"""Reproducible CPU benchmarks. Every number in the README comes from ``peftquant bench``."""

from __future__ import annotations

import copy
import tempfile
import time
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from peftquant.export import (
    export_artifact,
    load_artifact,
    model_nbytes,
    quantize_model,
)
from peftquant.grpo import GRPOConfig, train_grpo
from peftquant.inject import count_parameters, inject_adapters, merge_adapters
from peftquant.lora import LoRALinear
from peftquant.quant import QuantSpec, quantize, sqnr_db
from peftquant.verify import freivalds, verify_lora_merge

SPECS = {
    "int8 sym channel": QuantSpec(8, "sym", "channel"),
    "int4 sym tensor": QuantSpec(4, "sym", "tensor"),
    "int4 sym channel": QuantSpec(4, "sym", "channel"),
    "int4 sym group64": QuantSpec(4, "sym", "group", 64),
    "int4 asym group64": QuantSpec(4, "asym", "group", 64),
    "nf4 group64": QuantSpec(4, "nf4", "group", 64),
    "int3 asym group64": QuantSpec(3, "asym", "group", 64),
}

EXTRA_SPECS = {
    "int3 sym channel": QuantSpec(3, "sym", "channel"),
    "int2 asym group64": QuantSpec(2, "asym", "group", 64),
}


@dataclass
class BenchConfig:
    seed: int = 0
    dim: int = 128
    classes: int = 10
    n_train: int = 8192
    grpo_steps: int = 400


def _data(cfg: BenchConfig, n: int, gen: torch.Generator, mix: Tensor, centers: Tensor) -> tuple[Tensor, Tensor]:
    y = torch.randint(0, cfg.classes, (n,), generator=gen)
    x = (centers[y] + torch.randn(n, cfg.dim, generator=gen)) @ mix
    return x, y


def _mlp(cfg: BenchConfig) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(cfg.dim, 256),
        nn.GELU(),
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Linear(256, cfg.classes),
    )


def _train(model: nn.Module, x: Tensor, y: Tensor, epochs: int, lr: float) -> None:
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(epochs):
        for i in range(0, len(x), 256):
            opt.zero_grad()
            F.cross_entropy(model(x[i : i + 256]), y[i : i + 256]).backward()
            opt.step()


@torch.no_grad()
def _acc(model: nn.Module, x: Tensor, y: Tensor) -> float:
    return float((model.eval()(x).argmax(1) == y).float().mean())


def bench_weight_sqnr(cfg: BenchConfig) -> list[dict]:
    w = torch.randn(256, 1024, generator=torch.Generator().manual_seed(cfg.seed))
    rows = []
    for name, spec in SPECS.items():
        q = quantize(w, spec)
        rows.append(
            {
                "spec": name,
                "sqnr_db": sqnr_db(w, q.dequantize()),
                "bytes": q.nbytes,
                "ratio_vs_fp32": w.numel() * 4 / q.nbytes,
            }
        )
    return rows


def bench_model_quant(cfg: BenchConfig) -> dict:
    """Train an MLP on a correlated Gaussian-mixture task, then compare RTN vs GPTQ."""
    gen = torch.Generator().manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    mix = torch.randn(cfg.dim, cfg.dim, generator=gen) / cfg.dim**0.5
    centers = torch.randn(cfg.classes, cfg.dim, generator=gen) * 0.35
    xtr, ytr = _data(cfg, cfg.n_train, gen, mix, centers)
    xte, yte = _data(cfg, 4096, gen, mix, centers)
    model = _mlp(cfg)
    _train(model, xtr, ytr, epochs=10, lr=1e-3)
    ref_logits = model.eval()(xte).detach()
    rows: list[dict] = []
    out = {
        "fp32": {"acc": _acc(model, xte, yte), "bytes": model_nbytes(model)},
        "rows": rows,
    }
    calib = xtr[:1024]
    for name in [
        "int4 sym channel",
        "int4 asym group64",
        "int3 asym group64",
        "int3 sym channel",
        "int2 asym group64",
    ]:
        spec = SPECS.get(name) or EXTRA_SPECS[name]
        for method in ("rtn", "gptq"):
            q = quantize_model(model, spec, method, calib if method == "gptq" else None)  # type: ignore[arg-type]
            logits = q(xte)
            rows.append(
                {
                    "spec": name,
                    "method": method,
                    "acc": _acc(q, xte, yte),
                    "bytes": model_nbytes(q),
                    "logit_rel_err": float((logits - ref_logits).norm() / ref_logits.norm()),
                    "top1_agree": float((logits.argmax(1) == ref_logits.argmax(1)).float().mean()),
                }
            )
    return out


def bench_lora_pipeline(cfg: BenchConfig) -> dict:
    """Domain shift -> LoRA fine-tune -> merge (Freivalds-verified) -> GPTQ int4 -> export/load round trip."""
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    torch.manual_seed(cfg.seed)
    mix = torch.randn(cfg.dim, cfg.dim, generator=gen) / cfg.dim**0.5
    centers = torch.randn(cfg.classes, cfg.dim, generator=gen) * 0.35
    xtr, ytr = _data(cfg, cfg.n_train, gen, mix, centers)
    base = _mlp(cfg)
    _train(base, xtr, ytr, epochs=10, lr=1e-3)
    # target domain: rotated inputs (same labels)
    rot, _ = torch.linalg.qr(torch.randn(cfg.dim, cfg.dim, generator=gen))
    shift = 0.6 * rot + 0.4 * torch.eye(cfg.dim)
    xs, ys = _data(cfg, cfg.n_train, gen, mix @ shift, centers)
    xst, yst = _data(cfg, 4096, gen, mix @ shift, centers)
    res: dict = {"base_acc_on_shifted_domain": _acc(base, xst, yst)}
    model = copy.deepcopy(base)
    inject_adapters(model, kind="lora", rank=8, alpha=16.0)
    tr, tot = count_parameters(model)
    res.update(trainable_params=tr, total_params=tot)
    t0 = time.perf_counter()
    _train(model, xs, ys, epochs=5, lr=3e-3)
    res["lora_train_s"] = time.perf_counter() - t0
    res["lora_acc"] = _acc(model, xst, yst)
    reports = [
        verify_lora_merge(m.merge_weights().detach(), m.weight, m.lora_A, m.lora_B, m.scaling)
        for m in model.modules()
        if isinstance(m, LoRALinear)
    ]
    res["freivalds_merge_pass"] = all(r.passed for r in reports)
    res["freivalds_max_rel_residual"] = max(r.max_rel_residual for r in reports)
    merge_adapters(model)
    res["merged_acc"] = _acc(model, xst, yst)
    # A deliberately corrupted merge must be rejected
    m0 = next(m for m in base.modules() if isinstance(m, nn.Linear))
    bad = m0.weight.detach().clone()
    bad[3, 5] += 1e-2
    res["freivalds_detects_single_entry_corruption"] = not freivalds(
        bad, torch.eye(bad.shape[1]), m0.weight.detach()
    ).passed
    q = quantize_model(model, QuantSpec(4, "asym", "group", 64), "gptq", xs[:1024])
    res["gptq_int4_acc"] = _acc(q, xst, yst)
    with tempfile.TemporaryDirectory() as d:
        manifest = export_artifact(q, d, reference=model, probe=xst[:512])
        loaded = load_artifact(d, _mlp(cfg))
        res["loaded_acc"] = _acc(loaded, xst, yst)
        res["loaded_matches_exported"] = bool(torch.equal(loaded(xst[:512]), q(xst[:512])))
        res["artifact_file_bytes"] = manifest["file_bytes"]
        res["artifact_payload_bytes"] = manifest["payload_bytes"]
        res["fp32_bytes"] = model_nbytes(model)
        res["export_top1_agreement_vs_fp32"] = manifest["verification"]["top1_agreement"]
    return res


def bench_freivalds_speed(n: int = 2048, reps: int = 5) -> dict:
    g = torch.Generator().manual_seed(0)
    A, B = torch.randn(n, n, generator=g), torch.randn(n, n, generator=g)
    C = A @ B
    A64, B64, C64 = A.double(), B.double(), C.double()
    t0 = time.perf_counter()
    for _ in range(reps):
        torch.allclose(A64 @ B64, C64, rtol=1e-4, atol=1e-3)
    full = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        freivalds(A64, B64, C64, trials=8)
    fv = (time.perf_counter() - t0) / reps
    return {
        "n": n,
        "recompute_fp64_check_s": full,
        "freivalds_8_trials_fp64_s": fv,
        "speedup": full / fv,
    }


def run_all(cfg: BenchConfig = BenchConfig()) -> dict:
    t0 = time.perf_counter()
    torch.manual_seed(cfg.seed)
    out = {
        "config": asdict(cfg),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "weight_sqnr": bench_weight_sqnr(cfg),
        "model_quant": bench_model_quant(cfg),
        "lora_pipeline": bench_lora_pipeline(cfg),
        "freivalds_speed": bench_freivalds_speed(),
    }
    g = train_grpo(GRPOConfig(steps=cfg.grpo_steps, seed=cfg.seed))
    g.pop("sampled_reward_curve")
    out["grpo"] = g
    out["total_s"] = time.perf_counter() - t0
    return out
