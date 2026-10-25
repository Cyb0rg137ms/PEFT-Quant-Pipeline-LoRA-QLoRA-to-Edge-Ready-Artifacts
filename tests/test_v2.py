"""Tests for the v2 ``peftquant`` package."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from peftquant import (
    ActivationCalibrator,
    AdaLoRALinear,
    DoRALinear,
    LoRALinear,
    QuantLinear,
    QuantSpec,
    adapters_disabled,
    count_parameters,
    detect_device,
    export_artifact,
    freivalds,
    gptq_quantize,
    hessian,
    inject_adapters,
    load_artifact,
    merge_adapters,
    model_nbytes,
    nf4_codebook,
    pack_bits,
    pack_int4,
    quantize,
    quantize_model,
    sqnr_db,
    unpack_bits,
    unpack_int4,
    verify_lora_merge,
)
from peftquant.cli import main
from peftquant.grpo import GRPOConfig, group_advantages, reward_fn, train_grpo
from pipeline.compile import PEFTQuantPipeline
from pipeline.quantizer import quantize_symmetric


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def mlp() -> nn.Sequential:
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 4))


# ---------------- quantisation ----------------


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7])
def test_pack_bits_roundtrip_and_density(bits: int) -> None:
    codes = torch.randint(0, 2**bits, (1001,), dtype=torch.uint8)
    packed = pack_bits(codes, bits)
    assert packed.numel() == (1001 * bits + 7) // 8
    assert torch.equal(unpack_bits(packed, bits, 1001), codes)


def test_pack_int4_roundtrip_and_overflow() -> None:
    c = torch.randint(0, 16, (33,), dtype=torch.uint8)
    assert torch.equal(unpack_int4(pack_int4(c), 33), c)
    with pytest.raises(ValueError):
        pack_int4(torch.tensor([16], dtype=torch.uint8))
    with pytest.raises(ValueError):
        pack_bits(torch.tensor([8], dtype=torch.uint8), 3)


def test_nf4_codebook_properties() -> None:
    cb = nf4_codebook()
    assert cb.numel() == 16 and cb.min() == -1 and cb.max() == 1
    assert (cb == 0).sum() == 1 and torch.all(cb[1:] > cb[:-1])
    # published QLoRA values
    assert cb[0] == -1.0 and abs(float(cb[1]) + 0.6962) < 1e-3 and abs(float(cb[14]) - 0.7230) < 1e-3


@pytest.mark.parametrize(
    "spec",
    [
        QuantSpec(8, "sym", "channel"),
        QuantSpec(4, "sym", "tensor"),
        QuantSpec(4, "asym", "group", 32),
        QuantSpec(4, "nf4", "group", 64),
        QuantSpec(3, "asym", "channel"),
        QuantSpec(2, "sym", "group", 16),
    ],
)
def test_quantize_codes_in_range_and_shapes(spec: QuantSpec) -> None:
    w = torch.randn(24, 70)
    q = quantize(w, spec)
    assert q.codes.dtype == torch.uint8
    deq = q.dequantize()
    assert deq.shape == w.shape and torch.isfinite(deq).all()
    codes = unpack_bits(q.codes, spec.bits, w.numel()) if q.packed else q.codes
    assert int(codes.max()) < 2**spec.bits


def test_finer_granularity_and_more_bits_improve_sqnr() -> None:
    w = torch.randn(64, 256)
    s = {
        k: sqnr_db(w, quantize(w, sp).dequantize())
        for k, sp in {
            "t4": QuantSpec(4, "sym", "tensor"),
            "c4": QuantSpec(4, "sym", "channel"),
            "g4": QuantSpec(4, "sym", "group", 32),
            "c8": QuantSpec(8, "sym", "channel"),
        }.items()
    }
    assert s["t4"] < s["c4"] < s["g4"] < s["c8"]
    assert s["c8"] > 40


def test_nbytes_is_real_storage() -> None:
    w = torch.randn(64, 256)
    q4 = quantize(w, QuantSpec(4, "sym", "channel"))
    assert q4.nbytes == 64 * 256 // 2 + 64 * 2
    q3 = quantize(w, QuantSpec(3, "sym", "channel"))
    assert q3.nbytes == 64 * 256 * 3 // 8 + 64 * 2
    assert quantize(w, QuantSpec(4, "sym", "tensor")).nbytes == 64 * 256 // 2 + 2


def test_zero_and_constant_weights() -> None:
    for spec in [
        QuantSpec(4, "sym"),
        QuantSpec(4, "asym"),
        QuantSpec(4, "nf4", "group", 64),
    ]:
        assert torch.all(quantize(torch.zeros(4, 8), spec).dequantize() == 0)
    assert sqnr_db(torch.ones(3, 3), torch.ones(3, 3)) == float("inf")


def test_spec_validation() -> None:
    with pytest.raises(ValueError):
        QuantSpec(3, "nf4")
    with pytest.raises(ValueError):
        QuantSpec(9)
    with pytest.raises(ValueError):
        quantize(torch.randn(3), QuantSpec())


def test_quantlinear_matches_dequantized_matmul() -> None:
    lin = nn.Linear(20, 10)
    ql = QuantLinear.from_linear(lin, QuantSpec(4, "asym", "group", 8))
    x = torch.randn(5, 20)
    ref = nn.functional.linear(x, ql.qweight.dequantize(), lin.bias)
    assert torch.allclose(ql(x), ref, atol=1e-6)


# ---------------- GPTQ ----------------


def test_gptq_beats_rtn_on_correlated_inputs() -> None:
    X = torch.randn(1024, 64) @ torch.randn(64, 64)
    W = torch.randn(32, 64)
    H = hessian(X)
    for spec in [QuantSpec(4, "sym", "channel"), QuantSpec(3, "asym", "group", 16)]:
        e_rtn = ((X @ W.t() - X @ quantize(W, spec).dequantize().t()) ** 2).mean()
        e_gptq = ((X @ W.t() - X @ gptq_quantize(W, H, spec, block_size=16).dequantize().t()) ** 2).mean()
        assert e_gptq < 0.8 * e_rtn


def test_gptq_handles_dead_inputs_and_rejects_nf4() -> None:
    X = torch.randn(256, 16)
    X[:, 3] = 0
    q = gptq_quantize(torch.randn(8, 16), hessian(X), QuantSpec(4, "sym", "channel"))
    assert torch.isfinite(q.dequantize()).all()
    with pytest.raises(ValueError):
        gptq_quantize(torch.randn(8, 16), hessian(X), QuantSpec(4, "nf4", "group", 8))


# ---------------- adapters ----------------


def test_lora_from_linear_is_noop_and_merge_unmerge() -> None:
    lin = nn.Linear(12, 6)
    lo = LoRALinear.from_linear(lin, rank=3)
    x = torch.randn(4, 12)
    assert torch.allclose(lo(x), lin(x), atol=1e-6)
    nn.init.normal_(lo.lora_B)
    y = lo(x)
    lo.merge_()
    assert torch.allclose(lo(x), y, atol=1e-5)
    lo.merge_()  # idempotent
    assert torch.allclose(lo(x), y, atol=1e-5)
    lo.unmerge_()
    assert torch.allclose(lo(x), y, atol=1e-5) and torch.allclose(lo.weight, lin.weight, atol=1e-6)
    assert torch.allclose(lo.to_linear()(x), y, atol=1e-5)
    with pytest.raises(ValueError):
        LoRALinear(4, 4, rank=0)


def test_inject_merge_and_counts() -> None:
    m = mlp()
    x = torch.randn(3, 16)
    ref = m(x)
    names = inject_adapters(m, r"[02]", rank=4)
    assert names == ["0", "2"]
    tr, tot = count_parameters(m)
    assert tr == 4 * (16 + 32) + 4 * (32 + 32)
    assert torch.allclose(m(x), ref, atol=1e-6)
    for p in m.parameters():
        if p.requires_grad:
            nn.init.normal_(p, std=0.1)
    y = m(x)
    with adapters_disabled(m):
        assert torch.allclose(m(x), ref, atol=1e-6)
    assert merge_adapters(m) == 2
    assert all(type(c) is nn.Linear for c in [m[0], m[2], m[4]])
    assert torch.allclose(m(x), y, atol=1e-5)
    with pytest.raises(ValueError):
        inject_adapters(mlp(), r"nothing")


def test_inject_dora_is_noop_at_init_and_merges() -> None:
    m = mlp()
    x = torch.randn(3, 16)
    ref = m(x)
    inject_adapters(m, lambda n, _l: n == "4", kind="dora", rank=2)
    assert isinstance(m[4], DoRALinear)
    assert torch.allclose(m(x), ref, atol=1e-5)
    nn.init.normal_(m[4].lora_B, std=0.1)
    y = m(x)
    merge_adapters(m)
    assert torch.allclose(m(x), y, atol=1e-5)


def test_adalora_starts_as_noop_and_prune_validates() -> None:
    a = AdaLoRALinear(8, 4, rank=4)
    x = torch.randn(2, 8)
    assert torch.allclose(a(x), nn.functional.linear(x, a.weight, a.bias))
    with pytest.raises(ValueError):
        a.prune_rank(5)


# ---------------- verification & export ----------------


def test_freivalds_accepts_correct_and_rejects_wrong() -> None:
    A, B = torch.randn(50, 40), torch.randn(40, 30)
    assert freivalds(A, B, A @ B).passed
    C = A @ B
    C[7, 11] += 0.05
    assert not freivalds(A, B, C).passed
    with pytest.raises(ValueError):
        freivalds(A, B, torch.zeros(3, 3))


def test_verify_lora_merge() -> None:
    lo = LoRALinear(32, 16, rank=4)
    nn.init.normal_(lo.lora_B)
    merged = lo.merge_weights().detach()
    assert verify_lora_merge(merged, lo.weight, lo.lora_A, lo.lora_B, lo.scaling).passed
    merged[0, 0] += 0.1
    assert not verify_lora_merge(merged, lo.weight, lo.lora_A, lo.lora_B, lo.scaling).passed


@pytest.mark.parametrize("method", ["rtn", "gptq"])
def test_export_load_roundtrip(tmp_path, method: str) -> None:
    m = mlp()
    x = torch.randn(64, 16)
    q = quantize_model(m, QuantSpec(4, "asym", "group", 8), method, x)  # type: ignore[arg-type]
    assert all(isinstance(c, QuantLinear) for c in [q[0], q[2], q[4]])
    assert model_nbytes(q) < model_nbytes(m) / 3  # group-8 scales are costly on tiny layers
    man = export_artifact(q, tmp_path, reference=m, probe=x)
    assert man["verification"]["rel_err"] < 0.2
    assert json.loads((tmp_path / "manifest.json").read_text())["sha256"] == man["sha256"]
    loaded = load_artifact(tmp_path, mlp())
    assert torch.equal(loaded(x), q(x))


def test_load_detects_tampering(tmp_path) -> None:
    q = quantize_model(mlp(), QuantSpec(4))
    export_artifact(q, tmp_path)
    with (tmp_path / "weights.pt").open("ab") as fh:
        fh.write(b"x")
    with pytest.raises(ValueError, match="hash"):
        load_artifact(tmp_path, mlp())


def test_quantize_model_skip_and_gptq_needs_calib() -> None:
    q = quantize_model(mlp(), skip=("4",))
    assert type(q[4]) is nn.Linear and isinstance(q[0], QuantLinear)
    with pytest.raises(ValueError):
        quantize_model(mlp(), method="gptq")


# ---------------- calibration (fixed percentile) ----------------


def test_percentile_is_over_activations_not_batch_maxima() -> None:
    model = nn.Sequential(nn.Linear(4, 4))
    cal = ActivationCalibrator(model, num_bits=8, strategy="percentile", percentile=50.0)
    cal.attach_hooks()
    with torch.no_grad():
        model(torch.randn(1000, 4))
    cal.remove_hooks()
    st = cal.stats["0"]
    assert st.percentile_absmax(50.0) < 0.5 * st.global_absmax()  # median |N(0,1)| ~ 0.67, max ~ 3.5


# ---------------- legacy API fixes ----------------


def test_legacy_quantize_returns_integers_and_reports_bytes() -> None:
    q, _ = quantize_symmetric(torch.randn(8, 8), 4)
    assert q.dtype == torch.int8
    out = PEFTQuantPipeline(nn.Sequential(nn.Linear(8, 4))).compile_model(4)
    assert out["fp32_weight_bytes"] == 128 and out["ideal_packed_bytes"] == 16


# ---------------- GRPO / device / CLI ----------------


def test_group_advantages() -> None:
    adv = group_advantages(torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5]]))
    assert torch.allclose(adv[0].mean(), torch.tensor(0.0)) and torch.all(adv[1] == 0)
    p = torch.tensor([[1, 2, 3]])
    assert reward_fn(p, torch.tensor([[3, 2, 1]])).item() == 1.0


def test_grpo_improves_reward() -> None:
    r = train_grpo(GRPOConfig(steps=200, seed=0))
    assert r["greedy_reward_after"] > r["greedy_reward_before"] + 0.1  # type: ignore[operator]
    assert r["trainable_params"] < r["total_params"] / 5  # type: ignore[operator]


def test_grpo_with_kl_runs() -> None:
    r = train_grpo(GRPOConfig(steps=20, kl_coef=0.05))
    assert len(r["sampled_reward_curve"]) == 20  # type: ignore[arg-type]


def test_device_and_cli(capsys) -> None:
    assert detect_device("cpu").backend == "cpu"
    assert main(["device"]) == 0
    assert main(["grpo", "--steps", "5"]) == 0
    assert "greedy_reward_after" in capsys.readouterr().out
