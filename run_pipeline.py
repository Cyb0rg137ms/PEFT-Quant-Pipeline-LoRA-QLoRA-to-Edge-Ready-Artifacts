"""End-to-end demo: LoRA fine-tune -> Freivalds-verified merge -> GPTQ int4 -> verified export/load.

All numbers printed are measured in this run (v1 printed a hard-coded "8.0x").
"""

from __future__ import annotations

import tempfile

import torch
from torch import nn

from peftquant import (
    LoRALinear,
    QuantSpec,
    count_parameters,
    export_artifact,
    inject_adapters,
    load_artifact,
    merge_adapters,
    model_nbytes,
    quantize_model,
    verify_lora_merge,
)


def build() -> nn.Sequential:
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32))


def main() -> None:
    torch.manual_seed(0)
    x, y = torch.randn(512, 64), torch.randn(512, 32)
    model = build()
    inject_adapters(model, rank=4, alpha=8.0)
    tr, tot = count_parameters(model)
    print(f"trainable {tr} / {tot} params")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for epoch in range(20):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        if epoch % 5 == 0:
            print(f"epoch {epoch:2d} mse {loss.item():.4f}")
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            r = verify_lora_merge(m.merge_weights().detach(), m.weight, m.lora_A, m.lora_B, m.scaling)
            print(f"Freivalds merge check {name}: passed={r.passed} residual={r.max_rel_residual:.2e}")
    merge_adapters(model)
    q = quantize_model(model, QuantSpec(4, "asym", "group", 64), "gptq", x[:256])
    fp, qb = model_nbytes(model), model_nbytes(q)
    print(f"fp32 {fp} B -> GPTQ int4 {qb} B ({fp / qb:.2f}x, scales and fp32 biases included)")
    with tempfile.TemporaryDirectory() as d:
        man = export_artifact(q, d, reference=model, probe=x[:128])
        print("export verification:", man["verification"])
        loaded = load_artifact(d, build())
        print("loaded == exported outputs:", torch.equal(loaded(x[:8]), q(x[:8])))


if __name__ == "__main__":
    main()
