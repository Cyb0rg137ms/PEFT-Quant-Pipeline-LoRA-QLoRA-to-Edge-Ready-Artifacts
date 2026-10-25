"""Legacy merge-then-quantise compiler (API-compatible with v1).

Returns integer codes (``int8``) per layer and reports *measured* storage
(``nbytes``) instead of a hard-coded compression ratio. For packed 4-bit
storage, GPTQ and verified export use :func:`peftquant.quantize_model` and
:func:`peftquant.export_artifact`.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from peftquant.dora import AdaLoRALinear, DoRALinear
from peftquant.lora import LoRALinear
from pipeline.quantizer import quantize_symmetric


class PEFTQuantPipeline:
    """Merge adapters and quantise every linear layer per-tensor."""

    def __init__(self, model: nn.Module):
        self.model = model

    def compile_model(self, num_bits: int = 4) -> dict[str, Any]:
        weights: dict[str, Any] = {}
        scales: dict[str, float] = {}
        fp32_bytes = 0
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear | DoRALinear | AdaLoRALinear):
                w = module.weight if getattr(module, "merged", False) else module.merge_weights()
            elif isinstance(module, nn.Linear):
                w = module.weight
            else:
                continue
            q, s = quantize_symmetric(w.detach(), num_bits)
            weights[f"{name}.weight"] = q
            scales[f"{name}.scale"] = s
            fp32_bytes += w.numel() * 4
            if module.bias is not None:
                weights[f"{name}.bias"] = module.bias.detach().clone()
        n_codes = sum(v.numel() for k, v in weights.items() if k.endswith(".weight"))
        packed = (n_codes * num_bits + 7) // 8
        return {
            "weights": weights,
            "scales": scales,
            "quantization_bits": num_bits,
            "fp32_weight_bytes": fp32_bytes,
            "int8_storage_bytes": n_codes,
            "ideal_packed_bytes": packed,
        }
