"""Legacy per-tensor symmetric quantiser (kept for API compatibility).

Fixed in v2: ``quantize_symmetric`` now returns a real ``torch.int8`` code
tensor (it used to return rounded *floats*, so nothing was ever compressed).
Prefer :func:`peftquant.quantize`, which supports per-channel/group scales,
asymmetric and NF4 schemes, and packs 4-bit codes two per byte.
"""

from __future__ import annotations

import torch


def quantize_symmetric(weight: torch.Tensor, num_bits: int = 4) -> tuple[torch.Tensor, float]:
    """Per-tensor symmetric quantisation to ``int8`` codes in ``[-(2^(b-1)-1), 2^(b-1)-1]``."""
    if not 2 <= num_bits <= 8:
        raise ValueError("num_bits must be in [2, 8]")
    q_boundary = 2 ** (num_bits - 1) - 1
    max_abs = weight.detach().abs().max().item()
    scale = max_abs / q_boundary if max_abs > 0 else 1.0
    q = torch.clamp(torch.round(weight.detach() / scale), -q_boundary, q_boundary)
    return q.to(torch.int8), scale


def dequantize_symmetric(q_weight: torch.Tensor, scale: float) -> torch.Tensor:
    """Restore approximate float weights."""
    return q_weight.float() * scale
