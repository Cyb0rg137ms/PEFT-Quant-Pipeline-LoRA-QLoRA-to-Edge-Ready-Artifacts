"""GPTQ (Frantar et al., 2022): second-order post-training weight quantisation.

Given calibration inputs ``X`` to a linear layer, GPTQ quantises the weight one
input column at a time and spreads each column's rounding error over the
not-yet-quantised columns using the inverse Hessian ``H^-1``, with
``H = 2 X^T X / n + damp * I``. This minimises ``||X W^T - X Q^T||^2``
much better than round-to-nearest (RTN) at 3-4 bits.

Integer schemes (sym/asym; channel or group granularity) are supported;
NF4 is RTN-only here.
"""

from __future__ import annotations

import torch
from torch import Tensor

from peftquant.quant import QTensor, QuantSpec, _fp16, pack_bits


def _params(block: Tensor, spec: QuantSpec) -> tuple[Tensor, Tensor]:
    """Per-row scale and zero-point for ``block`` (rows x cols)."""
    if spec.scheme == "sym":
        qmax = 2 ** (spec.bits - 1) - 1
        scale = _fp16((block.abs().amax(1) / qmax).clamp_min(1e-6))
        return scale, torch.full_like(scale, float(qmax + 1))
    qmax = 2**spec.bits - 1
    lo, hi = block.amin(1).clamp(max=0), block.amax(1).clamp(min=0)
    scale = _fp16(((hi - lo) / qmax).clamp_min(1e-6))
    return scale, torch.round(-lo / scale)


def _q(col: Tensor, scale: Tensor, zero: Tensor, spec: QuantSpec) -> tuple[Tensor, Tensor]:
    """Quantise one column; returns (codes, dequantised values)."""
    if spec.scheme == "sym":
        qmax = 2 ** (spec.bits - 1) - 1
        q = torch.clamp(torch.round(col / scale), -qmax, qmax)
        return q + zero, q * scale
    qmax = 2**spec.bits - 1
    c = torch.clamp(torch.round(col / scale) + zero, 0, qmax)
    return c, (c - zero) * scale


def hessian(inputs: Tensor) -> Tensor:
    """``2 X^T X / n`` from calibration activations of shape ``(n, in)`` (or ``(..., in)``)."""
    x = inputs.reshape(-1, inputs.shape[-1]).double()
    return 2.0 * x.t() @ x / x.shape[0]


def gptq_quantize(
    weight: Tensor,
    H: Tensor,
    spec: QuantSpec = QuantSpec(),
    damp: float = 0.01,
    block_size: int = 128,
) -> QTensor:
    """Quantise ``weight (out, in)`` given the layer-input Hessian ``H (in, in)``."""
    if spec.scheme == "nf4":
        raise ValueError("gptq_quantize supports sym/asym integer schemes")
    W = weight.detach().double().clone()
    out, inp = W.shape
    H = H.double().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(inp, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)

    g = spec.group_size if spec.granularity == "group" else inp
    n_groups = (inp + g - 1) // g if spec.granularity != "tensor" else 1
    scales = torch.zeros(out, n_groups, dtype=torch.float64)
    zeros = torch.zeros(out, n_groups, dtype=torch.float64)
    codes = torch.zeros(out, inp, dtype=torch.uint8)
    if spec.granularity == "tensor":
        s, z = _params(W.reshape(1, -1), spec)
        scales[:], zeros[:] = s, z
    elif spec.granularity == "channel":
        s, z = _params(W, spec)
        scales[:, 0], zeros[:, 0] = s, z

    for b0 in range(0, inp, block_size):
        b1 = min(b0 + block_size, inp)
        Wb = W[:, b0:b1].clone()
        Err = torch.zeros_like(Wb)
        Hb = Hinv[b0:b1, b0:b1]
        for j in range(b1 - b0):
            col = b0 + j
            gi = 0 if spec.granularity == "tensor" else col // g
            if spec.granularity == "group" and col % g == 0:
                # group params from the *error-updated* weights, as in GPTQ
                s, z = _params(torch.cat([Wb[:, j:], W[:, b1 : col + g]], 1)[:, :g], spec)
                scales[:, gi], zeros[:, gi] = s, z
            c, deq = _q(Wb[:, j], scales[:, gi], zeros[:, gi], spec)
            codes[:, col] = c.to(torch.uint8)
            err = (Wb[:, j] - deq) / Hb[j, j]
            Wb[:, j:] -= err.unsqueeze(1) * Hb[j, j:].unsqueeze(0)
            Err[:, j] = err
        W[:, b1:] -= Err @ Hinv[b0:b1, b1:]

    packed = spec.bits < 8
    return QTensor(
        pack_bits(codes, spec.bits) if packed else codes,
        scales.float(),
        zeros.float(),
        (out, inp),
        spec,
        packed,
    )
