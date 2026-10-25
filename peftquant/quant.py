"""Weight quantisation: integer (sym/asym, tensor/channel/group) and NF4, with real bit-packing.

All quantisers return a :class:`QTensor` holding *integer codes* (``torch.uint8``),
scales/zero-points and enough metadata to dequantise. ``QTensor.nbytes`` counts
the bytes actually stored (codes packed two-per-byte for <=4 bits), so reported
compression ratios are measured, not assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

Scheme = Literal["sym", "asym", "nf4"]
Granularity = Literal["tensor", "channel", "group"]


@dataclass(frozen=True)
class QuantSpec:
    """How to quantise a 2-D weight ``(out, in)``.

    ``group`` granularity splits each row into blocks of ``group_size`` input
    columns, each with its own scale. ``nf4`` always uses 4 bits and absmax
    blocks (``group_size`` defaults to 64 as in QLoRA).
    """

    bits: int = 4
    scheme: Scheme = "sym"
    granularity: Granularity = "channel"
    group_size: int = 128

    def __post_init__(self) -> None:
        if self.scheme == "nf4" and self.bits != 4:
            raise ValueError("nf4 is a 4-bit format")
        if not 2 <= self.bits <= 8:
            raise ValueError("bits must be in [2, 8]")
        if self.group_size < 1:
            raise ValueError("group_size must be positive")


@dataclass
class QTensor:
    """Quantised 2-D tensor. ``codes`` are unsigned integers in ``[0, 2^bits)``."""

    codes: Tensor  # uint8; if ``packed``, a little-endian bitstream at ``spec.bits`` bits per code
    scale: Tensor  # (out, n_groups) float32
    zero: Tensor  # (out, n_groups) float32 (code value of real 0)
    shape: tuple[int, int]
    spec: QuantSpec
    packed: bool = field(default=False)

    @property
    def nbytes(self) -> int:
        """Bytes stored: codes + fp16 scales (+ fp16 zero-points for asym)."""
        n_scales = 1 if self.spec.granularity == "tensor" and self.spec.scheme != "nf4" else self.scale.numel()
        extra = n_scales * 2 if self.spec.scheme == "asym" else 0
        return self.codes.numel() * self.codes.element_size() + n_scales * 2 + extra

    def dequantize(self) -> Tensor:
        """Float32 reconstruction."""
        out, inp = self.shape
        codes = (
            unpack_bits(self.codes, self.spec.bits, out * inp).view(out, inp)
            if self.packed
            else self.codes.view(out, inp)
        )
        g = _group_cols(self.spec, inp)
        c = _pad_cols(codes.float(), g).view(out, -1, g)
        if self.spec.scheme == "nf4":
            vals = nf4_codebook().to(c.device)[c.long()] * self.scale.unsqueeze(-1)
        else:
            vals = (c - self.zero.unsqueeze(-1)) * self.scale.unsqueeze(-1)
        return vals.view(out, -1)[:, :inp].contiguous()


def _fp16(t: Tensor) -> Tensor:
    """Round scales to fp16 (what is stored), so in-memory and exported results match bit-for-bit."""
    return t.half().float()


def _group_cols(spec: QuantSpec, inp: int) -> int:
    if spec.scheme == "nf4" or spec.granularity == "group":
        return min(spec.group_size, inp)
    return inp


def _pad_cols(w: Tensor, g: int) -> Tensor:
    pad = (-w.shape[1]) % g
    return F.pad(w, (0, pad)) if pad else w


@lru_cache(maxsize=1)
def nf4_codebook() -> Tensor:
    """The 16 NormalFloat-4 levels (Dettmers et al., 2023), normalised to [-1, 1].

    Built as in the QLoRA reference: 8 positive and 7 negative normal quantiles
    with offset 0.9677, plus an exact zero.
    """
    normal = torch.distributions.Normal(0.0, 1.0)
    offset = 0.9677083
    pos = normal.icdf(torch.linspace(offset, 0.5, 9, dtype=torch.float64)[:-1].float())
    neg = -normal.icdf(torch.linspace(offset, 0.5, 8, dtype=torch.float64)[:-1].float())
    v = torch.cat([pos, torch.zeros(1), neg]).sort().values
    return (v / v.abs().max()).float()


def pack_int4(codes: Tensor) -> Tensor:
    """Pack values in [0, 16) two per byte (low nibble first)."""
    flat = codes.reshape(-1).to(torch.uint8)
    if flat.numel() and int(flat.max()) > 15:
        raise ValueError("codes must be < 16 to pack as int4")
    if flat.numel() % 2:
        flat = torch.cat([flat, flat.new_zeros(1)])
    return flat[0::2] | (flat[1::2] << 4)


def unpack_int4(packed: Tensor, numel: int) -> Tensor:
    """Inverse of :func:`pack_int4`."""
    out = torch.stack([packed & 0x0F, packed >> 4], dim=1).reshape(-1)
    return out[:numel]


def pack_bits(codes: Tensor, bits: int) -> Tensor:
    """Pack values in ``[0, 2^bits)`` into a dense little-endian bitstream (``ceil(n*bits/8)`` bytes)."""
    if bits == 4:
        return pack_int4(codes)
    flat = codes.reshape(-1).to(torch.uint8)
    if flat.numel() and int(flat.max()) >= 2**bits:
        raise ValueError(f"codes must be < {2**bits} to pack at {bits} bits")
    shifts = torch.arange(bits, dtype=torch.uint8, device=flat.device)
    bitstream = ((flat.unsqueeze(1) >> shifts) & 1).reshape(-1)
    bitstream = F.pad(bitstream, (0, (-bitstream.numel()) % 8))
    weights = 1 << torch.arange(8, dtype=torch.int32, device=flat.device)
    return (bitstream.view(-1, 8).to(torch.int32) * weights).sum(1).to(torch.uint8)


def unpack_bits(packed: Tensor, bits: int, numel: int) -> Tensor:
    """Inverse of :func:`pack_bits`."""
    if bits == 4:
        return unpack_int4(packed, numel)
    shifts = torch.arange(8, dtype=torch.uint8, device=packed.device)
    bitstream = ((packed.unsqueeze(1) >> shifts) & 1).reshape(-1)[: numel * bits].view(numel, bits)
    weights = 1 << torch.arange(bits, dtype=torch.int32, device=packed.device)
    return (bitstream.to(torch.int32) * weights).sum(1).to(torch.uint8)


def quantize(w: Tensor, spec: QuantSpec = QuantSpec()) -> QTensor:
    """Round-to-nearest quantisation of a 2-D weight."""
    if w.dim() != 2:
        raise ValueError("expected a 2-D weight (out, in)")
    w = w.detach().float()
    out, inp = w.shape
    if spec.granularity == "tensor" and spec.scheme != "nf4":
        g = inp
        blocks = w.reshape(1, 1, -1)
    else:
        g = _group_cols(spec, inp)
        blocks = _pad_cols(w, g).view(out, -1, g)
    if spec.scheme == "nf4":
        absmax = _fp16(blocks.abs().amax(-1).clamp_min(1e-6))
        cb = nf4_codebook().to(w.device)
        normed = blocks / absmax.unsqueeze(-1)
        codes = (normed.unsqueeze(-1) - cb).abs().argmin(-1).to(torch.uint8)
        scale, zero = absmax, torch.zeros_like(absmax)
    elif spec.scheme == "sym":
        qmax = 2 ** (spec.bits - 1) - 1
        scale = _fp16((blocks.abs().amax(-1) / qmax).clamp_min(1e-6))
        zero = torch.full_like(scale, float(qmax + 1)) if spec.bits > 1 else torch.zeros_like(scale)
        q = torch.clamp(torch.round(blocks / scale.unsqueeze(-1)), -qmax, qmax)
        codes = (q + zero.unsqueeze(-1)).to(torch.uint8)
    else:
        qmax = 2**spec.bits - 1
        lo, hi = blocks.amin(-1).clamp(max=0), blocks.amax(-1).clamp(min=0)
        scale = _fp16(((hi - lo) / qmax).clamp_min(1e-6))
        zero = torch.round(-lo / scale)
        codes = torch.clamp(torch.round(blocks / scale.unsqueeze(-1)) + zero.unsqueeze(-1), 0, qmax).to(torch.uint8)
    if spec.granularity == "tensor" and spec.scheme != "nf4":
        codes = codes.view(out, inp)
        scale, zero = (
            scale.expand(out, 1).contiguous(),
            zero.expand(out, 1).contiguous(),
        )
    else:
        codes = codes.view(out, -1)[:, :inp]
    packed = spec.bits < 8
    stored = pack_bits(codes, spec.bits) if packed else codes.contiguous()
    return QTensor(stored, scale.float(), zero.float(), (out, inp), spec, packed)


def sqnr_db(original: Tensor, reconstructed: Tensor) -> float:
    """Signal-to-quantisation-noise ratio in dB (``inf`` for exact reconstruction)."""
    noise = float(((original - reconstructed) ** 2).mean())
    if noise == 0.0:
        return math.inf
    return 10.0 * math.log10(float((original**2).mean()) / noise)


class QuantLinear(nn.Module):
    """Weight-only quantised linear layer (dequantise-on-the-fly reference kernel).

    The dequantised weight is cached after the first forward unless
    ``cache=False``; memory accounting uses :attr:`QTensor.nbytes`.
    """

    def __init__(self, qweight: QTensor, bias: Tensor | None = None, cache: bool = True) -> None:
        super().__init__()
        self.qweight = qweight
        self.out_features, self.in_features = qweight.shape
        self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False) if bias is not None else None
        self.cache = cache
        self._w: Tensor | None = None

    @classmethod
    def from_linear(cls, layer: nn.Linear, spec: QuantSpec = QuantSpec()) -> QuantLinear:
        """Quantise an existing linear layer with round-to-nearest."""
        return cls(quantize(layer.weight, spec), layer.bias)

    def weight(self) -> Tensor:
        """Dequantised weight."""
        if self._w is not None:
            return self._w
        w = self.qweight.dequantize()
        if self.cache:
            self._w = w
        return w

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight().to(device=x.device, dtype=x.dtype)
        return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))

    @property
    def nbytes(self) -> int:
        """Stored bytes (quantised weight + fp32 bias)."""
        return self.qweight.nbytes + (self.bias.numel() * 4 if self.bias is not None else 0)

    def extra_repr(self) -> str:
        s = self.qweight.spec
        return f"in={self.in_features}, out={self.out_features}, bits={s.bits}, scheme={s.scheme}, gran={s.granularity}"
