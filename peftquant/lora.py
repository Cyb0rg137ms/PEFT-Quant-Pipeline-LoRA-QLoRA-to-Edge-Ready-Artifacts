"""LoRA (Hu et al., 2021) linear adapter with merge/unmerge and wrapping of existing layers."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """``y = x W^T + b + (alpha / r) * dropout(x) A^T B^T`` with frozen ``W, b``.

    ``A`` is Kaiming-uniform initialised and ``B`` is zero, so a fresh adapter is
    an exact no-op. Use :meth:`from_linear` to wrap a pretrained ``nn.Linear``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.in_features, self.out_features, self.rank, self.alpha = (
            in_features,
            out_features,
            rank,
            alpha,
        )
        self.scaling = alpha / rank
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / math.sqrt(in_features),
            requires_grad=False,
        )
        self.bias: nn.Parameter | None = nn.Parameter(torch.zeros(out_features), requires_grad=False) if bias else None
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False
        self.enabled = True  # False -> behave as the frozen base layer

    @classmethod
    def from_linear(cls, layer: nn.Linear, rank: int = 4, alpha: float = 8.0, dropout: float = 0.0) -> LoRALinear:
        """Wrap a pretrained linear layer (weights are copied and frozen)."""
        m = cls(
            layer.in_features,
            layer.out_features,
            rank,
            alpha,
            dropout,
            bias=layer.bias is not None,
        )
        m = m.to(device=layer.weight.device, dtype=layer.weight.dtype)
        with torch.no_grad():
            m.weight.copy_(layer.weight)
            if layer.bias is not None and m.bias is not None:
                m.bias.copy_(layer.bias)
        return m

    def delta_weight(self) -> Tensor:
        """``scaling * B @ A``."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: Tensor) -> Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.merged or not self.enabled:
            return out
        return out + F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling

    def merge_weights(self) -> Tensor:
        """Dense weight ``W + scaling * B A`` (does not modify the module)."""
        return self.weight + self.delta_weight()

    @torch.no_grad()
    def merge_(self) -> None:
        """Fold the adapter into ``weight`` in place (idempotent)."""
        if not self.merged:
            self.weight.add_(self.delta_weight())
            self.merged = True

    @torch.no_grad()
    def unmerge_(self) -> None:
        """Undo :meth:`merge_` (exact up to floating-point rounding)."""
        if self.merged:
            self.weight.sub_(self.delta_weight())
            self.merged = False

    def to_linear(self) -> nn.Linear:
        """A plain ``nn.Linear`` with the merged weight."""
        lin = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        lin = lin.to(device=self.weight.device, dtype=self.weight.dtype)
        with torch.no_grad():
            lin.weight.copy_(self.weight if self.merged else self.merge_weights())
            if self.bias is not None:
                lin.bias.copy_(self.bias)
        return lin

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, alpha={self.alpha}, "
            f"merged={self.merged}"
        )
