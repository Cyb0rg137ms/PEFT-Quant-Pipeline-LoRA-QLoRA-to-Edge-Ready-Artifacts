"""
dora.py
=======
Weight-Decomposed Low-Rank Adaptation (DoRA).

Reference:
  Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", NeurIPS 2024.
  https://arxiv.org/abs/2402.09353

DoRA factorises the weight as:
    W = m * (W_0 + delta_W) / ||W_0 + delta_W||_col

where:
  - m   is a learnable magnitude vector (one scalar per output neuron)
  - W_0 is the frozen base weight
  - delta_W = B @ A * scaling  is the LoRA low-rank update
  - ||·||_col denotes the column-wise L2 norm

This preserves the directional component of the base weight while training
only the magnitude and direction adjustment separately.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class DoRALinear(nn.Module):
    """
    Linear layer with Weight-Decomposed Low-Rank Adaptation (DoRA).

    Compared to LoRA, DoRA additionally learns a per-column magnitude vector `m`
    that rescales the directional weight matrix. This gives the adapter more
    expressive power without increasing the number of trainable rank parameters.

    Trainable parameters:
        lora_A:  (rank, in_features)
        lora_B:  (out_features, rank)
        magnitude: (out_features,)

    Frozen parameters:
        weight: (out_features, in_features)
        bias:   (out_features,)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen base weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / math.sqrt(in_features)))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # DoRA magnitude vector — initialised to column-wise norm of base weight
        col_norms = self.weight.norm(dim=1).clamp(min=1e-6)  # shape: (out_features,)
        self.magnitude = nn.Parameter(col_norms.clone())

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    @property
    def effective_weight(self) -> torch.Tensor:
        """W_0 + delta_W, before magnitude normalisation."""
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        return self.weight + delta_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_eff = self.effective_weight  # (out, in)

        # Column-wise L2 normalisation: each row (= output column direction) to unit norm
        col_norms = W_eff.norm(dim=1, keepdim=True).clamp(min=1e-6)
        W_norm = W_eff / col_norms  # normalised directional weight

        # Rescale by learned magnitude
        W_dora = self.magnitude.unsqueeze(1) * W_norm  # (out, in)

        return F.linear(self.dropout(x), W_dora, self.bias)

    def merge_weights(self) -> torch.Tensor:
        """Returns the equivalent dense weight for zero-overhead inference."""
        W_eff = self.effective_weight
        col_norms = W_eff.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return self.magnitude.unsqueeze(1) * (W_eff / col_norms)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, alpha={self.alpha}"


class AdaLoRALinear(nn.Module):
    """
    Adaptive LoRA (AdaLoRA) Linear layer.

    Reference:
      Zhang et al., "Adaptive Budget Allocation for Parameter-Efficient
      Fine-Tuning", ICLR 2023. https://arxiv.org/abs/2303.10512

    AdaLoRA parameterises delta_W = P @ diag(Lambda) @ Q  (SVD-style), where:
      - P ∈ R^(out × rank): left singular vectors (trainable)
      - Lambda ∈ R^rank:    singular values (trainable, pruned by importance)
      - Q ∈ R^(rank × in): right singular vectors (trainable)

    During training, low-importance singular values are zeroed out, effectively
    reducing the rank of inactive layers and allocating budget to active ones.

    Trainable parameters:
        ada_P:      (out_features, rank)
        ada_lambda: (rank,)
        ada_Q:      (rank, in_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

        # SVD-factored adapter: delta_W = P @ diag(lam) @ Q
        self.ada_P = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        # Zero singular values at init (as in the AdaLoRA paper) so a fresh
        # adapter is an exact no-op; P and Q get small random values.
        self.ada_lambda = nn.Parameter(torch.zeros(rank))
        self.ada_Q = nn.Parameter(torch.randn(rank, in_features) * 0.01)

        # Importance mask: 1.0 = active, 0.0 = pruned
        self.importance_mask: torch.Tensor
        self.register_buffer("importance_mask", torch.ones(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Masked singular values
        lam = self.ada_lambda * self.importance_mask  # (rank,)
        # delta_W = P @ diag(lam) @ Q
        delta_w = (self.ada_P * lam.unsqueeze(0)) @ self.ada_Q  # (out, in)
        W_eff = self.weight + delta_w * self.scaling
        return F.linear(x, W_eff, self.bias)

    def prune_rank(self, keep_top_k: int) -> None:
        """
        Prunes the bottom (rank - keep_top_k) singular values by their magnitude.
        Called periodically during training to enforce budget constraints.

        Args:
            keep_top_k: Number of singular values to retain.
        """
        with torch.no_grad():
            sorted_indices = torch.argsort(self.ada_lambda.abs(), descending=True)
            if not 0 <= keep_top_k <= self.rank:
                raise ValueError("keep_top_k must be in [0, rank]")
            mask = torch.zeros(
                self.rank,
                dtype=self.importance_mask.dtype,
                device=self.importance_mask.device,
            )
            mask[sorted_indices[:keep_top_k]] = 1.0
            self.importance_mask.copy_(mask)

    def effective_rank(self) -> int:
        """Returns the number of currently active singular values."""
        return int(self.importance_mask.sum().item())

    def merge_weights(self) -> torch.Tensor:
        """Returns the merged dense weight for inference."""
        lam = self.ada_lambda * self.importance_mask
        delta_w = (self.ada_P * lam.unsqueeze(0)) @ self.ada_Q
        return self.weight + delta_w * self.scaling

    def orthogonal_regularization(self) -> torch.Tensor:
        """``||P^T P - I||_F^2 + ||Q Q^T - I||_F^2`` (keeps P, Q near-orthogonal)."""
        eye = torch.eye(self.rank, device=self.ada_P.device, dtype=self.ada_P.dtype)
        return ((self.ada_P.t() @ self.ada_P - eye) ** 2).sum() + ((self.ada_Q @ self.ada_Q.t() - eye) ** 2).sum()
