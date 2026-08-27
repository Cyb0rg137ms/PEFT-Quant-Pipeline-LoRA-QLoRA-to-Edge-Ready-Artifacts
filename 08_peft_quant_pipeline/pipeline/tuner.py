"""
tuner.py
========
Custom Parameter-Efficient Fine-Tuning (PEFT) module implementing Low-Rank Adaptation (LoRA)
layers in PyTorch. Freezes base weights and trains only low-rank matrices A and B.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Any

class LoRALinear(nn.Module):
    """Linear layer augmented with Low-Rank Adaptation (LoRA) adapters."""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 8.0):
        """
        Args:
            in_features: Input dimension of linear layer.
            out_features: Output dimension of linear layer.
            rank: Lower rank dimensions (r).
            alpha: Scaling parameter (alpha).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # Frozen base weight and bias (requires_grad = False)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        
        # Trainable low-rank adapter matrices
        # Initialize A to Gaussian and B to zero so adapter output is initially 0
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / math.sqrt(in_features)))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear path
        base_out = nn.functional.linear(x, self.weight, self.bias)
        
        # Low-rank adapter path: (x * A^T) * B^T * scaling
        lora_out = nn.functional.linear(
            nn.functional.linear(x, self.lora_A),
            self.lora_B
        ) * self.scaling
        
        return base_out + lora_out

    def merge_weights(self) -> torch.Tensor:
        """
        Fuses the adapter weights delta_W directly into the base weights.
        Used to compile the final model for zero-overhead inference.
        """
        # delta_W = (B * A) * scaling
        delta_w = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        return self.weight + delta_w
