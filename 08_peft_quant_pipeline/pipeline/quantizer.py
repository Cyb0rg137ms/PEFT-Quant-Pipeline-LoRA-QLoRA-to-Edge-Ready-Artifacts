"""
quantizer.py
============
Implements weight matrix quantization algorithms (uniform symmetric quantization)
to compress neural network parameters for deployment.
"""

import torch
from typing import Tuple

def quantize_symmetric(weight: torch.Tensor, num_bits: int = 4) -> Tuple[torch.Tensor, float]:
    """
    Applies uniform symmetric quantization to a weight matrix.
    Maps range [-max_abs, max_abs] to integer range [-q_boundary, q_boundary].
    
    Args:
        weight: The raw float weight tensor.
        num_bits: Bit-width representation (e.g. 4 or 8 bits).
        
    Returns:
        A tuple of (quantized_integer_tensor, scale_float)
    """
    q_boundary = (2 ** (num_bits - 1)) - 1
    
    # Calculate scale factor
    max_abs = weight.abs().max().item()
    scale = max_abs / q_boundary if max_abs > 0 else 1.0
    
    # Scale, round, and clamp
    q_weight = torch.round(weight / scale)
    q_weight = torch.clamp(q_weight, -q_boundary, q_boundary)
    
    return q_weight, scale

def dequantize_symmetric(q_weight: torch.Tensor, scale: float) -> torch.Tensor:
    """Restores the quantized integer weights back to approximate floats."""
    return q_weight * scale
