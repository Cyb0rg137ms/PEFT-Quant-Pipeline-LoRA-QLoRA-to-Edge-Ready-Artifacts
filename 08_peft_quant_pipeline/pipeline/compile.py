"""
compile.py
==========
Manages model compilation pipelines: merges trained LoRA adapter weights 
back into the base layers, and compiles the result into quantized matrices.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from pipeline.tuner import LoRALinear
from pipeline.quantizer import quantize_symmetric

class PEFTQuantPipeline:
    """Orchestrates model compilation, merging, and quantization."""
    
    def __init__(self, model: nn.Module):
        self.model = model

    def compile_model(self, num_bits: int = 4) -> Dict[str, Any]:
        """
        Walks the model, merges all LoRALinear layers, and quantizes the weights.
        
        Returns:
            Dictionary containing compiled model parameter matrices and scales.
        """
        compiled_weights = {}
        quantization_scales = {}
        
        # Traverse named submodules
        for name, module in self.model.named_modules():
            # If the module is a LoRALinear layer, merge and quantize
            if isinstance(module, LoRALinear):
                # 1. Merge weights: W = W_base + (B * A) * scaling
                merged_weight = module.merge_weights()
                
                # 2. Quantize W
                q_weight, scale = quantize_symmetric(merged_weight, num_bits)
                
                compiled_weights[f"{name}.weight"] = q_weight
                quantization_scales[f"{name}.scale"] = scale
                compiled_weights[f"{name}.bias"] = module.bias.clone()
                
            elif isinstance(module, nn.Linear):
                # Standard linear layer: just quantize
                q_weight, scale = quantize_symmetric(module.weight, num_bits)
                compiled_weights[f"{name}.weight"] = q_weight
                quantization_scales[f"{name}.scale"] = scale
                if module.bias is not None:
                    compiled_weights[f"{name}.bias"] = module.bias.clone()
                    
        return {
            "weights": compiled_weights,
            "scales": quantization_scales,
            "quantization_bits": num_bits
        }
