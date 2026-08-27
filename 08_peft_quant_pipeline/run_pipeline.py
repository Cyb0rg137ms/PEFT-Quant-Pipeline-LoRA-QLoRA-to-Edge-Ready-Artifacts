"""
run_pipeline.py
===============
Fine-tunes a model containing LoRALinear adapters, merges the weights, 
and compiles the model to a 4-bit quantized configuration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pipeline.tuner import LoRALinear
from pipeline.compile import PEFTQuantPipeline

class TextEncoder(nn.Module):
    """A dummy model implementing LoRALinear projection heads for evaluation."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj1 = LoRALinear(input_dim, output_dim, rank=4, alpha=8.0)
        self.relu = nn.ReLU()
        self.proj2 = LoRALinear(output_dim, output_dim, rank=4, alpha=8.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.relu(self.proj1(x)))

def run_peft_quantization_flow():
    print("==================================================")
    print("      PEFT & QUANTIZATION COMPILE PIPELINE       ")
    print("==================================================")
    
    input_dim = 64
    output_dim = 32
    
    # Initialize base model
    print(f"Initializing base model (Input: {input_dim}, Output: {output_dim})...")
    model = TextEncoder(input_dim, output_dim)
    
    # Verify that base weights are frozen
    print("Verifying trainable parameter configurations...")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    frozen_params = [p for p in model.parameters() if not p.requires_grad]
    
    print(f"  Trainable Parameters: {len(trainable_params)} tensor nodes")
    print(f"  Frozen Base Parameters: {len(frozen_params)} tensor nodes")
    
    # Generate dummy features and targets
    x = torch.randn(16, input_dim)
    y = torch.randn(16, output_dim)
    
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    
    # Run a quick training adaptation loop (fine-tuning only the adapters)
    print("\nRunning adapter fine-tuning iterations...")
    for epoch in range(5):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        print(f"  Epoch {epoch+1:02d} | Mean Squared Error Loss: {loss.item():.6f}")
        
    # Compile: merge and quantize
    print("\nCompiling model (Adapter Fusion + 4-bit Quantization)...")
    pipeline = PEFTQuantPipeline(model)
    compiled = pipeline.compile_model(num_bits=4)
    
    print(f"\nCompilation Successful (4-bit representation):")
    print(f"  Compiled parameter nodes: {len(compiled['weights'])}")
    print(f"  Stored quantization scale parameters: {len(compiled['scales'])}")
    
    # Calculate simulated compression ratio
    # Float32 = 32 bits per weight. Quantized = 4 bits per weight.
    # Compression: 8x reduction
    print(f"  Estimated Model Weight Compression: 8.0x reduction (Float32 -> Int4)")
    print("==================================================")

if __name__ == "__main__":
    run_peft_quantization_flow()
