import torch
from torch import nn

from pipeline.compile import PEFTQuantPipeline
from pipeline.quantizer import dequantize_symmetric, quantize_symmetric
from pipeline.tuner import LoRALinear


def test_lora_parameter_freezing():
    layer = LoRALinear(in_features=10, out_features=5, rank=2, alpha=4.0)

    # Check that base parameters do not require gradients
    assert not layer.weight.requires_grad
    assert not layer.bias.requires_grad

    # Check that low-rank adapters require gradients
    assert layer.lora_A.requires_grad
    assert layer.lora_B.requires_grad

    # Verify forward pass shape
    x = torch.randn(3, 10)
    out = layer(x)
    assert out.shape == (3, 5)


def test_lora_weight_merging():
    layer = LoRALinear(in_features=4, out_features=4, rank=2, alpha=4.0)

    # Initialize mock inputs
    x = torch.randn(1, 4)

    # Pre-merge evaluation
    out_before = layer(x)

    # Merge weights
    merged_w = layer.merge_weights()
    assert merged_w.shape == (4, 4)

    # Verify merged weights forward matches pre-merge output (without delta calculation)
    # W_merged x + bias
    out_after = torch.nn.functional.linear(x, merged_w, layer.bias)
    assert torch.allclose(out_before, out_after, atol=1e-5)


def test_uniform_quantization():
    weight = torch.randn(10, 10) * 5.0

    # Quantize to 4-bit
    q_weight, scale = quantize_symmetric(weight, num_bits=4)

    assert scale > 0.0
    # Values should be clamped within [-7, 7]
    assert q_weight.max() <= 7
    assert q_weight.min() >= -7

    # Dequantize
    deq_weight = dequantize_symmetric(q_weight, scale)
    assert deq_weight.shape == weight.shape
    # Mean squared error should be low
    mse = ((weight - deq_weight) ** 2).mean().item()
    assert mse < 0.5


def test_compilation_pipeline():
    model = nn.Sequential(LoRALinear(in_features=8, out_features=4), nn.ReLU(), nn.Linear(4, 2))

    pipeline = PEFTQuantPipeline(model)
    compiled = pipeline.compile_model(num_bits=4)

    assert "0.weight" in compiled["weights"]
    assert "0.scale" in compiled["scales"]
    assert "2.weight" in compiled["weights"]
    assert "2.scale" in compiled["scales"]
    assert compiled["quantization_bits"] == 4
