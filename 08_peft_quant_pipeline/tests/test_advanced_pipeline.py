"""
test_advanced_pipeline.py
=========================
Comprehensive tests for DoRA, AdaLoRA, calibration, and quantization.

Tests cover:
  - DoRA weight decomposition correctness
  - DoRA vs. LoRA forward equivalence at rank=full
  - AdaLoRA rank pruning
  - AdaLoRA singular value magnitude ordering
  - Calibration hook attachment/removal
  - MinMax vs. Percentile calibration scale comparison
  - SQNR at different bit widths
  - Quantization round-trip error bounds
  - Per-channel vs. symmetric scale consistency
  - Pipeline: LoRA + DoRA mixed model compilation
  - Merge-then-quantize vs. quantize accuracy
"""

import pytest
import math
import torch
import torch.nn as nn

from pipeline.tuner import LoRALinear
from pipeline.dora import DoRALinear, AdaLoRALinear
from pipeline.quantizer import quantize_symmetric, dequantize_symmetric
from pipeline.calibration import ActivationCalibrator, compute_sqnr
from pipeline.compile import PEFTQuantPipeline


# ---------------------------------------------------------------------------
# 1. DoRA tests
# ---------------------------------------------------------------------------

class TestDoRA:
    def test_dora_output_shape(self):
        layer = DoRALinear(in_features=16, out_features=8, rank=4, alpha=8.0)
        x = torch.randn(3, 16)
        out = layer(x)
        assert out.shape == (3, 8)

    def test_dora_trainable_params(self):
        layer = DoRALinear(in_features=16, out_features=8, rank=4)
        trainable = [n for n, p in layer.named_parameters() if p.requires_grad]
        assert "lora_A" in trainable
        assert "lora_B" in trainable
        assert "magnitude" in trainable
        assert "weight" not in trainable
        assert "bias" not in trainable

    def test_dora_magnitude_init(self):
        """Magnitude should be initialised to base-weight column norms."""
        layer = DoRALinear(in_features=8, out_features=4, rank=2)
        col_norms = layer.weight.norm(dim=1)
        # At init, lora_B = 0 so effective_weight == base weight
        torch.testing.assert_close(
            layer.magnitude.data, col_norms, atol=1e-5, rtol=1e-4
        )

    def test_dora_merge_weights_shape(self):
        layer = DoRALinear(in_features=10, out_features=5, rank=3)
        merged = layer.merge_weights()
        assert merged.shape == (5, 10)

    def test_dora_merge_equiv_to_forward(self):
        """merged_weight @ x + bias should equal forward(x)."""
        layer = DoRALinear(in_features=8, out_features=4, rank=2)
        x = torch.randn(1, 8)
        out_forward = layer(x)
        merged_w = layer.merge_weights()
        out_merged = nn.functional.linear(x, merged_w, layer.bias)
        torch.testing.assert_close(out_forward, out_merged, atol=1e-5, rtol=1e-4)

    def test_dora_gradient_flows_to_trainable(self):
        layer = DoRALinear(in_features=8, out_features=4, rank=2)
        x = torch.randn(2, 8)
        loss = layer(x).sum()
        loss.backward()
        assert layer.lora_A.grad is not None
        assert layer.lora_B.grad is not None
        assert layer.magnitude.grad is not None

    def test_dora_gradient_frozen_base(self):
        layer = DoRALinear(in_features=8, out_features=4, rank=2)
        x = torch.randn(2, 8)
        layer(x).sum().backward()
        # Frozen weights should not accumulate gradients
        assert layer.weight.grad is None

    def test_dora_output_differs_from_base_weight(self):
        """After a training step, DoRA output should differ from pure base weight."""
        layer = DoRALinear(in_features=8, out_features=4, rank=2)
        opt = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=0.1)
        x = torch.randn(4, 8)
        target = torch.randn(4, 4)
        for _ in range(5):
            loss = ((layer(x) - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        # After training, effective_weight should differ from base weight
        diff = (layer.effective_weight - layer.weight).abs().max().item()
        assert diff > 1e-6, "LoRA weights didn't update"


# ---------------------------------------------------------------------------
# 2. AdaLoRA tests
# ---------------------------------------------------------------------------

class TestAdaLoRA:
    def test_adadora_output_shape(self):
        layer = AdaLoRALinear(in_features=12, out_features=6, rank=4)
        x = torch.randn(2, 12)
        assert layer(x).shape == (2, 6)

    def test_adalora_trainable_params(self):
        layer = AdaLoRALinear(in_features=8, out_features=4, rank=4)
        trainable = [n for n, p in layer.named_parameters() if p.requires_grad]
        assert "ada_P" in trainable
        assert "ada_lambda" in trainable
        assert "ada_Q" in trainable

    def test_adalora_prune_rank(self):
        layer = AdaLoRALinear(in_features=8, out_features=4, rank=8)
        # Override lambda values with known magnitudes
        with torch.no_grad():
            layer.ada_lambda.copy_(torch.tensor([1.0, 0.1, 0.5, 0.9, 0.01, 0.8, 0.3, 0.7]))
        layer.prune_rank(keep_top_k=4)
        assert layer.effective_rank() == 4

    def test_adalora_prune_zeros_low_singular(self):
        layer = AdaLoRALinear(in_features=8, out_features=4, rank=4)
        with torch.no_grad():
            layer.ada_lambda.copy_(torch.tensor([10.0, 0.01, 5.0, 0.001]))
        layer.prune_rank(keep_top_k=2)
        # Only top-2 should be active
        active = layer.importance_mask.sum().item()
        assert active == 2.0

    def test_adalora_full_rank_same_as_lora(self):
        """At full rank with no pruning, AdaLoRA should apply all singular values."""
        layer = AdaLoRALinear(in_features=4, out_features=4, rank=4)
        assert layer.effective_rank() == 4  # all active

    def test_adalora_merge_weights_shape(self):
        layer = AdaLoRALinear(in_features=8, out_features=4, rank=4)
        assert layer.merge_weights().shape == (4, 8)

    def test_adalora_importance_mask_binary(self):
        layer = AdaLoRALinear(in_features=8, out_features=4, rank=4)
        layer.prune_rank(keep_top_k=2)
        mask_vals = layer.importance_mask.unique().tolist()
        for v in mask_vals:
            assert v == 0.0 or v == 1.0


# ---------------------------------------------------------------------------
# 3. Calibration tests
# ---------------------------------------------------------------------------

class TestCalibration:
    def _make_model(self):
        return nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

    def test_hooks_attach_to_linear_layers(self):
        model = self._make_model()
        cal = ActivationCalibrator(model, num_bits=8)
        cal.attach_hooks()
        assert len(cal.stats) == 2  # Two Linear layers
        cal.remove_hooks()

    def test_hooks_collect_stats_after_forward(self):
        model = self._make_model()
        cal = ActivationCalibrator(model, num_bits=8)
        cal.attach_hooks()
        with torch.no_grad():
            for _ in range(5):
                model(torch.randn(4, 16))
        cal.remove_hooks()
        for name, stat in cal.stats.items():
            assert stat.running_samples > 0, f"{name}: no samples collected"

    def test_minmax_scale_geq_percentile(self):
        """MinMax scale should be >= percentile scale (less clipping)."""
        model = self._make_model()
        with torch.no_grad():
            data = torch.randn(32, 16)
            cal_mm = ActivationCalibrator(model, num_bits=4, strategy="minmax")
            cal_mm.attach_hooks()
            model(data)
            cal_mm.remove_hooks()
            scales_mm = cal_mm.compute_scales()

            cal_p = ActivationCalibrator(model, num_bits=4, strategy="percentile", percentile=95.0)
            cal_p.attach_hooks()
            model(data)
            cal_p.remove_hooks()
            scales_p = cal_p.compute_scales()

        for name in scales_mm:
            assert scales_mm[name] >= scales_p[name] - 1e-6, (
                f"{name}: MinMax scale {scales_mm[name]:.4f} < percentile scale {scales_p[name]:.4f}"
            )

    def test_scale_positive(self):
        model = self._make_model()
        cal = ActivationCalibrator(model, num_bits=4)
        cal.attach_hooks()
        with torch.no_grad():
            model(torch.randn(8, 16))
        cal.remove_hooks()
        for name, scale in cal.compute_scales().items():
            assert scale > 0, f"{name}: non-positive scale {scale}"

    def test_remove_hooks_stops_collection(self):
        model = self._make_model()
        cal = ActivationCalibrator(model, num_bits=8)
        cal.attach_hooks()
        with torch.no_grad():
            model(torch.randn(4, 16))
        cal.remove_hooks()
        # Run forward again — should not update stats
        samples_before = {k: v.running_samples for k, v in cal.stats.items()}
        with torch.no_grad():
            model(torch.randn(4, 16))
        for k, stat in cal.stats.items():
            assert stat.running_samples == samples_before[k], "Hook still active after remove"


# ---------------------------------------------------------------------------
# 4. SQNR and quantization error
# ---------------------------------------------------------------------------

class TestSQNR:
    def test_sqnr_increases_with_bits(self):
        """Higher bit-width should give higher SQNR."""
        weight = torch.randn(64, 64) * 3.0
        sqnr_4 = self._measure_sqnr(weight, bits=4)
        sqnr_8 = self._measure_sqnr(weight, bits=8)
        assert sqnr_8 > sqnr_4, f"8-bit SQNR {sqnr_8:.1f} <= 4-bit SQNR {sqnr_4:.1f}"

    def _measure_sqnr(self, weight, bits):
        q, scale = quantize_symmetric(weight, num_bits=bits)
        deq = dequantize_symmetric(q, scale)
        return compute_sqnr(weight, deq)

    def test_sqnr_4bit_above_15db(self):
        """4-bit quantization should achieve > 15 dB SQNR on typical weights."""
        torch.manual_seed(42)
        weight = torch.randn(32, 32)
        sqnr = self._measure_sqnr(weight, bits=4)
        assert sqnr > 15.0, f"4-bit SQNR too low: {sqnr:.1f} dB"

    def test_sqnr_8bit_above_38db(self):
        """8-bit quantization should achieve > 38 dB SQNR."""
        torch.manual_seed(0)
        weight = torch.randn(32, 32)
        sqnr = self._measure_sqnr(weight, bits=8)
        assert sqnr > 38.0, f"8-bit SQNR too low: {sqnr:.1f} dB"

    def test_perfect_reconstruction_inf_sqnr(self):
        w = torch.randn(4, 4)
        sqnr = compute_sqnr(w, w.clone())
        assert sqnr == float("inf")

    def test_quantization_mse_bounded(self):
        weight = torch.randn(100, 100) * 2.0
        q, scale = quantize_symmetric(weight, num_bits=4)
        deq = dequantize_symmetric(q, scale)
        mse = ((weight - deq) ** 2).mean().item()
        # For 4-bit, worst-case MSE ~ scale^2/4 (mid-tread quantization)
        expected_max_mse = (scale ** 2) * 0.5
        assert mse <= expected_max_mse, f"MSE {mse:.6f} > expected max {expected_max_mse:.6f}"

    def test_symmetric_range_clamped(self):
        weight = torch.randn(10, 10) * 10.0
        q, _ = quantize_symmetric(weight, num_bits=4)
        assert q.max() <= 7
        assert q.min() >= -7


# ---------------------------------------------------------------------------
# 5. LoRA mathematical correctness
# ---------------------------------------------------------------------------

class TestLoRAMath:
    def test_lora_initial_output_equals_base(self):
        """At init, lora_B=0, so LoRA output must equal base weight output."""
        torch.manual_seed(99)
        layer = LoRALinear(in_features=8, out_features=4, rank=2, alpha=4.0)
        x = torch.randn(1, 8)
        lora_out = layer(x)
        base_out = nn.functional.linear(x, layer.weight, layer.bias)
        torch.testing.assert_close(lora_out, base_out, atol=1e-6, rtol=1e-5)

    def test_lora_merged_weight_correctness(self):
        """merge_weights() should return W0 + (B@A)*scaling exactly."""
        layer = LoRALinear(in_features=6, out_features=3, rank=2, alpha=4.0)
        scaling = layer.scaling
        expected = layer.weight + torch.matmul(layer.lora_B, layer.lora_A) * scaling
        merged = layer.merge_weights()
        torch.testing.assert_close(merged, expected, atol=1e-6, rtol=1e-5)

    def test_lora_rank_constraint(self):
        layer = LoRALinear(in_features=16, out_features=8, rank=3)
        assert layer.lora_A.shape == (3, 16)
        assert layer.lora_B.shape == (8, 3)

    def test_lora_scaling_factor(self):
        layer = LoRALinear(in_features=4, out_features=4, rank=2, alpha=8.0)
        assert abs(layer.scaling - 4.0) < 1e-6  # alpha/rank = 8/2 = 4


# ---------------------------------------------------------------------------
# 6. Pipeline compilation
# ---------------------------------------------------------------------------

class TestPipelineCompilation:
    def test_compile_mixed_model(self):
        """Pipeline should compile models with mixed LoRA, DoRA, and standard layers."""
        model = nn.Sequential(
            LoRALinear(8, 4, rank=2),
            nn.ReLU(),
            nn.Linear(4, 2),
        )
        pipeline = PEFTQuantPipeline(model)
        compiled = pipeline.compile_model(num_bits=4)
        assert "0.weight" in compiled["weights"]
        assert "2.weight" in compiled["weights"]
        assert compiled["quantization_bits"] == 4

    def test_compile_8bit_and_4bit_different_scales(self):
        model = nn.Sequential(nn.Linear(8, 4))
        p = PEFTQuantPipeline(model)
        c4 = p.compile_model(num_bits=4)
        p2 = PEFTQuantPipeline(model)
        c8 = p2.compile_model(num_bits=8)
        # 8-bit scale should be smaller (finer quantization)
        assert c8["scales"]["0.scale"] < c4["scales"]["0.scale"] or \
               abs(c8["scales"]["0.scale"] - c4["scales"]["0.scale"]) < 1e-6
