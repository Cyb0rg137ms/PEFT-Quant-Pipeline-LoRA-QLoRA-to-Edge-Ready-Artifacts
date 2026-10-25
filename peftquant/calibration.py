"""
calibration.py
==============
Post-training quantization calibration pass.

Collects activation statistics (min, max, per-channel absmax) over a
calibration dataset to compute optimal quantization scales before quantizing
the model. Two calibration strategies are supported:

  1. **MinMax**: Scale = max(|tensor|) / (2^(bits-1) - 1)
     Simple, fast, sensitive to outliers.

  2. **Percentile**: Scale = percentile(|tensor|, p) / (2^(bits-1) - 1)
     More robust to outlier activations at the cost of slight clipping.

Usage::

    calibrator = ActivationCalibrator(model, num_bits=4, strategy="percentile")
    calibrator.attach_hooks()

    with torch.no_grad():
        for batch_x, _ in calibration_loader:
            model(batch_x)

    scales = calibrator.compute_scales()
    calibrator.remove_hooks()
"""

from __future__ import annotations

import torch
import torch.utils.hooks
from torch import nn


class LayerCalibrationStats:
    """Accumulates per-layer activation statistics across calibration batches."""

    def __init__(self, max_samples_per_batch: int = 65536, seed: int = 0) -> None:
        self.min_vals: list[float] = []
        self.max_vals: list[float] = []
        self.abs_max_vals: list[float] = []
        self.running_samples: int = 0
        self.max_samples_per_batch = max_samples_per_batch
        self._samples: list[torch.Tensor] = []
        self._gen = torch.Generator().manual_seed(seed)

    def update(self, tensor: torch.Tensor) -> None:
        flat = tensor.detach().float().reshape(-1)
        self.min_vals.append(flat.min().item())
        self.max_vals.append(flat.max().item())
        self.abs_max_vals.append(flat.abs().max().item())
        self.running_samples += flat.numel()
        a = flat.abs().cpu()
        if a.numel() > self.max_samples_per_batch:
            a = a[torch.randint(0, a.numel(), (self.max_samples_per_batch,), generator=self._gen)]
        self._samples.append(a)

    def global_min(self) -> float:
        return min(self.min_vals) if self.min_vals else 0.0

    def global_max(self) -> float:
        return max(self.max_vals) if self.max_vals else 0.0

    def global_absmax(self) -> float:
        return max(self.abs_max_vals) if self.abs_max_vals else 1.0

    def percentile_absmax(self, p: float = 99.9) -> float:
        """p-th percentile of ``|activation|`` over all calibration batches.

        v1 took a percentile over the per-batch *maxima* (so it never clipped
        anything with a single batch). v2 keeps a uniform subsample of up to
        ``max_samples_per_batch`` absolute values per batch and returns the exact
        percentile of that sample (exact when batches are smaller than the cap).
        """
        if not self._samples:
            return 1.0
        allv = torch.cat(self._samples)
        k = min(allv.numel(), max(1, round(allv.numel() * p / 100.0)))
        return float(allv.kthvalue(k).values)


class ActivationCalibrator:
    """
    Attaches forward hooks to all quantizable layers and collects
    input activation statistics for calibration.

    Supports ``nn.Linear``, ``nn.Conv2d``, ``LoRALinear``, and ``DoRALinear``.
    """

    QUANTIZABLE_TYPES = (nn.Linear, nn.Conv2d)

    def __init__(
        self,
        model: nn.Module,
        num_bits: int = 4,
        strategy: str = "percentile",
        percentile: float = 99.9,
    ) -> None:
        self.model = model
        self.num_bits = num_bits
        self.strategy = strategy
        self.percentile = percentile
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self.stats: dict[str, LayerCalibrationStats] = {}

    def attach_hooks(self) -> None:
        """Attach forward hooks to all linear/conv layers in the model."""
        self._hooks.clear()
        self.stats.clear()

        for name, module in self.model.named_modules():
            if isinstance(module, self.QUANTIZABLE_TYPES):
                stats = LayerCalibrationStats()
                self.stats[name] = stats

                def make_hook(s: LayerCalibrationStats):
                    def hook(m, inp, out):
                        if inp and inp[0] is not None:
                            s.update(inp[0])

                    return hook

                h = module.register_forward_hook(make_hook(stats))
                self._hooks.append(h)

    def remove_hooks(self) -> None:
        """Remove all attached hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def compute_scales(self) -> dict[str, float]:
        """
        Computes optimal quantization scale for each calibrated layer.

        Returns:
            Dictionary mapping layer name to quantization scale factor.
        """
        q_bound = (2 ** (self.num_bits - 1)) - 1
        scales: dict[str, float] = {}

        for name, stat in self.stats.items():
            if self.strategy == "minmax":
                absmax = stat.global_absmax()
            elif self.strategy == "percentile":
                absmax = stat.percentile_absmax(self.percentile)
            else:
                raise ValueError(f"Unknown calibration strategy: {self.strategy!r}")

            scale = absmax / q_bound if absmax > 0 else 1.0
            scales[name] = scale

        return scales

    def calibrate_from_tensors(self, tensors: list[torch.Tensor]) -> dict[str, float]:
        """
        Convenience method: calibrate without running the model.
        Directly provides weight tensors for per-layer calibration.

        Args:
            tensors: List of weight tensors to calibrate.

        Returns:
            List of computed scale factors.
        """
        q_bound = (2 ** (self.num_bits - 1)) - 1
        result = {}
        for i, t in enumerate(tensors):
            absmax = t.abs().max().item()
            if self.strategy == "percentile":
                # Use the 99.9th percentile of absolute values
                flat = t.detach().float().reshape(-1).abs()
                k = max(1, int(len(flat) * self.percentile / 100.0))
                absmax = float(flat.kthvalue(k).values.item())
            scale = absmax / q_bound if absmax > 0 else 1.0
            result[f"tensor_{i}"] = scale
        return result


def compute_sqnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Computes Signal-to-Quantization-Noise Ratio (SQNR) in dB.

    SQNR = 10 * log10(E[signal^2] / E[noise^2])

    Higher is better. 4-bit quantization typically yields ~24 dB SQNR;
    8-bit typically yields ~48 dB SQNR.

    Args:
        original: The original float tensor.
        reconstructed: The dequantised tensor.

    Returns:
        SQNR in decibels (float).
    """
    signal_power = (original**2).mean().item()
    noise = original - reconstructed
    noise_power = (noise**2).mean().item()

    if noise_power < 1e-12:
        return float("inf")  # Perfect reconstruction

    import math

    return 10.0 * math.log10(signal_power / noise_power)
