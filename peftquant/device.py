"""Accelerator detection: CUDA, ROCm (HIP builds of PyTorch expose ``torch.cuda``), Apple MPS, CPU."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    backend: str  # "cuda" | "rocm" | "mps" | "cpu"
    name: str


def detect_device(prefer: str | None = None) -> DeviceInfo:
    """Pick the best available device (or honour ``prefer`` = cpu/cuda/mps)."""
    if prefer == "cpu":
        return DeviceInfo(torch.device("cpu"), "cpu", "cpu")
    if torch.cuda.is_available() and prefer in (None, "cuda", "rocm"):
        backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        return DeviceInfo(torch.device("cuda"), backend, torch.cuda.get_device_name(0))
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and prefer in (None, "mps"):
        return DeviceInfo(torch.device("mps"), "mps", "Apple MPS")
    if prefer not in (None, "cpu"):
        raise RuntimeError(f"requested device {prefer!r} is not available")
    return DeviceInfo(torch.device("cpu"), "cpu", "cpu")
