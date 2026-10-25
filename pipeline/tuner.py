"""Back-compat shim: ``pipeline.tuner`` now lives in :mod:`peftquant.lora`."""

from peftquant.lora import LoRALinear

__all__ = ["LoRALinear"]
