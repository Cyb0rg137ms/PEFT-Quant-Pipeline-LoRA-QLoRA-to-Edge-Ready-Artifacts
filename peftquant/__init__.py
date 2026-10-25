"""peftquant: LoRA/DoRA/AdaLoRA adapters, RTN/NF4/GPTQ weight quantisation, verified export, toy GRPO."""

from peftquant.calibration import ActivationCalibrator, compute_sqnr
from peftquant.device import DeviceInfo, detect_device
from peftquant.dora import AdaLoRALinear, DoRALinear
from peftquant.export import (
    export_artifact,
    load_artifact,
    model_nbytes,
    quantize_model,
)
from peftquant.gptq import gptq_quantize, hessian
from peftquant.inject import (
    adapters_disabled,
    count_parameters,
    inject_adapters,
    merge_adapters,
)
from peftquant.lora import LoRALinear
from peftquant.quant import (
    QTensor,
    QuantLinear,
    QuantSpec,
    nf4_codebook,
    pack_bits,
    pack_int4,
    quantize,
    sqnr_db,
    unpack_bits,
    unpack_int4,
)
from peftquant.verify import FreivaldsReport, freivalds, verify_lora_merge

__version__ = "2.0.0"
__all__ = [
    "ActivationCalibrator",
    "AdaLoRALinear",
    "DeviceInfo",
    "DoRALinear",
    "FreivaldsReport",
    "LoRALinear",
    "QTensor",
    "QuantLinear",
    "QuantSpec",
    "adapters_disabled",
    "compute_sqnr",
    "count_parameters",
    "detect_device",
    "export_artifact",
    "freivalds",
    "gptq_quantize",
    "hessian",
    "inject_adapters",
    "load_artifact",
    "merge_adapters",
    "model_nbytes",
    "nf4_codebook",
    "pack_bits",
    "pack_int4",
    "quantize",
    "quantize_model",
    "sqnr_db",
    "unpack_bits",
    "unpack_int4",
    "verify_lora_merge",
]
