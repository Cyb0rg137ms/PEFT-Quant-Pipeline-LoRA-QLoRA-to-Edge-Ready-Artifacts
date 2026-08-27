# PEFT-Quant-Pipeline — Architecture & Technical Reference

> **Full Project Name:** PEFT-Quant-Pipeline — Parameter-Efficient Fine-Tuning & Quantization
> **Category:** LLM Optimization / Production ML / Edge Deployment
> **Language:** Python 3.9+, PyTorch
> **Test Coverage:** 4/4 unit tests passing ✅

---

## 1. Architecture Overview

```
08_peft_quant_pipeline/
├── pipeline/
│   ├── lora.py          # LoRA adapter: rank decomposition + freezing + merging
│   ├── quantizer.py     # Uniform INT8 + dynamic quantization engine
│   ├── compiler.py      # Pipeline orchestrator: LoRA → quantize → export
│   └── __init__.py
├── tests/
│   └── test_pipeline.py
├── run_pipeline.py      # CLI: fine-tune + compress a model
└── README.md
```

### Component Interaction

```
┌────────────────────────────────────────────────────────────────┐
│                 PEFT-QUANT-PIPELINE                           │
│                                                               │
│  Pretrained Model (nn.Module)                                 │
│          │                                                    │
│          ▼                                                    │
│  LoRA Adapter Layer                                           │
│    ├── Freeze all original weights (requires_grad = False)    │
│    ├── Add rank-r matrices A (d_in × r), B (r × d_out)       │
│    └── Train only A and B                                     │
│          │                                                    │
│          ▼                                                    │
│  LoRA Weight Merging                                          │
│    └── W'  =  W  +  (α / r)  ×  B × A                       │
│          │                                                    │
│          ▼                                                    │
│  INT8 Quantizer                                               │
│    ├── Compute scale:  s = max(|W|) / 127                    │
│    └── Quantize:  W_q = clamp( round(W / s), -128, 127 )    │
│          │                                                    │
│          ▼                                                    │
│  Export: TorchScript / ONNX / SafeTensors                    │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Framework

### 2.1 LoRA — Low-Rank Adaptation

Standard fine-tuning updates the entire weight matrix `W` of shape `(d_out × d_in)`.
LoRA instead constrains the update to be a low-rank product:

```
W'  =  W  +  ΔW,   where   ΔW = B × A

A  has shape  (r × d_in)      — rank-r projection DOWN
B  has shape  (d_out × r)     — rank-r projection UP
r  << min(d_out, d_in)        — rank, e.g. r = 4 or r = 8

Initialization:
  A  = Kaiming uniform random
  B  = all zeros  (so ΔW = 0 at training start, preserving pretrained behavior)

Only A and B receive gradients — the original W is frozen.
```

**Parameter savings:**

```
Standard:  d_out × d_in   parameters updated
LoRA:      r × (d_out + d_in)  parameters updated

Savings ratio  =  1  -  r × (d_out + d_in) / (d_out × d_in)

Example: d=4096, r=4:
  Standard:  16,777,216  parameters
  LoRA:      32,768      parameters
  Savings:   99.8%
```

### 2.2 LoRA Scaled Forward Pass

During fine-tuning, the forward pass for the adapted layer is:

```
h  =  W × x  +  (α / r)  ×  B × (A × x)

x   = input vector
α   = LoRA scaling hyperparameter  (e.g. α = 16)
r   = LoRA rank  (e.g. r = 4)
α/r = effective scaling factor  (controls magnitude of LoRA update)
```

### 2.3 LoRA Weight Merging (Zero Inference Overhead)

After fine-tuning, the LoRA matrices are absorbed back into W:

```
W_merged  =  W  +  (α / r)  ×  B × A

The merged model has IDENTICAL architecture to the original.
Inference cost = zero extra computation compared to base model.
```

### 2.4 Uniform INT8 Quantization

Converts 32-bit floating point weights to 8-bit integers for 4× memory reduction.

**Calibration — find the scale factor:**

```
s  =  max( |W| )  /  127

max(|W|) = largest absolute value in the weight tensor
127      = largest representable positive INT8 value
```

**Quantization:**

```
W_q  =  clamp( round( W / s ),  low=-128,  high=127 )

round()  = round to nearest integer
clamp()  = clip to valid INT8 range [-128, 127]
```

**Dequantization during inference:**

```
W_reconstructed  =  s  ×  W_q
```

**Maximum quantization error (worst case):**

```
max_error  =  s / 2  =  max(|W|) / 254

This is the largest possible difference between the true and quantized weight.
```

### 2.5 Compression Summary

| Metric | Formula | Example (d=4096, r=4) |
|--------|---------|----------------------|
| LoRA trainable params | r × (d_in + d_out) | 32,768 |
| Savings vs full fine-tune | 1 - (r(d_in+d_out)) / (d_in×d_out) | 99.8% |
| INT8 memory vs FP32 | 1 byte / 4 bytes | 4× smaller |
| Combined (LoRA + INT8) | — | up to 4× smaller model |

---

## 3. Workflow

```
Load pretrained nn.Module
        │
        ▼
LoRAAdapter: inject rank-r matrices A, B into each Linear layer
Freeze all base weights:  requires_grad = False
        │
        ▼
Fine-tune on target dataset  (only A and B update)
        │
        ▼
LoRA weight merge:  W' = W + (α/r) × B × A
        │
        ▼
INT8 Quantizer: for each Linear layer
  compute scale  s = max(|W'|) / 127
  quantize:  W_q = clamp(round(W'/s), -128, 127)
        │
        ▼
Validate quantization error < threshold
        │
        ▼
Export in chosen format:
  TorchScript  →  model.pt        (mobile / C++ runtime)
  ONNX         →  model.onnx      (cross-platform inference)
  SafeTensors  →  model.safetensors (server / HuggingFace)
```

---

## 4. System Design

| Component | Module | Role |
|-----------|--------|------|
| **Adapter** | `lora.py` | Low-rank matrix injection, parameter freezing, weight merging |
| **Quantizer** | `quantizer.py` | Uniform INT8, dynamic per-tensor scale calibration |
| **Orchestrator** | `compiler.py` | Pipeline chaining: LoRA → quantize → multi-format export |
| **CLI** | `run_pipeline.py` | Argument parsing, model loading, pipeline execution |
| **Tests** | `test_pipeline.py` | Parameter counts, weight merge correctness, quantization bounds |

---

## 5. Key Advantages

| Advantage | Description |
|-----------|-------------|
| **Minimal trainable parameters** | LoRA reduces trainable params by 99%+ at rank r=4 |
| **Zero inference overhead** | Weight merging eliminates runtime LoRA computation |
| **4× memory reduction** | INT8 quantization vs FP32 baseline |
| **Multi-format export** | Targets edge (ONNX), mobile (TorchScript), and server (SafeTensors) |
| **Production-ready** | Full pipeline from fine-tuning to deployment-ready artifact |

---

## 6. Test Results

```
tests/test_pipeline.py::test_lora_parameter_freezing PASSED
tests/test_pipeline.py::test_lora_weight_merging     PASSED
tests/test_pipeline.py::test_uniform_quantization    PASSED
tests/test_pipeline.py::test_compilation_pipeline    PASSED
────────────────────────────────────────────────────
4 passed in 0.17s
```

---

## 7. Quick Start

```bash
pip install torch
pytest tests/
python run_pipeline.py --rank 4 --alpha 16 --bits 8
```

<div align="center">
  <a href="https://q.com"><img src="../../assets/https_q_com.png" width="80" /></a>
</div>
