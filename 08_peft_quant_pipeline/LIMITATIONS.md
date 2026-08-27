"""
LIMITATIONS.md — PEFT-Quant Pipeline
====================================

# Known Limitations and Future Work

## PEFT Adapters (LoRA / DoRA / AdaLoRA)

### 1. Sequential Layer Dependency
Our current weight merging mechanism (`merge_weights`) folds adapter updates into the base weight:
$$W_{\text{merged}} = W_0 + B \cdot A \cdot \text{scaling}$$
This assumes that the base layer and adapters are linear and can be mathematically combined. This does not work if intermediate operations (e.g. non-linearities, layer normalization, residual routing) exist between the base linear layer and the adapter.

### 2. High Memory Requirement for AdaLoRA SVD computation
AdaLoRA parameterises weight updates as $P \cdot \Lambda \cdot Q$. To prune parameters dynamically, it tracks the importance scores of singular values. For very large models:
- Computing importance scores requires keeping a running average of gradients.
- This doubles the memory footprint of optimizer states compared to standard LoRA.

---

## Quantization (Uniform / PTQ)

### 1. Lack of Outlier-Aware Mixed Precision
Standard symmetric 4-bit/8-bit quantization scales all values uniformly using a single scaling factor per tensor (or per channel).
In large language models (LLMs above 6.7B parameters), **systematic outlier features** emerge (activation values up to 100× the typical range). Quantizing these uniformly causes:
- Massive clipping error if scaling factor is small.
- Severe precision loss for normal-range values if scaling factor is large (due to outliers dominating).

To solve this, a production-grade quantizer requires mixed-precision quantization (e.g. keeping 1% outlier channels in FP16, and the remaining 99% in INT4).

### 2. No Dynamic Range Adaptation during Inference
Our quantization pipeline is post-training (PTQ) and static. Dynamic quantization, which computes activation scales dynamically per token at runtime, would preserve more precision for activation-sensitive workloads but is not supported.

---

## Performance & Dev Ops

### 1. Lacks C++/CUDA Kernels for INT4 MatMul
Because Python cannot execute 4-bit arithmetic natively (Python integers are at least 28 bytes), the model is compiled by packing weights, but inference is done by dequantizing weights back to float32 before calling `torch.matmul`.
- **Memory footprint**: Reduced on disk/RAM.
- **Inference speed**: No speedup (actually slightly slower due to dequantization overhead on the fly).
- To achieve real speedups, one must write a custom CUDA kernel that directly computes `INT4_weight @ FP16_activation` using bit-shifts and accumulator registers.
"""
