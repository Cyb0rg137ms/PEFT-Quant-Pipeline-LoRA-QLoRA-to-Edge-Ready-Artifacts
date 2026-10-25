# Limitations (read before using this for anything real)

- **Reference kernels only.** `QuantLinear` dequantises to fp32 and calls `F.linear`; it saves *storage*, not
  compute or peak memory at inference. No fused int4 GEMM, no CUDA/ROCm kernels.
- **GPTQ is the one-pass variant**: layer inputs are captured from the float model, not re-captured after
  quantising earlier layers (the sequential variant in the paper). No act-order / static groups.
- **NF4 is RTN only** (no GPTQ), and there is no double quantisation of scales (QLoRA's "DQ").
- **Benchmarks are synthetic**: a Gaussian-mixture MLP classifier and random matrices on CPU. They show the
  relative behaviour of RTN vs GPTQ and the real byte counts; they say nothing about LLM perplexity.
  On this easy task accuracy differences between RTN and GPTQ at 3-4 bits are within noise; the logit error
  differences are the meaningful signal.
- **GRPO is a toy** (factorised policy, reverse-the-prompt reward). It shows the algorithm is implemented
  correctly and improves reward; it is not an LLM RL result.
- **Freivalds is probabilistic with a float tolerance** (`rtol=1e-4`): corruptions smaller than that relative
  size are not detected. It certifies arithmetic, not model quality.
- **Artefact format is this project's own** (torch `weights.pt` + JSON manifest). Not GGUF/ONNX/safetensors.
- Tensor-granularity sym stores one scale per tensor in accounting but keeps a broadcast copy in memory.
- ROCm detection relies on `torch.version.hip`; nothing has been tested on real AMD/NVIDIA hardware here
  (CI and all numbers are CPU-only).
