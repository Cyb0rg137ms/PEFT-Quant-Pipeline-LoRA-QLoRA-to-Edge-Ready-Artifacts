# Architecture

```mermaid
flowchart LR
    M[nn.Module] -->|inject_adapters regex/predicate| A[LoRA / DoRA layers]
    A -->|train adapters only| T[fine-tuned model]
    G[grpo.py: group-normalised advantages] --> T
    T -->|verify_lora_merge: Freivalds| V{merge correct?}
    V -->|merge_adapters| F[plain nn.Linear model]
    F -->|quantize_model rtn| Q[QuantLinear layers]
    F -->|calib inputs -> hessian -> gptq_quantize| Q
    Q -->|export_artifact| E[(weights.pt + manifest.json<br/>sha256, bytes, probe error)]
    E -->|load_artifact: hash check| L[rebuilt quantised model]
```

| module | responsibility |
|---|---|
| `peftquant.lora` | `LoRALinear`: no-op init, `from_linear`, merge/unmerge, `to_linear`, enable flag |
| `peftquant.dora` | `DoRALinear` (magnitude/direction), `AdaLoRALinear` (SVD-form, rank pruning, orthogonality reg.) |
| `peftquant.inject` | in-place adapter injection, merge back to `nn.Linear`, parameter counts, `adapters_disabled` |
| `peftquant.quant` | `QuantSpec`, `QTensor` (bit-packed codes, fp16 scales), RTN sym/asym/NF4, `QuantLinear` |
| `peftquant.gptq` | Hessian from calibration activations, GPTQ with lazy block updates |
| `peftquant.verify` | Freivalds checks for `C = AB` and `W' = W + sBA` |
| `peftquant.export` | whole-model quantisation, artefact export/load with SHA-256 and recorded probe error |
| `peftquant.calibration` | activation statistics (minmax / true percentile) |
| `peftquant.grpo` | toy GRPO with LoRA (CPU, seconds) |
| `peftquant.device` | CUDA / ROCm (HIP) / MPS / CPU detection |
| `pipeline.*` | v1 API shims (`pipeline.tuner`, `pipeline.quantizer`, `pipeline.compile`, ...) |
