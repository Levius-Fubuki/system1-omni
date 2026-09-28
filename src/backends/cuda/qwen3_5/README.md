# Qwen3.5 prefill operations

CUDA kernels for a prefill-only Qwen3.5 forward pass, built into `libqwen3_5_cuda.so`:

```sh
src/backends/cuda/qwen3_5/build.sh <output dir> [compute capability, default 89]
```

The library has a C interface ([`ops.h`](ops.h)): the operations, plus the few CUDA runtime calls a caller needs (allocation, copies, streams, graph capture), so that a Rust model engine can load it at run time and build without a CUDA toolkit. The Cua-S1 native worker ([`src/models/cua_s1/native/`](../../../models/cua_s1/native/)) uses it; the layer loop, buffers, CUDA graphs and GEMM algorithm choice stay in that model engine.

| File | Operations |
| --- | --- |
| `norm.cu` | Zero-centred RMSNorm, the residual add fused with the next norm, and the gated RMSNorm of the Gated DeltaNet output. |
| `elementwise.cu` | Embedding lookup; the Gated DeltaNet causal convolution with SiLU and its gates; the attention output gate; SiLU(gate) * up. |
| `attention.cu` | q/k RMSNorm and partial rotary embedding; causal attention with grouped KV heads (head dim 256) on tensor cores, FlashAttention-2 style. |
| `gdn_prefill.cu` | The chunked gated delta rule (chunks of 64) in three kernels, split the way flash-linear-attention splits it: per-chunk preparation, the state carried from chunk to chunk, and the per-chunk output. |
| `gemm.cu` | bfloat16 GEMMs through cuBLASLt with float32 accumulation, algorithm tuning, and saving and loading the tuned choices. |
| `runtime.cu` | The CUDA runtime calls. |
| `mma.cuh`, `common.cuh` | `mma.sync`, `ldmatrix` and `cp.async` helpers, and shared device helpers. |

The norm, elementwise and q/k preparation kernels round to bfloat16 at the same points as the Transformers implementation (`modeling_qwen3_5.py`). Attention and the Gated DeltaNet prefill keep some intermediate results in bfloat16, as FlashAttention and flash-linear-attention do, where Transformers' float32 fallback for the gated delta rule does not; the model is checked end to end against the float32 reference worker. Tensor-core kernels need sm_80 or newer; they are measured on sm_89 (RTX 6000 Ada). Split-K reductions that accumulate into the output in place are not used, and plans that use them are refused on import, so a given GEMM algorithm always gives the same result. `build.sh` also embeds PTX, but only sm_89 has been run.
