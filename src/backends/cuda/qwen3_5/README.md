# Qwen3.5/3.8 prefill operations

CUDA kernels for a prefill-only Qwen3.5/3.8 forward pass, built into `libqwen3_5_cuda.so` with a C interface ([`ops.h`](ops.h)), so that a Rust model engine loads it at run time and builds without a CUDA toolkit. The Cua-S1 and Open-Jev native workers share the layer loop and buffers in [`src/models/qwen3_5/native/`](../../../models/qwen3_5/native/).

```sh
src/backends/cuda/qwen3_5/build.sh <output dir> [compute capability, default 89]
```

The norm, elementwise and q/k preparation kernels round to bfloat16 where Transformers (`modeling_qwen3_5.py`) does. Attention (FlashAttention-2 style, on tensor cores) and the chunked gated delta rule keep some intermediate results in bfloat16, as FlashAttention and flash-linear-attention do. GEMMs go through cuBLASLt with its first heuristic choice. Tensor-core kernels need sm_80 or newer; PR #19 validated the original kernels on sm_89. The current reference tests, including fused gating, cached residual RMSNorm and packed SiLU, passed on H200 (sm_90). Compilation passed for sm_80, sm_89 and sm_90; execution of the modified GDN kernel on sm_80/sm_89 remains unverified.

`cs1_attention_gated` fuses the sigmoid gate into the attention epilogue, preserving
the BF16 rounding of both attention and sigmoid before multiplication. The native
workers use this entry point; the separate operations remain available for kernel
comparisons. Rebuild the library and workers together for ABI version 8, which
includes shared vision and CUDA Graph entry points, request-local continuation
state/KV operations and fixed-algorithm GEMM selection alongside gated attention.

Gated DeltaNet preparation stores converted TF32 operands in three-byte component planes, preserves the original four-term TF32 accumulation, and writes U/W fragments directly as bfloat16. Dynamic shared memory is 72 KiB per block. The [H200 comparison](../../../../benchmarks/gdn/README.md) records complete GDN call latency, numerical checks, and the small end-to-end change measured with the Open-Jev worker from PR #55.

The shared Rust model can pack independent sequences for input and gate/up GEMMs.
Output/down GEMMs retain each prompt's original shape and reduction order;
attention, convolution and GDN calls remain sequence-local. Packed prefill preserves per-sequence state. Open-Jev uses this path within
requests; Cua-S1 keeps single-prompt calls. Decider optionally packs complete
question rows within one admitted request.

ABI8 retains `cs1_copy_rows`, `cs1_gdn_conv_history`, `cs1_gdn_prefill_state`,
`cs1_attention_gated_cached` and `cs1_gemm_create_fixed` for request-local prefix
continuations, alongside JEV-VL's `cs1_copy_dd`, `cs1_copy2d`, `cs1_gdn_prefill_x`
and `cs1_attention_gated_prefix`. Full-buffer prefix attention and suffix-only
cached attention retain separate kernels and indexing contracts. Shared spans end on 64-token GDN chunk boundaries. Fixed GEMM
selection fails explicitly if a requested shape cannot use the selected algorithm;
M-independent output equality is a device-tested requirement, not a portable
cuBLAS guarantee. See the [Decider recipe](../../../../recipe/decider/README.md).
Rebuild `libqwen3_5_cuda.so` and restart every Qwen consumer together; the Rust
loader rejects an older ABI at startup with a rebuild hint.

JEV-VL retains full-attention KV, three convolution input rows and FP32 GDN state
at 64-token chunk boundaries through its model-bound capture/continuation APIs.

ABI8 requires the union of the former ABI6 JEV-VL and ABI7 Decider interfaces.
Rebuild the library and every consumer together; both earlier versions are rejected.
The public model-bound `PrefixState` for multimodal continuation remains distinct
from the request-local `SharedPrefixState` used by Decider. Current-branch GPU
regression is required before release; earlier campaign results do not validate
this combined interface.
