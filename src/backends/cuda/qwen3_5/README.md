# Qwen3.5/3.8 prefill operations

CUDA kernels for a prefill-only Qwen3.5/3.8 forward pass, built into `libqwen3_5_cuda.so` with a C interface ([`ops.h`](ops.h)), so that a Rust model engine loads it at run time and builds without a CUDA toolkit. The Cua-S1, Open-Jev and JEMM native workers share the layer loop and buffers in [`src/models/qwen3_5/native/`](../../../models/qwen3_5/native/).

```sh
src/backends/cuda/qwen3_5/build.sh <output dir> [compute capability, default 89]
```

The norm, elementwise and q/k preparation kernels round to bfloat16 where Transformers (`modeling_qwen3_5.py`) does. Attention (FlashAttention-2 style, on tensor cores) and the chunked gated delta rule keep some intermediate results in bfloat16, as FlashAttention and flash-linear-attention do. GEMMs go through cuBLASLt with its first heuristic choice. Tensor-core kernels need sm_80 or newer; PR #19 validated the original kernels on sm_89. The current reference tests, including fused gating, cached residual RMSNorm and packed SiLU, passed on H200 (sm_90). Compilation passed for sm_80, sm_89 and sm_90. The current language kernel suite, including GDN, also passed on A800 (sm_80) during the [JEMM checks](../../../../recipe/jemm/validation.md); execution of the modified GDN kernel on sm_89 remains unverified.

`cs1_attention_gated` fuses the sigmoid gate into the attention epilogue, preserving
the BF16 rounding of both attention and sigmoid before multiplication. The native
workers use this entry point; the separate operations remain available for kernel
comparisons. Rebuild the library and workers together for ABI version 5, which
includes the shared vision and CUDA Graph entry points alongside gated attention.

Gated DeltaNet preparation stores converted TF32 operands in three-byte component planes, preserves the original four-term TF32 accumulation, and writes U/W fragments directly as bfloat16. Dynamic shared memory is 72 KiB per block. The [H200 comparison](../../../../benchmarks/gdn/README.md) records complete GDN call latency, numerical checks, and the small end-to-end change measured with the Open-Jev worker from PR #55.

The shared Rust model can pack independent sequences for input and gate/up GEMMs.
Output/down GEMMs retain each prompt's original shape and reduction order;
attention, convolution and GDN calls remain sequence-local. The CUDA ABI is
unchanged. Open-Jev uses this path within requests; Cua-S1 and JEMM keep
single-prompt language calls.


## Configurable Qwen vision

The shared Rust vision executor accepts the exact Qwen3.5-4B and Qwen3.8-27B
layouts: 1024 hidden with 16 heads of 64, or 1152 hidden with 16 heads of 72.
The new `cs1_vision_position_v2`, `cs1_vision_rope_v2` and
`cs1_vision_attention_v2` endpoints add explicit dimensions; the attention
endpoint also accepts caller-owned scratch space. Rotary uses the layout's
head dimension, including 36 frequencies for the 72-dimensional head.
Vision attention is full, image-local attention; separate images are not
concatenated into one attention sequence.

These are additive optional symbols under **ABI version 5**. The legacy vision
signatures remain unchanged. Existing 4B dispatch still calls the legacy
position/rotary/attention endpoints, and old ABI 5 libraries remain loadable by
legacy consumers. JEMM's 27B encoder requires all three v2 symbols, so rebuild
the library for it. Unsupported layouts are rejected before launching kernels.

```sh
# A800 build; pass the actual architecture for another GPU.
src/backends/cuda/qwen3_5/build.sh target/release 80
CUA_S1_CUDA_LIB="$PWD/target/release/libqwen3_5_cuda.so" \
  cargo test --release --locked -p omni-qwen3-5-native --test vision_kernels \
  -- --ignored --test-threads=1
```

The opt-in vision tests cover exact retained 4B operations, 1152-wide position
interpolation and rounding, 72-dimensional rotary multiplication order, and
full attention against a float64 reference with image isolation. A800 evidence
also compares retained 4B full features and stage hashes at three sizes with
the original implementation. These checks establish those shared boundaries.
[JEMM A800 validation](../../../../recipe/jemm/validation.md) also records exact
preprocessing, fixed-corpus response parity and matched warm HTTP timings
against the official unmerged reference. SDPA was configured and FLA was installed in that
reference, while optional `causal_conv1d` was absent and convolution used the
Torch fallback. This evidence does not validate other hardware or establish
general accuracy or production throughput.
