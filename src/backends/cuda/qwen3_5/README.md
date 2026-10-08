# Qwen3.5/3.8 prefill operations

CUDA kernels for a prefill-only Qwen3.5/3.8 forward pass, built into `libqwen3_5_cuda.so` with a C interface ([`ops.h`](ops.h)), so that a Rust model engine loads it at run time and builds without a CUDA toolkit. The Cua-S1, Open-Jev and Decider native workers share the layer loop and buffers in [`src/models/qwen3_5/native/`](../../../models/qwen3_5/native/).

```sh
src/backends/cuda/qwen3_5/build.sh <output dir> [compute capability, default 89]
```

The norm, elementwise and q/k preparation kernels round to bfloat16 where Transformers (`modeling_qwen3_5.py`) does. Attention (FlashAttention-2 style, on tensor cores) and the chunked gated delta rule keep some intermediate results in bfloat16, as FlashAttention and flash-linear-attention do. GEMMs go through cuBLASLt with its first heuristic choice. Tensor-core kernels need sm_80 or newer; PR #19 validated the original kernels on sm_89. The current reference tests, including fused gating, cached residual RMSNorm and packed SiLU, passed on H200 (sm_90). Compilation passed for sm_80, sm_89 and sm_90; execution of the modified GDN kernel on sm_80/sm_89 remains unverified.

`cs1_attention_gated` fuses the sigmoid gate into the attention epilogue, preserving
the BF16 rounding of both attention and sigmoid before multiplication. The shared
Qwen executor uses it, through `cs1_attention_gated_cached` with no cached positions;
the separate operations remain available for kernel comparisons. Rebuild the library and workers together for ABI version 7, which
includes the CUDA Graph entry points, gated attention, the vision operations, the
continuation operations and the fixed-algorithm GEMM handle below.

Three operations continue a sequence after a shared prefix, for prefix reuse
([#85](https://github.com/ThinkFlowLab/system1-omni/issues/85)). The shared Qwen
executor calls them in every pass; without history, state or cached positions they
give the plain operations' results. `forward_shared` continues a prefix; the Decider
worker uses it when `DECIDER_PREFIX=1` or the opt-in Auto policy selects reuse. Cua-S1 and Open-Jev keep their existing
independent prompt paths:

- `cs1_gdn_conv_history` reads the conv inputs of the three positions before its
  first token and can write those of its last three. Every output equals the
  unsplit conv's.
- `cs1_gdn_prefill_state` starts the chunked gated delta rule from a float32 state
  `[H, 128, 128]` and can write the final state. When every split falls on a multiple
  of 64 tokens, it reproduces the unsplit prefill bit for bit; elsewhere the chunks
  fall differently, within the float64-reference tolerance of the unsplit kernel.
- `cs1_attention_gated_cached` runs the queries of the last positions against keys
  and values that also cover the positions before them. Key tiles start at position
  0 either way, so each output row matches the unsplit call.

`cs1_copy_rows` queues a pitched device-to-device copy, for keeping cached values
in their own rows. The existing `cs1_gdn_conv`, `cs1_gdn_prefill` and
`cs1_attention_gated` are these operations without history, state or cached
positions. `tests/qwen3_5/kernels.rs` checks each against the unsplit call at
prefix lengths around and inside 64-token chunks.

cuBLASLt's heuristic picks a GEMM algorithm per M, so a row's result can change with
the number of rows in the call: a prefix and its branch, run separately, round
differently from the same tokens in one pass. `cs1_gemm_create_fixed` returns a handle
that keeps one algorithm per weight shape for every M, the heuristic's first choice at a
reference M among algorithms without split-K, so each row's result is the same whatever
M is and wherever the row sits. cuBLASLt doesn't document this property;
`fixed_gemm_rows_do_not_depend_on_m` checks it, so run it on a new GPU before relying
on it (it passed on sm_89). It is slower for some shapes and faster for others: on an
RTX 6000 Ada, the down and output projections take up to about 4 times as long at small
M without split-K. `cs1_gemm_create` remains the default.

Gated DeltaNet preparation stores converted TF32 operands in three-byte component planes, preserves the original four-term TF32 accumulation, and writes U/W fragments directly as bfloat16. Dynamic shared memory is 72 KiB per block. The [H200 comparison](../../../../benchmarks/gdn/README.md) records complete GDN call latency, numerical checks, and the small end-to-end change measured with the Open-Jev worker from PR #55.

The shared Rust model can pack independent sequences for input and gate/up GEMMs.
Output/down GEMMs retain each prompt's original shape and reduction order;
attention, convolution and GDN calls remain sequence-local. The CUDA ABI is
unchanged. Open-Jev uses this path within requests; Cua-S1 keeps single-prompt calls.

Decider-2B v11 reuses these operations for eager independent text rows and its
BF16 selected tied-embedding projection. It pads the 255 retained labels to 256
GEMM rows and excludes padding from the output. Optional request-local shared
prefix execution uses the continuation stack and requires ABI7; default eager
and Graph execution retain their independent-row math. See the [Decider recipe](../../../../recipe/decider/README.md) and
[RTX 4090 full-checkpoint scope](../../../../recipe/decider/validation.md).
