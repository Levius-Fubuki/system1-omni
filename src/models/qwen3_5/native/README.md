# Shared Qwen3.5/3.8 native executor

This crate provides the language prefill executor, runtime-loaded CUDA bindings,
ordered JSON helpers and shared image/vision execution used by Cua-S1,
Open-Jev and JEMM. Model-specific workers own request contracts, preparation,
calibration, response reconstruction and scheduler admission. The crate owns
model math, validated weight layouts, buffers and device synchronization.

The language model supports a single token sequence, multimodal sequences with
explicit image-token indices and three-axis positions, and bounded packed
prefill. Open-Jev-27B packs independent candidates for selected GEMMs; Cua-S1
and JEMM use single-prompt language calls. JEMM performs one prefill per question
with all candidates in that prompt. Sharing an executor does not make these
workers' prompt or head contracts interchangeable.

## Shared vision

`image_preprocess` provides RGB conversion, smart resizing, interpolation,
normalization, temporal duplication and spatially merged patch ordering.
Callers supply source-pixel, resized-pixel and patch limits; JEMM's pinned
processor limits differ from Cua-S1's. Shared transforms do not choose request
validation policy or resize images to fit a question's token budget.

`vision` owns structurally validated BF16 checkpoints and configurable GPU
execution. Only the exact supported configurations are accepted:

| Configuration | Blocks | Hidden width | MLP width | Attention | Output width |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B / Cua-S1 | 24 | 1024 | 4096 | 16 heads × 64 | 2560 |
| Qwen3.8-27B / JEMM | 27 | 1152 | 4304 | 16 heads × 72 | 5120 |

Both use RGB images, 16-pixel spatial patches, temporal patch size 2, spatial
merge size 2 and a 2304-entry position table, without DeepStack outputs.
Tensor inventories, dimensions, BF16 dtypes and language-output widths are
validated before device loading. Export files remain immutable while mapped.

Cua-S1's 4B vision path retains its legacy kernel calls and supported vision
LoRA behavior. JEMM uses unadapted base vision weights; its adapter targets
language modules. The 27B path requires optional ABI 5 `*_v2` position, rotary
and attention symbols. Rebuild `libqwen3_5_cuda.so` for JEMM; an older ABI 5
library can still serve legacy consumers but cannot execute 27B vision.

The [CUDA backend](../../../backends/cuda/qwen3_5/README.md) describes build
commands and these additive symbols. The [JEMM recipe](../../../../recipe/jemm/native.md)
explains pinned export, request limits and whole-request execution. Shared vision
per-image computation is reused across JEMM questions within that request;
there is no cross-request image cache or dynamic batching.

## Validation scope

CPU tests in [`tests/qwen3_5/`](../../../../tests/qwen3_5/) cover supported
configuration and tensor inventories, resizing limits, rotary geometry and
prepared multimodal input constraints. Opt-in kernel tests compare language
operations against rounded/float64 references, retain exact legacy 4B position,
rotary and attention outputs, and exercise 1152-wide, 72-dimensional vision
operations. The [JEMM evidence](../../../../recipe/jemm/validation.md) records
A800 kernel and retained 4B full-feature/stage checks alongside completed JEMM
preprocessing/response parity and matched warm HTTP timings for the fixed
corpus. That reference configured SDPA and had FLA installed, with the Torch convolution fallback
because optional `causal_conv1d` was absent; other hardware and general
accuracy or production throughput remain outside the evidence.

These checks do not expand model-specific hardware claims. Refer to each
worker's recipe for its observed execution and supported input contract.
