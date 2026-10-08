# Shared Qwen3.5/3.8 native executor

This crate provides language prefill, runtime-loaded CUDA bindings, ordered
JSON helpers and shared image/vision execution. Model-specific workers retain
request contracts, preparation policy, calibration and scheduler admission.

## Shared vision

`image_preprocess` accepts RGB bytes and caller-provided resizing and patch
limits. `vision` validates BF16 checkpoints and supports these exact layouts:

| Layout | Blocks | Hidden | MLP | Heads × dimension | Output |
| --- | ---: | ---: | ---: | --- | ---: |
| Qwen3.5-4B / Cua-S1 | 24 | 1024 | 4096 | 16 × 64 | 2560 |
| Qwen3.8-27B | 27 | 1152 | 4304 | 16 × 72 | 5120 |

Both use 16-pixel spatial patches, temporal patch size 2, spatial merge size 2
and a 2304-entry position table. Unsupported layouts are rejected.

Cua-S1 keeps its checkpoint provenance validation, vision LoRA and legacy CUDA
entry points. The 27B layout uses base vision weights and requires the three
optional ABI 5 v2 position, rotary and attention symbols. See the
[CUDA backend](../../../backends/cuda/qwen3_5/README.md) for build commands.
Older ABI 5 libraries remain loadable for existing 4B consumers.

## Validation

Root `tests/qwen3_5/` covers inventories, supported layouts, geometry and image
limits. A800 GPU checks cover head-72 attention against float64, rotary order,
1152-wide learned-position interpolation and exact legacy head-64 behavior.
The original and shared Cua-S1 full vision features and stage hashes match
exactly at 256×256, 512×256 and 512×512, including with the original ABI 5 library.

[Raw A800 component evidence](https://github.com/Levius-Fubuki/system1-omni/releases/tag/jemm-a800-20261008)
includes the untouched baseline commit, pinned checkpoints, source hashes and
retained outputs. These checks do not establish execution on other GPUs or
end-to-end behavior of a new model worker.
