# Cua-S1 native vision implementation plan

Goal: finish native RGB-to-decision inference and verify against the pinned unmerged reference on RTX 4090.

Architecture: combine #59 checkpoint validation, #63 CPU RGB preparation and #56 language boundary on an isolated integration branch. Keep original BF16 base and FP32 vision LoRA branches separate. Execute patch projection, interpolated positions, 24 bidirectional vision transformer blocks and merger on CUDA. Build single-image token positions in Rust and expose an end-to-end RGB API/example. Retain the text API. Reject invalid geometry before GPU execution.

- [x] Inspect dependencies and GPU environment; preserve original branches.
- [x] Implement CUDA vision execution with CPU geometry tests and numerical GPU checks against Transformers stage outputs.
- [x] Implement native prompt image insertion, three-axis positions, RGB orchestration and candidate scoring; test invalid inputs and position oracles.
- [x] Export deterministic multi-geometry reference cases using pinned weights, unmerged BF16 and full FP32 controls, then run native GPU end-to-end comparison. Acceptance is the repository probability criterion: max error <= 2*BF16 reference error + 0.01, matching top choices at FP32 margin >= 0.05. Record intermediate errors without inventing tolerances after measurements.
- [x] Run workspace fmt, strict Clippy, tests, release build, CUDA regressions and independent spec/code review. Resolve actionable findings. Preserve commands and evidence in recipe documentation.

The first implementation is eager; no acceleration claim. GPU validation must exercise actual native vision, never substitute exported reference image features. Large attention inputs require bounded-memory attention. Verification uses fresh source hashes and exact tool/environment versions.
