# Cua-S1 rule bucket worker integration

Approved scope: integrate the measured rule-only bucket runtime into the worker, keep exact Graph behavior and eager defaults, preserve strict numerical gates, validate HTTP and GPU parity/performance. Automatic strategy selection and simultaneous exact/bucket caches are separate work.

Use explicit model-instance computation, not global function replacement or instance-method mutation. A small cache-free prefill adapter reproduces the pinned Qwen3.5 GatedDeltaNet forward around a supplied rule callable. Norms, convolution, projections, MLP, full attention and positions retain original shapes. Only rule inputs are zero-padded. The adapter is restricted to the pinned Transformers implementation and CUDA-resident immutable model; unsupported execution forms fall back to eager. It has no persistent recurrent state.

A production RuleBucketRuntime owns one bounded cache/admission policy and serializes its requests. Preserve per-bucket capture equality, per-real-length exact vocabulary-logit gates, input-copy layout checks, independent shape pools/streams and eager fallback. Clearing/closing the engine explicitly retires all graphs before streams. Closed engines reject further prediction; shutdown drains current inference before releasing GPU resources.

CLI: no graph option stays eager; existing --graph stays exact; --graph-mode exact|rule-bucket also enables the selected mode. --graph-bucket-width defaults to 64, must be a positive multiple of 64, and is only meaningful in rule mode. Mode validation occurs before weight loading. Single-question requests retain the current reference path.

Acceptance: source runtime has no recipe dependency and performs no global/model-method mutation; CPU tests for modes/config/packing/gates/cleanup; CPU tensor adapter comparison against pinned upstream; changed-image/text, boundary and fallback GPU checks; real HTTP cold/capture/replay equality; same-workload comparison to previous recipe runtime and exact/eager. Document fixture-specific results and keep opt-in.
