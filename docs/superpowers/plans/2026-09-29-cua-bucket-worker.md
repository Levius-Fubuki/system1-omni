# Cua-S1 Bucket Worker Implementation Plan

Execute inline using the approved worker-integration scope.

**Goal:** Make rule-only buckets available as an explicit worker execution mode with local model calls and reproducible GPU/HTTP evidence.
**Architecture:** `rule_prefill.py` implements pinned cache-free prefill with an explicit rule callback. `graph_buckets.py` owns packing, cache/gates and direct per-model decoder traversal. `model.py` selects a runtime; `server.py` parses modes and closes the engine after draining requests.
**Tech Stack:** Existing pinned Transformers 5.17.0 / PEFT / PyTorch CUDA environment and pytest/Ruff.

- [x] Add failing tests in `tests/cua_s1/test_rule_prefill.py`: compare a small pinned Qwen3.5 module's CPU output to the explicit callback path, prove globals and forward identities unchanged, validate callback output behavior and input contracts. Implement `src/models/cua_s1/multimodal/rule_prefill.py`.
- [x] Add failing runtime/config/lifecycle tests in `tests/cua_s1/test_bucket_worker.py`. Promote the successful recipe logic to `src/models/cua_s1/multimodal/graph_buckets.py`; directly traverse model-instance layers using the prefill adapter; preserve capture/admission and full-logit gates. Reject static input shape/dtype/device changes before copies. Explicitly retire graph entries on invalidation/close.
- [x] Extend `GraphConfig` with validated mode/width defaults, `MultimodalEngine` runtime selection and idempotent close, and worker CLI parsing/shutdown. Test eager default, legacy --graph, explicit modes, malformed options, no post-close prediction and HTTP drain semantics.
- [x] Build a clean GPU recipe checking explicit prefill against original, independent runtimes without global mutation, complete logits under changed inputs and boundaries, eviction/budget fallback, and real HTTP cold/capture/replay. Run pinned tests and CI-scoped lint/format.
- [x] Compare synchronized full predictions for eager/exact/previous recipe/worker buckets on stable and rotating lengths, balancing order and counting cold cost. Record clean revision, full responses and budgets. Preserve historical evidence without rewriting it.
- [x] Independent code/evidence review; address important findings, verify report metrics, commit clean source and evidence, and keep the worktree for subsequent work.

Completed on 2026-09-29. Runtime revision `74760be`; 218 tests, 136 changed-content full-logit checks, 12 boundary checks, two-model HTTP isolation and 960 four-way timed predictions passed. Exact-boundary stride regression was reproduced and fixed. Independent code/evidence review completed; report records both positive and negative results.
