# Cua-S1 Multimodal Graph Runtime Implementation Plan

> Inline execution in this task; the user authorized implementation, GPU experiments, PR submission, and server shutdown.

**Goal:** Serve multimodal language forwards through reusable CUDA Graphs while preserving the existing worker's candidate probabilities.

**Architecture:** The pinned Qwen3.5 model has runs of three Gated DeltaNet layers separated by full attention layers. Capture only each three-layer run; execute full attention, rotary positions, normalization, and final projection eagerly. Cache all eight graph segments as one exact-layout entry, with limits and eager fallback.

**Tech Stack:** Python 3.12, PyTorch 2.14 CUDA Graphs, Transformers 5.17, PEFT 0.21, pytest, RTX 4090.

---

### Task 1: Diagnose numerical divergence

- [x] Reproduce the two rejected cases from #20.
- [x] Compare per-layer eager and full-graph outputs; identify the first divergent operation.
- [x] Prototype segmented capture and compare complete candidate logits to the original forward.

### Task 2: Build the bounded runtime

**Files:** `src/models/cua_s1/multimodal/graph_runtime.py`, `tests/cua_s1/test_graph_runtime.py`.

- [x] Add failing tests for configuration limits, exact tensor signatures, and LRU eviction.
- [x] Implement static input buffers, side-stream warmup, capture, and replay for contiguous linear-attention layers.
- [x] Build the exact model loop with eager full attention and the original final norm and head.
- [x] Validate a newly captured shape against the original forward before caching it.
- [x] Bound the cache by layout count and retained allocation; fall back to eager on unsupported inputs and resource limits.

### Task 3: Integrate the worker

**Files:** `src/models/cua_s1/multimodal/model.py`, `src/models/cua_s1/multimodal/server.py`, `tests/cua_s1/test_image_reuse.py`.

- [x] Add a failing test that an enabled runtime receives the prepared multimodal tensors.
- [x] Route only the reused multi-question path through the runtime; retain existing default behavior.
- [x] Add server flags for opt-in Graph mode and limits; keep the server's inference lock.

### Task 4: GPU acceptance and PR

**Files:** `recipe/cua_s1/experiments/rtx4090-graph-runtime/README.md`, `recipe/cua_s1/README.md`.

- [x] On RTX 4090, verify eager versus Graph for repeated, distinct, changed-image, changed-text, and long prompts, including the two previously rejected cases.
- [x] Measure synchronized whole-request p50/p95, capture cost, retained GPU memory, and fallback counts.
- [x] Run the full Cua-S1 test suite, Ruff, and format checks.
- [ ] Inspect the diff, commit, push, and open a PR.
- [ ] Stop GPU processes and shut down the user-provided server after evidence is saved.
