# Cua-S1 Image Reuse Implementation Plan

> **For agentic workers:** Use subagent-driven-development for implementation and independent review; controller owns GPU experiments and publication.

**Goal:** Complete request-local image reuse with reproducible correctness and speed evidence, submit a PR, then shut down the GPU server.

**Architecture:** Keep existing prepare/score as the reference path. Add isolated image preparation reuse and shared visual features for multi-question predict. Build per-question embeddings and positions using pinned model helpers.

**Tech Stack:** Python, PyTorch 2.14, Transformers 5.17, PEFT 0.21, RTX 4090.

- [x] Baseline: run all tests/cua_s1; preserve existing PR #15 data.
- [x] Implementation: tests/cua_s1/test_image_reuse.py first; prove failures; implement src/models/cua_s1/multimodal/model.py and focused helper module if needed. Preserve preflight validation, distinct prompts, no shared mutable state, PEFT vision layers and unchanged single-question behavior.
- [x] Harness: recipe/cua_s1/benchmark_image_reuse.py plus CPU tests. Check exact tensors, positions and probabilities before timed paired measurements; alternate order; record each sample and independent per-variant peak memory. Retain failed reports. Reuse profile_multimodal fixture and provenance helpers.
- [x] GPU: use clean committed checkout, run CPU tests, correctness smoke then 16-case paired experiment (5 warmups, 2 x 50 iterations per variant), plus distinct questions. Separate count/trace verification from all timed calls.
- [x] Review: independent spec review followed by quality review; resolve findings and re-run affected checks.
- [x] Publication: document measured results and tensor contract, retain raw JSON, run lint/full tests, commit/push and create linked PR identifying #12/#15 dependencies.
- [x] Shutdown: copy experiment artifacts locally, verify hashes, confirm no required GPU work remains, shut down server and verify provider/connection state.

## Completion evidence

Published non-draft [PR #17](https://github.com/ThinkFlowLab/system1-omni/pull/17). The complete experiment archive was copied locally and its SHA-256 matched the GPU host. After publication, the platform shutdown command exited successfully; a subsequent SSH attempt was refused. No experiment process remained on the GPU before shutdown.
