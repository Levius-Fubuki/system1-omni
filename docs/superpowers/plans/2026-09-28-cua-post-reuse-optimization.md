# Cua-S1 Post-Reuse Optimization Implementation Plan

> **For agentic workers:** Execute these steps in order in the isolated worktree; preserve independent profiling and timed results.

**Goal:** Select and verify one optimization after image reuse, publish a reproducible PR, and shut down the GPU server.

**Architecture:** Use the existing reference and reuse paths. Measure current kernel groups, then limit the output projection to one token only if the paired experiment shows a benefit. Store raw experiment evidence and a readable analysis under `recipe/cua_s1/experiments/`.

**Tech Stack:** Python 3.12, PyTorch 2.14, Transformers 5.17, PEFT 0.21, RTX 4090.

---

- [x] Profile the unchanged reuse path with 1 and 8 questions; retain raw kernel groups, module invocation counts, traces, and source revision.
- [x] Run a temporary paired exploration of full projection versus `logits_to_keep=1`, checking complete responses before timing. Select the candidate only if its effect is credible.
- [x] Add a failing tensor integration test in `tests/cua_s1/test_image_reuse.py` proving the reused path passes `logits_to_keep=1`; run it on the GPU host or a torch-equipped environment.
- [x] Modify `score_reused` in `src/models/cua_s1/multimodal/model.py` and clean up a nearby Ruff guard; rerun the focused test and the full Cua-S1 suite.
- [x] Add a paired benchmark recipe with explicit baseline selection, alternating order, raw samples, response parity, source and environment metadata, and peak allocated memory.
- [x] Commit a clean source revision, transfer it to the GPU host, and run correctness and timed measurements. Review raw samples, variability, kernel changes, and single-question behavior.
- [x] Publish result data and reproduction commands, run verification, push the branch, create and attach the PR.
- [x] Archive and checksum complete GPU evidence locally, ensure no experiment is running, shut down the server, and verify it no longer accepts SSH.

## Completion evidence

Published [PR #18](https://github.com/ThinkFlowLab/system1-omni/pull/18) with the 17-case, 2,040-prediction dataset. The independent audit recomputed every saved p50/p95 and passed the 13-fixture, 49-question exact correctness and eight-question HTTP checks. Clean measured source `224d88b` passed 105 Cua-S1 tests on the GPU host; local tests passed 104 with one torch-dependent skip. The full trace archive SHA-256 matched on the server and local host: `774d952e89a9586709419c6c8583e4aa7ee2d27b4bf836253c8c01881c4dccd9`. The platform shutdown command exited successfully, and a subsequent SSH attempt was refused.
