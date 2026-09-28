# Cua-S1 Post-Reuse Optimization Implementation Plan

> **For agentic workers:** Execute these steps in order in the isolated worktree; preserve independent profiling and timed results.

**Goal:** Select and verify one optimization after image reuse, publish a reproducible PR, and shut down the GPU server.

**Architecture:** Use the existing reference and reuse paths. Measure current kernel groups, then limit the output projection to one token only if the paired experiment shows a benefit. Store raw experiment evidence and a readable analysis under `recipe/cua_s1/experiments/`.

**Tech Stack:** Python 3.12, PyTorch 2.14, Transformers 5.17, PEFT 0.21, RTX 4090.

---

- [ ] Profile the unchanged reuse path with 1 and 8 questions; retain raw kernel groups, module invocation counts, traces, and source revision.
- [ ] Run a temporary paired exploration of full projection versus `logits_to_keep=1`, checking complete responses before timing. Select the candidate only if its effect is credible.
- [ ] Add a failing tensor integration test in `tests/cua_s1/test_image_reuse.py` proving the reused path passes `logits_to_keep=1`; run it on the GPU host or a torch-equipped environment.
- [ ] Modify only `score_reused` in `src/models/cua_s1/multimodal/model.py`; rerun the focused test and the full Cua-S1 suite.
- [ ] Add a paired benchmark recipe with explicit baseline selection, alternating order, raw samples, response parity, source and environment metadata, and peak allocated memory.
- [ ] Commit a clean source revision, transfer it to the GPU host, and run correctness and timed measurements. Review raw samples, variability, kernel changes, and single-question behavior.
- [ ] Publish result data and reproduction commands, run verification, push the branch, create and attach the PR.
- [ ] Archive and checksum complete GPU evidence locally, ensure no experiment is running, shut down the server, and verify it no longer accepts SSH.
