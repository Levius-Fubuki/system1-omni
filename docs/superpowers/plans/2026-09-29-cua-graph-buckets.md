# Cua-S1 Graph Bucket Experiment Implementation Plan

Execute inline in the isolated worktree. Scope is an experiment and reusable opt-in recipe runtime, not default production enablement.

**Goal:** Determine whether causal segment padding can safely reduce length churn and end-to-end latency.
**Architecture:** A recipe-only GraphRuntime subclass reuses #33 capture, admission, allocator and stream ownership. It changes segment input packing and cache keys, adds per-real-length exact validation, and keeps full attention at original length.
**Tech Stack:** Pinned PyTorch 2.14.0+cu130, Transformers 5.17.0, PEFT 0.21.0; RTX 4090.

- [ ] Establish baseline: bundle clean PR #33 source to a separate server checkout; run `PYTHONPATH=src python -m pytest tests/cua_s1 -q`. Preserve existing remote experiments.
- [ ] Add `tests/cua_s1/test_graph_buckets.py` before runtime code: bucket boundaries 63/64/65, invalid widths, prefix preservation/zero-tail reset, key sharing/separation, and per-length rejection without poisoning verified lengths. Run failures before implementing `recipe/cua_s1/graph_buckets.py`.
- [ ] Implement only dense batch-one/no-input-padding support. Use `_ShapeEntry` and `_GraphSegment` unchanged; pad each linear run then crop. Retain lock, request accounting, capture budget, strict full-logit capture gate and per-length validation/rejection metadata bounded by resident entries.
- [ ] Add `recipe/cua_s1/benchmark_graph_buckets.py`: clean revision/environment provenance, complete response comparison, cold-inclusive timing, balanced variant order and incremental report writes. First run correctness on 12 distinct lengths and changed image/text. If exact parity fails, keep diagnostics and investigate before timing claims.
- [ ] Run repeated mixed-length and stable-hot schedules. Separately measure each variant's reservation with no other graph runtime resident. Include padding ratio, p50/p95, total time, capture/validation counts, and regressions.
- [ ] Preserve raw evidence and a reproducible report under `recipe/cua_s1/experiments/rtx4090-graph-buckets/`; verify recomputed metrics and full responses, run CI-scoped lint/tests, and record limitations.
