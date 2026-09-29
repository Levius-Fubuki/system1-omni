# Cua-S1 Graph policy experiment

**Goal:** Establish whether mode selection is justified after correcting the exact admission-window confound, then validate shared resource ownership for exact and rule buckets.

**Architecture:** Start at #37 (`fd5c417`). Reuse the pinned worker, strict full-logit gates, and capture-inclusive benchmark. Compare eager, default exact, exact with a 32-request admission window, and worker rule buckets with otherwise identical limits. Keep both default worker behavior and numerical acceptance unchanged.

**Environment:** One authorized RTX 4090 session, existing Python 3.12 environment and weights. No installs or model downloads needed. Use an isolated local worktree and remote clone; preserve previous evidence.

## Execution checklist

- [x] Inspect GPU occupancy, reference revision, Python package versions and existing experiment harness.
- [x] Extend `recipe/cua_s1/benchmark_graph_buckets.py` with an optional tuned exact comparator and the existing hot/cold and shifting-hot schedules. Record complete per-variant configurations and runtime/admission source hashes. Keep old CLI behavior compatible.
- [x] Validate the comparison isolates admission-window changes using CPU tests, and verify the recorded schedule, response equality, statistics and per-variant limits independently.
- [x] Run two cold repetitions of hot-four, churn-twelve, hot/cold and shifting-hot, with all 24 execution orders cycling across requests. Use fresh output directories outside the remote Git repository.
- [x] Analyze total time (including capture and first-length gates), p50/p95, cache bytes, captures, replay, evictions and rejected requests. Record negative results and scope limits.
- [x] Build a narrowly scoped shared-budget experiment if the baseline supports continuing: one request clock, namespaced cache keys, one capture ledger, one aggregate resident-cache budget, deterministic retirement. Avoid claiming a hard peak-process-memory bound: in-flight candidate capture can temporarily exceed retained cache limits.
- [x] Validate cross-mode eviction, budget fallback, two questions counted once, cold/changed inputs, explicit close with live entry references, and replay after retirement of the other mode.
- [x] Preserve scripts, source revisions, raw results, verifier and a concise report locally. Do not claim auto selection, compilation, native integration or concurrent throughput without separate implementation and evidence.

## Acceptance and interpretation

All timed complete responses must match eager. Captures and new bucket lengths retain exact full-vocabulary-logit comparison. Do not relax tolerances. Tune only the admission window in the comparator; retain the same 8-shape / 1 GiB cache and capture limits. Report any negative or mixed outcome instead of selecting a favorable workload. Two repetitions are exploratory evidence, not confidence intervals. Shared-cache CPU tests must demonstrate real resource-retirement semantics before GPU validation. GPU measurements are serial synthetic workloads on the PyTorch fallback path, not production traffic.

## Recorded outcome

Completed the controlled baseline (2,304 timed predictions) and shared-budget prototype (119 full-logit comparisons, 16 complete-response comparisons). Both independent verifiers pass. Exact window tuning alone does not remove bucket gains under churn; the prototype validates resource semantics only. Worker promotion and automatic selection remain subsequent implementation work. See `recipe/cua_s1/experiments/rtx4090-graph-policy/README.md` for all results and limitations.
