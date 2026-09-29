# Cua-S1 automatic Graph worker implementation

**Goal:** Ship an explicit, validated `--graph-mode auto` with shared resources and a reproducible PR on top of #37.

**Architecture:** Promote the tested shared runtime to the model package. Keep eager/default and manual exact/rule modes. Route each supported question under a single request lock; decisions for an identical layout remain fixed within that request. Both children draw from one namespaced cache, admission clock and capture ledger. A bounded online selector observes only past/current requests, and has no workload-name or future-schedule input.

## Selector contract

- Retain at most 128 layouts over a 32-request horizon. Count distinct requests separately from same-request question multiplicity.
- Resident exact entries replay immediately. A previously rejected bucket length may not replay via auto; all existing strict gates remain active.
- Consider exact capture only after recurrence within the configured exact admission window, without working-set pressure or predicted eviction of useful graphs. Consider buckets for recurring dispersed lengths or a growing repeated working set.
- Compare estimated future saving against capture cost with 25% margin. Charge first-length validation when considering a new bucket. Predict promotion from resident bucket to exact using only the incremental saving.
- Initially use conservative 500 ms capture cost and replay ratios 0.45/0.65 of measured eager language latency. Replace priors with observed captures and sampled CUDA-event spans, without adding synchronization to normal replay. Priors are policy heuristics, not guaranteed latency predictions.
- Record observations per compatible bucket family, bounded like layout history. Skip capture/gate/rejection events as replay-cost samples. Maintain switch cooldown. Cache/capture limits remain authoritative when estimates are wrong.
- Default remains eager; `--graph` remains exact. Auto is opt-in and pinned to the same single-device fallback implementation as buckets.

## Tasks and validation

- [ ] Promote shared ownership; add explicit construction injection for cache/admission/lock, reusable exact keys, atomic retirement, and tests covering lifetime and cross-mode bounds.
- [ ] Implement bounded selector with tests for hot, dispersed, one-shot, same-request duplicates, pressure, cost rejection, cooldown, invalidation and bucket rejection.
- [ ] Integrate config/CLI/engine, document counters and expose no experimental imports in production.
- [ ] Run Cua-S1 CPU tests and changed-file lint/format.
- [ ] GPU check full logits, mixed layouts, changed images/text, boundaries, aggregate budgets and lifecycle; use actual CLI-selected HTTP worker and two loaded instances.
- [ ] Benchmark eager/default exact/tuned exact/bucket/auto on four existing workloads, plus an unseen mixed schedule. Use two cold runs and order balancing, retain capture/gates/eviction in measured time and per-mode evidence. Record negative results instead of tuning to future schedule labels.
- [ ] Independently verify raw data/source provenance and report scope. Run a focused code review, fix findings, rerun checks justified by changes, push branch, create and attach PR with incremental comparison against #37.

## Shipping criterion

All complete responses and full-logit checks match eager; budgets and lifecycle checks pass. The selector should improve total cost over eager on the tested recurrent workloads, with regressions versus fixed modes disclosed. It need not beat the hindsight-best fixed choice. If its cost assumptions fail materially, revise from measured evidence, preserve failed/intermediate runs, and revalidate. No automatic enabling by default and no claims about concurrent throughput or non-fallback kernels.
