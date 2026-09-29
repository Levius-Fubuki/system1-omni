# Cua-S1 Graph policy and shared-budget experiment

RTX 4090, 2026-09-29. Baseline parent: PR #37, `fd5c4172987bcbcc2eb8e10dcf04b3edd45af0b8`.

## Findings

Increasing exact Graph's admission window from 8 to 32 does not remove the
advantage of rule buckets under length churn. On twelve rotating lengths,
bucket total time is **24.69–25.19% lower than tuned exact**. On shifting hot
lengths it is **20.59–22.45% lower**. Conversely, buckets take **6.69–8.14% more
time on four stable hot lengths**, and **3.97–4.86% more on the tested hot/cold
mix**, than tuned exact. These are two exploratory cold runs, not confidence
intervals or a universal policy threshold.

A separate shared-resource prototype passed GPU parity and lifecycle checks:
one request clock, capture count/time ledger, aggregate resident-cache limit and
namespaced LRU for exact and rule-bucket execution. This is a building block for
mode selection; **automatic mode selection and production worker integration
are not implemented here**. No new compilation or model compute kernels.

## Controlled comparison

Measured source: `71531bb` (full revision and source hashes in `baseline.json`).
The only change between `exact` and `exact_tuned` is `admission_window=8 → 32`.
Both retain 8 resident shapes, 1 GiB cache, 4 captures per 32 requests and a
2,000 ms capture-work budget. A synchronous in-flight capture may overshoot the
time allowance; subsequent attempts are withheld until budget becomes available.
Worker buckets use width 64 and otherwise the same default resource limits.

Total synchronized `predict` time, seconds; cells are run 1 / run 2:

| Workload | Eager | Default exact | Exact window 32 | Rule buckets |
|---|---:|---:|---:|---:|
| Four hot lengths, 48 requests/run | 11.06 / 11.05 | 7.46 / 7.47 | 7.43 / 7.48 | 8.04 / 7.98 |
| Twelve rotating lengths, 72 requests/run | 16.88 / 16.93 | 16.43 / 16.43 | 16.32 / 16.33 | 12.29 / 12.22 |
| Hot/cold mix, 40 requests/run | 9.38 / 9.31 | 7.65 / 7.59 | 7.67 / 7.62 | 8.04 / 7.93 |
| Shifting hot lengths, 128 requests/run | 29.87 / 29.90 | 25.70 / 25.17 | 26.32 / 25.72 | 20.41 / 20.42 |

Capture-inclusive p95, milliseconds; cells are run 1 / run 2:

| Workload | Eager | Default exact | Exact window 32 | Rule buckets |
|---|---:|---:|---:|---:|
| Four hot lengths | 243.99 / 239.86 | 530.97 / 525.70 | 522.73 / 524.31 | 232.36 / 238.00 |
| Twelve rotating lengths | 240.89 / 240.29 | 236.12 / 233.96 | 570.31 / 565.53 | 241.81 / 243.34 |
| Hot/cold mix | 248.48 / 248.75 | 531.44 / 523.08 | 536.54 / 527.61 | 490.80 / 480.12 |
| Shifting hot lengths | 241.57 / 242.04 | 674.67 / 581.04 | 678.58 / 677.89 | 242.68 / 242.96 |

All **2,304 timed complete responses equal eager**. There are no numerical
rejections, capture errors or OOMs. An independent verifier recomputes totals,
p50/p95, response equality, source hashes, counter sums, cache limits and capture
count/time admission compliance from the raw events.

The twelve-length diagnostic isolates the earlier confound:

- Default exact captures/replays zero times: the recurrence gap exceeds its
  admission window, so it falls back to eager.
- Tuned exact captures 8 shapes and replays 56 question forwards in each run,
  but encounters 56 capture-budget fallbacks. It uses up to 972.06 MiB of
  resident Graph ownership, versus 598.53 MiB for buckets.
- Buckets capture 2 shapes and replay 138 question forwards, including 10
  first-length validation gates. The capture and validation savings persist
  after enabling exact admission; total speedup is not solely a window artifact.
- In this schedule tuned exact improves total time only about 0.6% relative to
  default exact, while its capture-inclusive p95 rises above 560 ms. Simply
  broadening admission is not a suitable default based on this evidence.

On the hot/cold mix, buckets retain 1,002.22 MiB compared with 463.45 MiB for
exact. Fewer bucket keys do not imply lower memory. A later selector must account
for retained bytes, capture/first-length cost and hotspot lifetime, alongside
reuse frequency. Its value is still to be measured; this experiment does not
establish that an online selector beats the best fixed mode.

## Shared-resource prototype

Measured source: `de1322a0837fc55088328689daa6d6684fb254ac`.
`recipe/cua_s1/graph_policy.py` composes the existing exact and worker bucket
runtimes. Mode selection is explicit and serialized by the experimental caller.
No worker CLI/default changes or global/model-method replacements are made.
Each captured shape retains its own allocator pool and owned stream; shared
budgeting does not share incompatible graph buffers or allocator pools.

- A single request clock advances once per `predict`, including 2- and
  8-question requests. Keys are namespaced by execution mode.
- All captures draw from one count/time ledger. Exact consuming the budget
  prevents a subsequent bucket capture while existing exact replay still works.
- One LRU enforces aggregate resident shape/byte limits. Cross-mode retirement
  explicitly closes entries even while diagnostics hold their Python references.
- Eviction cooldown refers to the evicted mode's key. Invalidation/close retire
  both modes, clear admission history and release their Graph ownership.

GPU checks in `shared.json`:

| Check | Result |
|---|---|
| 255/256/257/319/320/321 token boundaries, both directions and both modes | 24 exact full-vocabulary-logit matches; 9 captures, 15 replays |
| Global one-shape limit | Exact entry retired by bucket insertion; exact cooldown and surviving bucket replay succeed |
| Global 300 MiB resident limit | 115.96 MiB exact entry retired before retaining 271.12 MiB bucket; aggregate stays within the limit |
| Global one-capture allowance | Exact captures once; bucket falls back; exact replay remains valid |
| Exhausted shared time allowance | Bucket capture withheld after exact spends the allowance |
| 1 MiB resident limit | Both modes reject oversize candidates and return eager results; no resident entries |
| Two resolutions, 2/8 questions, cold/capture/changed image/changed text | 16 complete-response matches and 80 exact full-logit comparisons |
| Close with live entry references | Both modes' blocks/pools explicitly cleared, resident ownership zero |

The five budget/eviction scenarios add 15 full-logit comparisons, for **119
independently checked full-logit comparisons** in total. The boundary run retained
1,830,518,784 bytes across 9 exact/bucket entries before close and zero afterward.
The loaded model remains resident; zero Graph ownership does not mean zero
process GPU memory. Per-event memory snapshots are recorded with only one
shared runtime live at a time in this diagnostic.

**Memory limit scope:** limits are checked when admitting a fully captured
candidate. Capture may temporarily coexist with resident graphs and exceed the
resident-cache limit. This is not a hard cap on transient or total process VRAM.
The prototype's cross-mode scheduling performance and concurrent use are not
established by these correctness checks.

## Environment and measurement limits

Python 3.12.13, RTX 4090 sm_89, driver 595.71.05, PyTorch 2.14.0+cu130,
Transformers 5.17.0, PEFT 0.21.0. BF16 base and unmerged multimodal LoRA; pinned
weight revisions are recorded in both reports. No FLA or causal-conv1d is
installed: this measures the PyTorch fallback implementation.

The latency matrix uses a synthetic 320×240 image and two identical questions,
varying processed prompt lengths. It is serial engine latency, not HTTP or
concurrent throughput. Model load is excluded. Three eager warmups precede the
matrix; every workload/run starts with empty Graph caches. Timing includes
preprocessing, vision, both questions, cold capture, first-length checks and
in-request eviction, but excludes final cache retirement and writing reports.
All 24 four-variant orders cycle across requests; run two offsets by half a
cycle. Three variant caches coexist in the timing process, so allocator totals
are combined; only per-cache ownership is attributable to an individual mode.

These results do not establish real GUI task accuracy, broader hardware/backend
portability, concurrent throughput, or automatic-selection performance.

## Reproduction

Run the timing matrix at source `71531bb` and the shared diagnostic at
`de1322a`, from clean checkouts, using fresh output paths outside each checkout:

```sh
PYTHONPATH=src:recipe/cua_s1 python recipe/cua_s1/benchmark_graph_buckets.py \
  --weights /path/to/weights --output /path/to/policy-baseline \
  --kind worker --tuned-exact-window 32 --runs 2 \
  --case hot_four --case churn_twelve --case hot_cold --case shifting_hot

PYTHONPATH=src:recipe/cua_s1 python recipe/cua_s1/check_shared_graph_policy.py \
  --weights /path/to/weights --output /path/to/shared.json
```

Verify from the final branch (which contains both measured source revisions):

```sh
python recipe/cua_s1/verify_graph_policy.py \
  recipe/cua_s1/experiments/rtx4090-graph-policy/baseline.json --kind baseline
python recipe/cua_s1/verify_graph_policy.py \
  recipe/cua_s1/experiments/rtx4090-graph-policy/shared.json --kind shared
PYTHONPATH=src python -m pytest tests/cua_s1 -q
```

The final Cua-S1 suite passes **241 tests** (`tests.log`); changed Python files
pass Ruff lint and formatting. The verifier regression tests deliberately corrupt comparison limits, request
counts, schedules, logits, memory accounting, retirement evidence, complete
responses and cleanup state. See `verification.log`, `tests.log` and
`SHA256SUMS` for final validation and evidence integrity.

## Next implementation

Promote the validated ownership model into the worker with a reviewed lifecycle
API, then evaluate an explicit opt-in `auto` mode. Its admission decision should
include estimated capture amortization, first-length checking, aggregate cache
bytes and a switch cooldown. Keep exact/rule-bucket/eager manual modes and exact
numerical gates. Compare the implemented selector against the tuned exact
baseline here; do not infer selector gains by retrospectively choosing the best
fixed result for each workload.
