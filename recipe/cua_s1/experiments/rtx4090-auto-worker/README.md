# Cua-S1 automatic Graph worker with shared budgets

RTX 4090, 2026-09-29. Stacked on PR #37 (`fd5c417`).

## Findings

Opt-in automatic selection reduced total synchronized request time by **18.29–33.08% versus eager** across five tested workloads and two cold runs. All **3,720 timed complete responses equal eager** (744 eager baselines and 2,976 Graph-enabled variant predictions, including 744 auto predictions). The selector uses only past/current requests and shares resource limits between exact and rule-bucket execution.

It does not always match the best fixed mode. Tables below retain both benefits and regressions. Stable hot layouts tend toward exact, while recurring dispersed lengths can reuse buckets after an observation period. Manual modes remain available and eager remains the default.

## Controlled performance

Total synchronized `predict` time, seconds; cells are run 1 / run 2:

| Workload | Eager | Default exact | Exact window 32 | Rule bucket | Auto |
|---|---:|---:|---:|---:|---:|
| Four hot lengths | 11.09 / 11.11 | 7.49 / 7.49 | 7.51 / 7.51 | 8.01 / 8.01 | 7.56 / 7.49 |
| Twelve rotating lengths | 17.01 / 17.02 | 16.52 / 16.56 | 16.43 / 16.45 | 12.29 / 12.33 | 13.23 / 13.23 |
| Hot/cold mix | 9.38 / 9.39 | 7.65 / 7.64 | 7.64 / 7.63 | 8.02 / 8.04 | 7.66 / 7.63 |
| Shifting hot lengths | 29.73 / 29.66 | 25.57 / 25.34 | 25.48 / 25.42 | 20.17 / 20.09 | 20.00 / 19.85 |
| Unseen mixed schedule | 19.96 / 20.07 | 17.42 / 17.50 | 19.16 / 19.30 | 14.79 / 14.98 | 16.07 / 16.11 |

Auto reduction versus eager and extra time versus the hindsight-best fixed mode (minimum of eager, exact, tuned exact and bucket). Positive extra time means auto is slower:

| Workload | Reduction vs eager (%) | Extra time vs best fixed (%) |
|---|---:|---:|
| Four hot lengths | 31.86 / 32.62 | 0.92 / -0.01 |
| Twelve rotating lengths | 22.22 / 22.24 | 7.67 / 7.34 |
| Hot/cold mix | 18.29 / 18.75 | 0.26 / 0.00 |
| Shifting hot lengths | 32.74 / 33.08 | -0.84 / -1.19 |
| Unseen mixed schedule | 19.48 / 19.74 | 8.66 / 7.52 |

Capture-inclusive p95, milliseconds; cells are run 1 / run 2:

| Workload | Eager | Default exact | Exact window 32 | Rule bucket | Auto |
|---|---:|---:|---:|---:|---:|
| Four hot lengths | 242.78 / 239.81 | 531.67 / 523.00 | 533.11 / 523.78 | 237.34 / 236.93 | 536.09 / 523.70 |
| Twelve rotating lengths | 243.19 / 243.62 | 237.88 / 236.32 | 567.53 / 569.25 | 244.64 / 245.51 | 242.90 / 245.86 |
| Hot/cold mix | 252.00 / 250.15 | 525.31 / 522.62 | 523.01 / 515.39 | 486.68 / 482.23 | 527.99 / 518.05 |
| Shifting hot lengths | 240.79 / 242.07 | 668.34 / 669.54 | 670.36 / 677.89 | 242.60 / 237.07 | 246.54 / 243.56 |
| Unseen mixed schedule | 251.25 / 251.78 | 568.33 / 568.11 | 568.69 / 573.42 | 249.31 / 247.54 | 251.73 / 246.29 |

Automatic execution and resident cache ownership; cells are run 1 / run 2. Replay counters include replay during first-length validation. They are not a count of requests accelerated:

| Workload | Captures | Replays | Peak resident MiB |
|---|---:|---:|---:|
| Four hot lengths | 4 / 4 | 84 / 84 | 463.45 / 463.45 |
| Twelve rotating lengths | 2 / 2 | 116 / 116 | 598.53 / 598.53 |
| Hot/cold mix | 4 / 4 | 52 / 52 | 463.45 / 463.45 |
| Shifting hot lengths | 5 / 5 | 227 / 227 | 789.69 / 789.69 |
| Unseen mixed schedule | 4 / 4 | 118 / 118 | 961.77 / 961.77 |

Raw `benchmark.json` records all samples, routing counters, resident mode names and selector reasons. `verify_auto_worker.py` independently recomputes p50/p95/totals, response equality, source hashes, cumulative counters and shared capture/cache limits. There are no capture OOMs, capture errors or numeric rejections in the formal matrix.

## Worker behavior and budgets

`--graph-mode auto` is explicit opt-in. The default remains eager and `--graph`
continues to mean exact mode. The production package owns the shared runtime;
the earlier `recipe/cua_s1/graph_policy.py` is now only a compatibility re-export.
No production import depends on recipe code. Full attention, projection,
convolution, MLP and position processing retain their real lengths.

Exact and rule-bucket caches share one namespaced LRU, request clock, lock,
resident-memory limit and capture count/time ledger. Each cache entry retains its own
stream/pool and buffers; its captured segments share that owner. One mode cannot obtain a second budget by switching.
The resident limit is not a hard process VRAM limit: capture candidates coexist
with retained graphs, and weights/default-allocator memory are excluded. A
synchronous attempt may overshoot the capture-time allowance; later captures
wait for that work to expire from the request window.

The selector retains at most 128 layouts in the preceding 32 requests, with
bounded cost history, 16 pending event pairs and 128 decisions per request.
It estimates up to 64 future requests using observed frequency, counts same-request
copies separately, checks capture/first-length amortization with a 25% margin,
and applies a 32-request switching cooldown. The initial 500 ms capture and
0.45/0.65 exact/bucket replay ratios are heuristics, not device-independent
performance predictions. Observed captures and sampled CUDA-event spans replace
priors. Sampling occurs on the first eight supported forwards and every sixteenth
thereafter; readiness queries add no synchronization. Captures and length gates
are excluded from replay-cost samples. There is no future schedule or workload
name input to the selector. Numeric gates and the shared admission ledger remain
authoritative when a prediction is wrong.

`selected_*` counts routing decisions, not successful graph replays. A selected
route can fall back to eager because of admission, resource or numeric checks.
Invalidation clears both caches, disabled keys, admission, selector history and
pending timing events; lifetime counters remain cumulative. Close explicitly
retires entries even while diagnostics retain their Python references.

## Correctness and lifecycle evidence

All final GPU checks use clean source `e243a30d73ae3ae822bdf21fc4f96d8b739c4b76`.
Subsequent changes strengthen the independent verifier and add evidence/tests/docs;
they do not alter the measured production implementation.

- `parity.json`: **266 complete-vocabulary logits comparisons**, all bitwise
  equal to eager (`torch.equal`, max absolute error zero); **61 complete responses**.
  Five cases cover 320x240/640x480, 2/8 questions, long text and distinct questions,
  each under cold/capture/warm/changed-image/changed-text execution. A 36-request
  twelve-length rotation exercises actual bucket capture/replay, and 24 direct
  forwards cover 255/256/257/319/320/321-token boundaries. Case capture/replay
  counters and no-rejection assertions prevent eager-only parity from counting
  as Graph validation. This diagnostic uses a recorded 4 GiB/16-capture/10 s budget
  and 4096-token limit to exercise paths; it is not the default-budget benchmark.
- `shared.json`: **119 full-logit comparisons and 16 complete responses** against
  the production shared resource manager. It covers cross-mode one-shape eviction,
  a 300 MiB resident cap, one-capture and exhausted-time allowances, 1 MiB oversize
  candidates, boundaries, cooldown and explicit retirement with live references.
  The independent verifier reconstructs counter sums and resident ownership.
- `http.json`: actual CLI-selected auto worker, health and cold/capture/changed-image/
  changed-text HTTP parity; two independently loaded models, continued replay
  after the other model closes, no global/model-method replacement, and explicit
  cleanup with retained entry references. This is isolation/lifecycle validation,
  not a concurrent-throughput result.
- The existing exact full-logit admission and per-new-length bucket gates remain
  unchanged. These checks establish equality on the tested inputs; a first-input
  gate cannot prove equality for all possible image/text content.

## Automated checks

The measured GPU source passed all 266 Cua-S1 tests (`tests.log`). The later
independent-verifier regressions add 13 tests, including corrupted clocks,
budgets, routing records, missing captures/replays and incorrect responses.
The complete PR suite then passed **279 tests** on the GPU host at `543fade`
(`tests-pr.log`); subsequent changes only clarify documentation and retain logs.
The local CPU-only run passed 257 tests and skipped 22 requiring PyTorch
(`cpu-tests.log`). CI-scoped Ruff lint and formatting passed (`lint.log`).
A focused code review found two initial issues (unconditional bucket-mask
synchronization and insufficient resident-bucket cost gating); both were fixed
before the final measurement, with regressions. Event probes remain bounded even
when a selected mode repeatedly falls back. Final review found no production
correctness, shared-budget or lifecycle blockers.

## Scope and reproduction

RTX 4090 24 GB; driver 595.71.05, Python 3.12.13, PyTorch 2.14.0+cu130,
Transformers 5.17.0, PEFT 0.21.0, 64 CPU threads, BF16 base and unmerged PEFT
adapter. Neither FLA nor causal-conv1d is installed; results apply to the pinned
PyTorch fallback path. See JSON environment and source hashes for full provenance.
No native multimodal boundary, new model-compute kernel, compilation, batching,
GUI task accuracy or broader deployment claim is made.

Formal timing compares eager/default exact/exact admission-window 32/rule-bucket/
auto with otherwise identical defaults: 8 resident shapes, 1 GiB, 4 capture
attempts per 32 requests, 2000 ms capture-work allowance, bucket width 64 and
2048-token cap. Two cold-cache runs per workload use cyclic and reversed order
rotations with a half-cycle offset between runs. Each ten-request block balances
variant positions; incomplete final blocks are retained. The model stays loaded.
Timing synchronizes around `predict`, including capture, first-length checks,
eviction and fallback. Parsing/image decoding, HTTP, initial model load/warmup,
report I/O and final cache retirement are excluded. Each request has two identical
questions on a generated 320x240 image. All four Graph variants coexist, so process
allocator totals are combined; per-variant cache ownership is reported separately.

The four recurrent schedules retain the preceding experiment's shapes. The new
84-request mixed schedule is `[1,2]*6`, 48 draws from 1..16 using seed 20260929,
then `[20,21]*12`. It was fixed before the formal run and was not used to tune the
policy afterward. Two runs are exploratory observations, not confidence intervals
or evidence that the heuristic beats the best fixed mode on arbitrary traffic.
Capture-inclusive p95 can regress while total request time improves.

From a clean checkout with pinned dependencies and verified weights:

```sh
export PYTHONPATH=src:recipe/cua_s1
python recipe/cua_s1/check_auto_worker.py --weights /path/to/weights --output /tmp/auto-parity.json
python recipe/cua_s1/check_shared_graph_policy.py --weights /path/to/weights --output /tmp/auto-shared.json
python recipe/cua_s1/check_bucket_worker_http.py --weights /path/to/weights --mode auto --output /tmp/auto-http.json
python recipe/cua_s1/benchmark_graph_buckets.py --weights /path/to/weights --output /tmp/auto-matrix \
  --kind worker --tuned-exact-window 32 --include-auto --runs 2 \
  --case hot_four --case churn_twelve --case hot_cold --case shifting_hot --case mixed_holdout
python recipe/cua_s1/verify_auto_worker.py /tmp/auto-matrix/report.json --parity /tmp/auto-parity.json
python recipe/cua_s1/verify_graph_policy.py /tmp/auto-shared.json --kind shared
```

Use fresh output paths. Provenance verification needs the measured Git objects,
so use a full clone or fetch the recorded revisions. `SHA256SUMS` covers the raw
JSON/log files. `diagnostics/initial-probe.*` retains the earlier 600-prediction
two-workload probe at `0473de8`; `initial-parity.*` is the first 266-logit run at
`cb63dac`. `default-probe.*` records 240 predictions at the final source before
explicitly selecting all five formal workloads (the script defaults to `probe`).
These diagnostics are not mixed into formal performance tables. The preceding
[controlled exact-window and shared-prototype experiment](../rtx4090-graph-policy/README.md)
retains its historical results and source.
