# Request-aware Graph admission — RTX 4090, 2026-09-29

This change counts reuse across distinct `predict` requests, resets sparse heat,
adds eviction cooldown, and limits capture work in a sliding request window.
Cache hits continue to replay when the capture budget is exhausted. It also gives
each shape an exclusively owned CUDA stream and a shared private pool for its
ordered segments. Cache accounting includes all reserved pool segments plus
external static input buffers, instead of only live tensor deltas. Whole-shape
retirement ordering addresses a separately reproduced stream/workspace lifetime
hazard while sharing temporary storage between that shape’s serial segments.
Graph execution remains opt-in. Admission bounds wasted work; it does not predict
whether a future workload will amortize capture.

## Sources and measurement boundaries

After the server endpoint changed, the interrupted September 28 matrix was
retained separately. The first September 29 matrix exposed capture-OOM fallback
and is diagnostic only. Final performance, parity and HTTP checks use clean commit
`9f2c7ef1cb1778b573d0c8f844c2b5be0710cf1a` after the pool fix.
The 164-test suite and pool/stream checks use `dda1f0c`; the only following change
before measurement raises the correctness-only cache budget to cover all seven
large distinct layouts. Runtime and performance code are identical.
Later commits format tests and add reports, publication checks and documentation,
not runtime or benchmark changes.

The legacy comparator loads the original runtime bytes from PR #24 commit
`2336a085fc090c1c6ac3a297f91d826b66949f30`, verified against Git before import,
then substitutes the current `_GraphSegment`, `_ShapeEntry`, and `_run_segments`
implementations. Thus **both policies use the same execution, memory accounting,
and stream-lifetime fixes**, preserving historical admission/cache policy. These numbers compare admission policies;
they are not a rerun of unmodified #24. Original and current runtime SHA-256 values
are stored in the raw report and independently checked against the Git objects.

One RTX 4090 (24,564 MiB), Python 3.12.13, PyTorch 2.14.0+cu130, Transformers
5.17.0 and PEFT 0.21.0; BF16 base with the original PEFT adapter. The installed
PyTorch Git revision is `08187d9e0fba026dc8217405802ab5381dc88d90`.
The native PyTorch CUDA allocator is required for per-pool reservation accounting.
Neither `flash-linear-attention` nor `causal-conv1d` is installed. This measures
the existing Transformers fallback implementation, not new custom kernels.

All four schedules run twice from cold caches. There are 576 paired request
positions and **1,728 timed predictions** across eager, legacy and admission.
Each prediction contains two identical questions and one synthetic 320×240 image;
lengths vary with goal repetitions. Timing surrounds synchronized `engine.predict`,
including preparation, cold admission, capture, correctness gates, retirement and
replay. Model loading, parsing/HTTP and concurrent traffic are excluded. Execution
order reverses between adjacent positions and runs; legacy remains in the middle.
The two graph caches coexist on one engine: CUDA allocator totals include both
pools and are not isolated per-variant memory measurements. Raw cache occupancy
is recorded separately. There is no statistical significance or general GUI-task
accuracy claim from these two runs and synthetic fixtures.

## Results

| Workload (requests/run) | Eager total s | Legacy total s | New total s | New reduction % |
| --- | ---: | ---: | ---: | ---: |
| hot_four (48) | 11.71 / 11.35 | 7.58 / 7.30 | 7.73 / 7.58 | 33.95 / 33.23 |
| churn_twelve (72) | 17.46 / 17.58 | 51.34 / 51.96 | 17.08 / 17.08 | 2.14 / 2.88 |
| hot_cold (40) | 9.78 / 9.69 | 11.79 / 11.65 | 8.01 / 7.78 | 18.13 / 19.75 |
| shifting_hot (128) | 30.91 / 31.25 | 24.28 / 24.32 | 26.14 / 25.62 | 15.44 / 18.02 |

| Workload | Eager p50 / p95 ms (run 1; run 2) | Legacy p50 / p95 ms | New p50 / p95 ms |
| --- | ---: | ---: | ---: |
| hot_four | 238.90 / 258.75; 234.34 / 246.69 | 109.21 / 619.87; 108.27 / 594.07 | 110.36 / 563.04; 108.45 / 539.90 |
| churn_twelve | 244.08 / 253.28; 246.51 / 253.64 | 723.02 / 804.66; 732.03 / 794.51 | 236.50 / 254.70; 238.00 / 247.66 |
| hot_cold | 242.43 / 261.83; 240.45 / 260.87 | 111.78 / 856.01; 111.08 / 845.56 | 122.85 / 556.48; 121.98 / 536.67 |
| shifting_hot | 242.69 / 252.46; 243.30 / 256.96 | 120.76 / 706.05; 121.27 / 720.38 | 123.61 / 705.66; 123.39 / 586.47 |

| Workload | Legacy captures / evictions (run 1; run 2) | New captures / evictions | New sustained break-even request |
| --- | ---: | ---: | ---: |
| hot_four | 4 / 0; 4 / 0 | 4 / 0; 4 / 0 | 18 / 18 |
| churn_twelve | 72 / 64; 72 / 64 | 0 / 0; 0 / 0 | None / None |
| hot_cold | 12 / 5; 12 / 5 | 4 / 0; 4 / 0 | 22 / 22 |
| shifting_hot | 16 / 8; 16 / 8 | 16 / 8; 14 / 6 | 18 / 18 |

Each pair below is run 1 / run 2. A positive reduction means faster than eager;
a negative value means slower. Total time includes every cold request and capture.
P95 uses the nearest-rank definition. The full raw report records all responses,
latencies, counters, configurations, lengths and memory samples.

The new policy avoids all 72 captures and 64 evictions per run in the churn
schedule. Its 2.1–2.9% difference from eager is timing/order variation, **not a Graph
speedup**, because no replay occurs. The legacy policy takes about 2.94–2.96× eager
time there. In hot/cold traffic, captures fall from 12 to 4 and the new policy is
18.1–19.7% faster than eager, while legacy remains about 20% slower.

Conservative admission is not universally better: hot-four is 33.2–34.0% faster
than eager but trails legacy's 35.3–35.6%; shifting-hot is 15.4–18.0% faster than
eager but trails legacy's 21.5–22.2%. Delayed capture and budget fallback trade
some high-reuse speed for protection from low-reuse capture churn. Capturing
workloads still have substantially higher p95 than eager. The time budget is
soft for one non-preemptible capture, not a per-request latency cap.

Every final sample has exact full-response equality. Both policy variants record
zero capture OOMs, capture errors, numerical mismatches and rejected layouts.
Peak process reserved memory is 12,274,630,656 bytes (12.27 GB), including the
model and both policies' pools. Each runtime's retained cache stays within 1 GiB;
candidate/transient capture allocations and default allocator reservations are
outside that resident-cache limit.

## Correctness and verification

- `tests.log`: 164 Cua-S1 tests pass on Python 3.12.13. The baseline Ruff rule set
  (`--isolated --select E4,E7,E9,F`) passes; the final local tree also passes
  the exact CI rules (including import order) and format check. Expanded lint also reports nine existing
  findings in historical benchmark/server files; they are not silently represented
  as a clean expanded lint run. New admission test findings were fixed.
- `parity.json`: five complete-model cases, including 320×240 / 640×480, long
  prompts and distinct questions. Black-image replacement and same-length question
  replacement replay all questions and match the entire eager response dictionary.
  Four further cases exercise eviction cooldown, a 1 MiB cache, capture-count and
  capture-time budget fallback. Correctness-only limits are expanded to 4,096
  tokens, 16 captures, 10,000 ms and a 4 GiB cache so seven distinct question shapes actually
  capture; performance uses the default limits, including 2,048 tokens and a 1 GiB cache.
  The earlier 1 GiB correctness run preserved exact responses but evicted a large
  distinct layout, so it could not demonstrate replay of all seven layouts.
- `streams.json`: 40 simultaneously live standalone capture streams, 40 exact initial
  replays, then 60 exact changed-input survivor replays after retiring half the
  graphs. Idempotent cleanup is exercised. This exceeds the ordinary 32-stream
  pool and rules out merely rotating through pooled handles.
- `pools.json`: 40 independent shape owners, 120 sequential Graph segments and
  60 exact changed-input shape replays, including survivors after retirement.
  Retired pool snapshots are empty; reserved-memory budgets force eviction.
- `http.json`: real worker health and three inference requests, including cold
  admission, capture and replay, exactly match eager responses. This is a worker
  HTTP smoke test; it is not a Rust-frontend or throughput benchmark.
- `verification.log`: independent stdlib verifier recomputes performance metrics,
  compares complete response dictionaries, audits sliding capture budgets, and
  validates memory/counter consistency. Separate publication checks require all
  four cases × two runs and compare provenance hashes to the measured Git objects.

The observed post-capture sustained break-even point must remain nonnegative
through the end of that run. First cumulative crossing can occur before capture,
so it is reported separately and is not evidence of amortization. No-capture and
net-negative runs have no post-capture break-even point.

## Memory-pressure diagnostic

The first complete matrix with owned per-segment streams returned exact responses
but failed the publication verifier: both policy variants entered capture-OOM
fallback in the shifting-hot workload. Reported cache tensor allocations were
about 0.73 GB per runtime while process reserved memory reached about 23.9 GB.
Those timings are diagnostic only (`diagnostics/oom-policy-report.json.gz`) and
are excluded from the final comparison. The discrepancy exposed unaccounted
Graph private-pool reservations and redundant per-segment workspaces.

## Stream lifetime failure and limits of attribution

An earlier full run with unpatched legacy segments stopped at `shifting_hot`,
run 1, request 104, with `CUBLAS_STATUS_EXECUTION_FAILED` in PEFT projection.
Its partial timings are excluded from the table. An isolated repeat with
`CUDA_LAUNCH_BLOCKING=1` completed, so that particular failure was not deterministic.
The incomplete report and logs are preserved under `diagnostics/` and in the
full local evidence backup.

The installed PyTorch source clears cuBLAS workspace associated with a capture
stream when its graph is reset. A minimal shared-stream reproduction captures
forward/backward GEMMs, destroys temporary graphs on the same stream, and then
fails replaying the surviving graph with CUDA launch failure. Giving those graphs
independent owned streams makes the same reproduction pass with exact gradients.
See `diagnostics/workspace_repro.py.txt`, `workspace-shared.log` and
`workspace-owned.log`; execute the saved text with Python, with and without
`--owned`, in separate processes. This reproduces the shared-stream lifetime
hazard and supports the fix, but does not uniquely prove the cause of every
original model failure. The completed final matrix is the regression evidence.

Shared per-shape pools follow [PyTorch’s ordered, non-concurrent replay contract](https://docs.pytorch.org/docs/2.14/notes/cuda.html#sharing-memory-across-captures).

Primary references: [PyTorch issue #193402](https://github.com/pytorch/pytorch/issues/193402)
and the [installed wheel's CUDAGraph source](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/cuda/CUDAGraph.cpp).
The installed wheel's Git SHA matters; a release tag may have different code.

This does not fix whole-model BF16 attention differences by relaxing tolerances.
The pre-existing segmented executor still leaves full attention eager. No new
CUDA/Triton model kernel, native multimodal adapter, length bucketing, compilation,
dynamic batching or concurrent-throughput result is claimed.

## Reproduction

See [the experiment guide](../../graph-admission.md). Use the measured clean
commit above and the exact historical revision. Run the matrix, then:

```sh
python recipe/cua_s1/verify_graph_admission.py /path/to/output/report.json
python recipe/cua_s1/experiments/rtx4090-graph-admission/verify_publication.py
PYTHONPATH=src:recipe/cua_s1 python recipe/cua_s1/check_graph_capture_stream.py --output /tmp/streams.json
PYTHONPATH=src:recipe/cua_s1 python recipe/cua_s1/check_graph_pool.py --output /tmp/pools.json
PYTHONPATH=src:recipe/cua_s1 python recipe/cua_s1/check_graph_admission_runtime.py --weights /path/to/weights --output /tmp/parity.json
PYTHONPATH=src:recipe/cua_s1 python recipe/cua_s1/experiments/rtx4090-graph-admission/http_postflight.py.txt --weights /path/to/weights --output /tmp/http.json
```

Use fresh output paths. `SHA256SUMS` covers the archived evidence; the raw original
failure is compressed to keep the review manageable. No model weights or credentials
are included.
