# Open-Jev selective prefill packing on H200

Packing input and gate/up projections across candidates reduces mean warm HTTP
latency by **18.04% for four-question requests**, **20.11% for eight-question
requests**, and **10.67% for mixed Choice/Noul/Score examples**. All compared
answers and token counts match the current-main native reference exactly.
Single-question mean latency increases 0.21%; its p95 increases 1.09%,
within the declared 2% regression limit.

These are within-request, concurrency-1 measurements on one H200, BF16, from
2026-10-06. The multi-question workloads are synthetic transformations of real
JevBench requests. They do not establish production traffic performance,
general model accuracy, or cross-request batching throughput.

## Implementation and numerical constraint

The Open-Jev batch adapter groups prepared candidates in order, at most 16
sequences and 4096 tokens per group. A longer prompt executes alone at its
original length. Selected GEMMs share packed rows and weights; token-local
normalization and activation operations use the same rows without padding.
Each sequence keeps its own positions, causal attention range, convolution
boundary, GDN state and final-position readout. Scalars are regrouped before
the original question-level calibration. The scheduler still admits one whole
request at a time.

Output/down projections retain the original per-sequence GEMM shape. Changing
those shapes can change cuBLASLt split-K reductions and BF16 rounding. CUDA
Graphs are keyed by ordered sequence lengths, rather than total packed length,
so equal-size batches with different boundaries cannot share the wrong graph.
The PR uses upstream CUDA ABI5 without changing its interface, weights, precision,
temperature or truncation policy. Input and gate/up calls select packed GEMMs
explicitly; a separate helper keeps output/down calls sequence-local. Native
vision input support and upstream graph cache-miss handling are preserved.

## Matched warm HTTP results

Each cell gives the two measured pass means in milliseconds. Reduction uses
their arithmetic mean; the two observations are variability, not a confidence
interval. Each arm reuses one worker/frontend pair for all slices, after genuine
readiness, a validated first inference and one excluded feasibility pass per slice.

| Workload | Requests/pass | Current main, pass 1 / 2 | Packed, pass 1 / 2 | Mean reduction |
| --- | ---: | ---: | ---: | ---: |
| Real single-question JevBench | 74 | 48.048 / 48.143 | 48.307 / 48.083 | -0.21% |
| Four repeated questions | 60 | 98.703 / 98.618 | 80.800 / 80.915 | 18.04% |
| Eight repeated questions | 60 | 197.470 / 197.502 | 157.558 / 157.978 | 20.11% |
| Mixed Choice/Noul/Score | 12 | 258.378 / 257.593 | 230.449 / 230.490 | 10.67% |

Four/eight-question p95 decreases 7.66%/10.14%, using the mean of the two pass
p95 values. The original 74 cases contain one Noul candidate each and span
80–3399 tokens. The repeated-question slices select the 60 original cases of
at most 400 tokens, then duplicate each question four/eight times under distinct
IDs in the same request. The mixed slice uses the repository's three-question
example, with 12 progressively longer contexts, covering distinct candidate
lengths and packing across the token budget.

The two measured passes per arm total **824 successful HTTP requests and 3320
decisions**. All answer fields, usage, model identity and metadata other than
elapsed time match the corresponding baseline. Maximum probability/Score drift
is **0.0**, with **zero decision flips and zero request failures**. This is output
fidelity against native current main, not a new full-precision accuracy evaluation.

All predeclared gates pass: at least 15% mean reduction for both repeated-question
slices, at most 2% single-question mean/p95 regression, drift at most 0.001,
zero flips and zero failures. No measured repetitions were added.

## Frozen controls and raw evidence

- Baseline source: `47eff9cdeda01e4847a4fb9634a43f2cab6a233f`.
  [Source verification](artifacts/20261006/source-verification.json) pins the
  candidate runtime commit and verifies its files match the measured source hashes.
- Exact scheduler GPU 2, UUID `GPU-cbf66259-f4ab-0ede-1811-82037dde5924`, SM90,
  NUMA 0 and CPUs 0–15. The driver/scheduler calls this device L20X; the existing
  device records identify it as H200. CUDA 13.0, nvcc 13.0.88 and rustc 1.98.1.
- Same merged BF16 export, tokenizer, trained head, temperature
  `2.5343690298472983`, max length 16384 and 32 MiB cuBLASLt workspace.
  Base revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; checkpoint revision
  `28cf73067d5b337860bbef3c85b8b82ba8730956`.
- Same CUDA ABI5 library and frontend binary; primary comparison has
  `CUA_S1_GRAPH=0` and no profiler. Only native Rust packing/execution changes.
- Reuse the prepared model and environment. Downloads, copies, builds,
  process-to-readiness, first inference and feasibility passes are excluded.
  No shared caches are dropped and no clocks are changed.
- HTTP timing includes localhost frontend forwarding, tokenization, inference
  and UTF8 response-body receipt. Request JSON construction and response JSON
  parsing are outside the timer. P95 uses nearest rank.

[Plan and hashes](artifacts/20261006/plan.json),
[input manifest](artifacts/20261006/harness/requests.json),
[collector](artifacts/20261006/harness/collect.py),
[summary](artifacts/20261006/summary.json), and the adjacent feasibility/measured
JSONL files preserve raw requests, responses, latency and repeated-run results.
[Final cleanup](artifacts/20261006/final-cleanup.json) verifies task-owned processes
exited and GPU 2 returned to available with 0 MB used.

This comparison was refreshed after integrating native vision main. The complete
previous campaign remains in Git commit `03dbd11972389dadd06a945dc4dd6bc1e16687f8`
and in the local run archive; its ABI4 timings are separate historical results.

For reproduction, prepare a baseline worktree at the pinned SHA and the candidate
worktree, build both with the same release options and CUDA library, and reuse
the pinned export from the native recipe. Copy the artifact directory into a new
run directory, update its host-specific executable/model paths and freeze new
hashes before running its `harness/collect.py` through the verified scheduler:

```sh
gpu run --gpu-ids <available-exact-id> --wait 10m --timeout 40m \
  --note "Open-Jev selective packing HTTP A/B" -- \
  numactl --membind=<verified-numa> --physcpubind=<fixed-cpus> \
  <prepared-python> <new-run>/harness/collect.py
```

The collector also asserts the frozen visibility, GPU UUID and affinity; update
those assertions together with the plan when reproducing on another host.

## Rejected attempts and validation

The first cuBLASLt search tested 120 algorithm/shape configurations at 107, 936
and 3399 tokens. It did not meet its 10% weighted-GEMM improvement gate. A fully
packed prototype then failed feasibility at probability drift **0.0021800553**,
above the fixed **0.001** limit. Both attempts retain their local raw artifacts;
[rejected-attempts.json](artifacts/20261006/rejected-attempts.json) records their
outcomes. The selective variant uses a separate frozen experiment with the same
numerical and performance gates; no tolerance was relaxed.

Repository checks pass: formatting, strict Clippy, locked workspace tests
(**90 passed, 13 ignored**) and release workspace build. Reserved GPU validation
passes all **six CUDA reference tests** and the full-checkpoint packing test with
`CUA_S1_GRAPH=1`, including unequal lengths, reordered equal-total shapes,
17-candidate splitting, cache hits with changed token IDs, replay and a later singleton.
[GPU test records](artifacts/20261006/gpu-validation.json) and adjacent logs
preserve those commands/results. The graph checks are correctness tests, with no
additional performance measurements.
The final cardinality/isolation test also passes under Nsight Systems;
[trace evidence](artifacts/20261006/graph-replay-evidence.json) records **4 actual
CUDA Graph launches**, confirming replay executes during the regression check.

Other GPU architectures, concurrency above 1, full production distributions,
peak memory, cold capture latency and packing combined with graph performance
remain unmeasured. Packed scratch grows with the group's token count; the same
per-prompt maximum length remains supported for singletons.
