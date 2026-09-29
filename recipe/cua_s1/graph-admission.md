# Capture admission experiment

`benchmark_graph_admission.py` measures the entire synchronized `engine.predict`
request, including cold misses and capture cost. It pairs eager,
legacy-policy-with-execution-fix, and current admission on one engine. The legacy variant imports the original #24 runtime but replaces its
`_GraphSegment`, `_ShapeEntry`, and `GraphRuntime._run_segments` with their
current implementations. Both variants therefore use the same implementation
for per-shape graph pools, owned streams, and reserved-memory accounting. Historical admission and
cache policy remain unchanged. This is a comparison of admission policies
with the same execution and accounting fixes; it does not reproduce raw #24
performance. Each variant retains its own
runtime across a schedule; all runtimes are destroyed between runs. Execution
order reverses on alternating requests and runs. No allocator purge occurs
between paired variants. Both graph pools coexist, so allocator memory totals
include both pools and must not be interpreted as isolated variant memory.
Per-runtime cache occupancy is recorded separately.

Use a clean committed source checkout and put output outside it. Extract the
historical runtime outside the repository. Its full 40-character revision is
required, and its bytes must exactly match `git show` for that revision before
import. Reports preserve the original file SHA-256, revision, explicit segment
override, and current `graph_runtime.py` SHA-256:

```sh
git show 2336a085fc090c1c6ac3a297f91d826b66949f30:src/models/cua_s1/multimodal/graph_runtime.py > /tmp/legacy-graph-runtime.py
PYTHONPATH=src:recipe/cua_s1 python recipe/cua_s1/benchmark_graph_admission.py \
  --weights /path/to/weights --output /path/to/fresh-admission-output \
  --legacy-runtime /tmp/legacy-graph-runtime.py --legacy-revision 2336a085fc090c1c6ac3a297f91d826b66949f30 --runs 2
python recipe/cua_s1/verify_graph_admission.py /path/to/fresh-admission-output/report.json
```

`--case` can be repeated to select schedules; by default all four run:
48 hot-four requests, 72 twelve-shape churn requests, 40 hot/cold requests,
and 128 shifting-hot requests. Every request contains two identical questions;
prompt repetition counts produce distinct token lengths below 2048.

Reports are checkpointed after every paired request and preserve full responses,
configuration, environment, fixture hash and contents, source SHA, token counts,
latencies, memory and counter deltas. Equality requires both the probability
comparison and full response dictionary equality. The independent stdlib verifier
checks those records and recomputes totals, medians, nearest-rank p95 and cumulative
metrics. A failed or interrupted report is not accepted as complete evidence.

`first_cumulative_crossing_request` is the first nonnegative cumulative eager
minus graph balance and can precede capture. The separate
`post_capture_sustained_break_even_request` is the earliest request at or after
first capture with nonnegative cumulative balance for the rest of the observed
schedule. It is null when no capture occurs or the final balance is negative.
It describes the observed horizon, not a prediction of future amortization.
Historical #24 raw JSON and its verifier retain their original schema.

## Historical stream failure and diagnostic mode

The initial extended experiment using unpatched legacy #24 failed during
`shifting_hot` at request 104 after more than 400 completed request pairs, with
an SGEMM replay error. That incomplete run is excluded from performance
conclusions. An intermediate exclusive-stream run from `b094c92` completed the
matrix but failed the publication verifier because `shifting_hot` recorded capture OOMs;
it is also excluded from performance conclusions. Both policy variants now
use per-shape pools and owned streams, with reserved-memory accounting,
to avoid both cross-shape workspace invalidation and unaccounted pool growth.
Every published policy-comparison schedule must be rerun
from the updated clean source; earlier partial samples cannot be combined with
those runs.

`--unpatched-legacy` preserves the original segment implementation solely for
diagnosing the historical failure. It still requires verified original bytes
and a full revision. Reports label this mode `unpatched-legacy-diagnostic` and
record `runtime_segment_override` as `none; unpatched historical runtime
diagnostic`. Such runs are not the default policy comparison and must be
described separately, even if a particular schedule completes.
