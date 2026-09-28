# Mixed-length segmented CUDA Graph cost — RTX 4090, 2026-09-28

The existing [segmented Graph runtime](../rtx4090-graph-runtime/README.md) was
measured after capture on fixed inputs. This experiment includes first use,
capture, replay, and eviction in request latency. It tests whether the current
exact-length cache helps when prompt lengths change between requests. No serving
policy or model code was changed.

## Workload and measurement

One pinned BF16 Cua-S1 multimodal adapter ran on an RTX 4090 (sm_89), with
PyTorch 2.14.0, Transformers 5.17.0, and PEFT 0.21.0. The optional
`causal_conv1d` and `flash-linear-attention` packages were absent. The base and
adapter revisions, full package versions, CUDA version, GPU, and source commit
are recorded in each raw report.

Every request contains the same synthetic 320×240 screenshot and two identical
choice questions. Repeating the instruction produces 12 distinct processed
lengths, 236–302 tokens per question. The **hot four** schedule repeats lengths
1, 2, 4, and 8 six times (24 requests). The **churn twelve** schedule repeats
all 12 lengths three times (36 requests), exceeding the cache's eight-layout
limit. These are controlled stress cases, not a measured production traffic
distribution.

For each threshold and schedule, two runs start with an empty Graph cache. Every
request is timed once through eager and once through Graph, in alternating
order. Synchronization brackets `engine.predict`, so the totals include image
preprocessing, vision, language, scoring, and any eager gate or capture that
occurs during a request. Initial model load, three eager kernel warmups, request
parsing, HTTP, and concurrency are excluded. The complete response is compared
per pair; every candidate probability, choice, usage count, and model identity
matched exactly. There are **360 paired requests, 720 timed predictions**.

`min_uses` counts question forwards, not requests. With two repeated questions,
the default threshold of 2 captures on the first request for a new length. The
Graph cache holds at most eight layouts and 1 GiB of measured live allocations;
the reports separately record PyTorch's peak reserved memory, which includes
allocator pooling and capture-time allocations.

## Results

The percentage is the reduction in **total synchronized request time for the
whole schedule**, including first captures; a negative value means Graph was
slower. Each cell shows runs 1 / 2.

| Capture threshold | Hot four, 24 requests | Churn twelve, 36 requests | Captures / evictions per churn run |
| --- | ---: | ---: | ---: |
| 2 (worker default) | **10.2% / 9.6% faster** | **246.0% / 303.6% slower** | 36 / 28 |
| 4 | 2.9% / 1.8% faster | 157.2% / 166.8% slower | 24 / 16 |
| 8 | 13.3% / 12.8% slower | 3.7% / 3.1% faster* | 0 / 0 |

\* At threshold 8, the churn schedule never reaches capture: all Graph-path
forwards fall back to eager. Its small measured difference is run-to-run timing
variation, not Graph acceleration.

The default threshold captures each of the four hot lengths once. Its 24-request
Graph total first drops below eager at request 19 in both runs. For churn it
captures all 12 lengths on each pass, evicting 28 entries per run; capture alone
takes 14.8 and 19.1 seconds across the 36 requests. Raising the threshold to 4
reduces capture frequency but also delays hot-set replay. Raising it to 8
prevents capture under churn, while the hot schedule has too few subsequent
replays to repay its four captures. The maximum observed PyTorch reserved memory
was about 12.1 GiB for hot four and 16.1 GiB for churn with captures, versus
9.3 GiB for churn with no captures. This is allocator memory, not the Graph
cache's measured entry size.

The result shows a workload-dependent admission problem. It does not establish
that padding several lengths into one bucket is numerically safe: the current
runtime keys 3D positions, masks, and exact embedding layouts, and Gated
DeltaNet may consume padded tokens differently. A follow-up should measure
real length reuse and test a bounded admission or bucketing policy against
exact eager outputs before changing the worker default.

## Evidence and reproduction

[`default.json`](default.json) was measured from clean source commit `1019581`;
[`min4.json`](min4.json) and [`min8.json`](min8.json) were measured from clean
source commit `6e0a432`, which adds only the benchmark's threshold option.
Each report contains every latency sample, request order, runtime counter delta,
capture time, cache occupancy, and memory observation. The independent
[`verify_results.py`](verify_results.py) recomputes the totals and medians,
checks schedule coverage and cache limits, and requires exact response parity.
It verified all three reports and 360 paired requests. On the GPU host, all
118 Cua-S1 tests were reported as passing. The later source `6e0a432`
adds `test_capture_threshold_must_be_positive` relative to `1019581`,
accounting for the PR summary’s 119-test count; the original execution log
is not retained here, so this is a source-count explanation rather than a
new claim that all 119 ran on the GPU host. Ruff lint and format checks passed for the new
benchmark. SHA-256 hashes are in [`SHA256SUMS`](SHA256SUMS).

To reproduce the historical runtime behavior, check out the measured source
`6e0a432` in a separate checkout, with pinned weights and the GPU environment
described above. Current source changes admission and metric semantics.

```sh
git checkout 6e0a432
for threshold in 2 4 8; do
  PYTHONPATH=src:recipe/cua_s1 python recipe/cua_s1/benchmark_graph_mixed_shapes.py \
    --weights /path/to/weights --output "/path/to/fresh-output-$threshold" \
    --runs 2 --graph-min-uses "$threshold"
done
python recipe/cua_s1/experiments/rtx4090-graph-mixed-shapes/verify_results.py \
  recipe/cua_s1/experiments/rtx4090-graph-mixed-shapes
```
