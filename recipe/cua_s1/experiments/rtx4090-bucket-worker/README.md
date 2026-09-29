# Cua-S1 rule-bucket worker — RTX 4090, 2026-09-29

The worker now exposes explicit `--graph-mode rule-bucket` alongside existing
`--graph` / `--graph-mode exact`. Eager remains the default. Rule buckets use
instance-local decoder calls and a pinned cache-free prefill adapter; they do
not replace Transformers globals or model methods. Only internal DeltaNet rule
inputs are padded. Other model operations retain their actual token length.
Automatic strategy selection and simultaneous exact/bucket worker caches are
outside this change.

## Correctness and lifecycle

Acceptance retains exact complete-vocabulary-logit gates at capture and at the
first appearance of each real length in a resident bucket. Changed images and
text are separately tested. No numerical tolerance was relaxed. Unsupported
inputs, admission/budget limits and rejected lengths use eager inference.

A new strict rule-input signature check exposed a boundary bug: PyTorch's
zero-width padding preserves transposed QKV strides, whereas nonzero padding
produces contiguous inputs. Thus 255 and 256 tokens mapped to one bucket with
different layouts. The failed GPU check at `249b0e3` is retained in
`boundaries-before-fix.json` and its log. A CPU regression first reproduced the
failure, then `pack_rule_inputs(...).contiguous()` fixed it without weakening
layout validation. Final runtime revision is
`74760be2df0727d2d71467a52f0f0c2f4c92a398`.

The engine serializes lifecycle operations, rejects predictions after close,
and explicitly retires graph entries even with external references alive.
HTTP shutdown drains accepted handlers before engine cleanup. Mode/config,
rejected-length reuse, static layouts, upstream prefill equivalence and shutdown
are covered by 218 passing Cua-S1 tests. CI-scoped Ruff checks and formatting pass.

`boundaries.json` verifies 255/256/257/319/320/321 tokens in forward and reverse
order: 12 exact full-logit matches, three captures and nine replays. Separate
policies verify eviction/cooldown, 1 MiB memory-budget rejection, and a one-capture
budget with eager-equivalent outputs.

`http.json` exercises the actual CLI-selected engine and loopback HTTP server:
cold, capture, changed-image and changed-text responses equal eager completely.
It additionally loads two independent models/runtimes simultaneously. The second
runtime captures and replays without changing the first runtime's statistics or
entries; the first still replays after the second closes. Globals and forward
methods retain their identities. Closing the server empties graph blocks and
pool ownership even while the checker retains entry references, and subsequent
prediction fails explicitly. This interleaved isolation check is not a concurrent
GPU throughput benchmark.

`parity.json` covers five cases (320×240 with 2/8 questions, 640×480 with
8 questions, long prompts, and eight distinct questions). All 136 independent
complete-logit comparisons and all 20 cold/capture/changed-image/changed-text
responses match eager. These checks expand limits to 4,096 tokens, 16 captures,
10,000 ms capture work and 4 GiB cache; timing uses default limits. In this raw
checker output, per-case `config` is the base configuration before its local
worker factory replaces `mode` with `rule-bucket`; top-level `runtime: worker`
identifies the actual path. The HTTP check uses the real CLI/engine selection.

## Measurement boundaries

One RTX 4090, PyTorch 2.14.0+cu130, Transformers 5.17.0 and PEFT 0.21.0; pinned
BF16 base and unmerged multimodal LoRA. Neither flash-linear-attention nor
causal-conv1d is installed. Results concern this fallback implementation.

Four variants share the same loaded model: eager, exact Graph, worker rule
buckets (`bucket` in JSON), and the prior experimental rule algorithm (`recipe`).
The recipe comparison uses the current shared Graph runtime/lifecycle, not an
unchanged historical commit. All 24 variant permutations cycle across request
positions; run two offsets by half a cycle. Graph caches start empty for each
workload/run. Loading is excluded; synchronized `engine.predict` includes input
preparation, vision, two identical questions on a synthetic 320×240 screenshot,
cold admission/capture, first-length checks and replay. In-request eviction
would be included, but none occurs in this matrix; final cache retirement is
outside timing. This serial latency matrix does not measure HTTP throughput or
concurrent requests.

The three caches coexist during timing. Process allocator totals are combined,
not attributable to an individual mode. Separate parity diagnostics measure
modes with one live runtime at a time. A lower bucket count does not establish
lower memory use.
For the isolated 236-token diagnostic, exact cache ownership is 120,815,616 bytes
and worker rule ownership is 284,295,168 bytes. Peak process reservations are
10,101,981,184 and 10,259,267,584 bytes respectively. These are one-shape
diagnostics, not whole-matrix peak comparisons.

## Timing results

Two cold runs per workload, 240 request groups / **960 timed predictions**. Cells show run 1 / run 2; total time includes capture and first-length gates.

| Workload / mode | Total (s) | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|
| hot_four / eager | 11.05 / 10.95 | 227.79 / 225.34 | 238.62 / 235.62 |
| hot_four / exact | 7.48 / 7.45 | 108.17 / 107.62 | 529.83 / 522.27 |
| hot_four / bucket | 8.08 / 7.91 | 142.77 / 141.92 | 257.69 / 236.17 |
| hot_four / recipe | 8.05 / 7.97 | 144.94 / 143.84 | 238.65 / 239.12 |
| churn_twelve / eager | 16.98 / 16.75 | 236.19 / 234.80 | 242.68 / 239.21 |
| churn_twelve / exact | 16.50 / 16.34 | 229.86 / 228.89 | 235.44 / 237.82 |
| churn_twelve / bucket | 12.27 / 12.18 | 145.26 / 143.57 | 245.93 / 240.44 |
| churn_twelve / recipe | 12.35 / 12.33 | 146.49 / 145.31 | 246.63 / 243.26 |

For twelve rotating lengths, worker buckets reduce total time **27.30–27.73%
versus eager** and **25.48–25.64% versus exact mode**. Exact makes zero captures
or replays because the 12-request revisit interval exceeds its eight-request
admission window. This measures useful bucket reuse versus conservative eager
fallback, not an exact-cache eviction storm. Churn p95 is still slightly above
eager in both runs (240.44–245.93 versus 239.21–242.68 ms).

For four hot lengths, worker buckets reduce total time 26.83–27.76% versus eager,
but take **6.13–8.00% longer than exact Graph**. Each worker run captures two
buckets; exact captures four hot layouts. Relative to the prior recipe algorithm,
worker total time ranges from 0.44% slower to 1.25% faster across these four runs.
This small two-run comparison supports retaining the measured behavior, not a
statistically established integration speedup.

All 960 responses match eager completely; no capture errors, OOMs, numerical
rejections or length rejections occur in the matrix. The independent stdlib
verifier recomputes metrics and counters, checks response equality and cache
bounds, and checks source hashes against measured Git objects. Read-only code
review found no remaining actionable integration/lifecycle defects.

## Reproduction

Use the clean source revision recorded in each JSON and fresh output paths:

```sh
PYTHONPATH=src:recipe/cua_s1 python recipe/cua_s1/check_graph_buckets.py \
  --weights /path/to/weights --worker --output /path/to/parity.json
PYTHONPATH=src:recipe/cua_s1 python recipe/cua_s1/check_graph_buckets.py \
  --weights /path/to/weights --worker --boundaries-only \
  --output /path/to/boundaries.json
PYTHONPATH=src:recipe/cua_s1 python recipe/cua_s1/check_bucket_worker_http.py \
  --weights /path/to/weights --output /path/to/http.json
PYTHONPATH=src:recipe/cua_s1 python recipe/cua_s1/benchmark_graph_buckets.py \
  --weights /path/to/weights --output /path/to/matrix --runs 2 \
  --case hot_four --case churn_twelve --kind worker --include-recipe
python recipe/cua_s1/verify_graph_buckets.py /path/to/matrix/report.json --require-replay
```

The worker can be started with:

```sh
PYTHONPATH=src python -m models.cua_s1.multimodal.server \
  --base /path/to/weights/Qwen3.5-4B \
  --adapter /path/to/weights/cua-s1-4b-0.2/multimodal \
  --graph-mode rule-bucket --graph-bucket-width 64
```

No new CUDA/Triton kernels, compilation, native multimodal bridge, dynamic
batching, real-GUI task accuracy, or broad deployment claims are made here.
