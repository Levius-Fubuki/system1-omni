# Cua-S1 rule-level Graph buckets — RTX 4090, 2026-09-29

This experiment builds a recipe-only bucketed execution path on PR #33. It keeps
all projections, convolutions, layer normalization, MLP, rotary positions and full
attention at their original token length. Only the internal cache-free DeltaNet
rule receives zero-padded query/key/value/decay/update tensors. Twelve measured
lengths (236–302 tokens) map to two buckets (256 and 320).

The implementation is **not enabled in the worker**. It temporarily replaces the
installed Transformers rule function under the runtime lock and is limited to a
serial, single-engine experiment. It is not safe as a multi-engine global hook.
A production integration needs an explicit per-model call site. The original
function is restored on both successful and failed execution.

## Correctness and rejected alternative

Padding whole linear-attention segments changes projection results on this BF16
stack. The original probe rejects both buckets and falls back to eager. Four
layer diagnostics find the first observed leaf-module difference in QKV or output
projection; exact Graph equals exact eager, and padded Graph equals padded eager
for the first three-layer segment. Full-model logits differ by up to 0.25 in
these four fixtures. This isolates padding-related numerical changes from Graph
capture; it does not identify the exact cuBLAS kernel/accumulation mechanism.
No acceptance tolerance was relaxed. The negative probe is retained separately.

The subsequent rule-only path gates complete vocabulary logits with
`torch.equal` on capture and on each new real length in a resident bucket. A
mismatching real length falls back to eager; verified lengths can continue.
Verification metadata lives with the cached entry and is discarded on eviction.
This first-input gate is not a proof for all content; changed-content regressions
are therefore tested independently.

`parity-memory.json` covers five cases: 320×240 with 2/8 questions, 640×480 with
8 questions, a long-prompt case, and eight distinct questions. Cold requests,
capture, changed image and changed text produce identical complete responses.
All 136 independent complete-logit comparisons pass, including replay on changed
inputs. Correctness checks expand the Graph limits to 4,096 tokens, 16 captures,
a 10,000 ms capture budget and 4 GiB cache; timing uses #33 defaults.

## Measurement boundaries

The matrix uses clean revision `9142e23e7c798ee350de8e4b7c0f5ade28012a7f`, on
one RTX 4090, PyTorch 2.14.0+cu130, Transformers 5.17.0 and PEFT 0.21.0. We reuse
the pinned BF16 base and multimodal LoRA without merging or modifying weights.
Neither flash-linear-attention nor causal-conv1d is installed. Results apply to
the existing fallback implementation, not an already fused FLA deployment.

Timing surrounds synchronized `engine.predict`, including preparation, image
encoding, two identical questions, first-length verification, admission, capture,
replay and retirement. Loading is excluded. All six variant orderings cycle over
request positions; the second run shifts to the reversed permutations. Caches
start empty for each workload/run. Both graph caches coexist during paired
measurements, so process allocator totals cannot be attributed to one variant.

Separate reservation diagnostics run eager, exact and rule modes with only one
live runtime at a time and an allocator clear between modes. For one 236-token
shape, accounted Graph cache bytes are 120,815,616 (exact) and 284,295,168 (rule).
Peak process reservation is 10,101,981,184 and 10,259,267,584 bytes respectively.
These are one-shape diagnostics, not whole-matrix peak comparisons. Rule buckets
retain more static rule inputs; fewer buckets do not imply lower memory use.

## Timing results

Two cold runs per workload; 240 request triplets / **720 timed predictions**. Times below include capture and first-length validation. Each cell is run 1 / run 2.

| Workload / mode | Total (s) | p50 (ms) | p95 (ms) | Captures / run |
|---|---:|---:|---:|---:|
| hot_four / eager | 11.38 / 11.15 | 232.81 / 229.63 | 248.76 / 240.78 | 0 |
| hot_four / exact | 7.53 / 7.50 | 108.48 / 108.22 | 533.16 / 527.54 | 4 / 4 |
| hot_four / bucket | 8.25 / 8.18 | 147.81 / 146.81 | 251.78 / 252.64 | 2 / 2 |
| churn_twelve / eager | 17.24 / 17.17 | 241.04 / 240.39 | 245.89 / 244.38 | 0 |
| churn_twelve / exact | 16.85 / 16.72 | 235.05 / 234.59 | 240.58 / 239.99 | 0 / 0 |
| churn_twelve / bucket | 12.60 / 12.55 | 149.21 / 147.89 | 250.50 / 253.72 | 2 / 2 |

For twelve-length rotation, rule buckets reduce total time by **26.90–26.93% vs eager** and **24.92–25.20% vs exact mode**. Exact mode performs zero captures and zero replays: the 12-request revisit interval exceeds its eight-request admission window. This is useful bucket admission versus conservative eager fallback, not a comparison against repeatedly evicted exact graphs. Its 2.30–2.64% timing difference from eager is not a Graph speedup.

For four stable hot lengths, rule buckets reduce total time by **26.63–27.53% vs eager**, but take **8.99–9.44% more time than exact Graph mode**. Capturing only the internal rule leaves more operations eager. It also retains more static rule inputs per bucket. A workload-dependent choice is preferable to replacing exact mode globally.

Each rule run captures two buckets; hot-four has 90 replays and two additional first-length checks, while twelve-length rotation has 138 replays and ten checks. Across the matrix there are no capture errors, OOMs, numerical rejections, or response mismatches. Churn p95 remains slightly higher than eager (250.50–253.72 vs 244.38–245.89 ms), so this is not a universal tail-latency improvement. The two synthetic runs do not establish statistical significance or a broad production speedup.

## Validation

- `boundaries.json` verifies complete-logit equality at 255/256/257/319/320/321 tokens, then revisits them in reverse order: three captures and nine replays. Three additional policies exercise eviction cooldown, a 1 MiB memory budget and a one-capture budget, with eager equality.
- 193 Cua-S1 tests pass; CI-scoped Ruff checks and formatting pass.
- The independent stdlib verifier checks source hashes against measured Git objects, complete responses, all reported timing metrics, aggregated counters and cache bounds.
- Read-only independent code/evidence review found no significant defects within the declared serial experimental scope.

The matrix and five-case parity/memory checks use the revision above. Boundary checks use `d082d54804c31e92203cd8bb7b76ef14f53ccd06`, which only adds checks/tests; runtime and benchmark code are unchanged.

## Reproduction

Use the source revision in each raw JSON, with the pinned environment and existing
verified checkpoints. Choose fresh output paths:

```sh
PYTHONPATH=src:recipe/cua_s1 python recipe/cua_s1/benchmark_graph_buckets.py \
  --weights /path/to/weights --output /path/to/matrix --runs 2 \
  --case hot_four --case churn_twelve --kind rule
PYTHONPATH=src:recipe/cua_s1 python recipe/cua_s1/check_graph_buckets.py \
  --weights /path/to/weights --output /path/to/parity-memory.json
PYTHONPATH=src:recipe/cua_s1 python recipe/cua_s1/check_graph_buckets.py \
  --weights /path/to/weights --boundaries-only --output /path/to/boundaries.json
python recipe/cua_s1/verify_graph_buckets.py /path/to/matrix/report.json --require-replay
```

Reproduce the negative segment variant with `--kind segment --case probe --runs 1`.
`diagnose_graph_buckets.py` records leaf-module and Graph/eager differences. The
initial rule prototype rejected a harmless `use_cache=False` metadata argument;
its failed run is preserved, and the accepted call contract has a regression test.

This experiment does not establish HTTP throughput, concurrency, dynamic batching,
new CUDA/Triton kernels, compilation, native multimodal execution, task accuracy
on a real GUI dataset, or performance on another GPU/software stack.
