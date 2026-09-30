# Cua-S1 maintainer review follow-up: fresh RTX 4090 validation

Experiment started 2026-09-30 (Asia/Shanghai). This branch stores validation support and raw evidence separately from the runtime-only PRs.

The PR22 fix moves the existing PR33 per-shape stream/pool safety model into the earliest graph-execution PR. Each independently evictable shape owns its capture stream and allocator pool; graphs reset before stream destruction. Resident accounting includes pool reservations and external static buffers. Rejected and evicted candidates retire explicitly even when another reference remains. PR33/37/38 carry the same lifetime fixes.

## Measured sources and CPU validation

| PR | Runtime commit | CPU tests | Validation commit |
|---|---|---|---|
| #17 | [`1a339e1b6363`](https://github.com/Levius-Fubuki/system1-omni/commit/1a339e1b63631271c0a180cb3c848d8698d61bfa) | 115 passed in 17.51s | [`736f12d2cd86`](https://github.com/Levius-Fubuki/system1-omni/commit/736f12d2cd863d25a5344eeb765d7c246faaf5e5) |
| #18 | [`378d10134864`](https://github.com/Levius-Fubuki/system1-omni/commit/378d101348644f855b193402cce56605c7fc9f9e) | 116 passed in 17.07s | [`cbd92b626c08`](https://github.com/Levius-Fubuki/system1-omni/commit/cbd92b626c08374018efdbfbddceb73f639474b7) |
| #22 | [`552c02ce9ce4`](https://github.com/Levius-Fubuki/system1-omni/commit/552c02ce9ce48cfe1ada512dee3b9da55ce3b8ca) | 135 passed in 17.68s | [`da07a404a201`](https://github.com/Levius-Fubuki/system1-omni/commit/da07a404a201eb4b3d0aa1ff3797d35fc25e374c) |
| #33 | [`f4069e0aa4b3`](https://github.com/Levius-Fubuki/system1-omni/commit/f4069e0aa4b32c533d117a7821ff7b3161c3237b) | 187 passed in 17.76s | [`a4aa04c3603a`](https://github.com/Levius-Fubuki/system1-omni/commit/a4aa04c3603a2565068374271c26985c49b5e4bf) |
| #37 | [`d598de3ce03f`](https://github.com/Levius-Fubuki/system1-omni/commit/d598de3ce03f57dd20ff9b730279a4b4979623d4) | 227 passed in 21.31s | [`6743e254998c`](https://github.com/Levius-Fubuki/system1-omni/commit/6743e254998c2574f234c4722ddaa41c9fcb0c78) |
| #38 | [`10b1522e4839`](https://github.com/Levius-Fubuki/system1-omni/commit/10b1522e483999e1830f715bb9970128cfaf323f) | 264 passed in 21.68s | [`9a18da4cd4bf`](https://github.com/Levius-Fubuki/system1-omni/commit/9a18da4cd4bfdd1481ddc79d5f49a7c7440dbd3b) |

The validation commits restore archived test/recipe helpers. `git diff <runtime> <validation> -- src` is empty for each pair. Report source hashes are independently checked against the recorded Git objects. CPU tests use the pinned Python/GPU-host environment; the CPU suite itself uses mocked CUDA where appropriate.

## Real GPU correctness and lifecycle

- Pinned RTX 4090 24 GiB, driver 595.71.05, Torch 2.14.0+cu130, Transformers 5.17.0 and PEFT 0.21.0. BF16 base checkpoint with unmerged adapter; no flash-linear-attention or causal-conv1d.
- PR22 stream regression: 40 simultaneously unique capture streams, 40 initial exact replays and 60 changed-input survivor replays after retiring alternating graphs.
- PR22 pool regression: 40 shapes × 3 segments, exact changed-input replay, reserved-byte-driven eviction and released retired pools.
- Each of PR22/33/37 and the PR38 shared exact/bucket manager: 7 full-checkpoint full-vocabulary-logit comparisons are bitwise equal to eager, including 3 changed-input survivor replays after actual eviction. Retained references confirm explicit retirement. Reservations exceed live allocation and are fully included in the resident byte count.
- PR17: 13 image-reuse cases have exact prepared tensors, language embeddings/positions and full responses; includes changed images, JPEG, structured/non-ASCII candidates and question order changes.
- PR18: full projection versus final-token projection gives identical responses for the two measured cases; output-head shapes confirm the final-token restriction.
- PR22: five full-model cases retain exact original/changed-image/changed-text responses. The cache checker exercises eviction and small-budget eager fallback.
- PR33: five full-model changed-input cases and four admission/fallback policies pass.
- PR37: rule-bucket full-logit parity, 255/256/257/319/320/321 boundaries, real HTTP, independent-model isolation, shutdown and budgets pass.
- PR38: 266 full-logit comparisons and 61 complete responses pass; shared-manager boundary, cross-mode eviction, budgets and retirement pass. Real HTTP and independently loaded model isolation also pass.

The original PR22 stream/pool unit regression fails 7 of 8 tests; the corrected runtime passes all 8. Additional injected candidate cleanup tests fail all four cases against original PR38 and pass against corrected PR33/37/38. Those injected OOM/error tests mock CUDA; they do not claim a real poisoned-context recovery.

## Capture-inclusive bucket and automatic-mode performance

Two cold-cache runs per workload. Synchronized end-to-end `engine.predict` includes capture, per-length parity gates, eviction and fallback. Initial model loading, HTTP parsing and final cache destruction are excluded. Inputs are synthetic 320×240 images with two identical questions. Variant order rotates and caches coexist; these are serial latency measurements, not concurrent throughput or production traffic.

### PR37

| Workload | Run | Eager total (s) | Exact total (s) | Bucket total (s) | Bucket vs eager | Bucket vs exact |
|---|---:|---:|---:|---:|---:|---:|
| hot_four | 1 | 9.297 | 6.849 | 6.719 | 27.73% reduction | -1.90% time |
| hot_four | 2 | 9.309 | 6.810 | 6.708 | 27.94% reduction | -1.50% time |
| churn_twelve | 1 | 14.311 | 13.873 | 10.391 | 27.39% reduction | -25.10% time |
| churn_twelve | 2 | 14.197 | 13.761 | 10.315 | 27.34% reduction | -25.04% time |

### PR38

| Workload | Run | Eager (s) | Exact (s) | Exact window 32 (s) | Bucket (s) | Auto (s) | Auto extra time vs best fixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| hot_four | 1 | 9.496 | 6.881 | 6.893 | 6.859 | 6.892 | +0.48% |
| hot_four | 2 | 9.665 | 6.901 | 6.897 | 6.894 | 6.932 | +0.56% |
| churn_twelve | 1 | 14.499 | 14.007 | 14.357 | 10.577 | 11.352 | +7.32% |
| churn_twelve | 2 | 14.541 | 14.083 | 14.346 | 10.595 | 11.325 | +6.89% |
| hot_cold | 1 | 8.003 | 6.803 | 6.821 | 6.869 | 6.818 | +0.23% |
| hot_cold | 2 | 8.068 | 6.804 | 6.805 | 6.901 | 6.830 | +0.38% |
| shifting_hot | 1 | 25.850 | 23.555 | 23.451 | 17.657 | 17.947 | +1.64% |
| shifting_hot | 2 | 25.896 | 23.522 | 23.588 | 17.700 | 17.896 | +1.11% |
| mixed_holdout | 1 | 17.451 | 15.789 | 17.248 | 13.282 | 14.329 | +7.88% |
| mixed_holdout | 2 | 17.383 | 15.839 | 17.273 | 13.209 | 14.480 | +9.62% |

In the twelve-length rotation, default exact mode cannot admit a layout whose revisit interval exceeds its eight-request admission window; its fallback is part of the result. The PR38 matrix also includes exact admission-window 32 as a controlled comparison. Positive extra time means auto is slower. Auto is a heuristic and these measurements do not establish that it always selects the fastest mode. Eager remains the default. See `results/verified-summary.json` and raw reports for capture-inclusive p95 and all counters.

## Important negative result for PR22

The five-case PR22 matrix warms caches before timing, so it is separate from the cold-cache PR37/38 matrix. With eight distinct questions and the correctly enforced 1 GiB resident pool limit, PR22 repeatedly evicts and captures layouts: eager p50 is 841–847 ms while graph p50 is 4569–4646 ms. Other measured cases improve, but graph execution is not a universal speedup. PR33 separately adds admission/capture-work limits; those controls are intentionally not folded into the PR22 safety fix.

## Reproduction and provenance

Use the validation commit for the PR being reproduced, its archived pinned dependencies, and upstream-verified weights with `weights.lock.json` beside the checkpoint. Exact commands are in `results/checks.json`; replace the host-specific paths with your local paths. `source-manifest.json` identifies runtime, support and validation revisions. The independent verifiers recompute source hashes, response equality, timing summaries and resource counters.

A first custom eviction-checker attempt used out-of-place arithmetic, changing the singleton-batch stride and correctly selecting a new graph layout. The diagnostic and failed attempt are retained. The corrected checker changes content in-place while retaining captured strides; it does not weaken replay or equality assertions. Production code did not change during that checker correction.

All measurements concern the Python multimodal worker. They do not accelerate or validate the separate native Rust/CUDA worker. One GPU, two timing runs and synthetic serial workloads do not establish broad deployment performance, GUI task accuracy or concurrency behavior.
