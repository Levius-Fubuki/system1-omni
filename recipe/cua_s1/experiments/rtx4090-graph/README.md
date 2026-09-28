# Cua-S1 multimodal CUDA Graph feasibility — RTX 4090, 2026-09-28

CUDA Graph replay of the language forward reduces synchronized `engine.predict`
latency for these **fixed-shape, short-instruction** requests. This is an
experiment on the Transformers/PEFT multimodal worker after the image reuse and
last-token projection changes in #17 and #18. It does not add a production Graph
path or make a claim about the separate native text worker in #19.

The graph captures `model(inputs_embeds, position_ids, attention_mask,
logits_to_keep=1, use_cache=False)` after image encoding, text embedding, image
insertion, and three-dimensional position construction. Both measured variants
use `use_cache=False`; the eager variant returns exactly the original worker
response. Graph replay copies the current tensors into long-lived buffers whose
shape, stride, dtype, and device match capture. A graph is keyed by that tensor
layout. Image processing, image encoding, prompt preparation, candidate scoring,
and response assembly run normally on every request.

## Result and correctness boundary

The table gives p50 milliseconds from two alternating runs of 20 calls per
variant and case (240 timed predictions total). Each case used three warmup calls
per variant. Timings synchronize CUDA around each whole `engine.predict`; they
exclude request parsing, fixture creation, model load, graph capture, HTTP, and
concurrent traffic. Percentages compare Graph to eager within the same run.

| Synthetic case | Eager p50, runs 1 / 2 | Graph p50, runs 1 / 2 | Reduction, runs 1 / 2 | First capture with 3 warmups |
| --- | ---: | ---: | ---: | ---: |
| 320×240, short instruction, 2 repeated questions | 181.42 / 183.28 | 100.31 / 100.38 | 44.7% / 45.2% | 355 ms |
| 320×240, short instruction, 8 repeated questions | 669.81 / 684.26 | 337.57 / 338.21 | 49.6% / 50.6% | 347 ms |
| 640×480, short instruction, 8 repeated questions | 825.78 / 786.77 | 562.90 / 561.72 | 31.8% / 28.6% | 416 ms |

For all three rows, eager produced the original response exactly. Graph retained
the selected choices, model identity, usage, and answer keys. The largest
candidate probability difference from the original response was 0.000421;
the experiment's absolute probability threshold was 0.002. Replaying after
replacing the image with a same-size black image, then returning to the original,
passed with at most 0.000781 probability difference. Replacing “Save” with a
different same-token-length verb, then returning to the original, also passed.
The saved reports include these checks for each case. The post-capture
`torch.cuda.memory_allocated()` increase was 9–17 MiB for one graph. This
allocator measure is not a total GPU-card memory measurement.

Two additional cases **failed the correctness gate** and were not timed as
supported Graph paths:

| Case | Observed failure |
| --- | --- |
| 640×480, 8 distinct questions | Seven graph layouts were captured. Candidate probability difference reached **0.065181** on the original input, despite the same chosen options. |
| 640×480, long instruction, 8 repeated questions | The original input and changed image were within 0.002, but a same-length changed question produced **0.051902** maximum probability difference. |

Those deviations exceed the 0.002 experimental gate. The first appears during
the original distinct-question inference; the second appears only after changing
text. The cause is not yet isolated. This evidence supports a bounded
fixed-shape study and argues for tracing the captured language kernels before
adding Graph replay to a production worker. It does not establish correctness
for arbitrary questions, prompt lengths, GPUs, concurrent requests, or the
native engine. In particular, the fixed-input long-instruction speedup observed
during exploratory timing is not counted as usable.

## Provenance and reproduction

- Clean measured source: `809cacc70979b1b4e3a55b8b89be4691d09b0ce4`, on
  the #18 stack. All three reports record this revision with `dirty: false`.
- One NVIDIA GeForce RTX 4090 (24,564 MiB, sm_89), driver 595.71.05, CUDA
  runtime 13.0, Python 3.12.13, PyTorch 2.14.0, Transformers 5.17.0, PEFT
  0.21.0. Pinned BF16 base and unmerged multimodal LoRA were used. The
  `causal_conv1d` and `flash-linear-attention` optional packages were absent,
  so Transformers used its reference fallback kernels, as in #18.
- [paired.json](paired.json) contains every elapsed sample, run order, p50/p95,
  capture cost, allocator measures, and correctness checks. The two rejected
  reports, [distinct-rejected.json](distinct-rejected.json) and
  [long-rejected.json](long-rejected.json), preserve the numerical failures.
  [verify_results.py](verify_results.py) independently recomputes all 240
  samples and checks both rejections.
- The Cua-S1 test suite passed **108 tests** on this GPU host. Ruff lint and
  formatting checks passed. [SHA256SUMS](SHA256SUMS) identifies the published
  report files.
- A complete 186,201-byte evidence archive with fixtures, error logs, package
  versions, and a Git source bundle was copied off the GPU host and verified
  against server SHA-256
  `11831981ae8ea1c414075d563284b40b378a3471a8164c26b9a281986a12bee0`.

Run from the measured source with the pinned weights and environment:

```sh
PYTHONPATH=src:recipe/cua_s1 python recipe/cua_s1/benchmark_multimodal_graph.py \
  --weights /path/to/weights --output /path/to/new-output \
  --case 320x240-short-q2 --case 320x240-short-q8 \
  --case 640x480-short-q8 --warmup 3 --runs 2 --iterations 20
python recipe/cua_s1/experiments/rtx4090-graph/verify_results.py
```

To reproduce a rejection, run the same command with only
`--case 640x480-distinct-q8` or `--case 640x480-long-q8`, and use a fresh output
directory. Both intentionally exit nonzero after saving `report.json`. The
verification script reads the committed report files; copy fresh reports to a
separate copy of this directory to audit a new run.
