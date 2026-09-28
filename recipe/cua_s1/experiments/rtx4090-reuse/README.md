# RTX 4090 request-local image reuse — 2026-09-28

The optimized worker preprocesses and encodes a shared screenshot once per
multi-question request, retaining independent text processing, three-dimensional
positions, language forwards and candidate readouts. Single-question execution
uses the original path.

Across the 13 multi-question cases (two runs each), measured p50 latency fell
**6.8–19.9%**. For **eight distinct questions on a 640×480 screenshot**, p50 fell
from **1100.03 / 1094.87 ms to 887.54 / 888.45 ms** (19.3% / 18.9% lower latency).
These are end-to-end `engine.predict` measurements, not extrapolations from
profiler stage spans. Single-question p50 differences ranged from 0.5% slower to
1.1% faster across identical execution paths and are treated as measurement noise.

## Provenance and method

- Timed source: clean `c18a21b205d7a66449de845a8598b6e27d82424c`; inference implementation
  commit `cf899ff`. Later changes add tests, documentation and result-verification
  scripts without changing the measured model or benchmark paths.
- Final GPU-host tests, fresh upstream oracle and HTTP/layout validation use
  clean source `9f4a4fe`.
- RTX 4090 24,564 MiB; driver 595.71.05, CUDA runtime 13.0, Python 3.12.13,
  PyTorch 2.14.0, Transformers 5.17.0, PEFT 0.21.0. Full package/version and model
  provenance are in [paired.json](paired.json). Environment matches the original
  pinned upstream baseline.
- Unmerged BF16 model with fp32 LoRA branches; all 178 adapted modules present,
  including the vision modules. PyTorch Gated DeltaNet/causal-convolution fallback
  implementations and 64/64 PyTorch thread settings are unchanged.
- Original 16-case matrix plus one eight-distinct-question case. Five warmup
  requests per variant per case, then two runs of 50 samples per variant:
  **3,400 timed requests / 13,600 language forwards**. Cases use recorded shuffled
  order; baseline/reuse ordering alternates each sample and reverses per run.
- Timing synchronizes CUDA around each whole prediction, excluding HTTP, parsing,
  image decoding, fingerprinting, hooks and model load. Reset peak memory before
  each variant invocation. No profiler or counting hooks are installed during
  latency sampling. Reserved memory can carry across variants; compare allocated
  memory. Largest allocated peak: 9.083 GiB baseline, 9.035 GiB reuse.

## Correctness and service validation

- **13 correctness fixtures / 49 question forwards per path**, including eight
  different questions, PNG/JPEG, different image/grid sizes, reversed question
  order, 1–26 candidates, candidate reordering, Unicode/structured labels and
  consecutive image changes. The same validation also runs on all 17 timed cases.
- Prepared tensor fingerprints, language input embeddings, 3D positions,
  attention masks, candidate probabilities and complete responses are **exactly
  equal** between original and optimized paths. No tolerance or approximation is
  used. Actual multi-question image-preprocessing and vision counts fall from N
  to 1; language counts remain N. The raw report records these counts and hashes.
- Fresh pinned upstream [reference.json](reference.json) and worker
  [candidate.json](candidate.json) revalidate all nine upstream forwards. The
  independent result audit also compares the optimized report with that oracle.
- Final GPU-host CPU suite: **104 passed**. Local macOS: **103 passed, 1 skipped**
  (torch unavailable locally). The skipped tensor test runs on the GPU host.
  Tests cover preflight token rejection, encoder/later-question failures, feature
  lifetime and successful recovery with another image. Lint and formatting pass.
- [postflight.json](postflight.json) records actual tensor shape, dtype, strides,
  contiguity and ownership-relevant metadata, plus HTTP health and eight-question
  response equality against direct engine execution.

## Paired latency and memory

Values separated by `/` are run 1 / run 2. p95 is nearest rank over 50 samples;
allocated memory is the maximum across both runs. Negative reduction means slower.

| Case | Baseline p50 ms | Reuse p50 ms | p50 reduction | Baseline p95 ms | Reuse p95 ms | Peak allocated GiB, baseline → reuse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 320x240-long-q1 | 135.56 / 135.63 | 136.22 / 135.66 | -0.5% / -0.0% | 143.72 / 142.83 | 145.02 / 140.97 | 8.926 → 8.926 |
| 320x240-long-q2 | 286.12 / 271.52 | 261.18 / 251.31 | 8.7% / 7.4% | 329.01 / 337.47 | 272.06 / 312.39 | 8.931 → 8.922 |
| 320x240-long-q4 | 566.04 / 540.73 | 503.31 / 478.02 | 11.1% / 11.6% | 579.48 / 562.61 | 538.99 / 500.64 | 8.931 → 8.922 |
| 320x240-long-q8 | 1159.29 / 1137.77 | 967.33 / 950.53 | 16.6% / 16.5% | 1232.38 / 1236.30 | 1086.98 / 994.59 | 8.939 → 8.922 |
| 320x240-short-q1 | 108.11 / 108.17 | 108.09 / 108.04 | 0.0% / 0.1% | 112.40 / 113.68 | 109.76 / 116.19 | 8.728 → 8.728 |
| 320x240-short-q2 | 228.86 / 230.37 | 205.76 / 207.33 | 10.1% / 10.0% | 242.46 / 244.51 | 213.63 / 216.08 | 8.728 → 8.726 |
| 320x240-short-q4 | 448.85 / 443.57 | 382.62 / 379.76 | 14.8% / 14.4% | 464.28 / 470.91 | 395.06 / 394.91 | 8.734 → 8.726 |
| 320x240-short-q8 | 914.48 / 889.32 | 748.10 / 724.66 | 18.2% / 18.5% | 989.35 / 962.43 | 771.86 / 762.28 | 8.740 → 8.726 |
| 640x480-distinct-q8 | 1100.03 / 1094.87 | 887.54 / 888.45 | 19.3% / 18.9% | 1171.23 / 1162.61 | 984.33 / 963.19 | 9.015 → 8.992 |
| 640x480-long-q1 | 158.14 / 155.74 | 156.43 / 155.89 | 1.1% / -0.1% | 192.92 / 193.58 | 193.23 / 189.91 | 9.035 → 9.035 |
| 640x480-long-q2 | 309.13 / 310.54 | 286.97 / 289.53 | 7.2% / 6.8% | 325.38 / 330.10 | 298.02 / 302.77 | 9.043 → 9.033 |
| 640x480-long-q4 | 629.40 / 630.98 | 561.65 / 566.86 | 10.8% / 10.2% | 714.52 / 671.00 | 622.21 / 594.51 | 9.057 → 9.033 |
| 640x480-long-q8 | 1319.17 / 1316.22 | 1113.74 / 1112.65 | 15.6% / 15.5% | 1474.11 / 1446.67 | 1196.36 / 1159.36 | 9.083 → 9.033 |
| 640x480-short-q1 | 128.17 / 126.42 | 128.00 / 126.42 | 0.1% / 0.0% | 142.45 / 131.61 | 133.58 / 130.29 | 8.842 → 8.842 |
| 640x480-short-q2 | 256.55 / 256.11 | 234.05 / 236.78 | 8.8% / 7.6% | 270.34 / 273.75 | 240.42 / 250.98 | 8.849 → 8.838 |
| 640x480-short-q4 | 532.07 / 520.37 | 461.20 / 449.92 | 13.3% / 13.5% | 570.69 / 553.80 | 478.79 / 472.87 | 8.862 → 8.838 |
| 640x480-short-q8 | 1092.57 / 1096.68 | 874.72 / 891.88 | 19.9% / 18.7% | 1317.02 / 1271.33 | 954.19 / 933.10 | 8.890 → 8.838 |

## Concrete feature interface

For the 640×480 distinct-question fixture, the processor produces contiguous
CPU fp32 pixels `[1200, 1536]`. Shared adapted features are contiguous CUDA BF16
`[300, 2560]`, stride `[2560, 1]`, 1,536,000 bytes, with gradients disabled.
The first question uses CUDA BF16 embeddings `[1, 443, 2560]` and CUDA int64
positions `[3, 1, 443]`. All eight question layouts are recorded in
[postflight.json](postflight.json); sequence lengths remain question-specific.
These are measurements of the pinned processor/model, not a general native-engine
ABI. See [ownership and execution contract](../../image-reuse.md).

## Reproduce and independently audit

Run from the repository root with the pinned environment and downloaded weights.
The measured command was:

```sh
PYTHONPATH=src /root/autodl-tmp/system1-omni-work/venv/bin/python \
  recipe/cua_s1/benchmark_image_reuse.py \
  --weights /root/autodl-tmp/system1-omni-work/weights \
  --output /root/autodl-tmp/system1-omni-work/experiments/20260928-reuse/paired \
  --warmup 5 --runs 2 --iterations 50
```

Use a fresh output directory for a new run. Regenerate the independent oracle
with both `--mode reference` and `--mode candidate` in the same output directory:

```sh
PYTHONPATH=src python recipe/cua_s1/evaluate_multimodal.py \
  --weights /path/to/weights --reference /path/to/pinned-cua-checkout \
  --output /path/to/fresh-oracle --mode reference
PYTHONPATH=src python recipe/cua_s1/evaluate_multimodal.py \
  --weights /path/to/weights --reference /path/to/pinned-cua-checkout \
  --output /path/to/fresh-oracle --mode candidate
PYTHONPATH=src:recipe/cua_s1 python \
  recipe/cua_s1/experiments/rtx4090-reuse/postflight.py \
  /path/to/system1-omni /path/to/weights /path/to/fresh-postflight
```

The saved dataset can be audited without torch, a GPU or model weights:

```sh
python recipe/cua_s1/experiments/rtx4090-reuse/verify_results.py
```

This checks exact coverage of the 17 cases, all 3,400 sample values, alternating
order, medians and nearest-rank p95 calculations, memory bounds, invocation
counts, parity flags and all nine independent upstream input/probability records.
[SHA256SUMS](SHA256SUMS) identifies the public data files; [paired.log](paired.log)
and [cpu-tests.log](cpu-tests.log) retain execution progress and final test output.

The complete raw outputs, generated fixtures, smoke run, oracle and HTTP logs
were also copied off the GPU host before shutdown in
`20260928-image-reuse-evidence.tar.gz`, SHA-256
`65b365ecd10c49036519ac37e8a50bc6746f6630e0c98f03068ab39c4dd8f868`.
The table does not depend on that external archive; its raw samples are in Git.

## Limits

This is concurrency-1, prefill-only execution on synthetic screenshots and the
pinned software/hardware. It establishes no GUI task-quality improvement, HTTP
latency/throughput gain, p99 result, maximum input-size coverage, native CUDA
kernel speedup or Metal support. The HTTP postflight is a correctness check only.
Image preprocessing and visual encoding are reused inside one request; text
processing and language execution remain per question. No persistent cache,
batching, adapter merge, compiler or CUDA Graph change was made.
