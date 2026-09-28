# RTX 4090 multimodal profiling — 2026-09-28

This is a stage-level baseline of the existing Transformers/PEFT worker, not an
optimized implementation or a speedup claim. Source commit:
`767e21172a303e36da4765fe59027db63f09acd1`; the remote checkout was clean. Model execution
is unchanged from PR #12's `ee970c2`. The 9-question upstream parity check was rerun
at `ee970c2` before the profiling harness was added. Trace collection was repeated
at `ebd7adc7e43d1c8ef2144cc0426295bf22abfa3b` after fixing CPU/GPU annotation counting;
model execution and the latency measurement function were not changed by that fix.

## Preserved failure and recovery

All 16 cases completed their latency samples before the first trace failed its
stage-count check: PyTorch emits separate CPU and GPU annotations under the same
name, and the original check added both. The original [baseline.json](baseline.json)
and [baseline.log](baseline.log) retain their failed status and error. Their
**1,600 unprofiled samples are complete**, independently checked for sample counts
and percentile calculations; they were collected before the profiler ran.

The fix retains event device types, counts CPU user annotations only, and supports
an explicit trace-only run. [traces.json](traces.json) records the successful new
trace run, its own clean source revision and environment. It contains no benchmark
samples. Its fixture and prepared-input fingerprints match the corresponding
baseline cases. The failed manifest has not been relabeled or overwritten.

## Environment and method

- RTX 4090, 24,564 MiB; driver 595.71.05, CUDA runtime 13.0.
- Linux container allocated 16 vCPUs and 120 GiB RAM. PyTorch intra/inter-op
  thread counts remain 64/64, matching the earlier environment; no thread tuning.
- Python 3.12.13, unmerged BF16 base with PEFT fp32 LoRA branches;
  all 178 expected adapted modules loaded. Full-logit scoring and PyTorch
  Gated DeltaNet/causal-convolution fallbacks retained. Packages: [packages.txt](packages.txt).
- Two PNG sizes × two instruction lengths × 1/2/4/8 serial questions = 16 cases.
  Instructions and candidates repeat within a case. Long instructions repeat a
  fixed sentence 64 times. These are controlled synthetic workloads, not GUI accuracy tests.
- Five warmup requests per case, then two runs of 50 synchronized requests:
  **1,600 measured requests / 6,000 question forwards**. All latency measurements
  finish before any profiling. Raw per-case warmup and samples are in [baseline.json](baseline.json).
- Timing covers `engine.predict` after parsing/decoding; excludes HTTP, queueing,
  image decoding, input fingerprinting, model load and profiling. The load
  observation (19.65 s) includes artifact verification and is not
  a cold-start comparison. GPU peaks below are for unprofiled execution.

See [experiment instructions](../../profiling.md) for the complete hypothesis,
controls, setup, command sequence and measurement limitations. Exact measured command:

```sh
PYTHONPATH=src /root/autodl-tmp/system1-omni-work/venv/bin/python \
  recipe/cua_s1/profile_multimodal.py \
  --weights /root/autodl-tmp/system1-omni-work/weights \
  --output /root/autodl-tmp/system1-omni-work/experiments/20260928-profile/measured \
  --profile 640x480-short-q1 --profile 640x480-short-q8 \
  --warmup 5 --runs 2 --iterations 50
```

Trace-only recovery command, from the corrected source revision:

```sh
PYTHONPATH=src /root/autodl-tmp/system1-omni-work/venv/bin/python \
  recipe/cua_s1/profile_multimodal.py \
  --weights /root/autodl-tmp/system1-omni-work/weights \
  --output /root/autodl-tmp/system1-omni-work/experiments/20260928-profile/traces \
  --trace-only --profile 640x480-short-q1 --profile 640x480-short-q8
```

## Correctness

- [reference.json](reference.json) and [candidate.json](candidate.json): all nine
  question forwards have identical input tensor hashes and fp32 candidate
  probabilities; maximum absolute difference 0. Both processes used the pinned
  upstream source hash, model revisions and identical recorded environments.
- Two instrumented requests (one and eight questions) produce exactly the same
  responses as ordinary execution. All seven stage ranges occur exactly once per
  question. This tests instrumentation, not an independent model implementation.
- No numerical approximation, adapter merge, batching, caching or kernel change
  was made. All 87 CPU tests passed on local macOS and the Linux GPU host; Ruff
  checks and formatting passed locally. The added profiling suite contains 25 tests.

## Unprofiled request latency

Values separated by `/` are run 1 / run 2. Tokens are per question; all questions
within a case share the same prompt. p95 uses nearest rank over 50 samples.

| Case | Tokens/question | p50 ms | p95 ms | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| 320x240-short-q1 | 236 | 109.24 / 108.96 | 111.30 / 111.17 | 8.73 |
| 320x240-short-q2 | 236 | 218.49 / 218.23 | 225.28 / 229.10 | 8.73 |
| 320x240-short-q4 | 236 | 433.38 / 456.14 | 448.92 / 500.47 | 8.73 |
| 320x240-short-q8 | 236 | 905.78 / 899.87 | 984.60 / 923.22 | 8.74 |
| 320x240-long-q1 | 614 | 135.50 / 141.88 | 143.17 / 168.55 | 8.92 |
| 320x240-long-q2 | 614 | 285.31 / 283.31 | 345.52 / 348.03 | 8.93 |
| 320x240-long-q4 | 614 | 555.44 / 543.43 | 625.50 / 565.05 | 8.93 |
| 320x240-long-q8 | 614 | 1152.11 / 1116.13 | 1226.11 / 1350.77 | 8.94 |
| 640x480-short-q1 | 456 | 128.18 / 127.74 | 152.02 / 134.22 | 8.84 |
| 640x480-short-q2 | 456 | 256.98 / 255.78 | 271.74 / 272.01 | 8.85 |
| 640x480-short-q4 | 456 | 536.72 / 520.11 | 551.70 / 693.52 | 8.86 |
| 640x480-short-q8 | 456 | 1099.52 / 1106.41 | 1215.46 / 1253.40 | 8.89 |
| 640x480-long-q1 | 834 | 154.87 / 155.64 | 158.91 / 192.66 | 9.04 |
| 640x480-long-q2 | 834 | 310.97 / 310.65 | 368.06 / 326.56 | 9.04 |
| 640x480-long-q4 | 834 | 696.27 / 698.81 | 768.80 / 811.10 | 9.06 |
| 640x480-long-q8 | 834 | 1362.37 / 1393.84 | 1398.75 / 1503.62 | 9.09 |

## Instrumented stage attribution

Each row is an inclusive total across one profiled request. Parent ranges overlap
with child ranges; do not add these rows or combine them with operator/kernel
totals. Only CPU user-annotation rows are shown; same-name GPU annotations are
retained separately in the raw report and are not added again. CPU range time
includes host work and possible synchronization. PyTorch's raw inclusive device
aggregates can include synthetic annotation accounting; they are retained as raw
data but are not presented as stage GPU execution times. Individual device kernel
events and their shapes remain available for GPU investigation. These traces are excluded from the latency
table and do not establish production latency by themselves.

| Case | Range | Calls | Inclusive CPU ms |
| --- | --- | ---: | ---: |
| 640x480-short-q1 | cua.prepare | 1 | 8.93 |
| 640x480-short-q1 | cua.transfer | 1 | 1.49 |
| 640x480-short-q1 | cua.forward | 1 | 222.74 |
| 640x480-short-q1 | cua.vision | 1 | 37.40 |
| 640x480-short-q1 | cua.language | 1 | 182.43 |
| 640x480-short-q1 | cua.output_projection | 1 | 0.12 |
| 640x480-short-q1 | cua.readout | 1 | 4.26 |
| 640x480-short-q8 | cua.prepare | 8 | 57.37 |
| 640x480-short-q8 | cua.transfer | 8 | 11.49 |
| 640x480-short-q8 | cua.forward | 8 | 1786.88 |
| 640x480-short-q8 | cua.vision | 8 | 291.37 |
| 640x480-short-q8 | cua.language | 8 | 1473.94 |
| 640x480-short-q8 | cua.output_projection | 8 | 0.71 |
| 640x480-short-q8 | cua.readout | 8 | 37.56 |

Module paths, full operator shape groups, trace byte counts and SHA-256 checksums
are preserved in `traces.json`. Large Chrome traces remain outside Git, archived
alongside the experiment on the GPU host and copied locally before task completion.
Re-run the documented command to regenerate them; hashes identify the original
trace files, whose runtime timestamps are not deterministic.

Do not rank GPU bottlenecks by CPU launch spans. In particular, the output-head
CPU span is short while its device work can continue afterward. Filtering raw
operator groups to `DeviceType.CUDA` and `is_user_annotation == false` gives
individual device kernel groups without adding CPU parents or synthetic ranges.
For example, the following GEMM groups appear in the instrumented traces:

| Kernel group | 1-question calls / total ms | 8-question calls / total ms |
| --- | ---: | ---: |
| `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` | 96 / 16.38 | 768 / 136.14 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_tn` | 56 / 6.54 | 448 / 52.77 |
| `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn` | 1 / 4.08 | 8 / 35.72 |

These are instrumented kernel durations, not end-to-end latency. Kernel names
alone do not identify the owning model module; use trace correlations and shapes
before choosing a replacement kernel.

## Next experiment

The eight-question trace contains eight preprocessing calls and eight vision
forwards for the same image. Request-local image preprocessing and vision-feature
reuse are therefore concrete candidates for the next PR. The current data does
not measure the speedup from removing those calls.

Start with distinct questions sharing one image, verify unchanged processor
tensors and probabilities, then run paired baseline/candidate measurements on
the same matrix. Coordinate language-layer and output-head optimizations with
the text-engine contributor. The 456-token, one-question fixture has the same
input tensor fingerprints as the original baseline; its new p50 values are
128.18 / 127.74 ms, compared with the historical 126.39 / 128.28 ms. This is a
baseline reproduction, not an optimization comparison.

## Scope and limitations

Results apply to this GPU, environment, synthetic input matrix and concurrency 1.
The experiment does not establish maximum supported image sizes, real task quality,
HTTP latency, p99 behavior, concurrent throughput, native CUDA or Metal support.
Reserved allocator memory can carry over between cases; allocated peaks are the
primary memory comparison. Cases execute in a fixed order, so clock/cache/order
effects have not been randomized away. Follow-up optimization claims require paired
baseline/candidate runs under the same controls and expanded reuse correctness cases.
