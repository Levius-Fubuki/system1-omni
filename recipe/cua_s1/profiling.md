# Multimodal profiling experiment

This experiment attributes the existing unmerged BF16 Transformers/PEFT worker's
cost before changing inference. It extends the [original baseline](https://github.com/Levius-Fubuki/system1-omni/blob/683d470669f19d5e9530cd8233727a0d4f1f8d29/recipe/cua_s1/experiments/README.md)
with image-size, text-length and same-image question-count sweeps. It does not
implement image reuse, native CUDA kernels, CUDA Graphs or batching.

## Hardware and environment

Recommended for comparison with the original measurements:

| Resource | Configuration |
| --- | --- |
| GPU | One exclusive RTX 4090, 24 GB; no competing GPU jobs |
| CPU | 8–16 allocated vCPUs; record actual thread settings |
| Host RAM | 32 GB minimum planning allowance; 64 GB preferred |
| Disk | 80–100 GB available SSD for a fresh environment, weights and traces |
| OS | Linux x86_64, Ubuntu 22.04 or 24.04 |
| Python | 3.12 |
| CUDA runtime | Match the original PyTorch `2.14.0+cu130` wheel where available |

These are experiment planning recommendations, not measured maximum-input
requirements. The original 456-token, one-question run peaked at 8.84 GiB of
allocated GPU memory. Longer inputs and profiler tensor retention may use more.
Start with one short case before the full matrix. Stop and report an OOM rather
than silently changing precision, truncating inputs or reducing question counts.
The eight questions run serially; they are not a batch of eight.

If reusing the previous environment and weights, substantially less free disk is
needed. Check the available space before collecting traces. A separate CUDA
Toolkit is not required for this PyTorch profiling experiment; native kernel
development is a subsequent task. CUDA 13 requires a compatible R580-or-newer
driver; the previous run used 595.71.05. See NVIDIA's
[compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

Use the [worker setup](README.md#setup) and pinned requirements. Restore the
previous environment when possible. Do not install flash-linear-attention,
causal-conv1d, merge LoRA or enable compilation in the baseline environment.
Record any environment difference and establish a new baseline before comparing
performance. GPU scheduling/allocation rules of the host still apply.

From the repository root, with its Python environment activated:

```sh
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
df -h .
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -m pip freeze > /tmp/cua-profile-packages.txt
PYTHONPATH=src python recipe/cua_s1/download_weights.py --dest weights
PYTHONPATH=src python -m pytest tests/cua_s1 -q
```

The downloader verifies pinned sizes and hashes. Existing valid weights can be
reused. Keep model weights outside version control and select the allocated GPU
with `CUDA_VISIBLE_DEVICES` when appropriate.

## Hypotheses and controls

1. Repeated questions about one image repeat image preprocessing and vision
   computation. Measure how much those stages contribute before implementing
   reuse; do not assume they dominate.
2. Longer instructions and larger processed image grids change the mix of vision,
   language and projection work. Record actual token counts and tensor shapes;
   the fixture names are input categories, not promised token lengths.
3. The trace instrumentation should preserve the response exactly. Comparing
   traced and ordinary responses is an instrumentation check, not an independent
   upstream parity result.

Keep the GPU, model revisions, package versions, BF16 dtype, unmerged adapter,
full-logit path, thread configuration and serial execution fixed. Correctness
against the pinned upstream implementation remains the separate
[`evaluate_multimodal.py` reference/candidate procedure](README.md#reproduce-correctness-and-profiling).
Run those modes before accepting a new environment or inference change. Do not
use old GPU reports as evidence for a newly changed path.

## Run

List the 16 cases without importing PyTorch or loading weights:

```sh
PYTHONPATH=src python recipe/cua_s1/profile_multimodal.py --list-cases
```

The matrix covers 320×240 and 640×480 PNGs, short/long instructions and 1/2/4/8
questions per request. Each request uses a single synthetic image with three
candidate actions per question. Questions within a case repeat the same
instruction and criteria under different question names, isolating question
count from prompt variation. The long instruction repeats a fixed sentence 64
times; it is a controlled workload, not a realistic GUI task evaluation. Images
contain no private user data. Follow-up reuse correctness tests must also cover
different questions sharing an image.

Start with a feasibility run, using a new output directory:

```sh
PYTHONPATH=src python recipe/cua_s1/profile_multimodal.py \
  --weights weights --output /tmp/cua-profile-smoke \
  --case 320x240-short-q1 --profile 320x240-short-q1 \
  --warmup 1 --runs 1 --iterations 2
```

This checks execution and trace generation; two samples are not a performance
claim. Next run the complete matrix, profiling only two representative cases:

```sh
PYTHONPATH=src python recipe/cua_s1/profile_multimodal.py \
  --weights weights --output /tmp/cua-profile-measured \
  --profile 640x480-short-q1 --profile 640x480-short-q8 \
  --warmup 5 --runs 2 --iterations 50
```

Use repeated `--case` arguments for a selected subset. Retain both successful
and failed-run manifests. A failed case must be investigated before claiming
coverage of the full matrix. Reserve detailed traces for selected cases: shape
recording adds overhead and memory pressure.

Completed cases are written before the next case begins. Measurements within a
case are saved after all its runs finish; an interruption during that case can
lose its samples. This script does not resume a partial run. Repeat the failed
case in a new output directory and retain the original failure record.

If latency sampling completed but trace collection needs to be repeated, use a
separate trace-only run after fixing the failure:

```sh
PYTHONPATH=src python recipe/cua_s1/profile_multimodal.py \
  --weights weights --output /tmp/cua-profile-traces \
  --trace-only --profile 640x480-short-q1 --profile 640x480-short-q8
```

Trace-only mode collects no benchmark samples and records its own source and
environment. Keep both reports; do not replace the original failed manifest or
present the new trace run as a rerun of latency measurements. If `--case` is also
specified, its set must match the `--profile` set.

## Interpretation and publication

The unprofiled benchmark times `engine.predict` after JSON parsing and image
decoding, with CUDA synchronization at the boundaries. It includes processor
work, transfers, forward passes, candidate readout and response construction.
It excludes HTTP, admission/queueing, input decoding, load, setup metadata and
profiler overhead. Startup and per-case warmup are separate observations.

Each case retains raw latency samples and reports p50/p95, serial request rate,
question count and GPU memory peaks. Serial question rate, if derived, is request
rate multiplied by question count; it is not independent concurrent throughput.
Do not infer p99 reliability or GUI task quality from these measurements.

Trace ranges and grouped operators describe instrumented execution. Inclusive
stage ranges overlap with child operations and kernels; do not sum them or use
their totals as fractions of the unprofiled request time. CPU range duration is
not necessarily GPU execution time. Module-name discovery must identify the
vision, language and output-head boundaries explicitly. Missing boundaries are
an experiment failure, not zero time. See the
[PyTorch profiler documentation](https://docs.pytorch.org/docs/stable/profiler)
for shape-recording overhead and trace semantics.

PyTorch can emit CPU and GPU annotations with the same range name. Operator
records retain their device type and annotation flag. Invocation counts use CPU
user annotations only; same-name GPU annotations are separate records and must
not be added again to the CPU range's attributed CUDA duration. Raw inclusive
device aggregates can themselves contain synthetic annotation accounting. Use
stage CPU spans for the host timeline and individual device kernel events in the
trace for GPU investigation; these raw parent aggregates are not disjoint kernel
time or a reliable basis for stage GPU percentages.

For a reviewable experiment PR, include:

- Commit/source provenance, environment, exact commands and fixture settings.
- New upstream reference/candidate parity results, or a clear statement that
  they were not rerun.
- Both warm runs' raw samples, numerical summaries, memory peaks and stage/module
  mappings; document all exclusions and failed cases.
- A bounded next optimization hypothesis supported by the profile.

Keep large Chrome traces outside Git, with checksums in the experiment record.
Publish no speedup until an optimized implementation has a paired comparison
against this baseline. Native interface design, image reuse and kernel changes
remain separate follow-ups.
