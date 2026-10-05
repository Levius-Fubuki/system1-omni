---
title: "Optimizing Open-Jev in System1-Omni: Rust/CUDA, Graph Replay, and Gated DeltaNet"
date: 2026-10-05
author: "Hongsheng Liu"
summary: "Open-Jev optimization on H200: a matched 7.47× gain over raw HF Transformers, CUDA Graph cache behavior, GDN kernel improvements, and the numerical constraints behind the results."
tags:
  - performance
  - cuda
  - open-jev
---

# Optimizing Open-Jev in System1-Omni

Open-Jev-27B-v1.1 makes decisions with a prefill pass and a learned scoring
head. On our 74-request H200 workload, System1-Omni's native Rust/CUDA worker
reduced mean warm HTTP latency from **362.21 ms with raw HF Transformers to
48.50 ms**, a **7.47× speedup**. A separate native CUDA Graph experiment reached
**47.11 ms** after warming the workload's token lengths. Optimizing Gated
DeltaNet preparation then reduced longer standalone GDN calls by about **13%**,
while its separate HTTP comparison improved only **0.42%**.

Those results describe different experiments. They are not a cumulative
optimization ladder: **the merged GDN change with CUDA Graph replay enabled has
not been timed together**. This post records measurements from October 2–3 and
integration validation from October 4, 2026. We inspected main at
[`e3cb4a2`](https://github.com/ThinkFlowLab/system1-omni/tree/e3cb4a215d76e5df929f60e71552531408f95c72)
on October 5; later source changes do not turn historical timings into a new
benchmark of current main.

## Establishing the HTTP baseline

We compared three complete serving backends on one H200, using the same Rust
frontend, prepared model inputs and request order. The workload contains 74 real
JevBench `noul` requests with one candidate each, 80–3,399 tokens, BF16 and
concurrency 1. Each backend reused one server for an excluded feasibility pass
and two measured passes of 74 requests.

| Backend | Mean HTTP latency | Pass 1 / pass 2 |
| --- | ---: | ---: |
| Raw HF Transformers | 362.209 ms | 362.238 / 362.180 ms |
| Native Rust/CUDA, eager | 48.503 ms | 48.471 / 48.535 ms |
| Original OpenJev-Fast | 50.936 ms | 51.097 / 50.775 ms |

![Matched H200 comparison of raw HF Transformers, native Rust/CUDA and OpenJev-Fast warm HTTP latency](../assets/blog/open-jev-20261005/backend-http.svg)

*Figure 1. The matched October 3 full-backend comparison. Bars show reported
aggregate means; black dots show the two pass means. The plot includes the
localhost frontend, tokenization, worker execution and UTF8 response decoding.
Preparation, readiness, first inference and feasibility are excluded. Values
come from the [historical validation summary](https://github.com/ThinkFlowLab/system1-omni/blob/e3cb4a215d76e5df929f60e71552531408f95c72/recipe/open_jev/validation.md#raw-hf-transformers-comparison-2026-10-03).
The two dots are observed variation, not confidence intervals.*

The raw HF arm used the original server with unmerged PEFT LoRA modules,
stock SDPA and the PyTorch fallback for linear attention and convolution.
Optional FLA and causal-convolution dispatch was explicitly disabled. Native
used a merged adapter and the repository's CUDA kernels; Fast retained its
custom kernels and graph stack. This measures the entire backend change,
including adapter execution, rather than isolating the programming language.

Native and HF agreed on all 74 thresholded decisions and both scored 64/74,
but their probabilities were not identical: the maximum absolute difference
was 0.020423. Fast scored 63/74 with one different decision. Native's mean was
4.78% below Fast in this run, while Fast had a lower median. This subset does
not establish general accuracy superiority or cover multi-candidate prefix
sharing. The [validation record](../../recipe/open_jev/validation.md) retains
the percentile timings, numerical differences and exact baseline setup.

The [OpenJev-Fast write-up](https://yiqilyu.me/open-jev-fast/) motivated this
investigation. Its 17.3 ms example uses B300, pretokenized prompts and a
forward-plus-head timer; its 231-task HTTP result is 42.0 ms. Neither is the
74-request H200 HTTP measurement plotted here.

## The native request path

The current worker separates prompt preparation from model execution. Rust
validates and tokenizes each candidate; the shared Qwen3.5/3.8 executor runs
one prefill forward per candidate. The final hidden state is downloaded for
the learned CPU scalar head, then the processor calibrates and reconstructs
the complete question's answer. There is no autoregressive decode loop.
See the [model contract](../../src/models/open_jev/README.md),
[native recipe](../../recipe/open_jev/native.md) and
[architecture contracts](../architecture.md).

The backend reused the native prefill implementation from
[PR #19](https://github.com/ThinkFlowLab/system1-omni/pull/19), with support and
shared execution integrated through
[PR #55](https://github.com/ThinkFlowLab/system1-omni/pull/55). Several changes
removed work while preserving the existing BF16 rounding points:

- Attention applies the sigmoid gate in its epilogue, removing a separate
  launch and output read/write pass for each full-attention layer.
- Residual RMSNorm retains thread values in registers at the model widths.
- MLP SiLU uses 16-byte BF16 loads and stores for compatible layouts, with
  scalar paths for other widths, strides and alignments.

An earlier, separate packed-SiLU experiment reduced mean HTTP latency from
49.527 to 48.297 ms, or 2.483%, with unchanged native outputs. Its long-request
SiLU timeline total dropped from 21.217 to 6.380 ms across 64 launches. Those
profiled kernel totals are not HTTP latency; the October 2 HTTP run also had
a different instrumentation context from the unprofiled October 3 campaigns.
The [packed-SiLU record](https://github.com/ThinkFlowLab/system1-omni/blob/e3cb4a215d76e5df929f60e71552531408f95c72/recipe/open_jev/validation.md#packed-silu-ab-2026-10-02)
documents that distinction.

## CUDA Graphs: capture helps when the cache retains the workload

The shared native executor supports opt-in graph replay with `CUA_S1_GRAPH=1`.
For a new exact token length, it runs an eager forward to initialize GEMM plans,
captures the model forward, instantiates the graph and replays it. Subsequent
requests of that length update the input IDs and reuse the graph. Growing the
scratch allocation invalidates the graphs; capture failure disables graph mode
and falls back to eager execution. The
[implementation](https://github.com/ThinkFlowLab/system1-omni/blob/e3cb4a215d76e5df929f60e71552531408f95c72/src/models/qwen3_5/native/src/model.rs#L635)
keeps tokenization, transfers and CPU scoring outside capture.

An eight-entry cache worked for a repeated short request but regressed the
mixed workload: its 57 distinct lengths caused eviction and recapture. Retaining
up to 64 graphs changed the result.

| Graph configuration | Mixed 74-request mean | Mixed pass 1 / 2 | Fixed 107-token mean |
| --- | ---: | ---: | ---: |
| Eager | 48.086 ms | 48.062 / 48.110 ms | 19.829 ms |
| Eight entries | 95.289 ms | 95.425 / 95.153 ms | 18.882 ms |
| 64 entries | 47.112 ms | 47.050 / 47.174 ms | 19.081 ms |

![Separate mixed-length and fixed-short CUDA Graph experiments showing the eight-entry mixed-workload regression and 64-entry warm replay improvement](../assets/blog/open-jev-20261005/graph-cache.svg)

*Figure 2. The October 3 native graph-cache experiment. The mixed workload has
74 requests per pass; the fixed-short workload has 32. Both use two measured
passes after feasibility, with maximum scratch length warmed first. Panels use
different scales, each starting at zero. Values are from the public
[graph validation summary](https://github.com/ThinkFlowLab/system1-omni/blob/e3cb4a215d76e5df929f60e71552531408f95c72/recipe/open_jev/validation.md#native-cuda-graph-replay-2026-10-03),
not a new run with optimized GDN.*

The 64-entry configuration reduced the mixed mean by 2.03% and the fixed-short
mean by 3.77%. All 74 probabilities and decisions remained exactly unchanged.
The mixed gain only narrowly exceeded the declared 2% gate. Cold capture remains
costly: the excluded mixed feasibility means were 50.130, 97.694 and 110.255 ms
for eager, eight-entry and 64-entry configurations respectively. These single
observations are not repeated cold-latency benchmarks. Graph mode remains opt-in;
more than 64 distinct lengths can still trigger eviction.

### Reading the GPU gaps

A preceding Nsight Systems trace verified one graph launch, no recapture and
no individual runtime kernel-launch calls per warm short request, while keeping
the same 834 GPU kernels. Graph replay changes submission; it does not fuse all
those kernels into one. Model transfers and CPU work outside capture also remain.

Node-level graph tracing showed larger gaps even though unprofiled HTTP latency
improved. [NVIDIA documents additional overhead from tracing individual graph
nodes](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-graph-trace).
We therefore use unprofiled HTTP passes for the speedup and traces to verify the
execution path. Hardware counters were unavailable, so these results do not
identify a particular occupancy, stall or bandwidth bottleneck.

## Gated DeltaNet: reduce preparation without changing its operands

The native GDN call has three stages: preparation, recurrent state propagation
and output. Preparation normalizes Q/K, builds cumulative decays and pair
products, computes a triangular inverse, and produces U/W intermediates.
The original implementation stored normalized Q/K in FP32 shared memory,
reconverted loaded tensor-core fragments to TF32, and staged U/W through FP32
shared memory before BF16 output.

[PR #68](https://github.com/ThinkFlowLab/system1-omni/pull/68) changes this
[preparation kernel](https://github.com/ThinkFlowLab/system1-omni/blob/33df9d75a575833b04bd3292e9b6ff32c7f6b760/src/backends/cuda/qwen3_5/gdn_prefill.cu):

| Preparation work | Original | Packed implementation |
| --- | --- | --- |
| Converted Q/K storage | FP32 shared arrays | TF32 converted once; meaningful bits in three-byte planes |
| TF32 pair products | WMMA | Explicit four-term `mma.m16n8k4`, matching measured baseline accumulation |
| U/W output | FP32 shared staging, then BF16 conversion | Direct packed BF16 stores |
| Dynamic shared memory per block | 92 KiB | 72 KiB |

Packing preserves the converted TF32 operands and exponent range. Decayed Q/K
still use the original FP32 normalization values before BF16 rounding. Gate
loads are prefetched in groups of eight while retaining the sequential FP32 sum.
The inverse stays FP32, and the workspace, C ABI, state and output contracts
remain unchanged. The lower shared-memory allocation is an implementation fact;
we did not measure occupancy to assign a hardware-counter explanation to the gain.

### Kernel gains and the HTTP result

The standalone comparison uses seeded synthetic tensors at representative token
lengths, with batch 1, 16 Q/K heads, 48 value heads and head dimension 128. It
times the complete preparation/state/output call, including eager host submission.
Each shape/arm has one excluded feasibility run, ten warmups and two measured
passes of 100 calls; pass 2 reverses arm order.

| Tokens | Baseline pass 1 / 2 | Packed pass 1 / 2 | Reduction pass 1 / 2 |
| --- | --- | --- | --- |
| 107 | 0.051291 / 0.050976 ms | 0.050914 / 0.050957 ms | 0.74% / 0.04% |
| 936 | 0.247565 / 0.247337 ms | 0.214176 / 0.214178 ms | 13.49% / 13.41% |
| 3,399 | 0.800775 / 0.798935 ms | 0.696768 / 0.697262 ms | 12.99% / 12.73% |

![Three complete GDN call comparisons and a separate warm HTTP comparison, showing longer-call reductions of about 13% but only a 0.42% HTTP improvement](../assets/blog/open-jev-20261005/gdn-kernel-http.svg)

*Figure 3. Two separate October 3 A/B experiments. The three kernel panels use
synthetic inputs and 100 calls per measured pass; the HTTP panel uses 74 real
requests per pass. Bars are recomputed from the
[raw kernel timings](https://github.com/ThinkFlowLab/system1-omni/blob/33df9d75a575833b04bd3292e9b6ff32c7f6b760/benchmarks/gdn/artifacts/20261003/kernel/analysis/results.jsonl)
and [raw HTTP records](https://github.com/ThinkFlowLab/system1-omni/tree/33df9d75a575833b04bd3292e9b6ff32c7f6b760/benchmarks/gdn/artifacts/20261003/e2e).
Black dots are the two pass means. Each panel starts at zero and has its own
scale; CUDA Graph is disabled.*

The declared kernel gate was at least 10% improvement at both longer shapes in
each pass, without a short-input regression. It passed. Short-input differences
are too small to support a meaningful speedup claim.

The complete-model comparison reused the same frozen worker/frontend and
prepared checkpoint for both arms, changing only GDN preparation. Baseline HTTP
means were 48.375 / 48.441 ms; packed means were 48.105 / 48.305 ms. The aggregate
changed **48.408 → 48.205 ms**, or **0.42%**, missing the declared 2% gate.
All 74 probabilities, decisions and token counts matched the pinned native
reference. Other model operations and host work remain in the HTTP path, and
synthetic operator timings cannot determine their contribution. The measured
whole-request result supports a targeted kernel improvement, not a material
end-to-end win. The [GDN report](../../benchmarks/gdn/README.md) preserves both
results and their separate controls.

### The numerical constraints that shaped the implementation

Several faster-looking variants failed before promotion. BF16 pair products
and one FP16 rounding variant changed a decision near 0.5; another FP16 variant
exceeded the 0.01 probability-drift limit. Using eight-term TF32 MMA changed
intermediate bit patterns even with the same converted operands. Restoring
four-term accumulation resolved the sampled differences. A direct-store variant
with FP32 staging preserved outputs but its 7–8% kernel gain missed the gate.
The [rejected-variant records](https://github.com/ThinkFlowLab/system1-omni/blob/33df9d75a575833b04bd3292e9b6ff32c7f6b760/benchmarks/gdn/artifacts/20261003/rejected-variants.json)
keep these failures visible; tolerances and measured run budgets were not relaxed.

For the accepted candidate, all initialized sampled U/W/QD/KD/PB/decay elements
and final outputs matched the baseline bitwise. The float64 recurrent reference
kept its existing tolerance. After merging the shared engine changes, six GPU
tests passed, including 20 GDN configurations covering boundaries, grouped heads,
long inputs, weak decay and tiny/zero Q/K. Newly built worker/frontend binaries
also passed first inference and a complete 74-request fidelity pass with exact
native probabilities, decisions and usage. The
[October 4 integration record](https://github.com/ThinkFlowLab/system1-omni/blob/33df9d75a575833b04bd3292e9b6ff32c7f6b760/benchmarks/gdn/artifacts/20261004/integration.json)
is a correctness check, not another latency comparison. Compilation passed for
SM80, SM89 and SM90; modified-kernel execution is verified on SM90 only.

## What remains to measure

The next direct comparison is merged GDN preparation with graph replay disabled
and enabled, at a fixed current revision and with a declared run budget. It
needs both warm replay and visible capture costs for newly encountered lengths.
Adding the independently measured 2.03% graph and 0.42% GDN gains would invent
a result we do not have.

We also examined vLLM's ready GDN backends. The recorded dispatcher revision
selects FlashInfer on SM90; its fused prefill path helped the longest standalone
shape but was slower on the two shorter complete calls. That experiment used
installed versions distinct from the researched upstream revisions and did
not validate a full-model Rust integration. The
[ready-kernel comparison](../../benchmarks/gdn/README.md#ready-kernels-in-vllm)
is a useful starting point for a targeted adapter, not a measured replacement
for the accepted native path. Multi-candidate prefix sharing remains outside
this benchmark subset.

## Evidence and figure regeneration

All figures use the corrected H200 label. Archived driver output names the same
SM90 device `NVIDIA L20X`; the fixed device UUID, NUMA 0 / CPUs 0–15 affinity,
model revisions and timer boundaries are recorded in the
[figure inputs](../assets/blog/open-jev-20261005/source-data.json).
HF and graph figures use the published historical summaries, rounded to
0.001 ms; their complete raw campaigns remain in local archives outside those
PRs. GDN figures are derived from the raw records committed with PR #68.
The model is Open-Jev-27B-v1.1, with base revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, checkpoint revision
`28cf73067d5b337860bbef3c85b8b82ba8730956`, and JevBench revision
`f8ce71361165846101d02ebc83ad44e47ae44fc3`.

From the repository root, use an existing Python environment with Matplotlib to
regenerate the SVG and PNG files without loading a model or initializing CUDA:

```sh
python docs/assets/blog/open-jev-20261005/plot.py
mkdocs build --strict
```

The [plot script](../assets/blog/open-jev-20261005/plot.py) reads only the sourced
JSON values. Its assets are original plots of historical measurements. For
actual inference setup and scheduler requirements, follow the
[native recipe](../../recipe/open_jev/native.md); chart regeneration does not
run an inference benchmark.

This presentation follows the useful combination of serving results and concrete
implementation explanations in the
[Qwen3-Omni](https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization) and
[Kimi K3 optimization](https://vllm.ai/blog/2026-09-13-kimi-k3-performance-optimization)
posts. Their measurements are separate from this project's evidence.
