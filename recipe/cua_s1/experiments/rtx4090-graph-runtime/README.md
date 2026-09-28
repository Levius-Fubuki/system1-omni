# Segmented CUDA Graph runtime — RTX 4090, 2026-09-28

This experiment exercises the serving worker's optional `--graph` path on the
Transformers/PEFT Cua-S1 multimodal adapter. The worker captures runs of three
Gated DeltaNet decoder layers and executes each intervening full-attention layer
eagerly. Image preprocessing, the adapted vision encoder, token embedding, 3D
position construction, candidate scoring, and response assembly remain in the
normal request path.

## Why the graph is segmented

The whole-language-forward experiment in [#20](../rtx4090-graph/README.md)
rejected an eight-distinct-question case with a 0.065181 candidate-probability
difference and a long-instruction text-change case with a 0.051902 difference.
On the pinned RTX 4090 environment, the whole-graph replay is deterministic,
and an eager forward before and after capture is bitwise identical. Inspecting
every decoder layer for the 26-choice question found identical outputs through
layers 0–2. The first difference occurs in layer 3's full-attention output;
its input layer norm is identical. Differences then accumulate across later
layers. Forcing PyTorch's math SDPA backend makes its own eager and graph
forwards identical, but changes the 26-choice candidate probabilities by
0.045445 relative to the worker's normal attention backend. These observations
localize the problem to the full-attention calculation under capture; they do
not establish which CUDA kernel or rounding rule causes it.

Segmenting at full-attention layers keeps that calculation on the original
eager path. A manual layer loop using the pinned Qwen3.5 model was compared to
the original PEFT forward before graphing; logits were bitwise identical for
the four distinct-question shapes checked, including the 26-choice question.
Capturing only linear-attention runs retained bitwise-identical logits for
those inputs while holding 8–32 graph segments in memory across the four
shapes. The worker integration repeats the exact-logit comparison when each
layout is first captured. A failing layout falls back to eager.

## Runtime and measurement scope

The runtime keys entries by model identity, attention implementation, and the
shape, stride, dtype, and device of embeddings, 3D positions, and attention
mask. Each entry owns all eight contiguous linear-attention captures for one
layout. On replay it copies fresh hidden states and the recurrent mask into
static buffers. Full attention uses freshly generated rotary embeddings and
causal mask. The final norm and output projection run eagerly, producing a
fresh logits tensor before the entry can be replayed again. The worker's
existing request lock prevents concurrent inference.

The first use of a layout runs eagerly. At the second use, the runtime captures
and compares all vocabulary logits to the worker's eager forward before caching
the entry. The default cache retains up to eight layouts and 1 GiB of measured
live graph allocations. Unsupported layouts, oversized entries, capture errors
with a healthy CUDA context, and numerical mismatches use eager. The memory
limit covers retained allocations, not the temporary peak during capture or
PyTorch's reserved allocator pool.

The synchronized timing covers `engine.predict` from prepared request object to
response dict. It includes preprocessing, vision and language inference, and
response assembly. It excludes request parsing, HTTP, model loading, initial
capture, and concurrency. Each case alternates eager and Graph predictions.
Model identity, usage, choice, and candidate probabilities are compared for the
original image, a black replacement image of the same size, and a
same-token-length text change. The distinct-question case changes its first
question's text while preserving all eight layouts. Every recorded candidate
probability difference is **zero**.

## Clean-source result

The measured source was commit `de000560ce927a0db58f4928e91234d95df2ae23`
with a clean checkout. One RTX 4090 (24,564 MiB, sm_89) ran Python 3.12.13,
PyTorch 2.14.0+cu130, Transformers 5.17.0, and PEFT 0.21.0 with the pinned BF16
base and unmerged multimodal LoRA. Optional `causal_conv1d` and
`flash-linear-attention` packages were absent, matching #20's fallback path.
The five cases each have two alternating runs of 20 timed predictions per
variant: **400 timed predictions** total. Each case had three warmups per
variant before timing. The table gives synchronized whole-request p50
milliseconds; the raw report also contains every sample and p95.

| Case | Eager p50, runs 1 / 2 | Graph p50, runs 1 / 2 | Reduction | Capture cost | Retained graph allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 320×240, 2 repeated questions | 189.57 / 192.80 | 102.74 / 102.99 | 45.8% / 46.6% | 346 ms, 1 layout | 92.0 MiB |
| 320×240, 8 repeated questions | 702.32 / 701.88 | 349.60 / 349.04 | 50.2% / 50.3% | 329 ms, 1 layout | 92.0 MiB |
| 640×480, 8 repeated questions | 859.35 / 852.85 | 576.29 / 575.22 | 32.9% / 32.6% | 507 ms, 1 layout | 108.9 MiB |
| 640×480, 8 long-instruction questions | 1024.08 / 1026.02 | 971.69 / 971.54 | 5.1% / 5.3% | 618 ms, 1 layout | 130.3 MiB |
| 640×480, 8 distinct questions | 863.12 / 824.02 | 597.37 / 591.87 | 30.8% / 28.2% | 3493 ms, 7 layouts | 286.8 MiB |

The distinct-question case requires seven captures because two questions have
the same tensor layout. Its total capture cost is higher than the repeated
question cases. Capturing also runs one eager forward for an exact-logit gate;
neither this validation nor capture is included in the timed samples. These
measurements are serial and do not show HTTP throughput or concurrency.

The [raw report](report.json) records the repository SHA, fixtures, exact
comparison results, counters, allocation estimates, and all 400 samples.
[Cache checks](cache.json) exercised a one-layout LRU limit and a 1 MiB budget.
The first evicted three complete layouts over two requests while preserving
candidate probabilities. The second rejected both layouts, recorded the
budget fallback, kept zero cached layouts, and preserved candidate
probabilities. A real HTTP worker started with `--graph --graph-min-uses 1`,
returned a ready health response, and served two identical two-question
requests with byte-identical JSON bodies. On the clean checkout, all 116
`tests/cua_s1` tests passed; Ruff lint and format checks passed.

The [independent verifier](verify_results.py) recomputes p50/p95 from every
sample, checks the five correctness gates and cache limits, and requires a
clean measured revision. `SHA256SUMS` records the raw report hashes.

Reproduce from this branch's clean source and the pinned weights:

```sh
PYTHONPATH=src:recipe/cua_s1 python recipe/cua_s1/benchmark_graph_runtime.py \
  --weights /path/to/weights --output /path/to/fresh-output \
  --case 320x240-short-q2 --case 320x240-short-q8 \
  --case 640x480-short-q8 --case 640x480-long-q8 \
  --case 640x480-distinct-q8 --graph-max-tokens 4096 \
  --warmup 3 --runs 2 --iterations 20
```

The raw report records all samples, capture counters and cost, retained graph
allocations, package versions, fixture hashes, and repository revision. The
`--graph-max-tokens 4096` override exercises the long prompt in this
experiment; the worker's default is 2048 tokens, with longer prompts going
eager.
