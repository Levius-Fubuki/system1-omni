# JEMM A800 validation

The native Rust/CUDA worker passes the declared reference parity gates on the
fixed A800 corpus. The measurements below describe this tested environment and
workload.

## Frozen scope

The baseline System1-Omni commit is `4a79980d8a75fd063cb3f8247e06215e288b18ac`.
The official JEMM source is pinned to
`6822fe0fd53c5e6670af6ba99fb2c857a661e532`, its adapter to
`76e3c209e8441fa658221c7ba2725bad2f811176`, and Qwen/Qwen3.8-27B to
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

The host has one NVIDIA A800 80 GB PCIe, compute capability 8.0, driver
595.71.05 and CUDA toolkit 13.0.88. Its cgroup memory limit is 120 GiB;
the data disk is 170 GiB and system disk is 30 GiB. Reference dependencies and
launch commands are in [the native recipe](native.md).

The frozen corpus contains 13 requests and 16 questions: 11 text requests and
two synthetic deployment-console screenshots. Its SHA256 is
`5e729330b148aacec058abda8f5e0d81f9b9064a2e2bb8401426fd97e9815c4c`.
It covers 2/3/10/26/32 choices, 3/10/32 score levels, Noul, structured Unicode
state, tool JSON, literal special-token text and shared-image multiple questions.

Reference and native runs use exclusive GPU ownership in that order. Each
deployment receives three explicit warmup requests, one full feasibility pass
and exactly two measured repetitions at concurrency one. Latency is the client
HTTP round trip through the same Rust frontend, including complete response
JSON parsing. Downloads, compilation, model loading and warmup are excluded.
Sampled GPU resident allocation is reported separately from true peak memory.

Acceptance gates are fixed before execution: prompt text, token IDs, label IDs,
question/candidate order, image grids, modality markers, position IDs and input
token counts must match exactly. Maximum probability difference is 0.02 and
score expected-value difference is 0.1. Winners must agree when the reference
top-two margin is at least 0.05; all lower-margin cases remain in the report.
Failures are retained without relaxing these gates.

## Component checks

On the A800, the six original Qwen CUDA kernel regressions passed. Four new
vision checks passed: legacy versus v2 head-64 attention, head-72 attention
against float64, head-72 rotary order, and BF16 learned-position interpolation
at hidden size 1152. The selected-label CUDA projection regression passed.

For the pinned Cua-S1 4B checkpoint, the original and extracted shared vision
implementations produced identical BF16 output bytes and stage hashes at
256×256, 512×256 and 512×512. The shared Cua implementation also produced the
same results with the original ABI 5 CUDA library, whose new vision symbols
are absent. These checks protect existing Cua behavior; they do not establish
JEMM end-to-end parity or execution on other GPUs.

All 19 pinned JEMM base/adapter weight files passed SHA256 verification before
CPU export. The export completed and recorded checksums for 31 output files.
Export peak RAM was not sampled.

## End-to-end results

Both deployments completed three warmup requests, 13 feasibility requests and
26 measured requests, with every response HTTP 200. There are 32 measured
question comparisons. All winners agree; the smallest reference top-two margin
is 0.961086, so this corpus contains no low-margin cases. Maximum absolute
probability difference is **0.0015338802** and maximum Score expected-value
difference is **0.0008446101**, below the predeclared 0.02 and 0.1 gates.

All 16 preprocessing records match exactly: raw template text, ordered
question/candidate identities, label IDs, complete token IDs, grids, modality
markers and three-axis positions. The four image/question pixel comparisons
match exactly in FP32. Raw template text has one image placeholder per image;
native `chat` diagnostics additionally retain the expanded tokenizer input.
The initial comparison used these two different stages for its prompt field;
that failed report is preserved, and a diagnostic-only change now exposes
`template_chat` separately. Actual token IDs and pixels matched throughout.

| Warm HTTP round trip | Samples per deployment | Official PyTorch/PEFT median (range) | System1-Omni median (range) | Ratio of medians |
| --- | ---: | ---: | ---: | ---: |
| Text | 22 | 204.284 ms (196.686–369.751) | 85.171 ms (51.354–137.698) | 2.399× |
| Images | 4 | 405.712 ms (403.692–412.761) | 181.143 ms (174.235–188.712) | 2.240× |

These samples pool different request shapes from the same ordered corpus;
ratios are ratios of pooled medians. They are not production throughput or
statistical generalization claims. Screenshot requests contain two questions.
Reference execution recomputes vision per question, following upstream;
native execution reuses that request's vision embeddings. This is an intentional
implementation difference, alongside language LoRA merging and selected-head
projection.

The reference is the unmodified official `DecisionModel.from_pretrained` and
HTTP handler, with BF16 base parameters and 992 FP32 LoRA tensors, unmerged.
SDPA is configured and flash-linear-attention is installed. The recorded
Accelerate wrapper does not independently establish individual FLA kernel
dispatch. Optional `causal_conv1d` is absent from upstream's declared dependencies
and this environment; its PyTorch convolution fallback warning is preserved.
This is the tested official dependency environment, not a claim of comparison
against every optimized PyTorch configuration. Native language LoRA is merged
into BF16 and CUDA Graphs are disabled (`CUA_S1_GRAPH=0`).

Sampled resident GPU maxima are 53,883 MiB for the reference and 50,751 MiB
for native. `nvidia-smi` sampling ran every 0.2 seconds; reference sampling began
after loading, native sampling before loading. These are observed resident
allocations, not true allocator peaks. Reference checkpoint verification took
55.130 seconds and loading took 17.972 seconds. Its first explicit HTTP warmup
took 35.027 seconds, including first-forward compilation. Native performs a
real text warmup before listening; its startup phases were not individually
instrumented. None of these setup intervals enters the latency table.

The native JSONL diagnostic completed all 13 requests with 16 raw logit rows
and passing response parity. Twenty-three live HTTP probes passed, including
invalid requests, body/media limits, direct versus frontend parity, two
concurrent image requests and readiness during those in-flight requests.
The readiness probe took 0.350 ms. CUDA failure retirement is covered by CPU
ownership/cleanup regressions; a live device failure was not injected.

## Reproduction and provenance

Follow [the native recipe](native.md) for pins and launch commands. The corpus,
protocol, strict comparator and benchmark are maintained under `tests/jemm/`
and `recipe/jemm/`. The raw archive includes source and binary SHA256 inventories,
complete request/response JSONL, environment, export manifest, reference input
tensors, GPU samples, component stage hashes, HTTP probes and all failed setup
attempts. It contains no model weights or signed download URLs.

[Download the immutable A800 evidence](https://github.com/Levius-Fubuki/system1-omni/releases/tag/jemm-a800-20261008).

The completed export was produced by exporter SHA256
`8d27d3246bccb7f383f6d477d6dcdaf2b6751f9f97bc99bfc42c64c696c59793`;
its exact script is preserved. Later exporter changes add pinned-label checking
and strict producer/runtime identity for resumption, without changing numerical
merge code. The final exporter does not re-certify that completed directory.
The worker verifies its original manifest and every output checksum.

Final local checks pass: workspace formatting, Clippy with warnings denied,
127 ordinary Rust tests, release workspace build, 58 Python recipe tests and
strict MkDocs. Twenty Rust tests are opt-in and ignored by the ordinary suite;
applicable A800 kernel, head and Cua vision tests were run separately above,
and the pinned tokenizer test was run explicitly. Metal and other GPU opt-ins
remain unverified. Independent component reviews found and fixed input-budget,
readiness-locking, numeric-parser, exporter-resumption and benchmark-redirect
issues before publication.

Native CPU inference, Metal, other GPU architectures, general model accuracy
and production throughput are outside this validation scope.
