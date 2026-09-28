# RTX 4090 final-token projection after image reuse — 2026-09-28

The request-local reuse path previously projected every language hidden state to
the 248,320-word vocabulary, then read only the last position to score choices.
Passing `logits_to_keep=1` to the pinned Transformers model projects that final
position alone. This change is limited to multi-question requests; the
single-question path remains the existing reference path.

Across 13 multi-question cases and two runs each, synchronized `engine.predict`
p50 fell **0.5–5.3%** (median reduction **3.4%**). The 640×480 screenshot with
eight different questions fell from **852.40 / 881.26 ms** to **822.68 / 851.65
ms**. Four single-question controls changed by **−0.4% to +0.2%** and are treated
as measurement noise. All 17 cases returned exactly the same full response with
the full and limited projections; no numerical tolerance was needed.

## Provenance and measurement

- Clean measured source: `224d88b07be354b3cfc99fe8e9096bed777eafeb`.
  [paired.json](paired.json) records its revision, clean status, environment,
  every sample, alternating call order, p50/p95 and allocated/reserved peaks.
- NVIDIA RTX 4090 24,564 MiB, driver 595.71.05, CUDA runtime 13.0, Python
  3.12.13, PyTorch 2.14.0, Transformers 5.17.0, PEFT 0.21.0, BF16 base with
  unmerged multimodal LoRA. The 178 adapted modules and fallback Gated DeltaNet
  and causal-convolution paths match PR #17. No package or model revision changed.
- Original 16-case matrix (two screenshot sizes, short/long instructions,
  1/2/4/8 repeated questions) plus eight distinct questions on a 640×480 image.
  Five warmup calls per variant and case, then two runs of 30 calls per variant:
  **2,040 timed predictions**. The order alternates each pair and reverses in
  the second run. CUDA is synchronized around each prediction. Hooks and the
  profiler are absent from timed samples.
- Both timed variants use the **same request-local image reuse path**. A temporary
  benchmark wrapper removes `logits_to_keep` for the full-projection comparison;
  the candidate retains it. Thus this experiment isolates output projection,
  unlike PR #17's before/after image-reuse comparison.
- Timing includes processor work and engine inference after request parsing.
  It excludes image decoding, HTTP, queueing, model load, correctness checks and
  profiling. Measurements are serial, concurrency 1.

| Case | Full p50 ms, runs 1 / 2 | Final-token p50 ms, runs 1 / 2 | p50 reduction | Peak allocated GiB, full → final |
| --- | ---: | ---: | ---: | ---: |
| 320×240 short, 1 question | 107.78 / 107.98 | 107.88 / 107.72 | −0.1% / +0.2% | 8.728 → 8.728 |
| 320×240 short, 8 questions | 716.83 / 715.95 | 711.50 / 705.99 | 0.7% / 1.4% | 8.726 → 8.677 |
| 320×240 long, 8 questions | 957.85 / 929.54 | 924.68 / 892.63 | 3.5% / 4.0% | 8.920 → 8.776 |
| 640×480 short, 1 question | 126.08 / 125.59 | 125.95 / 125.65 | +0.1% / −0.0% | 8.843 → 8.843 |
| 640×480 short, 8 questions | 857.32 / 901.78 | 830.27 / 875.32 | 3.2% / 2.9% | 8.838 → 8.739 |
| 640×480 long, 8 questions | 1086.47 / 1126.04 | 1029.76 / 1066.46 | 5.2% / 5.3% | 9.034 → 8.840 |
| 640×480 distinct, 8 questions | 852.40 / 881.26 | 822.68 / 851.65 | 3.5% / 3.4% | 8.994 → 8.812 |

The independent [verification script](verify_results.py) recomputes all 17 rows,
including p95, from every saved sample; the displayed rows are representative.
For 640×480 long, eight questions, peak allocated memory fell by **0.194 GiB**.
Reserved memory may persist across variants, so allocated memory is the useful
peak comparison here.

## Correctness and hotspot evidence

- [correctness.json](correctness.json): 13 fixtures / 49 question forwards per
  path. Prepared tensors, language embeddings, masks, three-dimensional positions,
  candidate probabilities and full responses are exactly equal to the original
  execution path. Vision executes once and language executes once per question.
  The fixtures include different goals, candidate counts and ordering, Unicode,
  PNG/JPEG and consecutive image changes. Unit tests cover failure recovery.
- [paired.json](paired.json): exact full-response parity for each of the 17 timed
  cases. Its output-head pre-hooks record an input change from `[1, 456, 2560]`
  to `[1, 1, 2560]` for every question in the 640×480 short, eight-question case;
  for eight distinct questions, `[1, 443, 2560]` becomes `[1, 1, 2560]` for
  the first question. Single-question head shapes are unchanged.
- [reuse-profile.json](reuse-profile.json): an **untimed** trace on the prior
  reused path at clean source `9f4a4fe` shows one visual forward, eight language
  forwards and eight output projections. Its shape-grouped `aten::mm` with
  inputs `[456, 2560]` and
  `[2560, 248320]` occurs eight times and records 35.93 ms of device time. This
  identifies the targeted output projection; profiled kernel time is not an
  additive estimate of end-to-end savings. Full Chrome traces are retained in
  the separate evidence archive because they are hundreds of MiB.
- [postflight.json](postflight.json): actual CUDA layout, HTTP health 200 and
  eight-question worker response 200, exactly equal to direct engine output.
  The clean GPU-host Cua-S1 suite passed **105 tests**; local macOS passed
  **104**, with one torch-dependent tensor test skipped locally. Ruff lint and
  formatting checks passed.

## Reproduce

Run from a clean checkout with the pinned environment and verified weights:

```sh
PYTHONPATH=src:recipe/cua_s1 python recipe/cua_s1/benchmark_last_logits.py \
  --weights /path/to/weights --output /path/to/fresh-output \
  --warmup 5 --runs 2 --iterations 30
python recipe/cua_s1/experiments/rtx4090-last-logits/verify_results.py
```

For the full reference-versus-reuse correctness sweep and real HTTP check:

```sh
PYTHONPATH=src:recipe/cua_s1 python recipe/cua_s1/benchmark_image_reuse.py \
  --weights /path/to/weights --output /path/to/fresh-correctness \
  --correctness-only
PYTHONPATH=src:recipe/cua_s1 python \
  recipe/cua_s1/experiments/rtx4090-reuse/postflight.py \
  /path/to/clean-repository /path/to/weights /path/to/fresh-postflight
```

The checked-in audit reads this directory's `paired.json`. To audit a fresh
run, copy its `report.json` to `paired.json` in a copy of this directory. The
profiling groups and Chrome traces are retained as separate evidence from the
timed benchmark. [SHA256SUMS](SHA256SUMS) identifies the checked-in data files
and logs.

The complete experiment directory, including all three large Chrome traces,
generated fixtures, logs and the measured source bundle, was copied to the
local ignored `work/20260928-post-reuse-evidence.tar.gz` before server shutdown.
The server and local archive SHA-256 matched:
`774d952e89a9586709419c6c8583e4aa7ee2d27b4bf836253c8c01881c4dccd9`
(25,231,406 compressed bytes). The checked-in JSON is sufficient to audit the
latency claims without this archive.

## Limits and next step

These numbers describe synthetic screenshots and serial, prefill-only engine
calls on one pinned RTX 4090 configuration. They do not establish HTTP
throughput, p99, GUI-task quality, native-engine support, CUDA Graph benefits or
Metal performance. The output projection is now bounded by the last token;
the remaining repeated language forwards and PyTorch fallback kernels are the
next candidates for measurement and validation.
