# Cua-S1 0.2 multimodal CUDA worker

This recipe adds screenshot decisions using the multimodal adapter discussed in
[#10](https://github.com/ThinkFlowLab/system1-omni/issues/10). It loads Transformers
and PEFT directly. Upstream `FourBModel` is used only as an independent parity
oracle. The worker owns image decoding, the processor/chat template, vision and
language LoRA loading, and the candidate-letter probability readout.

The text mapping and pinned revisions follow
[PR #11](https://github.com/ThinkFlowLab/system1-omni/pull/11). The image `state`
format below is this PR's proposed extension. A separately launched multimodal
worker uses the same request model name; deployment routing selects its modality.

## Setup

Run from this repository's root on Linux with an NVIDIA GPU. The measured CUDA
wheel, driver, GPU memory and results are recorded in [experiments](https://github.com/Levius-Fubuki/system1-omni/blob/683d470669f19d5e9530cd8233727a0d4f1f8d29/recipe/cua_s1/experiments/README.md).
Python 3.12 is required by the pinned environment.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r recipe/cua_s1/requirements-multimodal.txt
PYTHONPATH=src python recipe/cua_s1/download_weights.py --dest weights
PYTHONPATH=src python -m frontend.cua_s1 \
  --base weights/Qwen3.5-4B \
  --adapter weights/cua-s1-4b-0.2/multimodal
```

The downloader fetches the upstream manifest at the fixed reference revision,
checks its pinned SHA-256 and saves it as `weights/weights.lock.json`. The worker
reads this local manifest and checks every loaded artifact's size and SHA-256.
Keep the manifest next to the base checkpoint directory when moving weights.
Extra files are rejected, except Hugging Face's `.cache` metadata, so another
checkpoint cannot silently override verified shards. Downloads require roughly
9 GB plus cache/install space. Loading is offline after the download completes.
Weights are not included in this repository.

The HTTP adapter lives in `src/frontend/cua_s1.py`; model execution stays in
`src/models/cua_s1/multimodal/`.

The worker binds to `127.0.0.1:8000` only after loading and a successful warmup.
`GET /health` returns `{"status":"ready","modality":"multimodal"}`. One request
runs at a time; concurrent requests return `503`. This is a loopback model worker,
with the Rust frontend and an ingress responsible for public serving.

To enable the optional CUDA Graph runtime for multi-question requests, add
`--graph` to the worker command. It captures each run of Gated DeltaNet layers
at its actual token length and keeps full-attention layers eager. A layout must
appear in two distinct `predict` requests, at most eight requests apart, before
capture. Repeated questions inside one request count once. The captured logits
must exactly match eager before the layout enters the cache. The defaults retain
at most eight layouts and 1 GiB of measured live Graph allocations.

The default remains eager. `--graph` is an alias for `--graph-mode exact`.
For workloads rotating through many nearby lengths, explicitly select
`--graph-mode rule-bucket --graph-bucket-width 64` instead. This mode captures
only the internal DeltaNet rule, padding its query/key/value/decay/update inputs
to a multiple of 64 tokens. Projections, convolutions, MLPs and full attention
keep their actual lengths. Width must be a positive multiple of 64; the padded
length must fit `--graph-max-tokens`. The worker uses direct per-model calls,
without replacing Transformers globals or model methods. This path requires
the pinned Transformers 5.17.0 implementation and an immutable resident model;
it does not support model offloading or sharding.

Rule buckets share the admission and resource limits below. Complete vocabulary
logits must match eager exactly at capture and for each new actual length in a
cached bucket. A rejected length uses eager while other verified lengths remain
eligible. Dense batch-one inputs are supported; unsupported inputs use eager.
These first-input checks do not prove equality for every possible content.
For bounded online selection, opt in with `--graph-mode auto`. It observes the
last 32 requests (at most 128 layouts) and chooses eager, exact or rule-bucket
per layout, keeping that choice fixed within a request. Exact and bucket entries
share one namespaced LRU cache, request clock, capture-count/time ledger and
resident-memory budget. No second budget is allocated when switching modes.
The selector estimates reuse over the next 64 requests, compares savings with
capture and new-length validation costs, and applies a 25% capture-cost margin
and a 32-request switching cooldown. Initial heuristics use 500 ms capture cost
and exact/bucket replay ratios of 0.45/0.65 relative to eager; observations replace
these priors. CUDA event spans are sampled for the first eight supported forwards
and every sixteenth thereafter, queried without extra synchronization. Capture
and length-check spans do not train replay costs. This heuristic is workload
dependent and does not guarantee the fastest fixed mode. It uses the same pinned,
single-device fallback implementation and strict gates as manual buckets.
`selected_eager`, `selected_exact` and `selected_rule_bucket` count chosen routes;
admission or validation may still make a selected Graph route run eagerly.
Invalidation also clears selector history and pending timing events. See the
[automatic worker report](https://github.com/Levius-Fubuki/system1-omni/blob/683d470669f19d5e9530cd8233727a0d4f1f8d29/recipe/cua_s1/experiments/rtx4090-auto-worker/README.md) for controlled
comparisons, strict parity and shared-budget validation.

Stable hot lengths can favor exact mode. See the
[worker integration report](https://github.com/Levius-Fubuki/system1-omni/blob/683d470669f19d5e9530cd8233727a0d4f1f8d29/recipe/cua_s1/experiments/rtx4090-bucket-worker/README.md) for
correctness, HTTP checks, and workload-dependent timing.

`--graph-min-uses` now counts **requests**, not question forwards. Use
`--graph-admission-window` to change the maximum gap between observations.
Evicted layouts lose their heat and wait `--graph-cooldown-requests` (default 32)
before accumulating new observations. History and cooldown tables retain at
most 128 layouts each. Every prediction, including the single-question reference
path, advances the request clock while Graph is enabled.

Capture is limited to `--graph-max-captures` (default 4) attempts per sliding
`--graph-capture-window` (default 32 requests). `--graph-capture-budget-ms`
(default 2000) also limits the accumulated elapsed attempt work, including the
eager correctness gate, failures and cleanup. A synchronous capture cannot be
interrupted; one attempt may overshoot this time budget and later attempts then
fall back to eager until budget expires. This is a request-count window, not a
wall-clock rate limit. Cached layouts can still replay when capture is paused.

`--graph-max-shapes`, `--graph-max-memory-mib` and `--graph-max-tokens` adjust
cache and token limits. Graph pool accounting requires the native PyTorch CUDA
allocator; other allocator backends fall back eagerly. Inputs above 2,048 tokens, rejected layouts, insufficient
reuse and exhausted budgets use eager inference. The memory budget covers
the full reservations of each retained Graph private pool plus external static
input buffers. It excludes the model, default allocator cache and capture-time
peaks; an in-progress candidate temporarily coexists with retained entries.
Segments of one shape share an exclusively owned stream and pool and replay
in capture order. Different shapes have independent owners. Cache entries own
their static buffers and all segments are retired together. The loaded model
must remain immutable; call `graph_runtime.invalidate()` before changing its
weights or adapters. Invalidation clears admission and cache state, while lifetime
statistics remain cumulative. `predict` opens a serialized Graph request context;
standalone runtime `forward` calls without that context use eager. Single-question
requests continue through the reference path.
`engine.close()` explicitly releases Graph resources and rejects later
predictions. HTTP shutdown drains accepted handlers before closing the engine;
the lifecycle lock also waits for an active direct prediction. Close is idempotent.

Whole-model capture changed BF16 attention results on the measured RTX 4090;
the earlier [Graph feasibility report](https://github.com/Levius-Fubuki/system1-omni/blob/683d470669f19d5e9530cd8233727a0d4f1f8d29/recipe/cua_s1/experiments/rtx4090-graph/README.md)
records those failures. The segmented runtime preserves eager attention
behavior. See [the runtime experiment](https://github.com/Levius-Fubuki/system1-omni/blob/683d470669f19d5e9530cd8233727a0d4f1f8d29/recipe/cua_s1/experiments/rtx4090-graph-runtime/README.md)
for fixed-layout correctness and latency, and the
[mixed-length experiment](https://github.com/Levius-Fubuki/system1-omni/blob/683d470669f19d5e9530cd8233727a0d4f1f8d29/recipe/cua_s1/experiments/rtx4090-graph-mixed-shapes/README.md) for
capture and eviction costs under changing lengths. The subsequent
[request-aware admission report](https://github.com/Levius-Fubuki/system1-omni/blob/683d470669f19d5e9530cd8233727a0d4f1f8d29/recipe/cua_s1/experiments/rtx4090-graph-admission/README.md)
compares bounded capture admission against the old policy with the same owned-stream
fix, including cold requests and negative results. See the
[reproduction guide](graph-admission.md) for the complete schedules.

To use the Rust frontend included in this repository:

```sh
cargo build --release --locked
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 ./target/release/omni-jev
```

## Request and response

Generate a self-contained example with a synthetic settings screenshot:

```sh
python recipe/cua_s1/make_example.py --output /tmp/cua-example.json
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @/tmp/cua-example.json
```

Replace the port with `8080` to send the same request through the frontend.
The request shape is:

```json
{
  "model": "cua-s1-4b-0.2",
  "state": {"image": "data:image/png;base64,<encoded screenshot>"},
  "questions": {
    "next": {
      "type": "choice",
      "instructions": "Save the changes",
      "criteria": {"save": "Save changes", "cancel": "Cancel"}
    }
  }
}
```

- `state` contains exactly one inline PNG or JPEG data URL. Images are decoded to
  RGB. Local filenames, remote URLs, video, animation and mixed text/image state
  are unsupported.
- There are 1–8 questions and 1–26 options per question. Option order assigns
  letters A–Z. Labels are strings, objects, arrays or `null` (which uses the key).
  Structured values use Python `json.dumps(..., ensure_ascii=False)` followed by
  the upstream chooser's label escaping. Instructions accept strings, objects
  or arrays and must be present; an empty string or `null` omits the goal block.
- Limits: 8 MiB body, 4 MiB decoded image, 2048 pixels per side, 1,048,576 pixels
  total, at most 200:1 image aspect ratio, 16,384 characters per question and
  4096 processed tokens per question. Unsupported image dimensions or aspect
  ratios return `422` before model inference.
  Every question is validated/preprocessed before any forward pass begins.
- `<|image_pad|>`, `<|video_pad|>`, `<|vision_start|>` and `<|vision_end|>` are
  rejected in user text because the processor interprets them as media controls.
  Other special-token spellings retain upstream tokenization behavior.
- Malformed JSON (including duplicate keys, non-finite numbers, invalid UTF-8,
  lone surrogates and non-object bodies) returns `400`. Well-formed unsupported
  inputs, missing `instructions`, unsupported models and `score`/`noul` questions
  return `422`. Error bodies use `{"detail": "<message>"}`. Oversized bodies
  return `413`; chunked uploads return `411`. Send `Content-Length` and `Content-Type: application/json`.

Each answer has `type`, `choice`, `probabilities` and `confidence`. The readout
uses the last position's candidate-letter logits, casts to fp32 and applies
softmax over those letters only. There is no decode. Ties select the earliest
option; confidence is `1 - H(p)/ln(n)`, or 1 for one option. Each question has a
separate language forward pass. Multi-question requests reuse the image
preprocessing and adapted vision features within the request; see the
[reuse contract and experiment](image-reuse.md). Usage sums processed input
tokens and reports zero output tokens.

Response identity:
`cua-ai/cua-s1-4b-0.2@16818868b0cc7813808aae4e87b417657046ab79:multimodal`.
The base is BF16; PEFT's rank-16, alpha-32 adapter remains unmerged with fp32 LoRA
branches, including 50 vision projection modules (178 total adapted modules).

## Reproduce correctness and profiling

For the staged image-size, text-length and same-image question-count experiment,
see [multimodal profiling](profiling.md), including hardware requirements and
separate unprofiled timing and instrumented traces.

```sh
git clone https://github.com/trycua/cua.git /tmp/cua-reference
git -C /tmp/cua-reference checkout 0e75660ce4c2edda519e0c795fa3ad98abf4e76f
for mode in reference candidate benchmark; do
  PYTHONPATH=src python recipe/cua_s1/evaluate_multimodal.py \
    --weights weights --reference /tmp/cua-reference \
    --output /tmp/cua-evidence --mode "$mode"
done
```

The evaluator hashes the pinned `four_b.py` before importing it and checks report
provenance/environment before comparison. Reference and candidate are separate
processes to avoid keeping two models in GPU memory. Eight synthetic requests
(nine question forwards) cover two resolutions, PNG/JPEG, 1/26 candidates,
structured/Unicode labels, special-token text and multiple questions. It requires
identical processor tensor shapes/dtypes/hashes and identical fp32 candidate
probabilities; numerical tolerance is zero. These are integration/parity fixtures,
not an evaluation of GUI task success.

Benchmark mode records two runs of 50 serial requests after five warmups on the
640×480 fixture, with synchronized end-to-end engine latency, p50/p95, serial
throughput, allocated/reserved GPU peaks and a separate operator profile. Load
and warmup are recorded separately; candidate load time includes artifact hash
verification. See the experiment report for measured scope and limitations.

CPU-only validation:

```sh
pip install Pillow==11.3.0 pytest==9.1.1 ruff==0.16.8
PYTHONPATH=src python -m pytest tests/cua_s1 -q
ruff check --select E4,E7,E9,F,I src/frontend/cua_s1.py src/models/cua_s1/multimodal recipe/cua_s1/*.py tests/cua_s1
ruff format --check src/frontend/cua_s1.py src/models/cua_s1/multimodal recipe/cua_s1/*.py tests/cua_s1
```

Metal, native CUDA kernels, text-adapter serving, batching and training
are outside this worker's scope. This implementation does not import or modify
another contributor's text engine.
