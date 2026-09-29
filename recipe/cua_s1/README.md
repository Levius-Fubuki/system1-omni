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
  total, at most 200:1 aspect ratio in either orientation, 16,384 characters per
  question and 4096 processed tokens per question.
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
separate forward pass over the same screenshot. Usage sums processed input tokens
and reports zero output tokens.

Response identity:
`cua-ai/cua-s1-4b-0.2@16818868b0cc7813808aae4e87b417657046ab79:multimodal`.
The base is BF16; PEFT's rank-16, alpha-32 adapter remains unmerged with fp32 LoRA
branches, including 50 vision projection modules (178 total adapted modules).

## Reproduce correctness and profiling

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

Metal, native CUDA kernels, text-adapter serving, batching, caching and training
are outside this worker's scope. This implementation does not import or modify
another contributor's text engine.
