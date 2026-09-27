# Cua-S1 4B 0.2 text worker

This recipe runs the Cua-S1 4B 0.2 `text` adapter behind the Rust frontend. The worker lives in [`src/models/cua_s1/text/`](../../src/models/cua_s1/text/), and [`src/models/cua_s1/README.md`](../../src/models/cua_s1/README.md) documents the inference contract and the request mapping. Only `choice` questions are supported.

Run all commands from the repository root, on Linux with an NVIDIA GPU.

## Install

Use Python 3.12. The pinned versions match the upstream reference environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r recipe/cua_s1/requirements-text.txt
```

## Download the weights

Download the pinned revisions (about 9.5 GB) into `weights/`:

```sh
.venv/bin/hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir weights/Qwen3.5-4B
.venv/bin/hf download cua-ai/cua-s1-4b-0.2 \
  --revision 16818868b0cc7813808aae4e87b417657046ab79 --local-dir weights/cua-s1-4b-0.2
```

To verify every file against upstream's lock, clone [trycua/cua](https://github.com/trycua/cua) next to this repository, check out `0e75660ce4c2edda519e0c795fa3ad98abf4e76f`, and run:

```sh
.venv/bin/python ../cua/libs/cua-s1/ci/fetch_pinned_weights.py --dest weights --verify-only
```

## Start the worker

```sh
PYTHONPATH=src .venv/bin/python -m models.cua_s1.text.server \
  --base weights/Qwen3.5-4B --adapter weights/cua-s1-4b-0.2/text \
  --device cuda --dtype bfloat16 --host 127.0.0.1 --port 8000
```

The worker loads the model and runs one warmup decision before it starts listening, so `GET /health` answers only once requests can be served; it then returns `{"status": "ready", "modality": "text", ...}`. The log reports load and warmup times separately. The worker refuses the `multimodal/` adapter and reports the adapter revision that `hf download` recorded. Every flag can also be set through an environment variable: `CUA_S1_BASE`, `CUA_S1_ADAPTER`, `CUA_S1_ADAPTER_REVISION`, `CUA_S1_DEVICE`, `CUA_S1_DTYPE`, `CUA_S1_HOST`, `CUA_S1_PORT`, `CUA_S1_MAX_BODY_BYTES`, `CUA_S1_MAX_QUESTIONS` and `CUA_S1_MAX_PROMPT_TOKENS`. `--adapter-revision` only sets the revision reported in `model` when the download metadata is missing; a value that contradicts the metadata stops the worker. Set `CUA_S1_API_KEY` to require `Authorization: Bearer <key>` on `/v1/systemone`.

Oversized requests get `413`: bodies over 4 MiB, more than 64 questions, or a question whose prompt is over 16,384 tokens (`--max-body-bytes`, `--max-questions`, `--max-prompt-tokens`). The worker computes logits for every prompt position, as upstream does, so memory grows with prompt length: serving the 15,446-token test input in bfloat16 peaked at about 21.3 GiB in use on the card. Requests run one at a time, and the frontend gives up after 60 seconds.

## Start the frontend

Build and start the frontend from the repository root, with stable Rust installed:

```sh
cargo build --release --locked
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
```

## Send a request

```sh
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"cua-s1-4b-0.2","state":"Dialog: Delete 3 files permanently? Buttons: Delete, Cancel","questions":{"pick":{"type":"choice","instructions":"Keep the files.","criteria":{"delete":"Click Delete","cancel":"Click Cancel"}}}}'
```

The answer has the Jev choice shape. On an RTX 6000 Ada in bfloat16, the response is:

```json
{"model":"cua-ai/cua-s1-4b-0.2@16818868b0cc7813808aae4e87b417657046ab79:text","answers":{"pick":{"type":"choice","choice":"cancel","probabilities":{"delete":0.0024726232513785362,"cancel":0.9975274205207825},"confidence":0.9750249565060322}},"usage":{"input_tokens":153,"output_tokens":0}}
```

## Check against upstream

`compare_text_with_upstream.py` scores the fixed input set (`tests/cua_s1/data/text_inputs.json`) with the worker's model and then with upstream `FourBModel`, one model at a time, and compares the results. It needs the trycua/cua checkout from above and two extra packages for upstream's processor:

```sh
.venv/bin/python -m pip install torchvision==0.29.0 pillow==11.3.0
.venv/bin/python recipe/cua_s1/compare_text_with_upstream.py --upstream ../cua \
  --base weights/Qwen3.5-4B --adapter weights/cua-s1-4b-0.2/text --device cuda
```

Every question must have identical prompt token ids and identical fp32 probabilities.

With the worker and the frontend running, `bench_text.py` checks that the frontend returns the same bytes as the worker for every input, then measures warm latency on both paths:

```sh
.venv/bin/python recipe/cua_s1/bench_text.py --direct http://127.0.0.1:8000 \
  --frontend http://127.0.0.1:8080 --warmup 3 --repeat 20
```

## Tests

The contract and HTTP tests need neither weights nor a GPU. The tokenizer tests also run when `CUA_S1_BASE` points to the downloaded base model; they read only its tokenizer files:

```sh
.venv/bin/python -m pip install pytest
CUA_S1_BASE=weights/Qwen3.5-4B PYTHONPATH=src .venv/bin/python -m pytest tests/cua_s1/test_text_*.py
```
