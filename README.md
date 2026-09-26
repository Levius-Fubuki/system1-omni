# omni-jev

Community-maintained serving infrastructure for prefill-only Jev models.

A small Rust frontend built with Axum, Tokio, and Reqwest. It forwards requests
unchanged to a separately running model worker and returns the worker's response.
The worker handles validation, media loading, tokenization, and inference.

Text, image, audio, video, and mixed payloads are supported at the transport level.
Actual inference support depends on the worker. Laya is used to verify text
decisions; scheduling and inference optimization are outside this milestone.

## Run

With stable Rust installed, build and start the frontend:

```sh
cargo build --release --locked
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
```

Both variables are optional; the values above are their defaults. The bind address
must be an IP address and port. The backend URL accepts an optional path prefix
(e.g. `http://localhost:8000/worker`), but no credentials, query, or fragment.
Backend connections bypass system HTTP proxies. Start the worker separately.

## Interface

- `POST /v1/systemone` forwards the `model`, `state`, and `questions` envelope
  unchanged. Workers return `choice`, `score`, or `noul` decisions; see the
  [Jev API reference](https://docs.typesafe.ai/api).
- `GET /health` forwards the worker's health response, including unhealthy status codes.
- Authorization and other end-to-end headers are forwarded. Backend status,
  content type, and body are preserved; redirects are returned without following them.
- A shared client reuses connections with a 60-second total timeout and no retries.
  Transport failures return `502`; timeouts, including response-body timeouts, return `504`.
- Uploads are streamed; responses are buffered before their status is sent.
  Request size and concurrency limits should be set at the ingress or worker.

## Try a Laya text worker

In another terminal, install the optional worker in a Python 3.12 environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install 'laya[serve]==0.3.20'
LAYA_HOST=127.0.0.1 LAYA_PORT=8000 LAYA_DEVICE=cpu \
LAYA_MODELS=english LAYA_PRELOAD=1 LAYA_THREADS=4 \
  .venv/bin/laya-serve
```

First startup downloads the English checkpoint. Once the worker is ready, send
a request through the frontend:

```sh
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"english","state":"Please refund the duplicate charge.","questions":{"refund":{"type":"noul","instructions":"Does the customer ask for a refund?"}}}'
```

## Check

The Rust tests use local mock workers and need no model weights or GPU. They cover
multimodal byte preservation, authentication, connection reuse, large uploads,
backend errors, timeouts, and a binary smoke test with SIGTERM shutdown on Unix.
The GitHub Actions workflow runs these checks on Linux:

```sh
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
```

With Laya and the frontend running, compare health and all three decision types,
separately and together (Python standard library only):

```sh
python3 scripts/compare_with_backend.py --model english \
  --backend http://127.0.0.1:8000 --frontend http://127.0.0.1:8080
```

Each check requires status `200` and identical status, content type, and body bytes
for direct and proxied requests. Set `OMNI_JEV_TEST_TOKEN` if the worker requires a
bearer token. Use a deterministic worker response for this byte-level comparison.
