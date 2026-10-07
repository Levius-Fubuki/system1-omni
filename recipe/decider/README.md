# Native Decider-2B v11 text decisions

Run these commands from the repository root on Linux with Rust, an NVIDIA GPU,
CUDA toolkit/nvcc and cuBLASLt. The existing Qwen backend targets compute capability
8.0 or newer; hardware validation is described in [validation.md](validation.md).
The worker itself uses no Python/PyTorch serving process. Python 3.10+ is used only
for downloading and optional reference validation. Keep at least 8 GB disk free for
weights and Rust build products; reference Python packages need additional space.
Device memory includes about 3.76 GB of BF16 checkpoint data plus model/head buffers
and length-dependent activation/workspace storage.

## Download and build

```sh
python3 recipe/decider/download_weights.py /path/to/models/decider-2b-v11
cargo build --release --locked -p omni-decider-native -p omni-jev
NVCC=/usr/local/cuda/bin/nvcc \
  bash src/backends/cuda/qwen3_5/build.sh target/release 89
```

Replace `89` with the target GPU's compute capability. The download helper pins
`Mapika/decider-2b` revision `533964dae8be954c5b5e19fa4948e48408094c1e` and checks
all five downloaded files; it never downloads unpinned remote Python code. Its
`--endpoint` option can select an accessible Hugging Face mirror; fixed hashes
remain required. Interrupted downloads leave no usable unverified final file.

Startup independently hashes the required checkpoint/config/calibration/tokenizer
and rejects changed artifacts before loading CUDA. Use the original single
`model.safetensors`, with no `model.safetensors.index.json`; no export or LoRA merge
is needed. Keep files immutable while loading and serving.

## Start the worker and frontend

Worker terminal:

```sh
CUA_S1_GRAPH=0 \
DECIDER_MODEL=/path/to/models/decider-2b-v11 \
DECIDER_CUDA_LIB="$PWD/target/release/libqwen3_5_cuda.so" \
DECIDER_HOST=127.0.0.1 DECIDER_PORT=8000 \
  ./target/release/omni-decider
```

Decider's first path is eager. `CUA_S1_GRAPH=1` is rejected at startup so a shared
backbone switch cannot silently change this path. An unset switch is also accepted.
`DECIDER_CUDA_LIB` defaults to the library beside the binary. The worker uses CUDA
device 0; use `CUDA_VISIBLE_DEVICES` to select a physical device.

The socket binds only after artifact validation, CUDA loading and a real warmup
request. Startup failure does not advertise readiness. In another terminal:

```sh
curl --fail http://127.0.0.1:8000/health
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
```

Health returns the model identity, checkpoint/reference revisions, BF16/eager mode
and effective per-type temperatures. A retired executor returns HTTP 503 and
`status: unavailable`. The frontend exposes its existing health behavior; see the
[frontend configuration](../../src/frontend/README.md) for its own limits/timeouts.

## First decision

```sh
curl --fail http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/decider/example-request.json
```

The example asks a Choice, Noul and three-level Score question about one text
state. The response has model `decider-2b-v11`, ordered typed answers and usage.
Question identities are omitted from model text. The exact output depends on the
checkpoint computation; this example is not a task-quality guarantee.

Errors use JSON `detail`: HTTP 415 for unsupported Content-Type, 413 for a body
above 8 MiB, 422 for invalid/unsupported requests, and 503 when the model becomes
unavailable after an execution failure. Request-preparation task failures return
500. Invalid inputs do not retire the executor. Empty questions return HTTP 200
with empty answers and zero input/output usage. No partial answers are returned.

Supported inputs are plain text/JSON state, independent Choice (2–255 options),
Noul and isolated Score (2–10 levels). State-prefix truncation is 32,768 tokens;
complete rows can include question suffixes up to 36,864 tokens. At most 1,024
expanded rows and 1,048,576 processed tokens are accepted per request. These bounds
do not bound all waiting requests together; runtime pending-queue limits and
cross-request batching remain separate work.

The pinned worker requires released calibration. Images, video, chat/schema-first,
packed questions, shared-prefix caching, quantized/CPU/Metal execution and Graph
optimizations are unsupported. See the [model contract](../../src/models/decider/README.md)
for native JSON restrictions and the processing/execution boundary.

## Validation and diagnostics

```sh
cargo fmt --all --check
cargo clippy --workspace --locked --all-targets -- -D warnings
cargo test --workspace --locked
cargo build --workspace --release --locked
DECIDER_MODEL=/path/to/models/decider-2b-v11 \
  cargo test --release --locked -p omni-decider-native --test contract -- --ignored
DECIDER_MODEL=/path/to/models/decider-2b-v11 \
DECIDER_CUDA_LIB="$PWD/target/release/libqwen3_5_cuda.so" CUA_S1_GRAPH=0 \
  cargo test --release --locked -p omni-decider-native --test gpu -- --ignored
DECIDER_CUDA_LIB="$PWD/target/release/libqwen3_5_cuda.so" \
  cargo test --release --locked -p omni-decider-native --lib \
  bf16_head_rounds_before_fp32_calibration -- --ignored
```

The CPU contract target requires only tokenizer/config; the GPU target requires the
complete pinned checkpoint. The projection test needs a built backend and a GPU.

For exact prepared rows and raw BF16-rounded logits, the diagnostic binary reads
one JSON request per line and emits rows, logits and complete response:

```sh
CUA_S1_GRAPH=0 ./target/release/decider-run \
  /path/to/models/decider-2b-v11 "$PWD/target/release/libqwen3_5_cuda.so" \
  < recipe/decider/example-request.json
```

See [validation.md](validation.md) for full-checkpoint reference comparison,
HTTP/frontend checks, prerequisites and the recorded scope.

## Optional request-local packing

Set `DECIDER_BATCH_MAX_ROWS=4 DECIDER_BATCH_MAX_TOKENS=4096` on `omni-decider`
to pack independent rows within one admitted request. Rows default to 1; accepted
limits are 1–4 rows and 1–4096 packed tokens. Longer complete rows execute alone.
Malformed or out-of-range settings fail startup. These limits do not change HTTP
admission, token usage, or the maximum complete-row length.

The selected-label head uses persistent buffers sized to the row limit and projects
each batch in one BF16 GEMM. No cross-request batching is introduced. Keep the
single-row default until numerical and performance results fit your workload.
