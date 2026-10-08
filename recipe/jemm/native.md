# JEMM native Rust/CUDA worker

JEMM serves text and image `choice`, `noul` and `score` decisions using
Qwen3.8-27B. The Rust worker validates and prepares the whole request, runs
vision once for its shared images, and performs one language prefill per
question. The selected BF16 LM head runs on CUDA; calibrated probabilities
and typed responses are finished on the CPU.

The [validation report](validation.md) records completed A800 validation on the
fixed corpus: exact text/image preprocessing, response parity, HTTP behavior
and matched warm HTTP timings. All measured requests succeeded within the
frozen numerical gates. Shared CUDA kernels, retained Cua-S1 4B vision behavior
and pinned artifacts were also checked. Native CPU inference, Metal and other
GPU configurations remain unverified. These results cover this corpus and
reference environment, without general accuracy or production throughput claims.

## Environment and storage

The current A800 environment is Linux with one NVIDIA A800 80 GB (`sm_80`),
driver 595.71.05, CUDA toolkit 13.0.88, Python 3.12.3, PyTorch 2.12.1+cu130,
torchvision 0.27.1, Transformers 5.17, PEFT 0.21, Accelerate 1.15 and
flash-linear-attention 0.5.2. The reference image processor requires the
matching torchvision installation.
Use stable Rust and a CUDA toolkit providing `nvcc` and cuBLASLt. Rust compilation
itself does not require CUDA; inference needs the separately built library.

The raw base and adapter use about 55.6 GB; the native export needs about
52 GB more. Preserve both until reference and native comparisons finish.
Place them on a data volume with room for downloads, both copies and temporary
shards. The validation host has a 170 GiB data disk and a 30 GiB system disk.
Its cgroup RAM limit is 120 GiB; that is a memory limit, separate from disk
capacity. Use direct local-directory downloads and inspect free disk space
before copying weights. Allow room for build caches and evidence.

The exporter processes one base shard at a time. RAM must accommodate that
shard, adapter tensors, merged shard output and bounded FP32 LoRA intermediates;
it does not instantiate two complete 27B models. An interrupted export can be
resumed with the same inputs, producer script and recorded CPU runtime. Keep checkpoint files immutable
while the worker maps them.

## Download the pinned artifacts

Run the following from the System1-Omni repository root. Set `JEMM_DATA` to your
large data volume; these examples use `/data/jemm`.

```sh
export JEMM_DATA=/data/jemm
mkdir -p "$JEMM_DATA"
python3 -m venv "$JEMM_DATA/venv"
. "$JEMM_DATA/venv/bin/activate"
python -m pip install 'torch==2.12.1' 'torchvision==0.27.1' \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install 'transformers==5.17.0' 'peft==0.21.0' \
  'accelerate==1.15.0' 'flash-linear-attention==0.5.2' \
  safetensors pillow 'huggingface-hub'

hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --include '*.safetensors' '*.json' '*.jinja' '*.txt' 'README.md' 'LICENSE' \
  --local-dir "$JEMM_DATA/base"
hf download MaestroYan/JEMM \
  --revision 76e3c209e8441fa658221c7ba2725bad2f811176 \
  --include 'adapter_config.json' 'adapter_model.safetensors' 'decision_config.json' 'README.md' 'LICENSE' \
  --local-dir "$JEMM_DATA/adapter"
git clone https://github.com/ypcypc/JEMM.git "$JEMM_DATA/reference"
git -C "$JEMM_DATA/reference" checkout --detach 6822fe0fd53c5e6670af6ba99fb2c857a661e532
```

The base pin includes tokenizer, chat template, image processor and configuration
files. Do not substitute a similarly named model or a newer revision. The
[checked-in inventory](https://github.com/ThinkFlowLab/system1-omni/blob/main/recipe/jemm/pinned_inventory.json)
records every required file's SHA256 and every input tensor's name, shard,
shape and dtype. Both export and reference startup verify the pinned raw bytes;
reference startup also checks the official source files.

Create the required download provenance JSON:

```sh
cat > "$JEMM_DATA/download-provenance.json" <<'JSON'
{
  "base_model_id": "Qwen/Qwen3.8-27B",
  "base_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  "checkpoint_revision": "76e3c209e8441fa658221c7ba2725bad2f811176",
  "source_revision": "6822fe0fd53c5e6670af6ba99fb2c857a661e532",
  "download_method": "hf download with explicit revisions; git detached checkout"
}
JSON
```

The first four fields are required and must match exactly. Additional provenance
fields are retained in the export manifest. Do not include credentials or signed
download URLs. Provenance declares the source; file SHA256 verification checks
the actual inputs.

## Export on the CPU

```sh
CUDA_VISIBLE_DEVICES='' python recipe/jemm/export.py \
  --base "$JEMM_DATA/base" \
  --adapter "$JEMM_DATA/adapter" \
  --out "$JEMM_DATA/native" \
  --provenance "$JEMM_DATA/download-provenance.json" \
  --threads 4
```

The export merges language LoRA with a finite-checked FP32 delta added into
BF16 base weights. It retains the unadapted base vision weights in
`vision.safetensors` and extracts the 32 rows of **untied `lm_head.weight`**
corresponding to `A`–`Z`, then `0`–`5`, into
`jemm_lm_head.safetensors` with shape `[32, 5120]`. Token embeddings or a scalar
head cannot substitute for this readout. The full LM head and MTP tensors are
omitted from the native export.

`jemm_export.json` is the completion marker, written last. Its
`jemm-native/1` format records pins, input/output checksums, the exporter hash and CPU runtime,
label token IDs, disabled-thinking chat boundaries, calibration, token budgets
and download provenance. `export_progress.json` tracks atomically completed
shards. The resume fingerprint includes the producer script's SHA256, Python,
PyTorch, safetensors, Transformers and tokenizers versions, CPU platform and
capability, thread count and FP32 matmul setting. Repeating the command under
that same producer verifies and reuses completed outputs; changed inputs,
producer code or recorded runtime are rejected for both partial and finalized
exports. Use a new output directory after an exporter or runtime change. The worker verifies the
completed export before loading GPU weights.

A completed validation artifact retains its original producer identity:
`jemm_export.json`'s `exporter_sha256` identifies the exact archived producer
script, even when the recipe later changes. Preserve that script together with
the manifest and output hashes, and cite its recorded hash in the validation
evidence. Later recipe fixes do not regenerate or reattribute existing weights.
An artifact created before the producer-bound resume fingerprint was added
remains an inference input under its original manifest; the current exporter
rejects reuse of that directory rather than certifying it as a new export.

## Build and launch

Build the library for A800's compute capability explicitly:

```sh
src/backends/cuda/qwen3_5/build.sh target/release 80
cargo build --release --locked -p omni-jemm-native -p omni-jev
CUDA_VISIBLE_DEVICES=0 \
JEMM_MODEL="$JEMM_DATA/native" \
JEMM_CUDA_LIB="$PWD/target/release/libqwen3_5_cuda.so" \
JEMM_HOST=127.0.0.1 JEMM_PORT=8000 \
  target/release/omni-jemm-native
```

Use exclusive GPU ownership for model runs and comparisons. Compute capability
8.0 or newer is required by the shared kernels, but the JEMM validation scope
here is the A800 configuration above. Rebuild the CUDA library for JEMM even if
an older ABI 5 library is present: the 27B vision encoder requires the new
optional `cs1_vision_position_v2`, `cs1_vision_rope_v2` and
`cs1_vision_attention_v2` symbols. Existing Cua-S1 4B execution retains the
legacy entry points.

`JEMM_MODEL` is required. `JEMM_CUDA_LIB` defaults to the library beside the
executable; `JEMM_HOST` and `JEMM_PORT` default to `127.0.0.1` and `8000`.
Visible CUDA device 0 receives the language, vision and selected-head weights.
The worker completes a real text warmup before binding its listener. A backend
error or panic synchronizes and retires loaded state; subsequent health checks
report unavailable until restart.

In another terminal, launch the Rust frontend and check readiness:

```sh
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  target/release/omni-jev
```

```sh
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8080/health
curl --fail-with-body http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' --data-binary '{
    "model": "JEMM",
    "state": {"service": "payments", "healthy": true, "errors": 0},
    "questions": {
      "action": {
        "type": "choice",
        "instructions": "Which action inspects logs?",
        "criteria": {
          "inspect": {"description": "Inspect logs."},
          "deploy": {"description": "Deploy a release."}
        }
      }
    }
  }'
```

Ready native health returns `{"status":"ready","model":"JEMM"}`. Decision
responses contain `model`, ordered `answers`, and `usage` with `input_tokens`,
zero `output_tokens` and `latency_ms`. A `choice` answer includes the selected
candidate ID, probabilities and maximum-probability confidence. Specific
probabilities depend on model execution.

## Image requests and input limits

Images are optional top-level `images`, shared by all questions and placed
before the question text. Supply at most four PNG, JPEG or WebP images as
base64 strings or base64 data URLs. For example, prepare a request from a local
PNG and send it through the same frontend:

```sh
python - <<'PY' > /tmp/jemm-image-request.json
import base64
import json
from pathlib import Path
image = base64.b64encode(Path("screenshot.png").read_bytes()).decode("ascii")
print(json.dumps({
    "model": "JEMM",
    "state": "The image shows the current deployment console.",
    "images": ["data:image/png;base64," + image],
    "questions": {"action": {
        "type": "choice",
        "instructions": "Which visible control opens deployment logs?",
        "criteria": {"logs": "Click Logs.", "restart": "Click Restart."}
    }}
}, ensure_ascii=False))
PY
curl --fail-with-body http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @/tmp/jemm-image-request.json
```

`choice` and `score` accept 2–32 candidates or levels. `noul` uses `yes`, then
`no`, and returns `p(yes)` as `noul`. Score levels have zero-based indices;
`expected_value` is the probability-weighted index. Candidate insertion order
selects labels and resolves exact ties. Structured state uses compact JSON
without sorting keys.

Each complete text prompt is limited to 8192 tokens; each image prompt to
3072 tokens, including images and chat framing. Usage sums the complete prompt
length for every question. Each source image is limited to 3,145,728 pixels.
The pinned processor uses a 65,536 minimum and 16,777,216 maximum resized pixel
setting. Over-budget prompts fail before GPU execution, without truncation or
budget-dependent resizing. Native server limits deliberately bound the HTTP
body, question count and aggregate rendered prompt size in addition to each
prompt's token budget. See the
[worker contract](https://github.com/ThinkFlowLab/system1-omni/blob/main/src/models/jemm/README.md#api-semantics)
for the current limits; these bounds can reject requests accepted by the
reference.

Native request JSON accepts finite floating-point numbers with Python-compatible
rendering and integers in the i64/u64 range. Integer `-0` renders as `0`; floating
`-0.0` retains its sign. Nonfinite/out-of-range floating-point values and lexical
integers outside i64/u64 are rejected before processing; Python's arbitrary-size
integers are not supported. The same numeric restriction applies inside JSON
tool descriptions. Duplicate keys and a requested model other than `JEMM` are
rejected. Invalid requests return 422; unsupported content types return 415.

## Run the pinned reference and compare

The official reference uses the raw base and **unmerged** PEFT adapter.
The wrapper preserves the upstream model and HTTP handler, validates source and
checkpoint hashes, and records observed classes, functions, kernel availability,
attention settings, parameter dtypes and actual adapter merge state. The
validated reference had a BF16 base and FP32 LoRA tensors, with SDPA and FLA
available. Optional `causal_conv1d` was absent from the official default
environment, so the reference used the Torch convolution fallback. The native
worker used merged BF16 language weights and eager execution. The measured
comparison therefore covers those recorded paths; it does not establish a
speedup against a reference with every optional optimization installed.
Run reference and native deployments sequentially on the same reserved GPU;
keep the frontend at `127.0.0.1:8080` and each worker at port 8000.

With the native process stopped, start the reference:

```sh
CUDA_VISIBLE_DEVICES=0 python recipe/jemm/reference_server.py \
  --source "$JEMM_DATA/reference" \
  --base "$JEMM_DATA/base" --adapter "$JEMM_DATA/adapter" \
  --device cuda:0 --host 127.0.0.1 --port 8000 \
  --capture-corpus tests/jemm/fixtures/corpus.jsonl \
  --evidence "$JEMM_DATA/evidence/reference-environment"
```

Then replay the checked-in frozen corpus through the frontend:

```sh
python recipe/jemm/benchmark.py --url http://127.0.0.1:8080 \
  --corpus tests/jemm/fixtures/corpus.jsonl \
  --out "$JEMM_DATA/evidence/reference"
```

Stop the reference, launch the native worker with the earlier command, check
health, then run the identical replay and strict response gate:

```sh
python recipe/jemm/benchmark.py --url http://127.0.0.1:8080 \
  --corpus tests/jemm/fixtures/corpus.jsonl \
  --out "$JEMM_DATA/evidence/native"
python recipe/jemm/compare.py \
  --reference "$JEMM_DATA/evidence/reference/measured.jsonl" \
  --actual "$JEMM_DATA/evidence/native/measured.jsonl" \
  --out "$JEMM_DATA/evidence/response-parity.json"
```

Evidence directories must be fresh. The frozen corpus has 13 synthetic cases
and SHA256 `5e729330b148aacec058abda8f5e0d81f9b9064a2e2bb8401426fd97e9815c4c`.
Use its checked-in bytes rather than regenerating images under another Pillow
version. Each deployment gets one full-corpus feasibility pass and two measured
passes, in fixed order at concurrency 1. Warm HTTP latency includes frontend,
CPU preparation, device execution and response transfer/parsing; model loading,
startup warmup and evidence writes are excluded. Report text and image samples
separately. This workload checks implementation parity, not general decision
accuracy or throughput.

Preprocessing acceptance requires exact prompts/token IDs, label IDs, grids,
positions, token usage and question/candidate identities. The response gate
requires maximum absolute probability drift ≤0.02, score expected-value drift
≤0.1, and the same winner when the reference margin is ≥0.05. Low-margin cases
and failures remain in the records. Never treat a response-only gate as a
complete preprocessing or vision-feature comparison; the
[validation report](validation.md) records each boundary separately.

The native `--prepare-jsonl` mode loads only export metadata without CUDA. It
emits the question `prompt`, raw `template_chat` with one image placeholder
per image, and `chat` with image placeholders expanded to the image grid
lengths. `template_chat` corresponds to the reference's raw processor template;
`chat` is the native text passed to tokenization. Diagnostics also include
token and label IDs, image grids/indices, three-axis positions, resized geometry
and FP32 pixel values. `--diagnostic-jsonl` loads and warms the full
model and additionally emits raw logits and responses. Each consumes one raw
SystemOne request per stdin line, sends diagnostics to stdout and logs to stderr,
and emits an `error` object for rejected lines.

## Focused checks

Default checks run without model weights or CUDA:

```sh
cargo test --locked -p omni-jemm-native -p omni-qwen3-5-native
cargo test --locked -p omni-jev --test frontend
```

After compiling the library, run the shared GPU checks under exclusive GPU
ownership:

```sh
CUA_S1_CUDA_LIB="$PWD/target/release/libqwen3_5_cuda.so" \
  cargo test --release --locked -p omni-qwen3-5-native --test kernels \
  -- --ignored --test-threads=1
CUA_S1_CUDA_LIB="$PWD/target/release/libqwen3_5_cuda.so" \
  cargo test --release --locked -p omni-qwen3-5-native --test vision_kernels \
  -- --ignored --test-threads=1
JEMM_CUDA_LIB="$PWD/target/release/libqwen3_5_cuda.so" \
  cargo test --release --locked -p omni-jemm-native bf16_label_head \
  -- --ignored --test-threads=1
```

Shared kernel tests retain the `CUA_S1_CUDA_LIB` variable. CPU fixtures and
isolated kernels alone do not establish full-model text or multimodal parity.
See the [worker contract](https://github.com/ThinkFlowLab/system1-omni/blob/main/src/models/jemm/README.md)
and [shared Qwen executor](https://github.com/ThinkFlowLab/system1-omni/blob/main/src/models/qwen3_5/native/README.md)
for ownership and supported vision layouts.
