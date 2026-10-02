# Cua-S1 native screenshot worker

`omni-cua-s1-vision` serves the same screenshot `/v1/systemone` contract as the
[Python multimodal model](../../src/models/cua_s1/multimodal/model.py). Runtime request handling, PNG/JPEG decoding,
RGB preprocessing, the 24-block vision encoder, 2×2 merger, image insertion,
three-axis positions, language forward and candidate scoring execute in Rust/CUDA.
Python is used only for the one-time language export and reference validation.

The worker combines the checkpoint loader (#59), RGB processor (#63) and language
input boundary (#56). These dependencies are included in the integration branch;
merging this branch does not require users to manually combine working trees.
The existing `omni-cua-s1-native` binary continues to serve the text adapter.

## Build and load

Download the pinned base and adapter revisions using the commands in the
[text recipe](text.md). Use its Python 3.12 environment with the additional image
packages and save the upstream lock beside the base directory:

```sh
.venv/bin/python -m pip install -r recipe/cua_s1/requirements-native-validation.txt
curl --fail -L https://raw.githubusercontent.com/trycua/cua/0e75660ce4c2edda519e0c795fa3ad98abf4e76f/libs/cua-s1/ci/weights.lock.json \
  -o weights/weights.lock.json
```

The exporter and native worker verify this lock's exact trusted SHA-256. Export
the language weights once to a new directory:

```sh
PYTHONPATH=src .venv/bin/python recipe/cua_s1/export_multimodal_language.py \
  --base weights/Qwen3.5-4B \
  --adapter weights/cua-s1-4b-0.2/multimodal \
  --out weights/cua-s1-multimodal-language
src/backends/cuda/qwen3_5/build.sh target/release 89
cargo build --release --locked -p omni-cua-s1-native --bins --examples
CUA_S1_BASE=weights/Qwen3.5-4B \
CUA_S1_VISION_ADAPTER=weights/cua-s1-4b-0.2/multimodal \
CUA_S1_MODEL=weights/cua-s1-multimodal-language \
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
  target/release/omni-cua-s1-vision
```

`CUA_S1_HOST` and `CUA_S1_PORT` default to `127.0.0.1:8000`. `/health` reports ready
after loading. The library and worker must both use **CUDA ABI 4**; rebuild both.
RTX 4090 / sm_89 is validated. Tensor-core operations require sm_80 or newer.

At startup the worker verifies the upstream lock's trusted SHA-256, all pinned
base/adapter files, and the language export's per-file hashes before CUDA loading.
The export manifest is locally generated, trusted provenance; it is not a signature
of an externally supplied checkpoint. Keep all checkpoint and manifest files
immutable during inference. Low-level `VisionCheckpoint` and `VisionModel` APIs
perform structural validation; callers of those APIs own artifact provenance.

The original vision tensors stay BF16 and the 50 rank-16 LoRA pairs stay FP32.
Each LoRA branch uses FP32 GEMMs, scaling and addition before BF16 rounding.
Language LoRA uses the existing merged BF16 path. The encoder uses bounded-memory,
bidirectional attention; it never allocates a full image-token-square score matrix.
Multimodal execution remains eager. This is a correctness implementation, with no
throughput or speedup claim.

## Request and API behavior

The screenshot contract accepts one inline PNG/JPEG, up to eight choice questions,
1–26 options each, a body up to 8 MiB and compressed image up to 4 MiB. Images are
bounded to 2048 per side, 1,048,576 pixels, and aspect ratio 200:1; animated PNGs
are rejected. PNG metadata allocations are bounded before decoding. Six 16-bit PNG
modes, including grayscale transparency, have Pillow-compatible RGB conversions.

All questions and tokenized prompt lengths (at most 4096) are checked before
vision execution. A request preprocesses and encodes its image once, then scores
its questions independently. Features are request-local. Changed images, prompt
lengths and language/vision scratch reuse are included in validation. Malformed
JSON returns 400; invalid screenshot/question inputs return 422, body-limit
violations 413, and inference failures 500, all with a `detail` envelope.

`VisionEngine::prepare` and `predict` accept decoded RGB requests. The lower-level
`VisionModel::forward(&ProcessedImage)` returns row-major BF16 `[image_tokens,2560]`.
`forward_with_trace` exposes synchronized intermediate downloads for diagnostics.
`prepare_prompt` constructs the token sequence and the T/H/W positions. Video,
padding, batching and CUDA Graph capture of vision are outside this worker.

JPEG decoding uses the Rust decoder; it is not byte-identical to Pillow/libjpeg.
The validation JPEG differed by at most 3 intensity levels per channel and its
complete native probabilities passed the same acceptance test. PNG fixture pixels
were identical. CPU preprocessing parity for identical decoded RGB remains the
separate [RGB processor guarantee](native_image_preprocess.md).

## Reproduce GPU alignment

Generate the standard and independently added boundary requests with Pillow 11.3.0:

```sh
.venv/bin/python recipe/cua_s1/native_vision_cases.py /tmp/cua-standard
.venv/bin/python recipe/cua_s1/native_vision_cases.py /tmp/cua-boundary --boundary
PYTHONPATH=src .venv/bin/python recipe/cua_s1/validate_native_vision.py \
  --base weights/Qwen3.5-4B --adapter weights/cua-s1-4b-0.2/multimodal \
  --requests /tmp/cua-standard --out /tmp/cua-standard-controls
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
  target/release/examples/native_vision \
  weights/Qwen3.5-4B weights/cua-s1-4b-0.2/multimodal \
  weights/cua-s1-multimodal-language /tmp/cua-standard-controls /tmp/cua-native.json
python recipe/cua_s1/verify_native_vision.py \
  /tmp/cua-standard-controls/manifest.json /tmp/cua-native.json
```

Repeat with `/tmp/cua-boundary` and separate output paths. Output directories/files
must not exist. Reference controls execute the **full vision and language path**
with unmerged BF16 and FP32 models, TF32 disabled; they do not feed BF16 reference
image embeddings into the native language model. The native replay starts from
request PNG/JPEG data and compares independently produced token IDs, image grids,
positions, candidate probabilities and repeated forwards. Stage feature errors
are reported separately. The acceptance rule is unchanged: maximum native
probability error over a set ≤ twice the maximum BF16 reference error plus 0.01;
top choices match where the FP32 top-two margin is at least 0.05.

Additional CUDA regressions:

```sh
.venv/bin/python tests/cua_s1/test_native_vision_cuda.py \
  target/release/libqwen3_5_cuda.so
CUA_S1_MODEL=weights/cua-s1-multimodal-language \
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
  cargo test --release --locked -p omni-cua-s1-native -- --ignored
```

The optional `vision_stage` example emits each vision block's BF16 tensors. These
are diagnostics, not a bitwise-equivalence claim. See the recorded results in
[native vision validation](experiments/native-vision/README.md).
