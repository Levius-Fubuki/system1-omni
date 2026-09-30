# Cua-S1 multimodal reference tensors

This offline recipe observes the Transformers/PEFT implementation merged in
[#12](https://github.com/ThinkFlowLab/system1-omni/pull/12). It provides concrete
inputs for the native vision integration discussed in
[#10](https://github.com/ThinkFlowLab/system1-omni/issues/10#issuecomment-5862276038).
The adapter stays **unmerged**, including the vision LoRA. It does not change the
worker or add a native vision implementation.

## Reproduce

Use Python 3.12 and an NVIDIA GPU with enough memory for Qwen3.5-4B and the full
reference output head. The validated setup is an RTX 4090 (24 GiB), driver
595.71.05, Torch 2.14.0+cu130 and the packages in
[requirements-reference.txt](requirements-reference.txt). Use a fresh environment
without flash-linear-attention, causal-conv1d or explicitly enabled hub kernels.
The model chooses its default attention implementation; the manifest records the
actual text and vision choices. These are reference exports, not timed benchmarks.

From the repository root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.14.0 torchvision==0.29.0 \
  --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -r recipe/cua_s1/requirements-reference.txt

git clone https://github.com/trycua/cua.git /tmp/cua-reference
git -C /tmp/cua-reference checkout 0e75660ce4c2edda519e0c795fa3ad98abf4e76f
.venv/bin/python /tmp/cua-reference/libs/cua-s1/ci/fetch_pinned_weights.py \
  --dest /tmp/cua-weights
cp /tmp/cua-reference/libs/cua-s1/ci/weights.lock.json /tmp/cua-weights/weights.lock.json

PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python recipe/cua_s1/export_multimodal_reference.py \
  --weights /tmp/cua-weights --output /tmp/cua-reference-a
PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python recipe/cua_s1/export_multimodal_reference.py \
  --weights /tmp/cua-weights --output /tmp/cua-reference-b
PYTHONPATH=src .venv/bin/python recipe/cua_s1/verify_multimodal_reference.py \
  /tmp/cua-reference-a --compare /tmp/cua-reference-b
```

The model verifies the upstream manifest's pinned SHA-256 and the size/hash of
every loaded base/adapter file before loading. For existing verified weights,
provide the same directory layout and the upstream manifest next to `Qwen3.5-4B/`.
The model pins are recorded in [the model contract](../../src/models/cua_s1/README.md#pinned-revisions).

Each output must be a new directory. An existing output is rejected before model
loading. `manifest.json` is written last; a directory without it is an incomplete
export. Save generated bundles outside the checkout. A bundle is about 44 MiB on
the validated setup and contains no checkpoint weights.

## Fixed inputs

The recipe generates redistributable synthetic account-settings screens using
Pillow; it needs no user screenshots or external images. The image bytes and the
full inline-image request JSON are included in `inputs/`.

| Case | Original image, width × height | Format | Questions / options |
| --- | --- | --- | --- |
| small | 320 × 240 | PNG | 1 / 3 |
| wide | 640 × 320 | PNG | 1 / 3 |
| portrait | 320 × 640 | PNG | 1 / 3 |
| jpeg | 640 × 480 | JPEG | 1 / 3 |
| single-option | 320 × 240 | PNG | 1 / 1 |
| 26-options | 256 × 256 | PNG | 1 / 26 |
| two-questions | 320 × 240 | PNG | 2 / 3 and 2 |

The second question includes structured, non-ASCII instructions and an object and
null criterion. Each question is an independent unpadded forward over the same
request image, preserving the #12 behavior and option order. Identical image bytes
must give identical preprocessing and adapted vision features across questions.

## Bundle format and native boundary

`manifest.json` uses schema `cua-s1-multimodal-reference-v1`. It records pinned
revisions, verified weight-manifest hash, package/GPU/CUDA/driver information,
source hashes, execution configuration, file sizes and SHA-256s. Each ordered
question entry contains its prompt, request/image paths, option keys, final answer,
and a safetensors file. Every tensor has its shape, dtype and SHA-256 of contiguous
CPU bytes, with no dtype conversion. BF16 bytes are retained directly.

The bundle also includes base, processor and adapter JSON configurations in
`configs/`. Tensor dimensions below use `S` for prompt length, `P = T×H×W` for raw
patch count and `I = P / merge_size²` for merged image-token count. In this pinned
model, hidden width `D = 2560`, patch size 16, temporal patch size 2 and merge size 2.
A still image has `T = 1`; the processor duplicates its frame for temporal patches.

| Tensor | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `pixel_values` | `[P, 1536]` | float32 | Processor-normalized, flattened image patches; not an RGB image or NCHW tensor |
| `image_grid_thw` | `[1, 3]` | int64 | Raw temporal/height/width patch grid before spatial merging |
| `input_ids` | `[1, S]` | int64 | Fully expanded chat prompt, including `I` image placeholders |
| `attention_mask` | `[1, S]` | int64 | All ones: one unpadded prompt |
| `mm_token_type_ids` | `[1, S]` | int64 | Text 0, image 1 |
| `image_features` | `[I, D]` | bfloat16 | Actual adapted vision output after the merger, before insertion into the language input |
| `image_token_indices` | `[I]` | int64 | Positions where the image features replace token embeddings, in sequence order |
| `inputs_embeds` | `[1, S, D]` | bfloat16 | Actual language-model input after image insertion |
| `position_ids` | `[3, 1, S]` | int64 | Actual temporal/height/width M-RoPE positions at the language-model boundary |
| `rope_deltas` | `[1, 1]` | int64 | `max(position_ids) + 1 - S`, recorded from the model |
| `last_hidden_state` | `[1, D]` | bfloat16 | Final normalized language hidden state at the last sequence position |
| `candidate_token_ids` | `[C]` | int64 | A–Z letter ids, aligned with `option_keys` |
| `candidate_logits` | `[C]` | bfloat16 | Final-position logits for the candidates |
| `candidate_probabilities` | `[C]` | float32 | GPU fp32 softmax of the candidate logits |

Temporary hooks observe the **actual** PEFT forward, including the vision merger
and the language input/last hidden state. They return no replacement outputs and
are removed in `finally`, including on failure. Position ids come from the model's
forward, rather than a separately reimplemented M-RoPE algorithm. Each captured
readout is checked against a second call to the ordinary #12 `score()` method;
the probabilities must match exactly before an export is marked complete.

For a native language-path test, use `inputs_embeds`, `position_ids`, the base text
configuration and the multimodal-adapted language weights. The text adapter from
#19 is a different adapter. For a native vision test, start from `pixel_values` and
`image_grid_thw` and compare with `image_features`. The exported embedding contains
the unmerged reference adapter's BF16 rounding behavior; merging LoRA may change
that behavior and needs its own declared tolerance.

## Verification and limits

The verifier checks file and tensor fingerprints, exact keys/shapes/dtypes,
finite values, patch/merge counts, image-placeholder ordering, feature insertion,
position delta, candidate ordering, answer reconstruction and same-image reuse.
CPU softmax reconstruction allows absolute error `1e-7` with zero relative
allowance because CPU and GPU reduction implementations can round differently.
`--compare` additionally requires the two manifests to match exactly, including
all file and tensor hashes, execution settings and source/environment metadata.
The safetensors header key order is not part of a portable format guarantee.

The checked [validation summary](reference-validation.json) records two
independent exports on the RTX 4090: 8 questions, 112 tensors, 25 files per bundle,
identical raw tensor bytes and identical ordinary/captured probabilities. CPU
softmax reconstruction differed by at most `3.73e-9`. This verifies the reference
artifact pipeline; it makes no native accuracy or speed claim. Bitwise equality
has only been checked on the recorded software/GPU setup. Other GPUs, CUDA builds,
FP32 controls, video, multiple images, padding/batching and Metal are unverified.
When evaluating a new native implementation, declare numerical tolerances before
comparison; the `--compare` mode is for repeating the same reference export.

Run the focused tests without weights or a GPU:

```sh
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install Pillow==11.3.0 numpy==2.5.3 safetensors==0.8.0 pytest==9.1.1
PYTHONPATH=src python -m pytest tests/cua_s1/test_export_multimodal_reference.py -q
```

Tests cover deterministic cases, existing-output rejection, raw BF16 bytes,
actual hook capture and cleanup on success/failure, safetensors round trips,
corruption/unlisted-file detection, safe paths, missing tensor rejection and CPU
softmax roundoff versus drift. Without Torch, the two image/output tests run and tensor tests are skipped;
the recipe CI installs CPU Torch and executes the complete suite.
