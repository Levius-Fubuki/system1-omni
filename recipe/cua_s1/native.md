# Cua-S1 4B 0.2 native text worker

The native worker ([`src/models/cua_s1/native/`](../../src/models/cua_s1/native/)) serves the `text` adapter like the reference worker in [`text.md`](text.md), with the Qwen3.5-4B forward pass on the CUDA kernels in [`src/backends/cuda/qwen3_5/`](../../src/backends/cuda/qwen3_5/) and no Python or PyTorch. It needs an NVIDIA GPU with compute capability 8.0 or newer; only an RTX 6000 Ada (sm_89) with CUDA 13.2 has been run.

Run the commands from the repository root. Pass your GPU's compute capability to `build.sh` (89 for Ada, 80 for A100, 90 for H100); the worker finds `libqwen3_5_cuda.so` next to its executable, or at `CUA_S1_CUDA_LIB`:

```sh
src/backends/cuda/qwen3_5/build.sh target/release 89   # needs nvcc and cuBLASLt
cargo build --release --locked -p omni-cua-s1-native
```

The worker loads the weights with the `text` adapter merged in. With the reference worker's environment and weights from `text.md`, export them once (about 8.5 GB):

```sh
PYTHONPATH=src .venv/bin/python recipe/cua_s1/export_text_merged.py \
  --base weights/Qwen3.5-4B --adapter weights/cua-s1-4b-0.2/text \
  --out weights/cua-s1-4b-0.2-text-merged
```

Start the worker (`CUA_S1_HOST` and `CUA_S1_PORT` default to `127.0.0.1` and `8000`), then the frontend and requests as in `text.md`:

```sh
CUA_S1_MODEL=weights/cua-s1-4b-0.2-text-merged target/release/omni-cua-s1-native
```

For the local CUDA Graph experiment, also set `CUA_S1_GRAPH=1`. The first use of
each exact prompt length warms the GEMM plans and captures the forward pass.
The first call returns the eager result; later requests replay it after fresh
token embedding. The graph contains the language layers, which update residuals
in place, so a cache miss must not replay those layers over its eager result.
At most eight lengths are cached. Growing the scratch allocation clears the captures before freeing
their buffers. Capture adds first-use latency; leave the variable unset to use
the eager control. Rebuild both the worker and CUDA library together (ABI 4).
If capture fails, the worker returns the completed eager result and disables
Graph capture/replay for its remaining lifetime, logging the failure to stderr.

Each question runs one forward pass over its prompt, eagerly by default or through exact-length CUDA Graph replay when enabled; the final hidden state at the last position times the 26 letter rows of the output projection gives the option probabilities. The probabilities are not bitwise identical to the reference worker's, since the adapter is merged and the kernels differ; they are held to the tolerance in [`src/models/cua_s1/README.md`](../../src/models/cua_s1/README.md#validation). Error messages are worded differently, and bodies nested more than 127 levels deep are refused.

The request tests need no GPU; the kernel tests compare attention and the chunked Gated DeltaNet prefill with float64 references:

```sh
cargo test -p omni-cua-s1-native
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
  cargo test --release -p omni-cua-s1-native --test kernels -- --ignored
```

For CPU-only structural inspection of the separate vision base weights and
multimodal adapter, see [native vision checkpoint inspection](native_vision.md).

For the separate decoded-RGB8 CPU preprocessing API and example, see
[native image preprocessing](native_image_preprocess.md).
## Multimodal language boundary

`Model::forward_multimodal` consumes token IDs, adapted BF16 image features
`[image_tokens, hidden_size]`, their sorted placeholder indices, and the T/H/W
slices of int64 `position_ids [3, 1, sequence]`. It returns the last position's
final-normalized hidden state, like `Model::forward`.

Inputs are one unpadded prompt. Every image placeholder must have exactly one
feature row; all other rows come from the token embedding table. The caller
calculates positions and runs the image processor, vision tower and vision LoRA.
Positions must be nonnegative and below `max_position_embeddings`. The language
path uses Qwen3.5's interleaved MRoPE sections, not three contiguous rotary blocks.
Text calls and their captured graphs use immutable text-position tables;
multimodal calls use separate device tables, so returning to text requires no
host table rebuild or restoration copy. Multimodal calls execute eagerly even
when `CUA_S1_GRAPH=1`. The extra tables use
`4 * scratch_capacity * rotary_half` bytes (2 MiB at 16,384 rows).

This is a Rust model API for integrating a vision producer. The HTTP worker
above continues to serve the text adapter. For native vision and image HTTP requests use the separate
[screenshot worker](native_multimodal.md). Padding, video and batching remain unsupported.

### Prepare a matching language checkpoint

The language weights must contain the **multimodal** adapter, not the `text`
adapter. In the pinned reference environment, with upstream-verified weights,
export just the merged language model (about 7.5 GB) to a new directory:

```sh
PYTHONPATH=src .venv/bin/python recipe/cua_s1/export_multimodal_language.py \
  --base weights/Qwen3.5-4B --adapter weights/cua-s1-4b-0.2/multimodal \
  --out weights/cua-s1-4b-0.2-multimodal-language-merged
```

No `cua_s1_export.json` text-worker marker is created. The low-level `Model` API
does not verify checkpoint provenance; retain the export metadata and use the
matching adapter for the supplied features. Standalone language safetensors
names and the existing full-model prefixes are supported.

### Replay a reference boundary

This optional example consumes the `cua-s1-multimodal-reference-v1` format
from [#53](https://github.com/ThinkFlowLab/system1-omni/pull/53), which is still
open. The exporter and checksum verifier are not yet available on `main`.
Use a separate checkout of exporter revision
`1b64fa2ceb0a82b6a66a69ecdc9bc5cc1b1a0b66` to generate and verify the bundle;
the Rust model API itself does not depend on that PR being merged.
Its eight questions include different image grids and question lengths, JPEG,
structured/non-ASCII text, and 1/3/26 candidates. Then run:

```sh
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
  cargo run --release --locked -p omni-cua-s1-native \
  --example multimodal_boundary -- \
  weights/cua-s1-4b-0.2-multimodal-language-merged \
  /path/to/verified-reference-bundle /tmp/native-language.json

CUA_S1_MODEL=$PWD/weights/cua-s1-4b-0.2-multimodal-language-merged \
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
  cargo test --release --locked -p omni-cua-s1-native \
  --test multimodal -- --ignored

# Compare graph misses, hits, eviction and scratch growth with eager hidden states.
CUA_S1_MODEL=$PWD/weights/cua-s1-4b-0.2-multimodal-language-merged \
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
  cargo test --release --locked -p omni-cua-s1-native \
  --lib graph_tests::graph_misses_hits_eviction_growth_and_multimodal_match_eager -- --ignored
```

The example checks repeated native hidden-state equality and writes last hidden
states, candidate logits and probabilities. It uses the text engine's FP32
letter-row readout with FP64 accumulation. Output must be a new file. Verify
bundle integrity before invoking the example; it checks tensor shapes and input
contracts but is not the bundle checksum verifier.

For accuracy validation, compare against an unmerged FP32 **language** control
with TF32 disabled, feeding the same fixed exported embeddings and positions.
Use the [declared native tolerance](../../src/models/cua_s1/README.md#validation):
maximum probability error over the set must be at most twice the BF16 reference
error plus 0.01, and the top option must match for FP32 margins at least 0.05.
The FP32 control starts after the BF16-exported vision boundary; it does not
validate a full FP32 vision pipeline. Native kernel and LoRA-merge rounding can
change hidden states and logits; bitwise equality to Transformers is not claimed.
