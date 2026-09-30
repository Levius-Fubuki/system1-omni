# Cua-S1 4B 0.2 native text worker

The native worker ([`src/models/cua_s1/native/`](../../src/models/cua_s1/native/)) serves the `text` adapter like the reference worker in [`text.md`](text.md), with the forward pass on the CUDA kernels in [`src/backends/cuda/qwen3_5/`](../../src/backends/cuda/qwen3_5/). It needs an NVIDIA GPU with compute capability 8.0 or newer and was measured on an RTX 6000 Ada (sm_89).

Run the commands from the repository root. The reference worker's setup from `text.md` is needed once, to export the merged weights.

## Build

```sh
src/backends/cuda/qwen3_5/build.sh target/release 89   # needs nvcc and cuBLASLt
cargo build --release --locked -p omni-cua-s1-native
```

Pass your GPU's compute capability to `build.sh` (89 for Ada, 80 for A100, 90 for H100). Only CUDA 13.2 on sm_89 has been run. The worker finds `libqwen3_5_cuda.so` next to its executable; `--cuda-lib` (or `CUA_S1_CUDA_LIB`) points elsewhere.

## Export the merged weights

The worker loads Qwen3.5-4B with the `text` adapter already merged into the bfloat16 weights. With the weights downloaded as in `text.md`, in the reference worker's environment:

```sh
PYTHONPATH=src .venv/bin/python recipe/cua_s1/export_text_merged.py \
  --base weights/Qwen3.5-4B --adapter weights/cua-s1-4b-0.2/text \
  --out weights/cua-s1-4b-0.2-text-merged
```

This writes about 8.5 GB: the checkpoint, the tokenizer files, and `cua_s1_export.json`, which records the revisions and the SHA-256 of `tokenizer.json`. The worker refuses a `tokenizer.json` that does not match it, and warns if the revisions are not the pinned ones.

## Start the worker

```sh
target/release/omni-cua-s1-native --model weights/cua-s1-4b-0.2-text-merged \
  --gemm-plans weights/gemm-plans.json --gemm-search --port 8000
```

The first start tunes the GEMMs for this GPU and writes the choices to `--gemm-plans`; with `--gemm-search` that takes about a minute. Later starts read the file and are ready in about 2 seconds, and give the same results each time. Without `--gemm-search`, tuning takes a few seconds and the worker is somewhat slower for prompts of 200 to 2,000 tokens. The file records the GPU, the cuBLASLt version and `--graph-max-tokens`; the worker refuses a file that does not match and leaves it alone, so keep one per GPU model and CUDA version, and remove it to tune again.

The same options exist as environment variables (`CUA_S1_MODEL`, `CUA_S1_PORT`, `CUA_S1_GEMM_PLANS`, ...; see `--help`), as do the reference worker's request limits and `CUA_S1_API_KEY`. The Rust frontend and the requests are as in `text.md`.

## Tests

Tests without a GPU, then the kernel checks (attention against a float32 kernel, the gated delta rule against a float64 token-by-token reference, and the GEMM plans):

```sh
cargo test -p omni-cua-s1-native
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
  cargo test --release -p omni-cua-s1-native --test kernels -- --ignored
```

## Not covered

- `score` and `noul` questions, and the `multimodal` adapter, as in the reference worker.
- More than one request at a time: the worker answers one decision at a time, like the reference worker.
- GPUs other than sm_89, and CUDA versions other than 13.2, have not been run. The GEMM plans are per GPU and cuBLASLt version.
