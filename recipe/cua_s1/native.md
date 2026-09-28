# Cua-S1 4B 0.2 native text worker

The native worker ([`src/models/cua_s1/native/`](../../src/models/cua_s1/native/)) serves the `text` adapter like the reference worker in [`text.md`](text.md), with the forward pass on the CUDA kernels in [`src/backends/cuda/qwen3_5/`](../../src/backends/cuda/qwen3_5/). It needs an NVIDIA GPU with compute capability 8.0 or newer and was measured on an RTX 6000 Ada (sm_89).

Run the commands from the repository root. The reference worker's setup from `text.md` is needed once, to export the merged weights and for the checks.

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

## Check it

Tests without a GPU, then the kernel checks (attention against a float32 kernel, the gated delta rule against a float64 token-by-token reference):

```sh
cargo test -p omni-cua-s1-native
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
  cargo test --release -p omni-cua-s1-native --test kernels -- --ignored
```

Accuracy against the float32 reference worker, under the tolerance in [`src/models/cua_s1/README.md`](../../src/models/cua_s1/README.md#validation). First write the reference results with the reference worker's check (`text.md`), once in each dtype:

```sh
mkdir -p parity
.venv/bin/python recipe/cua_s1/compare_text_with_upstream.py --upstream ../cua \
  --base weights/Qwen3.5-4B --adapter weights/cua-s1-4b-0.2/text --device cuda \
  --dtype float32 --no-tf32 --out parity/parity_float32.jsonl
# the same with --dtype bfloat16 --out parity/parity_bfloat16.jsonl
```

Then score the fixed input set eagerly and as served, in two separate processes, and compare:

```sh
for run in 1 2; do
  target/release/omni-cua-s1-native --model weights/cua-s1-4b-0.2-text-merged \
    --gemm-plans weights/gemm-plans.json --score-all tests/cua_s1/data/text_inputs.json > scores-$run.jsonl
done
python3 recipe/cua_s1/check_native.py scores-1.jsonl parity scores-2.jsonl
```

It passes when the largest difference and the top options are within the tolerance, the served results (CUDA graphs) equal the eager ones bit for bit, and the two processes agree bit for bit.

The HTTP behaviour against the reference worker: start the reference worker on port 8001 and the native worker on port 8002, then

```sh
python3 recipe/cua_s1/diff_corpus.py tests/cua_s1/data/text_inputs.json corpus.jsonl 150
python3 recipe/cua_s1/diff_workers.py corpus.jsonl 8001 8002 diff.jsonl
```

sends the fixed input set, edge cases for every error the workers return, and random bodies to both. Errors must be identical (status, content type and body); answers must have the same model, usage, keys and types, and the same choice wherever the reference's top-two margin is at least 0.05.

Latency, through the frontend and directly, as for the reference worker:

```sh
.venv/bin/python recipe/cua_s1/bench_text.py --direct http://127.0.0.1:8000 \
  --frontend http://127.0.0.1:8080 --warmup 3 --repeat 20
```

`--bench <ids.json>... --bench-gap-ms 50` times the forward pass alone, with idle time between passes.

## Results

On one RTX 6000 Ada (48 GB, sm_89), CUDA 13.2, driver 595.91.07, with the pinned revisions:

- Accuracy: over the 16 questions of the fixed input set, the largest difference from the float32 reference worker is 0.0126 (the allowance is 2 × 0.0145 + 0.01 = 0.039), and no top option changes. Served and eager results are bitwise identical, and so are two separate processes with the same plans file.
- HTTP: 567 requests to both workers (the fixed input set, error cases and random bodies) give identical status and body for every error and the same keys, types and choices for every answer; the largest probability difference is 0.029.
- Frontend: all 14 requests return identical status, content type and body bytes directly and through the frontend.
- Startup: 2.0 s to a ready `/health` with a plans file (about 70 s on the first start with `--gemm-search`); the first request after that took 16 ms. The card peaked at 12.6 GiB in use while serving.
- Latency, p50 in milliseconds, one request at a time. `bench_text.py` sends requests back to back, which keeps this card at its 300 W power limit; the forward-only numbers leave 50 ms idle before each pass, closer to decisions that arrive one by one. The reference worker's numbers come from the same two methods.

| Case | Prompt tokens | Reference worker, `bench_text.py` | Native, `bench_text.py` | Reference worker, forward only | Native, forward only |
| --- | --- | --- | --- | --- | --- |
| `one_option` | 139 | 46.2 | 15.7 | 44.6 | 12.2 |
| `fixture_positive` | 218 | 49.2 | 18.8 | 47.4 | 13.8 |
| `fixture_negative` | 292 | 52.7 | 25.6 | 51.1 | 17.0 |
| `max_26_options` | 712 | 109.3 | 50.3 | 103.0 | 33.4 |
| `long_state` | 15446 | 3084.7 | 1364.5 | 3047.6 | 1308.9 |

## Not covered

- `score` and `noul` questions, and the `multimodal` adapter, as in the reference worker.
- More than one request at a time: the worker answers one decision at a time, like the reference worker.
- GPUs other than sm_89, and CUDA versions other than 13.2, have not been run. The GEMM plans are per GPU and cuBLASLt version.
