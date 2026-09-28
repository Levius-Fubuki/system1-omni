# Cua-S1 4B 0.2 native text worker

A `/v1/systemone` worker for the `text` adapter in Rust. It answers every request the way the reference worker in [`../text/`](../text/) does (same validation, error bodies, prompt token ids and answer format) and runs the Qwen3.5-4B forward pass on the CUDA kernels in [`src/backends/cuda/qwen3_5/`](../../../backends/cuda/qwen3_5/), with no Python and no PyTorch. Setup, launch and checks are in [`recipe/cua_s1/native.md`](../../../../recipe/cua_s1/native.md).

| File | Contents |
| --- | --- |
| `src/pyjson.rs` | JSON parsing and output that behave like CPython 3.12's `json` module and `repr`, so request errors and answers match the Python worker byte for byte. |
| `src/contract.rs` | The request mapping, prompt text, confidence and answers of `../text/contract.py`. |
| `src/server.rs` | The HTTP worker (`GET /health`, `POST /v1/systemone`), with the reference worker's limits, status codes and error bodies. |
| `src/engine.rs` | Tokenization, the letter rows of the output projection, and scoring. |
| `src/model.rs` | The Qwen3.5 text model: weights, buffers, the layer loop, CUDA graphs and GEMM tuning. |
| `src/cuda.rs` | Loading `libqwen3_5_cuda.so` and the calls into it. |
| `tests/kernels.rs` | GPU checks of the attention and Gated DeltaNet kernels (ignored unless asked for; they need `CUA_S1_CUDA_LIB`). |
| `THIRD_PARTY_NOTICES.md` | The license of the prompt text and fixed values that `src/contract.rs` copies from trycua/cua. |

## How a question is answered

1. The request is parsed and mapped as in the reference worker, and each question's prompt is tokenized with the `tokenizer.json` exported next to the merged weights (the worker checks its SHA-256 against `cua_s1_export.json`).
2. One forward pass runs over the prompt: bfloat16 weights, with the `text` adapter merged into them by `recipe/cua_s1/export_text_merged.py`. The operations follow the Transformers implementation and round to bfloat16 where it does, except inside attention and the Gated DeltaNet prefill, which keep some intermediate results in bfloat16 as FlashAttention and flash-linear-attention do.
3. The final-norm hidden state at the last position is multiplied by the 26 letter rows of the output projection (float32 with float64 accumulation), and a softmax over the question's letters gives the option probabilities.

The whole path from request to answer is in this crate. `src/backends/cuda/qwen3_5/` provides the operations: RMSNorm variants, the Gated DeltaNet convolution, gates and chunked prefill, rotary embedding and attention, and bfloat16 GEMMs through cuBLASLt.

## CUDA library, graphs and GEMM plans

The kernels are built into `libqwen3_5_cuda.so` by `src/backends/cuda/qwen3_5/build.sh` and loaded when the worker starts (`--cuda-lib`, by default next to the executable), so building the crate needs no CUDA toolkit and the workspace checks run anywhere.

Prompts up to `--graph-max-tokens` (2048) run as a CUDA graph captured for their exact length on first use; the 128 most recently used lengths keep theirs. There is no padding, and a graph queues the same kernels with the same GEMM algorithms as the eager pass, so both give bitwise identical results. Longer prompts run eagerly.

Projections that share an input run as one GEMM (`gate_proj` and `up_proj`; `in_proj_qkv`, `in_proj_z`, `in_proj_b` and `in_proj_a`; `q_proj`, `k_proj` and `v_proj`), with their weights stored back to back. GEMM algorithms are tuned at startup for a set of prompt lengths, by timing cuBLASLt's candidates. For prompts up to `--graph-max-tokens`, `--gemm-search` enumerates far more configurations (each algorithm with its tiles, stage counts, swizzles and several split-K factors), times each once and times the 12 fastest properly. `--gemm-plans` keeps the choices in a file, so that later starts reuse them and give the same results. The file records the GPU, the cuBLASLt version, the workspace size and `--graph-max-tokens`; a file that does not match is refused, and removing it tunes again.

## Known differences from the reference worker

Request bodies are handled the same way. The nesting limits copy what the Python worker does on CPython 3.12.13 (parsing fails for arrays nested more than 9,990 deep, and the check after parsing past 969 nested calls). In Python both come from recursion limits, so they move with the interpreter's call stack, and some values need more of it: close to the limit, `NaN` inside 9,985 arrays is "nested too deeply" in Python and "NaN is not valid JSON" here. The status is 400 either way.

Outside the request body the HTTP stacks differ (Starlette and h11 in Python, axum and hyper here):

- A trailing slash (`/v1/systemone/`) gets a 307 redirect from Starlette and a 404 here; a percent-encoded path is decoded by uvicorn and not here.
- FastAPI also serves `/docs`, `/redoc` and `/openapi.json`; they are not served here.
- `HEAD /health` returns 200 here and 405 from FastAPI.
- The HTTP parsers reject different malformed requests (control bytes in header values, a missing `Host`, a `Content-Length` too large to parse), and those rejections are not JSON. A body that fails partway through reading gets a JSON 400 here.
- `GET /health` also reports `"mode": "native"`, and its `dtype` is always `bfloat16`.

The probabilities are not bitwise identical to the reference worker's: the adapter is merged, the kernels differ, and the GEMM algorithms depend on the prompt length and the GPU. They are checked against the float32 worker under the tolerance in [`../README.md`](../README.md#validation).

## Tests

`cargo test -p omni-cua-s1-native` runs the request-handling tests. Three more are ignored unless asked for with `-- --ignored`: a check of float formatting against Python, on a file that `tests/make_float_vectors.py` writes (`CUA_S1_FLOAT_VECTORS`), and the kernel checks in `tests/kernels.rs`, which need a GPU and `CUA_S1_CUDA_LIB` pointing to a built `libqwen3_5_cuda.so`.
