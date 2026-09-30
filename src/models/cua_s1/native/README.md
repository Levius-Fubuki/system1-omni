# Cua-S1 4B 0.2 native text worker

A `/v1/systemone` worker for the `text` adapter in Rust. It handles requests like the reference worker (same validation, status codes, prompt token ids and answer format) and runs the Qwen3.5-4B forward pass on the CUDA kernels in [`src/backends/cuda/qwen3_5/`](../../../backends/cuda/qwen3_5/), with no Python and no PyTorch. Setup and launch are in [`recipe/cua_s1/native.md`](../../../../recipe/cua_s1/native.md).

| File | Contents |
| --- | --- |
| `src/pyjson.rs` | JSON parsing and `json.dumps` output as the Python worker has them, since structured `state`, `instructions` and `criteria` values reach the prompt as `json.dumps` text. |
| `src/contract.rs` | The request mapping, prompt text, confidence and answers of `../text/contract.py`. |
| `src/server.rs` | The HTTP worker (`GET /health`, `POST /v1/systemone`) with the reference worker's limits and status codes. |
| `src/engine.rs` | Tokenization, the letter rows of the output projection, and scoring. |
| `src/model.rs` | The Qwen3.5 text model: weights, buffers, the layer loop, CUDA graphs and GEMM tuning. |
| `src/cuda.rs` | Loading `libqwen3_5_cuda.so` and the calls into it. |
| `tests/kernels.rs` | GPU checks of the attention and Gated DeltaNet kernels and of the GEMM plans. |

## How a question is answered

1. The request is parsed and mapped as in the reference worker, and each question's prompt is tokenized with the `tokenizer.json` exported next to the merged weights (the worker checks its SHA-256 against `cua_s1_export.json`).
2. One forward pass runs over the prompt: bfloat16 weights, with the `text` adapter merged into them by `recipe/cua_s1/export_text_merged.py`. The operations follow the Transformers implementation and round to bfloat16 where it does, except inside attention and the Gated DeltaNet prefill, which keep some intermediate results in bfloat16 as FlashAttention and flash-linear-attention do.
3. The final-norm hidden state at the last position is multiplied by the 26 letter rows of the output projection (float32 with float64 accumulation), and a softmax over the question's letters gives the option probabilities.

## CUDA library, graphs and GEMM plans

`libqwen3_5_cuda.so` is built by `src/backends/cuda/qwen3_5/build.sh` and loaded when the worker starts, so building the crate needs no CUDA toolkit.

Prompts up to `--graph-max-tokens` (2048) run as a CUDA graph captured for their exact length on first use, and the 128 most recently used lengths keep theirs. A graph queues the same kernels with the same GEMM algorithms as an eager pass, so both give bitwise identical results. Longer prompts run eagerly.

Projections that share an input run as one GEMM. GEMM algorithms are tuned at startup for a set of prompt lengths by timing cuBLASLt's candidates; `--gemm-search` times far more configurations for prompts up to `--graph-max-tokens`. `--gemm-plans` keeps the choices in a file, so later starts reuse them and give the same results. The file records the GPU, the cuBLASLt version, the workspace size and `--graph-max-tokens`; a file that does not match is refused, and removing it tunes again.

## Differences from the reference worker

- The probabilities are not bitwise identical: the adapter is merged, the kernels differ, and the GEMM algorithms depend on the prompt length and the GPU. They are held to the tolerance in [`../README.md`](../README.md#validation).
- Error messages quote names as Python's `repr` does, except that non-printable characters outside ASCII, such as U+00A0 or U+200B, are written as they are instead of escaped.
- The HTTP stacks differ outside the request body (Starlette and uvicorn there, axum and hyper here): trailing slashes, percent-encoded paths, `HEAD /health`, FastAPI's `/docs`, and which malformed HTTP requests are refused.
- `GET /health` also reports `"mode": "native"`, and its `dtype` is always `bfloat16`.

## Tests

`cargo test -p omni-cua-s1-native` runs the request-handling tests. The kernel checks in `tests/kernels.rs` need a GPU and run with `-- --ignored` and `CUA_S1_CUDA_LIB` pointing to a built `libqwen3_5_cuda.so`.
