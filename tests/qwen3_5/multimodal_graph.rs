//! CUDA graph regression evidence for the native multimodal language boundary.
//! First run `python3 tests/qwen3_5/prepare_graph_test.py` from the repo root.
//! Run with CUA_S1_TEST_MODEL and CUA_S1_CUDA_LIB pointing at the real checkpoint
//! and CUDA library, using `--ignored --test-threads=1`.
//!
//! The preparation script copies the executor bytes unchanged and appends only
//! root test module wiring, without production test hooks.

#[allow(dead_code)]
#[path = "../../src/models/qwen3_5/native/src/cuda.rs"]
mod cuda;
#[allow(dead_code)]
#[path = "../../src/models/qwen3_5/native/src/inputs.rs"]
mod inputs;
#[allow(dead_code)]
#[path = "model_under_test.rs"]
mod model;
