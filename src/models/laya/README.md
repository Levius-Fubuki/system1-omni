# LAYA model engine

LAYA is the first planned System1-Omni model. This directory owns its complete request-to-result path: preprocessing, postprocessing, batching policy, state, execution, and backend-specific kernel selection.

GPU operations and kernel implementations belong in [`backends/cuda/`](../../backends/cuda/) and [`backends/metal/`](../../backends/metal/). Setup and usage examples belong in the top-level [`recipe/`](../../../recipe/) directory.

The `omni-laya` crate currently reads and checks the English Laya 0.3.20 checkpoint. `Config::load` validates the architecture and temperatures; `Weights` checks tensor names and shapes and converts FP32, FP16 and BF16 values. `checkpoint_tensors()` lists the 206 expected tensors. Each backend chooses its own storage precision.

Keep checkpoint files unchanged while `Weights` holds a read-only memory mapping. This crate does not yet execute inference.

## CPU checks

The normal workspace tests cover configuration errors, malformed tensors, inventory mismatches and conversion boundaries without downloading weights.

To check the complete checkpoint, use `convaiinnovations/laya` revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` and a Python environment with PyTorch, safetensors and NumPy:

```sh
export LAYA_CHECKPOINT=/path/to/laya/snapshot
export LAYA_WEIGHT_ORACLE=/tmp/laya-weight-oracle.json
python recipe/laya/native/export_weights.py "$LAYA_CHECKPOINT" "$LAYA_WEIGHT_ORACLE"
cargo test --release --locked -p omni-laya --test weights -- --ignored
```

These two CPU tests check all 206 tensor names and shapes, 618 conversion hashes, and the legacy temperature buffer. The normal CI job skips them because it does not download the full checkpoint.

Status: the Python worker below serves LAYA through laya-serve on CPU and Apple Silicon (PyTorch MPS,
validated on an M1 Pro). No native CUDA or Metal backend yet.

## Worker

`worker.py` runs laya-serve (`laya[serve]==0.3.20`) with two changes:

- It binds only after a warmup of every loaded model over short, long and multi-question requests,
  so `/health` never answers for a worker that has not run a forward pass. Without it the first request after `/health`
  returned 200 took ~230 ms against ~45 ms warm on an M1 Pro (MPS); with it, ~73 ms.
- `/health` reports, for each loaded model under `models` and for `LAYA_WORKER_MODEL` at the top level,
  the device, weight and autocast dtypes, the checkpoint and the revision its weights were downloaded
  from, and `device_mismatch` when a model is not on the device `LAYA_DEVICE` asked for. These are
  read on every call, so a fallback to the CPU after startup shows up too.
  laya-serve reports `LAYA_DEVICE` as configured, and laya falls back to CPU with only a printed
  warning. `LAYA_REQUIRE_DEVICE=1` makes the worker exit instead.

Configuration is laya-serve's (`LAYA_HOST`, `LAYA_PORT`, `LAYA_DEVICE`, `LAYA_MODELS`, `LAYA_API_KEY`, ...),
plus `LAYA_WORKER_COMPILE=off|on` and `LAYA_WORKER_WEIGHTS=fp32|fp16`. `on` compiles one-question
requests end to end and, for several questions, only the encoder (Laya's decision head is slower
compiled on MPS); `/health` reports compiled graphs at readiness and now. `fp16` keeps the checkpoint's
fp16 weights (except `act_head`, which Laya feeds fp32 features) instead of the fp32 upcast; it is meant
for MPS, on CPU fp16 is about 2.4x slower. See the
[Apple Silicon recipe](../../../recipe/laya/apple-silicon.md).

```sh
LAYA_DEVICE=mps LAYA_MODELS=english python src/models/laya/worker.py
python -m pytest src/models/laya/tests                     # unit tests, no model
LAYA_CONTRACT=1 python -m pytest src/models/laya/tests     # plus contract tests on CPU, loads the checkpoint
```
