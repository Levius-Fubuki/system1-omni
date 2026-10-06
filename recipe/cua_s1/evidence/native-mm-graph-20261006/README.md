# Native multimodal language Graph evidence

Core change: `b418230f53a158fcbea706c1815c529a8b3832ec`, based on upstream
`47eff9cdeda01e4847a4fb9634a43f2cab6a233f`. The core PR changes only
`src/models/qwen3_5/native/src/model.rs`; this separate evidence branch holds
tests and reproduction material and must not be merged as the core PR.

`CUA_S1_GRAPH=1` opts both native text and multimodal language forwards into
exact-length replay. Vision remains eager. Every call uploads current token
embeddings, image rows and T/H/W rotary tables before language execution.
Text and multimodal own separate FIFO caches, each bounded to 64 lengths.
Scratch growth clears both caches before freeing buffers. A miss returns its
completed eager result and records a graph for subsequent calls, without
launching it on already-advanced residuals. Capture failure unconditionally
logs to stderr, clears both caches and disables Graph for the model lifetime.
`CUA_S1_GRAPH_TRACE=1` additionally logs successful capture/replay events.

## Root regression tests

The tests generate `tests/qwen3_5/model_under_test.rs`: the complete unchanged
core source bytes followed by root test module wiring. The preparation JSON
checks and records exact byte-prefix identity. No test hook is added to the
production source. The generated file is ignored; its implementation and all
helpers remain under root `tests/`.

With the pinned Cua-S1 multimodal language checkpoint exported using
`recipe/cua_s1/export_multimodal_language.py`, and a compiled ABI5 CUDA library:

```sh
cargo fetch --locked
export CUA_S1_TEST_MODEL=/absolute/path/to/cua-s1-multimodal-language
export CUA_S1_CUDA_LIB=/absolute/path/to/libqwen3_5_cuda.so
python3 tests/qwen3_5/run_graph_regressions.py \
  --log /tmp/native-mm-graph-gpu.log --result /tmp/native-mm-graph-result.json
```

The runner removes `CUA_S1_GRAPH_TRACE`, prepares the source copy, then runs:

```sh
cargo test --offline --release --locked -p omni-qwen3-5-native \
  --test multimodal_graph -- --ignored --nocapture --test-threads=1
```

Three real-GPU regressions cover:

- Exact eager equality on miss/replay, changed IDs/image embeddings/3D
  positions, same-length text interleave, scratch growth and new captures.
- 65 distinct lengths within one unchanged 1024-row scratch, FIFO eviction,
  an oldest-entry hit, recapture of an evicted length and surviving text replay.
- A real CUDA capture whose recording closure returns an error, followed by
  the shared production capture-result handler in both modes; prior eager
  hidden-state preservation, cache clearing, disabled Graph and subsequent
  public eager forwards. This does not inject every possible CUDA failure.

The separate diagnostic gate requires one unconditional failure warning for
each mode. In the preserved logging RED run, all three GPU numerical/lifetime
tests passed but this gate failed because warnings were trace-only. GREEN
passes three tests and the diagnostic gate. This RED is a logging regression,
not a main-baseline numerical failure.

## Native launch

Build the existing ABI5 library and workers from the core checkout:

```sh
src/backends/cuda/qwen3_5/build.sh target/release 89
cargo build --release --locked -p omni-cua-s1-native --bins
CUA_S1_GRAPH=1 CUA_S1_GRAPH_TRACE=1 \
CUA_S1_BASE=/absolute/path/to/Qwen3.5-4B \
CUA_S1_VISION_ADAPTER=/absolute/path/to/cua-s1-4b-0.2/multimodal \
CUA_S1_MODEL=/absolute/path/to/cua-s1-multimodal-language \
CUA_S1_CUDA_LIB="$PWD/target/release/libqwen3_5_cuda.so" \
  target/release/omni-cua-s1-vision
```

For text, use `omni-cua-s1-native` and its separately merged text adapter
export as described in `recipe/cua_s1/native.md`. Leave `CUA_S1_GRAPH` unset
or set it to `0` for the eager control. The HTTP/API and preparation contracts
are unchanged. These commands do not turn the vision encoder into a Graph.

## Recorded scope and environment

One RTX4090, driver 595.71.05, CUDA 13.0.88, Rust 1.98.1; base
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, adapters
`cua-ai/cua-s1-4b-0.2@16818868b0cc7813808aae4e87b417657046ab79`.
Native language uses the appropriate merged BF16 export; native vision retains
BF16 base weights and separate FP32 LoRA. Official control uses pinned upstream
`FourBModel.forward`, BF16, original PEFT, default SDPA, no Graph/compile, TF32 off.
Python 3.12.3, torch 2.14.0+cu130, Transformers 5.17.0, PEFT 0.21.0.

Core fmt, strict workspace Clippy, workspace tests and locked release build
passed; core tests report 89 passed/0 failed/12 ignored. Explicit GREEN GPU
tests report 3 passed/0 failed in 8.56 seconds, with tracing unset and two
unconditional diagnostics. Ignored core tests are not GPU validation.

Media and complete fresh measurement records are published separately from
the core diff. The recordings are illustrative synthetic product-filter
workflows at 1x; the fixed-input comparison is a distinct warmed server-local
HTTP benchmark. Whole-framework versus official speed and Graph-only ON/OFF
results must be reported separately. Capture costs and peak memory are not
measured; the result does not establish general model accuracy or a universal
Graph speedup. Raw failed attempts and protocol revisions are retained.
