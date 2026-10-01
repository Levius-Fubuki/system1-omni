# Native multimodal language-boundary evidence

Runtime commit: `93a6cb0e874a72efa47d9c8492d69335e29cc697`, based on upstream `566dec1bbd99329dc534a3af4bdaa4b91258beda`.
Reference exporter: [#53](https://github.com/ThinkFlowLab/system1-omni/pull/53), `1b64fa2ceb0a82b6a66a69ecdc9bc5cc1b1a0b66`.

[Download the full evidence archive](native-boundary-evidence.tar.gz) (15,319,545 bytes).
SHA256: `5a828a463e67a402760ba8a5b85b0aabd2e949f3da3444ae1b5fdf22dff95321`.
Its `SHA256SUMS.json` hashes all payload files. All six runtime source hashes were checked against the source actually executed on the server; see [source-manifest.json](source-manifest.json).

## Environment and checks

RTX 4090 24 GiB, driver 595.71.05, nvcc 13.0.88 targeting sm_89; Torch 2.14.0+cu130, Transformers 5.17.0, PEFT 0.21.0. TF32 was disabled for reference controls. The existing CUDA backend was rebuilt without kernel or ABI changes (ABI 2).

The following passed locally and on the GPU server:

```sh
cargo fmt --all --check
cargo clippy --workspace --locked --all-targets -- -D warnings
cargo test --workspace --locked
cargo build --workspace --release --locked
```

Workspace tests: 25 passed; three hardware tests are ignored by default. Those three were separately executed on the RTX 4090 and passed:

```sh
CUA_S1_MODEL=/path/to/multimodal-language-merged \
CUA_S1_CUDA_LIB=/path/to/libqwen3_5_cuda.so \
  cargo test --release --locked -p omni-cua-s1-native --test multimodal -- --ignored
CUA_S1_CUDA_LIB=/path/to/libqwen3_5_cuda.so \
  cargo test --release --locked -p omni-cua-s1-native --test kernels -- --ignored
```

The multimodal regression covers feature insertion, disjoint placeholder spans, explicit three-axis positions, rejection before GPU mutation, text-position restoration, a 1025-token scratch-buffer growth, and subsequent short/long text and multimodal calls. Independent agent review found no actionable issue; it ran CPU checks and reviewed the GPU evidence but did not independently execute a GPU run.

## Numerical result

The example replayed eight exported questions twice (16 native forwards); repeated hidden states were bitwise equal. The eight fixtures cover sequence lengths 216–518, 64–300 image tokens, varied image shapes, JPEG, structured/non-ASCII questions and 1/3/26 candidates.

At the **fixed exported BF16 vision/input-embedding boundary**, an unmerged FP32 language control gave:

| Measurement | Maximum |
|---|---:|
| BF16 reference probability error versus FP32 | 0.00736469030380249 |
| Native probability error versus FP32 | 0.0015364587306976318 |
| Declared allowance: twice BF16 reference error plus 0.01 | 0.024729380607604982 |
| Native hidden-state drift versus merged BF16 control | 0.5 |
| Native candidate-logit drift versus merged BF16 control | 0.258575439453125 |

All eight top choices matched. Every multi-option FP32 margin exceeded 0.05 (minimum about 0.476), so the declared top-choice gate applies to all of them. [verified-summary.json](verified-summary.json) contains per-question results.

This validates the native language boundary on one GPU and a small synthetic fixture set. It is not an end-to-end FP32 vision comparison, a broad accuracy benchmark, a performance claim, or bitwise parity with Transformers. Native vision/preprocessing, image HTTP serving, video, padding and batching remain outside this change. The native readout uses FP32 letter rows with FP64 accumulation, consistent with the existing text engine.

## Reproduce

1. Check out the runtime commit, build the CUDA library for your GPU and the `multimodal_boundary` example using `recipe/cua_s1/native.md`.
2. Extract the archive. Its `reference/` includes every fixed boundary tensor and synthetic source image; verify it with the archived `verify_multimodal_reference.py`. The independent second export manifest and exact-equality verification are also archived: eight questions, 112 tensors and 25 payloads. The original exporter CPU test log is included; all 11 tests were rerun successfully on October 1.
3. Prepare the pinned base and multimodal-adapter weights and reference Python environment. Edit the three host-specific `root`, `bundle` and `weights` paths in [prepare_language.py](prepare_language.py). Choose a new output root; set `bundle` to the extracted `reference/`. Run with `PYTHONPATH=src` from the reference checkout. This exports only the merged language weights and computes merged BF16 and unmerged FP32 language controls. It asserts that token embedding plus image feature insertion exactly reconstructs exported input embeddings.
4. Run the native example with those language weights, the verified bundle, and a new `<root>/native.json`:

```sh
CUA_S1_CUDA_LIB=/path/to/libqwen3_5_cuda.so \
  cargo run --release --locked -p omni-cua-s1-native \
  --example multimodal_boundary -- \
  /path/to/weights-language-merged /path/to/reference /path/to/root/native.json
python3 verify_results.py /path/to/root
```

Use Python 3.10 or later for the numerical verifier. To recompute the archived gates without a GPU, run [verify_results.py](verify_results.py) on the extracted `native-boundary-evidence/` directory. The archive includes native/control JSON, scripts, reference tensors, source hashes, build/test logs and numerical verification logs; model weights are not included.
