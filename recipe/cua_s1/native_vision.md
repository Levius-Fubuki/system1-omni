# Cua-S1 native vision checkpoint inspection

The model-owned Rust module `omni_cua_s1_native::vision` loads and structurally
validates the Qwen3.5-4B vision weights and Cua-S1 4B 0.2 **multimodal** LoRA on
CPU. It keeps the 297 BF16 base tensors and 100 FP32 adapter tensors separate.
The adapter contains 50 A/B pairs, rank 16, alpha 32, and scale 2. No weights are
merged or converted. This CPU inspection API is separate from the
[native CUDA vision encoder and screenshot worker](native_multimodal.md).

Prepare checkpoints from these pinned upstream revisions:

| Checkpoint | Revision | Required files |
| --- | --- | --- |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `config.json` and `model.safetensors`, or `model.safetensors.index.json` and its vision-bearing shards |
| `cua-ai/cua-s1-4b-0.2` multimodal adapter | `16818868b0cc7813808aae4e87b417657046ab79` | `adapter_config.json` and `adapter_model.safetensors` from the multimodal directory |

Verify downloaded files against trusted upstream hashes before loading them.
Structural validation checks names, shapes, dtypes, configuration, and shard
mapping; it does **not** establish cryptographic identity or validate tensor
values. A bare base checkpoint or the `text` adapter is insufficient.

From the repository root, pass exactly the base and multimodal adapter directories:

```sh
cargo run --locked -p omni-cua-s1-native --example inspect_vision -- \
  weights/Qwen3.5-4B weights/cua-s1-4b-0.2/multimodal
```

The example reports the vision configuration, LoRA rank and scale, and tensor
counts. It needs neither a GPU nor a CUDA shared library. The index loader opens
only shards assigned visual tensors; unrelated language tensors in those shards
are ignored. Unexpected visual tensors, incompatible configuration, invalid
safetensors, missing tensors, mismatched index entries, and paths escaping the
checkpoint directory (including symlinks) are rejected.

`VisionCheckpoint::load(base_dir, adapter_dir)` owns the memory maps and caches
validated tensor metadata. `config()` and `adapter()` expose the configuration;
`base_names()` / `adapter_names()` enumerate visual tensor names, and
`base_tensor(name)` / `adapter_tensor(name)` return borrowed safetensors
`TensorView`s with the original bytes. Views cannot outlive the checkpoint.
Accessors do not parse the file headers again.

**Do not modify, replace in place, or truncate checkpoint files while a loaded
checkpoint or any of its views exists.** As with the existing native text
loader, callers must ensure that the memory-mapped files remain immutable.

The supported layout is the pinned 4B configuration: 24 vision blocks, hidden
size 1024, intermediate size 4096, 16 heads, 2304 positions, output width 2560,
3 input channels, patch size 16, temporal patch size 2, spatial merge size 2,
`gelu_pytorch_tanh`, and no DeepStack. Text hidden size must equal vision output
width. LoRA targets both MLP matrices in every vision block and both merger
matrices. Math-changing PEFT options such as DoRA, rsLoRA, biases, target
exclusions, and custom rank/alpha patterns are unsupported.

Run the CPU integration tests with:

```sh
cargo test --locked -p omni-cua-s1-native --test vision_loader
```

Tests use sparse safetensors with real tensor shapes and small sentinel values;
no weight download or multi-gigabyte in-memory tensor allocation is required.
Their independent metadata oracle and provenance live in
[`tests/cua_s1/fixtures/vision/`](../../tests/cua_s1/fixtures/vision/).
