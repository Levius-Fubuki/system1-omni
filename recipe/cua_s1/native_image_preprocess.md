# Native CPU image preprocessing

The native crate exposes `image_preprocess::preprocess_rgb8(width, height, rgb)`
for **already decoded, interleaved RGB8** data. It prepares the image tensor for
the fixed Qwen3.5-4B / Cua-S1 4B processor. It does not decode PNG/JPEG, fetch
URLs, handle HTTP requests, run the vision encoder, or use a GPU. The existing
native text worker remains separate.

The input must contain exactly `width * height * 3` bytes, in row-major RGB
order. Both dimensions must be nonzero and at most 2048; the area must be at
most 1,048,576 pixels and the aspect ratio at most 200. The library checks
geometry, lengths, and allocation arithmetic before creating image buffers.
These are input limits; smart resize can produce a side longer than 2048 for
very narrow inputs.

The fixed processor uses a factor of 32, minimum area 65,536 and maximum area
16,777,216. Smart resize follows Python ties-to-even rounding and floating-point
square-root scaling with floor/ceil. Resampling matches the CPU torchvision
uint8 bicubic antialias path: Keys cubic coefficient `a = -0.5`, float64 weights,
per-axis int16 fixed-point coefficients, horizontal then vertical passes, and
rounding/clamping to uint8 after each pass. Unchanged axes bypass resampling.
The maximum-area downscale branch is retained for parity with the processor,
although the smaller input cap makes it unreachable through this API.

`ProcessedImage` contains:

- `pixel_values: Vec<f32>`, contiguous `[patches, 1536]` values normalized as
  `(pixel - 127.5) / 127.5` using float32 operations.
- `image_grid_thw: [usize; 3]`, equal to `[1, resized_height / 16, resized_width / 16]`.
- `resized_width` and `resized_height`.
- `image_tokens()`, the patch count divided by four for the 2×2 spatial merge.

Packing order is `block_y, block_x, merge_y (2), merge_x (2), channel (3),
temporal repeat (2), patch_y (16), patch_x (16)`. Each single image is repeated
across the two temporal positions. There is no video input support.

## Run without a GPU

From the repository root, provide a raw RGB8 file and its dimensions:

```sh
cargo run --locked -p omni-cua-s1-native --example preprocess_image -- \
  256 256 image.rgb pixel_values.f32
```

The example prints the tensor shape, grid, resized dimensions and image token
count. The optional fourth argument writes every output float as little-endian
float32, with no header. It validates argument count, decimal dimensions and
exact file length, and bounds the input read. No model weights, Python, CUDA
library or image decoder are needed for this command.

## Reference and validation

Reference hashes are produced by the actual Hugging Face `AutoImageProcessor`
on CPU, with Python 3.12 and these exact package pins: Transformers 5.17.0,
PyTorch 2.14.0, torchvision 0.29.0, NumPy 2.5.3 and Pillow 11.3.0. The processor
configuration is the Qwen3.5-4B file at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (see the fixture manifest for its URL
and SHA-256). Fixtures and regeneration instructions live in
[`tests/cua_s1/fixtures/image_preprocess/`](../../tests/cua_s1/fixtures/image_preprocess/).

```sh
cargo test --locked -p omni-cua-s1-native --test image_preprocess
cargo test --release --locked -p omni-cua-s1-native --test image_preprocess
```

The tests compare input hashes, shape/grid metadata and SHA-256 of **every
little-endian float32 output byte** for 14 deterministic images. Cases include
tiny images, noise, ramps, checkerboards, ties-to-even dimensions, unchanged
axes, one- and two-axis resizes, extreme aspect ratios and the input area cap.
Additional checks cover all RGB byte values, channel/temporal/patch order,
constant images, inclusive limits and malformed geometry/buffers. This
validates the pinned CPU preprocessing behavior; it does not establish CUDA
preprocessing, image decoding, vision inference, or end-to-end model parity.

The implementation adapts upstream algorithms; retained attributions and
license texts are in
[`THIRD_PARTY_NOTICES.md`](../../src/models/cua_s1/native/THIRD_PARTY_NOTICES.md).
