# Request-local image reuse

`MultimodalEngine.predict` reuses image preprocessing and the adapted visual
encoder for requests containing 2–8 questions. A one-question request keeps the
original path. Every question still has its own prompt, tokenization, candidate
ordering, input embeddings, three-dimensional positions, language forward and
readout. All processed token limits are checked before any model execution.

`predict_reference` retains the original implementation for paired measurements.
`profile_multimodal.py` explicitly selects it so the PR #15 stage baseline remains
reproducible. The model identity, HTTP request/response and usage contract do not
change. This is a Transformers/PEFT execution optimization, not a native CUDA
backend, a custom kernel, batching or a cross-request cache.

## Reuse boundary and tensor ownership

The implementation targets the pinned Transformers 5.17.0 Qwen3.5 model and
processor. A shallow processor copy owns a request-local image-processor proxy.
Its first invocation executes the original image processor; later invocations
return fresh mappings over the same image tensors. The full original processor
still expands image placeholders and tokenizes every question. Neither the
shared processor nor model methods are replaced.

| Value | Contract | Owner and lifetime |
| --- | --- | --- |
| `pixel_values` | Processor-produced patch tensor, unchanged shape/dtype/layout | CPU tensor shared only among prepared inputs in this request; transferred once for the shared visual forward |
| `image_grid_thw` | Integer `[1, 3]` temporal/height/width grid from the same processor call | Shared CPU metadata; transferred for visual encoding and each question's position construction |
| Visual features | Concatenated `get_image_features(...).pooler_output`, `[image_tokens, text_hidden_size]`; active vision LoRA layers included | Request-local device tensor under `no_grad`; converted to each question embedding dtype/device before scatter |
| `input_ids`, `attention_mask`, `mm_token_type_ids` | Full processor outputs, constructed independently for each prompt | Per-question mappings; image reuse does not replace text tokenization |
| `inputs_embeds` | `[1, sequence_length, text_hidden_size]`, new token embeddings with image placeholders filled by `masked_scatter` | Per-question device tensor; feature/placeholder count checked by the model helper |
| `position_ids` | Integer `[3, 1, sequence_length]` from the original `get_rope_index` using that question's tokens, modality types, grid and mask | Per-question device tensor passed explicitly to the full PEFT model; never reused |

No manual allocator, custom CUDA stream, KV-cache reuse or persistent feature
cache is introduced. Ordinary PyTorch references own the tensors; request-local
references disappear on return or exception. The CUDA allocator may retain
reserved memory. The existing server serializes inference. Native-engine
handoff and its stream/allocator contract remain future work.

Measured results and raw evidence: [RTX 4090 experiment](experiments/rtx4090-reuse/README.md).

## Correctness and performance experiment

```sh
PYTHONPATH=src python recipe/cua_s1/benchmark_image_reuse.py \
  --weights /path/to/weights --output /path/to/fresh-results \
  --warmup 5 --runs 2 --iterations 50
```

Use `--correctness-only` for a smoke run; `--case ID` selects performance cases.
Output directories cannot overwrite an existing report. Exceptions leave a
failed report with completed results intact.

Before timing, compare the reference and optimized paths for exact prepared
input fingerprints, language input embeddings, 3D positions, attention masks,
candidate probabilities and full responses. Count actual image preprocessing,
visual and language invocations using temporary test instrumentation. This
instrumentation is removed before timed calls. Correctness cases include PNG,
JPEG, different image sizes, 1 and 26 candidates, reordered candidates, structured
and Unicode labels, accepted text control tokens, eight distinct questions,
reversed question order and consecutive requests changing image/grid then
returning to the original image. CPU tests also cover preparation failures and
token-limit rejection before inference.

The performance matrix retains all 16 original size/length/question-count cases
and adds a distinct-question case. Shuffle case order with recorded seed
20260928. Each case warms both paths, then alternates baseline/reuse and
reuse/baseline for each sample, reversing the starting order for the second run.
Each sample synchronizes CUDA around the whole `predict` call. Reset memory
peaks for each variant invocation and retain raw latencies, execution order,
per-variant allocated/reserved peaks, environment and source revision.

Timing excludes HTTP, queuing, JSON parsing, image decoding, instrumentation,
input fingerprinting and model loading. Results apply to concurrency 1 and
controlled synthetic screenshots; they do not establish GUI task accuracy,
service throughput, p99 or maximum supported input sizes. Reserved-memory peaks
may carry over between variants, so allocated peaks are the useful comparison.
Single-question paths are identical and their measured differences represent
noise. Numerical equality is tested for this pinned environment, not guaranteed
for arbitrary library versions, adapters or devices.
