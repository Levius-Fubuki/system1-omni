# JEMM native worker

JEMM uses one Qwen3.8-27B prefill for each question, with all candidates in an
ordered prompt. The Rust worker implements the pinned SystemOne reference
contract and supports text and up to four PNG, JPEG or WebP images supplied as
base64 strings or base64 data URLs. Images appear before the question text and
are shared by every question in the request.

## Pinned artifacts

| Artifact | Revision |
| --- | --- |
| `ypcypc/JEMM` reference | `6822fe0fd53c5e6670af6ba99fb2c857a661e532` |
| `MaestroYan/JEMM` adapter | `76e3c209e8441fa658221c7ba2725bad2f811176` |
| `Qwen/Qwen3.8-27B` base, tokenizer and processor | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

The prompt and response adaptation retains the upstream Apache-2.0 license in
[native/LICENSE.jemm](native/LICENSE.jemm). The exporter must merge the adapter
into language weights and extract the 32 selected rows from the **untied
`lm_head.weight`**. Token embeddings and scalar decision heads do not implement
JEMM's readout. The export's `jemm_export.json` has format `jemm-native/1` and
validates artifact revisions, disabled-thinking chat prefix/suffix, 32 distinct
single-token label IDs, token budgets, and the selected BF16 head file. It
enforces the fixed adapter calibration: temperature `1.3480874159655591`, image
temperature `1.3954832341582943`, and threshold `0.9872681877423998`. Custom
calibration exports are rejected. Original processor/config files remain in
the export. Before CUDA initialization, the executor streams and verifies every
file in `export_sha256`, requiring metadata, vision/head weights, the language
index and every indexed language shard. Hash paths must remain within the export.
This adds a full sequential checkpoint read to pre-readiness loading.

## API semantics

`POST /v1/systemone` takes `state`, an ordered `questions` object and optional
`images`. Missing state defaults to an empty string. Structured state uses
compact JSON without sorting keys. Candidate maps preserve insertion order;
labels are `A`–`Z`, then `0`–`5`. Each choice or score question accepts 2–32
candidates. Descriptions and instructions use the reference's Python-style
coercion and whitespace flattening; tool descriptions render names, summaries,
required parameters and truncated defaults. Container string coercion uses
Python 3.12 printability rules from pinned Unicode 15.0 general categories;
[the generator](native/generate_unicode.py) records the official data checksum,
and [the Unicode license](native/LICENSE.unicode) accompanies the compact table.

- `choice` returns the first highest-probability candidate, probabilities and
  maximum-probability confidence.
- `noul` prepares `yes` followed by `no`, maps `true`/`false` criteria to their
  descriptions, and returns `noul = p(yes)`, both probabilities and confidence.
- `score` uses zero-based level indices, returns their probabilities and
  confidence, and computes `expected_value` over those indices.

Calibrated softmax uses FP64 host values and each complete question's logits.
The threshold is retained in the export for provenance; upstream SystemOne
responses do not use it. Responses identify `model: JEMM`; `usage` contains
`input_tokens`, zero `output_tokens`, and `latency_ms`. Input usage sums the
complete prompt length for each question, including image tokens.

Text prompts may contain at most 8192 tokens; image prompts at most 3072.
Every image is checked against the reference's 3,145,728 source-pixel limit,
then resized with the pinned processor's 65,536 minimum/16,777,216 maximum
pixel settings. Over-budget prompts are rejected without truncation or a
budget-dependent resize. All questions are validated before any GPU work.
Native preparation also caps requests at 64 questions and 16 MiB of aggregate compiled prompt text. It checks
the incremental text reservation before allocating each state-bearing prompt.
These native preparation bounds deliberately reject large requests that the
reference does not bound by question count or aggregate prompt bytes.

The native server additionally rejects duplicate JSON keys, nonfinite/out-of-range
JSON numbers, integer lexemes outside `i64`/`u64`, more than 64 questions,
and a mismatched requested `model`. Integer `-0` is preserved as Python integer
`0`; floating `-0.0` remains a float. Oversized integers are rejected before
conversion, including parsed tool-description JSON, rather than rounded to
FP64. Python arbitrary-precision integers outside this range are unsupported.
Malformed requests return 422; unsupported media types return 415. Backend
failure or panic synchronizes and retires the loaded model, then returns 503;
health reports unavailable thereafter. Literal special-token text is preserved
in text prompts. Image prompts require their image-placeholder count to match
the provided image grids.

## Ownership and execution

`contract.rs` owns ordered prompts and calibrated response semantics.
`processing.rs` validates and tokenizes the complete request, decodes/transforms
images through the shared Qwen processor, and creates three-axis positions.
Prepared work retains one unpadded token sequence and candidate count per
question, ordered image placeholders/positions, and shared image patches.
Response context owns original question identities, token usage and timing.

`executor.rs` owns the shared Qwen language model, shared configurable vision
model, and selected label-head GPU weights. Vision runs once per request and
its BF16 embeddings are reused for that request's questions. Language execution
uses `Model::forward` or `Model::forward_multimodal`; each question gets one
prefill. Readout uploads the final BF16 hidden state and projects the selected
BF16 LM-head rows with CUDA GEMM, padding the output dimension to 256. The
GEMM output is rounded to BF16 before FP32 logits are returned, matching the
reference dtype boundary.

One `SerialScheduler` admits an entire request, including vision, language,
readout and synchronized readback. A mutex owns mutable loaded state. Health
and pre-admission availability read a separate atomic flag and never wait for that execution mutex. Retirement
clears readiness before device synchronization. Cancellation after dispatch
keeps scheduler admission until work completes. Caught errors and panics synchronize all device streams before dropping failed state. There
is no cross-request batching or prefix cache. `server.rs` completes real model
warmup before binding the HTTP listener or reporting readiness.

## Running and diagnostics

See [the native recipe](../../../recipe/jemm/native.md) for pinned download,
export, backend build and launch commands. From the repository root:

```sh
cargo build --release --locked -p omni-jemm-native
JEMM_MODEL=/path/to/export JEMM_CUDA_LIB=/path/to/libqwen3_5_cuda.so \
  target/release/omni-jemm-native
```

Host/port default to `127.0.0.1:8000`, overridden by `JEMM_HOST`/`JEMM_PORT`.
`GET /health` is ready only after the real warmup. The JSON request body limit
is 16,000,000 bytes, matching the upstream HTTP limit; oversized HTTP bodies
return 413. Native preparation limits return 422 before inference.

`--prepare-jsonl` consumes one request per stdin line and emits prompts, complete
chat text, token IDs, image indices, three-axis positions, grids, resized
geometry and FP32 pixel values. It loads tokenizer/processor metadata without
CUDA. `--diagnostic-jsonl` loads and warms up the full native model and emits
`response`, raw per-question `logits`, and those preparation diagnostics per
line. Diagnostics go to stdout; worker logs go to stderr. Each mode emits an
`error` object for a rejected line and continues with subsequent input.

## Validation status

[Root tests/jemm](../../../tests/jemm/) covers ordered rendering, tool parameter
syntax, Python coercion, all 32 labels, candidate/type validation, calibrated
answers, Unicode printability and integer bounds, artifact integrity and shard
coverage, image decoding/positions, selected-head validation and padding,
tokenization/usage/limits, and synchronized state retirement after error/panic and nonblocking availability.
An explicit ignored CUDA check tests the real BF16 label projection with exact
constant rows. Synthetic tokenizer tests establish processor boundaries. The
[pinned text fixtures](../../../tests/jemm/fixtures/text-contract.json) preserve
upstream full chat prompts and token IDs for 12 questions across 11 corpus cases.
Their ordinary CPU test compares complete chat text exactly; a separate opt-in
CPU check loads the pinned tokenizer and compares all token IDs and lengths.
These checks do not establish full-model parity.

```sh
cargo test --locked -p omni-jemm-native -- --list
cargo test --locked -p omni-jemm-native
cargo clippy --locked -p omni-jemm-native --all-targets -- -D warnings
# CPU opt-in with pinned local tokenizer metadata:
JEMM_TOKENIZER_METADATA=/path/to/base-metadata \
  cargo test --locked -p omni-jemm-native pinned_tokenizer -- --ignored
# CUDA opt-in; select/reserve a device before running:
JEMM_CUDA_LIB=/path/to/libqwen3_5_cuda.so \
  cargo test --locked -p omni-jemm-native bf16_label_head -- --ignored
```

CPU tests and compilation do not establish A800 correctness, multimodal output
parity, or performance. Hardware evidence and the fixed comparison protocol
belong in the native recipe/evidence report once the actual run has completed.
