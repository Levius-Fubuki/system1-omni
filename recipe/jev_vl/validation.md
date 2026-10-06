# JEV-27B-VL validation status

**The reviewed ABI 6 candidate passes the bounded H800 checks below.** It remains
experimental: these checks cover one frozen synthetic corpus and a prepared-image
worker path. Clean installation, broad model-quality coverage, other shared-model
checkpoints and hosted GitHub CI remain unverified.

## Current integrated H800 validation

The candidate integrates upstream `47eff9cdeda01e4847a4fb9634a43f2cab6a233f`
(ABI 5) with ABI 6 prefix continuation and the review fixes. Build job 413586
prepared source snapshot SHA-256
`35aa8ecdf9af4d9ee5d84243c663a9428c7aa00ff86370bade4bc0df24695d2f`.
GPU jobs 413587 and 413597 verified the same library and worker binary hashes on
one H800 80 GB, CUDA 13.0.88, driver 580.159.03. The candidate was an uncommitted
review snapshot; use the archived [source and binary provenance](https://github.com/linear3735/system1-omni/blob/522f2256876a62ddd70873ee9e775055416aea1b/recipe/jev_vl/evidence/review-20261006/provenance.json)
rather than treating the original three commits as the tested final source.
Subsequent review edits change comments, test wiring and diagnostic header
constants only: the inaccurate L2 hit field was removed from `x-jev-cache`;
actual L2 counters remain available from `/v1/cache/stats`. The measured GPU
inference and cache algorithms are unchanged, but the final source hashes
are not identical to this measured snapshot.

- **12 GPU/kernel and checkpoint tests passed** in job 413587, including cached
  convolution history, empty-state graph behavior and rejection of uncaptured or
  foreign-model prefix states. A separate checkpoint/tokenizer prefix-splitting
  test passed on CPU; it is not a thirteenth GPU test.
- **48/48 worker decisions and eight error probes passed in each cache mode** in
  job 413597. Maximum probability drift against the frozen vLLM reference was
  0.021545 with caches off and 0.014215 with caches on, within the fixed 0.025
  tolerance. Neither result establishes bitwise equality or general quality.
- The public replay tool passed **48 requests × worker/frontend × off/on = 192
  measured decisions**, plus its recorded warmups. Frontend checks preserve all
  seven supported decision-error envelopes; unsupported chat routes separately
  return 404.
- Local formatting, strict workspace Clippy, locked release build and workspace
  tests passed: **121 passed, 19 ignored**. Ignored GPU/checkpoint tests are
  reported separately above. Python benchmark/replay tests and strict MkDocs
  also passed. Dependency resolution passed; a clean environment installation
  and deployment have not been reproduced.

### Paired cache measurement

Job 413597 completed with exit `0:0`. On the same GPU, checkpoint and binary,
cache-off and all-on each ran two measured passes of 12 image questions, with
12 excluded warmup requests per pass. All 48 measured responses succeeded.

| Configuration | Pass 1 p50 | Pass 2 p50 | Combined p50 |
| --- | ---: | ---: | ---: |
| Cache disabled, full language forward | 102.35 ms | 104.32 ms | 102.88 ms |
| All caches enabled, warm prefix hits | 37.32 ms | 36.78 ms | 37.04 ms |

The combined-median ratio is **2.78×**. This is worker-direct localhost HTTP,
including response JSON parsing, at concurrency 1. Loading, startup, offline
image decoding/vision encoding, warmup and the Rust frontend are excluded.
It measures repeated decisions on one prepared synthetic image, not live-image
end-to-end latency or a new speed comparison with vLLM. The two-pass spread is
observed variation, not a confidence interval.

All-on counters recorded 48 L1/L2/L3 hits across warmup plus measurement, with
three resident prefixes and no fallback. Resident L3 accounting was 643,384,320
bytes; peak memory was not measured. There is no current concurrency-8 A/B.

[Summary](https://github.com/linear3735/system1-omni/blob/522f2256876a62ddd70873ee9e775055416aea1b/recipe/jev_vl/evidence/review-20261006/summary.json),
[raw off pass 1](https://github.com/linear3735/system1-omni/blob/522f2256876a62ddd70873ee9e775055416aea1b/recipe/jev_vl/evidence/review-20261006/raw-off-c1-p0.jsonl),
[off pass 2](https://github.com/linear3735/system1-omni/blob/522f2256876a62ddd70873ee9e775055416aea1b/recipe/jev_vl/evidence/review-20261006/raw-off-c1-p1.jsonl),
[raw on pass 1](https://github.com/linear3735/system1-omni/blob/522f2256876a62ddd70873ee9e775055416aea1b/recipe/jev_vl/evidence/review-20261006/raw-all-c1-p0.jsonl),
[on pass 2](https://github.com/linear3735/system1-omni/blob/522f2256876a62ddd70873ee9e775055416aea1b/recipe/jev_vl/evidence/review-20261006/raw-all-c1-p1.jsonl), and
[checksummed evidence inventory](https://github.com/linear3735/system1-omni/blob/522f2256876a62ddd70873ee9e775055416aea1b/recipe/jev_vl/evidence/review-20261006/provenance.json)
at the reviewed commit preserve the original timing rows, parity verdicts,
public-replay responses and cache counters. The source archive and complete build
logs remain in the local review archive; their hashes identify the measured
snapshot.

### Preserved initial failure

Job 413587's frozen frontend comparator expected the worker error envelope on
`/v1/chat/completions` and failed. The frontend owns only `/health` and
`/v1/systemone`; it returns its own 404 for this unsupported route. The
[original failed verdict](https://github.com/linear3735/system1-omni/blob/522f2256876a62ddd70873ee9e775055416aea1b/recipe/jev_vl/evidence/review-20261006/initial-frontend-failed-verdict.json)
is retained. Job 413597 separates all seven supported error-envelope checks
from the unsupported-route 404 check. This corrects the validation boundary;
no production frontend route was added.

## Appendix: historical evidence audited on 2026-10-06

The reference is `autotrust/JEV-27B-VL` at
`f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc`, its `adapter_vllm`, calibration and
`serve_decide.py`, running with vLLM 0.31.0 and PyTorch 2.13.0+cu130.
The native experiment used base `7f39ac40902c374803992407bb26eeba29c8a588` plus
uncommitted changes. The archive identifies source snapshots but does not record
a source-to-binary build attestation. These measurements must not be relabeled
as tests of the later review head or current integrated branch.

The frozen 48-request corpus contains 36 synthetic text prompts and 12 questions
about a single synthetic 960×960 color/noise image. It covers `noul`, `score`
and 2–16-option `choice`. It is an implementation comparison, not a business
accuracy evaluation, real GUI benchmark or general multimodal quality result.
The source generator uses Python's process-dependent `hash()` for text seeds;
regenerating it does not establish the frozen input hash.

### Independent job 413512

One H800 80 GB ran both native cache configurations serially using the same
binary and prepared image assets. Timing covers localhost worker-direct HTTP,
including response parsing, at concurrency 1. Each configuration has two measured
passes of the 12 image questions, each preceded by 12 excluded warmup requests.
Loading, image decoding, vision encoding and the Rust frontend are outside this
measurement. All 48 measured HTTP responses succeeded.

| Historical configuration | Pass 1 p50 | Pass 2 p50 | Combined p50 |
| --- | ---: | ---: | ---: |
| Cache disabled, full language forward | 103.86 ms | 103.23 ms | 103.27 ms |
| All caches enabled, warm prefix hits | 35.55 ms | 35.06 ms | 35.49 ms |

The ratio of the combined medians is **2.91×**. This is a warm prepared-image
language-serving result, not an end-to-end vision speedup. The often cited
**3.02×** belongs to earlier job 413490; its summary survived while shared output
filenames were overwritten by job 413512. Concurrency 8 is a separate queued
workload: the later all-on median is 137.47 ms, without a paired off-mode result
at that concurrency.

The retained comparator reports 48/48 identical selected decisions for each
mode against historical vLLM responses, with maximum probability differences
0.021545 (off) and 0.010764 (all), within the declared 0.025 tolerance. Eight
error probes also passed that comparator. This checks selected fields and
probability drift; it does not compare complete response schemas, all error
details or token accounting. Native cache-on/off outputs are not bitwise equal.
Kernel job 413489 separately passed nine old-ABI kernel tests; those tests do
not substitute for current full-model continuation tests.

The historical L3 resident counter recorded 643,384,320 bytes for three prefixes;
this is cache accounting, not peak device memory. Cache snapshots bracket
warmup plus measurement, so 24 hits in a pass include 12 warmup requests and
12 measured requests. Historical `x-jev-cache` could label L2 as a hit even
when L2 was disabled; use the corresponding counters and configuration when
interpreting that field.

The earlier **3.34× text comparison** uses frozen R1 vLLM timings against later
native timings, rather than a fresh same-run same-device A/B. It is not a current
speed claim and should not be used as evidence of an isolated backend gain.

## Included records and reproducibility limits

[Provenance and checksums](https://github.com/linear3735/system1-omni/blob/522f2256876a62ddd70873ee9e775055416aea1b/recipe/jev_vl/evidence/provenance.json) identify the checkpoint,
input hashes, comparator and retained source archives. Archived records at the reviewed commit:

- Cache off: [pass 1](https://github.com/linear3735/system1-omni/blob/522f2256876a62ddd70873ee9e775055416aea1b/recipe/jev_vl/evidence/raw-r2d-off-c1-p0.jsonl), [pass 2](https://github.com/linear3735/system1-omni/blob/522f2256876a62ddd70873ee9e775055416aea1b/recipe/jev_vl/evidence/raw-r2d-off-c1-p1.jsonl).
- Cache on: [pass 1](https://github.com/linear3735/system1-omni/blob/522f2256876a62ddd70873ee9e775055416aea1b/recipe/jev_vl/evidence/raw-r2d-all-c1-p0.jsonl), [pass 2](https://github.com/linear3735/system1-omni/blob/522f2256876a62ddd70873ee9e775055416aea1b/recipe/jev_vl/evidence/raw-r2d-all-c1-p1.jsonl).
- Comparator verdicts: [off](https://github.com/linear3735/system1-omni/blob/522f2256876a62ddd70873ee9e775055416aea1b/recipe/jev_vl/evidence/parity-off.json), [all](https://github.com/linear3735/system1-omni/blob/522f2256876a62ddd70873ee9e775055416aea1b/recipe/jev_vl/evidence/parity-all.json).

Run outputs are retained in the linked commit archive. The current checkout
keeps the frozen input and reference fixtures needed for replay; new run outputs
belong in a task-owned output directory, as in the commands below.

The [manifest template](evidence/manifest-template.jsonl) and its single
[synthetic image](evidence/image.png) preserve the complete frozen corpus without
storing twelve copies of the same data URI. Restore its exact original bytes
with the standard-library script; it checks the original SHA-256 before writing:

```sh
python3 recipe/jev_vl/evidence/restore_manifest.py --out /tmp/jev-vl-manifest.jsonl
```

The restored JSONL is approximately 41 MiB. The
[reference responses](evidence/reference/) contain the 48 original vLLM answers
and eight archived error probes, with a [checksum inventory](evidence/reference/sha256.json).
The measured source/binary bundle is separate
evidence: the copied inputs and responses do not establish reproducibility of
the old executable.

## Replay on a new isolated run

Follow the [recipe](README.md) to export weights, build, preencode the restored
manifest, and start the worker. Run the standard-library replay tool on the full
corpus to compare successful decisions and probabilities with the frozen
reference. Its fixed absolute probability tolerance is 0.025; a failed request,
invalid probability vector or changed decision causes a nonzero exit:

```sh
python3 recipe/jev_vl/replay.py \
  --base http://127.0.0.1:8001 --manifest /tmp/jev-vl-manifest.jsonl \
  --reference recipe/jev_vl/evidence/reference \
  --out /tmp/jev-vl-parity-$(date +%Y%m%dT%H%M%S) --warmup 4 --passes 1
```

The tool writes warmup and measured responses to `results.jsonl`; `summary.json`
aggregates measured requests only. It refuses an existing output directory.
Repeat through `http://127.0.0.1:8080` to exercise the frontend, using a new output
directory. Replay compares the regular corpus; the archived error probes are
separate from these replay commands; include equivalent direct-worker and
frontend HTTP probes when validating a new revision.

For a new paired warm-image comparison, launch the worker with
`JEV_VL_CACHE=0`, then run:

```sh
python3 recipe/jev_vl/replay.py \
  --base http://127.0.0.1:8001 --manifest /tmp/jev-vl-manifest.jsonl \
  --reference recipe/jev_vl/evidence/reference --pattern img- \
  --out /tmp/jev-vl-cache-off-$(date +%Y%m%dT%H%M%S) --warmup 12 --passes 2
```

Stop that worker, restart the same binary with `JEV_VL_CACHE=1`, and repeat the
command with a new `cache-on` output directory. Keep the same allocated GPU,
checkpoint, image assets and request order. This measures one serial client;
it does not implement a concurrency-8 run. Both configurations exclude offline
vision encoding. Retain `GET /v1/cache/stats` snapshots from the direct worker
before and after each run to verify the intended path; the counter difference
includes warmup as well as measured requests. These commands provide a portable
path for collecting new evidence with the same declared scope.

## Remaining validation limits

The completed run covers the frozen 48-request corpus and the targeted prefix
state regressions. Broader changed-image/question-order workloads, exhaustive
cache-eviction numerical checks, complete usage/error-contract coverage and
full-checkpoint Cua-S1/Open-Jev shared-backend regression remain unverified.
Clean installation, model-quality evaluation and hosted GitHub CI are also
outstanding.

Freeze commits, build hashes, dependencies, inputs and cache policy before a new
paired timing run. Save each run in a unique directory, report repeated pass
variation, failures and memory measurement method, and keep startup, preprocessing,
cache misses and warm hits separate. No new GPU experiment is implied by the
documentation or historical evidence.
