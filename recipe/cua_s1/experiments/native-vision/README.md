# Native vision validation — 2026-10-02

Validated on NVIDIA GeForce RTX 4090 (sm_89, 24 GiB), driver 595.71.05,
CUDA toolkit 13.2; Python 3.12.3, Torch 2.14.0+cu130, Transformers 5.17.0,
PEFT 0.21.0, Pillow 11.3.0. Reference controls use unmerged adapters in BF16
and full FP32 with TF32 disabled. Weights match the pinned upstream lock.
Native execution uses unmerged FP32 vision LoRA and merged BF16 language LoRA.

| Set | Questions | Maximum native probability error | BF16 reference error | Allowed error | Matching choices |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Standard](standard.json) | 8 | 0.00331324 | 0.00526386 | 0.02052773 | 8/8 |
| [Boundary](boundary.json) | 11 | 0.07950398 | 0.09646034 | 0.20292068 | 11/11 |

The sets are evaluated separately so adding a difficult boundary case does not
relax the standard set's threshold. Every FP32 margin exceeds 0.05. Token IDs,
image grids and all three position axes match exactly. Repeated language calls
and vision reruns after language execution produce identical native outputs.
Standard PNG RGB pixels match Pillow; JPEG differs by up to 3 intensity levels.

The standard set includes square, landscape, portrait, JPEG, 1/26 choices,
structured Chinese instructions and two questions. Boundary images include
1024×1024, 200×1, 383×257 and 1×1, the latter with eight questions. These are a
finite synthetic validation corpus, not a general accuracy benchmark.

Vision stage checks identified and fixed Conv3d patch projection rounding:
FP32 accumulation is rounded to BF16 before the patch bias. On the standard
small image, patch mean absolute error is 2.33e-8; final feature RMS error is
0.005306 and maximum error 0.632813. Feature outputs are not bitwise identical.
Stage downloads are diagnostic and are excluded from any performance claim.
No speedup claim is made.

## Verification

- Workspace format, strict all-target Clippy, Rust tests and release build.
- 63 CPU Rust tests passed; 7 tests requiring external GPU/checkpoint data are
  skipped by ordinary CPU tests. Five Cua-S1 GPU regressions were executed
  separately, including graph misses/hits, changed tokens, eviction, allocation
  growth and multimodal/text mixing, kernel oracles and image insertion.
- CUDA primitives compared with PyTorch: patch Conv3d, biased projection,
  LayerNorm, FP32 LoRA GEMMs/addition, rotary, both GELUs and bidirectional attention.
- Live HTTP: 11 valid requests, 19 question outputs equal to the direct native
  replay (tolerance 1e-7); 9 malformed/unsupported/body-limit cases checked on
  each set. Model identity, usage and `detail` envelopes checked.
- Six 16-bit PNG modes independently checked against Pillow, including L16 and
  RGB16 with transparency. Same-size weight corruption and unlisted overrides
  are rejected by provenance tests.

See [reproduction instructions](../../native_multimodal.md). The case generator,
reference exporter, native replay, HTTP checker and acceptance verifier are all
included in this branch. Raw controls, per-stage BF16 tensors and logs are retained
in `/root/cua-native-vision-20261002/` on the authorized GPU host and the local
`artifacts/cua-native-vision-20261002/` evidence directory. The language checkpoint
reuses the previously verified multimodal export without changing its weights;
its tokenizer and SHA-256 manifest are saved in a new task directory.

## Review

Independent review covered specification and implementation. Five findings were
fixed and re-reviewed: pinned source/export provenance, pre-decode PNG allocation
limits, 16-bit PNG conversions, public question invariants, and CUDA device
selection when moving vision execution between threads. A final review also fixed
HTTP checker coverage truncation and added choice/confidence assertions. No unresolved actionable
finding remained in that review. Contributor and maintainer review remain separate.
