# Open-Jev H200 validation, 2026-10-02

A matched comparison of 74 real JevBench `noul` requests, each with one candidate,
measured packed MLP SiLU with cached residual RMSNorm fixed in both native variants.
All 74 native decisions and probabilities were exactly unchanged by packed SiLU;
the native worker scored 64/74 correct. OpenJev-Fast scored 63/74, with one different
decision (`hard-opus-a-temporal_numeric-09`) and maximum native/Fast probability
difference 0.03435. These counts do not establish statistical accuracy superiority.

## Warm HTTP latency

| Configuration | Mean, pass 1 / pass 2 (ms) | P50, pass 1 / pass 2 (ms) |
| --- | --- | --- |
| Native with cached residual RMSNorm; scalar SiLU | 49.563 /49.490 | 25.434 /25.539 |
| Native with cached residual RMSNorm and packed SiLU | 48.286 /48.307 | 24.925 /24.657 |
| Original OpenJev-Fast | 57.345 /50.908 | 25.026 /24.770 |

Packed SiLU reduces native mean 49.527→48.297 ms (2.483%). Its two-pass mean range
is disjoint from baseline's. These are observed two-pass ranges, not confidence
intervals. Fast's 6.436 ms mean spread prevents claiming a stable aggregate winner.
This subset does not measure multi-candidate prefix sharing. The author's 17.3 ms
B300 result is a different hardware/workload measurement.

HTTP includes the same Rust frontend, tokenization and worker execution; excludes
client body serialization and response JSON parsing, includes UTF8 decoding. One
server per configuration is reused. Real warmup, first request after readiness and
one complete 74-case feasibility pass are excluded; two subsequent passes are
measured. Nsight collection is inactive during HTTP passes, although CUPTI
instrumentation may remain loaded. No shared caches are dropped or clocks changed.

## Separate CUDA timelines

Each entry is the two-trace mean of 64 MLP SiLU launches per request, in milliseconds.

| Tokens | Scalar SiLU | Packed SiLU | Fast SiLU lookup |
| --- | --- | --- | --- |
| 107 | 0.680 | 0.305 | 0.359 |
| 936 | 5.643 | 1.746 | 1.513 |
| 3399 | 21.217 | 6.380 | 5.466 |

Long-request MLP SiLU decreases 69.93%, closing 94.19% of that measured kernel
family's gap to Fast. Other kernel families vary between traces; their separate
timeline totals are not HTTP latency. Fast uses padded/tree layouts and lookup
tables, so this does not imply identical rows, fusion boundaries or arithmetic.

The prespecified acceptance gates were five GPU tests, unchanged native decisions,
maximum probability delta ≤0.01, ≥25% long SiLU duration reduction and ≥2% warm HTTP
mean reduction with disjoint observed ranges. All passed; no extra measured runs
were added. Final integration also passed the six-test shared CUDA ABI 4 suite.

## Frozen controls and reproduction

- Device: exact scheduler GPU 2, UUID `GPU-cbf66259-f4ab-0ede-1811-82037dde5924`,
  NVIDIA H200, 143771 MiB (reported as `NVIDIA L20X` in the archived device metadata);
  CUDA driver and Nsight identify SM90 / 132 SMs. NUMA 0, CPUs 0–15.
- BF16, max length 16384, HTTP concurrency 1, native `CUA_S1_GRAPH=0`. Fast retains
  its original graph/kernel stack. Build with nvcc 13.0.88, SM90, `-O3 -std=c++17
  -lineinfo`; use the same cuBLASLt/runtime libraries for both native variants.
- Frozen native worker/frontend: `202c0e163f868334a99d88407056ebe61dbb2dce`.
  Both native libraries include the cached RMSNorm source; only `elementwise.cu`
  differs for packed SiLU. The shared CUDA ABI remains 4.
- [JevBench](https://github.com/fstandhartinger/jevbench)
  at `f8ce71361165846101d02ebc83ad44e47ae44fc3`; select its 74 `noul` cases, preserving
  request bodies and order. Frozen request JSON SHA256:
  `0c756a7b0b4c1f1352225f2e01770b5b3a0646fe86d5cc6930ded9e292ef37df`.
- [OpenJev-Fast](https://github.com/lyuyiqi/open-jev-fast)
  at `c52b8bb958c1f0d241d4eb7fce4ecd8d885bf1e4`; original server/model/kernels, with
  prepared SM90 extensions, PyTorch 2.13cu130/Triton 3.7.1, Transformers 5.10.2,
  PEFT 0.19.1 and FLA 0.5.2. These differ from the author's B300 environment.
- Model/export uses the pinned base, adapter and calibration in the
  [native recipe](native.md); temperature 2.5343690298472983. Reuse prepared weights
  and extensions. Keep copies, downloads, compilation, warmup and process-to-
  readiness separate from measured execution.

Reserve the exact device with `gpu run --gpu-ids 2 --timeout 45m --note <label> --`,
then bind the command with `numactl --membind=0 --physcpubind=0-15`. Run baseline,
packed native and Fast once each, reusing each server for one excluded feasibility
and two measured passes. Separately trace two requests at each 107/936/3399 tokens
per configuration with Nsight Systems CUDA/node tracing: 18 traces total.

Raw bodies, timing rows, SHA256 manifests, reproduction commands,
18 `.nsys-rep`/SQLite pairs and plots are archived locally in the benchmark
worktree's `profile/jev-single-candidate-silu-pack8-20261002/`, outside this PR.
Nsight Compute counters are denied by the host policy. Geometry,
register/shared-memory metadata and CUDA timelines are available; achieved
occupancy, per-SM tails, stalls, Tensor Core utilization and bandwidth/cache
efficiency are unmeasured. No hardware-cause claim follows from timeline data alone.
