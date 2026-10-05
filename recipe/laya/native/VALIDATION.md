# Laya CUDA validation

The combination contains the selected RoPE tile, short QKV/output dispatch,
down N64, short GEGLU selection and long GEGLU BN32/static registers. It preserves
the original arithmetic and checkpoint precision. Failed/HOLD candidates are
excluded; this does not claim zero GPU bubbles or maximum occupancy.

## Frozen native engine comparison

These measurements precede the current-main integration. They used one frozen
Rust CLI with the original and optimized CUDA configurations, on one shared
H800, concurrency 1, FP16 token embeddings, BF16 projection matrices/activations and FP32 residuals/norms. Clocks
were not locked. JSON parsing, tokenization, padding/upload, forward/heads,
readback, decoding and response JSON write are included. HTTP, process/model
startup and warmup/cold Graph construction are excluded.

| Graph input | Original p50 | Combined p50 | Latency reduction | Speedup |
| --- | ---: | ---: | ---: | ---: |
| choice, L48 | 2.5426 ms | 1.5957 ms | 37.24% | 1.59× |
| score, L64 | 2.5819 ms | 1.6101 ms | 37.64% | 1.60× |
| short, L48 | 2.5419 ms | 1.5959 ms | 37.22% | 1.59× |
| medium, L176 | 2.8666 ms | 1.9207 ms | 33.00% | 1.49× |
| long, L512 | 3.9926 ms | 3.0001 ms | 24.86% | 1.33× |

Each entry pools two 100-sample passes. The 12 processes measured 6000 requests
across original/retained/combined variants, Graph/eager, in forward/reverse order.
All raw engine/client timings and original file hashes are in
[measurements.json](measurements.json); private host paths are omitted.
The original uses `--original-rope` and the reverse-patched exporter; both sides
share the same runtime and Graph setting. There is no single workload-weighted
percentage. The full pipeline was measured directly, not by adding kernel gains.

The frozen CLI binary SHA256 is
`4d31e8593ff67127c5c908c3bdba26db393165036027e58a5554b70567a71c85`;
the combined library is
`eeb94478354b5b6f0b26c756263261dd61908a0aafd82f10b04b907c3d46cc36`.
Source kernels/runtime and exported library identity are retained in the evidence.
The earlier three-way numeric check covered 210 responses, 36 hidden-state
comparisons and six 17-request reuse sequences, bitwise equal to the original
native configuration. This tests implementation equivalence for fixed inputs,
not general model quality or all-input agreement with official PyTorch.

## Integration checks

The current-main integration retains checkpoint inventory checks and imports the
reviewed CPU parsing/decoding fixes. Its processor/packing and worker scheduling
are new consumers of the same device code. Normal CPU tests do not establish
full-checkpoint CUDA or HTTP parity. Integration-specific Linux/GPU/worker and
frontend checks must bind the actual new build; the table above is historical
engine evidence, not a measurement of the newly integrated HTTP service.

Build and reproduction commands are in the [recipe](README.md). Full model
weight/tokenizer oracle checks remain explicit opt-in tests, with their pinned
artifact hashes; see the [model contract](../../../src/models/laya/README.md).
