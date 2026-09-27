# System1-Omni

A community-maintained inference engine for prefill-only JEV models, designed around a Rust frontend, model-owned execution, and high-performance CUDA and Metal backends.

The project is in its initial design stage. The architecture below describes the intended implementation; model engines and GPU backends are not implemented yet.

## Architecture

Share serving infrastructure; let each model own its execution.

![System1-Omni architecture: Rust frontend, model-owned execution, and CUDA and Metal backends](docs/assets/architecture.svg)

| Layer | Responsibility |
| --- | --- |
| Rust frontend | API, request lifecycle, and response delivery through a small engine interface. |
| System1-Omni models | Model-specific preprocessing and postprocessing, batching, state, execution, and kernel selection. |
| CUDA backend | High-performance GPU operations for NVIDIA GPUs. |
| Metal backend | High-performance GPU operations for Apple GPUs. |

Each model owns its complete request-to-result path. Shared utilities stay minimal and are extracted when implementations need the same functionality. Backends can optimize for their hardware without requiring identical internal implementations.

## Supported models

No models are implemented yet. LAYA is the first planned model:

| Model | Status |
| --- | --- |
| LAYA | Planned |

CUDA and Metal coverage will be documented per model as implementations are added and validated.
