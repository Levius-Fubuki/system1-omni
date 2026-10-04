# CUDA backend

Planned home for high-performance NVIDIA GPU operations and kernel integration. Implement the operations required by the first model, with hardware-specific optimizations where needed.

In the [target architecture](../../../README.md#how-it-works), the shared worker
runtime owns processing orchestration, batching policy, and request bookkeeping.
Model executors own forward passes, device state, and kernel selection. The
shared runtime layers are planned; current workers retain model-specific
pipelines. CUDA and Metal implementations do not need identical internal
structures or a universal tensor abstraction.

Status: [`qwen3_5/`](qwen3_5/) has the operations of a prefill-only Qwen3.5 forward pass, used by the Cua-S1 native worker and measured on sm_89. Other models are planned.
