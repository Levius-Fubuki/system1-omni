# Rust frontend

Planned home for the API, request lifecycle, cancellation, response delivery, and serving metrics. Keep the small engine interface here initially.

Model-specific preprocessing, batching policy, and execution belong in [`models/`](../models/). This boundary accommodates a separately launched model worker and does not require in-process execution.

Status: layout only. Integrate the frontend work from [PR #2](https://github.com/ThinkFlowLab/system1-omni/pull/2) here, coordinating its package layout and Cargo configuration when implementation lands.
