# LAYA model engine

LAYA is the first planned System1-Omni model. This directory owns its complete request-to-result path: preprocessing, postprocessing, batching policy, state, execution, and backend-specific kernel selection.

GPU operations and kernel implementations belong in [`backends/cuda/`](../../backends/cuda/) and [`backends/metal/`](../../backends/metal/). Setup and usage examples belong in the top-level [`recipe/`](../../../recipe/) directory.

Status: planned; no model implementation or validated GPU backend support yet.
