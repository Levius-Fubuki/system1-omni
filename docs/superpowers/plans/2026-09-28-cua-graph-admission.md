# CUDA Graph Admission Implementation Plan

> **For agentic workers:** Use subagent-driven-development for the isolated benchmark task and independent review. Main agent owns runtime integration and GPU execution.

**Goal:** Prevent low-reuse requests from repeatedly paying Graph capture costs and publish reproducible end-to-end comparisons.

**Architecture:** A pure Python request-aware admission policy feeds the existing exact-layout Graph runtime. Request lifetime is explicit; cache hits bypass capture admission, and failed capture attempts count against the sliding budget.

**Tech Stack:** Python, PyTorch CUDA Graphs, Transformers/PEFT, pytest, RTX 4090.

- [x] Add failing policy tests for repeated questions, sparse lengths, eviction cooldown, metadata bounds, count/time budgets and reset; run `PYTHONPATH=src python3 -m pytest tests/cua_s1/test_graph_admission.py -q`.
- [x] Implement `graph_admission.py`; integrate request context, capture accounting, invalidation and CLI in runtime/model/server. Run new and existing CPU tests.
- [x] Fix mixed-shape summary naming and full-response equality; preserve old raw evidence, distinguish 118/119 historical test counts. Add extended benchmark and verifier with raw responses, parity and complete schedules. Run benchmark unit tests.
- [x] Review specification coverage, then code quality; resolve findings before final measurement.
- [x] Transfer committed clean source to a separate server checkout. Run complete Cua-S1 tests, Ruff and GPU smoke. Compare legacy/new/eager on extended workloads, changed inputs, cache memory fallback, and HTTP.
- [x] Independently verify raw results and hashes; document cost boundaries and limitations. Back up evidence and source locally.
- [x] Commit/push and create PR with measured source SHA and validation. Attach PR; execute server shutdown and verify connectivity stops.

Completed 2026-09-29: [PR #33](https://github.com/ThinkFlowLab/system1-omni/pull/33). Final measured source `9f2c7ef`, final implementation/evidence validation `fb8a47d`: 164 tests, CI lint/format, complete publication verifier. Local evidence archive SHA-256 `158d40ae84387cc3e246e0bcb81a3bd5dd03c1887801895c3600971d3142a80d`. Server shutdown returned successfully and the subsequent SSH connection was refused.
