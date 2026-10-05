//! Native Cua-S1 text worker, using the shared Qwen prefill executor.

pub mod contract;
pub use omni_qwen3_5_native::{cuda, inputs, json, model};
pub mod engine;
pub mod executor;
pub mod processing;
