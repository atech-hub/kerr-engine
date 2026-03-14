//! GPU compute backend — wgpu implementation of ComputeBackend.
//!
//! Split into three files for maintainability:
//! - gpu_pipelines.rs: struct, param structs, constructors, dispatch helpers (~760 lines)
//! - gpu_dispatch.rs: ComputeBackend trait implementation (~470 lines)
//! - gpu_validate.rs: validation and benchmark harness (~270 lines)

// Re-export the public API so external callers don't change
pub use crate::gpu_pipelines::GpuBackend;
pub use crate::gpu_validate::{validate_gpu_backend, benchmark_gpu_vs_cpu};
