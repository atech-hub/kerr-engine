//! WGPU device setup, buffer management, and kernel dispatch.
//!
//! Handles the host-side plumbing: creating the GPU device, allocating
//! storage buffers, loading the compiled SPIR-V kernel, dispatching
//! compute workgroups, and reading results back to CPU.

// TODO: Stage 1 implementation
// - init_device() -> (Device, Queue)
// - create_buffers(device, n_bands) -> input/output/params buffers
// - load_kernel(device) -> ComputePipeline
// - dispatch(queue, pipeline, buffers, n_bands) -> run one kernel invocation
// - readback(device, queue, output_buffer) -> Vec<f32>
