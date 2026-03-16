//! GPU buffer pool — persistent, reusable buffers for training dispatch.
//!
//! Eliminates the ~4,160 buffer create/destroy cycles per iteration.
//! Weight buffers cached by pointer identity (uploaded once, reused).
//! Scratch output buffer pre-allocated and reused between dispatches.
//! Single staging buffer for all readbacks.
//!
//! Two-phase API to satisfy the borrow checker:
//!   Phase 1: ensure_data/ensure_scratch/write_uniform (mutable)
//!   Phase 2: data_ref/scratch_ref/uniform_ref (immutable)

use std::collections::HashMap;
use wgpu;
use wgpu::util::DeviceExt;

/// Buffer pool that caches GPU buffers for reuse across dispatches.
pub struct GpuBufferPool {
    /// Weight/bias buffers cached by (data pointer, byte length).
    data_cache: HashMap<(usize, usize), wgpu::Buffer>,

    /// Single scratch output buffer (resized as needed).
    scratch: Option<wgpu::Buffer>,
    scratch_size: u64,

    /// Single staging buffer for readback.
    staging: Option<wgpu::Buffer>,
    staging_size: u64,

    /// Single uniform buffer for params.
    uniform: Option<wgpu::Buffer>,
    uniform_size: u64,
}

impl GpuBufferPool {
    pub fn new() -> Self {
        Self {
            data_cache: HashMap::new(),
            scratch: None,
            scratch_size: 0,
            staging: None,
            staging_size: 0,
            uniform: None,
            uniform_size: 0,
        }
    }

    // ─── Phase 1: Ensure buffers exist (mutable) ─────────────

    /// Ensure a data buffer exists for this slice. Cached by pointer identity.
    pub fn ensure_data(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[f32]) {
        let key = (data.as_ptr() as usize, data.len() * 4);
        if !self.data_cache.contains_key(&key) {
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            });
            self.data_cache.insert(key, buf);
        }
    }

    /// Ensure scratch output buffer is at least `n_bytes`.
    pub fn ensure_scratch(&mut self, device: &wgpu::Device, n_bytes: u64) {
        let size = n_bytes.max(16);
        if size > self.scratch_size {
            self.scratch = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pool_scratch"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.scratch_size = size;
        }
    }

    /// Write params to the uniform buffer (resizes if needed).
    pub fn write_uniform<T: bytemuck::Pod>(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &T) {
        let bytes = bytemuck::bytes_of(data);
        let size = bytes.len() as u64;
        if size > self.uniform_size {
            self.uniform = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pool_uniform"),
                size: size.max(16),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.uniform_size = size.max(16);
        }
        queue.write_buffer(self.uniform.as_ref().unwrap(), 0, bytes);
    }

    // ─── Phase 2: Get buffer references (immutable) ──────────

    /// Get reference to a cached data buffer.
    pub fn data_ref(&self, data: &[f32]) -> &wgpu::Buffer {
        let key = (data.as_ptr() as usize, data.len() * 4);
        self.data_cache.get(&key).expect("data buffer not ensured")
    }

    /// Get reference to the scratch output buffer.
    pub fn scratch_ref(&self) -> &wgpu::Buffer {
        self.scratch.as_ref().expect("scratch buffer not ensured")
    }

    /// Get reference to the uniform buffer.
    pub fn uniform_ref(&self) -> &wgpu::Buffer {
        self.uniform.as_ref().expect("uniform buffer not written")
    }

    // ─── Readback ────────────────────────────────────────────

    /// Read back f32 data from the scratch buffer.
    pub fn readback_scratch(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, n_floats: usize) -> Vec<f32> {
        let size = (n_floats * 4) as u64;

        // Ensure staging buffer
        if size > self.staging_size {
            self.staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pool_staging"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.staging_size = size;
        }

        let scratch = self.scratch.as_ref().unwrap();
        let staging = self.staging.as_ref().unwrap();

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(scratch, 0, staging, 0, size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    /// Invalidate all cached data buffers. Call after optimizer step.
    pub fn invalidate_weights(&mut self) {
        self.data_cache.clear();
    }

    /// Number of cached data buffers (for diagnostics).
    pub fn cached_count(&self) -> usize {
        self.data_cache.len()
    }
}
