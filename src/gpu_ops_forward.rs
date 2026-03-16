//! GPU forward dispatch operations — pooled buffer management.
//!
//! Single-position and batched forward dispatches: matvec, layer_norm,
//! kerr_derivative, kerr_ode. All use GpuBufferPool for weight caching
//! and scratch buffer reuse.

use crate::gpu_pipelines::*;
use crate::model::*;

use wgpu::util::DeviceExt;

impl GpuBackend {
    // ─── Legacy buffer helpers (used by non-pooled backward ops) ──

    /// Create a read-only storage buffer from f32 data.
    pub(crate) fn storage_buf(&self, label: &str, data: &[f32]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    /// Create a read-write storage buffer of given f32 count.
    pub(crate) fn output_buf(&self, label: &str, n_floats: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (n_floats * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Create a uniform buffer from a Pod struct.
    pub(crate) fn uniform_buf<T: bytemuck::Pod>(&self, label: &str, data: &T) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(data),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    /// Read back f32 data from a GPU buffer (legacy — non-pooled path).
    pub(crate) fn readback(&self, buf: &wgpu::Buffer, n_floats: usize) -> Vec<f32> {
        let size = (n_floats * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    // ─── Pooled forward dispatches ───────────────────────────────

    /// Matvec: y = W @ x + b. Pooled weight cache + scratch reuse.
    pub(crate) fn gpu_matvec(&self, w_flat: &[f32], x: &[f32], b: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        let use_bias = if b.is_empty() { 0u32 } else { 1u32 };
        {
            let mut pool = self.pool.lock().unwrap();
            pool.ensure_data(&self.device, &self.queue, w_flat);
            pool.ensure_data(&self.device, &self.queue, x);
            let b_data = if b.is_empty() { &[0.0f32][..] } else { b };
            pool.ensure_data(&self.device, &self.queue, b_data);
            pool.ensure_scratch(&self.device, 0, (out_dim * 4) as u64);
            let params = MatvecParams { out_dim: out_dim as u32, in_dim: in_dim as u32, use_bias, _pad: 0 };
            pool.write_uniform(&self.device, &self.queue, &params);
        }
        let b_data = if b.is_empty() { &[0.0f32][..] } else { b };
        let pool = self.pool.lock().unwrap();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.matvec_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pool.data_ref(w_flat).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pool.data_ref(x).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pool.data_ref(b_data).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pool.scratch_ref(0).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pool.uniform_ref().as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        { let mut pass = encoder.begin_compute_pass(&Default::default());
          pass.set_pipeline(&self.matvec_pipeline);
          pass.set_bind_group(0, &bind_group, &[]);
          pass.dispatch_workgroups((out_dim as u32 + 63) / 64, 1, 1); }
        self.queue.submit(Some(encoder.finish()));
        drop(pool);
        self.pool.lock().unwrap().readback_scratch(&self.device, &self.queue, 0, out_dim)
    }

    /// Layer norm on GPU. Pooled.
    pub(crate) fn gpu_layer_norm(&self, x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
        let dim = x.len();
        {
            let mut pool = self.pool.lock().unwrap();
            pool.ensure_data(&self.device, &self.queue, x);
            pool.ensure_data(&self.device, &self.queue, weight);
            pool.ensure_data(&self.device, &self.queue, bias);
            pool.ensure_scratch(&self.device, 0, (dim * 4) as u64);
            let params = LayerNormParams { dim: dim as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
            pool.write_uniform(&self.device, &self.queue, &params);
        }
        let pool = self.pool.lock().unwrap();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.layer_norm_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pool.data_ref(x).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pool.data_ref(weight).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pool.data_ref(bias).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pool.scratch_ref(0).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pool.uniform_ref().as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        { let mut pass = encoder.begin_compute_pass(&Default::default());
          pass.set_pipeline(&self.layer_norm_pipeline);
          pass.set_bind_group(0, &bind_group, &[]);
          pass.dispatch_workgroups(1, 1, 1); }
        self.queue.submit(Some(encoder.finish()));
        drop(pool);
        self.pool.lock().unwrap().readback_scratch(&self.device, &self.queue, 0, dim)
    }

    /// Kerr derivative on GPU. Pooled. Returns (dr, ds).
    pub(crate) fn gpu_kerr_derivative(&self, r: &[f32], s: &[f32], gamma: &[f32], omega: &[f32], alpha: f32, beta: f32) -> (Vec<f32>, Vec<f32>) {
        let n = r.len();
        let ab = [alpha, beta];
        {
            let mut pool = self.pool.lock().unwrap();
            pool.ensure_data(&self.device, &self.queue, r);
            pool.ensure_data(&self.device, &self.queue, s);
            pool.ensure_data(&self.device, &self.queue, gamma);
            pool.ensure_data(&self.device, &self.queue, omega);
            pool.ensure_data(&self.device, &self.queue, &ab);
            pool.ensure_scratch(&self.device, 0, (n * 4) as u64); // dr
            pool.ensure_scratch(&self.device, 1, (n * 4) as u64); // ds
            let params = KerrDerivParams { n_bands: n as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
            pool.write_uniform(&self.device, &self.queue, &params);
        }
        let pool = self.pool.lock().unwrap();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.kerr_deriv_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pool.data_ref(r).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pool.data_ref(s).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pool.scratch_ref(0).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pool.scratch_ref(1).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pool.data_ref(gamma).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: pool.data_ref(omega).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: pool.uniform_ref().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: pool.data_ref(&ab).as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        { let mut pass = encoder.begin_compute_pass(&Default::default());
          pass.set_pipeline(&self.kerr_deriv_pipeline);
          pass.set_bind_group(0, &bind_group, &[]);
          pass.dispatch_workgroups((n as u32 + 63) / 64, 1, 1); }
        self.queue.submit(Some(encoder.finish()));
        drop(pool);
        let dr = self.pool.lock().unwrap().readback_scratch(&self.device, &self.queue, 0, n);
        let ds = self.pool.lock().unwrap().readback_scratch(&self.device, &self.queue, 1, n);
        (dr, ds)
    }

    /// Kerr-ODE forward via host-side RK4 with GPU derivative evaluations. Pooled.
    pub(crate) fn gpu_kerr_ode(&self, weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
        let n_bands = weights.gamma_raw.len();
        let n_embd = n_bands * 2;
        let n_steps = weights.rk4_n_steps;
        let dt = 1.0 / n_steps as f32;

        let mut r = vec![0.0f32; n_bands];
        let mut s = vec![0.0f32; n_bands];
        for k in 0..n_bands { r[k] = x[k * 2]; s[k] = x[k * 2 + 1]; }

        fn softplus(x: f32) -> f32 { if x > 20.0 { x } else { (1.0 + x.exp()).ln() } }
        let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();

        for _ in 0..n_steps {
            let (k1r, k1s) = self.gpu_kerr_derivative(&r, &s, &gamma, &weights.omega, weights.alpha, weights.beta);
            let r2: Vec<f32> = r.iter().zip(&k1r).map(|(&ri, &k)| ri + 0.5 * dt * k).collect();
            let s2: Vec<f32> = s.iter().zip(&k1s).map(|(&si, &k)| si + 0.5 * dt * k).collect();
            let (k2r, k2s) = self.gpu_kerr_derivative(&r2, &s2, &gamma, &weights.omega, weights.alpha, weights.beta);
            let r3: Vec<f32> = r.iter().zip(&k2r).map(|(&ri, &k)| ri + 0.5 * dt * k).collect();
            let s3: Vec<f32> = s.iter().zip(&k2s).map(|(&si, &k)| si + 0.5 * dt * k).collect();
            let (k3r, k3s) = self.gpu_kerr_derivative(&r3, &s3, &gamma, &weights.omega, weights.alpha, weights.beta);
            let r4: Vec<f32> = r.iter().zip(&k3r).map(|(&ri, &k)| ri + dt * k).collect();
            let s4: Vec<f32> = s.iter().zip(&k3s).map(|(&si, &k)| si + dt * k).collect();
            let (k4r, k4s) = self.gpu_kerr_derivative(&r4, &s4, &gamma, &weights.omega, weights.alpha, weights.beta);
            for k in 0..n_bands {
                r[k] += dt / 6.0 * (k1r[k] + 2.0 * k2r[k] + 2.0 * k3r[k] + k4r[k]);
                s[k] += dt / 6.0 * (k1s[k] + 2.0 * k2s[k] + 2.0 * k3s[k] + k4s[k]);
            }
        }

        let mut out = vec![0.0f32; n_embd];
        for k in 0..n_bands { out[k * 2] = r[k]; out[k * 2 + 1] = s[k]; }
        out
    }

    // ─── Batched forward dispatches ──────────────────────────────

    /// Batched matvec: y[pos] = W @ x[pos] + b. Pooled.
    pub(crate) fn gpu_matvec_batch(&self, w_flat: &[f32], x_flat: &[f32], b: &[f32], out_dim: usize, in_dim: usize, n_pos: usize) -> Vec<f32> {
        let use_bias = if b.is_empty() { 0u32 } else { 1u32 };
        let out_total = n_pos * out_dim;
        {
            let mut pool = self.pool.lock().unwrap();
            pool.ensure_data(&self.device, &self.queue, w_flat);
            pool.ensure_data(&self.device, &self.queue, x_flat);
            let b_data = if b.is_empty() { &[0.0f32][..] } else { b };
            pool.ensure_data(&self.device, &self.queue, b_data);
            pool.ensure_scratch(&self.device, 0, (out_total * 4) as u64);
            let params = MatvecBatchParams { out_dim: out_dim as u32, in_dim: in_dim as u32, n_pos: n_pos as u32, use_bias };
            pool.write_uniform(&self.device, &self.queue, &params);
        }
        let b_data = if b.is_empty() { &[0.0f32][..] } else { b };
        let pool = self.pool.lock().unwrap();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.matvec_batch_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pool.data_ref(w_flat).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pool.data_ref(x_flat).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pool.data_ref(b_data).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pool.scratch_ref(0).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pool.uniform_ref().as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        { let mut pass = encoder.begin_compute_pass(&Default::default());
          pass.set_pipeline(&self.matvec_batch_pipeline);
          pass.set_bind_group(0, &bind_group, &[]);
          pass.dispatch_workgroups((out_total as u32 + 63) / 64, 1, 1); }
        self.queue.submit(Some(encoder.finish()));
        drop(pool);
        self.pool.lock().unwrap().readback_scratch(&self.device, &self.queue, 0, out_total)
    }

    /// Batched layer norm. Pooled.
    pub(crate) fn gpu_layer_norm_batch(&self, x_flat: &[f32], weight: &[f32], bias: &[f32], dim: usize, n_pos: usize) -> Vec<f32> {
        let out_total = n_pos * dim;
        {
            let mut pool = self.pool.lock().unwrap();
            pool.ensure_data(&self.device, &self.queue, x_flat);
            pool.ensure_data(&self.device, &self.queue, weight);
            pool.ensure_data(&self.device, &self.queue, bias);
            pool.ensure_scratch(&self.device, 0, (out_total * 4) as u64);
            let params = LayerNormBatchParams { dim: dim as u32, n_pos: n_pos as u32, _pad1: 0, _pad2: 0 };
            pool.write_uniform(&self.device, &self.queue, &params);
        }
        let pool = self.pool.lock().unwrap();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.layer_norm_batch_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pool.data_ref(x_flat).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pool.data_ref(weight).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pool.data_ref(bias).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pool.scratch_ref(0).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pool.uniform_ref().as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        { let mut pass = encoder.begin_compute_pass(&Default::default());
          pass.set_pipeline(&self.layer_norm_batch_pipeline);
          pass.set_bind_group(0, &bind_group, &[]);
          pass.dispatch_workgroups(n_pos as u32, 1, 1); }
        self.queue.submit(Some(encoder.finish()));
        drop(pool);
        self.pool.lock().unwrap().readback_scratch(&self.device, &self.queue, 0, out_total)
    }

    /// Batched Kerr derivative. Pooled. Returns (dr_flat, ds_flat).
    pub(crate) fn gpu_kerr_derivative_batch(
        &self, r_flat: &[f32], s_flat: &[f32],
        gamma: &[f32], omega: &[f32], alpha: f32, beta: f32,
        n_bands: usize, n_pos: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let total = n_pos * n_bands;
        let ab = [alpha, beta];
        {
            let mut pool = self.pool.lock().unwrap();
            pool.ensure_data(&self.device, &self.queue, r_flat);
            pool.ensure_data(&self.device, &self.queue, s_flat);
            pool.ensure_data(&self.device, &self.queue, gamma);
            pool.ensure_data(&self.device, &self.queue, omega);
            pool.ensure_data(&self.device, &self.queue, &ab);
            pool.ensure_scratch(&self.device, 0, (total * 4) as u64); // dr
            pool.ensure_scratch(&self.device, 1, (total * 4) as u64); // ds
            let params = KerrDerivBatchParams { n_bands: n_bands as u32, n_pos: n_pos as u32, _pad1: 0, _pad2: 0 };
            pool.write_uniform(&self.device, &self.queue, &params);
        }
        let pool = self.pool.lock().unwrap();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.kerr_deriv_batch_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pool.data_ref(r_flat).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pool.data_ref(s_flat).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pool.scratch_ref(0).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pool.scratch_ref(1).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pool.data_ref(gamma).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: pool.data_ref(omega).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: pool.uniform_ref().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: pool.data_ref(&ab).as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        { let mut pass = encoder.begin_compute_pass(&Default::default());
          pass.set_pipeline(&self.kerr_deriv_batch_pipeline);
          pass.set_bind_group(0, &bind_group, &[]);
          pass.dispatch_workgroups((total as u32 + 63) / 64, 1, 1); }
        self.queue.submit(Some(encoder.finish()));
        drop(pool);
        let dr = self.pool.lock().unwrap().readback_scratch(&self.device, &self.queue, 0, total);
        let ds = self.pool.lock().unwrap().readback_scratch(&self.device, &self.queue, 1, total);
        (dr, ds)
    }

    /// Batched Kerr-ODE forward: RK4 for all positions. Uses pooled gpu_kerr_derivative_batch.
    pub(crate) fn gpu_kerr_ode_batch(&self, weights: &KerrWeights, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n_pos = xs.len();
        let n_bands = weights.gamma_raw.len();
        let n_embd = n_bands * 2;
        let n_steps = weights.rk4_n_steps;
        let dt = 1.0 / n_steps as f32;

        let mut r = vec![0.0f32; n_pos * n_bands];
        let mut s = vec![0.0f32; n_pos * n_bands];
        for pos in 0..n_pos {
            for k in 0..n_bands { r[pos * n_bands + k] = xs[pos][k * 2]; s[pos * n_bands + k] = xs[pos][k * 2 + 1]; }
        }

        fn softplus(x: f32) -> f32 { if x > 20.0 { x } else { (1.0 + x.exp()).ln() } }
        let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();
        let total = n_pos * n_bands;

        for _ in 0..n_steps {
            let (k1r, k1s) = self.gpu_kerr_derivative_batch(&r, &s, &gamma, &weights.omega, weights.alpha, weights.beta, n_bands, n_pos);
            let r2: Vec<f32> = r.iter().zip(&k1r).map(|(&ri, &k)| ri + 0.5 * dt * k).collect();
            let s2: Vec<f32> = s.iter().zip(&k1s).map(|(&si, &k)| si + 0.5 * dt * k).collect();
            let (k2r, k2s) = self.gpu_kerr_derivative_batch(&r2, &s2, &gamma, &weights.omega, weights.alpha, weights.beta, n_bands, n_pos);
            let r3: Vec<f32> = r.iter().zip(&k2r).map(|(&ri, &k)| ri + 0.5 * dt * k).collect();
            let s3: Vec<f32> = s.iter().zip(&k2s).map(|(&si, &k)| si + 0.5 * dt * k).collect();
            let (k3r, k3s) = self.gpu_kerr_derivative_batch(&r3, &s3, &gamma, &weights.omega, weights.alpha, weights.beta, n_bands, n_pos);
            let r4: Vec<f32> = r.iter().zip(&k3r).map(|(&ri, &k)| ri + dt * k).collect();
            let s4: Vec<f32> = s.iter().zip(&k3s).map(|(&si, &k)| si + dt * k).collect();
            let (k4r, k4s) = self.gpu_kerr_derivative_batch(&r4, &s4, &gamma, &weights.omega, weights.alpha, weights.beta, n_bands, n_pos);
            for i in 0..total {
                r[i] += dt / 6.0 * (k1r[i] + 2.0 * k2r[i] + 2.0 * k3r[i] + k4r[i]);
                s[i] += dt / 6.0 * (k1s[i] + 2.0 * k2s[i] + 2.0 * k3s[i] + k4s[i]);
            }
        }

        (0..n_pos).map(|pos| {
            let mut out = vec![0.0f32; n_embd];
            for k in 0..n_bands { out[k * 2] = r[pos * n_bands + k]; out[k * 2 + 1] = s[pos * n_bands + k]; }
            out
        }).collect()
    }
}
