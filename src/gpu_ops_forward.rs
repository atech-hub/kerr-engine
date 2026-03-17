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

    /// Matvec: y = W @ x + b. Pooled scratch + staging reuse. Inputs always fresh.
    pub(crate) fn gpu_matvec(&self, w_flat: &[f32], x: &[f32], b: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        let use_bias = if b.is_empty() { 0u32 } else { 1u32 };

        // Inputs always fresh (never cached — data changes every call)
        let w_buf = self.storage_buf("w", w_flat);
        let x_buf = self.storage_buf("x", x);
        let b_data = if b.is_empty() { &[0.0f32][..] } else { b };
        let b_buf = self.storage_buf("b", b_data);

        // Scratch + uniform from pool (reused across dispatches)
        {
            let mut pool = self.pool.lock().unwrap();
            pool.ensure_scratch(&self.device, 0, (out_dim * 4) as u64);
            let params = MatvecParams { out_dim: out_dim as u32, in_dim: in_dim as u32, use_bias, _pad: 0 };
            pool.write_uniform(&self.device, &self.queue, &params);
        }
        let pool = self.pool.lock().unwrap();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.matvec_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pool.scratch_ref(0).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pool.uniform_ref().as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        { let mut pass = encoder.begin_compute_pass(&Default::default());
          pass.set_pipeline(&self.matvec_pipeline);
          pass.set_bind_group(0, &bind_group, &[]);
          pass.dispatch_workgroups(out_dim as u32, 1, 1); } // one workgroup per row (tiled)
        self.queue.submit(Some(encoder.finish()));
        drop(pool);
        self.pool.lock().unwrap().readback_scratch(&self.device, &self.queue, 0, out_dim)
    }

    /// Layer norm on GPU. Pooled scratch + staging.
    pub(crate) fn gpu_layer_norm(&self, x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
        let dim = x.len();
        let x_buf = self.storage_buf("x", x);
        let w_buf = self.storage_buf("weight", weight);
        let b_buf = self.storage_buf("bias", bias);
        {
            let mut pool = self.pool.lock().unwrap();
            pool.ensure_scratch(&self.device, 0, (dim * 4) as u64);
            let params = LayerNormParams { dim: dim as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
            pool.write_uniform(&self.device, &self.queue, &params);
        }
        let pool = self.pool.lock().unwrap();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.layer_norm_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
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

    /// Kerr derivative on GPU. Pooled scratch + staging.
    pub(crate) fn gpu_kerr_derivative(&self, r: &[f32], s: &[f32], gamma: &[f32], omega: &[f32], alpha: f32, beta: f32) -> (Vec<f32>, Vec<f32>) {
        let n = r.len();
        let ab = [alpha, beta];
        let r_buf = self.storage_buf("r", r);
        let s_buf = self.storage_buf("s", s);
        let gamma_buf = self.storage_buf("gamma", gamma);
        let omega_buf = self.storage_buf("omega", omega);
        let ab_buf = self.storage_buf("ab", &ab);
        {
            let mut pool = self.pool.lock().unwrap();
            pool.ensure_scratch(&self.device, 0, (n * 4) as u64);
            pool.ensure_scratch(&self.device, 1, (n * 4) as u64);
            let params = KerrDerivParams { n_bands: n as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
            pool.write_uniform(&self.device, &self.queue, &params);
        }
        let pool = self.pool.lock().unwrap();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.kerr_deriv_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: r_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: s_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pool.scratch_ref(0).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pool.scratch_ref(1).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: gamma_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: omega_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: pool.uniform_ref().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: ab_buf.as_entire_binding() },
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

        const MAG_BOUND: f32 = 50.0;
        const DERIV_BOUND: f32 = 1000.0;
        fn clamp_single(r: &mut [f32], s: &mut [f32], bound: f32) {
            for k in 0..r.len() {
                let mag = (r[k] * r[k] + s[k] * s[k]).sqrt();
                if mag > bound { let sc = bound / mag; r[k] *= sc; s[k] *= sc; }
            }
        }

        for _ in 0..n_steps {
            let (mut k1r, mut k1s) = self.gpu_kerr_derivative(&r, &s, &gamma, &weights.omega, weights.alpha, weights.beta);
            clamp_single(&mut k1r, &mut k1s, DERIV_BOUND);

            let mut r2: Vec<f32> = r.iter().zip(&k1r).map(|(&ri, &k)| ri + 0.5 * dt * k).collect();
            let mut s2: Vec<f32> = s.iter().zip(&k1s).map(|(&si, &k)| si + 0.5 * dt * k).collect();
            clamp_single(&mut r2, &mut s2, MAG_BOUND);

            let (mut k2r, mut k2s) = self.gpu_kerr_derivative(&r2, &s2, &gamma, &weights.omega, weights.alpha, weights.beta);
            clamp_single(&mut k2r, &mut k2s, DERIV_BOUND);

            let mut r3: Vec<f32> = r.iter().zip(&k2r).map(|(&ri, &k)| ri + 0.5 * dt * k).collect();
            let mut s3: Vec<f32> = s.iter().zip(&k2s).map(|(&si, &k)| si + 0.5 * dt * k).collect();
            clamp_single(&mut r3, &mut s3, MAG_BOUND);

            let (mut k3r, mut k3s) = self.gpu_kerr_derivative(&r3, &s3, &gamma, &weights.omega, weights.alpha, weights.beta);
            clamp_single(&mut k3r, &mut k3s, DERIV_BOUND);

            let mut r4: Vec<f32> = r.iter().zip(&k3r).map(|(&ri, &k)| ri + dt * k).collect();
            let mut s4: Vec<f32> = s.iter().zip(&k3s).map(|(&si, &k)| si + dt * k).collect();
            clamp_single(&mut r4, &mut s4, MAG_BOUND);

            let (mut k4r, mut k4s) = self.gpu_kerr_derivative(&r4, &s4, &gamma, &weights.omega, weights.alpha, weights.beta);
            clamp_single(&mut k4r, &mut k4s, DERIV_BOUND);

            for k in 0..n_bands {
                r[k] += dt / 6.0 * (k1r[k] + 2.0 * k2r[k] + 2.0 * k3r[k] + k4r[k]);
                s[k] += dt / 6.0 * (k1s[k] + 2.0 * k2s[k] + 2.0 * k3s[k] + k4s[k]);
            }
            clamp_single(&mut r, &mut s, MAG_BOUND);
        }

        let mut out = vec![0.0f32; n_embd];
        for k in 0..n_bands { out[k * 2] = r[k]; out[k * 2 + 1] = s[k]; }
        out
    }

    // ─── Batched forward dispatches ──────────────────────────────

    /// Batched matvec: y[pos] = W @ x[pos] + b. Pooled scratch + staging.
    pub(crate) fn gpu_matvec_batch(&self, w_flat: &[f32], x_flat: &[f32], b: &[f32], out_dim: usize, in_dim: usize, n_pos: usize) -> Vec<f32> {
        let use_bias = if b.is_empty() { 0u32 } else { 1u32 };
        let out_total = n_pos * out_dim;
        let w_buf = self.storage_buf("w", w_flat);
        let x_buf = self.storage_buf("x", x_flat);
        let b_data = if b.is_empty() { &[0.0f32][..] } else { b };
        let b_buf = self.storage_buf("b", b_data);
        {
            let mut pool = self.pool.lock().unwrap();
            pool.ensure_scratch(&self.device, 0, (out_total * 4) as u64);
            let params = MatvecBatchParams { out_dim: out_dim as u32, in_dim: in_dim as u32, n_pos: n_pos as u32, use_bias };
            pool.write_uniform(&self.device, &self.queue, &params);
        }
        let pool = self.pool.lock().unwrap();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.matvec_batch_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pool.scratch_ref(0).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pool.uniform_ref().as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        { let mut pass = encoder.begin_compute_pass(&Default::default());
          pass.set_pipeline(&self.matvec_batch_pipeline);
          pass.set_bind_group(0, &bind_group, &[]);
          // Split across x,y dimensions to stay under 65535 limit
          pass.dispatch_workgroups(out_dim as u32, n_pos as u32, 1); } // tiled: (out_dim, n_pos)
        self.queue.submit(Some(encoder.finish()));
        drop(pool);
        self.pool.lock().unwrap().readback_scratch(&self.device, &self.queue, 0, out_total)
    }

    /// Batched layer norm. Pooled scratch + staging.
    pub(crate) fn gpu_layer_norm_batch(&self, x_flat: &[f32], weight: &[f32], bias: &[f32], dim: usize, n_pos: usize) -> Vec<f32> {
        let out_total = n_pos * dim;
        let x_buf = self.storage_buf("x", x_flat);
        let w_buf = self.storage_buf("w", weight);
        let b_buf = self.storage_buf("b", bias);
        {
            let mut pool = self.pool.lock().unwrap();
            pool.ensure_scratch(&self.device, 0, (out_total * 4) as u64);
            let params = LayerNormBatchParams { dim: dim as u32, n_pos: n_pos as u32, _pad1: 0, _pad2: 0 };
            pool.write_uniform(&self.device, &self.queue, &params);
        }
        let pool = self.pool.lock().unwrap();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.layer_norm_batch_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
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

    /// Batched Kerr derivative. Pooled scratch + staging.
    pub(crate) fn gpu_kerr_derivative_batch(
        &self, r_flat: &[f32], s_flat: &[f32],
        gamma: &[f32], omega: &[f32], alpha: f32, beta: f32,
        n_bands: usize, n_pos: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let total = n_pos * n_bands;
        let ab = [alpha, beta];
        let r_buf = self.storage_buf("r", r_flat);
        let s_buf = self.storage_buf("s", s_flat);
        let gamma_buf = self.storage_buf("gamma", gamma);
        let omega_buf = self.storage_buf("omega", omega);
        let ab_buf = self.storage_buf("ab", &ab);
        {
            let mut pool = self.pool.lock().unwrap();
            pool.ensure_scratch(&self.device, 0, (total * 4) as u64);
            pool.ensure_scratch(&self.device, 1, (total * 4) as u64);
            let params = KerrDerivBatchParams { n_bands: n_bands as u32, n_pos: n_pos as u32, _pad1: 0, _pad2: 0 };
            pool.write_uniform(&self.device, &self.queue, &params);
        }
        let pool = self.pool.lock().unwrap();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.kerr_deriv_batch_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: r_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: s_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pool.scratch_ref(0).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pool.scratch_ref(1).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: gamma_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: omega_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: pool.uniform_ref().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: ab_buf.as_entire_binding() },
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

    /// Fused batched Kerr-ODE forward: chains all derivative + midpoint + combine
    /// dispatches in ONE command encoder, ONE submit, ONE readback.
    /// Uses existing validated kerr_step_batch.wgsl + vec_scale_add + rk4_combine.
    /// Per RK4 step: 4 derivative dispatches + 6 midpoint dispatches + 2 combine = 12.
    /// Total: 12 × n_steps dispatches, 1 submit. Down from n_steps × 4 submits.
    pub(crate) fn gpu_kerr_ode_batch_fused(&self, weights: &KerrWeights, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n_pos = xs.len();
        let n_bands = weights.gamma_raw.len();
        let n_embd = n_bands * 2;
        let n_steps = weights.rk4_n_steps;
        let dt = 1.0 / n_steps as f32;
        let total = n_pos * n_bands;

        // Deinterleave input into flat r/s arrays
        let mut r_flat = vec![0.0f32; total];
        let mut s_flat = vec![0.0f32; total];
        for pos in 0..n_pos {
            for k in 0..n_bands {
                r_flat[pos * n_bands + k] = xs[pos][k * 2];
                s_flat[pos * n_bands + k] = xs[pos][k * 2 + 1];
            }
        }

        fn softplus(x: f32) -> f32 { if x > 20.0 { x } else { (1.0 + x.exp()).ln() } }
        let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();
        let ab = [weights.alpha, weights.beta];

        let buf_size = (total * 4) as u64;
        let make_rw = |label| self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label), size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let make_init = |label, data: &[f32]| self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label), contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        // State buffers (r/s are read-write, updated in-place via rk4_combine)
        let r_buf = make_init("f_r", &r_flat);
        let s_buf = make_init("f_s", &s_flat);
        // Scratch buffers for k1-k4 derivatives
        let k1r = make_rw("k1r"); let k1s = make_rw("k1s");
        let k2r = make_rw("k2r"); let k2s = make_rw("k2s");
        let k3r = make_rw("k3r"); let k3s = make_rw("k3s");
        let k4r = make_rw("k4r"); let k4s = make_rw("k4s");
        // Midpoint buffers
        let r_mid = make_rw("r_mid"); let s_mid = make_rw("s_mid");
        // Output buffers for rk4_combine (swap with r_buf/s_buf)
        let r_new = make_rw("r_new"); let s_new = make_rw("s_new");

        let gamma_buf = self.storage_buf("f_gamma", &gamma);
        let omega_buf = self.storage_buf("f_omega", &weights.omega);
        let ab_buf = self.storage_buf("f_ab", &ab);

        // Uniform buffers for kerr derivative
        let deriv_params = KerrDerivBatchParams { n_bands: n_bands as u32, n_pos: n_pos as u32, _pad1: 0, _pad2: 0 };
        let deriv_params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("f_deriv_params"), contents: bytemuck::bytes_of(&deriv_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Pre-create uniform bufs for vec_scale_add (different scales)
        let vsa_half_dt = VecScaleAddParams { len: total as u32, scale: 0.5 * dt, _pad1: 0, _pad2: 0 };
        let vsa_full_dt = VecScaleAddParams { len: total as u32, scale: dt, _pad1: 0, _pad2: 0 };
        let vsa_half_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vsa_half"), contents: bytemuck::bytes_of(&vsa_half_dt),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let vsa_full_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vsa_full"), contents: bytemuck::bytes_of(&vsa_full_dt),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // rk4_combine params
        let rc_params = Rk4CombineParams { len: total as u32, dt_over_6: dt / 6.0, _pad1: 0, _pad2: 0 };
        let rc_params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rc_params"), contents: bytemuck::bytes_of(&rc_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let wg = (total as u32 + 63) / 64;

        // Helper: create derivative bind group (r_in, s_in → dr_out, ds_out)
        let deriv_bg = |r_in: &wgpu::Buffer, s_in: &wgpu::Buffer, dr: &wgpu::Buffer, ds: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &self.kerr_deriv_batch_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: r_in.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: s_in.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: dr.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: ds.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: gamma_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: omega_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: deriv_params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: ab_buf.as_entire_binding() },
                ],
            })
        };

        // Helper: create vec_scale_add bind group (a + scale * b → y)
        let vsa_bg = |a: &wgpu::Buffer, b: &wgpu::Buffer, y: &wgpu::Buffer, params: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &self.vec_scale_add_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
                ],
            })
        };

        // Pre-create all bind groups (they reference persistent buffers)
        // k1: deriv(r, s) → k1r, k1s
        let bg_k1 = deriv_bg(&r_buf, &s_buf, &k1r, &k1s);
        // midpoint for k2: r + 0.5*dt*k1r → r_mid, s + 0.5*dt*k1s → s_mid
        let bg_mid_r2 = vsa_bg(&r_buf, &k1r, &r_mid, &vsa_half_buf);
        let bg_mid_s2 = vsa_bg(&s_buf, &k1s, &s_mid, &vsa_half_buf);
        // k2: deriv(r_mid, s_mid) → k2r, k2s
        let bg_k2 = deriv_bg(&r_mid, &s_mid, &k2r, &k2s);
        // midpoint for k3: r + 0.5*dt*k2r → r_mid, s + 0.5*dt*k2s → s_mid
        let bg_mid_r3 = vsa_bg(&r_buf, &k2r, &r_mid, &vsa_half_buf);
        let bg_mid_s3 = vsa_bg(&s_buf, &k2s, &s_mid, &vsa_half_buf);
        // k3: deriv(r_mid, s_mid) → k3r, k3s
        let bg_k3 = deriv_bg(&r_mid, &s_mid, &k3r, &k3s);
        // midpoint for k4: r + dt*k3r → r_mid, s + dt*k3s → s_mid
        let bg_mid_r4 = vsa_bg(&r_buf, &k3r, &r_mid, &vsa_full_buf);
        let bg_mid_s4 = vsa_bg(&s_buf, &k3s, &s_mid, &vsa_full_buf);
        // k4: deriv(r_mid, s_mid) → k4r, k4s
        let bg_k4 = deriv_bg(&r_mid, &s_mid, &k4r, &k4s);
        // combine: r_new = r + dt/6*(k1r + 2*k2r + 2*k3r + k4r)
        let bg_combine_r = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.rk4_combine_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: r_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k1r.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: k2r.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: k3r.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: k4r.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: r_new.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: rc_params_buf.as_entire_binding() },
            ],
        });
        let bg_combine_s = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.rk4_combine_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: s_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k1s.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: k2s.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: k3s.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: k4s.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: s_new.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: rc_params_buf.as_entire_binding() },
            ],
        });
        // Copy r_new→r, s_new→s for next step
        // wgpu copy_buffer_to_buffer is used between encoder passes

        // Chain ALL RK4 steps in ONE command encoder
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for _ in 0..n_steps {
            // k1 = f(r, s)
            { let mut p = encoder.begin_compute_pass(&Default::default());
              p.set_pipeline(&self.kerr_deriv_batch_pipeline); p.set_bind_group(0, &bg_k1, &[]); p.dispatch_workgroups(wg, 1, 1); }
            // r_mid = r + 0.5*dt*k1r, s_mid = s + 0.5*dt*k1s
            { let mut p = encoder.begin_compute_pass(&Default::default());
              p.set_pipeline(&self.vec_scale_add_pipeline); p.set_bind_group(0, &bg_mid_r2, &[]); p.dispatch_workgroups(wg, 1, 1); }
            { let mut p = encoder.begin_compute_pass(&Default::default());
              p.set_pipeline(&self.vec_scale_add_pipeline); p.set_bind_group(0, &bg_mid_s2, &[]); p.dispatch_workgroups(wg, 1, 1); }
            // k2 = f(r_mid, s_mid)
            { let mut p = encoder.begin_compute_pass(&Default::default());
              p.set_pipeline(&self.kerr_deriv_batch_pipeline); p.set_bind_group(0, &bg_k2, &[]); p.dispatch_workgroups(wg, 1, 1); }
            // r_mid = r + 0.5*dt*k2r, s_mid = s + 0.5*dt*k2s
            { let mut p = encoder.begin_compute_pass(&Default::default());
              p.set_pipeline(&self.vec_scale_add_pipeline); p.set_bind_group(0, &bg_mid_r3, &[]); p.dispatch_workgroups(wg, 1, 1); }
            { let mut p = encoder.begin_compute_pass(&Default::default());
              p.set_pipeline(&self.vec_scale_add_pipeline); p.set_bind_group(0, &bg_mid_s3, &[]); p.dispatch_workgroups(wg, 1, 1); }
            // k3 = f(r_mid, s_mid)
            { let mut p = encoder.begin_compute_pass(&Default::default());
              p.set_pipeline(&self.kerr_deriv_batch_pipeline); p.set_bind_group(0, &bg_k3, &[]); p.dispatch_workgroups(wg, 1, 1); }
            // r_mid = r + dt*k3r, s_mid = s + dt*k3s
            { let mut p = encoder.begin_compute_pass(&Default::default());
              p.set_pipeline(&self.vec_scale_add_pipeline); p.set_bind_group(0, &bg_mid_r4, &[]); p.dispatch_workgroups(wg, 1, 1); }
            { let mut p = encoder.begin_compute_pass(&Default::default());
              p.set_pipeline(&self.vec_scale_add_pipeline); p.set_bind_group(0, &bg_mid_s4, &[]); p.dispatch_workgroups(wg, 1, 1); }
            // k4 = f(r_mid, s_mid)
            { let mut p = encoder.begin_compute_pass(&Default::default());
              p.set_pipeline(&self.kerr_deriv_batch_pipeline); p.set_bind_group(0, &bg_k4, &[]); p.dispatch_workgroups(wg, 1, 1); }
            // combine: r_new = r + dt/6*(k1r + 2*k2r + 2*k3r + k4r)
            { let mut p = encoder.begin_compute_pass(&Default::default());
              p.set_pipeline(&self.rk4_combine_pipeline); p.set_bind_group(0, &bg_combine_r, &[]); p.dispatch_workgroups(wg, 1, 1); }
            { let mut p = encoder.begin_compute_pass(&Default::default());
              p.set_pipeline(&self.rk4_combine_pipeline); p.set_bind_group(0, &bg_combine_s, &[]); p.dispatch_workgroups(wg, 1, 1); }
            // Copy r_new→r, s_new→s for next step
            encoder.copy_buffer_to_buffer(&r_new, 0, &r_buf, 0, buf_size);
            encoder.copy_buffer_to_buffer(&s_new, 0, &s_buf, 0, buf_size);
        }
        self.queue.submit(Some(encoder.finish()));

        // ONE readback
        let r_out = self.readback(&r_buf, total);
        let s_out = self.readback(&s_buf, total);

        // Reinterleave into output
        (0..n_pos).map(|pos| {
            let mut out = vec![0.0f32; n_embd];
            for k in 0..n_bands {
                out[k * 2] = r_out[pos * n_bands + k];
                out[k * 2 + 1] = s_out[pos * n_bands + k];
            }
            out
        }).collect()
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

        // Magnitude bound for GPU numerical stability at 768-dim+.
        // GPU FP differences in Kerr derivative can cause |Z| to grow past
        // what damping can control. Clamp intermediates AND final state.
        // CPU-validated range is ±7; threshold 50.0 gives 7x headroom.
        // At correct operating range this is a no-op.
        const MAG_BOUND: f32 = 50.0;
        #[inline(always)]
        fn clamp_mag(v: &mut [f32], s: &mut [f32], total: usize, bound: f32) {
            for i in 0..total {
                let mag = (v[i] * v[i] + s[i] * s[i]).sqrt();
                if mag > bound {
                    let sc = bound / mag;
                    v[i] *= sc;
                    s[i] *= sc;
                }
            }
        }

        // Clamp both states AND derivatives. The GPU derivative shader can
        // return Inf/NaN if inputs have large magnitudes from FP differences.
        // Clamping derivatives prevents Inf from propagating into the RK4 sum.
        const DERIV_BOUND: f32 = 1000.0;

        for _ in 0..n_steps {
            let (mut k1r, mut k1s) = self.gpu_kerr_derivative_batch(&r, &s, &gamma, &weights.omega, weights.alpha, weights.beta, n_bands, n_pos);
            clamp_mag(&mut k1r, &mut k1s, total, DERIV_BOUND);

            let mut r2: Vec<f32> = r.iter().zip(&k1r).map(|(&ri, &k)| ri + 0.5 * dt * k).collect();
            let mut s2: Vec<f32> = s.iter().zip(&k1s).map(|(&si, &k)| si + 0.5 * dt * k).collect();
            clamp_mag(&mut r2, &mut s2, total, MAG_BOUND);

            let (mut k2r, mut k2s) = self.gpu_kerr_derivative_batch(&r2, &s2, &gamma, &weights.omega, weights.alpha, weights.beta, n_bands, n_pos);
            clamp_mag(&mut k2r, &mut k2s, total, DERIV_BOUND);

            let mut r3: Vec<f32> = r.iter().zip(&k2r).map(|(&ri, &k)| ri + 0.5 * dt * k).collect();
            let mut s3: Vec<f32> = s.iter().zip(&k2s).map(|(&si, &k)| si + 0.5 * dt * k).collect();
            clamp_mag(&mut r3, &mut s3, total, MAG_BOUND);

            let (mut k3r, mut k3s) = self.gpu_kerr_derivative_batch(&r3, &s3, &gamma, &weights.omega, weights.alpha, weights.beta, n_bands, n_pos);
            clamp_mag(&mut k3r, &mut k3s, total, DERIV_BOUND);

            let mut r4: Vec<f32> = r.iter().zip(&k3r).map(|(&ri, &k)| ri + dt * k).collect();
            let mut s4: Vec<f32> = s.iter().zip(&k3s).map(|(&si, &k)| si + dt * k).collect();
            clamp_mag(&mut r4, &mut s4, total, MAG_BOUND);

            let (mut k4r, mut k4s) = self.gpu_kerr_derivative_batch(&r4, &s4, &gamma, &weights.omega, weights.alpha, weights.beta, n_bands, n_pos);
            clamp_mag(&mut k4r, &mut k4s, total, DERIV_BOUND);

            for i in 0..total {
                r[i] += dt / 6.0 * (k1r[i] + 2.0 * k2r[i] + 2.0 * k3r[i] + k4r[i]);
                s[i] += dt / 6.0 * (k1s[i] + 2.0 * k2s[i] + 2.0 * k3s[i] + k4s[i]);
            }
            clamp_mag(&mut r, &mut s, total, MAG_BOUND);
        }

        (0..n_pos).map(|pos| {
            let mut out = vec![0.0f32; n_embd];
            for k in 0..n_bands { out[k * 2] = r[pos * n_bands + k]; out[k * 2 + 1] = s[pos * n_bands + k]; }
            out
        }).collect()
    }
}
