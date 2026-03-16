//! GPU dispatch operations — buffer management + shader execution.
//!
//! All gpu_matvec, gpu_layer_norm, gpu_kerr_* methods live here.
//! Both single-position and batched variants. Called by gpu_dispatch.rs
//! (the ComputeBackend trait impl).

use crate::gpu_pipelines::*;
use crate::model::*;

use wgpu::util::DeviceExt;

impl GpuBackend {
    // ─── Buffer helpers ──────────────────────────────────────────

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

    /// Run matvec: y = W @ x + b (or y = W @ x if bias is empty).
    /// Uses buffer pool for weight caching and scratch reuse.
    pub(crate) fn gpu_matvec(&self, w_flat: &[f32], x: &[f32], b: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        let use_bias = if b.is_empty() { 0u32 } else { 1u32 };

        // Phase 1: Ensure all buffers exist (mutable borrow of pool)
        {
            let mut pool = self.pool.lock().unwrap();
            pool.ensure_data(&self.device, &self.queue, w_flat);
            pool.ensure_data(&self.device, &self.queue, x);
            let b_data = if b.is_empty() { &[0.0f32][..] } else { b };
            pool.ensure_data(&self.device, &self.queue, b_data);
            pool.ensure_scratch(&self.device, 0, (out_dim * 4) as u64);
            let params = MatvecParams {
                out_dim: out_dim as u32,
                in_dim: in_dim as u32,
                use_bias,
                _pad: 0,
            };
            pool.write_uniform(&self.device, &self.queue, &params);
        }

        // Phase 2: Borrow immutably for bind group + dispatch
        let b_data = if b.is_empty() { &[0.0f32][..] } else { b };
        let pool = self.pool.lock().unwrap();
        let w_buf = pool.data_ref(w_flat);
        let x_buf = pool.data_ref(x);
        let b_buf = pool.data_ref(b_data);
        let y_buf = pool.scratch_ref(0);
        let params_buf = pool.uniform_ref();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matvec_bg"),
            layout: &self.matvec_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: y_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.matvec_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (out_dim as u32 + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        drop(pool);

        // Phase 3: Readback (mutable borrow for staging)
        self.pool.lock().unwrap().readback_scratch(&self.device, &self.queue, 0, out_dim)
    }

    /// Run layer normalization on GPU.
    pub(crate) fn gpu_layer_norm(&self, x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
        let dim = x.len();

        let x_buf = self.storage_buf("x", x);
        let w_buf = self.storage_buf("weight", weight);
        let b_buf = self.storage_buf("bias", bias);
        let y_buf = self.output_buf("y", dim);
        let params = LayerNormParams { dim: dim as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
        let params_buf = self.uniform_buf("params", &params);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ln_bg"),
            layout: &self.layer_norm_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: y_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.layer_norm_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Single workgroup of 128 threads — matches our dim
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.readback(&y_buf, dim)
    }

    /// Run one Kerr derivative evaluation on GPU.
    /// Returns (dr, ds) each of length N_BANDS.
    pub(crate) fn gpu_kerr_derivative(&self, r: &[f32], s: &[f32], gamma: &[f32], omega: &[f32], alpha: f32, beta: f32) -> (Vec<f32>, Vec<f32>) {
        let n = r.len();

        let r_buf = self.storage_buf("r_in", r);
        let s_buf = self.storage_buf("s_in", s);
        let dr_buf = self.output_buf("dr_out", n);
        let ds_buf = self.output_buf("ds_out", n);
        let gamma_buf = self.storage_buf("gamma", gamma);
        let omega_buf = self.storage_buf("omega", omega);
        let params = KerrDerivParams { n_bands: n as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
        let params_buf = self.uniform_buf("params", &params);
        let ab = [alpha, beta];
        let ab_buf = self.storage_buf("alpha_beta", &ab);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("kerr_bg"),
            layout: &self.kerr_deriv_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: r_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: s_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dr_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: ds_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: gamma_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: omega_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: ab_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.kerr_deriv_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n as u32 + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        // Need two readbacks — submit is already done, just copy both
        let dr = self.readback(&dr_buf, n);
        let ds = self.readback(&ds_buf, n);
        (dr, ds)
    }

    /// Kerr-ODE forward via host-side RK4 with GPU derivative evaluations.
    pub(crate) fn gpu_kerr_ode(&self, weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
        let n_bands = weights.gamma_raw.len();
        let n_embd = n_bands * 2;
        let n_steps = weights.rk4_n_steps;
        let dt = 1.0 / n_steps as f32;

        // Unpack interleaved [r0, s0, r1, s1, ...] into separate r, s arrays
        let mut r = vec![0.0f32; n_bands];
        let mut s = vec![0.0f32; n_bands];
        for k in 0..n_bands {
            r[k] = x[k * 2];
            s[k] = x[k * 2 + 1];
        }

        // Pre-compute softplus(gamma_raw) on CPU (trivial)
        fn softplus(x: f32) -> f32 {
            if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
        }
        let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();

        // RK4 integration
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

        // Re-interleave
        let mut out = vec![0.0f32; n_embd];
        for k in 0..n_bands {
            out[k * 2] = r[k];
            out[k * 2 + 1] = s[k];
        }
        out
    }

    /// Batched matvec forward: y[pos] = W @ x[pos] + b for all positions in one dispatch.
    pub(crate) fn gpu_matvec_batch(&self, w_flat: &[f32], x_flat: &[f32], b: &[f32], out_dim: usize, in_dim: usize, n_pos: usize) -> Vec<f32> {
        let use_bias = if b.is_empty() { 0u32 } else { 1u32 };
        let out_total = n_pos * out_dim;

        // Phase 1: ensure pooled buffers
        {
            let mut pool = self.pool.lock().unwrap();
            pool.ensure_data(&self.device, &self.queue, w_flat);
            pool.ensure_data(&self.device, &self.queue, x_flat);
            let b_data = if b.is_empty() { &[0.0f32][..] } else { b };
            pool.ensure_data(&self.device, &self.queue, b_data);
            pool.ensure_scratch(&self.device, 0, (out_total * 4) as u64);
            let params = MatvecBatchParams {
                out_dim: out_dim as u32, in_dim: in_dim as u32,
                n_pos: n_pos as u32, use_bias,
            };
            pool.write_uniform(&self.device, &self.queue, &params);
        }

        // Phase 2: bind + dispatch
        let b_data = if b.is_empty() { &[0.0f32][..] } else { b };
        let pool = self.pool.lock().unwrap();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matvec_batch_bg"),
            layout: &self.matvec_batch_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pool.data_ref(w_flat).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pool.data_ref(x_flat).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pool.data_ref(b_data).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pool.scratch_ref(0).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pool.uniform_ref().as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.matvec_batch_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let total = out_total as u32;
            pass.dispatch_workgroups((total + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        drop(pool);

        self.pool.lock().unwrap().readback_scratch(&self.device, &self.queue, 0, out_total)
    }

    /// Batched layer norm: one workgroup per position, all positions in one dispatch.
    pub(crate) fn gpu_layer_norm_batch(&self, x_flat: &[f32], weight: &[f32], bias: &[f32], dim: usize, n_pos: usize) -> Vec<f32> {
        let out_total = n_pos * dim;

        {
            let mut pool = self.pool.lock().unwrap();
            pool.ensure_data(&self.device, &self.queue, x_flat);
            pool.ensure_data(&self.device, &self.queue, weight);
            pool.ensure_data(&self.device, &self.queue, bias);
            pool.ensure_scratch(&self.device, 0, (out_total * 4) as u64);
            let params = LayerNormBatchParams {
                dim: dim as u32, n_pos: n_pos as u32, _pad1: 0, _pad2: 0,
            };
            pool.write_uniform(&self.device, &self.queue, &params);
        }

        let pool = self.pool.lock().unwrap();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer_norm_batch_bg"),
            layout: &self.layer_norm_batch_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pool.data_ref(x_flat).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pool.data_ref(weight).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pool.data_ref(bias).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pool.scratch_ref(0).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pool.uniform_ref().as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.layer_norm_batch_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_pos as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        drop(pool);

        self.pool.lock().unwrap().readback_scratch(&self.device, &self.queue, 0, out_total)
    }

    /// Batched Kerr derivative: compute dr/ds for all positions in one dispatch.
    /// r_flat/s_flat are [n_pos * n_bands]. Returns (dr_flat, ds_flat).
    pub(crate) fn gpu_kerr_derivative_batch(
        &self, r_flat: &[f32], s_flat: &[f32],
        gamma: &[f32], omega: &[f32], alpha: f32, beta: f32,
        n_bands: usize, n_pos: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let r_buf = self.storage_buf("kdb_r", r_flat);
        let s_buf = self.storage_buf("kdb_s", s_flat);
        let dr_buf = self.output_buf("kdb_dr", n_pos * n_bands);
        let ds_buf = self.output_buf("kdb_ds", n_pos * n_bands);
        let gamma_buf = self.storage_buf("kdb_gamma", gamma);
        let omega_buf = self.storage_buf("kdb_omega", omega);
        let params = KerrDerivBatchParams {
            n_bands: n_bands as u32, n_pos: n_pos as u32, _pad1: 0, _pad2: 0,
        };
        let params_buf = self.uniform_buf("kdb_params", &params);
        let ab = [alpha, beta];
        let ab_buf = self.storage_buf("kdb_ab", &ab);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("kerr_deriv_batch_bg"),
            layout: &self.kerr_deriv_batch_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: r_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: s_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dr_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: ds_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: gamma_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: omega_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: ab_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.kerr_deriv_batch_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let total = (n_pos * n_bands) as u32;
            pass.dispatch_workgroups((total + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        let dr = self.readback(&dr_buf, n_pos * n_bands);
        let ds = self.readback(&ds_buf, n_pos * n_bands);
        (dr, ds)
    }

    /// Batched Kerr derivative backward: compute input + parameter gradients for all positions.
    /// Returns (d_r, d_s, d_gamma, d_omega, d_alpha, d_beta) all flat or scalar.
    /// d_gamma/d_omega are [n_pos * n_bands] — caller reduces across positions.
    pub(crate) fn gpu_kerr_derivative_backward_batch(
        &self,
        d_dr_flat: &[f32], d_ds_flat: &[f32],  // upstream gradients [n_pos * n_bands]
        r_flat: &[f32], s_flat: &[f32],          // cached forward state [n_pos * n_bands]
        gamma: &[f32], omega: &[f32],            // shared params [n_bands]
        alpha: f32, beta: f32,
        n_bands: usize, n_pos: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32, f32) {
        let total = n_pos * n_bands;

        let r_buf = self.storage_buf("kbb_r", r_flat);
        let s_buf = self.storage_buf("kbb_s", s_flat);
        let gamma_buf = self.storage_buf("kbb_gamma", gamma);
        let omega_buf = self.storage_buf("kbb_omega", omega);
        let ddr_buf = self.storage_buf("kbb_ddr", d_dr_flat);
        let dds_buf = self.storage_buf("kbb_dds", d_ds_flat);
        let dr_buf = self.output_buf("kbb_dr", total);
        let ds_buf = self.output_buf("kbb_ds", total);
        let dg_buf = self.output_buf("kbb_dg", total);
        let dom_buf = self.output_buf("kbb_dom", total);
        let da_buf = self.output_buf("kbb_da", total);
        let db_buf = self.output_buf("kbb_db", total);

        let params = KerrBwdBatchParams {
            n_bands: n_bands as u32, n_pos: n_pos as u32, alpha, beta,
        };
        let params_buf = self.uniform_buf("kbb_params", &params);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("kerr_bwd_batch_bg"),
            layout: &self.kerr_bwd_batch_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: r_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: s_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gamma_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: omega_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: ddr_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: dds_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: dr_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: ds_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: dg_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: dom_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: da_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: db_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.kerr_bwd_batch_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((total as u32 + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        let d_r = self.readback(&dr_buf, total);
        let d_s = self.readback(&ds_buf, total);
        let d_gamma = self.readback(&dg_buf, total);
        let d_omega = self.readback(&dom_buf, total);
        let da_partials = self.readback(&da_buf, total);
        let db_partials = self.readback(&db_buf, total);

        // CPU reduction for d_alpha and d_beta
        let d_alpha: f32 = da_partials.iter().sum();
        let d_beta: f32 = db_partials.iter().sum();

        (d_r, d_s, d_gamma, d_omega, d_alpha, d_beta)
    }

    /// Batched Kerr-ODE forward: RK4 integration for all positions simultaneously.
    /// x is [n_pos][n_embd] interleaved. Returns [n_pos][n_embd].
    pub(crate) fn gpu_kerr_ode_batch(&self, weights: &KerrWeights, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n_pos = xs.len();
        let n_bands = weights.gamma_raw.len();
        let n_embd = n_bands * 2;
        let n_steps = weights.rk4_n_steps;
        let dt = 1.0 / n_steps as f32;

        // Unpack all positions: interleaved → separate r, s flat arrays
        let mut r = vec![0.0f32; n_pos * n_bands];
        let mut s = vec![0.0f32; n_pos * n_bands];
        for pos in 0..n_pos {
            for k in 0..n_bands {
                r[pos * n_bands + k] = xs[pos][k * 2];
                s[pos * n_bands + k] = xs[pos][k * 2 + 1];
            }
        }

        fn softplus(x: f32) -> f32 {
            if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
        }
        let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();

        let total = n_pos * n_bands;

        // RK4 integration — 4 derivative evals per step, each batched across all positions
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

        // Re-interleave per position
        (0..n_pos).map(|pos| {
            let mut out = vec![0.0f32; n_embd];
            for k in 0..n_bands {
                out[k * 2] = r[pos * n_bands + k];
                out[k * 2 + 1] = s[pos * n_bands + k];
            }
            out
        }).collect()
    }

    /// Batched Kerr-ODE backward: full RK4 backward for all positions simultaneously.
    /// d_outputs is [n_pos][n_embd], inputs is [n_pos][n_embd].
    /// Returns (d_inputs, d_gamma_raw, d_omega, d_alpha, d_beta).
    pub(crate) fn gpu_kerr_ode_backward_batch(
        &self,
        d_outputs: &[Vec<f32>],  // [n_pos][n_embd]
        inputs: &[Vec<f32>],      // [n_pos][n_embd]
        weights: &KerrWeights,
    ) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>, f32, f32) {
        let n_pos = d_outputs.len();
        let n_bands = weights.gamma_raw.len();
        let n_embd = n_bands * 2;
        let n_steps = weights.rk4_n_steps;
        let dt = 1.0 / n_steps as f32;
        let total = n_pos * n_bands;

        fn softplus(x: f32) -> f32 {
            if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
        }
        let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();

        // Unpack inputs: interleaved → separate r, s flat arrays
        let mut r0 = vec![0.0f32; total];
        let mut s0 = vec![0.0f32; total];
        for pos in 0..n_pos {
            for k in 0..n_bands {
                r0[pos * n_bands + k] = inputs[pos][k * 2];
                s0[pos * n_bands + k] = inputs[pos][k * 2 + 1];
            }
        }

        // Forward recompute: save all intermediate states
        let mut states: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(n_steps + 1);
        let mut r = r0.clone();
        let mut s = s0.clone();
        states.push((r.clone(), s.clone()));

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
            states.push((r.clone(), s.clone()));
        }

        // Unpack d_outputs into d_r, d_s
        let mut d_r: Vec<f32> = vec![0.0f32; total];
        let mut d_s: Vec<f32> = vec![0.0f32; total];
        for pos in 0..n_pos {
            for k in 0..n_bands {
                d_r[pos * n_bands + k] = d_outputs[pos][k * 2];
                d_s[pos * n_bands + k] = d_outputs[pos][k * 2 + 1];
            }
        }

        let mut d_gamma_acc = vec![0.0f32; n_bands];
        let mut d_omega_acc = vec![0.0f32; n_bands];
        let mut d_alpha_acc = 0.0f32;
        let mut d_beta_acc = 0.0f32;

        // Backward through steps in reverse
        for step in (0..n_steps).rev() {
            let (ref r_step, ref s_step) = states[step];

            // RK4 step backward: recompute forward within this step, then backward
            let (d_r_new, d_s_new, dg, dom, da, db) = self.gpu_rk4_step_backward_batch(
                &d_r, &d_s, r_step, s_step, dt,
                &gamma, &weights.omega, weights.alpha, weights.beta,
                n_bands, n_pos,
            );

            d_r = d_r_new;
            d_s = d_s_new;
            // Accumulate per-band parameter gradients (reduce across positions)
            for pos in 0..n_pos {
                for k in 0..n_bands {
                    d_gamma_acc[k] += dg[pos * n_bands + k];
                    d_omega_acc[k] += dom[pos * n_bands + k];
                }
            }
            d_alpha_acc += da;
            d_beta_acc += db;
        }

        // Chain through softplus for gamma_raw
        fn softplus_backward(d_y: f32, x: f32) -> f32 {
            let s = 1.0 / (1.0 + (-x).exp()); // sigmoid(x)
            d_y * s
        }
        let d_gamma_raw: Vec<f32> = (0..n_bands)
            .map(|k| softplus_backward(d_gamma_acc[k], weights.gamma_raw[k]))
            .collect();

        // Re-interleave d_r, d_s → d_inputs
        let d_inputs: Vec<Vec<f32>> = (0..n_pos).map(|pos| {
            let mut d_input = vec![0.0f32; n_embd];
            for k in 0..n_bands {
                d_input[k * 2] = d_r[pos * n_bands + k];
                d_input[k * 2 + 1] = d_s[pos * n_bands + k];
            }
            d_input
        }).collect();

        (d_inputs, d_gamma_raw, d_omega_acc, d_alpha_acc, d_beta_acc)
    }

    /// Batched RK4 step backward for all positions.
    /// Returns (d_r, d_s, d_gamma_flat, d_omega_flat, d_alpha, d_beta).
    fn gpu_rk4_step_backward_batch(
        &self,
        d_r_new: &[f32], d_s_new: &[f32],  // [n_pos * n_bands]
        r: &[f32], s: &[f32],               // [n_pos * n_bands] (state at start of step)
        dt: f32,
        gamma: &[f32], omega: &[f32],
        alpha: f32, beta: f32,
        n_bands: usize, n_pos: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32, f32) {
        let total = n_pos * n_bands;
        let dt6 = dt / 6.0;

        // Forward recompute within this step to get intermediate states
        let (k1r, k1s) = self.gpu_kerr_derivative_batch(r, s, gamma, omega, alpha, beta, n_bands, n_pos);

        let r2: Vec<f32> = r.iter().zip(&k1r).map(|(&ri, &k)| ri + 0.5 * dt * k).collect();
        let s2: Vec<f32> = s.iter().zip(&k1s).map(|(&si, &k)| si + 0.5 * dt * k).collect();
        let (k2r, k2s) = self.gpu_kerr_derivative_batch(&r2, &s2, gamma, omega, alpha, beta, n_bands, n_pos);

        let r3: Vec<f32> = r.iter().zip(&k2r).map(|(&ri, &k)| ri + 0.5 * dt * k).collect();
        let s3: Vec<f32> = s.iter().zip(&k2s).map(|(&si, &k)| si + 0.5 * dt * k).collect();
        let (k3r, k3s) = self.gpu_kerr_derivative_batch(&r3, &s3, gamma, omega, alpha, beta, n_bands, n_pos);

        let r4: Vec<f32> = r.iter().zip(&k3r).map(|(&ri, &k)| ri + dt * k).collect();
        let s4: Vec<f32> = s.iter().zip(&k3s).map(|(&si, &k)| si + dt * k).collect();

        // Accumulate parameter gradients
        let mut d_gamma_acc = vec![0.0f32; total];
        let mut d_omega_acc = vec![0.0f32; total];
        let mut d_alpha_acc = 0.0f32;
        let mut d_beta_acc = 0.0f32;

        // d_r_new -> d_r (direct path: r_new = r + ...)
        let mut d_r = d_r_new.to_vec();
        let mut d_s = d_s_new.to_vec();

        // ── k4 backward ──
        let d_dr4: Vec<f32> = d_r_new.iter().map(|&v| v * dt6).collect();
        let d_ds4: Vec<f32> = d_s_new.iter().map(|&v| v * dt6).collect();

        let (d_r4, d_s4, dg, dom, da, db) = self.gpu_kerr_derivative_backward_batch(
            &d_dr4, &d_ds4, &r4, &s4, gamma, omega, alpha, beta, n_bands, n_pos,
        );
        for i in 0..total { d_gamma_acc[i] += dg[i]; d_omega_acc[i] += dom[i]; }
        d_alpha_acc += da; d_beta_acc += db;

        // r4 = r + dt*dr3, so d_r += d_r4, d_dr3 += d_r4*dt
        for i in 0..total { d_r[i] += d_r4[i]; d_s[i] += d_s4[i]; }
        let d_dr3: Vec<f32> = (0..total).map(|i| d_r4[i] * dt + d_r_new[i] * dt6 * 2.0).collect();
        let d_ds3: Vec<f32> = (0..total).map(|i| d_s4[i] * dt + d_s_new[i] * dt6 * 2.0).collect();

        // ── k3 backward ──
        let (d_r3_in, d_s3_in, dg, dom, da, db) = self.gpu_kerr_derivative_backward_batch(
            &d_dr3, &d_ds3, &r3, &s3, gamma, omega, alpha, beta, n_bands, n_pos,
        );
        for i in 0..total { d_gamma_acc[i] += dg[i]; d_omega_acc[i] += dom[i]; }
        d_alpha_acc += da; d_beta_acc += db;

        for i in 0..total { d_r[i] += d_r3_in[i]; d_s[i] += d_s3_in[i]; }
        let d_dr2: Vec<f32> = (0..total).map(|i| d_r3_in[i] * 0.5 * dt + d_r_new[i] * dt6 * 2.0).collect();
        let d_ds2: Vec<f32> = (0..total).map(|i| d_s3_in[i] * 0.5 * dt + d_s_new[i] * dt6 * 2.0).collect();

        // ── k2 backward ──
        let (d_r2_in, d_s2_in, dg, dom, da, db) = self.gpu_kerr_derivative_backward_batch(
            &d_dr2, &d_ds2, &r2, &s2, gamma, omega, alpha, beta, n_bands, n_pos,
        );
        for i in 0..total { d_gamma_acc[i] += dg[i]; d_omega_acc[i] += dom[i]; }
        d_alpha_acc += da; d_beta_acc += db;

        for i in 0..total { d_r[i] += d_r2_in[i]; d_s[i] += d_s2_in[i]; }
        let d_dr1: Vec<f32> = (0..total).map(|i| d_r2_in[i] * 0.5 * dt + d_r_new[i] * dt6).collect();
        let d_ds1: Vec<f32> = (0..total).map(|i| d_s2_in[i] * 0.5 * dt + d_s_new[i] * dt6).collect();

        // ── k1 backward ──
        let (d_r1_in, d_s1_in, dg, dom, da, db) = self.gpu_kerr_derivative_backward_batch(
            &d_dr1, &d_ds1, r, s, gamma, omega, alpha, beta, n_bands, n_pos,
        );
        for i in 0..total { d_gamma_acc[i] += dg[i]; d_omega_acc[i] += dom[i]; }
        d_alpha_acc += da; d_beta_acc += db;

        for i in 0..total { d_r[i] += d_r1_in[i]; d_s[i] += d_s1_in[i]; }

        (d_r, d_s, d_gamma_acc, d_omega_acc, d_alpha_acc, d_beta_acc)
    }
}
