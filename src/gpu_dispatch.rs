//! GPU ComputeBackend trait implementation.
//!
//! All forward and backward dispatch methods for GpuBackend.
//! Uses pipelines and helpers defined in gpu_pipelines.rs.

use crate::backend::ComputeBackend;
use crate::gpu_pipelines::*;
use crate::model::*;
use wgpu::util::DeviceExt;

// ─── ComputeBackend implementation ──────────────────────────────

impl ComputeBackend for GpuBackend {
    fn invalidate_weight_cache(&self) {
        self.pool.lock().unwrap().invalidate_weights();
    }

    fn load_weights(&self, model: &ModelWeights) {
        let resident = crate::gpu_resident::ResidentWeightBuffers::from_model(
            &self.device, &self.queue, model,
        );
        *self.resident.lock().unwrap() = Some(resident);
    }

    fn update_weights(&self, model: &ModelWeights) {
        if let Some(ref resident) = *self.resident.lock().unwrap() {
            resident.update_from_model(&self.queue, model);
        }
    }

    fn reset_ffn_counter(&self) {
        self.ffn_block_counter.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    fn linear(&self, w: &[Vec<f32>], b: &[f32], x: &[f32]) -> Vec<f32> {
        let out_dim = w.len();
        let in_dim = if out_dim > 0 { w[0].len() } else { 0 };
        // Flatten row-major: w[i][j] → flat[i * in_dim + j]
        let mut w_flat = Vec::with_capacity(out_dim * in_dim);
        for row in w {
            w_flat.extend_from_slice(row);
        }
        self.gpu_matvec(&w_flat, x, b, out_dim, in_dim)
    }

    fn linear_no_bias(&self, w: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
        let out_dim = w.len();
        let in_dim = if out_dim > 0 { w[0].len() } else { 0 };
        let mut w_flat = Vec::with_capacity(out_dim * in_dim);
        for row in w {
            w_flat.extend_from_slice(row);
        }
        self.gpu_matvec(&w_flat, x, &[], out_dim, in_dim)
    }

    fn layer_norm(&self, x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
        self.gpu_layer_norm(x, weight, bias)
    }

    fn kerr_ode(&self, weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
        self.gpu_kerr_ode(weights, x)
    }

    fn kerr_ode_batch(&self, weights: &KerrWeights, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.gpu_kerr_ode_batch_fused(weights, xs)
    }

    fn maestro(&self, weights: &MaestroWeights, x: &[f32]) -> Vec<f32> {
        // Maestro = squeeze(linear+gelu) → process(linear)
        // Use GPU matvec for both linear operations
        let squeezed = self.linear(&weights.squeeze.w, &weights.squeeze.b, x);
        let activated: Vec<f32> = squeezed.iter().map(|&v| gelu_cpu(v)).collect();
        self.linear(&weights.process_1.w, &weights.process_1.b, &activated)
    }

    fn attention(&self, weights: &AttentionWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let t = x.len();
        let n_embd = x[0].len();
        let n_head = weights.n_head;
        let head_dim = n_embd / n_head;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut q_all = vec![vec![0.0f32; n_embd]; t];
        let mut k_all = vec![vec![0.0f32; n_embd]; t];
        let mut v_all = vec![vec![0.0f32; n_embd]; t];

        for pos in 0..t {
            let qkv = self.linear(&weights.c_attn.w, &weights.c_attn.b, &x[pos]);
            for i in 0..n_embd {
                q_all[pos][i] = qkv[i];
                k_all[pos][i] = qkv[n_embd + i];
                v_all[pos][i] = qkv[2 * n_embd + i];
            }
        }

        let mut out = vec![vec![0.0f32; n_embd]; t];
        for head in 0..n_head {
            let offset = head * head_dim;
            for qi in 0..t {
                let mut att = vec![f32::NEG_INFINITY; t];
                for ki in 0..=qi {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_all[qi][offset + d] * k_all[ki][offset + d];
                    }
                    att[ki] = dot * scale;
                }

                let max_att = att[..=qi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for ki in 0..=qi {
                    att[ki] = (att[ki] - max_att).exp();
                    exp_sum += att[ki];
                }
                for ki in 0..=qi {
                    att[ki] /= exp_sum;
                }

                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for ki in 0..=qi {
                        sum += att[ki] * v_all[ki][offset + d];
                    }
                    out[qi][offset + d] = sum;
                }
            }
        }

        out.iter()
            .map(|o| self.linear(&weights.c_proj.w, &weights.c_proj.b, o))
            .collect()
    }

    fn per_band_linear(&self, weights: &PerBandLinearWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Increment FFN block counter (keeps resident dispatch aligned with block index)
        self.ffn_block_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let t = x.len();
        let n_bands = weights.band_w.len();
        let n_embd = n_bands * 2;

        // Per-band transform is trivial (2x2 per band) — CPU
        let bands_out_all: Vec<Vec<f32>> = (0..t).map(|pos| {
            let mut bands_out = vec![0.0f32; n_embd];
            for band in 0..n_bands {
                let r_in = x[pos][band * 2];
                let s_in = x[pos][band * 2 + 1];
                let w = &weights.band_w[band];
                let b = &weights.band_b[band];
                bands_out[band * 2] = w[0][0] * r_in + w[1][0] * s_in + b[0];
                bands_out[band * 2 + 1] = w[0][1] * r_in + w[1][1] * s_in + b[1];
            }
            bands_out
        }).collect();

        // Batched output projection on GPU
        self.linear_batch(&weights.out_proj.w, &weights.out_proj.b, &bands_out_all)
    }

    fn kerr_maestro_add(&self, weights: &KerrMaestroAddWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.ffn_block_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let t = x.len();
        let n_embd = x[0].len();

        // Batched Kerr-ODE: fused path chains all dispatches in one encoder
        let kerr_outs = self.gpu_kerr_ode_batch_fused(&weights.kerr, x);

        // NaN detection: Kerr-ODE vs Maestro
        if kerr_outs.iter().any(|h| h.iter().any(|v| v.is_nan())) {
            eprintln!("    [NaN source: Kerr-ODE output]");
        }

        // Batched maestro: squeeze(linear+gelu) → process(linear)
        let maestro_outs = self.linear_batch(&weights.maestro.squeeze.w, &weights.maestro.squeeze.b, x);
        let activated: Vec<f32> = maestro_outs.iter()
            .flat_map(|v| v.iter().map(|&val| gelu_cpu(val)))
            .collect();
        let maestro_dim = weights.maestro.squeeze.w.len();
        let activated_vecs: Vec<Vec<f32>> = activated.chunks(maestro_dim).map(|c| c.to_vec()).collect();
        let maestro_projected = self.linear_batch(&weights.maestro.process_1.w, &weights.maestro.process_1.b, &activated_vecs);

        // Combine kerr + maestro, then batched out_proj
        let combined: Vec<Vec<f32>> = (0..t).map(|pos| {
            let mut c = vec![0.0f32; n_embd];
            for i in 0..n_embd {
                c[i] = kerr_outs[pos][i] + maestro_projected[pos][i];
            }
            c
        }).collect();

        self.linear_batch(&weights.out_proj.w, &weights.out_proj.b, &combined)
    }

    fn kerr_dual_maestro_add(&self, weights: &KerrDualMaestroWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let t = x.len();
        let n_embd = x[0].len();
        let maestro_dim = weights.maestro_in.squeeze.w.len();

        // Check for resident buffers — if available, use them (no weight upload)
        let resident = self.resident.lock().unwrap();
        let block_idx = self.ffn_block_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Helper: resident linear batch — uses pre-uploaded weight buffer
        let linear_batch_res = |w_buf: &wgpu::Buffer, b_buf: &wgpu::Buffer, xs: &[Vec<f32>], out_dim: usize, in_dim: usize| -> Vec<Vec<f32>> {
            let n_pos = xs.len();
            let x_flat: Vec<f32> = xs.iter().flat_map(|v| v.iter().copied()).collect();
            let y_flat = self.gpu_matvec_batch_resident(w_buf, b_buf, &x_flat, out_dim, in_dim, n_pos, true);
            y_flat.chunks(out_dim).map(|c| c.to_vec()).collect()
        };

        // Fully fused on-GPU chain: all activations stay as GPU buffers.
        // ONE upload (input x), all intermediate ops on GPU, ONE readback (final output).
        if let Some(ref res) = *resident {
            if let Some(crate::gpu_resident::FfnResidentBuffers::KerrDualMaestro {
                    gamma, omega, alpha_beta: _,
                    in_squeeze_w, in_squeeze_b, in_process_w, in_process_b,
                    out_squeeze_w, out_squeeze_b, out_process_w, out_process_b,
                    out_proj_w, out_proj_b,
                }) = res.ffn_buffers.get(block_idx) {

                    let n_bands = n_embd / 2;
                    let n_pos = t;
                    let n_steps = weights.kerr.rk4_n_steps;
                    let dt = 1.0 / n_steps as f32;
                    let total_bands = n_pos * n_bands;

                    // Upload input to pre-allocated scratch buffer (only PCIe transfer per call)
                    let x_flat: Vec<f32> = x.iter().flat_map(|v| v.iter().copied()).collect();
                    let sc = &res.ffn_scratch[block_idx];
                    self.queue.write_buffer(&sc.x_buf, 0, bytemuck::cast_slice(&x_flat));

                    // All scratch buffers are pre-allocated — zero allocations during training
                    let x_buf = &sc.x_buf;
                    let sq_buf = &sc.sq_buf;
                    let act_buf = &sc.act_buf;
                    let mae_out_buf = &sc.mae_out_buf;
                    let precond_buf = &sc.precond_buf;
                    let r_buf = &sc.r_buf;
                    let s_buf = &sc.s_buf;
                    let r_tmp = &sc.r_tmp;
                    let s_tmp = &sc.s_tmp;
                    let kerr_buf = &sc.kerr_buf;
                    let sq2_buf = &sc.sq2_buf;
                    let act2_buf = &sc.act2_buf;
                    let mae_out2_buf = &sc.mae_out2_buf;
                    let regulated_buf = &sc.regulated_buf;
                    let output_buf = &sc.output_buf;
                    let k1r = &sc.k1r; let k1s = &sc.k1s;
                    let k2r = &sc.k2r; let k2s = &sc.k2s;
                    let k3r = &sc.k3r; let k3s = &sc.k3s;
                    let k4r = &sc.k4r; let k4s = &sc.k4s;
                    let r_mid = &sc.r_mid; let s_mid = &sc.s_mid;
                    let r_new = &sc.r_new; let s_new = &sc.s_new;

                    // Uniform params
                    let mv_sq_params = MatvecBatchParams { out_dim: maestro_dim as u32, in_dim: n_embd as u32, n_pos: n_pos as u32, use_bias: 1 };
                    let mv_sq_u = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None, contents: bytemuck::bytes_of(&mv_sq_params), usage: wgpu::BufferUsages::UNIFORM,
                    });
                    let mv_pr_params = MatvecBatchParams { out_dim: n_embd as u32, in_dim: maestro_dim as u32, n_pos: n_pos as u32, use_bias: 1 };
                    let mv_pr_u = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None, contents: bytemuck::bytes_of(&mv_pr_params), usage: wgpu::BufferUsages::UNIFORM,
                    });
                    let mv_out_params = MatvecBatchParams { out_dim: n_embd as u32, in_dim: n_embd as u32, n_pos: n_pos as u32, use_bias: 1 };
                    let mv_out_u = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None, contents: bytemuck::bytes_of(&mv_out_params), usage: wgpu::BufferUsages::UNIFORM,
                    });
                    let gelu_params = GeluParams { len: (n_pos * maestro_dim) as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
                    let gelu_u = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None, contents: bytemuck::bytes_of(&gelu_params), usage: wgpu::BufferUsages::UNIFORM,
                    });
                    let va_params = VecAddParams { len: (n_pos * n_embd) as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
                    let va_u = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None, contents: bytemuck::bytes_of(&va_params), usage: wgpu::BufferUsages::UNIFORM,
                    });
                    let di_params = DeinterleaveParams { n_bands: n_bands as u32, n_pos: n_pos as u32, _pad1: 0, _pad2: 0 };
                    let di_u = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None, contents: bytemuck::bytes_of(&di_params), usage: wgpu::BufferUsages::UNIFORM,
                    });
                    // Kerr derivative params
                    let kd_params = KerrDerivBatchParams { n_bands: n_bands as u32, n_pos: n_pos as u32, _pad1: 0, _pad2: 0 };
                    let kd_u = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None, contents: bytemuck::bytes_of(&kd_params), usage: wgpu::BufferUsages::UNIFORM,
                    });
                    let ab = [weights.kerr.alpha, weights.kerr.beta];
                    let ab_buf = self.storage_buf("ab", &ab);
                    // RK4 scale params
                    let vsa_half = VecScaleAddParams { len: total_bands as u32, scale: 0.5 * dt, _pad1: 0, _pad2: 0 };
                    let vsa_half_u = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None, contents: bytemuck::bytes_of(&vsa_half), usage: wgpu::BufferUsages::UNIFORM,
                    });
                    let vsa_full = VecScaleAddParams { len: total_bands as u32, scale: dt, _pad1: 0, _pad2: 0 };
                    let vsa_full_u = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None, contents: bytemuck::bytes_of(&vsa_full), usage: wgpu::BufferUsages::UNIFORM,
                    });
                    let rc_params = Rk4CombineParams { len: total_bands as u32, dt_over_6: dt / 6.0, _pad1: 0, _pad2: 0 };
                    let rc_u = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None, contents: bytemuck::bytes_of(&rc_params), usage: wgpu::BufferUsages::UNIFORM,
                    });

                    let wg_maestro = ((n_pos * maestro_dim) as u32 + 63) / 64;
                    let wg_embd = ((n_pos * n_embd) as u32 + 63) / 64;
                    let wg_bands = (total_bands as u32 + 63) / 64;

                    // ═══ Build ONE command encoder for the entire FFN ═══
                    let mut enc = self.device.create_command_encoder(&Default::default());

                    // 1. Maestro_in squeeze: x → sq_buf  (matvec: [maestro_dim, n_embd] @ x)
                    { let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.matvec_batch_layout, entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: in_squeeze_w.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: in_squeeze_b.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: sq_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: mv_sq_u.as_entire_binding() },
                    ]});
                    let mut p = enc.begin_compute_pass(&Default::default());
                    p.set_pipeline(&self.matvec_batch_pipeline); p.set_bind_group(0, &bg, &[]);
                    p.dispatch_workgroups(maestro_dim as u32, n_pos as u32, 1); }

                    // 2. GELU: sq_buf → act_buf
                    { let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.gelu_layout, entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: sq_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: act_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: gelu_u.as_entire_binding() },
                    ]});
                    let mut p = enc.begin_compute_pass(&Default::default());
                    p.set_pipeline(&self.gelu_pipeline); p.set_bind_group(0, &bg, &[]);
                    p.dispatch_workgroups(wg_maestro, 1, 1); }

                    // 3. Maestro_in process: act_buf → mae_out_buf (matvec: [n_embd, maestro_dim] @ act)
                    { let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.matvec_batch_layout, entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: in_process_w.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: act_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: in_process_b.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: mae_out_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: mv_pr_u.as_entire_binding() },
                    ]});
                    let mut p = enc.begin_compute_pass(&Default::default());
                    p.set_pipeline(&self.matvec_batch_pipeline); p.set_bind_group(0, &bg, &[]);
                    p.dispatch_workgroups(n_embd as u32, n_pos as u32, 1); }

                    // 4. Vec add: x + mae_out → precond_buf
                    { let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.vec_add_layout, entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: mae_out_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: precond_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: va_u.as_entire_binding() },
                    ]});
                    let mut p = enc.begin_compute_pass(&Default::default());
                    p.set_pipeline(&self.vec_add_pipeline); p.set_bind_group(0, &bg, &[]);
                    p.dispatch_workgroups(wg_embd, 1, 1); }

                    // 5. Deinterleave: precond_buf → r_buf, s_buf
                    { let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.deinterleave_layout, entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: precond_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: r_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: s_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: di_u.as_entire_binding() },
                    ]});
                    let mut p = enc.begin_compute_pass(&Default::default());
                    p.set_pipeline(&self.deinterleave_pipeline); p.set_bind_group(0, &bg, &[]);
                    p.dispatch_workgroups(wg_bands, 1, 1); }

                    // 6. Fused RK4 ODE (all n_steps in this encoder)
                    // Pre-create bind groups for ODE chain
                    let deriv_bg = |r_in: &wgpu::Buffer, s_in: &wgpu::Buffer, dr: &wgpu::Buffer, ds: &wgpu::Buffer| {
                        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.kerr_deriv_batch_layout, entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: r_in.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: s_in.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: dr.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 3, resource: ds.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 4, resource: gamma.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 5, resource: omega.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 6, resource: kd_u.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 7, resource: ab_buf.as_entire_binding() },
                        ]})
                    };
                    let vsa_bg = |a: &wgpu::Buffer, b: &wgpu::Buffer, y: &wgpu::Buffer, u: &wgpu::Buffer| {
                        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.vec_scale_add_layout, entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 3, resource: u.as_entire_binding() },
                        ]})
                    };
                    let bg_k1 = deriv_bg(&r_buf, &s_buf, &k1r, &k1s);
                    let bg_mid_r2 = vsa_bg(&r_buf, &k1r, &r_mid, &vsa_half_u);
                    let bg_mid_s2 = vsa_bg(&s_buf, &k1s, &s_mid, &vsa_half_u);
                    let bg_k2 = deriv_bg(&r_mid, &s_mid, &k2r, &k2s);
                    let bg_mid_r3 = vsa_bg(&r_buf, &k2r, &r_mid, &vsa_half_u);
                    let bg_mid_s3 = vsa_bg(&s_buf, &k2s, &s_mid, &vsa_half_u);
                    let bg_k3 = deriv_bg(&r_mid, &s_mid, &k3r, &k3s);
                    let bg_mid_r4 = vsa_bg(&r_buf, &k3r, &r_mid, &vsa_full_u);
                    let bg_mid_s4 = vsa_bg(&s_buf, &k3s, &s_mid, &vsa_full_u);
                    let bg_k4 = deriv_bg(&r_mid, &s_mid, &k4r, &k4s);
                    let bg_comb_r = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.rk4_combine_layout, entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: r_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: k1r.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: k2r.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: k3r.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: k4r.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: r_new.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: rc_u.as_entire_binding() },
                    ]});
                    let bg_comb_s = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.rk4_combine_layout, entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: s_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: k1s.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: k2s.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: k3s.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: k4s.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: s_new.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: rc_u.as_entire_binding() },
                    ]});

                    let buf_size = (total_bands * 4) as u64;
                    for _ in 0..n_steps {
                        // k1, midpoints, k2, midpoints, k3, midpoints, k4, combine
                        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&self.kerr_deriv_batch_pipeline); p.set_bind_group(0, &bg_k1, &[]); p.dispatch_workgroups(wg_bands, 1, 1); }
                        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&self.vec_scale_add_pipeline); p.set_bind_group(0, &bg_mid_r2, &[]); p.dispatch_workgroups(wg_bands, 1, 1); }
                        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&self.vec_scale_add_pipeline); p.set_bind_group(0, &bg_mid_s2, &[]); p.dispatch_workgroups(wg_bands, 1, 1); }
                        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&self.kerr_deriv_batch_pipeline); p.set_bind_group(0, &bg_k2, &[]); p.dispatch_workgroups(wg_bands, 1, 1); }
                        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&self.vec_scale_add_pipeline); p.set_bind_group(0, &bg_mid_r3, &[]); p.dispatch_workgroups(wg_bands, 1, 1); }
                        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&self.vec_scale_add_pipeline); p.set_bind_group(0, &bg_mid_s3, &[]); p.dispatch_workgroups(wg_bands, 1, 1); }
                        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&self.kerr_deriv_batch_pipeline); p.set_bind_group(0, &bg_k3, &[]); p.dispatch_workgroups(wg_bands, 1, 1); }
                        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&self.vec_scale_add_pipeline); p.set_bind_group(0, &bg_mid_r4, &[]); p.dispatch_workgroups(wg_bands, 1, 1); }
                        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&self.vec_scale_add_pipeline); p.set_bind_group(0, &bg_mid_s4, &[]); p.dispatch_workgroups(wg_bands, 1, 1); }
                        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&self.kerr_deriv_batch_pipeline); p.set_bind_group(0, &bg_k4, &[]); p.dispatch_workgroups(wg_bands, 1, 1); }
                        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&self.rk4_combine_pipeline); p.set_bind_group(0, &bg_comb_r, &[]); p.dispatch_workgroups(wg_bands, 1, 1); }
                        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&self.rk4_combine_pipeline); p.set_bind_group(0, &bg_comb_s, &[]); p.dispatch_workgroups(wg_bands, 1, 1); }
                        enc.copy_buffer_to_buffer(&r_new, 0, &r_buf, 0, buf_size);
                        enc.copy_buffer_to_buffer(&s_new, 0, &s_buf, 0, buf_size);
                    }

                    // 7. Reinterleave: r_buf, s_buf → kerr_buf
                    { let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.reinterleave_layout, entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: r_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: s_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: kerr_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: di_u.as_entire_binding() },
                    ]});
                    let mut p = enc.begin_compute_pass(&Default::default());
                    p.set_pipeline(&self.reinterleave_pipeline); p.set_bind_group(0, &bg, &[]);
                    p.dispatch_workgroups(wg_bands, 1, 1); }

                    // 8. Maestro_out squeeze: kerr_buf → sq2_buf
                    { let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.matvec_batch_layout, entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: out_squeeze_w.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: kerr_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: out_squeeze_b.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: sq2_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: mv_sq_u.as_entire_binding() },
                    ]});
                    let mut p = enc.begin_compute_pass(&Default::default());
                    p.set_pipeline(&self.matvec_batch_pipeline); p.set_bind_group(0, &bg, &[]);
                    p.dispatch_workgroups(maestro_dim as u32, n_pos as u32, 1); }

                    // 9. GELU: sq2_buf → act2_buf
                    { let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.gelu_layout, entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: sq2_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: act2_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: gelu_u.as_entire_binding() },
                    ]});
                    let mut p = enc.begin_compute_pass(&Default::default());
                    p.set_pipeline(&self.gelu_pipeline); p.set_bind_group(0, &bg, &[]);
                    p.dispatch_workgroups(wg_maestro, 1, 1); }

                    // 10. Maestro_out process: act2_buf → mae_out2_buf
                    { let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.matvec_batch_layout, entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: out_process_w.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: act2_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: out_process_b.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: mae_out2_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: mv_pr_u.as_entire_binding() },
                    ]});
                    let mut p = enc.begin_compute_pass(&Default::default());
                    p.set_pipeline(&self.matvec_batch_pipeline); p.set_bind_group(0, &bg, &[]);
                    p.dispatch_workgroups(n_embd as u32, n_pos as u32, 1); }

                    // 11. Vec add: kerr_buf + mae_out2_buf → regulated_buf
                    { let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.vec_add_layout, entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: kerr_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: mae_out2_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: regulated_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: va_u.as_entire_binding() },
                    ]});
                    let mut p = enc.begin_compute_pass(&Default::default());
                    p.set_pipeline(&self.vec_add_pipeline); p.set_bind_group(0, &bg, &[]);
                    p.dispatch_workgroups(wg_embd, 1, 1); }

                    // 12. Out projection: regulated_buf → output_buf
                    { let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.matvec_batch_layout, entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: out_proj_w.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: regulated_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: out_proj_b.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: output_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: mv_out_u.as_entire_binding() },
                    ]});
                    let mut p = enc.begin_compute_pass(&Default::default());
                    p.set_pipeline(&self.matvec_batch_pipeline); p.set_bind_group(0, &bg, &[]);
                    p.dispatch_workgroups(n_embd as u32, n_pos as u32, 1); }

                    // ONE submit, ONE readback
                    self.queue.submit(Some(enc.finish()));
                    let result_flat = self.readback(&output_buf, n_pos * n_embd);
                    return result_flat.chunks(n_embd).map(|c| c.to_vec()).collect();
            }
        }
        drop(resident);

        // Fallback: non-resident path (uploads weights every call)
        let mae_in_sq = self.linear_batch(&weights.maestro_in.squeeze.w, &weights.maestro_in.squeeze.b, x);
        let mae_in_act: Vec<f32> = mae_in_sq.iter()
            .flat_map(|v| v.iter().map(|&val| gelu_cpu(val))).collect();
        let mae_in_act_vecs: Vec<Vec<f32>> = mae_in_act.chunks(maestro_dim).map(|c| c.to_vec()).collect();
        let mae_in_out = self.linear_batch(&weights.maestro_in.process_1.w, &weights.maestro_in.process_1.b, &mae_in_act_vecs);
        let precond: Vec<Vec<f32>> = (0..t).map(|pos| {
            let mut p = vec![0.0f32; n_embd];
            for i in 0..n_embd { p[i] = x[pos][i] + mae_in_out[pos][i]; }
            p
        }).collect();
        let kerr_outs = self.gpu_kerr_ode_batch_fused(&weights.kerr, &precond);
        if kerr_outs.iter().any(|h| h.iter().any(|v| v.is_nan())) {
            eprintln!("    [NaN source: dual-maestro Kerr-ODE output]");
        }
        let mae_out_sq = self.linear_batch(&weights.maestro_out.squeeze.w, &weights.maestro_out.squeeze.b, &kerr_outs);
        let mae_out_act: Vec<f32> = mae_out_sq.iter()
            .flat_map(|v| v.iter().map(|&val| gelu_cpu(val))).collect();
        let mae_out_act_vecs: Vec<Vec<f32>> = mae_out_act.chunks(maestro_dim).map(|c| c.to_vec()).collect();
        let mae_out_out = self.linear_batch(&weights.maestro_out.process_1.w, &weights.maestro_out.process_1.b, &mae_out_act_vecs);
        let regulated: Vec<Vec<f32>> = (0..t).map(|pos| {
            let mut r = vec![0.0f32; n_embd];
            for i in 0..n_embd { r[i] = kerr_outs[pos][i] + mae_out_out[pos][i]; }
            r
        }).collect();
        self.linear_batch(&weights.out_proj.w, &weights.out_proj.b, &regulated)
    }

    fn linear_batch(&self, w: &[Vec<f32>], b: &[f32], xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if xs.is_empty() { return vec![]; }
        let out_dim = w.len();
        let in_dim = if out_dim > 0 { w[0].len() } else { return vec![vec![]; xs.len()] };
        let n_pos = xs.len();

        // Cache weight buffer by pointer — avoid re-upload within same iteration
        let w_ptr = w.as_ptr() as usize;
        let b_ptr = b.as_ptr() as usize;
        let x_flat: Vec<f32> = xs.iter().flat_map(|v| v.iter().copied()).collect();
        let out_total = n_pos * out_dim;
        let use_bias = 1u32;

        let mut pool = self.pool.lock().unwrap();
        if !pool.has_weight(w_ptr) {
            let mut w_flat = Vec::with_capacity(out_dim * in_dim);
            for row in w { w_flat.extend_from_slice(row); }
            pool.cache_weight(&self.device, &self.queue, w_ptr, &w_flat);
        }
        if !pool.has_weight(b_ptr) {
            pool.cache_weight(&self.device, &self.queue, b_ptr, b);
        }
        // Dispatch with cached weight buffers — hold pool lock for buffer refs
        let x_buf = self.storage_buf("x", &x_flat);
        pool.ensure_scratch(&self.device, 0, (out_total * 4) as u64);
        let params = MatvecBatchParams { out_dim: out_dim as u32, in_dim: in_dim as u32, n_pos: n_pos as u32, use_bias };
        pool.write_uniform(&self.device, &self.queue, &params);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.matvec_batch_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pool.weight_ref(w_ptr).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pool.weight_ref(b_ptr).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pool.scratch_ref(0).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pool.uniform_ref().as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        { let mut pass = encoder.begin_compute_pass(&Default::default());
          pass.set_pipeline(&self.matvec_batch_pipeline); pass.set_bind_group(0, &bind_group, &[]);
          pass.dispatch_workgroups(out_dim as u32, n_pos as u32, 1); }
        self.queue.submit(Some(encoder.finish()));
        let y_flat = pool.readback_scratch(&self.device, &self.queue, 0, out_total);
        drop(pool);
        y_flat.chunks(out_dim).map(|c| c.to_vec()).collect()
    }

    fn linear_no_bias_batch(&self, w: &[Vec<f32>], xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if xs.is_empty() { return vec![]; }
        let out_dim = w.len();
        let in_dim = if out_dim > 0 { w[0].len() } else { return vec![vec![]; xs.len()] };
        let n_pos = xs.len();

        let w_ptr = w.as_ptr() as usize;
        let x_flat: Vec<f32> = xs.iter().flat_map(|v| v.iter().copied()).collect();
        let out_total = n_pos * out_dim;

        let mut pool = self.pool.lock().unwrap();
        if !pool.has_weight(w_ptr) {
            let mut w_flat = Vec::with_capacity(out_dim * in_dim);
            for row in w { w_flat.extend_from_slice(row); }
            pool.cache_weight(&self.device, &self.queue, w_ptr, &w_flat);
        }
        let x_buf = self.storage_buf("x", &x_flat);
        let dummy_bias = [0.0f32];
        let dummy_buf = self.storage_buf("nb", &dummy_bias);
        pool.ensure_scratch(&self.device, 0, (out_total * 4) as u64);
        let params = MatvecBatchParams { out_dim: out_dim as u32, in_dim: in_dim as u32, n_pos: n_pos as u32, use_bias: 0 };
        pool.write_uniform(&self.device, &self.queue, &params);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.matvec_batch_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pool.weight_ref(w_ptr).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dummy_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pool.scratch_ref(0).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pool.uniform_ref().as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        { let mut pass = encoder.begin_compute_pass(&Default::default());
          pass.set_pipeline(&self.matvec_batch_pipeline); pass.set_bind_group(0, &bind_group, &[]);
          pass.dispatch_workgroups(out_dim as u32, n_pos as u32, 1); }
        self.queue.submit(Some(encoder.finish()));
        let y_flat = pool.readback_scratch(&self.device, &self.queue, 0, out_total);
        drop(pool);
        y_flat.chunks(out_dim).map(|c| c.to_vec()).collect()
    }

    fn layer_norm_batch(&self, xs: &[Vec<f32>], weight: &[f32], bias: &[f32]) -> Vec<Vec<f32>> {
        if xs.is_empty() { return vec![]; }
        let dim = xs[0].len();
        let n_pos = xs.len();

        // Cache weight/bias buffers — avoid re-upload within same iteration
        let w_ptr = weight.as_ptr() as usize;
        let b_ptr = bias.as_ptr() as usize;
        {
            let mut pool = self.pool.lock().unwrap();
            if !pool.has_weight(w_ptr) { pool.cache_weight(&self.device, &self.queue, w_ptr, weight); }
            if !pool.has_weight(b_ptr) { pool.cache_weight(&self.device, &self.queue, b_ptr, bias); }
        }
        let x_flat: Vec<f32> = xs.iter().flat_map(|v| v.iter().copied()).collect();
        // Use gpu_layer_norm_batch — weight/bias still re-uploaded inside it,
        // but the cache means the NEXT call with same weights skips upload.
        // TODO: add gpu_layer_norm_batch_resident for full caching
        let y_flat = self.gpu_layer_norm_batch(&x_flat, weight, bias, dim, n_pos);
        y_flat.chunks(dim).map(|c| c.to_vec()).collect()
    }

    fn kerr_ode_backward_batch(
        &self,
        d_outputs: &[Vec<f32>],
        inputs: &[Vec<f32>],
        weights: &KerrWeights,
    ) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>, f32, f32) {
        self.gpu_kerr_ode_backward_batch(d_outputs, inputs, weights)
    }

    fn linear_backward_dx(&self, d_y: &[f32], w: &[Vec<f32>]) -> Vec<f32> {
        let out_dim = w.len();
        let in_dim = if out_dim > 0 { w[0].len() } else { return vec![] };
        let mut w_flat = Vec::with_capacity(out_dim * in_dim);
        for row in w { w_flat.extend_from_slice(row); }

        let w_buf = self.storage_buf("bwd_w", &w_flat);
        let dy_buf = self.storage_buf("bwd_dy", d_y);
        let dx_buf = self.output_buf("bwd_dx", in_dim);
        let params = MatvecBwdParams {
            out_dim: out_dim as u32, in_dim: in_dim as u32, _pad1: 0, _pad2: 0,
        };
        let params_buf = self.uniform_buf("bwd_params", &params);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matvec_bwd_bg"),
            layout: &self.matvec_bwd_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dy_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dx_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.matvec_bwd_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(in_dim as u32, 1, 1); // tiled: one workgroup per input element
        }
        self.queue.submit(Some(encoder.finish()));

        self.readback(&dx_buf, in_dim)
    }

    fn layer_norm_backward(&self, d_y: &[f32], x: &[f32], weight: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let dim = x.len();
        let dy_buf = self.storage_buf("lnb_dy", d_y);
        let x_buf = self.storage_buf("lnb_x", x);
        let w_buf = self.storage_buf("lnb_w", weight);
        let out_buf = self.output_buf("lnb_out", dim * 3); // d_x, d_weight, d_bias concatenated
        let params = LayerNormParams { dim: dim as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
        let params_buf = self.uniform_buf("lnb_params", &params);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer_norm_bwd_bg"),
            layout: &self.layer_norm_bwd_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dy_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.layer_norm_bwd_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1); // single workgroup handles all
        }
        self.queue.submit(Some(encoder.finish()));

        let result = self.readback(&out_buf, dim * 3);
        let d_x = result[..dim].to_vec();
        let d_weight = result[dim..dim * 2].to_vec();
        let d_bias = result[dim * 2..].to_vec();
        (d_x, d_weight, d_bias)
    }

    fn gelu_backward(&self, d_y: &[f32], x: &[f32]) -> Vec<f32> {
        let n = x.len();
        let dy_buf = self.storage_buf("gb_dy", d_y);
        let x_buf = self.storage_buf("gb_x", x);
        let dx_buf = self.output_buf("gb_dx", n);
        let params = GeluBwdParams { len: n as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
        let params_buf = self.uniform_buf("gb_params", &params);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gelu_bwd_bg"),
            layout: &self.gelu_bwd_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dy_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dx_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.gelu_bwd_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n as u32 + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.readback(&dx_buf, n)
    }

    fn linear_backward_dx_batch(&self, d_y: &[Vec<f32>], w: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n_pos = d_y.len();
        let out_dim = w.len();
        let in_dim = if out_dim > 0 { w[0].len() } else { return vec![vec![]; n_pos] };

        // Flatten W and d_y
        let mut w_flat = Vec::with_capacity(out_dim * in_dim);
        for row in w { w_flat.extend_from_slice(row); }
        let dy_flat: Vec<f32> = d_y.iter().flat_map(|v| v.iter().copied()).collect();

        let w_buf = self.storage_buf("mvbb_w", &w_flat);
        let dy_buf = self.storage_buf("mvbb_dy", &dy_flat);
        let dx_buf = self.output_buf("mvbb_dx", n_pos * in_dim);
        let params = MatvecBwdBatchParams {
            out_dim: out_dim as u32, in_dim: in_dim as u32,
            n_pos: n_pos as u32, _pad: 0,
        };
        let params_buf = self.uniform_buf("mvbb_params", &params);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mvbb_bg"),
            layout: &self.matvec_bwd_batch_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dy_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dx_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.matvec_bwd_batch_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(in_dim as u32, n_pos as u32, 1); // tiled: (in_dim, n_pos)
        }
        self.queue.submit(Some(encoder.finish()));

        let dx_flat = self.readback(&dx_buf, n_pos * in_dim);
        dx_flat.chunks(in_dim).map(|c| c.to_vec()).collect()
    }

    fn outer_product_accum(
        &self,
        d_y: &[Vec<f32>],
        x: &[Vec<f32>],
        compute_bias: bool,
    ) -> (Vec<Vec<f32>>, Vec<f32>) {
        let n_pos = d_y.len();
        let out_dim = d_y[0].len();
        let in_dim = x[0].len();

        // Flatten inputs: d_y[pos][i] → d_y_flat[pos * out_dim + i]
        let d_y_flat: Vec<f32> = d_y.iter().flat_map(|v| v.iter().copied()).collect();
        let x_flat: Vec<f32> = x.iter().flat_map(|v| v.iter().copied()).collect();

        let dy_buf = self.storage_buf("op_dy", &d_y_flat);
        let x_buf = self.storage_buf("op_x", &x_flat);
        let dw_buf = self.output_buf("op_dw", out_dim * in_dim);
        // d_b buffer — always create it (even if not used) for binding
        let db_buf = self.output_buf("op_db", out_dim);
        let params = OuterProductParams {
            out_dim: out_dim as u32,
            in_dim: in_dim as u32,
            n_pos: n_pos as u32,
            compute_bias: if compute_bias { 1 } else { 0 },
        };
        let params_buf = self.uniform_buf("op_params", &params);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outer_product_bg"),
            layout: &self.outer_product_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dy_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dw_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: db_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.outer_product_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per output row
            pass.dispatch_workgroups(out_dim as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        // Readback and unflatten
        let dw_flat = self.readback(&dw_buf, out_dim * in_dim);
        let d_w: Vec<Vec<f32>> = dw_flat.chunks(in_dim).map(|c| c.to_vec()).collect();
        let d_b = if compute_bias {
            self.readback(&db_buf, out_dim)
        } else {
            vec![0.0f32; out_dim]
        };

        (d_w, d_b)
    }

    fn attention_backward(
        &self,
        d_pre_proj: &[Vec<f32>],
        q_all: &[Vec<f32>],
        k_all: &[Vec<f32>],
        v_all: &[Vec<f32>],
        att_weights: &[Vec<Vec<f32>>],
        n_head: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let t = d_pre_proj.len();
        let n_embd = d_pre_proj[0].len();
        let head_dim = n_embd / n_head;

        // Flatten inputs to contiguous f32 arrays
        let d_out_flat: Vec<f32> = d_pre_proj.iter().flat_map(|v| v.iter().copied()).collect();
        let q_flat: Vec<f32> = q_all.iter().flat_map(|v| v.iter().copied()).collect();
        let k_flat: Vec<f32> = k_all.iter().flat_map(|v| v.iter().copied()).collect();
        let v_flat: Vec<f32> = v_all.iter().flat_map(|v| v.iter().copied()).collect();

        // att_weights: [n_head][T][T] → flatten
        // att_weights[head][pos][ki] → att_flat[head * T * T + pos * T + ki]
        let mut att_flat = vec![0.0f32; n_head * t * t];
        for (head, head_weights) in att_weights.iter().enumerate() {
            for (pos, pos_weights) in head_weights.iter().enumerate() {
                for (ki, &w) in pos_weights.iter().enumerate() {
                    att_flat[head * t * t + pos * t + ki] = w;
                }
            }
        }

        // Create GPU buffers
        let d_out_buf = self.storage_buf("ab_d_out", &d_out_flat);
        let q_buf = self.storage_buf("ab_q", &q_flat);
        let k_buf = self.storage_buf("ab_k", &k_flat);
        let v_buf = self.storage_buf("ab_v", &v_flat);
        let att_buf = self.storage_buf("ab_att", &att_flat);
        let dq_buf = self.output_buf("ab_dq", t * n_embd);
        let dk_buf = self.output_buf("ab_dk", t * n_embd);
        let dv_buf = self.output_buf("ab_dv", t * n_embd);
        let dscore_buf = self.output_buf("ab_dscore", t * n_head * t);

        let params = AttnBwdParams {
            seq_len: t as u32,
            n_head: n_head as u32,
            head_dim: head_dim as u32,
            n_embd: n_embd as u32,
        };
        let params_buf = self.uniform_buf("ab_params", &params);
        // Dispatch 2 needs its own uniform buffer (same data, different bind group)
        let params_buf2 = self.uniform_buf("ab_params2", &params);

        // ─── Dispatch 1: attn_backward_scores ──────────────────────
        // One workgroup per (pos, head)
        let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attn_bwd_scores_bg"),
            layout: &self.attn_bwd_scores_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: d_out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: att_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: dq_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: dscore_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: params_buf.as_entire_binding() },
            ],
        });

        // ─── Dispatch 2: attn_backward_dkv ─────────────────────────
        // One thread per (ki, d_global)
        let bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attn_bwd_dkv_bg"),
            layout: &self.attn_bwd_dkv_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: d_out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: att_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dscore_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dk_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: dv_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: params_buf2.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            // Dispatch 1: (T, n_head, 1) workgroups of 64 threads each
            pass.set_pipeline(&self.attn_bwd_scores_pipeline);
            pass.set_bind_group(0, &bg1, &[]);
            pass.dispatch_workgroups(t as u32, n_head as u32, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            // Dispatch 2: ceil(T * n_embd / 64) workgroups of 64 threads
            pass.set_pipeline(&self.attn_bwd_dkv_pipeline);
            pass.set_bind_group(0, &bg2, &[]);
            let total_threads = (t * n_embd) as u32;
            pass.dispatch_workgroups((total_threads + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        // Readback d_q, d_k, d_v and unflatten
        let dq_flat = self.readback(&dq_buf, t * n_embd);
        let dk_flat = self.readback(&dk_buf, t * n_embd);
        let dv_flat = self.readback(&dv_buf, t * n_embd);

        let unflatten = |flat: Vec<f32>| -> Vec<Vec<f32>> {
            flat.chunks(n_embd).map(|c| c.to_vec()).collect()
        };

        (unflatten(dq_flat), unflatten(dk_flat), unflatten(dv_flat))
    }
}

// ─── CPU helper (GELU for maestro activation — trivial, not worth a shader) ─

pub(crate) fn gelu_cpu(x: f32) -> f32 {
    use std::f32::consts::PI;
    0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}
