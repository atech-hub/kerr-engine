//! GPU ComputeBackend trait implementation.
//!
//! All forward and backward dispatch methods for GpuBackend.
//! Uses pipelines and helpers defined in gpu_pipelines.rs.

use crate::backend::ComputeBackend;
use crate::gpu_pipelines::*;
use crate::model::*;

// ─── ComputeBackend implementation ──────────────────────────────

impl ComputeBackend for GpuBackend {
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
        let t = x.len();
        let n_embd = x[0].len();

        // Batched Kerr-ODE: all positions in one dispatch sequence
        let kerr_outs = self.gpu_kerr_ode_batch(&weights.kerr, x);

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

    fn linear_batch(&self, w: &[Vec<f32>], b: &[f32], xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if xs.is_empty() { return vec![]; }
        let out_dim = w.len();
        let in_dim = if out_dim > 0 { w[0].len() } else { return vec![vec![]; xs.len()] };
        let n_pos = xs.len();

        let mut w_flat = Vec::with_capacity(out_dim * in_dim);
        for row in w { w_flat.extend_from_slice(row); }
        let x_flat: Vec<f32> = xs.iter().flat_map(|v| v.iter().copied()).collect();

        let y_flat = self.gpu_matvec_batch(&w_flat, &x_flat, b, out_dim, in_dim, n_pos);
        y_flat.chunks(out_dim).map(|c| c.to_vec()).collect()
    }

    fn linear_no_bias_batch(&self, w: &[Vec<f32>], xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if xs.is_empty() { return vec![]; }
        let out_dim = w.len();
        let in_dim = if out_dim > 0 { w[0].len() } else { return vec![vec![]; xs.len()] };
        let n_pos = xs.len();

        let mut w_flat = Vec::with_capacity(out_dim * in_dim);
        for row in w { w_flat.extend_from_slice(row); }
        let x_flat: Vec<f32> = xs.iter().flat_map(|v| v.iter().copied()).collect();

        let y_flat = self.gpu_matvec_batch(&w_flat, &x_flat, &[], out_dim, in_dim, n_pos);
        y_flat.chunks(out_dim).map(|c| c.to_vec()).collect()
    }

    fn layer_norm_batch(&self, xs: &[Vec<f32>], weight: &[f32], bias: &[f32]) -> Vec<Vec<f32>> {
        if xs.is_empty() { return vec![]; }
        let dim = xs[0].len();
        let n_pos = xs.len();

        let x_flat: Vec<f32> = xs.iter().flat_map(|v| v.iter().copied()).collect();
        let y_flat = self.gpu_layer_norm_batch(&x_flat, weight, bias, dim, n_pos);
        y_flat.chunks(dim).map(|c| c.to_vec()).collect()
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
            let workgroups = (in_dim as u32 + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
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
            let total = (n_pos * in_dim) as u32;
            pass.dispatch_workgroups((total + 63) / 64, 1, 1);
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
