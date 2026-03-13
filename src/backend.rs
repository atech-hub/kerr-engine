//! Compute backend trait — abstraction over CPU/GPU execution.
//!
//! Defines the interface that both CpuBackend and GpuBackend implement.
//! The training loop and inference call through this trait.
//! Auto-selection: CPU below crossover dim, GPU above (if available).

use crate::model::*;

/// GPU dispatch overhead crossover point (measured on RTX 4070 Ti, 2026-03-12).
/// Below this: CPU wins. Above this: GPU persistent pipeline wins.
/// Derived from benchmark: CPU linear O(n^2) crosses ~120us GPU dispatch overhead.
const GPU_CROSSOVER_DIM: usize = 768;

/// Auto-select the best backend for the given embedding dimension.
/// Returns a boxed trait object so the caller doesn't know which backend it got.
///
/// Rules:
/// - N_EMBD < GPU_CROSSOVER_DIM → CpuBackend (dispatch overhead exceeds compute)
/// - N_EMBD >= GPU_CROSSOVER_DIM and GPU available → GpuBackend
/// - N_EMBD >= GPU_CROSSOVER_DIM but no GPU → CpuBackend (fallback)
/// - --force-cpu / --force-gpu override auto-selection
pub fn auto_select(n_embd: usize, force_cpu: bool, force_gpu: bool, gpu_device: Option<usize>) -> Box<dyn ComputeBackend + Send + Sync> {
    if force_cpu {
        println!("  Backend: CPU (forced)");
        return Box::new(CpuBackend);
    }

    if force_gpu || n_embd >= GPU_CROSSOVER_DIM {
        match try_gpu_backend(gpu_device) {
            Some(gpu) => {
                println!("  Backend: GPU (n_embd={n_embd} >= {GPU_CROSSOVER_DIM} crossover)");
                return Box::new(gpu);
            }
            None => {
                if force_gpu {
                    println!("  Backend: CPU (GPU requested but unavailable)");
                } else {
                    println!("  Backend: CPU (no GPU detected, fallback)");
                }
                return Box::new(CpuBackend);
            }
        }
    }

    println!("  Backend: CPU (n_embd={n_embd} < {GPU_CROSSOVER_DIM} crossover)");
    Box::new(CpuBackend)
}

/// Try to initialize GPU backend. Returns None if no GPU available.
/// If gpu_device is Some(idx), select that specific adapter by index.
fn try_gpu_backend(gpu_device: Option<usize>) -> Option<crate::gpu_backend::GpuBackend> {
    let instance = wgpu::Instance::default();

    let adapter = if let Some(idx) = gpu_device {
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        if idx >= adapters.len() {
            println!("  GPU device {idx} not found ({} adapters available)", adapters.len());
            return None;
        }
        // enumerate_adapters returns owned adapters; index into the vec
        let adapters: Vec<_> = adapters.into_iter().collect();
        let info = adapters[idx].get_info();
        println!("  GPU: {} ({:?}) [device {}]", info.name, info.backend, idx);
        // We can't easily pass the adapter to GpuBackend::new(), so we use
        // GpuBackend::with_device_index for explicit selection
        return Some(crate::gpu_backend::GpuBackend::with_device_index(idx));
    } else {
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))?
    };

    let name = adapter.get_info().name;
    let backend = adapter.get_info().backend;
    println!("  GPU: {name} ({backend:?})");

    Some(crate::gpu_backend::GpuBackend::new())
}

/// Core compute operations that can run on CPU or GPU.
pub trait ComputeBackend {
    /// Linear: y = W @ x + b
    fn linear(&self, w: &[Vec<f32>], b: &[f32], x: &[f32]) -> Vec<f32>;

    /// Linear without bias: y = W @ x
    fn linear_no_bias(&self, w: &[Vec<f32>], x: &[f32]) -> Vec<f32>;

    /// Layer normalization
    fn layer_norm(&self, x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32>;

    /// Kerr-ODE forward: input [N_EMBD] → output [N_EMBD]
    fn kerr_ode(&self, weights: &KerrWeights, x: &[f32]) -> Vec<f32>;

    /// Maestro forward: input [N_EMBD] → output [N_EMBD]
    fn maestro(&self, weights: &MaestroWeights, x: &[f32]) -> Vec<f32>;

    /// Causal self-attention: [T][N_EMBD] → [T][N_EMBD]
    fn attention(&self, weights: &AttentionWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>>;

    /// Per-band linear: [T][N_EMBD] → [T][N_EMBD]
    fn per_band_linear(&self, weights: &PerBandLinearWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>>;

    /// Kerr-Maestro-Add: [T][N_EMBD] → [T][N_EMBD]
    fn kerr_maestro_add(&self, weights: &KerrMaestroAddWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>>;

    /// Batched linear: y[i] = W @ x[i] + b for each x in the batch.
    /// Default loops over single linear(). Override for cache-efficient batching.
    fn linear_batch(&self, w: &[Vec<f32>], b: &[f32], xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        xs.iter().map(|x| self.linear(w, b, x)).collect()
    }

    /// Batched linear without bias: y[i] = W @ x[i] for each x in the batch.
    fn linear_no_bias_batch(&self, w: &[Vec<f32>], xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        xs.iter().map(|x| self.linear_no_bias(w, x)).collect()
    }

    /// Batched layer norm.
    fn layer_norm_batch(&self, xs: &[Vec<f32>], weight: &[f32], bias: &[f32]) -> Vec<Vec<f32>> {
        xs.iter().map(|x| self.layer_norm(x, weight, bias)).collect()
    }

    // ─── Backward operations ─────────────────────────────────────

    /// Backward through linear: d_x = W^T @ d_y
    fn linear_backward_dx(&self, d_y: &[f32], w: &[Vec<f32>]) -> Vec<f32>;

    /// Backward through layer norm. Returns (d_x, d_weight, d_bias).
    fn layer_norm_backward(&self, d_y: &[f32], x: &[f32], weight: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>);

    /// Backward through GELU activation.
    fn gelu_backward(&self, d_y: &[f32], x: &[f32]) -> Vec<f32>;

    /// Batched linear backward: d_x[pos] = W^T @ d_y[pos] for all positions in one dispatch.
    fn linear_backward_dx_batch(&self, d_y: &[Vec<f32>], w: &[Vec<f32>]) -> Vec<Vec<f32>>;

    /// Batched outer product accumulation: d_w[i][j] += sum_pos d_y[pos][i] * x[pos][j]
    /// Also computes d_b[i] += sum_pos d_y[pos][i] when compute_bias is true.
    /// d_y: [n_pos][out_dim], x: [n_pos][in_dim] → d_w: [out_dim][in_dim], d_b: [out_dim]
    fn outer_product_accum(
        &self,
        d_y: &[Vec<f32>],
        x: &[Vec<f32>],
        compute_bias: bool,
    ) -> (Vec<Vec<f32>>, Vec<f32>);

    /// Backward through causal self-attention.
    /// Returns (d_q, d_k, d_v) each [T][n_embd].
    fn attention_backward(
        &self,
        d_pre_proj: &[Vec<f32>],     // [T][n_embd] gradient into attention output
        q_all: &[Vec<f32>],          // [T][n_embd]
        k_all: &[Vec<f32>],          // [T][n_embd]
        v_all: &[Vec<f32>],          // [T][n_embd]
        att_weights: &[Vec<Vec<f32>>], // [n_head][T][T] post-softmax
        n_head: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>);
}

/// CPU backend — all operations run on the host processor.
/// Uses the existing implementations in model.rs.
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    #[inline]
    fn linear(&self, w: &[Vec<f32>], b: &[f32], x: &[f32]) -> Vec<f32> {
        // Iterator pattern: eliminates bounds checks, enables autovectorization
        w.iter()
            .zip(b.iter())
            .map(|(row, &bias)| {
                bias + row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum::<f32>()
            })
            .collect()
    }

    #[inline]
    fn linear_no_bias(&self, w: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
        w.iter()
            .map(|row| row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum::<f32>())
            .collect()
    }

    #[inline]
    fn layer_norm(&self, x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
        let n = x.len() as f32;
        let mean: f32 = x.iter().sum::<f32>() / n;
        let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        x.iter()
            .zip(weight.iter().zip(bias.iter()))
            .map(|(&xi, (&wi, &bi))| (xi - mean) * inv_std * wi + bi)
            .collect()
    }

    fn kerr_ode(&self, weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
        // Delegate to ModelWeights method via a temporary
        // This is the one place where the delegation is slightly awkward.
        // When GPU lands, this method will dispatch to a WGSL shader instead.
        kerr_ode_cpu(weights, x)
    }

    fn maestro(&self, weights: &MaestroWeights, x: &[f32]) -> Vec<f32> {
        maestro_cpu(weights, x)
    }

    fn attention(&self, weights: &AttentionWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        attention_cpu(weights, x)
    }

    fn per_band_linear(&self, weights: &PerBandLinearWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        per_band_linear_cpu(weights, x)
    }

    fn kerr_maestro_add(&self, weights: &KerrMaestroAddWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        kerr_maestro_add_cpu(weights, x)
    }

    fn linear_batch(&self, w: &[Vec<f32>], b: &[f32], xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        linear_batch_cpu(w, b, xs)
    }

    fn linear_no_bias_batch(&self, w: &[Vec<f32>], xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        linear_no_bias_batch_cpu(w, xs)
    }

    fn linear_backward_dx(&self, d_y: &[f32], w: &[Vec<f32>]) -> Vec<f32> {
        crate::backward::linear_backward_dx_only(d_y, w)
    }

    fn layer_norm_backward(&self, d_y: &[f32], x: &[f32], weight: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        crate::backward::layer_norm_backward(d_y, x, weight)
    }

    fn gelu_backward(&self, d_y: &[f32], x: &[f32]) -> Vec<f32> {
        crate::backward::gelu_backward(d_y, x)
    }

    fn linear_backward_dx_batch(&self, d_y: &[Vec<f32>], w: &[Vec<f32>]) -> Vec<Vec<f32>> {
        d_y.iter().map(|dy| crate::backward::linear_backward_dx_only(dy, w)).collect()
    }

    fn outer_product_accum(
        &self,
        d_y: &[Vec<f32>],
        x: &[Vec<f32>],
        compute_bias: bool,
    ) -> (Vec<Vec<f32>>, Vec<f32>) {
        let out_dim = d_y[0].len();
        let in_dim = x[0].len();
        let mut d_w = vec![vec![0.0f32; in_dim]; out_dim];
        let mut d_b = vec![0.0f32; out_dim];
        for pos in 0..d_y.len() {
            for i in 0..out_dim {
                let dy_i = d_y[pos][i];
                for j in 0..in_dim {
                    d_w[i][j] += dy_i * x[pos][j];
                }
                if compute_bias {
                    d_b[i] += dy_i;
                }
            }
        }
        (d_w, d_b)
    }

    fn attention_backward(
        &self,
        d_pre_proj: &[Vec<f32>],
        q_all: &[Vec<f32>],
        k_all: &[Vec<f32>],
        v_all: &[Vec<f32>],
        att_weights: &[Vec<Vec<f32>>],
        _n_head: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let t = d_pre_proj.len();
        let n_embd = d_pre_proj[0].len();
        let mut d_q = vec![vec![0.0f32; n_embd]; t];
        let mut d_k = vec![vec![0.0f32; n_embd]; t];
        let mut d_v = vec![vec![0.0f32; n_embd]; t];

        for pos in 0..t {
            let att_for_pos: Vec<Vec<f32>> = att_weights.iter()
                .map(|h| h[pos].clone()).collect();
            crate::backward::attention_backward_single(
                &d_pre_proj[pos],
                q_all, k_all, v_all,
                &att_for_pos,
                pos,
                &mut d_q, &mut d_k, &mut d_v,
            );
        }

        (d_q, d_k, d_v)
    }
}

// ─── CPU implementations (thin wrappers around model.rs logic) ──

use std::f32::consts::PI;

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

fn kerr_ode_cpu(weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
    let n_bands = weights.gamma_raw.len();
    let n_embd = n_bands * 2;
    let n_steps = weights.rk4_n_steps;
    let dt = 1.0 / n_steps as f32;

    let mut r = vec![0.0f32; n_bands];
    let mut s = vec![0.0f32; n_bands];
    for k in 0..n_bands {
        r[k] = x[k * 2];
        s[k] = x[k * 2 + 1];
    }

    let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();

    for _ in 0..n_steps {
        let (r_new, s_new) = rk4_step_public(
            &r, &s, dt, &gamma, &weights.omega, weights.alpha, weights.beta,
        );
        r = r_new;
        s = s_new;
    }

    let mut out = vec![0.0f32; n_embd];
    for k in 0..n_bands {
        out[k * 2] = r[k];
        out[k * 2 + 1] = s[k];
    }
    out
}

fn maestro_cpu(weights: &MaestroWeights, x: &[f32]) -> Vec<f32> {
    let squeezed = linear_fn(&weights.squeeze.w, &weights.squeeze.b, x);
    let activated: Vec<f32> = squeezed.iter().map(|&v| gelu(v)).collect();
    linear_fn(&weights.process_1.w, &weights.process_1.b, &activated)
}

fn attention_cpu(weights: &AttentionWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let t = x.len();
    let n_embd = x[0].len();
    let n_head = weights.n_head;
    let head_dim = n_embd / n_head;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut q_all = vec![vec![0.0f32; n_embd]; t];
    let mut k_all = vec![vec![0.0f32; n_embd]; t];
    let mut v_all = vec![vec![0.0f32; n_embd]; t];

    for pos in 0..t {
        let qkv = linear_fn(&weights.c_attn.w, &weights.c_attn.b, &x[pos]);
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
        .map(|o| linear_fn(&weights.c_proj.w, &weights.c_proj.b, o))
        .collect()
}

fn per_band_linear_cpu(weights: &PerBandLinearWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let t = x.len();
    let n_bands = weights.band_w.len();
    let n_embd = n_bands * 2;
    let mut result = Vec::with_capacity(t);

    for pos in 0..t {
        let mut bands_out = vec![0.0f32; n_embd];
        for band in 0..n_bands {
            let r_in = x[pos][band * 2];
            let s_in = x[pos][band * 2 + 1];
            let w = &weights.band_w[band];
            let b = &weights.band_b[band];
            bands_out[band * 2] = w[0][0] * r_in + w[1][0] * s_in + b[0];
            bands_out[band * 2 + 1] = w[0][1] * r_in + w[1][1] * s_in + b[1];
        }
        let projected = linear_fn(&weights.out_proj.w, &weights.out_proj.b, &bands_out);
        result.push(projected);
    }

    result
}

/// Batched linear: process all inputs against the same weight matrix.
/// Weight row stays in L1 while cycling through all batch inputs.
fn linear_batch_cpu(w: &[Vec<f32>], b: &[f32], xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if xs.is_empty() { return vec![]; }
    let out_dim = w.len();
    let batch = xs.len();
    let mut results = vec![vec![0.0f32; out_dim]; batch];

    for (i, (row, &bias)) in w.iter().zip(b.iter()).enumerate() {
        for (b_idx, x) in xs.iter().enumerate() {
            results[b_idx][i] = bias + row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum::<f32>();
        }
    }
    results
}

/// Batched linear without bias.
fn linear_no_bias_batch_cpu(w: &[Vec<f32>], xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if xs.is_empty() { return vec![]; }
    let out_dim = w.len();
    let batch = xs.len();
    let mut results = vec![vec![0.0f32; out_dim]; batch];

    for (i, row) in w.iter().enumerate() {
        for (b_idx, x) in xs.iter().enumerate() {
            results[b_idx][i] = row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum::<f32>();
        }
    }
    results
}

fn kerr_maestro_add_cpu(weights: &KerrMaestroAddWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let t = x.len();
    let n_embd = x[0].len();
    let mut result = Vec::with_capacity(t);

    for pos in 0..t {
        let kerr_out = kerr_ode_cpu(&weights.kerr, &x[pos]);
        let maestro_out = maestro_cpu(&weights.maestro, &x[pos]);
        let mut combined = vec![0.0f32; n_embd];
        for i in 0..n_embd {
            combined[i] = kerr_out[i] + maestro_out[i];
        }
        let projected = linear_fn(&weights.out_proj.w, &weights.out_proj.b, &combined);
        result.push(projected);
    }

    result
}
