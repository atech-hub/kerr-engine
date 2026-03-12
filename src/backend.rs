//! Compute backend trait — abstraction over CPU/GPU execution.
//!
//! Defines the interface that both CpuBackend and (future) GpuBackend
//! implement. The training loop and inference call through this trait.
//! Today: one implementation (CPU). The trait exists so GPU training
//! slots in without restructuring.

use crate::model::*;

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
}

/// CPU backend — all operations run on the host processor.
/// Uses the existing implementations in model.rs.
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn linear(&self, w: &[Vec<f32>], b: &[f32], x: &[f32]) -> Vec<f32> {
        linear_fn(w, b, x)
    }

    fn linear_no_bias(&self, w: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
        let out_dim = w.len();
        let mut y = vec![0.0f32; out_dim];
        for i in 0..out_dim {
            let mut sum = 0.0f32;
            for j in 0..x.len() {
                sum += w[i][j] * x[j];
            }
            y[i] = sum;
        }
        y
    }

    fn layer_norm(&self, x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
        let n = x.len();
        let mean: f32 = x.iter().sum::<f32>() / n as f32;
        let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let std = (var + 1e-5).sqrt();
        (0..n).map(|i| (x[i] - mean) / std * weight[i] + bias[i]).collect()
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
    let mut r = vec![0.0f32; N_BANDS];
    let mut s = vec![0.0f32; N_BANDS];
    for k in 0..N_BANDS {
        r[k] = x[k * 2];
        s[k] = x[k * 2 + 1];
    }

    let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();

    for _ in 0..RK4_N_STEPS {
        let (r_new, s_new) = rk4_step_public(
            &r, &s, RK4_DT, &gamma, &weights.omega, weights.alpha, weights.beta,
        );
        r = r_new;
        s = s_new;
    }

    let mut out = vec![0.0f32; N_EMBD];
    for k in 0..N_BANDS {
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
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut q_all = vec![vec![0.0f32; N_EMBD]; t];
    let mut k_all = vec![vec![0.0f32; N_EMBD]; t];
    let mut v_all = vec![vec![0.0f32; N_EMBD]; t];

    for pos in 0..t {
        let qkv = linear_fn(&weights.c_attn.w, &weights.c_attn.b, &x[pos]);
        for i in 0..N_EMBD {
            q_all[pos][i] = qkv[i];
            k_all[pos][i] = qkv[N_EMBD + i];
            v_all[pos][i] = qkv[2 * N_EMBD + i];
        }
    }

    let mut out = vec![vec![0.0f32; N_EMBD]; t];

    for head in 0..N_HEAD {
        let offset = head * HEAD_DIM;
        for qi in 0..t {
            let mut att = vec![f32::NEG_INFINITY; t];
            for ki in 0..=qi {
                let mut dot = 0.0f32;
                for d in 0..HEAD_DIM {
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

            for d in 0..HEAD_DIM {
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
    let mut result = Vec::with_capacity(t);

    for pos in 0..t {
        let mut bands_out = vec![0.0f32; N_EMBD];
        for band in 0..N_BANDS {
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

fn kerr_maestro_add_cpu(weights: &KerrMaestroAddWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let t = x.len();
    let mut result = Vec::with_capacity(t);

    for pos in 0..t {
        let kerr_out = kerr_ode_cpu(&weights.kerr, &x[pos]);
        let maestro_out = maestro_cpu(&weights.maestro, &x[pos]);
        let mut combined = vec![0.0f32; N_EMBD];
        for i in 0..N_EMBD {
            combined[i] = kerr_out[i] + maestro_out[i];
        }
        let projected = linear_fn(&weights.out_proj.w, &weights.out_proj.b, &combined);
        result.push(projected);
    }

    result
}
