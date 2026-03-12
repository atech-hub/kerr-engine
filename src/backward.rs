//! Backward pass — analytical gradients for the full model.
//!
//! No autograd. Every gradient is hand-derived.
//!
//! Notation: d_x means d(loss)/d(x).

use crate::model::*;
use std::f32::consts::PI;

// ─── Primitive backward ops ─────────────────────────────────────

/// Backward through y = W @ x + b
/// Returns (d_x, d_w, d_b)
pub fn linear_backward(
    d_y: &[f32],      // [out_dim]
    x: &[f32],        // [in_dim]
    w: &[Vec<f32>],   // [out_dim][in_dim]
) -> (Vec<f32>, Vec<Vec<f32>>, Vec<f32>) {
    let out_dim = w.len();
    let in_dim = x.len();

    // d_x[j] = sum_i d_y[i] * w[i][j]
    let mut d_x = vec![0.0f32; in_dim];
    for j in 0..in_dim {
        let mut sum = 0.0f32;
        for i in 0..out_dim {
            sum += d_y[i] * w[i][j];
        }
        d_x[j] = sum;
    }

    // d_w[i][j] = d_y[i] * x[j]
    let mut d_w = vec![vec![0.0f32; in_dim]; out_dim];
    for i in 0..out_dim {
        for j in 0..in_dim {
            d_w[i][j] = d_y[i] * x[j];
        }
    }

    // d_b[i] = d_y[i]
    let d_b = d_y.to_vec();

    (d_x, d_w, d_b)
}

/// Backward through y = W @ x (no bias, for lm_head)
/// Returns (d_x, d_w)
pub fn linear_no_bias_backward(
    d_y: &[f32],
    x: &[f32],
    w: &[Vec<f32>],
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let out_dim = w.len();
    let in_dim = x.len();

    let mut d_x = vec![0.0f32; in_dim];
    for j in 0..in_dim {
        for i in 0..out_dim {
            d_x[j] += d_y[i] * w[i][j];
        }
    }

    let mut d_w = vec![vec![0.0f32; in_dim]; out_dim];
    for i in 0..out_dim {
        for j in 0..in_dim {
            d_w[i][j] = d_y[i] * x[j];
        }
    }

    (d_x, d_w)
}

/// Backward through layer norm.
/// y = (x - mean) / std * weight + bias
/// where std = sqrt(var + eps), var = mean((x - mean)^2)
pub fn layer_norm_backward(
    d_y: &[f32],
    x: &[f32],
    weight: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = x.len();
    let nf = n as f32;
    let eps = 1e-5f32;

    let mean: f32 = x.iter().sum::<f32>() / nf;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / nf;
    let std = (var + eps).sqrt();
    let inv_std = 1.0 / std;

    let x_hat: Vec<f32> = x.iter().map(|v| (v - mean) * inv_std).collect();

    // d_weight[i] = d_y[i] * x_hat[i]
    let d_weight: Vec<f32> = (0..n).map(|i| d_y[i] * x_hat[i]).collect();

    // d_bias[i] = d_y[i]
    let d_bias = d_y.to_vec();

    // d_x_hat[i] = d_y[i] * weight[i]
    let d_x_hat: Vec<f32> = (0..n).map(|i| d_y[i] * weight[i]).collect();

    // d_x through layer norm (standard formula)
    let d_x_hat_sum: f32 = d_x_hat.iter().sum();
    let d_x_hat_x_hat_sum: f32 = (0..n).map(|i| d_x_hat[i] * x_hat[i]).sum();

    let mut d_x = vec![0.0f32; n];
    for i in 0..n {
        d_x[i] = inv_std / nf * (nf * d_x_hat[i] - d_x_hat_sum - x_hat[i] * d_x_hat_x_hat_sum);
    }

    (d_x, d_weight, d_bias)
}

/// Backward through GELU activation.
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu_backward(d_y: &[f32], x: &[f32]) -> Vec<f32> {
    let sqrt_2_pi = (2.0f32 / PI).sqrt();

    x.iter().zip(d_y.iter()).map(|(&xi, &dy)| {
        let x3 = xi * xi * xi;
        let inner = sqrt_2_pi * (xi + 0.044715 * x3);
        let tanh_inner = inner.tanh();
        let sech2 = 1.0 - tanh_inner * tanh_inner;
        let d_inner = sqrt_2_pi * (1.0 + 3.0 * 0.044715 * xi * xi);

        // d_gelu/dx = 0.5 * (1 + tanh) + 0.5 * x * sech^2 * d_inner
        let grad = 0.5 * (1.0 + tanh_inner) + 0.5 * xi * sech2 * d_inner;
        dy * grad
    }).collect()
}

/// Backward through softplus: y = log(1 + exp(x))
/// dy/dx = sigmoid(x) = exp(x) / (1 + exp(x))
pub fn softplus_backward(d_y: f32, x: f32) -> f32 {
    let sig = if x > 20.0 { 1.0 } else { x.exp() / (1.0 + x.exp()) };
    d_y * sig
}

// ─── Cross-entropy loss backward ────────────────────────────────

/// Backward through cross-entropy loss with logits.
/// Returns d_logits = softmax(logits) - one_hot(target)
pub fn cross_entropy_backward(logits: &[f32], target: usize) -> Vec<f32> {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let exp_l: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum_exp: f32 = exp_l.iter().sum();

    let mut d_logits: Vec<f32> = exp_l.iter().map(|&e| e / sum_exp).collect();
    d_logits[target] -= 1.0;

    // Normalize by 1 (single token loss, not batch-averaged here)
    d_logits
}

// ─── Kerr-ODE backward ─────────────────────────────────────────

/// Saved state from one derivative evaluation (needed for backward).
pub struct DerivativeCache {
    r: Vec<f32>,
    s: Vec<f32>,
    mag_sq: Vec<f32>,
    ns: Vec<f32>,
    phi: Vec<f32>,
}

/// Forward derivative with cache for backward.
pub fn kerr_derivative_with_cache_pub(
    r: &[f32], s: &[f32],
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32,
) -> (Vec<f32>, Vec<f32>, DerivativeCache) {
    let n = r.len();

    let mag_sq: Vec<f32> = (0..n).map(|k| r[k] * r[k] + s[k] * s[k]).collect();

    let mut ns = vec![0.0f32; n];
    for k in 0..n {
        if k >= 2 { ns[k] += mag_sq[k - 2]; }
        if k >= 1 { ns[k] += mag_sq[k - 1]; }
        if k + 1 < n { ns[k] += mag_sq[k + 1]; }
        if k + 2 < n { ns[k] += mag_sq[k + 2]; }
    }

    let phi: Vec<f32> = (0..n).map(|k| omega[k] + alpha * mag_sq[k] + beta * ns[k]).collect();

    let mut dr = vec![0.0f32; n];
    let mut ds = vec![0.0f32; n];
    for k in 0..n {
        dr[k] = -gamma[k] * r[k] - phi[k] * s[k];
        ds[k] = -gamma[k] * s[k] + phi[k] * r[k];
    }

    let cache = DerivativeCache {
        r: r.to_vec(), s: s.to_vec(), mag_sq, ns, phi,
    };

    (dr, ds, cache)
}

/// Backward through the Kerr derivative.
///
/// Given d_loss/d_dr and d_loss/d_ds, compute:
/// - d_loss/d_r, d_loss/d_s (input gradients)
/// - d_loss/d_gamma, d_loss/d_omega, d_loss/d_alpha, d_loss/d_beta (param gradients)
pub fn kerr_derivative_backward_pub(
    d_dr: &[f32], d_ds: &[f32],
    cache: &DerivativeCache,
    gamma: &[f32],
    alpha: f32, beta: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32, f32) {
    let n = cache.r.len();
    let r = &cache.r;
    let s = &cache.s;
    let mag_sq = &cache.mag_sq;
    let ns = &cache.ns;
    let phi = &cache.phi;

    // dr[k] = -gamma[k]*r[k] - phi[k]*s[k]
    // ds[k] = -gamma[k]*s[k] + phi[k]*r[k]

    // d_gamma[k] = d_dr[k] * (-r[k]) + d_ds[k] * (-s[k])
    let d_gamma: Vec<f32> = (0..n)
        .map(|k| d_dr[k] * (-r[k]) + d_ds[k] * (-s[k]))
        .collect();

    // d_phi[k] = d_dr[k] * (-s[k]) + d_ds[k] * r[k]
    let d_phi: Vec<f32> = (0..n)
        .map(|k| d_dr[k] * (-s[k]) + d_ds[k] * r[k])
        .collect();

    // phi[k] = omega[k] + alpha*mag_sq[k] + beta*ns[k]
    // d_omega[k] = d_phi[k]
    let d_omega = d_phi.clone();

    // d_alpha = sum_k d_phi[k] * mag_sq[k]
    let d_alpha: f32 = (0..n).map(|k| d_phi[k] * mag_sq[k]).sum();

    // d_beta = sum_k d_phi[k] * ns[k]
    let d_beta: f32 = (0..n).map(|k| d_phi[k] * ns[k]).sum();

    // d_mag_sq[k] from phi: d_phi[k] * alpha
    let mut d_mag_sq: Vec<f32> = (0..n).map(|k| d_phi[k] * alpha).collect();

    // d_mag_sq from ns (conv1d transpose with kernel [1,1,0,1,1])
    // ns[k] = mag_sq[k-2] + mag_sq[k-1] + mag_sq[k+1] + mag_sq[k+2]
    // So d_ns -> d_mag_sq: transpose convolution
    let d_ns: Vec<f32> = (0..n).map(|k| d_phi[k] * beta).collect();
    for k in 0..n {
        if k >= 2 { d_mag_sq[k - 2] += d_ns[k]; }
        if k >= 1 { d_mag_sq[k - 1] += d_ns[k]; }
        if k + 1 < n { d_mag_sq[k + 1] += d_ns[k]; }
        if k + 2 < n { d_mag_sq[k + 2] += d_ns[k]; }
    }

    // mag_sq[k] = r[k]^2 + s[k]^2
    // d_r[k] from mag_sq: d_mag_sq[k] * 2*r[k]
    // d_s[k] from mag_sq: d_mag_sq[k] * 2*s[k]

    // d_r[k] = d_dr[k] * (-gamma[k]) + d_ds[k] * phi[k] + d_mag_sq[k] * 2*r[k]
    let d_r: Vec<f32> = (0..n)
        .map(|k| d_dr[k] * (-gamma[k]) + d_ds[k] * phi[k] + d_mag_sq[k] * 2.0 * r[k])
        .collect();

    // d_s[k] = d_dr[k] * (-phi[k]) + d_ds[k] * (-gamma[k]) + d_mag_sq[k] * 2*s[k]
    let d_s: Vec<f32> = (0..n)
        .map(|k| d_dr[k] * (-phi[k]) + d_ds[k] * (-gamma[k]) + d_mag_sq[k] * 2.0 * s[k])
        .collect();

    (d_r, d_s, d_gamma, d_omega, d_alpha, d_beta)
}

/// Backward through one RK4 step.
///
/// Forward: r_new = r + (dt/6)(k1 + 2*k2 + 2*k3 + k4)
///          where k1 = f(r,s), k2 = f(r+dt/2*k1_r, s+dt/2*k1_s), etc.
///
/// Returns: (d_r, d_s) input gradients + accumulated parameter gradients.
pub fn rk4_step_backward(
    d_r_new: &[f32], d_s_new: &[f32],
    // Forward caches (saved during forward pass)
    r: &[f32], s: &[f32],
    dt: f32,
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32, f32) {
    let n = r.len();

    // Recompute forward (we need intermediate states)
    let (dr1, ds1, cache1) = kerr_derivative_with_cache_pub(r, s, gamma, omega, alpha, beta);

    let r2: Vec<f32> = (0..n).map(|k| r[k] + 0.5 * dt * dr1[k]).collect();
    let s2: Vec<f32> = (0..n).map(|k| s[k] + 0.5 * dt * ds1[k]).collect();
    let (dr2, ds2, cache2) = kerr_derivative_with_cache_pub(&r2, &s2, gamma, omega, alpha, beta);

    let r3: Vec<f32> = (0..n).map(|k| r[k] + 0.5 * dt * dr2[k]).collect();
    let s3: Vec<f32> = (0..n).map(|k| s[k] + 0.5 * dt * ds2[k]).collect();
    let (dr3, ds3, cache3) = kerr_derivative_with_cache_pub(&r3, &s3, gamma, omega, alpha, beta);

    let r4: Vec<f32> = (0..n).map(|k| r[k] + dt * dr3[k]).collect();
    let s4: Vec<f32> = (0..n).map(|k| s[k] + dt * ds3[k]).collect();
    let (_dr4, _ds4, cache4) = kerr_derivative_with_cache_pub(&r4, &s4, gamma, omega, alpha, beta);

    // r_new = r + dt/6 * (dr1 + 2*dr2 + 2*dr3 + dr4)
    let dt6 = dt / 6.0;

    // Accumulate parameter gradients
    let mut d_gamma_acc = vec![0.0f32; n];
    let mut d_omega_acc = vec![0.0f32; n];
    let mut d_alpha_acc = 0.0f32;
    let mut d_beta_acc = 0.0f32;

    // d_r_new -> d_r (direct path: r_new = r + ...)
    let mut d_r = d_r_new.to_vec();
    let mut d_s = d_s_new.to_vec();

    // d_r_new -> d_dr4 (coefficient: dt/6)
    let d_dr4: Vec<f32> = (0..n).map(|k| d_r_new[k] * dt6).collect();
    let d_ds4: Vec<f32> = (0..n).map(|k| d_s_new[k] * dt6).collect();

    // Backward through k4 = f(r4, s4)
    let (d_r4, d_s4, dg, dom, da, db) =
        kerr_derivative_backward_pub(&d_dr4, &d_ds4, &cache4, gamma, alpha, beta);
    for k in 0..n { d_gamma_acc[k] += dg[k]; d_omega_acc[k] += dom[k]; }
    d_alpha_acc += da; d_beta_acc += db;

    // r4 = r + dt*dr3, so d_r += d_r4, d_dr3 += d_r4 * dt
    for k in 0..n { d_r[k] += d_r4[k]; d_s[k] += d_s4[k]; }
    let d_dr3: Vec<f32> = (0..n).map(|k| d_r4[k] * dt + d_r_new[k] * dt6 * 2.0).collect();
    let d_ds3: Vec<f32> = (0..n).map(|k| d_s4[k] * dt + d_s_new[k] * dt6 * 2.0).collect();

    // Backward through k3 = f(r3, s3)
    let (d_r3, d_s3, dg, dom, da, db) =
        kerr_derivative_backward_pub(&d_dr3, &d_ds3, &cache3, gamma, alpha, beta);
    for k in 0..n { d_gamma_acc[k] += dg[k]; d_omega_acc[k] += dom[k]; }
    d_alpha_acc += da; d_beta_acc += db;

    // r3 = r + 0.5*dt*dr2
    for k in 0..n { d_r[k] += d_r3[k]; d_s[k] += d_s3[k]; }
    let d_dr2: Vec<f32> = (0..n).map(|k| d_r3[k] * 0.5 * dt + d_r_new[k] * dt6 * 2.0).collect();
    let d_ds2: Vec<f32> = (0..n).map(|k| d_s3[k] * 0.5 * dt + d_s_new[k] * dt6 * 2.0).collect();

    // Backward through k2 = f(r2, s2)
    let (d_r2, d_s2, dg, dom, da, db) =
        kerr_derivative_backward_pub(&d_dr2, &d_ds2, &cache2, gamma, alpha, beta);
    for k in 0..n { d_gamma_acc[k] += dg[k]; d_omega_acc[k] += dom[k]; }
    d_alpha_acc += da; d_beta_acc += db;

    // r2 = r + 0.5*dt*dr1
    for k in 0..n { d_r[k] += d_r2[k]; d_s[k] += d_s2[k]; }
    let d_dr1: Vec<f32> = (0..n).map(|k| d_r2[k] * 0.5 * dt + d_r_new[k] * dt6).collect();
    let d_ds1: Vec<f32> = (0..n).map(|k| d_s2[k] * 0.5 * dt + d_s_new[k] * dt6).collect();

    // Backward through k1 = f(r, s)
    let (d_r1_in, d_s1_in, dg, dom, da, db) =
        kerr_derivative_backward_pub(&d_dr1, &d_ds1, &cache1, gamma, alpha, beta);
    for k in 0..n { d_gamma_acc[k] += dg[k]; d_omega_acc[k] += dom[k]; }
    d_alpha_acc += da; d_beta_acc += db;

    for k in 0..n { d_r[k] += d_r1_in[k]; d_s[k] += d_s1_in[k]; }

    (d_r, d_s, d_gamma_acc, d_omega_acc, d_alpha_acc, d_beta_acc)
}

/// Backward through the full Kerr-ODE (8 RK4 steps).
pub fn kerr_ode_backward(
    d_output: &[f32],  // [N_EMBD] gradient from downstream
    input: &[f32],     // [N_EMBD] original input (saved from forward)
    weights: &KerrWeights,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, f32, f32) {
    // Split d_output into d_r, d_s
    let mut d_r: Vec<f32> = (0..N_BANDS).map(|k| d_output[k * 2]).collect();
    let mut d_s: Vec<f32> = (0..N_BANDS).map(|k| d_output[k * 2 + 1]).collect();

    // Recompute forward states for each RK4 step
    let mut r: Vec<f32> = (0..N_BANDS).map(|k| input[k * 2]).collect();
    let mut s: Vec<f32> = (0..N_BANDS).map(|k| input[k * 2 + 1]).collect();

    let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| {
        if g > 20.0 { g } else { (1.0 + g.exp()).ln() }
    }).collect();

    // Save all intermediate states
    let mut states: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(RK4_N_STEPS + 1);
    states.push((r.clone(), s.clone()));

    for _ in 0..RK4_N_STEPS {
        let (r_new, s_new) = crate::model::rk4_step_public(
            &r, &s, RK4_DT, &gamma, &weights.omega, weights.alpha, weights.beta,
        );
        r = r_new;
        s = s_new;
        states.push((r.clone(), s.clone()));
    }

    // Backward through steps in reverse
    let mut d_gamma_raw_acc = vec![0.0f32; N_BANDS];
    let mut d_omega_acc = vec![0.0f32; N_BANDS];
    let mut d_alpha_acc = 0.0f32;
    let mut d_beta_acc = 0.0f32;

    for step in (0..RK4_N_STEPS).rev() {
        let (ref r_step, ref s_step) = states[step];

        let (d_r_new, d_s_new, d_gamma, d_omega, d_alpha, d_beta) =
            rk4_step_backward(
                &d_r, &d_s,
                r_step, s_step,
                RK4_DT,
                &gamma, &weights.omega,
                weights.alpha, weights.beta,
            );

        d_r = d_r_new;
        d_s = d_s_new;
        for k in 0..N_BANDS {
            d_gamma_raw_acc[k] += d_gamma[k];
            d_omega_acc[k] += d_omega[k];
        }
        d_alpha_acc += d_alpha;
        d_beta_acc += d_beta;
    }

    // Chain through softplus for gamma_raw
    let d_gamma_raw: Vec<f32> = (0..N_BANDS)
        .map(|k| softplus_backward(d_gamma_raw_acc[k], weights.gamma_raw[k]))
        .collect();

    // Reinterleave d_r, d_s -> d_input
    let mut d_input = vec![0.0f32; N_EMBD];
    for k in 0..N_BANDS {
        d_input[k * 2] = d_r[k];
        d_input[k * 2 + 1] = d_s[k];
    }

    (d_input, d_gamma_raw, d_omega_acc, d_alpha_acc, d_beta_acc)
}

// ─── Attention backward ─────────────────────────────────────────

/// Backward through causal self-attention for a single query position.
/// This is the full attention backward — scores, softmax, value aggregation.
pub fn attention_backward_single(
    d_out: &[f32],           // [N_EMBD] gradient for this position
    q_all: &[Vec<f32>],     // [T][N_EMBD] all queries
    k_all: &[Vec<f32>],     // [T][N_EMBD] all keys
    v_all: &[Vec<f32>],     // [T][N_EMBD] all values
    att_weights: &[Vec<f32>], // [N_HEAD][T] post-softmax weights per head at this position
    pos: usize,              // which position we're computing grad for
    d_q_all: &mut [Vec<f32>],
    d_k_all: &mut [Vec<f32>],
    d_v_all: &mut [Vec<f32>],
) {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    for head in 0..N_HEAD {
        let offset = head * HEAD_DIM;
        let att = &att_weights[head]; // attention weights for this head at this position

        // Backward through: out[pos][offset+d] = sum_ki att[ki] * v[ki][offset+d]
        // d_att[ki] = sum_d d_out[offset+d] * v[ki][offset+d]
        // d_v[ki][offset+d] += att[ki] * d_out[offset+d]
        let mut d_att = vec![0.0f32; pos + 1];
        for ki in 0..=pos {
            let mut dot = 0.0f32;
            for d in 0..HEAD_DIM {
                dot += d_out[offset + d] * v_all[ki][offset + d];
                d_v_all[ki][offset + d] += att[ki] * d_out[offset + d];
            }
            d_att[ki] = dot;
        }

        // Backward through softmax
        // d_score[ki] = att[ki] * (d_att[ki] - sum_j att[j] * d_att[j])
        let att_d_att_sum: f32 = (0..=pos).map(|j| att[j] * d_att[j]).sum();
        let mut d_score = vec![0.0f32; pos + 1];
        for ki in 0..=pos {
            d_score[ki] = att[ki] * (d_att[ki] - att_d_att_sum);
        }

        // Backward through: score[ki] = (q[pos] . k[ki]) * scale
        // d_q[pos][d] += d_score[ki] * k[ki][d] * scale
        // d_k[ki][d] += d_score[ki] * q[pos][d] * scale
        for ki in 0..=pos {
            for d in 0..HEAD_DIM {
                d_q_all[pos][offset + d] += d_score[ki] * k_all[ki][offset + d] * scale;
                d_k_all[ki][offset + d] += d_score[ki] * q_all[pos][offset + d] * scale;
            }
        }
    }
}

// ─── Maestro backward ───────────────────────────────────────────

/// Backward through Maestro: squeeze -> GELU -> process
pub fn maestro_backward(
    d_output: &[f32],
    input: &[f32],
    weights: &MaestroWeights,
) -> (Vec<f32>, LinearGrads, LinearGrads) {
    // Forward recompute
    let squeezed = crate::model::linear_fn(&weights.squeeze.w, &weights.squeeze.b, input);
    let activated: Vec<f32> = squeezed.iter().map(|&v| {
        let sqrt_2_pi = (2.0f32 / PI).sqrt();
        0.5 * v * (1.0 + (sqrt_2_pi * (v + 0.044715 * v * v * v)).tanh())
    }).collect();

    // Backward through process linear
    let (d_activated, d_process_w, d_process_b) =
        linear_backward(d_output, &activated, &weights.process_1.w);

    // Backward through GELU
    let d_squeezed = gelu_backward(&d_activated, &squeezed);

    // Backward through squeeze linear
    let (d_input, d_squeeze_w, d_squeeze_b) =
        linear_backward(&d_squeezed, input, &weights.squeeze.w);

    (
        d_input,
        LinearGrads { d_w: d_squeeze_w, d_b: d_squeeze_b },
        LinearGrads { d_w: d_process_w, d_b: d_process_b },
    )
}

/// Gradient storage for a Linear layer.
pub struct LinearGrads {
    pub d_w: Vec<Vec<f32>>,
    pub d_b: Vec<f32>,
}

