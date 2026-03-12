//! Optimizer — Adam with bias correction, gradient clipping,
//! and parameter flatten/unflatten for structured weights.

use crate::model::*;
use crate::pipeline::{GradAccum, FfnGrads};

// ─── Adam optimizer ─────────────────────────────────────────────

pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,
    m: Vec<f32>,  // first moment
    v: Vec<f32>,  // second moment
}

impl Adam {
    pub fn new(lr: f32, param_count: usize) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m: vec![0.0; param_count],
            v: vec![0.0; param_count],
        }
    }

    /// Step: update params in-place given gradients.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

// ─── Gradient clipping ────────────────────────────────────────

pub fn clip_grad_norm(grads: &mut [f32], max_norm: f32) {
    let norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grads.iter_mut() { *g *= scale; }
    }
}

// ─── Parameter count ──────────────────────────────────────────

pub fn count_params(model: &ModelWeights) -> usize {
    let mut n = 0;

    for block in &model.blocks {
        n += N_EMBD * 2; // ln_1 weight + bias
        n += 3 * N_EMBD * N_EMBD + 3 * N_EMBD; // c_attn
        n += N_EMBD * N_EMBD + N_EMBD; // c_proj
        n += N_EMBD * 2; // ln_2 weight + bias

        match &block.ffn {
            FfnWeights::PerBand(_) => {
                n += N_BANDS * 4 + N_BANDS * 2; // band_w + band_b
                n += N_EMBD * N_EMBD + N_EMBD; // out_proj
            }
            FfnWeights::KerrMaestro(_) => {
                n += N_BANDS + N_BANDS + 1 + 1; // gamma_raw, omega, alpha, beta
                n += MAESTRO_DIM * N_EMBD + MAESTRO_DIM; // squeeze
                n += N_EMBD * MAESTRO_DIM + N_EMBD; // process
                n += N_EMBD * N_EMBD + N_EMBD; // out_proj
            }
        }
    }

    n += N_EMBD * 2; // ln_f
    n += model.vocab_size * N_EMBD; // lm_head

    n
}

// ─── Parameter flattening ─────────────────────────────────────

/// Flatten all trainable parameters into a single Vec.
pub fn flatten_params(model: &ModelWeights) -> Vec<f32> {
    let n = count_params(model);
    let mut params = Vec::with_capacity(n);

    for block in &model.blocks {
        params.extend_from_slice(&block.ln_1.weight);
        params.extend_from_slice(&block.ln_1.bias);
        for row in &block.attn.c_attn.w { params.extend_from_slice(row); }
        params.extend_from_slice(&block.attn.c_attn.b);
        for row in &block.attn.c_proj.w { params.extend_from_slice(row); }
        params.extend_from_slice(&block.attn.c_proj.b);
        params.extend_from_slice(&block.ln_2.weight);
        params.extend_from_slice(&block.ln_2.bias);

        match &block.ffn {
            FfnWeights::PerBand(w) => {
                for band in &w.band_w {
                    params.extend_from_slice(&band[0]);
                    params.extend_from_slice(&band[1]);
                }
                for band in &w.band_b {
                    params.extend_from_slice(band);
                }
                for row in &w.out_proj.w { params.extend_from_slice(row); }
                params.extend_from_slice(&w.out_proj.b);
            }
            FfnWeights::KerrMaestro(w) => {
                params.extend_from_slice(&w.kerr.gamma_raw);
                params.extend_from_slice(&w.kerr.omega);
                params.push(w.kerr.alpha);
                params.push(w.kerr.beta);
                for row in &w.maestro.squeeze.w { params.extend_from_slice(row); }
                params.extend_from_slice(&w.maestro.squeeze.b);
                for row in &w.maestro.process_1.w { params.extend_from_slice(row); }
                params.extend_from_slice(&w.maestro.process_1.b);
                for row in &w.out_proj.w { params.extend_from_slice(row); }
                params.extend_from_slice(&w.out_proj.b);
            }
        }
    }

    params.extend_from_slice(&model.ln_f.weight);
    params.extend_from_slice(&model.ln_f.bias);
    for row in &model.lm_head { params.extend_from_slice(row); }

    params
}

/// Flatten gradients in the same order as flatten_params.
pub fn flatten_grads(grads: &GradAccum) -> Vec<f32> {
    let mut flat = Vec::new();

    for bg in &grads.blocks {
        flat.extend_from_slice(&bg.ln_1_weight);
        flat.extend_from_slice(&bg.ln_1_bias);
        for row in &bg.attn_c_attn_w { flat.extend_from_slice(row); }
        flat.extend_from_slice(&bg.attn_c_attn_b);
        for row in &bg.attn_c_proj_w { flat.extend_from_slice(row); }
        flat.extend_from_slice(&bg.attn_c_proj_b);
        flat.extend_from_slice(&bg.ln_2_weight);
        flat.extend_from_slice(&bg.ln_2_bias);

        match &bg.ffn {
            FfnGrads::PerBand { band_w, band_b, out_proj_w, out_proj_b } => {
                for band in band_w {
                    flat.extend_from_slice(&band[0]);
                    flat.extend_from_slice(&band[1]);
                }
                for band in band_b {
                    flat.extend_from_slice(band);
                }
                for row in out_proj_w { flat.extend_from_slice(row); }
                flat.extend_from_slice(out_proj_b);
            }
            FfnGrads::KerrMaestro {
                gamma_raw, omega, alpha, beta,
                squeeze_w, squeeze_b, process_w, process_b,
                out_proj_w, out_proj_b,
            } => {
                flat.extend_from_slice(gamma_raw);
                flat.extend_from_slice(omega);
                flat.push(*alpha);
                flat.push(*beta);
                for row in squeeze_w { flat.extend_from_slice(row); }
                flat.extend_from_slice(squeeze_b);
                for row in process_w { flat.extend_from_slice(row); }
                flat.extend_from_slice(process_b);
                for row in out_proj_w { flat.extend_from_slice(row); }
                flat.extend_from_slice(out_proj_b);
            }
        }
    }

    flat.extend_from_slice(&grads.ln_f_weight);
    flat.extend_from_slice(&grads.ln_f_bias);
    for row in &grads.lm_head { flat.extend_from_slice(row); }

    flat
}

/// Unflatten parameters back into the model.
pub fn unflatten_params(model: &mut ModelWeights, params: &[f32]) {
    let mut idx = 0;

    for block in &mut model.blocks {
        block.ln_1.weight.copy_from_slice(&params[idx..idx + N_EMBD]); idx += N_EMBD;
        block.ln_1.bias.copy_from_slice(&params[idx..idx + N_EMBD]); idx += N_EMBD;
        for row in &mut block.attn.c_attn.w {
            row.copy_from_slice(&params[idx..idx + N_EMBD]); idx += N_EMBD;
        }
        block.attn.c_attn.b.copy_from_slice(&params[idx..idx + 3 * N_EMBD]); idx += 3 * N_EMBD;
        for row in &mut block.attn.c_proj.w {
            row.copy_from_slice(&params[idx..idx + N_EMBD]); idx += N_EMBD;
        }
        block.attn.c_proj.b.copy_from_slice(&params[idx..idx + N_EMBD]); idx += N_EMBD;
        block.ln_2.weight.copy_from_slice(&params[idx..idx + N_EMBD]); idx += N_EMBD;
        block.ln_2.bias.copy_from_slice(&params[idx..idx + N_EMBD]); idx += N_EMBD;

        match &mut block.ffn {
            FfnWeights::PerBand(w) => {
                for band in &mut w.band_w {
                    band[0].copy_from_slice(&params[idx..idx + 2]); idx += 2;
                    band[1].copy_from_slice(&params[idx..idx + 2]); idx += 2;
                }
                for band in &mut w.band_b {
                    band.copy_from_slice(&params[idx..idx + 2]); idx += 2;
                }
                for row in &mut w.out_proj.w {
                    row.copy_from_slice(&params[idx..idx + N_EMBD]); idx += N_EMBD;
                }
                w.out_proj.b.copy_from_slice(&params[idx..idx + N_EMBD]); idx += N_EMBD;
            }
            FfnWeights::KerrMaestro(w) => {
                w.kerr.gamma_raw.copy_from_slice(&params[idx..idx + N_BANDS]); idx += N_BANDS;
                w.kerr.omega.copy_from_slice(&params[idx..idx + N_BANDS]); idx += N_BANDS;
                w.kerr.alpha = params[idx]; idx += 1;
                w.kerr.beta = params[idx]; idx += 1;
                for row in &mut w.maestro.squeeze.w {
                    row.copy_from_slice(&params[idx..idx + N_EMBD]); idx += N_EMBD;
                }
                w.maestro.squeeze.b.copy_from_slice(&params[idx..idx + MAESTRO_DIM]); idx += MAESTRO_DIM;
                for row in &mut w.maestro.process_1.w {
                    row.copy_from_slice(&params[idx..idx + MAESTRO_DIM]); idx += MAESTRO_DIM;
                }
                w.maestro.process_1.b.copy_from_slice(&params[idx..idx + N_EMBD]); idx += N_EMBD;
                for row in &mut w.out_proj.w {
                    row.copy_from_slice(&params[idx..idx + N_EMBD]); idx += N_EMBD;
                }
                w.out_proj.b.copy_from_slice(&params[idx..idx + N_EMBD]); idx += N_EMBD;
            }
        }
    }

    model.ln_f.weight.copy_from_slice(&params[idx..idx + N_EMBD]); idx += N_EMBD;
    model.ln_f.bias.copy_from_slice(&params[idx..idx + N_EMBD]); idx += N_EMBD;
    for row in &mut model.lm_head {
        row.copy_from_slice(&params[idx..idx + N_EMBD]); idx += N_EMBD;
    }

    assert_eq!(idx, params.len(), "Param count mismatch in unflatten");
}
