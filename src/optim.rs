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

    /// Restore from checkpoint.
    pub fn from_checkpoint(lr: f32, t: usize, m: Vec<f32>, v: Vec<f32>) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t,
            m,
            v,
        }
    }

    /// Get state for checkpointing.
    pub fn checkpoint_state(&self) -> (usize, &[f32], &[f32]) {
        (self.t, &self.m, &self.v)
    }

    /// Extend m/v vectors with zeros for new parameters (e.g. vocab resize).
    pub fn extend(&mut self, extra: usize) {
        self.m.extend(std::iter::repeat(0.0f32).take(extra));
        self.v.extend(std::iter::repeat(0.0f32).take(extra));
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

/// Count params from vocab_size and config.
pub fn count_params_for_vocab(vocab_size: usize, config: &ModelConfig) -> usize {
    let n_embd = config.n_embd();
    let n_bands = config.n_bands;
    let maestro_dim = config.maestro_dim;
    let mut n = 0;

    // Block 0: PerBandLinear
    n += n_embd * 2; // ln_1
    n += 3 * n_embd * n_embd + 3 * n_embd; // c_attn
    n += n_embd * n_embd + n_embd; // c_proj
    n += n_embd * 2; // ln_2
    n += n_bands * 4 + n_bands * 2; // band_w + band_b
    n += n_embd * n_embd + n_embd; // out_proj

    // Blocks 1-(n_layers-1): KerrMaestro
    for _ in 0..(config.n_layers - 1) {
        n += n_embd * 2; // ln_1
        n += 3 * n_embd * n_embd + 3 * n_embd; // c_attn
        n += n_embd * n_embd + n_embd; // c_proj
        n += n_embd * 2; // ln_2
        n += n_bands + n_bands + 1 + 1; // gamma_raw, omega, alpha, beta
        n += maestro_dim * n_embd + maestro_dim; // squeeze
        n += n_embd * maestro_dim + n_embd; // process
        n += n_embd * n_embd + n_embd; // out_proj
    }

    n += n_embd * 2; // ln_f
    n += vocab_size * n_embd; // lm_head
    n
}

pub fn count_params(model: &ModelWeights) -> usize {
    let n_embd = model.config.n_embd();
    let n_bands = model.config.n_bands;
    let maestro_dim = model.config.maestro_dim;
    let mut n = 0;

    for block in &model.blocks {
        n += n_embd * 2; // ln_1 weight + bias
        n += 3 * n_embd * n_embd + 3 * n_embd; // c_attn
        n += n_embd * n_embd + n_embd; // c_proj
        n += n_embd * 2; // ln_2 weight + bias

        match &block.ffn {
            FfnWeights::PerBand(_) => {
                n += n_bands * 4 + n_bands * 2; // band_w + band_b
                n += n_embd * n_embd + n_embd; // out_proj
            }
            FfnWeights::KerrMaestro(_) => {
                n += n_bands + n_bands + 1 + 1; // gamma_raw, omega, alpha, beta
                n += maestro_dim * n_embd + maestro_dim; // squeeze
                n += n_embd * maestro_dim + n_embd; // process
                n += n_embd * n_embd + n_embd; // out_proj
            }
        }
    }

    n += n_embd * 2; // ln_f
    n += model.vocab_size * n_embd; // lm_head

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
    let n_embd = model.config.n_embd();
    let mut idx = 0;

    for block in &mut model.blocks {
        block.ln_1.weight.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
        block.ln_1.bias.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
        for row in &mut block.attn.c_attn.w {
            row.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
        }
        let attn_b_len = block.attn.c_attn.b.len();
        block.attn.c_attn.b.copy_from_slice(&params[idx..idx + attn_b_len]); idx += attn_b_len;
        for row in &mut block.attn.c_proj.w {
            row.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
        }
        block.attn.c_proj.b.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
        block.ln_2.weight.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
        block.ln_2.bias.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;

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
                    row.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
                }
                w.out_proj.b.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
            }
            FfnWeights::KerrMaestro(w) => {
                let n_bands = w.kerr.gamma_raw.len();
                let maestro_dim = w.maestro.squeeze.b.len();
                w.kerr.gamma_raw.copy_from_slice(&params[idx..idx + n_bands]); idx += n_bands;
                w.kerr.omega.copy_from_slice(&params[idx..idx + n_bands]); idx += n_bands;
                w.kerr.alpha = params[idx]; idx += 1;
                w.kerr.beta = params[idx]; idx += 1;
                for row in &mut w.maestro.squeeze.w {
                    row.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
                }
                w.maestro.squeeze.b.copy_from_slice(&params[idx..idx + maestro_dim]); idx += maestro_dim;
                for row in &mut w.maestro.process_1.w {
                    row.copy_from_slice(&params[idx..idx + maestro_dim]); idx += maestro_dim;
                }
                w.maestro.process_1.b.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
                for row in &mut w.out_proj.w {
                    row.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
                }
                w.out_proj.b.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
            }
        }
    }

    model.ln_f.weight.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
    model.ln_f.bias.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
    for row in &mut model.lm_head {
        row.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
    }

    assert_eq!(idx, params.len(), "Param count mismatch in unflatten");
}
