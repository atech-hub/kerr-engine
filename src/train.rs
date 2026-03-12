//! Training loop — full backward pass + Adam optimizer.
//!
//! No autograd, no computation graph. Every gradient flows through
//! hand-derived analytical backward passes.

use crate::model::*;
use crate::backward::*;
use std::fs;
use std::collections::HashMap;

// ─── Adam optimizer ─────────────────────────────────────────────

pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,
    // First and second moment estimates for each parameter group
    states: Vec<AdamState>,
}

struct AdamState {
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
            states: vec![AdamState {
                m: vec![0.0; param_count],
                v: vec![0.0; param_count],
            }],
        }
    }

    /// Step: update params in-place given gradients.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.t += 1;
        let state = &mut self.states[0];
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..params.len() {
            state.m[i] = self.beta1 * state.m[i] + (1.0 - self.beta1) * grads[i];
            state.v[i] = self.beta2 * state.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            let m_hat = state.m[i] / bc1;
            let v_hat = state.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

// ─── Gradient accumulator ───────────────────────────────────────

/// Flat storage for all trainable parameter gradients.
pub struct GradAccum {
    // Per block: ln_1, attn, ln_2, ffn
    pub blocks: Vec<BlockGrads>,
    pub ln_f_weight: Vec<f32>,
    pub ln_f_bias: Vec<f32>,
    pub lm_head: Vec<Vec<f32>>,
}

pub struct BlockGrads {
    pub ln_1_weight: Vec<f32>,
    pub ln_1_bias: Vec<f32>,
    pub attn_c_attn_w: Vec<Vec<f32>>,
    pub attn_c_attn_b: Vec<f32>,
    pub attn_c_proj_w: Vec<Vec<f32>>,
    pub attn_c_proj_b: Vec<f32>,
    pub ln_2_weight: Vec<f32>,
    pub ln_2_bias: Vec<f32>,
    pub ffn: FfnGrads,
}

pub enum FfnGrads {
    PerBand {
        band_w: Vec<[[f32; 2]; 2]>,
        band_b: Vec<[f32; 2]>,
        out_proj_w: Vec<Vec<f32>>,
        out_proj_b: Vec<f32>,
    },
    KerrMaestro {
        gamma_raw: Vec<f32>,
        omega: Vec<f32>,
        alpha: f32,
        beta: f32,
        squeeze_w: Vec<Vec<f32>>,
        squeeze_b: Vec<f32>,
        process_w: Vec<Vec<f32>>,
        process_b: Vec<f32>,
        out_proj_w: Vec<Vec<f32>>,
        out_proj_b: Vec<f32>,
    },
}

impl GradAccum {
    pub fn zeros(weights: &ModelWeights) -> Self {
        let mut blocks = Vec::new();
        for block in &weights.blocks {
            let ffn = match &block.ffn {
                FfnWeights::PerBand(w) => FfnGrads::PerBand {
                    band_w: vec![[[0.0; 2]; 2]; N_BANDS],
                    band_b: vec![[0.0; 2]; N_BANDS],
                    out_proj_w: vec![vec![0.0; N_EMBD]; N_EMBD],
                    out_proj_b: vec![0.0; N_EMBD],
                },
                FfnWeights::KerrMaestro(_) => FfnGrads::KerrMaestro {
                    gamma_raw: vec![0.0; N_BANDS],
                    omega: vec![0.0; N_BANDS],
                    alpha: 0.0,
                    beta: 0.0,
                    squeeze_w: vec![vec![0.0; N_EMBD]; MAESTRO_DIM],
                    squeeze_b: vec![0.0; MAESTRO_DIM],
                    process_w: vec![vec![0.0; MAESTRO_DIM]; N_EMBD],
                    process_b: vec![0.0; N_EMBD],
                    out_proj_w: vec![vec![0.0; N_EMBD]; N_EMBD],
                    out_proj_b: vec![0.0; N_EMBD],
                },
            };
            blocks.push(BlockGrads {
                ln_1_weight: vec![0.0; N_EMBD],
                ln_1_bias: vec![0.0; N_EMBD],
                attn_c_attn_w: vec![vec![0.0; N_EMBD]; 3 * N_EMBD],
                attn_c_attn_b: vec![0.0; 3 * N_EMBD],
                attn_c_proj_w: vec![vec![0.0; N_EMBD]; N_EMBD],
                attn_c_proj_b: vec![0.0; N_EMBD],
                ln_2_weight: vec![0.0; N_EMBD],
                ln_2_bias: vec![0.0; N_EMBD],
                ffn,
            });
        }
        Self {
            blocks,
            ln_f_weight: vec![0.0; N_EMBD],
            ln_f_bias: vec![0.0; N_EMBD],
            lm_head: vec![vec![0.0; N_EMBD]; weights.vocab_size],
        }
    }
}

// ─── Full backward pass ─────────────────────────────────────────

impl ModelWeights {
    /// Forward pass with saved activations for backward.
    pub fn forward_with_cache(&self, tokens: &[usize]) -> ForwardCache {
        let t = tokens.len();

        // Embedding + positional
        let mut hidden: Vec<Vec<f32>> = Vec::with_capacity(t);
        for (pos, &tok) in tokens.iter().enumerate() {
            let mut h = vec![0.0f32; N_EMBD];
            for i in 0..N_EMBD {
                h[i] = self.wte_phase[tok][i] + self.wpe[pos][i];
            }
            hidden.push(h);
        }

        let mut block_caches = Vec::new();

        for block in &self.blocks {
            let cache = self.forward_block_with_cache(block, &hidden);
            hidden = cache.output.clone();
            block_caches.push(cache);
        }

        // Final layer norm
        let normed: Vec<Vec<f32>> = hidden.iter()
            .map(|h| layer_norm_fn(h, &self.ln_f.weight, &self.ln_f.bias))
            .collect();

        // LM head
        let logits: Vec<Vec<f32>> = normed.iter()
            .map(|n| linear_no_bias_fn(&self.lm_head, n))
            .collect();

        ForwardCache {
            tokens: tokens.to_vec(),
            block_caches,
            pre_ln_f: hidden.iter().map(|h| h.clone()).collect(),
            post_ln_f: normed,
            logits,
        }
    }

    fn forward_block_with_cache(&self, block: &BlockWeights, hidden: &[Vec<f32>]) -> BlockCache {
        let t = hidden.len();

        // LN1
        let normed_1: Vec<Vec<f32>> = hidden.iter()
            .map(|h| layer_norm_fn(h, &block.ln_1.weight, &block.ln_1.bias))
            .collect();

        // Attention
        let (attn_out, attn_cache) = self.attention_with_cache(&block.attn, &normed_1);

        // Residual 1
        let h1: Vec<Vec<f32>> = (0..t).map(|i| {
            let mut v = vec![0.0f32; N_EMBD];
            for j in 0..N_EMBD { v[j] = hidden[i][j] + attn_out[i][j]; }
            v
        }).collect();

        // LN2
        let normed_2: Vec<Vec<f32>> = h1.iter()
            .map(|h| layer_norm_fn(h, &block.ln_2.weight, &block.ln_2.bias))
            .collect();

        // FFN
        let ffn_out = match &block.ffn {
            FfnWeights::PerBand(w) => self.per_band_linear(w, &normed_2),
            FfnWeights::KerrMaestro(w) => self.kerr_maestro_add(w, &normed_2),
        };

        // Residual 2
        let output: Vec<Vec<f32>> = (0..t).map(|i| {
            let mut v = vec![0.0f32; N_EMBD];
            for j in 0..N_EMBD { v[j] = h1[i][j] + ffn_out[i][j]; }
            v
        }).collect();

        BlockCache {
            input: hidden.to_vec(),
            normed_1,
            attn_cache,
            h1,
            ffn_input: normed_2.clone(),
            normed_2,
            output,
        }
    }

    fn attention_with_cache(&self, weights: &AttentionWeights, x: &[Vec<f32>]) -> (Vec<Vec<f32>>, AttnCache) {
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

        // Compute attention weights and output per head
        let mut att_weights_all = vec![vec![vec![0.0f32; t]; t]; N_HEAD];
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

                att_weights_all[head][qi] = att.clone();

                for d in 0..HEAD_DIM {
                    let mut sum = 0.0f32;
                    for ki in 0..=qi {
                        sum += att[ki] * v_all[ki][offset + d];
                    }
                    out[qi][offset + d] = sum;
                }
            }
        }

        // c_proj
        let pre_proj = out.clone();
        let result: Vec<Vec<f32>> = out.iter()
            .map(|o| linear_fn(&weights.c_proj.w, &weights.c_proj.b, o))
            .collect();

        let cache = AttnCache {
            input: x.to_vec(),
            q_all, k_all, v_all,
            att_weights: att_weights_all,
            pre_proj,
        };

        (result, cache)
    }

    /// Full backward pass: compute gradients for all parameters.
    pub fn backward(&self, cache: &ForwardCache, targets: &[usize]) -> (f32, GradAccum) {
        let t = cache.tokens.len();
        let mut grads = GradAccum::zeros(self);

        // Loss: cross-entropy per position, averaged
        let mut total_loss = 0.0f32;
        let mut d_logits: Vec<Vec<f32>> = Vec::with_capacity(t);

        for pos in 0..t {
            let logits = &cache.logits[pos];
            let target = targets[pos];

            // Forward loss
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_l: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
            let sum_exp: f32 = exp_l.iter().sum();
            total_loss += -(exp_l[target] / sum_exp).ln();

            // Gradient of cross-entropy
            let mut dl = cross_entropy_backward(logits, target);
            // Average over positions
            for v in &mut dl { *v /= t as f32; }
            d_logits.push(dl);
        }
        total_loss /= t as f32;

        // Backward through LM head (no bias)
        let mut d_hidden: Vec<Vec<f32>> = Vec::with_capacity(t);
        for pos in 0..t {
            let (d_normed, d_lm_head) = linear_no_bias_backward(
                &d_logits[pos], &cache.post_ln_f[pos], &self.lm_head,
            );
            // Accumulate lm_head grads
            for i in 0..self.vocab_size {
                for j in 0..N_EMBD {
                    grads.lm_head[i][j] += d_lm_head[i][j];
                }
            }

            // Backward through final layer norm
            let (d_h, d_w, d_b) = layer_norm_backward(
                &d_normed, &cache.pre_ln_f[pos], &self.ln_f.weight,
            );
            for i in 0..N_EMBD {
                grads.ln_f_weight[i] += d_w[i];
                grads.ln_f_bias[i] += d_b[i];
            }
            d_hidden.push(d_h);
        }

        // Backward through blocks in reverse
        for (block_idx, block) in self.blocks.iter().enumerate().rev() {
            let bc = &cache.block_caches[block_idx];
            let bg = &mut grads.blocks[block_idx];

            // Backward through residual 2: d_h1 = d_hidden, d_ffn_out = d_hidden
            let d_ffn_out = d_hidden.clone();
            let mut d_h1 = d_hidden.clone();

            // Backward through FFN
            for pos in 0..t {
                let d_normed_2 = match (&block.ffn, &mut bg.ffn) {
                    (FfnWeights::PerBand(w), FfnGrads::PerBand {
                        band_w, band_b, out_proj_w, out_proj_b
                    }) => {
                        // Backward through out_proj
                        let (d_bands, d_w, d_b) = linear_backward(
                            &d_ffn_out[pos], &bc.normed_2[pos], &w.out_proj.w,
                        );
                        // Note: d_bands is wrong here - we need the pre-projection input
                        // Let me fix: recompute per-band output
                        let mut bands_out = vec![0.0f32; N_EMBD];
                        for band in 0..N_BANDS {
                            let r_in = bc.normed_2[pos][band * 2];
                            let s_in = bc.normed_2[pos][band * 2 + 1];
                            let bw = &w.band_w[band];
                            let bb = &w.band_b[band];
                            bands_out[band * 2] = bw[0][0] * r_in + bw[1][0] * s_in + bb[0];
                            bands_out[band * 2 + 1] = bw[0][1] * r_in + bw[1][1] * s_in + bb[1];
                        }

                        let (d_bands_out, d_op_w, d_op_b) = linear_backward(
                            &d_ffn_out[pos], &bands_out, &w.out_proj.w,
                        );
                        for i in 0..N_EMBD {
                            for j in 0..N_EMBD { out_proj_w[i][j] += d_op_w[i][j]; }
                            out_proj_b[i] += d_op_b[i];
                        }

                        // Backward through per-band 2x2 matmul
                        let mut d_input = vec![0.0f32; N_EMBD];
                        for band in 0..N_BANDS {
                            let d_out_r = d_bands_out[band * 2];
                            let d_out_s = d_bands_out[band * 2 + 1];
                            let r_in = bc.normed_2[pos][band * 2];
                            let s_in = bc.normed_2[pos][band * 2 + 1];

                            // y = W @ x + b, where W is [2][2], x is [r,s]
                            // d_W[i][j] += d_y[j] * x[i]
                            band_w[band][0][0] += d_out_r * r_in;
                            band_w[band][0][1] += d_out_s * r_in;
                            band_w[band][1][0] += d_out_r * s_in;
                            band_w[band][1][1] += d_out_s * s_in;
                            band_b[band][0] += d_out_r;
                            band_b[band][1] += d_out_s;

                            // d_x = W^T @ d_y
                            d_input[band * 2] += w.band_w[band][0][0] * d_out_r
                                               + w.band_w[band][0][1] * d_out_s;
                            d_input[band * 2 + 1] += w.band_w[band][1][0] * d_out_r
                                                   + w.band_w[band][1][1] * d_out_s;
                        }
                        d_input
                    }
                    (FfnWeights::KerrMaestro(w), FfnGrads::KerrMaestro {
                        gamma_raw, omega, alpha, beta,
                        squeeze_w, squeeze_b, process_w, process_b,
                        out_proj_w, out_proj_b,
                    }) => {
                        // Recompute kerr and maestro outputs
                        let kerr_out = self.kerr_ode_forward(&w.kerr, &bc.normed_2[pos]);
                        let maestro_out = self.maestro_forward(&w.maestro, &bc.normed_2[pos]);
                        let mut combined = vec![0.0f32; N_EMBD];
                        for i in 0..N_EMBD { combined[i] = kerr_out[i] + maestro_out[i]; }

                        // Backward through out_proj
                        let (d_combined, d_op_w, d_op_b) = linear_backward(
                            &d_ffn_out[pos], &combined, &w.out_proj.w,
                        );
                        for i in 0..N_EMBD {
                            for j in 0..N_EMBD { out_proj_w[i][j] += d_op_w[i][j]; }
                            out_proj_b[i] += d_op_b[i];
                        }

                        // d_combined splits to kerr and maestro
                        let d_kerr = &d_combined;
                        let d_maestro = &d_combined;

                        // Backward through Kerr-ODE
                        let (d_kerr_input, d_gr, d_om, d_al, d_be) =
                            kerr_ode_backward(d_kerr, &bc.normed_2[pos], &w.kerr);
                        for k in 0..N_BANDS {
                            gamma_raw[k] += d_gr[k];
                            omega[k] += d_om[k];
                        }
                        *alpha += d_al;
                        *beta += d_be;

                        // Backward through Maestro
                        let (d_maestro_input, d_sq, d_pr) =
                            maestro_backward(d_maestro, &bc.normed_2[pos], &w.maestro);
                        for i in 0..MAESTRO_DIM {
                            for j in 0..N_EMBD { squeeze_w[i][j] += d_sq.d_w[i][j]; }
                            squeeze_b[i] += d_sq.d_b[i];
                        }
                        for i in 0..N_EMBD {
                            for j in 0..MAESTRO_DIM { process_w[i][j] += d_pr.d_w[i][j]; }
                            process_b[i] += d_pr.d_b[i];
                        }

                        // Combine input gradients
                        let mut d_input = vec![0.0f32; N_EMBD];
                        for i in 0..N_EMBD {
                            d_input[i] = d_kerr_input[i] + d_maestro_input[i];
                        }
                        d_input
                    }
                    _ => unreachable!(),
                };

                // Backward through LN2
                let (d_h1_from_ln2, d_w, d_b) = layer_norm_backward(
                    &d_normed_2, &bc.h1[pos], &block.ln_2.weight,
                );
                for i in 0..N_EMBD {
                    bg.ln_2_weight[i] += d_w[i];
                    bg.ln_2_bias[i] += d_b[i];
                    d_h1[pos][i] += d_h1_from_ln2[i];
                }
            }

            // Backward through attention (all positions)
            let mut d_attn_out = d_h1.clone();

            // Backward through c_proj for each position
            let mut d_pre_proj = Vec::with_capacity(t);
            for pos in 0..t {
                let (d_pp, d_w, d_b) = linear_backward(
                    &d_attn_out[pos], &bc.attn_cache.pre_proj[pos], &block.attn.c_proj.w,
                );
                for i in 0..N_EMBD {
                    for j in 0..N_EMBD { bg.attn_c_proj_w[i][j] += d_w[i][j]; }
                    bg.attn_c_proj_b[i] += d_b[i];
                }
                d_pre_proj.push(d_pp);
            }

            // Backward through attention scores
            let mut d_q = vec![vec![0.0f32; N_EMBD]; t];
            let mut d_k = vec![vec![0.0f32; N_EMBD]; t];
            let mut d_v = vec![vec![0.0f32; N_EMBD]; t];

            for pos in 0..t {
                attention_backward_single(
                    &d_pre_proj[pos],
                    &bc.attn_cache.q_all,
                    &bc.attn_cache.k_all,
                    &bc.attn_cache.v_all,
                    &bc.attn_cache.att_weights.iter()
                        .map(|h| h[pos].clone()).collect::<Vec<_>>(),
                    pos,
                    &mut d_q, &mut d_k, &mut d_v,
                );
            }

            // Backward through c_attn for each position
            let mut d_normed_1 = vec![vec![0.0f32; N_EMBD]; t];
            for pos in 0..t {
                // Combine q, k, v gradients into qkv
                let mut d_qkv = vec![0.0f32; 3 * N_EMBD];
                for i in 0..N_EMBD {
                    d_qkv[i] = d_q[pos][i];
                    d_qkv[N_EMBD + i] = d_k[pos][i];
                    d_qkv[2 * N_EMBD + i] = d_v[pos][i];
                }

                let (d_x, d_w, d_b) = linear_backward(
                    &d_qkv, &bc.attn_cache.input[pos], &block.attn.c_attn.w,
                );
                for i in 0..3 * N_EMBD {
                    for j in 0..N_EMBD { bg.attn_c_attn_w[i][j] += d_w[i][j]; }
                    bg.attn_c_attn_b[i] += d_b[i];
                }
                d_normed_1[pos] = d_x;
            }

            // Backward through LN1
            d_hidden = Vec::with_capacity(t);
            for pos in 0..t {
                let (d_input, d_w, d_b) = layer_norm_backward(
                    &d_normed_1[pos], &bc.input[pos], &block.ln_1.weight,
                );
                for i in 0..N_EMBD {
                    bg.ln_1_weight[i] += d_w[i];
                    bg.ln_1_bias[i] += d_b[i];
                }
                // Add residual gradient
                let mut d_h = d_input;
                for i in 0..N_EMBD { d_h[i] += d_h1[pos][i]; }
                d_hidden.push(d_h);
            }
        }

        (total_loss, grads)
    }
}

// ─── Forward cache structures ───────────────────────────────────

pub struct ForwardCache {
    pub tokens: Vec<usize>,
    pub block_caches: Vec<BlockCache>,
    pub pre_ln_f: Vec<Vec<f32>>,
    pub post_ln_f: Vec<Vec<f32>>,
    pub logits: Vec<Vec<f32>>,
}

pub struct BlockCache {
    pub input: Vec<Vec<f32>>,
    pub normed_1: Vec<Vec<f32>>,
    pub attn_cache: AttnCache,
    pub h1: Vec<Vec<f32>>,
    pub normed_2: Vec<Vec<f32>>,
    pub ffn_input: Vec<Vec<f32>>,
    pub output: Vec<Vec<f32>>,
}

pub struct AttnCache {
    pub input: Vec<Vec<f32>>,
    pub q_all: Vec<Vec<f32>>,
    pub k_all: Vec<Vec<f32>>,
    pub v_all: Vec<Vec<f32>>,
    pub att_weights: Vec<Vec<Vec<f32>>>,  // [N_HEAD][T][T]
    pub pre_proj: Vec<Vec<f32>>,
}

// ─── Helper wrappers ────────────────────────────────────────────

fn layer_norm_fn(x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let std = (var + 1e-5).sqrt();
    (0..n).map(|i| (x[i] - mean) / std * weight[i] + bias[i]).collect()
}

fn linear_no_bias_fn(w: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    w.iter().map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum()).collect()
}

// ─── Simple RNG (xorshift64) ──────────────────────────────────

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Uniform float in [0, 1)
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform float in [-limit, +limit]
    fn uniform(&mut self, limit: f32) -> f32 {
        (self.next_f32() * 2.0 - 1.0) * limit
    }

    /// Random index in [0, n)
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ─── Character-level dataset ──────────────────────────────────

pub struct Dataset {
    pub data: Vec<usize>,
    pub vocab_size: usize,
    pub char_to_idx: HashMap<char, usize>,
    pub idx_to_char: Vec<char>,
}

impl Dataset {
    pub fn from_file(path: &str) -> Self {
        let text = fs::read_to_string(path).expect("Failed to read training data");

        // Build vocabulary from unique chars, sorted for determinism
        let mut chars: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>()
            .into_iter().collect();
        chars.sort();

        let char_to_idx: HashMap<char, usize> = chars.iter().enumerate()
            .map(|(i, &c)| (c, i)).collect();
        let vocab_size = chars.len();

        let data: Vec<usize> = text.chars()
            .map(|c| char_to_idx[&c])
            .collect();

        println!("  Dataset: {} chars, vocab_size={}", data.len(), vocab_size);

        Self { data, vocab_size, char_to_idx, idx_to_char: chars }
    }

    /// Sample a random batch of (input, target) sequences.
    /// Returns (inputs, targets) where each is [batch_size][seq_len].
    fn sample_batch(&self, rng: &mut Rng, batch_size: usize, seq_len: usize)
        -> (Vec<Vec<usize>>, Vec<Vec<usize>>)
    {
        let max_start = self.data.len() - seq_len - 1;
        let mut inputs = Vec::with_capacity(batch_size);
        let mut targets = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let start = rng.next_usize(max_start);
            inputs.push(self.data[start..start + seq_len].to_vec());
            targets.push(self.data[start + 1..start + seq_len + 1].to_vec());
        }

        (inputs, targets)
    }
}

// ─── Weight initialization ────────────────────────────────────

fn init_linear(rng: &mut Rng, out_dim: usize, in_dim: usize) -> LinearWeights {
    let limit = 1.0 / (in_dim as f32).sqrt();
    let w: Vec<Vec<f32>> = (0..out_dim)
        .map(|_| (0..in_dim).map(|_| rng.uniform(limit)).collect())
        .collect();
    let b = vec![0.0f32; out_dim];
    LinearWeights { w, b }
}

fn init_layer_norm() -> LayerNormWeights {
    LayerNormWeights {
        weight: vec![1.0f32; N_EMBD],
        bias: vec![0.0f32; N_EMBD],
    }
}

fn init_attention(rng: &mut Rng) -> AttentionWeights {
    AttentionWeights {
        c_attn: init_linear(rng, 3 * N_EMBD, N_EMBD),
        c_proj: init_linear(rng, N_EMBD, N_EMBD),
    }
}

fn init_per_band_linear(rng: &mut Rng) -> PerBandLinearWeights {
    let limit = 1.0 / 2.0f32.sqrt();
    let band_w: Vec<[[f32; 2]; 2]> = (0..N_BANDS)
        .map(|_| [
            [rng.uniform(limit), rng.uniform(limit)],
            [rng.uniform(limit), rng.uniform(limit)],
        ])
        .collect();
    let band_b = vec![[0.0f32; 2]; N_BANDS];

    PerBandLinearWeights {
        band_w,
        band_b,
        out_proj: init_linear(rng, N_EMBD, N_EMBD),
    }
}

fn init_kerr_weights() -> KerrWeights {
    let gamma_raw_val = ((0.1f32).exp() - 1.0).ln(); // softplus^-1(0.1)
    KerrWeights {
        gamma_raw: vec![gamma_raw_val; N_BANDS],
        omega: (0..N_BANDS).map(|k| (k + 1) as f32 / N_BANDS as f32).collect(),
        alpha: 0.1,
        beta: 0.1,
    }
}

fn init_maestro_weights(rng: &mut Rng) -> MaestroWeights {
    MaestroWeights {
        squeeze: init_linear(rng, MAESTRO_DIM, N_EMBD),
        process_1: init_linear(rng, N_EMBD, MAESTRO_DIM),
    }
}

fn init_kerr_maestro_add(rng: &mut Rng) -> KerrMaestroAddWeights {
    KerrMaestroAddWeights {
        kerr: init_kerr_weights(),
        maestro: init_maestro_weights(rng),
        out_proj: init_linear(rng, N_EMBD, N_EMBD),
    }
}

pub fn init_model(vocab_size: usize, seed: u64) -> ModelWeights {
    let mut rng = Rng::new(seed);

    // Block 0: PerBandLinear
    let block0 = BlockWeights {
        ln_1: init_layer_norm(),
        attn: init_attention(&mut rng),
        ln_2: init_layer_norm(),
        ffn: FfnWeights::PerBand(init_per_band_linear(&mut rng)),
    };

    // Blocks 1-3: KerrMaestroAdd
    let mut blocks = vec![block0];
    for _ in 0..3 {
        blocks.push(BlockWeights {
            ln_1: init_layer_norm(),
            attn: init_attention(&mut rng),
            ln_2: init_layer_norm(),
            ffn: FfnWeights::KerrMaestro(init_kerr_maestro_add(&mut rng)),
        });
    }

    // LM head
    let limit = 1.0 / (N_EMBD as f32).sqrt();
    let lm_head: Vec<Vec<f32>> = (0..vocab_size)
        .map(|_| (0..N_EMBD).map(|_| rng.uniform(limit)).collect())
        .collect();

    ModelWeights {
        vocab_size,
        wte_phase: build_harmonic_table(vocab_size),
        wpe: build_positional_table(),
        blocks,
        ln_f: init_layer_norm(),
        lm_head,
    }
}

// ─── Parameter flattening for Adam ────────────────────────────

fn count_params(model: &ModelWeights) -> usize {
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

/// Flatten all trainable parameters into a single Vec.
fn flatten_params(model: &ModelWeights) -> Vec<f32> {
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
fn flatten_grads(grads: &GradAccum) -> Vec<f32> {
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
fn unflatten_params(model: &mut ModelWeights, params: &[f32]) {
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

// ─── Gradient clipping ────────────────────────────────────────

fn clip_grad_norm(grads: &mut [f32], max_norm: f32) {
    let norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grads.iter_mut() { *g *= scale; }
    }
}

// ─── Training loop ────────────────────────────────────────────

pub fn train(data_path: &str, n_iters: usize, batch_size: usize, seq_len: usize, lr: f32) {
    println!("Stage 4 — training from scratch\n");

    // Load dataset
    println!("Loading dataset from {data_path}...");
    let dataset = Dataset::from_file(data_path);

    // Init model
    println!("Initializing model...");
    let mut model = init_model(dataset.vocab_size, 42);
    let n_params = count_params(&model);
    println!("  Trainable parameters: {}", n_params);

    // Init Adam
    let mut optimizer = Adam::new(lr, n_params);

    // Training
    println!("\nTraining for {n_iters} iterations (batch_size={batch_size}, seq_len={seq_len}, lr={lr})");
    println!("{:>6} {:>10} {:>10}", "Iter", "Loss", "Time");
    println!("{}", "-".repeat(30));

    let mut rng = Rng::new(1337);
    let log_every = 50;
    let eval_every = 300;

    let train_start = std::time::Instant::now();

    for iter in 0..n_iters {
        let iter_start = std::time::Instant::now();

        // Sample batch — process one sequence at a time (no batched matmul)
        let (inputs, targets) = dataset.sample_batch(&mut rng, batch_size, seq_len);

        let mut total_loss = 0.0f32;
        let mut accumulated_grads: Option<Vec<f32>> = None;

        for b in 0..batch_size {
            let cache = model.forward_with_cache(&inputs[b]);
            let (loss, grads) = model.backward(&cache, &targets[b]);
            total_loss += loss;

            let flat_grads = flatten_grads(&grads);
            match &mut accumulated_grads {
                None => accumulated_grads = Some(flat_grads),
                Some(acc) => {
                    for (a, g) in acc.iter_mut().zip(flat_grads.iter()) {
                        *a += g;
                    }
                }
            }
        }

        // Average over batch
        total_loss /= batch_size as f32;
        let mut grads = accumulated_grads.unwrap();
        for g in grads.iter_mut() { *g /= batch_size as f32; }

        // Gradient clipping
        clip_grad_norm(&mut grads, 1.0);

        // Adam step
        let mut params = flatten_params(&model);
        optimizer.step(&mut params, &grads);
        unflatten_params(&mut model, &params);

        let iter_time = iter_start.elapsed();

        if iter % log_every == 0 || iter == n_iters - 1 {
            println!("{:>6} {:>10.4} {:>10.1?}", iter, total_loss, iter_time);
        }

        // Eval: generate sample text
        if iter > 0 && iter % eval_every == 0 {
            let sample = generate(&model, &dataset, 200, &mut rng);
            println!("\n--- Sample (iter {iter}) ---");
            println!("{sample}");
            println!("---\n");
        }
    }

    let total_time = train_start.elapsed();
    println!("\nTraining complete. Total time: {total_time:.1?}");

    // Final generation
    let sample = generate(&model, &dataset, 500, &mut rng);
    println!("\n=== Final sample ===");
    println!("{sample}");
    println!("===");
}

/// Generate text by sampling from the model.
fn generate(model: &ModelWeights, dataset: &Dataset, n_tokens: usize, rng: &mut Rng) -> String {
    // Start with newline
    let newline_idx = dataset.char_to_idx.get(&'\n').copied().unwrap_or(0);
    let mut tokens = vec![newline_idx];

    for _ in 0..n_tokens {
        // Use last BLOCK_SIZE tokens
        let start = if tokens.len() > BLOCK_SIZE { tokens.len() - BLOCK_SIZE } else { 0 };
        let context = &tokens[start..];

        let logits_all = model.forward(context);
        let logits = logits_all.last().unwrap();

        // Temperature sampling (temperature = 0.8)
        let temp = 0.8f32;
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_l: Vec<f32> = logits.iter().map(|&l| ((l - max_l) / temp).exp()).collect();
        let sum_exp: f32 = exp_l.iter().sum();
        let probs: Vec<f32> = exp_l.iter().map(|e| e / sum_exp).collect();

        // Sample from distribution
        let mut r = rng.next_f32();
        let mut chosen = probs.len() - 1;
        for (i, &p) in probs.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                chosen = i;
                break;
            }
        }

        tokens.push(chosen);
    }

    tokens.iter()
        .skip(1) // skip initial newline
        .map(|&t| {
            if t < dataset.idx_to_char.len() {
                dataset.idx_to_char[t]
            } else {
                '?'
            }
        })
        .collect()
}
