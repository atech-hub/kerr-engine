//! Training pipeline — cached forward pass + backward chain.
//!
//! What the forward saves, the backward consumes. These two halves
//! share a contract (the cache structs) and change together.
//! Separate from the training loop (train.rs) and the gradient
//! primitives (backward.rs).

use crate::model::*;
use crate::backward::*;

// ─── Gradient accumulator ───────────────────────────────────────

/// Flat storage for all trainable parameter gradients.
pub struct GradAccum {
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
                FfnWeights::PerBand(_) => FfnGrads::PerBand {
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

// ─── Forward cache structures ───────────────────────────────────

pub struct ForwardCache {
    pub tokens: Vec<usize>,
    pub block_caches: Vec<BlockCache>,
    pub pre_ln_f: Vec<Vec<f32>>,
    pub post_ln_f: Vec<Vec<f32>>,
    pub logits: Vec<Vec<f32>>,
}

#[allow(dead_code)]
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

// ─── Cached forward pass ────────────────────────────────────────

impl ModelWeights {
    /// Forward pass with saved activations for backward.
    #[allow(dead_code)]
    pub fn forward_with_cache(&self, tokens: &[usize]) -> ForwardCache {
        self.forward_with_cache_curriculum(tokens, N_BANDS)
    }

    /// Forward pass with curriculum band masking.
    /// Bands beyond `active_bands` are zeroed after embedding.
    pub fn forward_with_cache_curriculum(&self, tokens: &[usize], active_bands: usize) -> ForwardCache {
        let t = tokens.len();

        // Embedding + positional
        let mut hidden: Vec<Vec<f32>> = Vec::with_capacity(t);
        for (pos, &tok) in tokens.iter().enumerate() {
            let mut h = vec![0.0f32; N_EMBD];
            for i in 0..N_EMBD {
                h[i] = self.wte_phase[tok][i] + self.wpe[pos][i];
            }
            // Curriculum: zero inactive bands
            if active_bands < N_BANDS {
                for i in (active_bands * 2)..N_EMBD {
                    h[i] = 0.0;
                }
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

        let normed_1: Vec<Vec<f32>> = hidden.iter()
            .map(|h| layer_norm_fn(h, &block.ln_1.weight, &block.ln_1.bias))
            .collect();

        let (attn_out, attn_cache) = self.attention_with_cache(&block.attn, &normed_1);

        let h1: Vec<Vec<f32>> = (0..t).map(|i| {
            let mut v = vec![0.0f32; N_EMBD];
            for j in 0..N_EMBD { v[j] = hidden[i][j] + attn_out[i][j]; }
            v
        }).collect();

        let normed_2: Vec<Vec<f32>> = h1.iter()
            .map(|h| layer_norm_fn(h, &block.ln_2.weight, &block.ln_2.bias))
            .collect();

        let ffn_out = match &block.ffn {
            FfnWeights::PerBand(w) => self.per_band_linear(w, &normed_2),
            FfnWeights::KerrMaestro(w) => self.kerr_maestro_add(w, &normed_2),
        };

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

    // ─── Backward chain ─────────────────────────────────────────

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

            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_l: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
            let sum_exp: f32 = exp_l.iter().sum();
            total_loss += -(exp_l[target] / sum_exp).ln();

            let mut dl = cross_entropy_backward(logits, target);
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
            for i in 0..self.vocab_size {
                for j in 0..N_EMBD {
                    grads.lm_head[i][j] += d_lm_head[i][j];
                }
            }

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

            let d_ffn_out = d_hidden.clone();
            let mut d_h1 = d_hidden.clone();

            // Backward through FFN
            for pos in 0..t {
                let d_normed_2 = match (&block.ffn, &mut bg.ffn) {
                    (FfnWeights::PerBand(w), FfnGrads::PerBand {
                        band_w, band_b, out_proj_w, out_proj_b
                    }) => {
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

                        let mut d_input = vec![0.0f32; N_EMBD];
                        for band in 0..N_BANDS {
                            let d_out_r = d_bands_out[band * 2];
                            let d_out_s = d_bands_out[band * 2 + 1];
                            let r_in = bc.normed_2[pos][band * 2];
                            let s_in = bc.normed_2[pos][band * 2 + 1];

                            band_w[band][0][0] += d_out_r * r_in;
                            band_w[band][0][1] += d_out_s * r_in;
                            band_w[band][1][0] += d_out_r * s_in;
                            band_w[band][1][1] += d_out_s * s_in;
                            band_b[band][0] += d_out_r;
                            band_b[band][1] += d_out_s;

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
                        let kerr_out = self.kerr_ode_forward(&w.kerr, &bc.normed_2[pos]);
                        let maestro_out = self.maestro_forward(&w.maestro, &bc.normed_2[pos]);
                        let mut combined = vec![0.0f32; N_EMBD];
                        for i in 0..N_EMBD { combined[i] = kerr_out[i] + maestro_out[i]; }

                        let (d_combined, d_op_w, d_op_b) = linear_backward(
                            &d_ffn_out[pos], &combined, &w.out_proj.w,
                        );
                        for i in 0..N_EMBD {
                            for j in 0..N_EMBD { out_proj_w[i][j] += d_op_w[i][j]; }
                            out_proj_b[i] += d_op_b[i];
                        }

                        let d_kerr = &d_combined;
                        let d_maestro = &d_combined;

                        let (d_kerr_input, d_gr, d_om, d_al, d_be) =
                            kerr_ode_backward(d_kerr, &bc.normed_2[pos], &w.kerr);
                        for k in 0..N_BANDS {
                            gamma_raw[k] += d_gr[k];
                            omega[k] += d_om[k];
                        }
                        *alpha += d_al;
                        *beta += d_be;

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

                        let mut d_input = vec![0.0f32; N_EMBD];
                        for i in 0..N_EMBD {
                            d_input[i] = d_kerr_input[i] + d_maestro_input[i];
                        }
                        d_input
                    }
                    _ => unreachable!(),
                };

                let (d_h1_from_ln2, d_w, d_b) = layer_norm_backward(
                    &d_normed_2, &bc.h1[pos], &block.ln_2.weight,
                );
                for i in 0..N_EMBD {
                    bg.ln_2_weight[i] += d_w[i];
                    bg.ln_2_bias[i] += d_b[i];
                    d_h1[pos][i] += d_h1_from_ln2[i];
                }
            }

            // Backward through attention
            let d_attn_out = d_h1.clone();

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

            let mut d_normed_1 = vec![vec![0.0f32; N_EMBD]; t];
            for pos in 0..t {
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
                let mut d_h = d_input;
                for i in 0..N_EMBD { d_h[i] += d_h1[pos][i]; }
                d_hidden.push(d_h);
            }
        }

        (total_loss, grads)
    }
}
