//! Training pipeline — cached forward pass + backward chain.
//!
//! What the forward saves, the backward consumes. These two halves
//! share a contract (the cache structs) and change together.
//! Separate from the training loop (train.rs) and the gradient
//! primitives (backward.rs).

use crate::model::*;
use crate::backward::*;
use crate::backend::ComputeBackend;

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
        let n_embd = weights.config.n_embd();
        let n_bands = weights.config.n_bands;
        let maestro_dim = weights.config.maestro_dim;
        let mut blocks = Vec::new();
        for block in &weights.blocks {
            let ffn = match &block.ffn {
                FfnWeights::PerBand(_) => FfnGrads::PerBand {
                    band_w: vec![[[0.0; 2]; 2]; n_bands],
                    band_b: vec![[0.0; 2]; n_bands],
                    out_proj_w: vec![vec![0.0; n_embd]; n_embd],
                    out_proj_b: vec![0.0; n_embd],
                },
                FfnWeights::KerrMaestro(_) => FfnGrads::KerrMaestro {
                    gamma_raw: vec![0.0; n_bands],
                    omega: vec![0.0; n_bands],
                    alpha: 0.0,
                    beta: 0.0,
                    squeeze_w: vec![vec![0.0; n_embd]; maestro_dim],
                    squeeze_b: vec![0.0; maestro_dim],
                    process_w: vec![vec![0.0; maestro_dim]; n_embd],
                    process_b: vec![0.0; n_embd],
                    out_proj_w: vec![vec![0.0; n_embd]; n_embd],
                    out_proj_b: vec![0.0; n_embd],
                },
            };
            blocks.push(BlockGrads {
                ln_1_weight: vec![0.0; n_embd],
                ln_1_bias: vec![0.0; n_embd],
                attn_c_attn_w: vec![vec![0.0; n_embd]; 3 * n_embd],
                attn_c_attn_b: vec![0.0; 3 * n_embd],
                attn_c_proj_w: vec![vec![0.0; n_embd]; n_embd],
                attn_c_proj_b: vec![0.0; n_embd],
                ln_2_weight: vec![0.0; n_embd],
                ln_2_bias: vec![0.0; n_embd],
                ffn,
            });
        }
        Self {
            blocks,
            ln_f_weight: vec![0.0; n_embd],
            ln_f_bias: vec![0.0; n_embd],
            lm_head: vec![vec![0.0; n_embd]; weights.vocab_size],
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

// ─── Cached forward pass ────────────────────────────────────────

impl ModelWeights {
    /// Forward pass with saved activations for backward.
    #[allow(dead_code)]
    pub fn forward_with_cache(&self, tokens: &[usize], backend: &dyn ComputeBackend) -> ForwardCache {
        self.forward_with_cache_curriculum(tokens, self.config.n_bands, backend)
    }

    /// Forward pass with curriculum band masking.
    /// Bands beyond `active_bands` are zeroed after embedding.
    pub fn forward_with_cache_curriculum(&self, tokens: &[usize], active_bands: usize, backend: &dyn ComputeBackend) -> ForwardCache {
        let t = tokens.len();
        let n_embd = self.config.n_embd();
        let n_bands = self.config.n_bands;

        // Embedding + positional
        let mut hidden: Vec<Vec<f32>> = Vec::with_capacity(t);
        for (pos, &tok) in tokens.iter().enumerate() {
            let mut h = vec![0.0f32; n_embd];
            for i in 0..n_embd {
                h[i] = self.wte_phase[tok][i] + self.wpe[pos][i];
            }
            // Curriculum: zero inactive bands
            if active_bands < n_bands {
                for i in (active_bands * 2)..n_embd {
                    h[i] = 0.0;
                }
            }
            hidden.push(h);
        }

        let mut block_caches = Vec::new();

        for block in &self.blocks {
            let mut cache = self.forward_block_with_cache(block, &hidden, backend);
            // Take output instead of clone — backward never reads bc.output
            hidden = std::mem::take(&mut cache.output);
            block_caches.push(cache);
        }

        // Final layer norm (batched)
        let normed = backend.layer_norm_batch(&hidden, &self.ln_f.weight, &self.ln_f.bias);

        // LM head (batched)
        let logits = backend.linear_no_bias_batch(&self.lm_head, &normed);

        ForwardCache {
            tokens: tokens.to_vec(),
            block_caches,
            pre_ln_f: hidden, // move, not clone — hidden not used after this
            post_ln_f: normed,
            logits,
        }
    }

    fn forward_block_with_cache(&self, block: &BlockWeights, hidden: &[Vec<f32>], backend: &dyn ComputeBackend) -> BlockCache {
        let t = hidden.len();
        let n_embd = hidden[0].len();

        let normed_1 = backend.layer_norm_batch(hidden, &block.ln_1.weight, &block.ln_1.bias);

        let (attn_out, attn_cache) = self.attention_with_cache(&block.attn, &normed_1, backend);

        let h1: Vec<Vec<f32>> = (0..t).map(|i| {
            let mut v = vec![0.0f32; n_embd];
            for j in 0..n_embd { v[j] = hidden[i][j] + attn_out[i][j]; }
            v
        }).collect();

        let normed_2 = backend.layer_norm_batch(&h1, &block.ln_2.weight, &block.ln_2.bias);

        let ffn_out = match &block.ffn {
            FfnWeights::PerBand(w) => backend.per_band_linear(w, &normed_2),
            FfnWeights::KerrMaestro(w) => backend.kerr_maestro_add(w, &normed_2),
        };

        let output: Vec<Vec<f32>> = (0..t).map(|i| {
            let mut v = vec![0.0f32; n_embd];
            for j in 0..n_embd { v[j] = h1[i][j] + ffn_out[i][j]; }
            v
        }).collect();

        BlockCache {
            input: hidden.to_vec(),
            normed_1,
            attn_cache,
            h1,
            normed_2,
            output,
        }
    }

    fn attention_with_cache(&self, weights: &AttentionWeights, x: &[Vec<f32>], backend: &dyn ComputeBackend) -> (Vec<Vec<f32>>, AttnCache) {
        let t = x.len();
        let n_embd = x[0].len();
        let n_head = weights.n_head;
        let head_dim = n_embd / n_head;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Batched QKV projection: 1 call instead of T separate calls
        let qkv_all = backend.linear_batch(&weights.c_attn.w, &weights.c_attn.b, x);

        let mut q_all = vec![vec![0.0f32; n_embd]; t];
        let mut k_all = vec![vec![0.0f32; n_embd]; t];
        let mut v_all = vec![vec![0.0f32; n_embd]; t];

        for pos in 0..t {
            let qkv = &qkv_all[pos];
            for i in 0..n_embd {
                q_all[pos][i] = qkv[i];
                k_all[pos][i] = qkv[n_embd + i];
                v_all[pos][i] = qkv[2 * n_embd + i];
            }
        }

        let mut att_weights_all = vec![vec![vec![0.0f32; t]; t]; n_head];
        let mut out = vec![vec![0.0f32; n_embd]; t];

        // Pre-allocate attention scratch — reused across all heads and positions
        let mut att = vec![0.0f32; t];

        for head in 0..n_head {
            let offset = head * head_dim;
            for qi in 0..t {
                // Reset to -inf for causal masking
                for ki in 0..t { att[ki] = f32::NEG_INFINITY; }

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

                // Copy to cache (direct copy, not clone — att is reused)
                att_weights_all[head][qi][..t].copy_from_slice(&att);

                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for ki in 0..=qi {
                        sum += att[ki] * v_all[ki][offset + d];
                    }
                    out[qi][offset + d] = sum;
                }
            }
        }

        // Batched output projection: 1 call instead of T separate calls
        let result = backend.linear_batch(&weights.c_proj.w, &weights.c_proj.b, &out);
        let pre_proj = out; // move, not clone — out not used after this

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
    pub fn backward(&self, cache: &ForwardCache, targets: &[usize], backend: &dyn ComputeBackend) -> (f32, GradAccum) {
        // Timing instrumentation — set to true for backward profiling
        const TIMING: bool = false;
        let _bwd_start = std::time::Instant::now();
        let t = cache.tokens.len();
        let n_embd = self.config.n_embd();
        let n_bands = self.config.n_bands;
        let maestro_dim = self.config.maestro_dim;
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
        let _t_lm = std::time::Instant::now();
        // Batched: all positions in one GPU dispatch
        let d_normed_all = backend.linear_backward_dx_batch(&d_logits, &self.lm_head);
        let mut d_hidden: Vec<Vec<f32>> = Vec::with_capacity(t);
        for pos in 0..t {
            let (d_h, d_w, d_b) = backend.layer_norm_backward(
                &d_normed_all[pos], &cache.pre_ln_f[pos], &self.ln_f.weight,
            );
            for i in 0..n_embd {
                grads.ln_f_weight[i] += d_w[i];
                grads.ln_f_bias[i] += d_b[i];
            }
            d_hidden.push(d_h);
        }
        // d_lm_head via batched outer product on GPU (no bias)
        let (lm_dw, _) = backend.outer_product_accum(
            &d_logits, &cache.post_ln_f, false,
        );
        for i in 0..self.vocab_size {
            for j in 0..n_embd {
                grads.lm_head[i][j] += lm_dw[i][j];
            }
        }

        if TIMING { eprintln!("  lm_head bwd:     {:?}", _t_lm.elapsed()); }

        // Backward through blocks in reverse
        for (block_idx, block) in self.blocks.iter().enumerate().rev() {
            let bc = &cache.block_caches[block_idx];
            let bg = &mut grads.blocks[block_idx];

            let d_ffn_out = d_hidden.clone();
            let mut d_h1 = d_hidden.clone();

            let _t_ffn = std::time::Instant::now();
            // Backward through FFN — batched across all positions
            let d_normed_2_all: Vec<Vec<f32>> = match (&block.ffn, &mut bg.ffn) {
                (FfnWeights::PerBand(w), FfnGrads::PerBand {
                    band_w, band_b, out_proj_w, out_proj_b
                }) => {
                    // Forward recompute per-band transform for all positions
                    let bands_out_all: Vec<Vec<f32>> = (0..t).map(|pos| {
                        let mut bands_out = vec![0.0f32; n_embd];
                        for band in 0..n_bands {
                            let r_in = bc.normed_2[pos][band * 2];
                            let s_in = bc.normed_2[pos][band * 2 + 1];
                            let bw = &w.band_w[band];
                            let bb = &w.band_b[band];
                            bands_out[band * 2] = bw[0][0] * r_in + bw[1][0] * s_in + bb[0];
                            bands_out[band * 2 + 1] = bw[0][1] * r_in + bw[1][1] * s_in + bb[1];
                        }
                        bands_out
                    }).collect();
                    // Batched out_proj backward: all positions in one dispatch
                    let d_bands_out_all = backend.linear_backward_dx_batch(&d_ffn_out, &w.out_proj.w);
                    let (op_dw, op_db) = backend.outer_product_accum(&d_ffn_out, &bands_out_all, true);
                    for i in 0..n_embd {
                        for j in 0..n_embd { out_proj_w[i][j] += op_dw[i][j]; }
                        out_proj_b[i] += op_db[i];
                    }
                    // Per-band backward (trivial per-band math, not worth GPU)
                    (0..t).map(|pos| {
                        let mut d_input = vec![0.0f32; n_embd];
                        for band in 0..n_bands {
                            let d_out_r = d_bands_out_all[pos][band * 2];
                            let d_out_s = d_bands_out_all[pos][band * 2 + 1];
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
                    }).collect()
                }
                (FfnWeights::KerrMaestro(w), FfnGrads::KerrMaestro {
                    gamma_raw, omega, alpha, beta,
                    squeeze_w, squeeze_b, process_w, process_b,
                    out_proj_w, out_proj_b,
                }) => {
                    // Phase 1: Forward recompute all positions
                    let combined_all: Vec<Vec<f32>> = (0..t).map(|pos| {
                        let kerr_out = self.kerr_ode_forward(&w.kerr, &bc.normed_2[pos]);
                        let maestro_out = self.maestro_forward(&w.maestro, &bc.normed_2[pos]);
                        let mut combined = vec![0.0f32; n_embd];
                        for i in 0..n_embd { combined[i] = kerr_out[i] + maestro_out[i]; }
                        combined
                    }).collect();
                    // Phase 2: Batched out_proj backward
                    let d_combined_all = backend.linear_backward_dx_batch(&d_ffn_out, &w.out_proj.w);
                    let (op_dw, op_db) = backend.outer_product_accum(&d_ffn_out, &combined_all, true);
                    for i in 0..n_embd {
                        for j in 0..n_embd { out_proj_w[i][j] += op_dw[i][j]; }
                        out_proj_b[i] += op_db[i];
                    }
                    // Phase 3: Kerr-ODE backward per position (accumulate shared grads)
                    let mut d_input_all = vec![vec![0.0f32; n_embd]; t];
                    for pos in 0..t {
                        let (d_kerr_input, d_gr, d_om, d_al, d_be) =
                            kerr_ode_backward(&d_combined_all[pos], &bc.normed_2[pos], &w.kerr);
                        for k in 0..n_bands {
                            gamma_raw[k] += d_gr[k];
                            omega[k] += d_om[k];
                        }
                        *alpha += d_al;
                        *beta += d_be;
                        for i in 0..n_embd { d_input_all[pos][i] += d_kerr_input[i]; }
                    }
                    // Phase 4: Batched maestro backward
                    // Forward recompute squeeze + GELU for all positions
                    let squeezed_all: Vec<Vec<f32>> = (0..t).map(|pos| {
                        linear_fn(&w.maestro.squeeze.w, &w.maestro.squeeze.b, &bc.normed_2[pos])
                    }).collect();
                    let activated_all: Vec<Vec<f32>> = squeezed_all.iter().map(|sq| {
                        sq.iter().map(|&v| crate::model::gelu(v)).collect()
                    }).collect();
                    // Backward through process linear (batched)
                    let d_activated_all = backend.linear_backward_dx_batch(
                        &d_combined_all, &w.maestro.process_1.w,
                    );
                    let (pr_dw, pr_db) = backend.outer_product_accum(
                        &d_combined_all, &activated_all, true,
                    );
                    for i in 0..n_embd {
                        for j in 0..maestro_dim { process_w[i][j] += pr_dw[i][j]; }
                        process_b[i] += pr_db[i];
                    }
                    // GELU backward per position (maestro_dim elements, trivial)
                    let d_squeezed_all: Vec<Vec<f32>> = (0..t).map(|pos| {
                        backend.gelu_backward(&d_activated_all[pos], &squeezed_all[pos])
                    }).collect();
                    // Backward through squeeze linear (batched)
                    let d_maestro_inputs = backend.linear_backward_dx_batch(
                        &d_squeezed_all, &w.maestro.squeeze.w,
                    );
                    let (sq_dw, sq_db) = backend.outer_product_accum(
                        &d_squeezed_all, &bc.normed_2, true,
                    );
                    for i in 0..maestro_dim {
                        for j in 0..n_embd { squeeze_w[i][j] += sq_dw[i][j]; }
                        squeeze_b[i] += sq_db[i];
                    }
                    // Combine kerr + maestro input grads
                    for pos in 0..t {
                        for i in 0..n_embd {
                            d_input_all[pos][i] += d_maestro_inputs[pos][i];
                        }
                    }
                    d_input_all
                }
                _ => unreachable!(),
            };

            // Layer norm backward for all positions
            for pos in 0..t {
                let (d_h1_from_ln2, d_w, d_b) = backend.layer_norm_backward(
                    &d_normed_2_all[pos], &bc.h1[pos], &block.ln_2.weight,
                );
                for i in 0..n_embd {
                    bg.ln_2_weight[i] += d_w[i];
                    bg.ln_2_bias[i] += d_b[i];
                    d_h1[pos][i] += d_h1_from_ln2[i];
                }
            }

            if TIMING && block_idx == 0 { eprintln!("    ffn+ln2 bwd:   {:?}", _t_ffn.elapsed()); }

            // Backward through attention
            let d_attn_out = d_h1.clone();

            let _t_cproj = std::time::Instant::now();
            // Batched: all positions in one GPU dispatch
            let d_pre_proj = backend.linear_backward_dx_batch(&d_attn_out, &block.attn.c_proj.w);
            let (cp_dw, cp_db) = backend.outer_product_accum(
                &d_attn_out, &bc.attn_cache.pre_proj, true,
            );
            for i in 0..n_embd {
                for j in 0..n_embd { bg.attn_c_proj_w[i][j] += cp_dw[i][j]; }
                bg.attn_c_proj_b[i] += cp_db[i];
            }
            if TIMING && block_idx == 0 { eprintln!("    c_proj bwd:    {:?}", _t_cproj.elapsed()); }

            let _t_attn = std::time::Instant::now();
            let n_head = bc.attn_cache.att_weights.len();
            let (d_q, d_k, d_v) = backend.attention_backward(
                &d_pre_proj,
                &bc.attn_cache.q_all,
                &bc.attn_cache.k_all,
                &bc.attn_cache.v_all,
                &bc.attn_cache.att_weights,
                n_head,
            );
            if TIMING && block_idx == 0 { eprintln!("    attn bwd:      {:?}", _t_attn.elapsed()); }

            let _t_cattn = std::time::Instant::now();
            // Concatenate d_q, d_k, d_v into d_qkv[pos][3*n_embd]
            let mut d_qkv_all = Vec::with_capacity(t);
            for pos in 0..t {
                let mut d_qkv = vec![0.0f32; 3 * n_embd];
                for i in 0..n_embd {
                    d_qkv[i] = d_q[pos][i];
                    d_qkv[n_embd + i] = d_k[pos][i];
                    d_qkv[2 * n_embd + i] = d_v[pos][i];
                }
                d_qkv_all.push(d_qkv);
            }
            // Batched: all positions in one GPU dispatch
            let d_normed_1 = backend.linear_backward_dx_batch(&d_qkv_all, &block.attn.c_attn.w);
            let (ca_dw, ca_db) = backend.outer_product_accum(
                &d_qkv_all, &bc.attn_cache.input, true,
            );
            for i in 0..3 * n_embd {
                for j in 0..n_embd { bg.attn_c_attn_w[i][j] += ca_dw[i][j]; }
                bg.attn_c_attn_b[i] += ca_db[i];
            }
            if TIMING && block_idx == 0 { eprintln!("    c_attn bwd:    {:?}", _t_cattn.elapsed()); }

            // Backward through LN1
            d_hidden = Vec::with_capacity(t);
            for pos in 0..t {
                let (d_input, d_w, d_b) = backend.layer_norm_backward(
                    &d_normed_1[pos], &bc.input[pos], &block.ln_1.weight,
                );
                for i in 0..n_embd {
                    bg.ln_1_weight[i] += d_w[i];
                    bg.ln_1_bias[i] += d_b[i];
                }
                let mut d_h = d_input;
                for i in 0..n_embd { d_h[i] += d_h1[pos][i]; }
                d_hidden.push(d_h);
            }
        }

        if TIMING { eprintln!("  TOTAL bwd:       {:?}", _bwd_start.elapsed()); }
        (total_loss, grads)
    }
}
