//! Full Kerr-ODE model — CPU reference implementation.
//!
//! Matches phaseC_integrated.py exactly for inference.
//! Architecture: 4 blocks, each with CausalSelfAttention + FFN.
//!   Block 0: Attention + PerBandLinear
//!   Blocks 1-3: Attention + KerrMaestroAdd (Kerr-ODE + Maestro)

use std::f32::consts::PI;

// Architecture constants (must match phaseC_integrated.py)
pub const N_BANDS: usize = 64;
pub const N_EMBD: usize = 128;  // = N_BANDS * 2
pub const N_HEAD: usize = 4;
pub const HEAD_DIM: usize = N_EMBD / N_HEAD;  // = 32
pub const BLOCK_SIZE: usize = 256;
pub const MAESTRO_DIM: usize = 16;
pub const RK4_N_STEPS: usize = 8;
pub const RK4_DT: f32 = 1.0 / RK4_N_STEPS as f32;  // = 0.125
pub const N_LAYERS: usize = 4;

/// Softplus: log(1 + exp(x))
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x  // avoid overflow
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// GELU activation (approximate version matching PyTorch default)
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// Build frozen harmonic embedding table.
/// Returns [vocab_size][N_EMBD] array.
pub fn build_harmonic_table(vocab_size: usize) -> Vec<Vec<f32>> {
    let nh = N_EMBD / 2;
    let scale = 1.0 / (nh as f32).sqrt();
    let mut table = vec![vec![0.0f32; N_EMBD]; vocab_size];

    for c in 0..vocab_size {
        let theta = c as f32 * 2.0 * PI / vocab_size as f32;
        for h in 0..nh {
            let angle = (h + 1) as f32 * theta;
            table[c][h * 2] = angle.cos() * scale;
            table[c][h * 2 + 1] = angle.sin() * scale;
        }
    }
    table
}

/// Build positional encoding table.
/// Returns [BLOCK_SIZE][N_EMBD] array.
pub fn build_positional_table() -> Vec<Vec<f32>> {
    let nh = N_EMBD / 2;
    let scale = 1.0 / (nh as f32).sqrt();
    let mut table = vec![vec![0.0f32; N_EMBD]; BLOCK_SIZE];

    for pos in 0..BLOCK_SIZE {
        for h in 0..nh {
            let freq = 1.0 / 10000.0_f32.powf(2.0 * h as f32 / N_EMBD as f32);
            table[pos][h * 2] = (pos as f32 * freq).cos() * scale;
            table[pos][h * 2 + 1] = (pos as f32 * freq).sin() * scale;
        }
    }
    table
}

// ─── Linear algebra helpers ──────────────────────────────────────

/// Matrix-vector multiply: y = W @ x + b
/// W is [out_dim][in_dim], x is [in_dim], b is [out_dim]
fn linear(w: &[Vec<f32>], b: &[f32], x: &[f32]) -> Vec<f32> {
    let out_dim = w.len();
    let mut y = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut sum = b[i];
        for j in 0..x.len() {
            sum += w[i][j] * x[j];
        }
        y[i] = sum;
    }
    y
}

/// Layer normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let std = (var + 1e-5).sqrt();

    let mut y = vec![0.0f32; n];
    for i in 0..n {
        y[i] = (x[i] - mean) / std * weight[i] + bias[i];
    }
    y
}

// ─── Weight structures ──────────────────────────────────────────

/// Weights for a Linear layer.
#[derive(Clone)]
pub struct LinearWeights {
    pub w: Vec<Vec<f32>>,  // [out_dim][in_dim]
    pub b: Vec<f32>,       // [out_dim]
}

/// Weights for LayerNorm.
#[derive(Clone)]
pub struct LayerNormWeights {
    pub weight: Vec<f32>,  // [dim]
    pub bias: Vec<f32>,    // [dim]
}

/// Weights for CausalSelfAttention.
#[derive(Clone)]
pub struct AttentionWeights {
    pub c_attn: LinearWeights,  // [3*N_EMBD, N_EMBD]
    pub c_proj: LinearWeights,  // [N_EMBD, N_EMBD]
}

/// Weights for PerBandLinear (Block 0 FFN).
#[derive(Clone)]
pub struct PerBandLinearWeights {
    pub band_w: Vec<[[f32; 2]; 2]>,  // [N_BANDS][2][2]
    pub band_b: Vec<[f32; 2]>,       // [N_BANDS][2]
    pub out_proj: LinearWeights,
}

/// Weights for Kerr-ODE layer.
#[derive(Clone)]
pub struct KerrWeights {
    pub gamma_raw: Vec<f32>,  // [N_BANDS] (before softplus)
    pub omega: Vec<f32>,      // [N_BANDS]
    pub alpha: f32,
    pub beta: f32,
}

/// Weights for Maestro.
#[derive(Clone)]
pub struct MaestroWeights {
    pub squeeze: LinearWeights,   // [MAESTRO_DIM, N_EMBD]
    pub process_1: LinearWeights, // [N_EMBD, MAESTRO_DIM]
}

/// Weights for KerrMaestroAdd block (Blocks 1-3 FFN).
#[derive(Clone)]
pub struct KerrMaestroAddWeights {
    pub kerr: KerrWeights,
    pub maestro: MaestroWeights,
    pub out_proj: LinearWeights,
}

/// Weights for one Block.
#[derive(Clone)]
pub struct BlockWeights {
    pub ln_1: LayerNormWeights,
    pub attn: AttentionWeights,
    pub ln_2: LayerNormWeights,
    pub ffn: FfnWeights,
}

/// FFN can be PerBandLinear (block 0) or KerrMaestroAdd (blocks 1-3).
#[derive(Clone)]
pub enum FfnWeights {
    PerBand(PerBandLinearWeights),
    KerrMaestro(KerrMaestroAddWeights),
}

/// Full model weights.
pub struct ModelWeights {
    pub vocab_size: usize,
    pub wte_phase: Vec<Vec<f32>>,  // [vocab_size][N_EMBD] (frozen)
    pub wpe: Vec<Vec<f32>>,        // [BLOCK_SIZE][N_EMBD] (frozen)
    pub blocks: Vec<BlockWeights>, // 4 blocks
    pub ln_f: LayerNormWeights,
    pub lm_head: Vec<Vec<f32>>,    // [vocab_size][N_EMBD] (no bias)
}

// ─── Forward pass ───────────────────────────────────────────────

impl ModelWeights {
    /// Full forward pass: token indices → logits.
    pub fn forward(&self, tokens: &[usize]) -> Vec<Vec<f32>> {
        let t = tokens.len();
        assert!(t <= BLOCK_SIZE);

        // Embedding + positional encoding
        let mut hidden: Vec<Vec<f32>> = Vec::with_capacity(t);
        for (pos, &tok) in tokens.iter().enumerate() {
            let mut h = vec![0.0f32; N_EMBD];
            for i in 0..N_EMBD {
                h[i] = self.wte_phase[tok][i] + self.wpe[pos][i];
            }
            hidden.push(h);
        }

        // Process through blocks
        for block in &self.blocks {
            hidden = self.forward_block(block, &hidden);
        }

        // Final layer norm + LM head
        let mut logits = Vec::with_capacity(t);
        for h in &hidden {
            let normed = layer_norm(h, &self.ln_f.weight, &self.ln_f.bias);
            let l = linear_no_bias(&self.lm_head, &normed);
            logits.push(l);
        }

        logits
    }

    fn forward_block(&self, block: &BlockWeights, hidden: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let t = hidden.len();

        // x = x + attn(ln_1(x))
        let normed_1: Vec<Vec<f32>> = hidden.iter()
            .map(|h| layer_norm(h, &block.ln_1.weight, &block.ln_1.bias))
            .collect();
        let attn_out = self.causal_self_attention(&block.attn, &normed_1);
        let mut h: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                let mut v = vec![0.0f32; N_EMBD];
                for j in 0..N_EMBD { v[j] = hidden[i][j] + attn_out[i][j]; }
                v
            })
            .collect();

        // x = x + ffn(ln_2(x))
        let normed_2: Vec<Vec<f32>> = h.iter()
            .map(|x| layer_norm(x, &block.ln_2.weight, &block.ln_2.bias))
            .collect();
        let ffn_out = match &block.ffn {
            FfnWeights::PerBand(w) => self.per_band_linear(w, &normed_2),
            FfnWeights::KerrMaestro(w) => self.kerr_maestro_add(w, &normed_2),
        };
        for i in 0..t {
            for j in 0..N_EMBD { h[i][j] += ffn_out[i][j]; }
        }

        h
    }

    fn causal_self_attention(&self, weights: &AttentionWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let t = x.len();

        // Compute Q, K, V for all positions
        let mut q_all = vec![vec![0.0f32; N_EMBD]; t];
        let mut k_all = vec![vec![0.0f32; N_EMBD]; t];
        let mut v_all = vec![vec![0.0f32; N_EMBD]; t];

        for pos in 0..t {
            let qkv = linear(&weights.c_attn.w, &weights.c_attn.b, &x[pos]);
            for i in 0..N_EMBD {
                q_all[pos][i] = qkv[i];
                k_all[pos][i] = qkv[N_EMBD + i];
                v_all[pos][i] = qkv[2 * N_EMBD + i];
            }
        }

        // Multi-head attention
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut out = vec![vec![0.0f32; N_EMBD]; t];

        for head in 0..N_HEAD {
            let offset = head * HEAD_DIM;

            // Compute attention scores for this head
            for qi in 0..t {
                // Compute attention weights
                let mut att = vec![f32::NEG_INFINITY; t];
                for ki in 0..=qi {  // causal: only attend to past
                    let mut dot = 0.0f32;
                    for d in 0..HEAD_DIM {
                        dot += q_all[qi][offset + d] * k_all[ki][offset + d];
                    }
                    att[ki] = dot * scale;
                }

                // Softmax
                let max_att = att[..=qi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for ki in 0..=qi {
                    att[ki] = (att[ki] - max_att).exp();
                    exp_sum += att[ki];
                }
                for ki in 0..=qi {
                    att[ki] /= exp_sum;
                }

                // Weighted sum of values
                for d in 0..HEAD_DIM {
                    let mut sum = 0.0f32;
                    for ki in 0..=qi {
                        sum += att[ki] * v_all[ki][offset + d];
                    }
                    out[qi][offset + d] = sum;
                }
            }
        }

        // Output projection
        let result: Vec<Vec<f32>> = out.iter()
            .map(|o| linear(&weights.c_proj.w, &weights.c_proj.b, o))
            .collect();

        result
    }

    fn per_band_linear(&self, weights: &PerBandLinearWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let t = x.len();
        let mut result = Vec::with_capacity(t);

        for pos in 0..t {
            let mut bands_out = vec![0.0f32; N_EMBD];

            for band in 0..N_BANDS {
                let r_in = x[pos][band * 2];
                let s_in = x[pos][band * 2 + 1];
                let w = &weights.band_w[band];
                let b = &weights.band_b[band];

                // y = W @ [r, s] + b  (2x2 matrix)
                bands_out[band * 2] = w[0][0] * r_in + w[1][0] * s_in + b[0];
                bands_out[band * 2 + 1] = w[0][1] * r_in + w[1][1] * s_in + b[1];
            }

            let projected = linear(&weights.out_proj.w, &weights.out_proj.b, &bands_out);
            result.push(projected);
        }

        result
    }

    fn kerr_maestro_add(&self, weights: &KerrMaestroAddWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let t = x.len();
        let mut result = Vec::with_capacity(t);

        for pos in 0..t {
            // Kerr path
            let kerr_out = self.kerr_ode_forward(&weights.kerr, &x[pos]);

            // Maestro path
            let maestro_out = self.maestro_forward(&weights.maestro, &x[pos]);

            // Combine + project
            let mut combined = vec![0.0f32; N_EMBD];
            for i in 0..N_EMBD {
                combined[i] = kerr_out[i] + maestro_out[i];
            }

            let projected = linear(&weights.out_proj.w, &weights.out_proj.b, &combined);
            result.push(projected);
        }

        result
    }

    fn kerr_ode_forward(&self, weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
        // Split into real and imaginary parts
        let mut r = vec![0.0f32; N_BANDS];
        let mut s = vec![0.0f32; N_BANDS];
        for k in 0..N_BANDS {
            r[k] = x[k * 2];
            s[k] = x[k * 2 + 1];
        }

        // Compute gamma (softplus of raw)
        let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();

        // 8 RK4 steps
        for _ in 0..RK4_N_STEPS {
            let (r_new, s_new) = rk4_step(&r, &s, RK4_DT, &gamma,
                                           &weights.omega, weights.alpha, weights.beta);
            r = r_new;
            s = s_new;
        }

        // Reinterleave
        let mut out = vec![0.0f32; N_EMBD];
        for k in 0..N_BANDS {
            out[k * 2] = r[k];
            out[k * 2 + 1] = s[k];
        }
        out
    }

    fn maestro_forward(&self, weights: &MaestroWeights, x: &[f32]) -> Vec<f32> {
        // Squeeze: 128 → 16
        let squeezed = linear(&weights.squeeze.w, &weights.squeeze.b, x);

        // GELU activation
        let activated: Vec<f32> = squeezed.iter().map(|&v| gelu(v)).collect();

        // Process: 16 → 128
        linear(&weights.process_1.w, &weights.process_1.b, &activated)
    }
}

// ─── Kerr-ODE derivative and RK4 ───────────────────────────────

fn kerr_derivative(
    r: &[f32], s: &[f32],
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = r.len();
    let mut dr = vec![0.0f32; n];
    let mut ds = vec![0.0f32; n];

    // Compute mag_sq for all bands
    let mag_sq: Vec<f32> = (0..n).map(|k| r[k] * r[k] + s[k] * s[k]).collect();

    // Conv1d with kernel [1, 1, 0, 1, 1], padding=2
    let mut ns = vec![0.0f32; n];
    for k in 0..n {
        if k >= 2 { ns[k] += mag_sq[k - 2]; }
        if k >= 1 { ns[k] += mag_sq[k - 1]; }
        if k + 1 < n { ns[k] += mag_sq[k + 1]; }
        if k + 2 < n { ns[k] += mag_sq[k + 2]; }
    }

    for k in 0..n {
        let phi = omega[k] + alpha * mag_sq[k] + beta * ns[k];
        dr[k] = -gamma[k] * r[k] - phi * s[k];
        ds[k] = -gamma[k] * s[k] + phi * r[k];
    }

    (dr, ds)
}

fn rk4_step(
    r: &[f32], s: &[f32], dt: f32,
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = r.len();

    // k1
    let (dr1, ds1) = kerr_derivative(r, s, gamma, omega, alpha, beta);

    // k2
    let r2: Vec<f32> = (0..n).map(|k| r[k] + 0.5 * dt * dr1[k]).collect();
    let s2: Vec<f32> = (0..n).map(|k| s[k] + 0.5 * dt * ds1[k]).collect();
    let (dr2, ds2) = kerr_derivative(&r2, &s2, gamma, omega, alpha, beta);

    // k3
    let r3: Vec<f32> = (0..n).map(|k| r[k] + 0.5 * dt * dr2[k]).collect();
    let s3: Vec<f32> = (0..n).map(|k| s[k] + 0.5 * dt * ds2[k]).collect();
    let (dr3, ds3) = kerr_derivative(&r3, &s3, gamma, omega, alpha, beta);

    // k4
    let r4: Vec<f32> = (0..n).map(|k| r[k] + dt * dr3[k]).collect();
    let s4: Vec<f32> = (0..n).map(|k| s[k] + dt * ds3[k]).collect();
    let (dr4, ds4) = kerr_derivative(&r4, &s4, gamma, omega, alpha, beta);

    // Combine: y_new = y + (dt/6)(k1 + 2k2 + 2k3 + k4)
    let dt6 = dt / 6.0;
    let r_new: Vec<f32> = (0..n)
        .map(|k| r[k] + dt6 * (dr1[k] + 2.0 * dr2[k] + 2.0 * dr3[k] + dr4[k]))
        .collect();
    let s_new: Vec<f32> = (0..n)
        .map(|k| s[k] + dt6 * (ds1[k] + 2.0 * ds2[k] + 2.0 * ds3[k] + ds4[k]))
        .collect();

    (r_new, s_new)
}

/// Linear without bias: y[i] = sum_j w[i][j] * x[j]
fn linear_no_bias(w: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
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
