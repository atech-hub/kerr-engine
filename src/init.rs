//! Weight initialization — random init for trainable parameters.
//!
//! Matches PyTorch default initialization (uniform +/- 1/sqrt(fan_in)).
//! Frozen harmonic embeddings and positional encoding are not initialized here.

use crate::model::*;
use crate::rng::Rng;

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
