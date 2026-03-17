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

fn init_layer_norm(n_embd: usize) -> LayerNormWeights {
    LayerNormWeights {
        weight: vec![1.0f32; n_embd],
        bias: vec![0.0f32; n_embd],
    }
}

fn init_attention(rng: &mut Rng, n_embd: usize, n_head: usize) -> AttentionWeights {
    AttentionWeights {
        c_attn: init_linear(rng, 3 * n_embd, n_embd),
        c_proj: init_linear(rng, n_embd, n_embd),
        n_head,
    }
}

fn init_per_band_linear(rng: &mut Rng, n_bands: usize, n_embd: usize) -> PerBandLinearWeights {
    let limit = 1.0 / 2.0f32.sqrt();
    let band_w: Vec<[[f32; 2]; 2]> = (0..n_bands)
        .map(|_| [
            [rng.uniform(limit), rng.uniform(limit)],
            [rng.uniform(limit), rng.uniform(limit)],
        ])
        .collect();
    let band_b = vec![[0.0f32; 2]; n_bands];

    PerBandLinearWeights {
        band_w,
        band_b,
        out_proj: init_linear(rng, n_embd, n_embd),
    }
}

fn init_kerr_weights(config: &ModelConfig) -> KerrWeights {
    let n_bands = config.n_bands;
    let gamma_raw_val = ((0.1f32).exp() - 1.0).ln(); // softplus^-1(0.1)
    KerrWeights {
        gamma_raw: vec![gamma_raw_val; n_bands],
        omega: (0..n_bands).map(|k| (k + 1) as f32 / n_bands as f32).collect(),
        alpha: 0.1,
        beta: 0.1,
        rk4_n_steps: config.rk4_n_steps,
    }
}

fn init_maestro_weights(rng: &mut Rng, n_embd: usize, maestro_dim: usize) -> MaestroWeights {
    MaestroWeights {
        squeeze: init_linear(rng, maestro_dim, n_embd),
        process_1: init_linear(rng, n_embd, maestro_dim),
    }
}

fn init_kerr_maestro_add(rng: &mut Rng, config: &ModelConfig) -> KerrMaestroAddWeights {
    let n_embd = config.n_embd();
    KerrMaestroAddWeights {
        kerr: init_kerr_weights(config),
        maestro: init_maestro_weights(rng, n_embd, config.maestro_dim),
        out_proj: init_linear(rng, n_embd, n_embd),
    }
}

fn init_kerr_dual_maestro(rng: &mut Rng, config: &ModelConfig) -> KerrDualMaestroWeights {
    let n_embd = config.n_embd();
    KerrDualMaestroWeights {
        kerr: init_kerr_weights(config),
        maestro_in: init_maestro_weights(rng, n_embd, config.maestro_dim),
        maestro_out: init_maestro_weights(rng, n_embd, config.maestro_dim),
        out_proj: init_linear(rng, n_embd, n_embd),
    }
}

pub fn init_model(vocab_size: usize, seed: u64, config: ModelConfig) -> ModelWeights {
    init_model_ext(vocab_size, seed, config, false)
}

pub fn init_model_ext(vocab_size: usize, seed: u64, config: ModelConfig, dual_maestro: bool) -> ModelWeights {
    config.validate();
    let mut rng = Rng::new(seed);
    let n_embd = config.n_embd();
    let n_bands = config.n_bands;

    // Block 0: PerBandLinear
    let block0 = BlockWeights {
        ln_1: init_layer_norm(n_embd),
        attn: init_attention(&mut rng, n_embd, config.n_head),
        ln_2: init_layer_norm(n_embd),
        ffn: FfnWeights::PerBand(init_per_band_linear(&mut rng, n_bands, n_embd)),
    };

    // Blocks 1-(n_layers-1): KerrMaestroAdd or KerrDualMaestro
    // IMPORTANT: RNG consumption order must match original — attention before FFN.
    let mut blocks = vec![block0];
    for _ in 0..(config.n_layers - 1) {
        let ln_1 = init_layer_norm(n_embd);
        let attn = init_attention(&mut rng, n_embd, config.n_head);
        let ln_2 = init_layer_norm(n_embd);
        let ffn = if dual_maestro {
            FfnWeights::KerrDualMaestro(init_kerr_dual_maestro(&mut rng, &config))
        } else {
            FfnWeights::KerrMaestro(init_kerr_maestro_add(&mut rng, &config))
        };
        blocks.push(BlockWeights { ln_1, attn, ln_2, ffn });
    }

    // LM head
    let limit = 1.0 / (n_embd as f32).sqrt();
    let lm_head: Vec<Vec<f32>> = (0..vocab_size)
        .map(|_| (0..n_embd).map(|_| rng.uniform(limit)).collect())
        .collect();

    ModelWeights {
        config,
        vocab_size,
        wte_phase: build_harmonic_table_sized(vocab_size, n_embd),
        wpe: build_positional_table(config.block_size, n_embd),
        blocks,
        ln_f: init_layer_norm(n_embd),
        lm_head,
    }
}
