//! Checkpoint save/load — full training state to/from binary file.
//!
//! Format v1: magic + version(1) + vocab_size + metadata + Adam + weights
//! Format v2: magic + version(2) + ModelConfig + vocab_size + metadata + Adam + weights
//!
//! v2 checkpoints are self-describing — the architecture is stored in the header.
//! v1 checkpoints require external ModelConfig (defaults to 128-dim).

use crate::model::*;
use crate::optim::{self, Adam};
use crate::init::init_model;
use crate::rng::Rng;

use std::io::{self, Read, Write};
use std::fs::File;

const MAGIC: [u8; 4] = *b"KCHK"; // Kerr CHecKpoint
const VERSION: u32 = 2;

/// Everything needed to resume training.
#[allow(dead_code)]
pub struct TrainingState {
    pub model: ModelWeights,
    pub optimizer: Adam,
    pub rng: Rng,
    pub iter: usize,
    pub lr: f32,
}

/// Save full training state to a checkpoint file (v2 format — self-describing).
pub fn save(path: &str, model: &ModelWeights, optimizer: &Adam, rng: &Rng, iter: usize, lr: f32) -> io::Result<()> {
    let mut f = File::create(path)?;

    // Header
    f.write_all(&MAGIC)?;
    f.write_all(&VERSION.to_le_bytes())?;

    // ModelConfig (v2)
    write_u32(&mut f, model.config.n_bands as u32)?;
    write_u32(&mut f, model.config.n_head as u32)?;
    write_u32(&mut f, model.config.n_layers as u32)?;
    write_u32(&mut f, model.config.maestro_dim as u32)?;
    write_u32(&mut f, model.config.block_size as u32)?;
    write_u32(&mut f, model.config.rk4_n_steps as u32)?;

    // Metadata
    write_u64(&mut f, model.vocab_size as u64)?;
    write_u64(&mut f, iter as u64)?;
    write_f32(&mut f, lr)?;
    write_u64(&mut f, rng.state())?;

    // Adam state
    let (adam_t, adam_m, adam_v) = optimizer.checkpoint_state();
    write_u64(&mut f, adam_t as u64)?;
    write_f32_slice(&mut f, adam_m)?;
    write_f32_slice(&mut f, adam_v)?;

    // Model weights (flat)
    let params = optim::flatten_params(model);
    write_f32_slice(&mut f, &params)?;

    Ok(())
}

/// Load a checkpoint, auto-detecting v1 or v2 format.
/// v2 checkpoints self-describe (no config needed).
/// v1 checkpoints use the provided fallback config.
pub fn load(path: &str) -> io::Result<TrainingState> {
    load_with_fallback(path, ModelConfig::default_128())
}

/// Load a checkpoint with an explicit fallback config for v1 checkpoints.
/// v2 checkpoints ignore the fallback and use the embedded config.
pub fn load_with_config(path: &str, fallback: ModelConfig) -> io::Result<TrainingState> {
    load_with_fallback(path, fallback)
}

fn load_with_fallback(path: &str, fallback: ModelConfig) -> io::Result<TrainingState> {
    let mut f = File::open(path)?;

    // Header
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a kerr-engine checkpoint"));
    }
    let version = read_u32(&mut f)?;

    let config = match version {
        1 => fallback,
        2 => {
            let n_bands = read_u32(&mut f)? as usize;
            let n_head = read_u32(&mut f)? as usize;
            let n_layers = read_u32(&mut f)? as usize;
            let maestro_dim = read_u32(&mut f)? as usize;
            let block_size = read_u32(&mut f)? as usize;
            let rk4_n_steps = read_u32(&mut f)? as usize;
            ModelConfig { n_bands, n_head, n_layers, maestro_dim, block_size, rk4_n_steps }
        }
        _ => {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("Checkpoint version {version}, expected 1 or 2")));
        }
    };

    // Metadata
    let vocab_size = read_u64(&mut f)? as usize;
    let iter = read_u64(&mut f)? as usize;
    let lr = read_f32(&mut f)?;
    let rng_state = read_u64(&mut f)?;

    // Adam state
    let adam_t = read_u64(&mut f)? as usize;
    let n_params = optim::count_params_for_vocab(vocab_size, &config);
    let adam_m = read_f32_vec(&mut f, n_params)?;
    let adam_v = read_f32_vec(&mut f, n_params)?;

    // Model weights
    let params = read_f32_vec(&mut f, n_params)?;
    let mut model = init_model(vocab_size, 0, config); // seed irrelevant, we overwrite
    optim::unflatten_params(&mut model, &params);

    let optimizer = Adam::from_checkpoint(lr, adam_t, adam_m, adam_v);
    let rng = Rng::from_state(rng_state);

    if version == 2 {
        println!("  Checkpoint v2: {}x{} ({} bands, {} heads, {} layers, maestro {})",
            config.n_embd(), config.n_embd(), config.n_bands, config.n_head,
            config.n_layers, config.maestro_dim);
    } else {
        println!("  Checkpoint v1: using provided config ({} bands)", config.n_bands);
    }

    Ok(TrainingState { model, optimizer, rng, iter, lr })
}

// ─── Binary helpers ──────────────────────────────────────────

fn write_u32(f: &mut File, v: u32) -> io::Result<()> {
    f.write_all(&v.to_le_bytes())
}

fn write_u64(f: &mut File, v: u64) -> io::Result<()> {
    f.write_all(&v.to_le_bytes())
}

fn write_f32(f: &mut File, v: f32) -> io::Result<()> {
    f.write_all(&v.to_le_bytes())
}

fn write_f32_slice(f: &mut File, data: &[f32]) -> io::Result<()> {
    write_u64(f, data.len() as u64)?;
    for &v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_u32(f: &mut File) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(f: &mut File) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f32(f: &mut File) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f32_vec(f: &mut File, expected_len: usize) -> io::Result<Vec<f32>> {
    let len = read_u64(f)? as usize;
    if len != expected_len {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("Expected {expected_len} floats, got {len}")));
    }
    let mut data = vec![0.0f32; len];
    for v in &mut data {
        *v = read_f32(f)?;
    }
    Ok(data)
}
