//! Binary weight loader for exported Phase C models.
//!
//! Format: flat f32 little-endian binary.
//! Header: [magic=0x4B455252, version=1, vocab_size, n_layers]

use std::io::{self, Read, BufReader};
use std::fs::File;
use crate::model::*;

const MAGIC: u32 = 0x4B455252;  // 'KERR'
const VERSION: u32 = 1;

struct BinReader {
    reader: BufReader<File>,
}

impl BinReader {
    fn new(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        Ok(Self { reader: BufReader::new(file) })
    }

    fn read_u32(&mut self) -> io::Result<u32> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_f32(&mut self) -> io::Result<f32> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_f32_vec(&mut self, n: usize) -> io::Result<Vec<f32>> {
        let mut v = vec![0.0f32; n];
        for i in 0..n {
            v[i] = self.read_f32()?;
        }
        Ok(v)
    }

    fn read_f32_matrix(&mut self, rows: usize, cols: usize) -> io::Result<Vec<Vec<f32>>> {
        let mut m = Vec::with_capacity(rows);
        for _ in 0..rows {
            m.push(self.read_f32_vec(cols)?);
        }
        Ok(m)
    }

    fn read_linear(&mut self, out_dim: usize, in_dim: usize) -> io::Result<LinearWeights> {
        let w = self.read_f32_matrix(out_dim, in_dim)?;
        let b = self.read_f32_vec(out_dim)?;
        Ok(LinearWeights { w, b })
    }

    fn read_layernorm(&mut self, dim: usize) -> io::Result<LayerNormWeights> {
        let weight = self.read_f32_vec(dim)?;
        let bias = self.read_f32_vec(dim)?;
        Ok(LayerNormWeights { weight, bias })
    }
}

/// Load model weights from binary file.
pub fn load_weights(path: &str) -> io::Result<ModelWeights> {
    let mut r = BinReader::new(path)?;

    // Header
    let magic = r.read_u32()?;
    if magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("Bad magic: expected 0x{MAGIC:08X}, got 0x{magic:08X}")));
    }
    let version = r.read_u32()?;
    if version != VERSION {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("Bad version: expected {VERSION}, got {version}")));
    }
    let vocab_size = r.read_u32()? as usize;
    let n_layers = r.read_u32()? as usize;
    assert_eq!(n_layers, N_LAYERS);

    println!("  Loading weights: vocab={vocab_size}, layers={n_layers}");

    // Frozen embeddings
    let wte_phase = r.read_f32_matrix(vocab_size, N_EMBD)?;
    let wpe = r.read_f32_matrix(BLOCK_SIZE, N_EMBD)?;

    // Blocks
    let mut blocks = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let ln_1 = r.read_layernorm(N_EMBD)?;
        let c_attn = r.read_linear(3 * N_EMBD, N_EMBD)?;
        let c_proj = r.read_linear(N_EMBD, N_EMBD)?;
        let attn = AttentionWeights { c_attn, c_proj };
        let ln_2 = r.read_layernorm(N_EMBD)?;

        let ffn = if i == 0 {
            // PerBandLinear
            let band_w_flat = r.read_f32_vec(N_BANDS * 2 * 2)?;
            let band_b_flat = r.read_f32_vec(N_BANDS * 2)?;
            let out_proj = r.read_linear(N_EMBD, N_EMBD)?;

            let mut band_w = vec![[[0.0f32; 2]; 2]; N_BANDS];
            let mut band_b = vec![[0.0f32; 2]; N_BANDS];
            for n in 0..N_BANDS {
                for i in 0..2 {
                    for j in 0..2 {
                        band_w[n][i][j] = band_w_flat[n * 4 + i * 2 + j];
                    }
                }
                band_b[n][0] = band_b_flat[n * 2];
                band_b[n][1] = band_b_flat[n * 2 + 1];
            }

            FfnWeights::PerBand(PerBandLinearWeights { band_w, band_b, out_proj })
        } else {
            // KerrMaestroAdd
            let gamma_raw = r.read_f32_vec(N_BANDS)?;
            let omega = r.read_f32_vec(N_BANDS)?;
            let alpha = r.read_f32()?;
            let beta = r.read_f32()?;
            let kerr = KerrWeights { gamma_raw, omega, alpha, beta };

            let squeeze = r.read_linear(MAESTRO_DIM, N_EMBD)?;
            let process_1 = r.read_linear(N_EMBD, MAESTRO_DIM)?;
            let maestro = MaestroWeights { squeeze, process_1 };

            let out_proj = r.read_linear(N_EMBD, N_EMBD)?;

            FfnWeights::KerrMaestro(KerrMaestroAddWeights { kerr, maestro, out_proj })
        };

        blocks.push(BlockWeights { ln_1, attn, ln_2, ffn });
    }

    // Final layer norm
    let ln_f = r.read_layernorm(N_EMBD)?;

    // LM head (no bias)
    let lm_head = r.read_f32_matrix(vocab_size, N_EMBD)?;

    println!("  Weights loaded successfully.");

    Ok(ModelWeights {
        vocab_size,
        wte_phase,
        wpe,
        blocks,
        ln_f,
        lm_head,
    })
}

/// Load test vectors for validation.
pub struct TestVectors {
    pub tokens: Vec<usize>,
    pub expected_logits: Vec<Vec<f32>>,
}

pub fn load_test_vectors(path: &str) -> io::Result<TestVectors> {
    let mut r = BinReader::new(path)?;

    let n_tokens = r.read_u32()? as usize;
    let vocab_size = r.read_u32()? as usize;

    let mut tokens = vec![0usize; n_tokens];
    for i in 0..n_tokens {
        tokens[i] = r.read_u32()? as usize;
    }

    let expected_logits = r.read_f32_matrix(n_tokens, vocab_size)?;

    Ok(TestVectors { tokens, expected_logits })
}
