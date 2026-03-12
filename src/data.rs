//! Character-level dataset with train/val split and batch sampling.

use std::collections::HashMap;
use std::fs;
use crate::rng::Rng;

pub struct Dataset {
    pub train_data: Vec<usize>,
    pub val_data: Vec<usize>,
    pub vocab_size: usize,
    pub char_to_idx: HashMap<char, usize>,
    pub idx_to_char: Vec<char>,
}

impl Dataset {
    pub fn from_file(path: &str) -> Self {
        Self::from_file_with_split(path, 0.9)
    }

    pub fn from_file_with_split(path: &str, train_frac: f32) -> Self {
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

        let split = (data.len() as f32 * train_frac) as usize;
        let train_data = data[..split].to_vec();
        let val_data = data[split..].to_vec();

        println!("  Dataset: {} chars (train={}, val={}), vocab_size={}",
            data.len(), train_data.len(), val_data.len(), vocab_size);

        Self { train_data, val_data, vocab_size, char_to_idx, idx_to_char: chars }
    }

    /// Sample a random batch from training data.
    pub fn sample_batch(&self, rng: &mut Rng, batch_size: usize, seq_len: usize)
        -> (Vec<Vec<usize>>, Vec<Vec<usize>>)
    {
        Self::sample_from(&self.train_data, rng, batch_size, seq_len)
    }

    /// Sample a random batch from validation data.
    pub fn sample_val_batch(&self, rng: &mut Rng, batch_size: usize, seq_len: usize)
        -> (Vec<Vec<usize>>, Vec<Vec<usize>>)
    {
        Self::sample_from(&self.val_data, rng, batch_size, seq_len)
    }

    fn sample_from(data: &[usize], rng: &mut Rng, batch_size: usize, seq_len: usize)
        -> (Vec<Vec<usize>>, Vec<Vec<usize>>)
    {
        let max_start = data.len() - seq_len - 1;
        let mut inputs = Vec::with_capacity(batch_size);
        let mut targets = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let start = rng.next_usize(max_start);
            inputs.push(data[start..start + seq_len].to_vec());
            targets.push(data[start + 1..start + seq_len + 1].to_vec());
        }

        (inputs, targets)
    }
}
