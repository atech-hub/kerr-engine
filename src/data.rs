//! Character-level dataset and batch sampling.

use std::collections::HashMap;
use std::fs;
use crate::rng::Rng;

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
    pub fn sample_batch(&self, rng: &mut Rng, batch_size: usize, seq_len: usize)
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
