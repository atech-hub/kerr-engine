//! Training loop — orchestration, generation, evaluation.
//!
//! The forward/backward pipeline lives in pipeline.rs.
//! The optimizer lives in optim.rs. Data in data.rs.

use crate::model::*;
use crate::data::Dataset;
use crate::init::init_model;
use crate::optim::{self, Adam};
use crate::rng::Rng;

// ─── Training loop ────────────────────────────────────────────

pub fn train(data_path: &str, n_iters: usize, batch_size: usize, seq_len: usize, lr: f32) {
    println!("Stage 4 — training from scratch\n");

    // Load dataset
    println!("Loading dataset from {data_path}...");
    let dataset = Dataset::from_file(data_path);

    // Init model
    println!("Initializing model...");
    let mut model = init_model(dataset.vocab_size, 42);
    let n_params = optim::count_params(&model);
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

        let (inputs, targets) = dataset.sample_batch(&mut rng, batch_size, seq_len);

        let mut total_loss = 0.0f32;
        let mut accumulated_grads: Option<Vec<f32>> = None;

        for b in 0..batch_size {
            let cache = model.forward_with_cache(&inputs[b]);
            let (loss, grads) = model.backward(&cache, &targets[b]);
            total_loss += loss;

            let flat_grads = optim::flatten_grads(&grads);
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
        optim::clip_grad_norm(&mut grads, 1.0);

        // Adam step
        let mut params = optim::flatten_params(&model);
        optimizer.step(&mut params, &grads);
        optim::unflatten_params(&mut model, &params);

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
    let newline_idx = dataset.char_to_idx.get(&'\n').copied().unwrap_or(0);
    let mut tokens = vec![newline_idx];

    for _ in 0..n_tokens {
        let start = if tokens.len() > BLOCK_SIZE { tokens.len() - BLOCK_SIZE } else { 0 };
        let context = &tokens[start..];

        let logits_all = model.forward(context);
        let logits = logits_all.last().unwrap();

        let temp = 0.8f32;
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_l: Vec<f32> = logits.iter().map(|&l| ((l - max_l) / temp).exp()).collect();
        let sum_exp: f32 = exp_l.iter().sum();
        let probs: Vec<f32> = exp_l.iter().map(|e| e / sum_exp).collect();

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
        .skip(1)
        .map(|&t| {
            if t < dataset.idx_to_char.len() {
                dataset.idx_to_char[t]
            } else {
                '?'
            }
        })
        .collect()
}
