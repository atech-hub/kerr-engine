//! Training loop — orchestration, generation, evaluation.
//!
//! The forward/backward pipeline lives in pipeline.rs.
//! The optimizer lives in optim.rs. Data in data.rs.
//! Checkpoint save/load in checkpoint.rs.

use crate::model::*;
use crate::checkpoint;
use crate::data::Dataset;
use crate::init::init_model;
use crate::optim::{self, Adam};
use crate::rng::Rng;

use std::thread;

// ─── Curriculum schedule ─────────────────────────────────────

/// Band stages for progressive curriculum.
/// Each entry is (bands, fraction_of_training).
/// Fractions must sum to 1.0. Default: Phase C schedule.
pub struct CurriculumSchedule {
    stages: Vec<(usize, f32)>,
}

impl CurriculumSchedule {
    /// Default 3-stage schedule matching Phase C Python.
    pub fn default_3stage() -> Self {
        Self {
            stages: vec![(8, 0.333), (24, 0.333), (N_BANDS, 0.334)],
        }
    }

    /// No curriculum — all bands from the start.
    #[allow(dead_code)]
    pub fn none() -> Self {
        Self {
            stages: vec![(N_BANDS, 1.0)],
        }
    }

    /// Returns active bands for the given iteration.
    pub fn active_bands(&self, iter: usize, n_iters: usize) -> usize {
        let mut cumulative = 0.0f32;
        for &(bands, frac) in &self.stages {
            cumulative += frac;
            if iter < (cumulative * n_iters as f32) as usize {
                return bands;
            }
        }
        self.stages.last().unwrap().0
    }

    /// Print schedule for logging.
    pub fn describe(&self, n_iters: usize) {
        let mut start = 0;
        let mut cumulative = 0.0f32;
        for &(bands, frac) in &self.stages {
            cumulative += frac;
            let end = (cumulative * n_iters as f32) as usize;
            print!("{bands} bands ({start}-{end})");
            start = end;
            if end < n_iters { print!(", "); }
        }
        println!();
    }
}

// ─── Training config ─────────────────────────────────────────

pub struct TrainConfig {
    pub data_path: String,
    pub n_iters: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub lr: f32,
    pub use_curriculum: bool,
    pub resume_path: Option<String>,
    pub checkpoint_every: usize,
}

// ─── Training loop ────────────────────────────────────────────

pub fn train_with_config(config: TrainConfig) {
    println!("Stage 4 — training from scratch\n");

    // Load dataset
    println!("Loading dataset from {}...", config.data_path);
    let dataset = Dataset::from_file(&config.data_path);

    // Init or resume
    let (mut model, mut optimizer, mut rng, start_iter) = if let Some(ref path) = config.resume_path {
        println!("Resuming from checkpoint: {path}");
        let state = checkpoint::load(path).expect("Failed to load checkpoint");
        println!("  Resumed at iteration {}", state.iter);
        let rng = state.rng;
        (state.model, state.optimizer, rng, state.iter)
    } else {
        println!("Initializing model...");
        let model = init_model(dataset.vocab_size, 42);
        let n_params = optim::count_params(&model);
        println!("  Trainable parameters: {}", n_params);
        let optimizer = Adam::new(config.lr, n_params);
        let rng = Rng::new(1337);
        (model, optimizer, rng, 0)
    };

    // Detect thread count
    let n_threads = thread::available_parallelism()
        .map(|n| n.get().min(config.batch_size))
        .unwrap_or(1);

    let n_params = optim::count_params(&model);

    // Curriculum schedule
    let curriculum = if config.use_curriculum {
        CurriculumSchedule::default_3stage()
    } else {
        CurriculumSchedule::none()
    };
    print!("  Curriculum: ");
    curriculum.describe(config.n_iters);

    // Training
    println!("\nTraining for {} iterations (batch_size={}, seq_len={}, lr={}, threads={})",
        config.n_iters, config.batch_size, config.seq_len, config.lr, n_threads);
    if start_iter > 0 {
        println!("  Resuming from iteration {start_iter}");
    }
    println!("{:>6} {:>10} {:>6} {:>10}", "Iter", "Loss", "Bands", "Time");
    println!("{}", "-".repeat(40));

    let log_every = 50;
    let eval_every = 300;

    let train_start = std::time::Instant::now();

    for iter in start_iter..config.n_iters {
        let iter_start = std::time::Instant::now();
        let active_bands = curriculum.active_bands(iter, config.n_iters);

        let (inputs, targets) = dataset.sample_batch(&mut rng, config.batch_size, config.seq_len);

        // Parallel forward+backward across batch elements
        let batch_results: Vec<(f32, Vec<f32>)> = thread::scope(|s| {
            let handles: Vec<_> = (0..config.batch_size).map(|b| {
                let model_ref = &model;
                let input_ref = &inputs[b];
                let target_ref = &targets[b];
                s.spawn(move || {
                    let cache = model_ref.forward_with_cache_curriculum(input_ref, active_bands);
                    let (loss, grads) = model_ref.backward(&cache, target_ref);
                    let flat_grads = optim::flatten_grads(&grads);
                    (loss, flat_grads)
                })
            }).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        // Reduce: sum losses and gradients
        let mut total_loss = 0.0f32;
        let mut grads = vec![0.0f32; n_params];
        for (loss, fg) in &batch_results {
            total_loss += loss;
            for (a, g) in grads.iter_mut().zip(fg.iter()) {
                *a += g;
            }
        }

        // Average over batch
        total_loss /= config.batch_size as f32;
        for g in grads.iter_mut() { *g /= config.batch_size as f32; }

        // Gradient clipping
        optim::clip_grad_norm(&mut grads, 1.0);

        // Adam step
        let mut params = optim::flatten_params(&model);
        optimizer.step(&mut params, &grads);
        optim::unflatten_params(&mut model, &params);

        let iter_time = iter_start.elapsed();

        if iter % log_every == 0 || iter == config.n_iters - 1 {
            println!("{:>6} {:>10.4} {:>6} {:>10.1?}", iter, total_loss, active_bands, iter_time);
        }

        // Eval: generate sample text
        if iter > 0 && iter % eval_every == 0 {
            let sample = generate(&model, &dataset, 200, &mut rng);
            println!("\n--- Sample (iter {iter}, {active_bands} bands) ---");
            println!("{sample}");
            println!("---\n");
        }

        // Checkpoint
        if config.checkpoint_every > 0 && iter > 0 && iter % config.checkpoint_every == 0 {
            let path = format!("checkpoint_iter{iter}.bin");
            match checkpoint::save(&path, &model, &optimizer, &rng, iter + 1, config.lr) {
                Ok(()) => println!("  [checkpoint saved: {path}]"),
                Err(e) => println!("  [checkpoint failed: {e}]"),
            }
        }
    }

    let total_time = train_start.elapsed();
    println!("\nTraining complete. Total time: {total_time:.1?}");

    // Save final checkpoint
    let path = format!("checkpoint_final.bin");
    match checkpoint::save(&path, &model, &optimizer, &rng, config.n_iters, config.lr) {
        Ok(()) => println!("  [final checkpoint saved: {path}]"),
        Err(e) => println!("  [final checkpoint failed: {e}]"),
    }

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
