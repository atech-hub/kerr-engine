//! Training loop — orchestration, generation, evaluation.
//!
//! The forward/backward pipeline lives in pipeline.rs.
//! The optimizer lives in optim.rs. Data in data.rs.
//! Checkpoint save/load in checkpoint.rs.

use crate::model::*;
use crate::backend;
use crate::checkpoint;
use crate::data::Dataset;
use crate::init::init_model;
use crate::optim::{self, Adam};
use crate::rng::Rng;

use std::io::Write;
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
    pub fn default_3stage(n_bands: usize) -> Self {
        Self {
            stages: vec![(8.min(n_bands), 0.333), (24.min(n_bands), 0.333), (n_bands, 0.334)],
        }
    }

    /// 4-stage schedule for high-dim GPU training.
    /// Adds an intermediate stage at n_bands/4 to avoid the 16x energy jump
    /// from 24 → full bands that triggers NaN on GPU at 768-dim+.
    pub fn default_4stage(n_bands: usize) -> Self {
        let s2 = 24.min(n_bands);
        let s3 = (n_bands / 4).max(s2); // ~96 at 384 bands
        Self {
            stages: vec![
                (8.min(n_bands), 0.20),
                (s2,             0.25),
                (s3,             0.25),
                (n_bands,        0.30),
            ],
        }
    }

    /// No curriculum — all bands from the start.
    #[allow(dead_code)]
    pub fn none(n_bands: usize) -> Self {
        Self {
            stages: vec![(n_bands, 1.0)],
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

    /// LR multiplier for GPU stability at transitions.
    /// Returns 0.1 within `margin` iters of a band transition, 1.0 elsewhere.
    /// The GPU FP drift accumulates proportional to lr — reducing lr at
    /// transitions prevents the energy spike from amplifying accumulated drift.
    pub fn lr_multiplier(&self, iter: usize, n_iters: usize, margin: usize) -> f32 {
        let mut cumulative = 0.0f32;
        for &(_bands, frac) in &self.stages[..self.stages.len().saturating_sub(1)] {
            cumulative += frac;
            let transition = (cumulative * n_iters as f32) as usize;
            if iter + margin >= transition && iter <= transition + margin {
                return 0.1;
            }
        }
        1.0
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
    pub word_level: bool,
    pub bpe_path: Option<String>,
    pub resume_path: Option<String>,
    pub checkpoint_every: usize,
    pub model_seed: u64,
    pub train_seed: u64,
    pub threads: Option<usize>,  // None = auto-detect
    pub force_cpu: bool,
    pub force_gpu: bool,
    pub gpu_device: Option<usize>,
    pub model_config: ModelConfig,
    pub dual_maestro: bool,
}

// ─── Training loop ────────────────────────────────────────────

pub fn train_with_config(config: TrainConfig) {
    println!("Stage 4 — training from scratch\n");

    // Load dataset
    println!("Loading dataset from {}...", config.data_path);
    let dataset = if let Some(ref bpe_path) = config.bpe_path {
        Dataset::from_file_bpe(&config.data_path, bpe_path, 0.9)
    } else if config.word_level {
        Dataset::from_file_words(&config.data_path, 0.9, 3)
    } else {
        Dataset::from_file(&config.data_path)
    };

    // Init or resume
    let (mut model, mut optimizer, mut rng, start_iter) = if let Some(ref path) = config.resume_path {
        println!("Resuming from checkpoint: {path}");
        let mut state = checkpoint::load(path).expect("Failed to load checkpoint");
        println!("  Resumed at iteration {}", state.iter);

        // Handle vocab size mismatch (cross-corpus sequential training)
        if dataset.vocab_size > state.model.vocab_size {
            let old_vs = state.model.vocab_size;
            let new_vs = dataset.vocab_size;
            let extra_tokens = new_vs - old_vs;
            println!("  Resizing vocab: {} → {} (+{} tokens)", old_vs, new_vs, extra_tokens);

            // Extend lm_head with random init for new tokens
            let n_embd = state.model.config.n_embd();
            let limit = 1.0 / (n_embd as f32).sqrt();
            let mut resize_rng = crate::rng::Rng::new(state.iter as u64 + 777);
            for _ in 0..extra_tokens {
                state.model.lm_head.push(
                    (0..n_embd).map(|_| resize_rng.uniform(limit)).collect()
                );
            }

            // Rebuild harmonic table for new vocab size
            state.model.wte_phase = crate::model::build_harmonic_table_sized(new_vs, n_embd);
            state.model.vocab_size = new_vs;

            // Extend Adam m/v for the extra lm_head params
            state.optimizer.extend(extra_tokens * n_embd);
        } else if dataset.vocab_size < state.model.vocab_size {
            println!("  Note: dataset vocab ({}) < checkpoint vocab ({}), keeping larger model",
                     dataset.vocab_size, state.model.vocab_size);
        }

        let rng = state.rng;
        (state.model, state.optimizer, rng, state.iter)
    } else {
        println!("Initializing model (seed={}{})...", config.model_seed,
            if config.dual_maestro { ", dual-maestro" } else { "" });
        let model = crate::init::init_model_ext(dataset.vocab_size, config.model_seed, config.model_config, config.dual_maestro);
        let n_params = optim::count_params(&model);
        println!("  Trainable parameters: {}", n_params);
        let optimizer = Adam::new(config.lr, n_params);
        let rng = Rng::new(config.train_seed);
        (model, optimizer, rng, 0)
    };

    // Select compute backend (CPU at 128-dim, GPU at larger scales)
    let compute_backend = backend::auto_select(config.model_config.n_embd(), config.force_cpu, config.force_gpu, config.gpu_device);

    // Upload weights to GPU VRAM (no-op for CPU backend)
    compute_backend.load_weights(&model);

    // Thread count: explicit or auto-detect (capped at batch_size)
    let n_threads = config.threads.unwrap_or_else(|| {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }).min(config.batch_size);

    let n_params = optim::count_params(&model);

    // Curriculum schedule — use 4-stage at high dims for GPU stability
    let n_bands = config.model_config.n_bands;
    let curriculum = if !config.use_curriculum {
        CurriculumSchedule::none(n_bands)
    } else if n_bands > 64 {
        CurriculumSchedule::default_4stage(n_bands)
    } else {
        CurriculumSchedule::default_3stage(n_bands)
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

    // Track loss history for summary
    let mut loss_history: Vec<(usize, f32)> = Vec::new();
    let mut val_history: Vec<(usize, f32)> = Vec::new();

    let train_start = std::time::Instant::now();

    for iter in start_iter..config.n_iters {
        let iter_start = std::time::Instant::now();
        let active_bands = curriculum.active_bands(iter, config.n_iters);

        let (inputs, targets) = dataset.sample_batch(&mut rng, config.batch_size, config.seq_len);

        // Parallel forward+backward across batch elements.
        // 4 threads submit GPU work concurrently — overlapping submissions keep GPU busy.
        // Each thread processes seq_len positions independently.
        let batch_results: Vec<(f32, Vec<f32>)> = thread::scope(|s| {
            let handles: Vec<_> = (0..config.batch_size).map(|b| {
                let model_ref = &model;
                let backend_ref = &*compute_backend;
                let input_ref = &inputs[b];
                let target_ref = &targets[b];
                s.spawn(move || {
                    let cache = model_ref.forward_with_cache_curriculum(input_ref, active_bands, backend_ref);
                    let (loss, grads) = model_ref.backward(&cache, target_ref, backend_ref);
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

        // Two-system weight update:
        // 1. Pool cache invalidation — forces re-upload on next access for attention, LN, LM head
        // 2. Resident buffer update — writes FFN weights back for fused chain
        compute_backend.invalidate_weight_cache();
        compute_backend.update_weights(&model);

        let iter_time = iter_start.elapsed();

        if iter % log_every == 0 || iter == config.n_iters - 1 {
            println!("{:>6} {:>10.4} {:>6} {:>10.1?}", iter, total_loss, active_bands, iter_time);
            loss_history.push((iter, total_loss));
        }

        // Eval: validation loss + sample text
        if iter > 0 && iter % eval_every == 0 {
            let val_loss = eval_val_loss(&model, &dataset, config.batch_size, config.seq_len, active_bands, 8, &*compute_backend);
            println!("  [val_loss={val_loss:.4}]");
            val_history.push((iter, val_loss));
            let sample = generate(&model, &dataset, 200, &mut rng, &*compute_backend);
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

    // Final val loss
    let final_active = curriculum.active_bands(config.n_iters.saturating_sub(1), config.n_iters);
    let final_val = eval_val_loss(&model, &dataset, config.batch_size, config.seq_len, final_active, 8, &*compute_backend);
    println!("  Final val_loss: {final_val:.4}");
    val_history.push((config.n_iters, final_val));

    // Write training summary
    write_summary(&config, &dataset, n_params, total_time.as_secs_f32(), &loss_history, &val_history);

    // Save final checkpoint
    let path = format!("checkpoint_final.bin");
    match checkpoint::save(&path, &model, &optimizer, &rng, config.n_iters, config.lr) {
        Ok(()) => println!("  [final checkpoint saved: {path}]"),
        Err(e) => println!("  [final checkpoint failed: {e}]"),
    }

    // Final generation
    let sample = generate(&model, &dataset, 500, &mut rng, &*compute_backend);
    println!("\n=== Final sample ===");
    println!("{sample}");
    println!("===");
}

/// Write JSON training summary for cross-run comparison.
fn write_summary(
    config: &TrainConfig,
    dataset: &Dataset,
    n_params: usize,
    total_secs: f32,
    loss_history: &[(usize, f32)],
    val_history: &[(usize, f32)],
) {
    let path = "training_summary.json";
    let mut f = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => { println!("  [summary write failed: {e}]"); return; }
    };

    let mode = if config.bpe_path.is_some() { "bpe" } else if config.word_level { "word" } else { "char" };
    let final_train = loss_history.last().map(|(_, l)| *l).unwrap_or(0.0);
    let final_val = val_history.last().map(|(_, l)| *l).unwrap_or(0.0);

    // Hand-write JSON to avoid serde dependency
    let _ = writeln!(f, "{{");
    let _ = writeln!(f, "  \"data_path\": \"{}\",", config.data_path.replace('\\', "/"));
    let _ = writeln!(f, "  \"mode\": \"{mode}\",");
    let _ = writeln!(f, "  \"vocab_size\": {},", dataset.vocab_size);
    let _ = writeln!(f, "  \"n_params\": {n_params},");
    let _ = writeln!(f, "  \"n_iters\": {},", config.n_iters);
    let _ = writeln!(f, "  \"batch_size\": {},", config.batch_size);
    let _ = writeln!(f, "  \"seq_len\": {},", config.seq_len);
    let _ = writeln!(f, "  \"lr\": {},", config.lr);
    let _ = writeln!(f, "  \"model_seed\": {},", config.model_seed);
    let _ = writeln!(f, "  \"train_seed\": {},", config.train_seed);
    let _ = writeln!(f, "  \"curriculum\": {},", config.use_curriculum);
    let _ = writeln!(f, "  \"total_seconds\": {total_secs:.1},");
    let _ = writeln!(f, "  \"final_train_loss\": {final_train:.4},");
    let _ = writeln!(f, "  \"final_val_loss\": {final_val:.4},");

    // Loss curve
    let _ = write!(f, "  \"train_loss\": [");
    for (i, (iter, loss)) in loss_history.iter().enumerate() {
        if i > 0 { let _ = write!(f, ", "); }
        let _ = write!(f, "[{iter}, {loss:.4}]");
    }
    let _ = writeln!(f, "],");

    // Val loss curve
    let _ = write!(f, "  \"val_loss\": [");
    for (i, (iter, loss)) in val_history.iter().enumerate() {
        if i > 0 { let _ = write!(f, ", "); }
        let _ = write!(f, "[{iter}, {loss:.4}]");
    }
    let _ = writeln!(f, "]");

    let _ = writeln!(f, "}}");
    println!("  [summary saved: {path}]");
}

/// Evaluate validation loss over multiple batches.
/// Uses a fixed RNG seed so val loss is comparable across evals.
fn eval_val_loss(model: &ModelWeights, dataset: &Dataset, batch_size: usize, seq_len: usize, active_bands: usize, n_batches: usize, backend: &dyn backend::ComputeBackend) -> f32 {
    let mut val_rng = Rng::new(9999); // fixed seed, independent of training RNG
    let mut total_loss = 0.0f32;
    let mut n_samples = 0usize;

    for _ in 0..n_batches {
        let (inputs, targets) = dataset.sample_val_batch(&mut val_rng, batch_size, seq_len);
        for b in 0..batch_size {
            let cache = model.forward_with_cache_curriculum(&inputs[b], active_bands, backend);
            // Compute loss only (no gradients needed)
            let t = cache.logits.len();
            let mut batch_loss = 0.0f32;
            for pos in 0..t {
                let logits = &cache.logits[pos];
                let target = targets[b][pos];
                let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_l: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
                let sum_exp: f32 = exp_l.iter().sum();
                batch_loss += -(exp_l[target] / sum_exp).ln();
            }
            total_loss += batch_loss / t as f32;
            n_samples += 1;
        }
    }

    total_loss / n_samples as f32
}

/// Generate text by sampling from the model.
fn generate(model: &ModelWeights, dataset: &Dataset, n_tokens: usize, rng: &mut Rng, backend: &dyn crate::backend::ComputeBackend) -> String {
    let start_idx = *dataset.token_to_idx.get("\n").unwrap_or(&0);
    let mut tokens = vec![start_idx];

    for _ in 0..n_tokens {
        let block_size = model.config.block_size;
        let start = if tokens.len() > block_size { tokens.len() - block_size } else { 0 };
        let context = &tokens[start..];

        let cache = model.forward_with_cache(context, backend);
        let logits = cache.logits.last().unwrap();

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

    dataset.decode(&tokens[1..])
}
