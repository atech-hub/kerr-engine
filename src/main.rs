//! Kerr Engine — wave-native compute for Kerr-ODE architecture
//!
//! Stage 1: Single Euler step GPU validation (COMPLETE)
//! Stage 2: Full forward pass, load Python weights, validate against Python logits.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[allow(dead_code)]
mod backend;
mod bpe;
mod checkpoint;
mod backward;
mod data;
mod gpu;
mod gpu_backend;
mod gpu_dispatch;
mod gpu_pipelines;
mod gpu_persistent;
mod gpu_validate;
mod grad_test;
mod init;
mod model;
mod optim;
mod pipeline;
mod rng;
mod train;
mod weights;

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Top-level help: no args, --help, or -h
    match args.get(1).map(|s| s.as_str()) {
        None | Some("--help") | Some("-h") | Some("help") => {
            print_main_help();
            return;
        }
        _ => {}
    }

    println!("kerr-engine v0.2.0\n");

    match args.get(1).map(|s| s.as_str()) {
        Some("validate") => {
            if has_help_flag(&args[2..]) {
                print_validate_help();
                return;
            }
            let weights_path = args.get(2).map(|s| s.as_str()).unwrap_or("model.bin");
            let default_test = weights_path.replace(".bin", "_test.bin");
            let test_path = args.get(3).map(|s| s.as_str()).unwrap_or(&default_test);
            validate_forward_pass(weights_path, test_path);
        }
        Some("gpu-test") => {
            if has_help_flag(&args[2..]) {
                print_gpu_test_help();
                return;
            }
            gpu_kernel_test();
        }
        Some("gpu-backend-test") => {
            if has_help_flag(&args[2..]) {
                print_gpu_backend_test_help();
                return;
            }
            gpu_backend::validate_gpu_backend();
        }
        Some("gpu-bench") => {
            if has_help_flag(&args[2..]) {
                print_gpu_bench_help();
                return;
            }
            gpu_backend::benchmark_gpu_vs_cpu();
        }
        Some("gpu-persistent-bench") => {
            if has_help_flag(&args[2..]) {
                print_gpu_persistent_bench_help();
                return;
            }
            gpu_persistent::benchmark_persistent();
        }
        Some("backend-select") => {
            if has_help_flag(&args[2..]) {
                print_backend_select_help();
                return;
            }
            println!("Backend auto-selection test\n");
            for dim in [128, 256, 512, 768, 1024] {
                print!("  dim={dim}: ");
                let _ = backend::auto_select(dim, false, false, None);
            }
            println!();
            print!("  dim=128 --force-gpu: ");
            let _ = backend::auto_select(128, false, true, None);
            print!("  dim=1024 --force-cpu: ");
            let _ = backend::auto_select(1024, true, false, None);
        }
        Some("grad-test") => {
            if has_help_flag(&args[2..]) {
                print_grad_test_help();
                return;
            }
            let test_path = args.get(2).map(|s| s.as_str()).unwrap_or("reference/gradient_test.bin");
            grad_test::validate_gradients(test_path);
        }
        Some("list-gpus") => {
            if has_help_flag(&args[2..]) {
                print_list_gpus_help();
                return;
            }
            list_gpus();
        }
        Some("train") => {
            if has_help_flag(&args[2..]) {
                print_train_help();
                return;
            }
            run_train(&args);
        }
        Some(cmd) => {
            eprintln!("Unknown command: {cmd}\n");
            eprintln!("Run 'kerr-engine --help' to see available commands.");
            std::process::exit(1);
        }
        _ => unreachable!(),
    }
}

// ─── Help system ──────────────────────────────────────────────

fn has_help_flag(args: &[String]) -> bool {
    args.iter().any(|a| a == "--help" || a == "-h")
}

fn print_main_help() {
    println!("kerr-engine v0.2.0 — wave-native training and inference for Kerr-ODE models");
    println!();
    println!("USAGE:");
    println!("    kerr-engine <COMMAND> [OPTIONS]");
    println!();
    println!("COMMANDS:");
    println!("    train                Train a model from scratch or resume from checkpoint");
    println!("    validate             Validate forward pass against Python reference");
    println!("    grad-test            Validate analytical gradients against PyTorch autograd");
    println!("    list-gpus            List available GPU adapters");
    println!("    gpu-test             Stage 1: single Euler step GPU kernel validation");
    println!("    gpu-backend-test     Validate all GPU primitives against CPU reference");
    println!("    gpu-bench            Benchmark GPU vs CPU per-primitive timing");
    println!("    gpu-persistent-bench Benchmark persistent GPU pipeline vs per-call");
    println!("    backend-select       Test auto-selection logic at various dimensions");
    println!();
    println!("Run 'kerr-engine <COMMAND> --help' for detailed help on any command.");
    println!();
    println!("QUICK START:");
    println!("    # Train on Shakespeare with defaults (char-level, 128-dim, 4 layers)");
    println!("    kerr-engine train data/input.txt");
    println!();
    println!("    # Train with BPE tokenizer (subword vocabulary)");
    println!("    kerr-engine train data/corpus.txt --bpe tokenizer.json");
    println!();
    println!("    # Train a larger model (768-dim, 12 heads)");
    println!("    kerr-engine train data/input.txt 5000 --n-bands 384 --n-head 12");
    println!();
    println!("    # Resume training from a checkpoint");
    println!("    kerr-engine train data/input.txt 5000 --resume checkpoint_iter3000.bin");
    println!();
    println!("ARCHITECTURE:");
    println!("    The Kerr-ODE replaces the standard MLP in transformer blocks with a");
    println!("    nonlinear ODE derived from coupled optical resonators. Frozen harmonic");
    println!("    embeddings encode tokens as phase positions on the unit circle.");
    println!();
    println!("    Block 0: Attention + PerBandLinear (analytical, near-identity)");
    println!("    Blocks 1-N: Attention + KerrMaestroAdd (Kerr-ODE + global maestro sync)");
    println!();
    println!("    At 128-dim, trains 3x faster than PyTorch+CUDA on CPU alone.");
    println!("    At 768-dim, GPU acceleration via WGSL shaders on any vendor hardware.");
    println!();
    println!("SOURCE: https://github.com/atech-hub/kerr-engine");
    println!("PAPER:  https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive");
}

fn print_train_help() {
    println!("kerr-engine train — Train a Kerr-ODE model");
    println!();
    println!("USAGE:");
    println!("    kerr-engine train [DATA] [ITERS] [BATCH] [SEQ_LEN] [LR] [OPTIONS]");
    println!();
    println!("POSITIONAL ARGUMENTS:");
    println!("    DATA         Path to training data text file         [default: data/input.txt]");
    println!("    ITERS        Number of training iterations           [default: 3000]");
    println!("    BATCH        Batch size (parallelised across threads) [default: 4]");
    println!("    SEQ_LEN      Sequence length per sample              [default: 64]");
    println!("    LR           Learning rate for Adam optimizer         [default: 3e-4]");
    println!();
    println!("TOKENIZER:");
    println!("    --word              Word-level tokenization (split on whitespace, lowercase,");
    println!("                        separate punctuation, min_count=3 frequency threshold).");
    println!("                        Vocab is built from the training data.");
    println!();
    println!("    --bpe FILE          BPE subword tokenization. FILE is a HuggingFace");
    println!("                        tokenizer.json (GPT-2, Llama, Qwen, etc). Vocab comes");
    println!("                        from the tokenizer file, not the training data.");
    println!("                        Download from huggingface.co/<model>/raw/main/tokenizer.json");
    println!();
    println!("    (default)           Character-level tokenization. One token per character.");
    println!("                        Vocab is the set of unique characters in the data.");
    println!();
    println!("TRAINING:");
    println!("    --seed N            Model initialisation seed         [default: 42]");
    println!("    --train-seed N      Training RNG seed (sampling)      [default: seed+1295]");
    println!("    --resume FILE       Resume from a .bin checkpoint. Training continues from");
    println!("                        the saved iteration with restored weights, optimizer");
    println!("                        state, and RNG. Supports cross-corpus resume with");
    println!("                        automatic vocab resizing.");
    println!("    --no-curriculum     Disable progressive curriculum. By default, training");
    println!("                        starts with 8 bands and opens to all bands in 3 stages.");
    println!("                        Use this flag to train with all bands from iteration 0.");
    println!();
    println!("COMPUTE:");
    println!("    --threads N         Number of worker threads (batch parallelism).");
    println!("                        Default: auto-detect, capped at batch size.");
    println!("    --cpu               Force CPU backend even at large dimensions.");
    println!("    --gpu               Force GPU backend even at small dimensions.");
    println!("    --gpu-device N      Select GPU adapter by index (see 'list-gpus').");
    println!();
    println!("ARCHITECTURE:");
    println!("    These flags define the model structure. All are stored in v2 checkpoints");
    println!("    and auto-detected on resume. Only needed for initial training.");
    println!();
    println!("    --n-bands N         Harmonic frequency bands. Embedding dim = 2 * N_BANDS.");
    println!("                        64 bands = 128-dim, 384 bands = 768-dim.  [default: 64]");
    println!("    --n-head N          Number of attention heads. Must divide n_embd.");
    println!("                                                                  [default: 4]");
    println!("    --n-layers N        Number of transformer blocks. Block 0 uses PerBandLinear,");
    println!("                        blocks 1+ use Kerr-ODE + Maestro.         [default: 4]");
    println!("    --maestro-dim N     Maestro bottleneck width. Controls global sync capacity.");
    println!("                        16 is optimal for 128-dim.                [default: 16]");
    println!("    --block-size N      Maximum sequence length (context window).  [default: 256]");
    println!("    --rk4-steps N       ODE integration steps per layer. More steps = more");
    println!("                        accurate but slower. 8 is well-validated.  [default: 8]");
    println!();
    println!("OUTPUT:");
    println!("    Checkpoints saved every 500 iterations as checkpoint_iter<N>.bin");
    println!("    Final checkpoint saved as checkpoint_final.bin");
    println!("    Training summary saved as training_summary.json");
    println!("    Validation loss + sample text printed every 300 iterations");
    println!();
    println!("EXAMPLES:");
    println!("    # Minimal: train on Shakespeare with all defaults");
    println!("    kerr-engine train data/input.txt");
    println!();
    println!("    # Custom: 5000 iterations, batch 8, sequence length 128");
    println!("    kerr-engine train data/input.txt 5000 8 128 3e-4");
    println!();
    println!("    # BPE tokenizer with Qwen vocabulary");
    println!("    kerr-engine train data/corpus.txt 10000 --bpe qwen_tokenizer.json");
    println!();
    println!("    # Larger model on GPU");
    println!("    kerr-engine train data/input.txt 5000 --n-bands 384 --n-head 12 --gpu");
    println!();
    println!("    # Resume training on new data");
    println!("    kerr-engine train data/new_corpus.txt 3000 --resume checkpoint_final.bin");
    println!();
    println!("    # Deterministic training with specific seeds");
    println!("    kerr-engine train data/input.txt --seed 123 --train-seed 456");
    println!();
    println!("    # No curriculum, word-level, 4 threads");
    println!("    kerr-engine train data/input.txt --word --no-curriculum --threads 4");
}

fn print_validate_help() {
    println!("kerr-engine validate — Validate forward pass against Python reference");
    println!();
    println!("USAGE:");
    println!("    kerr-engine validate [MODEL] [TEST]");
    println!();
    println!("ARGUMENTS:");
    println!("    MODEL    Path to model weights (.bin)     [default: model.bin]");
    println!("    TEST     Path to test vectors (.bin)      [default: <MODEL>_test.bin]");
    println!();
    println!("Stage 2 validation. Loads a model exported from Python, runs a forward pass");
    println!("on test input tokens, and compares output logits against Python reference.");
    println!("Tolerance: 1e-3 (PASS), <0.01 (CLOSE, acceptable), >=0.01 (FAIL).");
    println!();
    println!("The test vectors file contains input tokens and expected logits, exported");
    println!("from the Python training script using export_model_and_test_vectors().");
    println!();
    println!("EXAMPLES:");
    println!("    kerr-engine validate");
    println!("    kerr-engine validate model.bin");
    println!("    kerr-engine validate model.bin model_test.bin");
}

fn print_grad_test_help() {
    println!("kerr-engine grad-test — Validate analytical gradients against PyTorch autograd");
    println!();
    println!("USAGE:");
    println!("    kerr-engine grad-test [TEST_FILE]");
    println!();
    println!("ARGUMENTS:");
    println!("    TEST_FILE    Path to gradient reference file    [default: reference/gradient_test.bin]");
    println!();
    println!("Stage 3 validation. Loads PyTorch autograd reference gradients and compares");
    println!("against Rust hand-derived analytical gradients. Validates that the backward");
    println!("pass produces correct gradients for all parameter types.");
    println!();
    println!("EXAMPLES:");
    println!("    kerr-engine grad-test");
    println!("    kerr-engine grad-test reference/gradient_test.bin");
}

fn print_list_gpus_help() {
    println!("kerr-engine list-gpus — List available GPU adapters");
    println!();
    println!("USAGE:");
    println!("    kerr-engine list-gpus");
    println!();
    println!("Enumerates all GPU adapters visible to WGPU (Vulkan, DirectX 12, Metal).");
    println!("Shows adapter name, backend, and device type for each.");
    println!("Use the index with --gpu-device N when training to select a specific GPU.");
    println!();
    println!("EXAMPLE OUTPUT:");
    println!("    [0] NVIDIA GeForce RTX 4070 Ti (Vulkan)");
    println!("        Device type: DiscreteGpu");
    println!("    [1] NVIDIA GeForce RTX 4070 Ti (Dx12)");
    println!("        Device type: DiscreteGpu");
}

fn print_gpu_test_help() {
    println!("kerr-engine gpu-test — Stage 1 GPU kernel validation");
    println!();
    println!("USAGE:");
    println!("    kerr-engine gpu-test");
    println!();
    println!("Runs a single Kerr-ODE Euler step on GPU and compares against CPU reference.");
    println!("This is the most basic GPU validation — confirms that the WGSL shader produces");
    println!("correct results for the core computation. Tolerance: 1e-5.");
    println!();
    println!("Uses 64 bands with dt=0.01, gamma=0.1. No model or data required.");
}

fn print_gpu_backend_test_help() {
    println!("kerr-engine gpu-backend-test — Validate all GPU primitives against CPU");
    println!();
    println!("USAGE:");
    println!("    kerr-engine gpu-backend-test");
    println!();
    println!("Runs every ComputeBackend operation (linear, layer_norm, GELU, kerr_ode,");
    println!("attention, backward passes) on both GPU and CPU, comparing results.");
    println!("Tolerance: 1e-4 per primitive. This validates the full GpuBackend.");
    println!();
    println!("No model or data required. Uses randomly generated test inputs.");
}

fn print_gpu_bench_help() {
    println!("kerr-engine gpu-bench — Benchmark GPU vs CPU per-primitive timing");
    println!();
    println!("USAGE:");
    println!("    kerr-engine gpu-bench");
    println!();
    println!("Times every ComputeBackend operation on GPU vs CPU at multiple dimensions.");
    println!("Shows per-primitive speedup/slowdown and identifies the crossover point");
    println!("where GPU becomes faster than CPU.");
    println!();
    println!("At 128-dim, CPU wins (GPU dispatch overhead dominates).");
    println!("At 768-dim+, GPU wins (O(n^2) matmul overtakes fixed dispatch cost).");
}

fn print_gpu_persistent_bench_help() {
    println!("kerr-engine gpu-persistent-bench — Benchmark persistent GPU pipeline");
    println!();
    println!("USAGE:");
    println!("    kerr-engine gpu-persistent-bench");
    println!();
    println!("Benchmarks the persistent GPU pipeline (weights uploaded once, scratch buffers");
    println!("pre-allocated, single command encoder submit) against per-call GPU and CPU.");
    println!("Shows the theoretical maximum GPU speedup when dispatch overhead is eliminated.");
    println!();
    println!("Persistent pipeline is ~296x faster than per-call GPU for Kerr-ODE at 128-dim.");
    println!("CPU still wins at 128-dim. Estimated crossover: ~700-800 dim.");
}

fn print_backend_select_help() {
    println!("kerr-engine backend-select — Test auto-selection logic");
    println!();
    println!("USAGE:");
    println!("    kerr-engine backend-select");
    println!();
    println!("Shows which compute backend (CPU or GPU) would be auto-selected at various");
    println!("embedding dimensions (128, 256, 512, 768, 1024). Also shows the effect of");
    println!("--cpu and --gpu override flags.");
    println!();
    println!("Default threshold: CPU below 768-dim, GPU at 768-dim and above.");
}

// ─── Train command ────────────────────────────────────────────

fn run_train(args: &[String]) {
    let use_curriculum = !args.iter().any(|a| a == "--no-curriculum");
    let word_level = args.iter().any(|a| a == "--word");
    let force_cpu = args.iter().any(|a| a == "--cpu");
    let force_gpu = args.iter().any(|a| a == "--gpu");
    let resume_path = args.iter()
        .position(|a| a == "--resume")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.to_string());
    let seed: u64 = args.iter()
        .position(|a| a == "--seed")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let train_seed: u64 = args.iter()
        .position(|a| a == "--train-seed")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(seed + 1295);
    let threads: Option<usize> = args.iter()
        .position(|a| a == "--threads")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());
    let gpu_device: Option<usize> = args.iter()
        .position(|a| a == "--gpu-device")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());
    // Architecture flags
    let n_bands: usize = args.iter()
        .position(|a| a == "--n-bands")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let n_head: usize = args.iter()
        .position(|a| a == "--n-head")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let n_layers: usize = args.iter()
        .position(|a| a == "--n-layers")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let maestro_dim: usize = args.iter()
        .position(|a| a == "--maestro-dim")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let block_size: usize = args.iter()
        .position(|a| a == "--block-size")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let rk4_steps: usize = args.iter()
        .position(|a| a == "--rk4-steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let bpe_path: Option<String> = args.iter()
        .position(|a| a == "--bpe")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.to_string());
    let model_config = model::ModelConfig {
        n_bands, n_head, n_layers, maestro_dim, block_size,
        rk4_n_steps: rk4_steps,
    };
    // Collect positional args, skipping flags and their values
    let flags_with_values = [
        "--resume", "--seed", "--train-seed", "--threads",
        "--n-bands", "--n-head", "--n-layers", "--maestro-dim",
        "--block-size", "--rk4-steps", "--gpu-device", "--bpe",
    ];
    let mut positional: Vec<&str> = Vec::new();
    let mut skip_next = false;
    for a in &args[2..] {
        if skip_next { skip_next = false; continue; }
        if flags_with_values.contains(&a.as_str()) { skip_next = true; continue; }
        if a.starts_with("--") { continue; }
        positional.push(a);
    }
    let data_path = positional.first().copied().unwrap_or("data/input.txt");
    let n_iters: usize = positional.get(1).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let batch_size: usize = positional.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let seq_len: usize = positional.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let lr: f32 = positional.get(4).and_then(|s| s.parse().ok()).unwrap_or(3e-4);
    train::train_with_config(train::TrainConfig {
        data_path: data_path.to_string(),
        n_iters,
        batch_size,
        seq_len,
        lr,
        use_curriculum,
        word_level,
        bpe_path,
        resume_path,
        checkpoint_every: 500,
        model_seed: seed,
        train_seed,
        threads,
        force_cpu,
        force_gpu,
        gpu_device,
        model_config,
    });
}

// ─── Validation functions ─────────────────────────────────────

fn gpu_kernel_test() {
    println!("Stage 1 — single Euler step GPU validation\n");

    let n_bands: u32 = 64;
    let dt: f32 = 0.01;
    let gamma: f32 = 0.1;

    let mut input = vec![0.0f32; n_bands as usize * 2];
    for k in 0..n_bands as usize {
        input[k * 2] = (k as f32).cos();
        input[k * 2 + 1] = (k as f32).sin();
    }

    // CPU reference (simple Euler from Stage 1)
    let cpu_output = cpu_euler_step(&input, n_bands as usize, dt, gamma);

    println!("Initializing GPU...");
    let kern = gpu::KernGpu::new(n_bands);

    println!("Running Kerr-ODE step on GPU...");
    let gpu_output = kern.step(&input, dt, gamma);

    println!("\nValidation (CPU vs GPU):");
    let mut max_diff: f32 = 0.0;
    for i in 0..input.len() {
        max_diff = max_diff.max((cpu_output[i] - gpu_output[i]).abs());
    }
    println!("  Max absolute difference: {:.2e}", max_diff);

    if max_diff < 1e-5 {
        println!("  PASS. Validation gate cleared.");
    } else {
        println!("  FAIL. Max diff = {:.2e}", max_diff);
    }
}

fn validate_forward_pass(weights_path: &str, test_path: &str) {
    println!("Stage 2 — full forward pass validation\n");

    // Load weights
    println!("Loading weights from {weights_path}...");
    let model = weights::load_weights(weights_path)
        .expect("Failed to load weights");

    // Load test vectors
    println!("Loading test vectors from {test_path}...");
    let test = weights::load_test_vectors(test_path)
        .expect("Failed to load test vectors");

    println!("  Tokens: {:?}...", &test.tokens[..8.min(test.tokens.len())]);
    println!("  Expected logits shape: [{}][{}]", test.tokens.len(), test.expected_logits[0].len());

    // Run forward pass
    println!("\nRunning forward pass...");
    let start = std::time::Instant::now();
    let logits = model.forward(&test.tokens);
    let elapsed = start.elapsed();
    println!("  Forward pass took: {:.1?}", elapsed);

    // Compare
    println!("\nValidation (Rust vs Python):");
    let mut max_diff: f32 = 0.0;
    let mut total_diff: f64 = 0.0;
    let mut n_values = 0usize;

    for (pos, (rust_logits, py_logits)) in logits.iter().zip(test.expected_logits.iter()).enumerate() {
        for (i, (&r, &p)) in rust_logits.iter().zip(py_logits.iter()).enumerate() {
            let diff = (r - p).abs();
            if diff > max_diff {
                max_diff = diff;
                if diff > 0.01 {
                    println!("  Large diff at pos={pos} vocab={i}: rust={r:.6} python={p:.6} diff={diff:.6}");
                }
            }
            total_diff += diff as f64;
            n_values += 1;
        }
    }

    let mean_diff = total_diff / n_values as f64;
    println!("\n  Max absolute difference:  {:.2e}", max_diff);
    println!("  Mean absolute difference: {:.2e}", mean_diff);

    // Show first position logits comparison
    if !logits.is_empty() {
        println!("\n  Position 0, first 5 logit values:");
        println!("  {:>5} {:>12} {:>12} {:>10}", "Vocab", "Rust", "Python", "Diff");
        for i in 0..5.min(logits[0].len()) {
            let r = logits[0][i];
            let p = test.expected_logits[0][i];
            println!("  {:>5} {:>12.6} {:>12.6} {:>10.2e}", i, r, p, (r - p).abs());
        }
    }

    // Verdict
    let tolerance = 1e-3;  // f32 accumulation across layers allows more drift
    println!();
    if max_diff < tolerance {
        println!("  PASS: Rust matches Python within {:.0e} tolerance.", tolerance);
        println!("  Validation gate CLEARED. Stage 2 complete.");
    } else if max_diff < 0.01 {
        println!("  CLOSE: Max diff {:.2e} is small but above {:.0e} tolerance.", max_diff, tolerance);
        println!("  Likely f32 accumulation drift. Acceptable for inference.");
    } else {
        println!("  FAIL: Rust differs from Python by {:.2e}.", max_diff);
        println!("  Validation gate NOT cleared. Debug required.");
    }
}

fn list_gpus() {
    println!("Available GPU adapters:\n");
    let instance = wgpu::Instance::default();
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    if adapters.is_empty() {
        println!("  No GPU adapters found.");
        return;
    }
    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        println!("  [{}] {} ({:?})", i, info.name, info.backend);
        println!("      Device type: {:?}", info.device_type);
    }
    println!("\nUse --gpu-device N to select a specific adapter.");
}

/// Simple Euler step for Stage 1 GPU validation (not the full Kerr-ODE).
fn cpu_euler_step(input: &[f32], n_bands: usize, dt: f32, gamma: f32) -> Vec<f32> {
    let mut output = vec![0.0f32; n_bands * 2];
    for band in 0..n_bands {
        let re = input[band * 2];
        let im = input[band * 2 + 1];
        let (left_re, left_im) = if band > 0 {
            (input[(band - 1) * 2], input[(band - 1) * 2 + 1])
        } else { (0.0, 0.0) };
        let (right_re, right_im) = if band < n_bands - 1 {
            (input[(band + 1) * 2], input[(band + 1) * 2 + 1])
        } else { (0.0, 0.0) };
        let coupling = (left_re * left_re + left_im * left_im)
                     + (right_re * right_re + right_im * right_im);
        let dre = -gamma * coupling * im;
        let dim = gamma * coupling * re;
        output[band * 2] = re + dt * dre;
        output[band * 2 + 1] = im + dt * dim;
    }
    output
}
