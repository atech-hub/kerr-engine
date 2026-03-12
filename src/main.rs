//! Kerr Engine — wave-native compute for Kerr-ODE architecture
//!
//! Stage 1: Single Euler step GPU validation (COMPLETE)
//! Stage 2: Full forward pass, load Python weights, validate against Python logits.

mod backward;
mod gpu;
mod grad_test;
mod model;
mod train;
mod weights;

use std::env;

fn main() {
    println!("kerr-engine v0.1.0\n");

    let args: Vec<String> = env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("validate") => {
            // Stage 2: validate full forward pass against Python reference
            let weights_path = args.get(2).map(|s| s.as_str()).unwrap_or("model.bin");
            let default_test = weights_path.replace(".bin", "_test.bin");
            let test_path = args.get(3).map(|s| s.as_str()).unwrap_or(&default_test);
            validate_forward_pass(weights_path, test_path);
        }
        Some("gpu-test") => {
            // Stage 1: GPU kernel validation
            gpu_kernel_test();
        }
        Some("grad-test") => {
            // Stage 3: gradient validation
            let test_path = args.get(2).map(|s| s.as_str()).unwrap_or("reference/gradient_test.bin");
            grad_test::validate_gradients(test_path);
        }
        Some("train") => {
            // Stage 4: training from scratch
            let data_path = args.get(2).map(|s| s.as_str()).unwrap_or("data/input.txt");
            let n_iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3000);
            let batch_size: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);
            let seq_len: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(64);
            let lr: f32 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(3e-4);
            train::train(data_path, n_iters, batch_size, seq_len, lr);
        }
        _ => {
            println!("Usage:");
            println!("  kerr-engine gpu-test              Stage 1: GPU kernel validation");
            println!("  kerr-engine validate <model.bin>  Stage 2: full forward pass validation");
            println!("  kerr-engine grad-test [test.bin]  Stage 3: gradient validation");
            println!("  kerr-engine train [data] [iters] [batch] [seq_len] [lr]");
            println!("                                    Stage 4: train from scratch");
        }
    }
}

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
