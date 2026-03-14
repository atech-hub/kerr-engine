//! GPU backend validation and benchmarking.
//!
//! validate_gpu_backend() — correctness check against CpuBackend.
//! benchmark_gpu_vs_cpu() — per-primitive timing comparison.

use crate::backend::ComputeBackend;
use crate::gpu_pipelines::GpuBackend;
use crate::model::*;

/// Validate GPU backend against CPU backend on all primitives.
pub fn validate_gpu_backend() {
    use crate::backend::CpuBackend;

    println!("GPU Backend Validation\n");

    let gpu = GpuBackend::new();
    let cpu = CpuBackend;

    // Test 1: Linear (128→128 with bias)
    print!("  linear (128→128, bias)... ");
    {
        let in_dim = 128;
        let out_dim = 128;
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let w: Vec<Vec<f32>> = (0..out_dim).map(|i| {
            (0..in_dim).map(|j| ((i * in_dim + j) as f32 * 0.001).cos()).collect()
        }).collect();
        let b: Vec<f32> = (0..out_dim).map(|i| i as f32 * 0.01).collect();

        let cpu_y = cpu.linear(&w, &b, &x);
        let gpu_y = gpu.linear(&w, &b, &x);

        let max_diff = cpu_y.iter().zip(&gpu_y).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        if max_diff < 1e-4 {
            println!("PASS (max_diff={max_diff:.2e})");
        } else {
            println!("FAIL (max_diff={max_diff:.2e})");
        }
    }

    // Test 2: Linear no bias (384→128 — QKV projection size)
    print!("  linear_no_bias (384→128)... ");
    {
        let in_dim = 128;
        let out_dim = 384;
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.03).cos()).collect();
        let w: Vec<Vec<f32>> = (0..out_dim).map(|i| {
            (0..in_dim).map(|j| ((i * in_dim + j) as f32 * 0.0007).sin()).collect()
        }).collect();

        let cpu_y = cpu.linear_no_bias(&w, &x);
        let gpu_y = gpu.linear_no_bias(&w, &x);

        let max_diff = cpu_y.iter().zip(&gpu_y).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        if max_diff < 1e-4 {
            println!("PASS (max_diff={max_diff:.2e})");
        } else {
            println!("FAIL (max_diff={max_diff:.2e})");
        }
    }

    // Test 3: Layer norm (dim=128)
    print!("  layer_norm (dim=128)... ");
    {
        let dim = 128;
        let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.05).sin() * 2.0).collect();
        let weight: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32 * 0.01).cos() * 0.1).collect();
        let bias: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).sin() * 0.05).collect();

        let cpu_y = cpu.layer_norm(&x, &weight, &bias);
        let gpu_y = gpu.layer_norm(&x, &weight, &bias);

        let max_diff = cpu_y.iter().zip(&gpu_y).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        if max_diff < 1e-4 {
            println!("PASS (max_diff={max_diff:.2e})");
        } else {
            println!("FAIL (max_diff={max_diff:.2e})");
        }
    }

    // Test 4: Kerr-ODE (full RK4, 8 steps)
    print!("  kerr_ode (RK4, 8 steps)... ");
    {
        let mut x = vec![0.0f32; N_EMBD];
        for k in 0..N_BANDS {
            x[k * 2] = (k as f32 * 0.1).cos() * 0.5;
            x[k * 2 + 1] = (k as f32 * 0.1).sin() * 0.5;
        }
        let weights = KerrWeights {
            gamma_raw: (0..N_BANDS).map(|k| -2.0 + k as f32 * 0.05).collect(), // softplus → ~0.1
            omega: (0..N_BANDS).map(|k| k as f32 / N_BANDS as f32).collect(),
            alpha: 0.1,
            beta: 0.1,
            rk4_n_steps: RK4_N_STEPS,
        };

        let cpu_y = cpu.kerr_ode(&weights, &x);
        let gpu_y = gpu.kerr_ode(&weights, &x);

        let max_diff = cpu_y.iter().zip(&gpu_y).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        if max_diff < 1e-4 {
            println!("PASS (max_diff={max_diff:.2e})");
        } else {
            println!("FAIL (max_diff={max_diff:.2e})");
        }
    }

    println!("\nGPU backend validation complete.");
}

/// Benchmark GPU vs CPU on all primitives. Runs each operation many times
/// and reports median timing. This tells us whether GPU is worth wiring
/// into the training loop at our current scale (128-dim).
pub fn benchmark_gpu_vs_cpu() {
    use crate::backend::CpuBackend;

    println!("GPU vs CPU Benchmark\n");

    let gpu = GpuBackend::new();
    let cpu = CpuBackend;

    // ─── Shared test data ───────────────────────────────────────

    // Linear 128→128 (output projection, maestro layers)
    let in_dim = 128;
    let out_dim = 128;
    let x128: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let w128: Vec<Vec<f32>> = (0..out_dim).map(|i| {
        (0..in_dim).map(|j| ((i * in_dim + j) as f32 * 0.001).cos()).collect()
    }).collect();
    let b128: Vec<f32> = (0..out_dim).map(|i| i as f32 * 0.01).collect();

    // Linear 384→128 (QKV attention projection)
    let qkv_out = 384;
    let w384: Vec<Vec<f32>> = (0..qkv_out).map(|i| {
        (0..in_dim).map(|j| ((i * in_dim + j) as f32 * 0.0007).sin()).collect()
    }).collect();
    let b384: Vec<f32> = (0..qkv_out).map(|i| i as f32 * 0.005).collect();

    // Layer norm
    let ln_x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin() * 2.0).collect();
    let ln_w: Vec<f32> = (0..128).map(|i| 1.0 + (i as f32 * 0.01).cos() * 0.1).collect();
    let ln_b: Vec<f32> = (0..128).map(|i| (i as f32 * 0.02).sin() * 0.05).collect();

    // Kerr-ODE
    let mut kerr_x = vec![0.0f32; N_EMBD];
    for k in 0..N_BANDS {
        kerr_x[k * 2] = (k as f32 * 0.1).cos() * 0.5;
        kerr_x[k * 2 + 1] = (k as f32 * 0.1).sin() * 0.5;
    }
    let kerr_w = KerrWeights {
        gamma_raw: (0..N_BANDS).map(|k| -2.0 + k as f32 * 0.05).collect(),
        omega: (0..N_BANDS).map(|k| k as f32 / N_BANDS as f32).collect(),
        alpha: 0.1,
        beta: 0.1,
        rk4_n_steps: RK4_N_STEPS,
    };

    // Maestro
    let maestro_w = MaestroWeights {
        squeeze: LinearWeights {
            w: (0..MAESTRO_DIM).map(|i| {
                (0..N_EMBD).map(|j| ((i * N_EMBD + j) as f32 * 0.002).sin()).collect()
            }).collect(),
            b: vec![0.01; MAESTRO_DIM],
        },
        process_1: LinearWeights {
            w: (0..N_EMBD).map(|i| {
                (0..MAESTRO_DIM).map(|j| ((i * MAESTRO_DIM + j) as f32 * 0.003).cos()).collect()
            }).collect(),
            b: vec![0.01; N_EMBD],
        },
    };

    // ─── Warmup (GPU shader compilation, buffer caching) ────────

    print!("  Warmup...");
    for _ in 0..5 {
        let _ = gpu.linear(&w128, &b128, &x128);
        let _ = gpu.layer_norm(&ln_x, &ln_w, &ln_b);
    }
    println!(" done\n");

    let n_iters = 200;

    println!("  {:>30} {:>12} {:>12} {:>8}", "Operation", "CPU (us)", "GPU (us)", "Ratio");
    println!("  {}", "-".repeat(66));

    // ─── Benchmark each primitive ───────────────────────────────

    // Linear 128→128
    let cpu_us = bench(n_iters, || { let _ = cpu.linear(&w128, &b128, &x128); });
    let gpu_us = bench(n_iters, || { let _ = gpu.linear(&w128, &b128, &x128); });
    print_row("linear (128→128, bias)", cpu_us, gpu_us);

    // Linear 384→128 (QKV)
    let cpu_us = bench(n_iters, || { let _ = cpu.linear(&w384, &b384, &x128); });
    let gpu_us = bench(n_iters, || { let _ = gpu.linear(&w384, &b384, &x128); });
    print_row("linear (384→128, QKV)", cpu_us, gpu_us);

    // Linear no bias 128→128
    let cpu_us = bench(n_iters, || { let _ = cpu.linear_no_bias(&w128, &x128); });
    let gpu_us = bench(n_iters, || { let _ = gpu.linear_no_bias(&w128, &x128); });
    print_row("linear_no_bias (128→128)", cpu_us, gpu_us);

    // Layer norm
    let cpu_us = bench(n_iters, || { let _ = cpu.layer_norm(&ln_x, &ln_w, &ln_b); });
    let gpu_us = bench(n_iters, || { let _ = gpu.layer_norm(&ln_x, &ln_w, &ln_b); });
    print_row("layer_norm (dim=128)", cpu_us, gpu_us);

    // Kerr-ODE (full RK4, 8 steps × 4 derivative evals = 32 dispatches)
    let cpu_us = bench(n_iters / 4, || { let _ = cpu.kerr_ode(&kerr_w, &kerr_x); });
    let gpu_us = bench(n_iters / 4, || { let _ = gpu.kerr_ode(&kerr_w, &kerr_x); });
    print_row("kerr_ode (RK4, 8 steps)", cpu_us, gpu_us);

    // Maestro (squeeze + GELU + process = 2 linear + activation)
    let cpu_us = bench(n_iters, || { let _ = cpu.maestro(&maestro_w, &kerr_x); });
    let gpu_us = bench(n_iters, || { let _ = gpu.maestro(&maestro_w, &kerr_x); });
    print_row("maestro (128→16→128)", cpu_us, gpu_us);

    // ─── Composite: simulate one forward position ───────────────
    // One position through block 1-3: layer_norm + attn_proj(QKV) + attn_proj(out) +
    //   layer_norm + kerr_ode + maestro + out_proj
    // That's: 2 layer_norms + 3 linears (QKV, out_proj, kerr out_proj) + kerr + maestro

    println!("\n  {:>30} {:>12} {:>12} {:>8}", "Composite", "CPU (us)", "GPU (us)", "Ratio");
    println!("  {}", "-".repeat(66));

    let cpu_us = bench(n_iters / 4, || {
        let _ = cpu.layer_norm(&ln_x, &ln_w, &ln_b);
        let _ = cpu.linear(&w384, &b384, &x128);   // QKV
        let _ = cpu.linear(&w128, &b128, &x128);   // attn out_proj
        let _ = cpu.layer_norm(&ln_x, &ln_w, &ln_b);
        let _ = cpu.kerr_ode(&kerr_w, &kerr_x);
        let _ = cpu.maestro(&maestro_w, &kerr_x);
        let _ = cpu.linear(&w128, &b128, &x128);   // block out_proj
    });
    let gpu_us = bench(n_iters / 4, || {
        let _ = gpu.layer_norm(&ln_x, &ln_w, &ln_b);
        let _ = gpu.linear(&w384, &b384, &x128);
        let _ = gpu.linear(&w128, &b128, &x128);
        let _ = gpu.layer_norm(&ln_x, &ln_w, &ln_b);
        let _ = gpu.kerr_ode(&kerr_w, &kerr_x);
        let _ = gpu.maestro(&maestro_w, &kerr_x);
        let _ = gpu.linear(&w128, &b128, &x128);
    });
    print_row("one block (1 position)", cpu_us, gpu_us);

    println!();
    println!("  Ratio < 1.0 = GPU wins, > 1.0 = CPU wins at this scale.");
    println!("  GPU dispatch overhead is fixed ~50-200us per call.");
    println!("  At 128-dim, CPU compute per call is comparable to dispatch cost.");
}

fn bench(n: usize, mut f: impl FnMut()) -> f64 {
    let mut times = Vec::with_capacity(n);
    for _ in 0..n {
        let start = std::time::Instant::now();
        f();
        times.push(start.elapsed().as_micros() as f64);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Median
    times[times.len() / 2]
}

fn print_row(name: &str, cpu_us: f64, gpu_us: f64) {
    let ratio = gpu_us / cpu_us;
    let marker = if ratio < 1.0 { " <GPU" } else { "" };
    println!("  {:>30} {:>10.0} {:>10.0} {:>7.2}x{}", name, cpu_us, gpu_us, ratio, marker);
}
