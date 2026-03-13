//! Gradient validation: compare Rust backward pass against PyTorch reference.

use std::io::{self, Read, BufReader};
use std::fs::File;
use crate::backward::*;
use crate::model::*;

struct BinReader {
    reader: BufReader<File>,
}

impl BinReader {
    fn new(path: &str) -> io::Result<Self> {
        Ok(Self { reader: BufReader::new(File::open(path)?) })
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
        (0..n).map(|_| self.read_f32()).collect()
    }
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn mean_diff(a: &[f32], b: &[f32]) -> f32 {
    let sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs() as f64).sum();
    (sum / a.len() as f64) as f32
}

pub fn validate_gradients(test_path: &str) {
    println!("Stage 3 — gradient validation\n");

    let mut r = BinReader::new(test_path).expect("Failed to open gradient test file");

    let magic = r.read_u32().unwrap();
    assert_eq!(magic, 0x47524144, "Bad magic");
    let version = r.read_u32().unwrap();
    assert_eq!(version, 1);

    // ─── Test 1: Kerr derivative backward ───────────────────────
    let test_id = r.read_u32().unwrap();
    assert_eq!(test_id, 1);

    // Read inputs (squeeze out batch dimension — PyTorch uses [1, N_BANDS])
    let py_r = r.read_f32_vec(N_BANDS).unwrap();
    let py_s = r.read_f32_vec(N_BANDS).unwrap();
    let py_gamma_raw = r.read_f32_vec(N_BANDS).unwrap();
    let py_omega = r.read_f32_vec(N_BANDS).unwrap();
    let py_alpha = r.read_f32().unwrap();
    let py_beta = r.read_f32().unwrap();
    let py_d_dr = r.read_f32_vec(N_BANDS).unwrap();
    let py_d_ds = r.read_f32_vec(N_BANDS).unwrap();

    // Expected gradients
    let expected_d_r = r.read_f32_vec(N_BANDS).unwrap();
    let expected_d_s = r.read_f32_vec(N_BANDS).unwrap();
    let expected_d_gamma_raw = r.read_f32_vec(N_BANDS).unwrap();
    let expected_d_omega = r.read_f32_vec(N_BANDS).unwrap();
    let expected_d_alpha = r.read_f32().unwrap();
    let expected_d_beta = r.read_f32().unwrap();

    // Compute gamma = softplus(gamma_raw)
    let gamma: Vec<f32> = py_gamma_raw.iter().map(|&g| {
        if g > 20.0 { g } else { (1.0 + g.exp()).ln() }
    }).collect();

    // Run Rust backward for derivative
    // First, forward to build cache
    let (_dr, _ds, cache) = crate::backward::kerr_derivative_with_cache_pub(
        &py_r, &py_s, &gamma, &py_omega, py_alpha, py_beta,
    );

    let (rust_d_r, rust_d_s, rust_d_gamma, rust_d_omega, rust_d_alpha, rust_d_beta) =
        crate::backward::kerr_derivative_backward_pub(
            &py_d_dr, &py_d_ds, &cache, &gamma, py_alpha, py_beta,
        );

    // Chain through softplus for gamma_raw
    let rust_d_gamma_raw: Vec<f32> = (0..N_BANDS)
        .map(|k| softplus_backward(rust_d_gamma[k], py_gamma_raw[k]))
        .collect();

    println!("Test 1: Kerr derivative backward");
    println!("  d_r:         max_diff={:.2e}  mean_diff={:.2e}",
             max_diff(&rust_d_r, &expected_d_r), mean_diff(&rust_d_r, &expected_d_r));
    println!("  d_s:         max_diff={:.2e}  mean_diff={:.2e}",
             max_diff(&rust_d_s, &expected_d_s), mean_diff(&rust_d_s, &expected_d_s));
    println!("  d_gamma_raw: max_diff={:.2e}  mean_diff={:.2e}",
             max_diff(&rust_d_gamma_raw, &expected_d_gamma_raw),
             mean_diff(&rust_d_gamma_raw, &expected_d_gamma_raw));
    println!("  d_omega:     max_diff={:.2e}  mean_diff={:.2e}",
             max_diff(&rust_d_omega, &expected_d_omega), mean_diff(&rust_d_omega, &expected_d_omega));
    println!("  d_alpha:     diff={:.2e}", (rust_d_alpha - expected_d_alpha).abs());
    println!("  d_beta:      diff={:.2e}", (rust_d_beta - expected_d_beta).abs());

    let test1_max = [
        max_diff(&rust_d_r, &expected_d_r),
        max_diff(&rust_d_s, &expected_d_s),
        max_diff(&rust_d_gamma_raw, &expected_d_gamma_raw),
        max_diff(&rust_d_omega, &expected_d_omega),
        (rust_d_alpha - expected_d_alpha).abs(),
        (rust_d_beta - expected_d_beta).abs(),
    ].iter().cloned().fold(0.0f32, f32::max);

    if test1_max < 1e-4 {
        println!("  PASS (max diff {:.2e})\n", test1_max);
    } else {
        println!("  FAIL (max diff {:.2e})\n", test1_max);
    }

    // ─── Test 2: Full Kerr-ODE backward ─────────────────────────
    let test_id = r.read_u32().unwrap();
    assert_eq!(test_id, 2);

    let py_input = r.read_f32_vec(N_EMBD).unwrap();
    let py_ode_gamma_raw = r.read_f32_vec(N_BANDS).unwrap();
    let py_ode_omega = r.read_f32_vec(N_BANDS).unwrap();
    let py_ode_alpha = r.read_f32().unwrap();
    let py_ode_beta = r.read_f32().unwrap();
    let py_d_output = r.read_f32_vec(N_EMBD).unwrap();

    let expected_ode_d_input = r.read_f32_vec(N_EMBD).unwrap();
    let expected_ode_d_gamma_raw = r.read_f32_vec(N_BANDS).unwrap();
    let expected_ode_d_omega = r.read_f32_vec(N_BANDS).unwrap();
    let expected_ode_d_alpha = r.read_f32().unwrap();
    let expected_ode_d_beta = r.read_f32().unwrap();

    let weights = KerrWeights {
        gamma_raw: py_ode_gamma_raw,
        omega: py_ode_omega,
        alpha: py_ode_alpha,
        beta: py_ode_beta,
        rk4_n_steps: RK4_N_STEPS,
    };

    let (rust_d_input, rust_ode_d_gamma_raw, rust_ode_d_omega, rust_ode_d_alpha, rust_ode_d_beta) =
        kerr_ode_backward(&py_d_output, &py_input, &weights);

    println!("Test 2: Full Kerr-ODE backward (8 RK4 steps)");
    println!("  d_input:     max_diff={:.2e}  mean_diff={:.2e}",
             max_diff(&rust_d_input, &expected_ode_d_input),
             mean_diff(&rust_d_input, &expected_ode_d_input));
    println!("  d_gamma_raw: max_diff={:.2e}  mean_diff={:.2e}",
             max_diff(&rust_ode_d_gamma_raw, &expected_ode_d_gamma_raw),
             mean_diff(&rust_ode_d_gamma_raw, &expected_ode_d_gamma_raw));
    println!("  d_omega:     max_diff={:.2e}  mean_diff={:.2e}",
             max_diff(&rust_ode_d_omega, &expected_ode_d_omega),
             mean_diff(&rust_ode_d_omega, &expected_ode_d_omega));
    println!("  d_alpha:     diff={:.2e}", (rust_ode_d_alpha - expected_ode_d_alpha).abs());
    println!("  d_beta:      diff={:.2e}", (rust_ode_d_beta - expected_ode_d_beta).abs());

    let test2_max = [
        max_diff(&rust_d_input, &expected_ode_d_input),
        max_diff(&rust_ode_d_gamma_raw, &expected_ode_d_gamma_raw),
        max_diff(&rust_ode_d_omega, &expected_ode_d_omega),
        (rust_ode_d_alpha - expected_ode_d_alpha).abs(),
        (rust_ode_d_beta - expected_ode_d_beta).abs(),
    ].iter().cloned().fold(0.0f32, f32::max);

    if test2_max < 1e-2 {
        println!("  PASS (max diff {:.2e})\n", test2_max);
    } else {
        println!("  FAIL (max diff {:.2e})\n", test2_max);
    }

    // Overall verdict
    println!("Stage 3 validation gate:");
    if test1_max < 1e-4 && test2_max < 1e-2 {
        println!("  PASS. Analytical gradients match PyTorch autograd.");
    } else {
        println!("  NOT YET. Debug needed.");
        if test1_max >= 1e-4 { println!("  - Derivative backward has issues"); }
        if test2_max >= 1e-2 { println!("  - Full ODE backward has issues (may be accumulation)"); }
    }
}
