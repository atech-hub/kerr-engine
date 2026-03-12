//! Kerr Engine — wave-native compute for Kerr-ODE architecture
//!
//! Stage 1: Single layer forward pass, 64 bands, validate against Python.

mod gpu;

fn main() {
    println!("kerr-engine v0.1.0");
    println!("Stage 1 — single layer Kerr-ODE forward pass");

    // TODO: Stage 1 implementation
    // 1. Initialize WGPU device
    // 2. Load or generate test amplitudes (64 bands = 128 f32 values)
    // 3. Dispatch kerr_step kernel (4x for RK4, or 1x Euler for initial validation)
    // 4. Read back results
    // 5. Compare against Python reference values
    println!("Not yet implemented. Kernel is in kernels/kerr-ode/src/lib.rs");
}
