//! Kerr-ODE GPU compute kernel
//!
//! One Euler step of the Kerr-ODE: dZ_k/dt = i * gamma * |Z_neighbours|^2 * Z_k
//! where Z_k is the complex amplitude of band k, and neighbours are bands k-1, k+1.
//!
//! Data layout: amplitudes are stored as interleaved [re_0, im_0, re_1, im_1, ...]
//! so band k has real part at index 2*k and imaginary part at 2*k+1.

#![no_std]

use spirv_std::spirv;

/// Single Euler step of the Kerr-ODE for one band.
///
/// Each thread handles one band. Reads neighbour amplitudes, computes
/// |Z_left|^2 + |Z_right|^2 coupling, applies the ODE step.
///
/// Parameters (in `params` buffer):
///   [0] = dt (step size)
///   [1] = gamma (Kerr nonlinearity coefficient)
///   [2] = n_bands (number of bands, as f32)
#[spirv(compute(threads(64)))]
pub fn kerr_step(
    #[spirv(global_invocation_id)] id: spirv_std::glam::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] params: &[f32],
) {
    let band = id.x as usize;
    let dt = params[0];
    let gamma = params[1];
    let n_bands = params[2] as usize;

    if band >= n_bands {
        return;
    }

    // Current band complex amplitude
    let re = input[band * 2];
    let im = input[band * 2 + 1];

    // Neighbour amplitudes (clamped at boundaries)
    let (left_re, left_im) = if band > 0 {
        (input[(band - 1) * 2], input[(band - 1) * 2 + 1])
    } else {
        (0.0_f32, 0.0_f32)
    };

    let (right_re, right_im) = if band < n_bands - 1 {
        (input[(band + 1) * 2], input[(band + 1) * 2 + 1])
    } else {
        (0.0_f32, 0.0_f32)
    };

    // |Z_left|^2 + |Z_right|^2
    let coupling = (left_re * left_re + left_im * left_im)
                 + (right_re * right_re + right_im * right_im);

    // dZ/dt = i * gamma * coupling * Z
    // i * Z = i * (re + i*im) = -im + i*re
    let dre = -gamma * coupling * im;
    let dim = gamma * coupling * re;

    // Euler step
    output[band * 2] = re + dt * dre;
    output[band * 2 + 1] = im + dt * dim;
}
