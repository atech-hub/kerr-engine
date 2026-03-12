// Kerr-ODE GPU compute kernel — FULL derivative
//
// Derivative: dr = -gamma*r - phi*s
//             ds = -gamma*s + phi*r
// where:
//   phi = omega + alpha*mag_sq + beta*ns
//   mag_sq = r^2 + s^2
//   ns = conv1d(mag_sq, [1,1,0,1,1], padding=2)
//
// This kernel computes ONE derivative evaluation.
// RK4 requires 4 calls per step, orchestrated from host.
//
// Data layout: separate real and imaginary arrays, each [N_BANDS] f32.

struct Params {
    n_bands: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

// Per-band learned parameters
@group(0) @binding(0) var<storage, read> r_in: array<f32>;
@group(0) @binding(1) var<storage, read> s_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> dr_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> ds_out: array<f32>;
@group(0) @binding(4) var<storage, read> gamma: array<f32>;   // [N_BANDS] softplus(raw)
@group(0) @binding(5) var<storage, read> omega: array<f32>;   // [N_BANDS]
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var<storage, read> alpha_beta: array<f32>; // [2]: alpha, beta

@compute @workgroup_size(64)
fn kerr_derivative(@builtin(global_invocation_id) id: vec3<u32>) {
    let band = id.x;
    let n = params.n_bands;

    if (band >= n) {
        return;
    }

    let r = r_in[band];
    let s = s_in[band];
    let mag_sq = r * r + s * s;

    // Conv1d with kernel [1, 1, 0, 1, 1] and padding=2
    // ns[i] = mag_sq[i-2] + mag_sq[i-1] + mag_sq[i+1] + mag_sq[i+2]
    // (center weight is 0, so we skip mag_sq[i])
    var ns: f32 = 0.0;

    if (band >= 2u) {
        let b2 = band - 2u;
        let r2 = r_in[b2]; let s2 = s_in[b2];
        ns += r2 * r2 + s2 * s2;
    }
    if (band >= 1u) {
        let b1 = band - 1u;
        let r1 = r_in[b1]; let s1 = s_in[b1];
        ns += r1 * r1 + s1 * s1;
    }
    if (band + 1u < n) {
        let bp1 = band + 1u;
        let rp1 = r_in[bp1]; let sp1 = s_in[bp1];
        ns += rp1 * rp1 + sp1 * sp1;
    }
    if (band + 2u < n) {
        let bp2 = band + 2u;
        let rp2 = r_in[bp2]; let sp2 = s_in[bp2];
        ns += rp2 * rp2 + sp2 * sp2;
    }

    let alpha = alpha_beta[0];
    let beta = alpha_beta[1];

    let phi = omega[band] + alpha * mag_sq + beta * ns;
    let g = gamma[band];

    dr_out[band] = -g * r - phi * s;
    ds_out[band] = -g * s + phi * r;
}
