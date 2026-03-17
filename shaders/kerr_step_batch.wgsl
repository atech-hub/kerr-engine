// Batched Kerr-ODE derivative: compute dr/ds for ALL positions in one dispatch.
//
// One thread per (pos, band). Shared parameters (gamma, omega, alpha, beta)
// are read once by all positions. Replaces N_POS separate kerr_step dispatches.
//
// Data layout: r_in/s_in/dr_out/ds_out are [n_pos * n_bands], contiguous per position.

struct Params {
    n_bands: u32,
    n_pos: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> r_in: array<f32>;       // [n_pos * n_bands]
@group(0) @binding(1) var<storage, read> s_in: array<f32>;       // [n_pos * n_bands]
@group(0) @binding(2) var<storage, read_write> dr_out: array<f32>; // [n_pos * n_bands]
@group(0) @binding(3) var<storage, read_write> ds_out: array<f32>; // [n_pos * n_bands]
@group(0) @binding(4) var<storage, read> gamma: array<f32>;      // [n_bands]
@group(0) @binding(5) var<storage, read> omega: array<f32>;      // [n_bands]
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var<storage, read> alpha_beta: array<f32>; // [2]: alpha, beta

@compute @workgroup_size(64)
fn kerr_derivative_batch(@builtin(global_invocation_id) id: vec3<u32>) {
    let flat_id = id.x;
    let n = params.n_bands;
    let n_pos = params.n_pos;

    let pos = flat_id / n;
    let band = flat_id % n;

    if (pos >= n_pos) {
        return;
    }

    let base = pos * n;
    let r = r_in[base + band];
    let s = s_in[base + band];
    let mag_sq = r * r + s * s;

    // Conv1d with kernel [1, 1, 0, 1, 1] and padding=2
    var ns: f32 = 0.0;

    if (band >= 2u) {
        let idx = base + band - 2u;
        let r2 = r_in[idx]; let s2 = s_in[idx];
        ns += r2 * r2 + s2 * s2;
    }
    if (band >= 1u) {
        let idx = base + band - 1u;
        let r1 = r_in[idx]; let s1 = s_in[idx];
        ns += r1 * r1 + s1 * s1;
    }
    if (band + 1u < n) {
        let idx = base + band + 1u;
        let rp1 = r_in[idx]; let sp1 = s_in[idx];
        ns += rp1 * rp1 + sp1 * sp1;
    }
    if (band + 2u < n) {
        let idx = base + band + 2u;
        let rp2 = r_in[idx]; let sp2 = s_in[idx];
        ns += rp2 * rp2 + sp2 * sp2;
    }

    let alpha = alpha_beta[0];
    let beta = alpha_beta[1];

    // Clamp magnitude terms to prevent phi overflow at 768-dim+.
    // GPU FP differences can cause |Z| to drift, making mag_sq/ns explode.
    // Clamp at 2500 (50² — matches RK4 magnitude bound of 50.0).
    let mag_sq_c = min(mag_sq, 2500.0);
    let ns_c = min(ns, 10000.0);

    let phi = omega[band] + alpha * mag_sq_c + beta * ns_c;
    let g = gamma[band];

    dr_out[base + band] = -g * r - phi * s;
    ds_out[base + band] = -g * s + phi * r;
}
