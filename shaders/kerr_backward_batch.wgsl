// Batched Kerr-ODE derivative backward: compute input + parameter gradients
// for ALL positions in one dispatch.
//
// One thread per (pos, band). Outputs per (pos, band):
//   d_r, d_s           — input gradients
//   d_gamma, d_omega   — per-band parameter gradients
//   d_alpha_partial     — d_phi[k] * mag_sq[k] (reduce on CPU for d_alpha)
//   d_beta_partial      — d_phi[k] * ns[k]     (reduce on CPU for d_beta)
//
// The transpose convolution (d_ns → d_mag_sq) is computed per-thread using
// the same neighbor pattern as the forward [1,1,0,1,1] kernel.

struct Params {
    n_bands: u32,
    n_pos: u32,
    alpha: f32,
    beta: f32,
}

// Forward cache inputs (saved during forward derivative eval)
@group(0) @binding(0) var<storage, read> r: array<f32>;         // [n_pos * n_bands]
@group(0) @binding(1) var<storage, read> s: array<f32>;         // [n_pos * n_bands]
@group(0) @binding(2) var<storage, read> gamma: array<f32>;     // [n_bands]
@group(0) @binding(3) var<storage, read> omega: array<f32>;     // [n_bands]

// Upstream gradients
@group(0) @binding(4) var<storage, read> d_dr: array<f32>;      // [n_pos * n_bands]
@group(0) @binding(5) var<storage, read> d_ds: array<f32>;      // [n_pos * n_bands]

// Outputs
@group(0) @binding(6) var<storage, read_write> d_r_out: array<f32>;  // [n_pos * n_bands]
@group(0) @binding(7) var<storage, read_write> d_s_out: array<f32>;  // [n_pos * n_bands]
@group(0) @binding(8) var<storage, read_write> d_gamma_out: array<f32>;  // [n_pos * n_bands]
@group(0) @binding(9) var<storage, read_write> d_omega_out: array<f32>;  // [n_pos * n_bands]
@group(0) @binding(10) var<storage, read_write> d_alpha_partial: array<f32>; // [n_pos * n_bands]
@group(0) @binding(11) var<storage, read_write> d_beta_partial: array<f32>;  // [n_pos * n_bands]

@group(0) @binding(12) var<uniform> params: Params;

@compute @workgroup_size(64)
fn kerr_backward_batch(@builtin(global_invocation_id) id: vec3<u32>) {
    let flat_id = id.x;
    let n = params.n_bands;
    let n_pos = params.n_pos;

    let pos = flat_id / n;
    let band = flat_id % n;

    if (pos >= n_pos) {
        return;
    }

    let base = pos * n;
    let idx = base + band;

    // Read cached forward state
    let r_k = r[idx];
    let s_k = s[idx];
    let g_k = gamma[band];

    // Recompute forward intermediates for this band
    let mag_sq_k = r_k * r_k + s_k * s_k;

    // Neighbour sum (conv1d [1,1,0,1,1])
    var ns_k: f32 = 0.0;
    if (band >= 2u) {
        let i = base + band - 2u;
        ns_k += r[i] * r[i] + s[i] * s[i];
    }
    if (band >= 1u) {
        let i = base + band - 1u;
        ns_k += r[i] * r[i] + s[i] * s[i];
    }
    if (band + 1u < n) {
        let i = base + band + 1u;
        ns_k += r[i] * r[i] + s[i] * s[i];
    }
    if (band + 2u < n) {
        let i = base + band + 2u;
        ns_k += r[i] * r[i] + s[i] * s[i];
    }

    let phi_k = omega[band] + params.alpha * mag_sq_k + params.beta * ns_k;

    // Read upstream gradients
    let ddr_k = d_dr[idx];
    let dds_k = d_ds[idx];

    // dr[k] = -gamma[k]*r[k] - phi[k]*s[k]
    // ds[k] = -gamma[k]*s[k] + phi[k]*r[k]

    // d_gamma[k] = d_dr[k] * (-r[k]) + d_ds[k] * (-s[k])
    let dg_k = ddr_k * (-r_k) + dds_k * (-s_k);

    // d_phi[k] = d_dr[k] * (-s[k]) + d_ds[k] * r[k]
    let dphi_k = ddr_k * (-s_k) + dds_k * r_k;

    // d_omega[k] = d_phi[k]
    let dom_k = dphi_k;

    // d_alpha partial = d_phi[k] * mag_sq[k]
    let da_k = dphi_k * mag_sq_k;

    // d_beta partial = d_phi[k] * ns[k]
    let db_k = dphi_k * ns_k;

    // d_mag_sq[k] from phi: d_phi[k] * alpha
    var d_mag_sq_k: f32 = dphi_k * params.alpha;

    // d_mag_sq from ns: transpose convolution
    // ns[j] uses mag_sq[k] when k == j-2, j-1, j+1, or j+2
    // So d_mag_sq[k] += d_ns[k-2] + d_ns[k-1] + d_ns[k+1] + d_ns[k+2]
    // where d_ns[j] = d_phi[j] * beta
    // We need d_phi for neighbors — recompute from d_dr/d_ds
    if (band >= 2u) {
        let j = band - 2u;
        let ji = base + j;
        let dphi_j = d_dr[ji] * (-s[ji]) + d_ds[ji] * r[ji];
        d_mag_sq_k += dphi_j * params.beta;
    }
    if (band >= 1u) {
        let j = band - 1u;
        let ji = base + j;
        let dphi_j = d_dr[ji] * (-s[ji]) + d_ds[ji] * r[ji];
        d_mag_sq_k += dphi_j * params.beta;
    }
    if (band + 1u < n) {
        let j = band + 1u;
        let ji = base + j;
        let dphi_j = d_dr[ji] * (-s[ji]) + d_ds[ji] * r[ji];
        d_mag_sq_k += dphi_j * params.beta;
    }
    if (band + 2u < n) {
        let j = band + 2u;
        let ji = base + j;
        let dphi_j = d_dr[ji] * (-s[ji]) + d_ds[ji] * r[ji];
        d_mag_sq_k += dphi_j * params.beta;
    }

    // d_r[k] = d_dr[k] * (-gamma[k]) + d_ds[k] * phi[k] + d_mag_sq[k] * 2*r[k]
    let dr_k = ddr_k * (-g_k) + dds_k * phi_k + d_mag_sq_k * 2.0 * r_k;

    // d_s[k] = d_dr[k] * (-phi[k]) + d_ds[k] * (-gamma[k]) + d_mag_sq[k] * 2*s[k]
    let ds_k = ddr_k * (-phi_k) + dds_k * (-g_k) + d_mag_sq_k * 2.0 * s_k;

    // Write outputs
    d_r_out[idx] = dr_k;
    d_s_out[idx] = ds_k;
    d_gamma_out[idx] = dg_k;
    d_omega_out[idx] = dom_k;
    d_alpha_partial[idx] = da_k;
    d_beta_partial[idx] = db_k;
}
