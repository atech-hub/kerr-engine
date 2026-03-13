// Fused Kerr-ODE RK4 step — one full RK4 step in a single dispatch.
//
// Each thread handles one band. Computes k1, k2, k3, k4 derivatives
// and combines them: r_new = r + dt/6*(k1 + 2*k2 + 2*k3 + k4).
//
// Uses workgroup shared memory to synchronise between derivative stages
// (neighbour coupling requires all bands' values at each sub-step).
//
// Supports up to 256 bands (n_embd up to 512). For larger models,
// the host falls back to non-fused kerr_step.wgsl with separate dispatches.

struct Params {
    n_bands: u32,
    dt: f32,
    alpha: f32,
    beta: f32,
}

@group(0) @binding(0) var<storage, read_write> r: array<f32>;     // [N_BANDS], in-place update
@group(0) @binding(1) var<storage, read_write> s: array<f32>;     // [N_BANDS], in-place update
@group(0) @binding(2) var<storage, read> gamma: array<f32>;       // [N_BANDS]
@group(0) @binding(3) var<storage, read> omega: array<f32>;       // [N_BANDS]
@group(0) @binding(4) var<uniform> params: Params;

// Shared memory for intermediate state (need all bands visible for neighbour coupling)
var<workgroup> shared_r: array<f32, 256>;
var<workgroup> shared_s: array<f32, 256>;

// Compute derivative for one band given shared r,s state
fn kerr_deriv_r(band: u32, n: u32, ri: f32, si: f32) -> f32 {
    let mag_sq = ri * ri + si * si;
    var ns: f32 = 0.0;
    if (band >= 2u) { let r2 = shared_r[band-2u]; let s2 = shared_s[band-2u]; ns += r2*r2 + s2*s2; }
    if (band >= 1u) { let r1 = shared_r[band-1u]; let s1 = shared_s[band-1u]; ns += r1*r1 + s1*s1; }
    if (band+1u < n) { let rp = shared_r[band+1u]; let sp = shared_s[band+1u]; ns += rp*rp + sp*sp; }
    if (band+2u < n) { let rp = shared_r[band+2u]; let sp = shared_s[band+2u]; ns += rp*rp + sp*sp; }
    let phi = omega[band] + params.alpha * mag_sq + params.beta * ns;
    return -gamma[band] * ri - phi * si;
}

fn kerr_deriv_s(band: u32, n: u32, ri: f32, si: f32) -> f32 {
    let mag_sq = ri * ri + si * si;
    var ns: f32 = 0.0;
    if (band >= 2u) { let r2 = shared_r[band-2u]; let s2 = shared_s[band-2u]; ns += r2*r2 + s2*s2; }
    if (band >= 1u) { let r1 = shared_r[band-1u]; let s1 = shared_s[band-1u]; ns += r1*r1 + s1*s1; }
    if (band+1u < n) { let rp = shared_r[band+1u]; let sp = shared_s[band+1u]; ns += rp*rp + sp*sp; }
    if (band+2u < n) { let rp = shared_r[band+2u]; let sp = shared_s[band+2u]; ns += rp*rp + sp*sp; }
    let phi = omega[band] + params.alpha * mag_sq + params.beta * ns;
    return -gamma[band] * si + phi * ri;
}

@compute @workgroup_size(256)
fn kerr_rk4_step(@builtin(local_invocation_id) lid: vec3<u32>) {
    let band = lid.x;
    let n = params.n_bands;
    let dt = params.dt;

    if (band >= n) { return; }

    // Load initial state
    let r0 = r[band];
    let s0 = s[band];

    // === k1 = f(r0, s0) ===
    shared_r[band] = r0;
    shared_s[band] = s0;
    workgroupBarrier();

    let k1r = kerr_deriv_r(band, n, r0, s0);
    let k1s = kerr_deriv_s(band, n, r0, s0);

    // === k2 = f(r0 + dt/2*k1, s0 + dt/2*k1) ===
    let r1 = r0 + 0.5 * dt * k1r;
    let s1 = s0 + 0.5 * dt * k1s;
    shared_r[band] = r1;
    shared_s[band] = s1;
    workgroupBarrier();

    let k2r = kerr_deriv_r(band, n, r1, s1);
    let k2s = kerr_deriv_s(band, n, r1, s1);

    // === k3 = f(r0 + dt/2*k2, s0 + dt/2*k2) ===
    let r2 = r0 + 0.5 * dt * k2r;
    let s2 = s0 + 0.5 * dt * k2s;
    shared_r[band] = r2;
    shared_s[band] = s2;
    workgroupBarrier();

    let k3r = kerr_deriv_r(band, n, r2, s2);
    let k3s = kerr_deriv_s(band, n, r2, s2);

    // === k4 = f(r0 + dt*k3, s0 + dt*k3) ===
    let r3 = r0 + dt * k3r;
    let s3 = s0 + dt * k3s;
    shared_r[band] = r3;
    shared_s[band] = s3;
    workgroupBarrier();

    let k4r = kerr_deriv_r(band, n, r3, s3);
    let k4s = kerr_deriv_s(band, n, r3, s3);

    // === Combine: r_new = r0 + dt/6*(k1 + 2*k2 + 2*k3 + k4) ===
    r[band] = r0 + dt / 6.0 * (k1r + 2.0*k2r + 2.0*k3r + k4r);
    s[band] = s0 + dt / 6.0 * (k1s + 2.0*k2s + 2.0*k3s + k4s);
}
