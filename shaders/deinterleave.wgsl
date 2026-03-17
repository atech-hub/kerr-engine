// Deinterleave: [r0,s0,r1,s1,...] → separate r[] and s[] buffers.
// One thread per band. Batched: handles n_pos positions.

struct Params {
    n_bands: u32,
    n_pos: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> interleaved: array<f32>;   // [n_pos * n_bands * 2]
@group(0) @binding(1) var<storage, read_write> r_out: array<f32>;   // [n_pos * n_bands]
@group(0) @binding(2) var<storage, read_write> s_out: array<f32>;   // [n_pos * n_bands]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn deinterleave(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let total = params.n_bands * params.n_pos;
    if (flat >= total) { return; }

    let pos = flat / params.n_bands;
    let band = flat % params.n_bands;
    let interleaved_base = pos * params.n_bands * 2u;

    r_out[flat] = interleaved[interleaved_base + band * 2u];
    s_out[flat] = interleaved[interleaved_base + band * 2u + 1u];
}
