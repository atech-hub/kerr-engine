// y[i] = a[i] + scale * b[i]
// Used for RK4 midpoint: r_mid = r + 0.5*dt*k1r, etc.

struct Params {
    len: u32,
    scale: f32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn vec_scale_add(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.len) {
        return;
    }
    y[i] = a[i] + params.scale * b[i];
}
