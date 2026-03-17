// RK4 combination: y[i] = base[i] + (dt/6) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])

struct Params {
    len: u32,
    dt_over_6: f32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> base: array<f32>;
@group(0) @binding(1) var<storage, read> k1: array<f32>;
@group(0) @binding(2) var<storage, read> k2: array<f32>;
@group(0) @binding(3) var<storage, read> k3: array<f32>;
@group(0) @binding(4) var<storage, read> k4: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn rk4_combine(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.len) {
        return;
    }
    y[i] = base[i] + params.dt_over_6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
}
