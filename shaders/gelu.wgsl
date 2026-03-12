// GELU activation: y[i] = 0.5 * x[i] * (1 + tanh(sqrt(2/pi) * (x[i] + 0.044715 * x[i]^3)))
// Applied element-wise. One thread per element.

struct Params {
    len: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const SQRT_2_OVER_PI: f32 = 0.7978845608;

@compute @workgroup_size(64)
fn gelu(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.len) {
        return;
    }
    let v = x[i];
    let inner = SQRT_2_OVER_PI * (v + 0.044715 * v * v * v);
    y[i] = 0.5 * v * (1.0 + tanh(inner));
}
