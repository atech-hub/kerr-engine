// Backward through GELU activation.
// d_x[i] = d_y[i] * gelu'(x[i])
// where gelu'(x) = 0.5*(1+tanh(inner)) + 0.5*x*sech^2(inner)*d_inner
// inner = sqrt(2/pi) * (x + 0.044715*x^3)
// d_inner = sqrt(2/pi) * (1 + 3*0.044715*x^2)
//
// Element-wise. One thread per element.

struct Params {
    len: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> d_y: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> d_x: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const SQRT_2_OVER_PI: f32 = 0.7978845608;

@compute @workgroup_size(64)
fn gelu_backward(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.len) {
        return;
    }

    let v = x[i];
    let x3 = v * v * v;
    let inner = SQRT_2_OVER_PI * (v + 0.044715 * x3);
    let tanh_inner = tanh(inner);
    let sech2 = 1.0 - tanh_inner * tanh_inner;
    let d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * v * v);

    let grad = 0.5 * (1.0 + tanh_inner) + 0.5 * v * sech2 * d_inner;
    d_x[i] = d_y[i] * grad;
}
