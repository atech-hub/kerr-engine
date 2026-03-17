// Backward through linear: d_x = W^T @ d_y
//
// W is [out_dim][in_dim], stored row-major as flat array.
// d_y is [out_dim], d_x is [in_dim].
// One thread per input element (column of W).
//
// This computes: d_x[j] = sum_i W[i][j] * d_y[i]
// which is the transpose matvec needed for gradient backprop.

struct Params {
    out_dim: u32,
    in_dim: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> w: array<f32>;         // [out_dim * in_dim]
@group(0) @binding(1) var<storage, read> d_y: array<f32>;       // [out_dim]
@group(0) @binding(2) var<storage, read_write> d_x: array<f32>; // [in_dim]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn matvec_backward(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.x;
    if (j >= params.in_dim) {
        return;
    }

    var sum: f32 = 0.0;
    var comp: f32 = 0.0;
    for (var i: u32 = 0u; i < params.out_dim; i++) {
        let product = w[i * params.in_dim + j] * d_y[i];
        let y_val = product - comp;
        let t = sum + y_val;
        comp = (t - sum) - y_val;
        sum = t;
    }

    d_x[j] = sum;
}
