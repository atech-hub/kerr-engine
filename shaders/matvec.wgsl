// Matrix-vector multiply: y = W @ x + b
//
// W is [out_dim][in_dim], stored row-major as flat array.
// x is [in_dim], b is [out_dim], y is [out_dim].
// One thread per output element.

struct Params {
    out_dim: u32,
    in_dim: u32,
    use_bias: u32,  // 1 = add bias, 0 = no bias
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> w: array<f32>;       // [out_dim * in_dim]
@group(0) @binding(1) var<storage, read> x: array<f32>;       // [in_dim]
@group(0) @binding(2) var<storage, read> b: array<f32>;       // [out_dim]
@group(0) @binding(3) var<storage, read_write> y: array<f32>; // [out_dim]
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn matvec(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= params.out_dim) {
        return;
    }

    var sum: f32 = 0.0;
    let base = row * params.in_dim;
    for (var j: u32 = 0u; j < params.in_dim; j++) {
        sum += w[base + j] * x[j];
    }

    if (params.use_bias == 1u) {
        sum += b[row];
    }

    y[row] = sum;
}
