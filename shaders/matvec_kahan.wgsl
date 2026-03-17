// Matrix-vector with Kahan compensated summation. Single position.
// Fallback when f64 not available.

struct Params {
    out_dim: u32,
    in_dim: u32,
    use_bias: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn matvec(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= params.out_dim) { return; }

    var sum: f32 = 0.0;
    var comp: f32 = 0.0;
    let base = row * params.in_dim;
    for (var j: u32 = 0u; j < params.in_dim; j++) {
        let product = w[base + j] * x[j];
        let y_val = product - comp;
        let t = sum + y_val;
        comp = (t - sum) - y_val;
        sum = t;
    }
    if (params.use_bias == 1u) {
        let y_val = b[row] - comp;
        sum = sum + y_val;
    }
    y[row] = sum;
}
