// Batched matrix-vector with Kahan compensated summation.
// Fallback for GPUs without f64 support. Reduces accumulation error
// from O(n) to O(1) using error compensation tracking.

struct Params {
    out_dim: u32,
    in_dim: u32,
    n_pos: u32,
    use_bias: u32,
}

@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn matvec_batch(@builtin(global_invocation_id) id: vec3<u32>) {
    let flat_id = id.x;
    let out_dim = params.out_dim;
    let in_dim = params.in_dim;
    let n_pos = params.n_pos;

    let pos = flat_id / out_dim;
    let i = flat_id % out_dim;

    if (pos >= n_pos) {
        return;
    }

    // Kahan compensated summation — tracks rounding error
    var sum: f32 = 0.0;
    var comp: f32 = 0.0;  // compensation for lost low-order bits
    let x_base = pos * in_dim;
    let w_base = i * in_dim;
    for (var j: u32 = 0u; j < in_dim; j++) {
        let product = w[w_base + j] * x[x_base + j];
        let y_val = product - comp;
        let t = sum + y_val;
        comp = (t - sum) - y_val;
        sum = t;
    }

    if (params.use_bias == 1u) {
        let y_val = b[i] - comp;
        let t = sum + y_val;
        sum = t;
    }

    y[pos * out_dim + i] = sum;
}
