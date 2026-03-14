// Batched matrix-vector forward: y[pos] = W @ x[pos] + b for all positions.
//
// One thread per (pos, i) where i is the output element index.
// Each thread computes one element of one position's output.
// Replaces N separate matvec dispatches with one.

struct Params {
    out_dim: u32,   // rows of W (y dimension)
    in_dim: u32,    // cols of W (x dimension)
    n_pos: u32,     // number of positions
    use_bias: u32,  // 1 = add bias, 0 = no bias
}

@group(0) @binding(0) var<storage, read> w: array<f32>;     // [out_dim * in_dim] row-major
@group(0) @binding(1) var<storage, read> x: array<f32>;     // [n_pos * in_dim]
@group(0) @binding(2) var<storage, read> b: array<f32>;     // [out_dim]
@group(0) @binding(3) var<storage, read_write> y: array<f32>; // [n_pos * out_dim]
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

    // y[pos][i] = sum_j W[i][j] * x[pos][j] + b[i]
    var sum: f32 = 0.0;
    let x_base = pos * in_dim;
    let w_base = i * in_dim;
    for (var j: u32 = 0u; j < in_dim; j++) {
        sum += w[w_base + j] * x[x_base + j];
    }

    if (params.use_bias == 1u) {
        sum += b[i];
    }

    y[pos * out_dim + i] = sum;
}
