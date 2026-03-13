// Batched matrix-vector backward: d_x[pos] = W^T @ d_y[pos] for all positions.
//
// One workgroup per (pos, j) where j is the output element index.
// Each thread computes one element of one position's d_x.
// Replaces N separate matvec_backward dispatches with one.

struct Params {
    out_dim: u32,   // rows of W (d_y dimension)
    in_dim: u32,    // cols of W (d_x dimension)
    n_pos: u32,     // number of positions
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> w: array<f32>;     // [out_dim * in_dim] row-major
@group(0) @binding(1) var<storage, read> d_y: array<f32>;   // [n_pos * out_dim]
@group(0) @binding(2) var<storage, read_write> d_x: array<f32>; // [n_pos * in_dim]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn matvec_backward_batch(@builtin(global_invocation_id) id: vec3<u32>) {
    let flat_id = id.x;
    let in_dim = params.in_dim;
    let out_dim = params.out_dim;
    let n_pos = params.n_pos;

    let pos = flat_id / in_dim;
    let j = flat_id % in_dim;

    if (pos >= n_pos) {
        return;
    }

    // d_x[pos][j] = sum_i W[i][j] * d_y[pos][i]  (W^T @ d_y)
    var sum: f32 = 0.0;
    let dy_base = pos * out_dim;
    for (var i: u32 = 0u; i < out_dim; i++) {
        sum += w[i * in_dim + j] * d_y[dy_base + i];
    }
    d_x[pos * in_dim + j] = sum;
}
