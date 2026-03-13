// Batched outer product accumulation:
//   d_w[i][j] = sum_{pos=0..n_pos} d_y[pos][i] * x[pos][j]
//   d_b[i]    = sum_{pos=0..n_pos} d_y[pos][i]
//
// One workgroup per output row i. Threads parallelize over columns j.
// Each thread loops over all positions (typically 64) — trivial inner loop.

struct Params {
    out_dim: u32,   // rows of d_w (e.g. 2304 for c_attn, 768 for c_proj)
    in_dim: u32,    // cols of d_w (e.g. 768)
    n_pos: u32,     // number of positions to sum over
    compute_bias: u32, // 1 = also compute d_b, 0 = skip
}

@group(0) @binding(0) var<storage, read> d_y: array<f32>;   // [n_pos * out_dim]
@group(0) @binding(1) var<storage, read> x: array<f32>;     // [n_pos * in_dim]
@group(0) @binding(2) var<storage, read_write> d_w: array<f32>; // [out_dim * in_dim]
@group(0) @binding(3) var<storage, read_write> d_b: array<f32>; // [out_dim]
@group(0) @binding(4) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn outer_product(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = wg_id.x;  // output row
    let tid = lid.x;
    let out_dim = params.out_dim;
    let in_dim = params.in_dim;
    let n_pos = params.n_pos;

    if (i >= out_dim) {
        return;
    }

    // Each thread handles columns j = tid, tid+WG_SIZE, tid+2*WG_SIZE, ...
    var j = tid;
    while (j < in_dim) {
        var sum: f32 = 0.0;
        for (var pos: u32 = 0u; pos < n_pos; pos++) {
            sum += d_y[pos * out_dim + i] * x[pos * in_dim + j];
        }
        d_w[i * in_dim + j] += sum;
        j += WG_SIZE;
    }

    // Bias: one thread per row computes the sum
    if (params.compute_bias != 0u && tid == 0u) {
        var bias_sum: f32 = 0.0;
        for (var pos: u32 = 0u; pos < n_pos; pos++) {
            bias_sum += d_y[pos * out_dim + i];
        }
        d_b[i] += bias_sum;
    }
}
