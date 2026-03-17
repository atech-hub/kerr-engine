// Batched matrix-vector: tiled workgroup reduction.
// One workgroup per (pos, output_row). 64 threads per workgroup.
// Dispatch: (n_pos * out_dim, 1, 1) workgroups.

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

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn matvec_batch(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let out_dim = params.out_dim;
    let in_dim = params.in_dim;
    let n_pos = params.n_pos;

    let row = wid.x;  // one workgroup per output row
    let pos = wid.y;  // one workgroup per position

    if (pos >= n_pos) { return; }

    let x_base = pos * in_dim;
    let w_base = row * in_dim;

    // Each thread accumulates its strided chunk
    var acc: f32 = 0.0;
    var j: u32 = tid;
    loop {
        if (j >= in_dim) { break; }
        acc += w[w_base + j] * x[x_base + j];
        j += 64u;
    }
    partial[tid] = acc;
    workgroupBarrier();

    // Tree reduction
    for (var stride: u32 = 32u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        var result = partial[0];
        if (params.use_bias == 1u) {
            result += b[row];
        }
        y[pos * out_dim + row] = result;
    }
}
