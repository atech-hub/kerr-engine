// Matrix-vector: tiled workgroup reduction for numerical stability at 768-dim+.
// One workgroup per output row. 64 threads each accumulate in_dim/64 terms,
// then tree-reduce via shared memory. Error: O(in_dim/64 + log2(64)) vs O(in_dim).

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

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn matvec(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x;
    if (row >= params.out_dim) { return; }

    let in_dim = params.in_dim;
    let w_base = row * in_dim;

    // Each thread accumulates its strided chunk
    var acc: f32 = 0.0;
    var j: u32 = tid;
    loop {
        if (j >= in_dim) { break; }
        acc += w[w_base + j] * x[j];
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
        y[row] = result;
    }
}
