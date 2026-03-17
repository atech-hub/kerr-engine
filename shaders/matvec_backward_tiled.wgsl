// Backward through linear: d_x = W^T @ d_y — tiled workgroup reduction.
// One workgroup per input element j. 64 threads each accumulate out_dim/64 terms,
// then tree-reduce. Error: O(out_dim/64 + log2(64)) vs O(out_dim).

struct Params {
    out_dim: u32,
    in_dim: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> d_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> d_x: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn matvec_backward(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let j = wid.x;
    if (j >= params.in_dim) { return; }

    let out_dim = params.out_dim;
    let in_dim = params.in_dim;

    var acc: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= out_dim) { break; }
        acc += w[i * in_dim + j] * d_y[i];
        i += 64u;
    }
    partial[tid] = acc;
    workgroupBarrier();

    for (var stride: u32 = 32u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        d_x[j] = partial[0];
    }
}
