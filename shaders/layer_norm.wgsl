// Layer normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
//
// Strided approach: each thread handles ceil(dim/256) elements.
// Supports any dim up to 256 * MAX_ELEMS_PER_THREAD = 2048.
// Tree reduction on 256 partial sums in shared memory.

struct Params {
    dim: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;         // [dim]
@group(0) @binding(1) var<storage, read> weight: array<f32>;    // [dim]
@group(0) @binding(2) var<storage, read> bias: array<f32>;      // [dim]
@group(0) @binding(3) var<storage, read_write> y: array<f32>;   // [dim]
@group(0) @binding(4) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;

var<workgroup> shared_sum: array<f32, 256>;  // one per thread

@compute @workgroup_size(256)
fn layer_norm(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let dim = params.dim;

    // Pass 1a: each thread accumulates its strided elements for mean
    var local_sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < dim) {
        local_sum += x[i];
        i += WG_SIZE;
    }

    shared_sum[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction for sum
    for (var stride: u32 = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        workgroupBarrier();
    }

    let mean = shared_sum[0] / f32(dim);
    workgroupBarrier();

    // Pass 1b: compute variance (strided)
    var var_sum: f32 = 0.0;
    i = tid;
    while (i < dim) {
        let diff = x[i] - mean;
        var_sum += diff * diff;
        i += WG_SIZE;
    }

    shared_sum[tid] = var_sum;
    workgroupBarrier();

    for (var stride: u32 = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        workgroupBarrier();
    }

    let variance = shared_sum[0] / f32(dim);
    let inv_std = 1.0 / sqrt(variance + 1e-5);

    // Pass 2: normalize (strided — each thread writes its own elements)
    i = tid;
    while (i < dim) {
        y[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
        i += WG_SIZE;
    }
}
