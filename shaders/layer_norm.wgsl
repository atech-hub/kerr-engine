// Layer normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
//
// Two-pass approach:
//   Pass 1 (reduce): compute mean and variance over x[0..dim]
//   Pass 2 (normalize): apply normalization per element
//
// For our sizes (dim=128), a single workgroup handles both passes
// using workgroup shared memory for the reduction.

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

var<workgroup> shared_sum: array<f32, 128>;  // for parallel reduction

@compute @workgroup_size(128)
fn layer_norm(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let dim = params.dim;

    // Each thread loads one element (dim <= 128 for our architecture)
    var val: f32 = 0.0;
    if (tid < dim) {
        val = x[tid];
    }

    // Pass 1a: compute mean via parallel reduction
    shared_sum[tid] = val;
    workgroupBarrier();

    // Tree reduction for sum
    for (var stride: u32 = 64u; stride > 0u; stride >>= 1u) {
        if (tid < stride && tid + stride < dim) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        workgroupBarrier();
    }

    let mean = shared_sum[0] / f32(dim);
    workgroupBarrier();

    // Pass 1b: compute variance
    var diff = val - mean;
    shared_sum[tid] = select(0.0, diff * diff, tid < dim);
    workgroupBarrier();

    for (var stride: u32 = 64u; stride > 0u; stride >>= 1u) {
        if (tid < stride && tid + stride < dim) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        workgroupBarrier();
    }

    let variance = shared_sum[0] / f32(dim);
    let inv_std = 1.0 / sqrt(variance + 1e-5);

    // Pass 2: normalize
    if (tid < dim) {
        y[tid] = (val - mean) * inv_std * weight[tid] + bias[tid];
    }
}
