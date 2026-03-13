// Backward through layer normalization.
//
// Forward: y = (x - mean) / std * weight + bias
// Backward: computes d_x, d_weight, d_bias from d_y.
//
// Uses strided workgroup reduction (same pattern as forward layer_norm).
// Supports any dim up to 2048 (256 threads × 8 elements each).
//
// Output layout: d_x in first dim floats, d_weight in next dim, d_bias in next dim.
// Total output: 3 * dim floats.

struct Params {
    dim: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> d_y: array<f32>;       // [dim]
@group(0) @binding(1) var<storage, read> x: array<f32>;         // [dim]
@group(0) @binding(2) var<storage, read> weight: array<f32>;    // [dim]
@group(0) @binding(3) var<storage, read_write> out: array<f32>; // [3*dim]: d_x, d_weight, d_bias
@group(0) @binding(4) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn layer_norm_backward(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let dim = params.dim;
    let dimf = f32(dim);
    let eps: f32 = 1e-5;

    // === Recompute forward stats: mean, variance, inv_std ===

    // Mean via strided reduction
    var local_sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < dim) {
        local_sum += x[i];
        i += WG_SIZE;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    for (var stride: u32 = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
        workgroupBarrier();
    }
    let mean = shared_sum[0] / dimf;
    workgroupBarrier();

    // Variance via strided reduction
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
        if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
        workgroupBarrier();
    }
    let variance = shared_sum[0] / dimf;
    let inv_std = 1.0 / sqrt(variance + eps);
    workgroupBarrier();

    // === Compute x_hat and d_x_hat ===
    // d_weight[i] = d_y[i] * x_hat[i]
    // d_bias[i] = d_y[i]
    // d_x_hat[i] = d_y[i] * weight[i]

    // We need two global sums: sum(d_x_hat) and sum(d_x_hat * x_hat)
    // Compute them via strided reduction

    // Sum of d_x_hat
    var dxh_sum: f32 = 0.0;
    var dxh_xh_sum: f32 = 0.0;
    i = tid;
    while (i < dim) {
        let x_hat_i = (x[i] - mean) * inv_std;
        let d_x_hat_i = d_y[i] * weight[i];
        dxh_sum += d_x_hat_i;
        dxh_xh_sum += d_x_hat_i * x_hat_i;
        i += WG_SIZE;
    }

    // Reduce dxh_sum
    shared_sum[tid] = dxh_sum;
    workgroupBarrier();
    for (var stride: u32 = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
        workgroupBarrier();
    }
    let d_x_hat_sum = shared_sum[0];
    workgroupBarrier();

    // Reduce dxh_xh_sum
    shared_sum[tid] = dxh_xh_sum;
    workgroupBarrier();
    for (var stride: u32 = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
        workgroupBarrier();
    }
    let d_x_hat_x_hat_sum = shared_sum[0];
    workgroupBarrier();

    // === Write outputs (strided) ===
    // d_x[i] = inv_std / n * (n * d_x_hat[i] - d_x_hat_sum - x_hat[i] * d_x_hat_x_hat_sum)
    // d_weight[i] = d_y[i] * x_hat[i]
    // d_bias[i] = d_y[i]
    i = tid;
    while (i < dim) {
        let x_hat_i = (x[i] - mean) * inv_std;
        let d_x_hat_i = d_y[i] * weight[i];

        out[i] = inv_std / dimf * (dimf * d_x_hat_i - d_x_hat_sum - x_hat_i * d_x_hat_x_hat_sum);
        out[dim + i] = d_y[i] * x_hat_i;
        out[2u * dim + i] = d_y[i];
        i += WG_SIZE;
    }
}
