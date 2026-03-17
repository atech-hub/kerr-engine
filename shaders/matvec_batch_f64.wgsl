// Batched matrix-vector with f64 accumulation for numerical stability.
// Accumulates dot product in double precision, stores result in f32.
// Eliminates floating-point accumulation error that causes NaN at 768-dim+.
//
// Requires: enable f64; (device must support shader-f64 feature)

enable f64;

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

    // Accumulate in f64 for numerical stability
    var sum: f64 = 0.0;
    let x_base = pos * in_dim;
    let w_base = i * in_dim;
    for (var j: u32 = 0u; j < in_dim; j++) {
        sum += f64(w[w_base + j]) * f64(x[x_base + j]);
    }

    if (params.use_bias == 1u) {
        sum += f64(b[i]);
    }

    y[pos * out_dim + i] = f32(sum);
}
