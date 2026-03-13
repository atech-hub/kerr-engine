// Attention backward — Phase 2: compute d_k and d_v from d_score and att_weights.
//
// One thread per (ki, d_global). No shared memory needed.
// d_k[ki][d] = sum_{pos >= ki} d_score[pos][head][ki] * q[pos][d] * scale
// d_v[ki][d] = sum_{pos >= ki} att_weights[head][pos][ki] * d_out[pos][d]

struct Params {
    seq_len: u32,
    n_head: u32,
    head_dim: u32,
    n_embd: u32,
}

@group(0) @binding(0) var<storage, read> q_all: array<f32>;
@group(0) @binding(1) var<storage, read> d_out: array<f32>;
@group(0) @binding(2) var<storage, read> att_weights: array<f32>;
@group(0) @binding(3) var<storage, read> d_score_buf: array<f32>;
@group(0) @binding(4) var<storage, read_write> d_k: array<f32>;
@group(0) @binding(5) var<storage, read_write> d_v: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn attn_backward_dkv(@builtin(global_invocation_id) id: vec3<u32>) {
    let flat_id = id.x;
    let T = params.seq_len;
    let n_embd = params.n_embd;
    let n_head = params.n_head;
    let hd = params.head_dim;

    let ki = flat_id / n_embd;
    let d_global = flat_id % n_embd;

    if (ki >= T) {
        return;
    }

    let head = d_global / hd;
    let scale = 1.0 / sqrt(f32(hd));

    var dk_sum: f32 = 0.0;
    var dv_sum: f32 = 0.0;
    for (var pos: u32 = ki; pos < T; pos++) {
        // d_score_buf layout: [pos * n_head * T + head * T + ki]
        let d_score_val = d_score_buf[pos * n_head * T + head * T + ki];
        dk_sum += d_score_val * q_all[pos * n_embd + d_global];

        // att_weights layout: [head * T * T + pos * T + ki]
        let att_val = att_weights[head * T * T + pos * T + ki];
        dv_sum += att_val * d_out[pos * n_embd + d_global];
    }

    d_k[ki * n_embd + d_global] = dk_sum * scale;
    d_v[ki * n_embd + d_global] = dv_sum;
}
