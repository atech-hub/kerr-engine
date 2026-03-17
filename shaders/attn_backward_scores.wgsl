// Attention backward — Phase 1: compute d_score and d_q.
//
// One workgroup per (pos, head). Threads parallelize over ki values.
// Writes d_q directly (no race — heads partition n_embd, pos is unique per wg).
// Writes d_score to scratch buffer for Phase 2.

struct Params {
    seq_len: u32,
    n_head: u32,
    head_dim: u32,
    n_embd: u32,
}

@group(0) @binding(0) var<storage, read> d_out: array<f32>;
@group(0) @binding(1) var<storage, read> q_all: array<f32>;
@group(0) @binding(2) var<storage, read> k_all: array<f32>;
@group(0) @binding(3) var<storage, read> v_all: array<f32>;
@group(0) @binding(4) var<storage, read> att_weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> d_q: array<f32>;
@group(0) @binding(6) var<storage, read_write> d_score_buf: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

const WG_SIZE: u32 = 64u;

var<workgroup> shared_reduce: array<f32, 64>;

@compute @workgroup_size(64)
fn attn_backward_scores(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let pos = wg_id.x;
    let head = wg_id.y;
    let tid = lid.x;
    let T = params.seq_len;
    let hd = params.head_dim;
    let n_embd = params.n_embd;
    let n_head = params.n_head;
    let offset = head * hd;
    let scale = 1.0 / sqrt(f32(hd));

    if (pos >= T || head >= n_head) {
        return;
    }

    let att_base = head * T * T + pos * T;
    let dscore_base = pos * n_head * T + head * T;

    // --- Pass A: d_att[ki] = dot(d_out[pos][offset..], v[ki][offset..]) ---
    // Each thread handles ki = tid, tid+WG_SIZE, tid+2*WG_SIZE, ...
    var ki = tid;
    while (ki <= pos) {
        var dot: f32 = 0.0;
        let d_out_base = pos * n_embd + offset;
        let v_base = ki * n_embd + offset;
        for (var d: u32 = 0u; d < hd; d++) {
            dot += d_out[d_out_base + d] * v_all[v_base + d];
        }
        // Store d_att temporarily in d_score_buf
        d_score_buf[dscore_base + ki] = dot;
        ki += WG_SIZE;
    }
    workgroupBarrier();

    // --- Pass B: sum_j att[j]*d_att[j] via reduction ---
    var local_sum: f32 = 0.0;
    ki = tid;
    while (ki <= pos) {
        local_sum += att_weights[att_base + ki] * d_score_buf[dscore_base + ki];
        ki += WG_SIZE;
    }
    shared_reduce[tid] = local_sum;
    workgroupBarrier();

    for (var stride: u32 = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_reduce[tid] += shared_reduce[tid + stride];
        }
        workgroupBarrier();
    }
    let att_d_att_sum = shared_reduce[0];
    workgroupBarrier();

    // --- Pass C: softmax backward ---
    // d_score[ki] = att[ki] * (d_att[ki] - att_d_att_sum)
    ki = tid;
    while (ki <= pos) {
        let att_ki = att_weights[att_base + ki];
        let d_att_ki = d_score_buf[dscore_base + ki];
        d_score_buf[dscore_base + ki] = att_ki * (d_att_ki - att_d_att_sum);
        ki += WG_SIZE;
    }
    workgroupBarrier();

    // --- Pass D: d_q[pos][offset+d] = sum_ki d_score[ki] * k[ki][offset+d] * scale ---
    // Kahan compensated summation
    var d_idx = tid;
    while (d_idx < hd) {
        var sum: f32 = 0.0;
        var comp: f32 = 0.0;
        for (var k: u32 = 0u; k <= pos; k++) {
            let product = d_score_buf[dscore_base + k] * k_all[k * n_embd + offset + d_idx];
            let y_val = product - comp;
            let t = sum + y_val;
            comp = (t - sum) - y_val;
            sum = t;
        }
        d_q[pos * n_embd + offset + d_idx] = sum * scale;
        d_idx += WG_SIZE;
    }
}
