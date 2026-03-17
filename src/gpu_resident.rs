//! Resident weight buffers — uploaded once to GPU VRAM, updated after Adam step.
//!
//! Eliminates per-dispatch PCIe transfers for weights. Only activations (inputs)
//! are uploaded per call. Weights stay in VRAM for the entire training session.

use crate::model::*;

/// Pre-allocated scratch buffers for the fused dual-maestro FFN chain.
/// One set per block. Sized at from_model() time based on max_pos × n_embd.
pub struct FfnScratch {
    pub x_buf: wgpu::Buffer,          // input [n_pos * n_embd]
    pub sq_buf: wgpu::Buffer,         // squeeze output [n_pos * maestro_dim]
    pub act_buf: wgpu::Buffer,        // GELU output [n_pos * maestro_dim]
    pub mae_out_buf: wgpu::Buffer,    // maestro process output [n_pos * n_embd]
    pub precond_buf: wgpu::Buffer,    // x + maestro_in [n_pos * n_embd]
    pub r_buf: wgpu::Buffer,          // deinterleaved r [n_pos * n_bands]
    pub s_buf: wgpu::Buffer,          // deinterleaved s [n_pos * n_bands]
    pub r_tmp: wgpu::Buffer,          // ODE scratch [n_pos * n_bands]
    pub s_tmp: wgpu::Buffer,          // ODE scratch [n_pos * n_bands]
    pub kerr_buf: wgpu::Buffer,       // reinterleaved ODE output [n_pos * n_embd]
    pub sq2_buf: wgpu::Buffer,        // maestro_out squeeze [n_pos * maestro_dim]
    pub act2_buf: wgpu::Buffer,       // maestro_out GELU [n_pos * maestro_dim]
    pub mae_out2_buf: wgpu::Buffer,   // maestro_out process [n_pos * n_embd]
    pub regulated_buf: wgpu::Buffer,  // kerr + maestro_out [n_pos * n_embd]
    pub output_buf: wgpu::Buffer,     // final out_proj [n_pos * n_embd]
    // RK4 scratch
    pub k1r: wgpu::Buffer, pub k1s: wgpu::Buffer,
    pub k2r: wgpu::Buffer, pub k2s: wgpu::Buffer,
    pub k3r: wgpu::Buffer, pub k3s: wgpu::Buffer,
    pub k4r: wgpu::Buffer, pub k4s: wgpu::Buffer,
    pub r_mid: wgpu::Buffer, pub s_mid: wgpu::Buffer,
    pub r_new: wgpu::Buffer, pub s_new: wgpu::Buffer,
}

impl FfnScratch {
    pub fn new(device: &wgpu::Device, n_pos: usize, n_embd: usize, maestro_dim: usize) -> Self {
        let n_bands = n_embd / 2;
        let total_bands = n_pos * n_bands;
        let make = |n: usize| device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            x_buf: make(n_pos * n_embd),
            sq_buf: make(n_pos * maestro_dim),
            act_buf: make(n_pos * maestro_dim),
            mae_out_buf: make(n_pos * n_embd),
            precond_buf: make(n_pos * n_embd),
            r_buf: make(total_bands), s_buf: make(total_bands),
            r_tmp: make(total_bands), s_tmp: make(total_bands),
            kerr_buf: make(n_pos * n_embd),
            sq2_buf: make(n_pos * maestro_dim),
            act2_buf: make(n_pos * maestro_dim),
            mae_out2_buf: make(n_pos * n_embd),
            regulated_buf: make(n_pos * n_embd),
            output_buf: make(n_pos * n_embd),
            k1r: make(total_bands), k1s: make(total_bands),
            k2r: make(total_bands), k2s: make(total_bands),
            k3r: make(total_bands), k3s: make(total_bands),
            k4r: make(total_bands), k4s: make(total_bands),
            r_mid: make(total_bands), s_mid: make(total_bands),
            r_new: make(total_bands), s_new: make(total_bands),
        }
    }
}

/// All model weights resident in GPU VRAM.
pub struct ResidentWeightBuffers {
    // Per-block attention weights [n_layers]
    pub c_attn_w: Vec<wgpu::Buffer>,
    pub c_attn_b: Vec<wgpu::Buffer>,
    pub c_proj_w: Vec<wgpu::Buffer>,
    pub c_proj_b: Vec<wgpu::Buffer>,

    // Per-block layer norms [n_layers × 2: ln1_w, ln1_b, ln2_w, ln2_b]
    pub ln1_w: Vec<wgpu::Buffer>,
    pub ln1_b: Vec<wgpu::Buffer>,
    pub ln2_w: Vec<wgpu::Buffer>,
    pub ln2_b: Vec<wgpu::Buffer>,

    // Per-block FFN — stored per block, layout depends on FFN type
    pub ffn_buffers: Vec<FfnResidentBuffers>,

    // Pre-allocated scratch buffers for fused FFN chain (one per block)
    pub ffn_scratch: Vec<FfnScratch>,

    // Final layer norm + LM head
    pub ln_f_w: wgpu::Buffer,
    pub ln_f_b: wgpu::Buffer,
    pub lm_head: wgpu::Buffer,
}

/// FFN weight buffers for one block.
pub enum FfnResidentBuffers {
    PerBand {
        band_w: wgpu::Buffer,   // flattened [n_bands * 4]
        band_b: wgpu::Buffer,   // flattened [n_bands * 2]
        out_proj_w: wgpu::Buffer,
        out_proj_b: wgpu::Buffer,
    },
    KerrMaestro {
        gamma: wgpu::Buffer,     // pre-softplus'd [n_bands]
        omega: wgpu::Buffer,     // [n_bands]
        alpha_beta: wgpu::Buffer, // [2]
        squeeze_w: wgpu::Buffer,
        squeeze_b: wgpu::Buffer,
        process_w: wgpu::Buffer,
        process_b: wgpu::Buffer,
        out_proj_w: wgpu::Buffer,
        out_proj_b: wgpu::Buffer,
    },
    KerrDualMaestro {
        gamma: wgpu::Buffer,
        omega: wgpu::Buffer,
        alpha_beta: wgpu::Buffer,
        in_squeeze_w: wgpu::Buffer,
        in_squeeze_b: wgpu::Buffer,
        in_process_w: wgpu::Buffer,
        in_process_b: wgpu::Buffer,
        out_squeeze_w: wgpu::Buffer,
        out_squeeze_b: wgpu::Buffer,
        out_process_w: wgpu::Buffer,
        out_process_b: wgpu::Buffer,
        out_proj_w: wgpu::Buffer,
        out_proj_b: wgpu::Buffer,
    },
}

fn create_buf(device: &wgpu::Device, queue: &wgpu::Queue, data: &[f32]) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
    buf
}

fn flatten_weights(w: &[Vec<f32>]) -> Vec<f32> {
    w.iter().flat_map(|row| row.iter().copied()).collect()
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

impl ResidentWeightBuffers {
    /// Upload all model weights to GPU VRAM.
    /// Upload all model weights to GPU VRAM and pre-allocate scratch buffers.
    /// `max_pos` is the maximum number of positions per dispatch (batch_size * seq_len).
    pub fn from_model(device: &wgpu::Device, queue: &wgpu::Queue, model: &ModelWeights) -> Self {
        let max_pos = 256; // batch_size(4) * seq_len(64) — covers the batched forward path
        let mut c_attn_w = Vec::new();
        let mut c_attn_b = Vec::new();
        let mut c_proj_w = Vec::new();
        let mut c_proj_b = Vec::new();
        let mut ln1_w = Vec::new();
        let mut ln1_b = Vec::new();
        let mut ln2_w = Vec::new();
        let mut ln2_b = Vec::new();
        let mut ffn_buffers = Vec::new();

        for block in &model.blocks {
            ln1_w.push(create_buf(device, queue, &block.ln_1.weight));
            ln1_b.push(create_buf(device, queue, &block.ln_1.bias));
            c_attn_w.push(create_buf(device, queue, &flatten_weights(&block.attn.c_attn.w)));
            c_attn_b.push(create_buf(device, queue, &block.attn.c_attn.b));
            c_proj_w.push(create_buf(device, queue, &flatten_weights(&block.attn.c_proj.w)));
            c_proj_b.push(create_buf(device, queue, &block.attn.c_proj.b));
            ln2_w.push(create_buf(device, queue, &block.ln_2.weight));
            ln2_b.push(create_buf(device, queue, &block.ln_2.bias));

            let ffn = match &block.ffn {
                FfnWeights::PerBand(w) => {
                    let bw: Vec<f32> = w.band_w.iter()
                        .flat_map(|b| b[0].iter().chain(b[1].iter()).copied())
                        .collect();
                    let bb: Vec<f32> = w.band_b.iter().flat_map(|b| b.iter().copied()).collect();
                    FfnResidentBuffers::PerBand {
                        band_w: create_buf(device, queue, &bw),
                        band_b: create_buf(device, queue, &bb),
                        out_proj_w: create_buf(device, queue, &flatten_weights(&w.out_proj.w)),
                        out_proj_b: create_buf(device, queue, &w.out_proj.b),
                    }
                }
                FfnWeights::KerrMaestro(w) => {
                    let gamma: Vec<f32> = w.kerr.gamma_raw.iter().map(|&g| softplus(g)).collect();
                    FfnResidentBuffers::KerrMaestro {
                        gamma: create_buf(device, queue, &gamma),
                        omega: create_buf(device, queue, &w.kerr.omega),
                        alpha_beta: create_buf(device, queue, &[w.kerr.alpha, w.kerr.beta]),
                        squeeze_w: create_buf(device, queue, &flatten_weights(&w.maestro.squeeze.w)),
                        squeeze_b: create_buf(device, queue, &w.maestro.squeeze.b),
                        process_w: create_buf(device, queue, &flatten_weights(&w.maestro.process_1.w)),
                        process_b: create_buf(device, queue, &w.maestro.process_1.b),
                        out_proj_w: create_buf(device, queue, &flatten_weights(&w.out_proj.w)),
                        out_proj_b: create_buf(device, queue, &w.out_proj.b),
                    }
                }
                FfnWeights::KerrDualMaestro(w) => {
                    let gamma: Vec<f32> = w.kerr.gamma_raw.iter().map(|&g| softplus(g)).collect();
                    FfnResidentBuffers::KerrDualMaestro {
                        gamma: create_buf(device, queue, &gamma),
                        omega: create_buf(device, queue, &w.kerr.omega),
                        alpha_beta: create_buf(device, queue, &[w.kerr.alpha, w.kerr.beta]),
                        in_squeeze_w: create_buf(device, queue, &flatten_weights(&w.maestro_in.squeeze.w)),
                        in_squeeze_b: create_buf(device, queue, &w.maestro_in.squeeze.b),
                        in_process_w: create_buf(device, queue, &flatten_weights(&w.maestro_in.process_1.w)),
                        in_process_b: create_buf(device, queue, &w.maestro_in.process_1.b),
                        out_squeeze_w: create_buf(device, queue, &flatten_weights(&w.maestro_out.squeeze.w)),
                        out_squeeze_b: create_buf(device, queue, &w.maestro_out.squeeze.b),
                        out_process_w: create_buf(device, queue, &flatten_weights(&w.maestro_out.process_1.w)),
                        out_process_b: create_buf(device, queue, &w.maestro_out.process_1.b),
                        out_proj_w: create_buf(device, queue, &flatten_weights(&w.out_proj.w)),
                        out_proj_b: create_buf(device, queue, &w.out_proj.b),
                    }
                }
            };
            ffn_buffers.push(ffn);
        }

        // Pre-allocate scratch buffers for fused FFN chain
        let n_embd = model.config.n_embd();
        let maestro_dim = model.config.maestro_dim;
        let ffn_scratch: Vec<FfnScratch> = model.blocks.iter().map(|block| {
            match &block.ffn {
                FfnWeights::KerrDualMaestro(_) | FfnWeights::KerrMaestro(_) => {
                    FfnScratch::new(device, max_pos, n_embd, maestro_dim)
                }
                FfnWeights::PerBand(_) => {
                    // PerBand doesn't use the fused chain, allocate minimal
                    FfnScratch::new(device, max_pos, n_embd, maestro_dim)
                }
            }
        }).collect();

        Self {
            c_attn_w, c_attn_b, c_proj_w, c_proj_b,
            ln1_w, ln1_b, ln2_w, ln2_b,
            ffn_buffers,
            ffn_scratch,
            ln_f_w: create_buf(device, queue, &model.ln_f.weight),
            ln_f_b: create_buf(device, queue, &model.ln_f.bias),
            lm_head: create_buf(device, queue, &flatten_weights(&model.lm_head)),
        }
    }

    /// Write updated weights back to resident GPU buffers after Adam step.
    pub fn update_from_model(&self, queue: &wgpu::Queue, model: &ModelWeights) {
        for (layer, block) in model.blocks.iter().enumerate() {
            queue.write_buffer(&self.ln1_w[layer], 0, bytemuck::cast_slice(&block.ln_1.weight));
            queue.write_buffer(&self.ln1_b[layer], 0, bytemuck::cast_slice(&block.ln_1.bias));
            queue.write_buffer(&self.c_attn_w[layer], 0, bytemuck::cast_slice(&flatten_weights(&block.attn.c_attn.w)));
            queue.write_buffer(&self.c_attn_b[layer], 0, bytemuck::cast_slice(&block.attn.c_attn.b));
            queue.write_buffer(&self.c_proj_w[layer], 0, bytemuck::cast_slice(&flatten_weights(&block.attn.c_proj.w)));
            queue.write_buffer(&self.c_proj_b[layer], 0, bytemuck::cast_slice(&block.attn.c_proj.b));
            queue.write_buffer(&self.ln2_w[layer], 0, bytemuck::cast_slice(&block.ln_2.weight));
            queue.write_buffer(&self.ln2_b[layer], 0, bytemuck::cast_slice(&block.ln_2.bias));

            match (&block.ffn, &self.ffn_buffers[layer]) {
                (FfnWeights::PerBand(w), FfnResidentBuffers::PerBand { band_w, band_b, out_proj_w, out_proj_b }) => {
                    let bw: Vec<f32> = w.band_w.iter()
                        .flat_map(|b| b[0].iter().chain(b[1].iter()).copied())
                        .collect();
                    let bb: Vec<f32> = w.band_b.iter().flat_map(|b| b.iter().copied()).collect();
                    queue.write_buffer(band_w, 0, bytemuck::cast_slice(&bw));
                    queue.write_buffer(band_b, 0, bytemuck::cast_slice(&bb));
                    queue.write_buffer(out_proj_w, 0, bytemuck::cast_slice(&flatten_weights(&w.out_proj.w)));
                    queue.write_buffer(out_proj_b, 0, bytemuck::cast_slice(&w.out_proj.b));
                }
                (FfnWeights::KerrMaestro(w), FfnResidentBuffers::KerrMaestro {
                    gamma, omega, alpha_beta, squeeze_w, squeeze_b, process_w, process_b, out_proj_w, out_proj_b,
                }) => {
                    let g: Vec<f32> = w.kerr.gamma_raw.iter().map(|&g| softplus(g)).collect();
                    queue.write_buffer(gamma, 0, bytemuck::cast_slice(&g));
                    queue.write_buffer(omega, 0, bytemuck::cast_slice(&w.kerr.omega));
                    queue.write_buffer(alpha_beta, 0, bytemuck::cast_slice(&[w.kerr.alpha, w.kerr.beta]));
                    queue.write_buffer(squeeze_w, 0, bytemuck::cast_slice(&flatten_weights(&w.maestro.squeeze.w)));
                    queue.write_buffer(squeeze_b, 0, bytemuck::cast_slice(&w.maestro.squeeze.b));
                    queue.write_buffer(process_w, 0, bytemuck::cast_slice(&flatten_weights(&w.maestro.process_1.w)));
                    queue.write_buffer(process_b, 0, bytemuck::cast_slice(&w.maestro.process_1.b));
                    queue.write_buffer(out_proj_w, 0, bytemuck::cast_slice(&flatten_weights(&w.out_proj.w)));
                    queue.write_buffer(out_proj_b, 0, bytemuck::cast_slice(&w.out_proj.b));
                }
                (FfnWeights::KerrDualMaestro(w), FfnResidentBuffers::KerrDualMaestro {
                    gamma, omega, alpha_beta,
                    in_squeeze_w, in_squeeze_b, in_process_w, in_process_b,
                    out_squeeze_w, out_squeeze_b, out_process_w, out_process_b,
                    out_proj_w, out_proj_b,
                }) => {
                    let g: Vec<f32> = w.kerr.gamma_raw.iter().map(|&g| softplus(g)).collect();
                    queue.write_buffer(gamma, 0, bytemuck::cast_slice(&g));
                    queue.write_buffer(omega, 0, bytemuck::cast_slice(&w.kerr.omega));
                    queue.write_buffer(alpha_beta, 0, bytemuck::cast_slice(&[w.kerr.alpha, w.kerr.beta]));
                    queue.write_buffer(in_squeeze_w, 0, bytemuck::cast_slice(&flatten_weights(&w.maestro_in.squeeze.w)));
                    queue.write_buffer(in_squeeze_b, 0, bytemuck::cast_slice(&w.maestro_in.squeeze.b));
                    queue.write_buffer(in_process_w, 0, bytemuck::cast_slice(&flatten_weights(&w.maestro_in.process_1.w)));
                    queue.write_buffer(in_process_b, 0, bytemuck::cast_slice(&w.maestro_in.process_1.b));
                    queue.write_buffer(out_squeeze_w, 0, bytemuck::cast_slice(&flatten_weights(&w.maestro_out.squeeze.w)));
                    queue.write_buffer(out_squeeze_b, 0, bytemuck::cast_slice(&w.maestro_out.squeeze.b));
                    queue.write_buffer(out_process_w, 0, bytemuck::cast_slice(&flatten_weights(&w.maestro_out.process_1.w)));
                    queue.write_buffer(out_process_b, 0, bytemuck::cast_slice(&w.maestro_out.process_1.b));
                    queue.write_buffer(out_proj_w, 0, bytemuck::cast_slice(&flatten_weights(&w.out_proj.w)));
                    queue.write_buffer(out_proj_b, 0, bytemuck::cast_slice(&w.out_proj.b));
                }
                _ => {} // FFN type mismatch — shouldn't happen
            }
        }

        queue.write_buffer(&self.ln_f_w, 0, bytemuck::cast_slice(&model.ln_f.weight));
        queue.write_buffer(&self.ln_f_b, 0, bytemuck::cast_slice(&model.ln_f.bias));
        queue.write_buffer(&self.lm_head, 0, bytemuck::cast_slice(&flatten_weights(&model.lm_head)));
    }
}
