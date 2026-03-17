//! Resident weight buffers — uploaded once to GPU VRAM, updated after Adam step.
//!
//! Eliminates per-dispatch PCIe transfers for weights. Only activations (inputs)
//! are uploaded per call. Weights stay in VRAM for the entire training session.

use crate::model::*;

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
    pub fn from_model(device: &wgpu::Device, queue: &wgpu::Queue, model: &ModelWeights) -> Self {
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

        Self {
            c_attn_w, c_attn_b, c_proj_w, c_proj_b,
            ln1_w, ln1_b, ln2_w, ln2_b,
            ffn_buffers,
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
