//! GPU compute backend — wgpu implementation of ComputeBackend.
//!
//! Hybrid approach: linear and layer_norm run on GPU,
//! compound operations (attention, kerr_maestro_add) fall through
//! to CPU until their shaders are written.

use wgpu::util::DeviceExt;

use crate::backend::ComputeBackend;
use crate::model::*;

/// GPU backend — dispatches to WGSL compute shaders.
pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    matvec_pipeline: wgpu::ComputePipeline,
    matvec_layout: wgpu::BindGroupLayout,
    layer_norm_pipeline: wgpu::ComputePipeline,
    layer_norm_layout: wgpu::BindGroupLayout,
    kerr_deriv_pipeline: wgpu::ComputePipeline,
    kerr_deriv_layout: wgpu::BindGroupLayout,
}

// ─── Uniform param structs ──────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatvecParams {
    out_dim: u32,
    in_dim: u32,
    use_bias: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerNormParams {
    dim: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct KerrDerivParams {
    n_bands: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

// ─── Helper: build bind group layout entries ────────────────────

fn storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl GpuBackend {
    /// Initialize GPU device and compile all compute shaders.
    pub fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("Failed to find GPU adapter");

        println!("  GPU adapter: {}", adapter.get_info().name);
        println!("  Backend:     {:?}", adapter.get_info().backend);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("kerr-engine-backend"),
                ..Default::default()
            },
            None,
        ))
        .expect("Failed to get GPU device");

        // ─── Compile matvec shader ──────────────────────────────
        let matvec_src = include_str!("../shaders/matvec.wgsl");
        let matvec_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matvec"),
            source: wgpu::ShaderSource::Wgsl(matvec_src.into()),
        });

        // bindings: 0=w(ro), 1=x(ro), 2=b(ro), 3=y(rw), 4=params(uniform)
        let matvec_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matvec_layout"),
            entries: &[
                storage_ro(0),
                storage_ro(1),
                storage_ro(2),
                storage_rw(3),
                uniform_entry(4),
            ],
        });

        let matvec_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matvec_pl"),
            bind_group_layouts: &[&matvec_layout],
            push_constant_ranges: &[],
        });

        let matvec_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matvec_pipeline"),
            layout: Some(&matvec_pl),
            module: &matvec_module,
            entry_point: Some("matvec"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ─── Compile layer_norm shader ──────────────────────────
        let ln_src = include_str!("../shaders/layer_norm.wgsl");
        let ln_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("layer_norm"),
            source: wgpu::ShaderSource::Wgsl(ln_src.into()),
        });

        // bindings: 0=x(ro), 1=weight(ro), 2=bias(ro), 3=y(rw), 4=params(uniform)
        let layer_norm_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layer_norm_layout"),
            entries: &[
                storage_ro(0),
                storage_ro(1),
                storage_ro(2),
                storage_rw(3),
                uniform_entry(4),
            ],
        });

        let ln_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layer_norm_pl"),
            bind_group_layouts: &[&layer_norm_layout],
            push_constant_ranges: &[],
        });

        let layer_norm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("layer_norm_pipeline"),
            layout: Some(&ln_pl),
            module: &ln_module,
            entry_point: Some("layer_norm"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ─── Compile Kerr derivative shader ─────────────────────
        let kerr_src = include_str!("../shaders/kerr_step.wgsl");
        let kerr_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kerr_derivative"),
            source: wgpu::ShaderSource::Wgsl(kerr_src.into()),
        });

        // bindings: 0=r_in(ro), 1=s_in(ro), 2=dr_out(rw), 3=ds_out(rw),
        //           4=gamma(ro), 5=omega(ro), 6=params(uniform), 7=alpha_beta(ro)
        let kerr_deriv_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("kerr_deriv_layout"),
            entries: &[
                storage_ro(0),
                storage_ro(1),
                storage_rw(2),
                storage_rw(3),
                storage_ro(4),
                storage_ro(5),
                uniform_entry(6),
                storage_ro(7),
            ],
        });

        let kerr_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("kerr_deriv_pl"),
            bind_group_layouts: &[&kerr_deriv_layout],
            push_constant_ranges: &[],
        });

        let kerr_deriv_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("kerr_deriv_pipeline"),
            layout: Some(&kerr_pl),
            module: &kerr_module,
            entry_point: Some("kerr_derivative"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            matvec_pipeline,
            matvec_layout,
            layer_norm_pipeline,
            layer_norm_layout,
            kerr_deriv_pipeline,
            kerr_deriv_layout,
        }
    }

    // ─── GPU dispatch helpers ───────────────────────────────────

    /// Create a read-only storage buffer from f32 data.
    fn storage_buf(&self, label: &str, data: &[f32]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    /// Create a read-write storage buffer of given f32 count.
    fn output_buf(&self, label: &str, n_floats: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (n_floats * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Create a uniform buffer from a Pod struct.
    fn uniform_buf<T: bytemuck::Pod>(&self, label: &str, data: &T) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(data),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    /// Read back f32 data from a GPU buffer.
    fn readback(&self, buf: &wgpu::Buffer, n_floats: usize) -> Vec<f32> {
        let size = (n_floats * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    /// Run matvec: y = W @ x + b (or y = W @ x if bias is empty).
    fn gpu_matvec(&self, w_flat: &[f32], x: &[f32], b: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        let use_bias = if b.is_empty() { 0u32 } else { 1u32 };

        let w_buf = self.storage_buf("w", w_flat);
        let x_buf = self.storage_buf("x", x);
        // Bias buffer: if no bias, still need a valid buffer (1 element placeholder)
        let b_data = if b.is_empty() { &[0.0f32][..] } else { b };
        let b_buf = self.storage_buf("b", b_data);
        let y_buf = self.output_buf("y", out_dim);
        let params = MatvecParams {
            out_dim: out_dim as u32,
            in_dim: in_dim as u32,
            use_bias,
            _pad: 0,
        };
        let params_buf = self.uniform_buf("params", &params);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matvec_bg"),
            layout: &self.matvec_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: y_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.matvec_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (out_dim as u32 + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.readback(&y_buf, out_dim)
    }

    /// Run layer normalization on GPU.
    fn gpu_layer_norm(&self, x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
        let dim = x.len();

        let x_buf = self.storage_buf("x", x);
        let w_buf = self.storage_buf("weight", weight);
        let b_buf = self.storage_buf("bias", bias);
        let y_buf = self.output_buf("y", dim);
        let params = LayerNormParams { dim: dim as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
        let params_buf = self.uniform_buf("params", &params);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ln_bg"),
            layout: &self.layer_norm_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: y_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.layer_norm_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Single workgroup of 128 threads — matches our dim
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.readback(&y_buf, dim)
    }

    /// Run one Kerr derivative evaluation on GPU.
    /// Returns (dr, ds) each of length N_BANDS.
    fn gpu_kerr_derivative(&self, r: &[f32], s: &[f32], gamma: &[f32], omega: &[f32], alpha: f32, beta: f32) -> (Vec<f32>, Vec<f32>) {
        let n = r.len();

        let r_buf = self.storage_buf("r_in", r);
        let s_buf = self.storage_buf("s_in", s);
        let dr_buf = self.output_buf("dr_out", n);
        let ds_buf = self.output_buf("ds_out", n);
        let gamma_buf = self.storage_buf("gamma", gamma);
        let omega_buf = self.storage_buf("omega", omega);
        let params = KerrDerivParams { n_bands: n as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
        let params_buf = self.uniform_buf("params", &params);
        let ab = [alpha, beta];
        let ab_buf = self.storage_buf("alpha_beta", &ab);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("kerr_bg"),
            layout: &self.kerr_deriv_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: r_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: s_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dr_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: ds_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: gamma_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: omega_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: ab_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.kerr_deriv_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n as u32 + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        // Need two readbacks — submit is already done, just copy both
        let dr = self.readback(&dr_buf, n);
        let ds = self.readback(&ds_buf, n);
        (dr, ds)
    }

    /// Kerr-ODE forward via host-side RK4 with GPU derivative evaluations.
    fn gpu_kerr_ode(&self, weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
        // Unpack interleaved [r0, s0, r1, s1, ...] into separate r, s arrays
        let mut r = vec![0.0f32; N_BANDS];
        let mut s = vec![0.0f32; N_BANDS];
        for k in 0..N_BANDS {
            r[k] = x[k * 2];
            s[k] = x[k * 2 + 1];
        }

        // Pre-compute softplus(gamma_raw) on CPU (64 values, trivial)
        fn softplus(x: f32) -> f32 {
            if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
        }
        let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();

        // RK4 integration: 8 steps, each step needs 4 derivative evaluations
        for _ in 0..RK4_N_STEPS {
            let dt = RK4_DT;

            // k1 = f(r, s)
            let (k1r, k1s) = self.gpu_kerr_derivative(&r, &s, &gamma, &weights.omega, weights.alpha, weights.beta);

            // k2 = f(r + dt/2 * k1r, s + dt/2 * k1s)
            let r2: Vec<f32> = r.iter().zip(&k1r).map(|(&ri, &k)| ri + 0.5 * dt * k).collect();
            let s2: Vec<f32> = s.iter().zip(&k1s).map(|(&si, &k)| si + 0.5 * dt * k).collect();
            let (k2r, k2s) = self.gpu_kerr_derivative(&r2, &s2, &gamma, &weights.omega, weights.alpha, weights.beta);

            // k3 = f(r + dt/2 * k2r, s + dt/2 * k2s)
            let r3: Vec<f32> = r.iter().zip(&k2r).map(|(&ri, &k)| ri + 0.5 * dt * k).collect();
            let s3: Vec<f32> = s.iter().zip(&k2s).map(|(&si, &k)| si + 0.5 * dt * k).collect();
            let (k3r, k3s) = self.gpu_kerr_derivative(&r3, &s3, &gamma, &weights.omega, weights.alpha, weights.beta);

            // k4 = f(r + dt * k3r, s + dt * k3s)
            let r4: Vec<f32> = r.iter().zip(&k3r).map(|(&ri, &k)| ri + dt * k).collect();
            let s4: Vec<f32> = s.iter().zip(&k3s).map(|(&si, &k)| si + dt * k).collect();
            let (k4r, k4s) = self.gpu_kerr_derivative(&r4, &s4, &gamma, &weights.omega, weights.alpha, weights.beta);

            // Combine: r_new = r + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            for k in 0..N_BANDS {
                r[k] += dt / 6.0 * (k1r[k] + 2.0 * k2r[k] + 2.0 * k3r[k] + k4r[k]);
                s[k] += dt / 6.0 * (k1s[k] + 2.0 * k2s[k] + 2.0 * k3s[k] + k4s[k]);
            }
        }

        // Re-interleave
        let mut out = vec![0.0f32; N_EMBD];
        for k in 0..N_BANDS {
            out[k * 2] = r[k];
            out[k * 2 + 1] = s[k];
        }
        out
    }
}

// ─── ComputeBackend implementation ──────────────────────────────

impl ComputeBackend for GpuBackend {
    fn linear(&self, w: &[Vec<f32>], b: &[f32], x: &[f32]) -> Vec<f32> {
        let out_dim = w.len();
        let in_dim = if out_dim > 0 { w[0].len() } else { 0 };
        // Flatten row-major: w[i][j] → flat[i * in_dim + j]
        let mut w_flat = Vec::with_capacity(out_dim * in_dim);
        for row in w {
            w_flat.extend_from_slice(row);
        }
        self.gpu_matvec(&w_flat, x, b, out_dim, in_dim)
    }

    fn linear_no_bias(&self, w: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
        let out_dim = w.len();
        let in_dim = if out_dim > 0 { w[0].len() } else { 0 };
        let mut w_flat = Vec::with_capacity(out_dim * in_dim);
        for row in w {
            w_flat.extend_from_slice(row);
        }
        self.gpu_matvec(&w_flat, x, &[], out_dim, in_dim)
    }

    fn layer_norm(&self, x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
        self.gpu_layer_norm(x, weight, bias)
    }

    fn kerr_ode(&self, weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
        self.gpu_kerr_ode(weights, x)
    }

    fn maestro(&self, weights: &MaestroWeights, x: &[f32]) -> Vec<f32> {
        // Maestro = squeeze(linear+gelu) → process(linear)
        // Use GPU matvec for both linear operations
        let squeezed = self.linear(&weights.squeeze.w, &weights.squeeze.b, x);
        let activated: Vec<f32> = squeezed.iter().map(|&v| gelu_cpu(v)).collect();
        self.linear(&weights.process_1.w, &weights.process_1.b, &activated)
    }

    fn attention(&self, weights: &AttentionWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Attention: GPU-accelerated projections, CPU softmax
        let t = x.len();
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        let mut q_all = vec![vec![0.0f32; N_EMBD]; t];
        let mut k_all = vec![vec![0.0f32; N_EMBD]; t];
        let mut v_all = vec![vec![0.0f32; N_EMBD]; t];

        for pos in 0..t {
            // QKV projection on GPU
            let qkv = self.linear(&weights.c_attn.w, &weights.c_attn.b, &x[pos]);
            for i in 0..N_EMBD {
                q_all[pos][i] = qkv[i];
                k_all[pos][i] = qkv[N_EMBD + i];
                v_all[pos][i] = qkv[2 * N_EMBD + i];
            }
        }

        // Attention scores + softmax on CPU (small T, dominated by projection cost)
        let mut out = vec![vec![0.0f32; N_EMBD]; t];
        for head in 0..N_HEAD {
            let offset = head * HEAD_DIM;
            for qi in 0..t {
                let mut att = vec![f32::NEG_INFINITY; t];
                for ki in 0..=qi {
                    let mut dot = 0.0f32;
                    for d in 0..HEAD_DIM {
                        dot += q_all[qi][offset + d] * k_all[ki][offset + d];
                    }
                    att[ki] = dot * scale;
                }

                let max_att = att[..=qi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for ki in 0..=qi {
                    att[ki] = (att[ki] - max_att).exp();
                    exp_sum += att[ki];
                }
                for ki in 0..=qi {
                    att[ki] /= exp_sum;
                }

                for d in 0..HEAD_DIM {
                    let mut sum = 0.0f32;
                    for ki in 0..=qi {
                        sum += att[ki] * v_all[ki][offset + d];
                    }
                    out[qi][offset + d] = sum;
                }
            }
        }

        // Output projection on GPU
        out.iter()
            .map(|o| self.linear(&weights.c_proj.w, &weights.c_proj.b, o))
            .collect()
    }

    fn per_band_linear(&self, weights: &PerBandLinearWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Per-band 2x2 is tiny — CPU is fine, output projection goes to GPU
        let t = x.len();
        let mut result = Vec::with_capacity(t);

        for pos in 0..t {
            let mut bands_out = vec![0.0f32; N_EMBD];
            for band in 0..N_BANDS {
                let r_in = x[pos][band * 2];
                let s_in = x[pos][band * 2 + 1];
                let w = &weights.band_w[band];
                let b = &weights.band_b[band];
                bands_out[band * 2] = w[0][0] * r_in + w[1][0] * s_in + b[0];
                bands_out[band * 2 + 1] = w[0][1] * r_in + w[1][1] * s_in + b[1];
            }
            // Output projection on GPU
            let projected = self.linear(&weights.out_proj.w, &weights.out_proj.b, &bands_out);
            result.push(projected);
        }

        result
    }

    fn kerr_maestro_add(&self, weights: &KerrMaestroAddWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let t = x.len();
        let mut result = Vec::with_capacity(t);

        for pos in 0..t {
            let kerr_out = self.kerr_ode(&weights.kerr, &x[pos]);
            let maestro_out = self.maestro(&weights.maestro, &x[pos]);
            let mut combined = vec![0.0f32; N_EMBD];
            for i in 0..N_EMBD {
                combined[i] = kerr_out[i] + maestro_out[i];
            }
            let projected = self.linear(&weights.out_proj.w, &weights.out_proj.b, &combined);
            result.push(projected);
        }

        result
    }
}

// ─── CPU helper (GELU for maestro activation — trivial, not worth a shader) ─

#[allow(dead_code)]
fn gelu_cpu(x: f32) -> f32 {
    use std::f32::consts::PI;
    0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

// ─── Validation ─────────────────────────────────────────────────

/// Validate GPU backend against CPU backend on all primitives.
pub fn validate_gpu_backend() {
    use crate::backend::CpuBackend;

    println!("GPU Backend Validation\n");

    let gpu = GpuBackend::new();
    let cpu = CpuBackend;

    // Test 1: Linear (128→128 with bias)
    print!("  linear (128→128, bias)... ");
    {
        let in_dim = 128;
        let out_dim = 128;
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let w: Vec<Vec<f32>> = (0..out_dim).map(|i| {
            (0..in_dim).map(|j| ((i * in_dim + j) as f32 * 0.001).cos()).collect()
        }).collect();
        let b: Vec<f32> = (0..out_dim).map(|i| i as f32 * 0.01).collect();

        let cpu_y = cpu.linear(&w, &b, &x);
        let gpu_y = gpu.linear(&w, &b, &x);

        let max_diff = cpu_y.iter().zip(&gpu_y).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        if max_diff < 1e-4 {
            println!("PASS (max_diff={max_diff:.2e})");
        } else {
            println!("FAIL (max_diff={max_diff:.2e})");
        }
    }

    // Test 2: Linear no bias (384→128 — QKV projection size)
    print!("  linear_no_bias (384→128)... ");
    {
        let in_dim = 128;
        let out_dim = 384;
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.03).cos()).collect();
        let w: Vec<Vec<f32>> = (0..out_dim).map(|i| {
            (0..in_dim).map(|j| ((i * in_dim + j) as f32 * 0.0007).sin()).collect()
        }).collect();

        let cpu_y = cpu.linear_no_bias(&w, &x);
        let gpu_y = gpu.linear_no_bias(&w, &x);

        let max_diff = cpu_y.iter().zip(&gpu_y).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        if max_diff < 1e-4 {
            println!("PASS (max_diff={max_diff:.2e})");
        } else {
            println!("FAIL (max_diff={max_diff:.2e})");
        }
    }

    // Test 3: Layer norm (dim=128)
    print!("  layer_norm (dim=128)... ");
    {
        let dim = 128;
        let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.05).sin() * 2.0).collect();
        let weight: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32 * 0.01).cos() * 0.1).collect();
        let bias: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).sin() * 0.05).collect();

        let cpu_y = cpu.layer_norm(&x, &weight, &bias);
        let gpu_y = gpu.layer_norm(&x, &weight, &bias);

        let max_diff = cpu_y.iter().zip(&gpu_y).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        if max_diff < 1e-4 {
            println!("PASS (max_diff={max_diff:.2e})");
        } else {
            println!("FAIL (max_diff={max_diff:.2e})");
        }
    }

    // Test 4: Kerr-ODE (full RK4, 8 steps)
    print!("  kerr_ode (RK4, 8 steps)... ");
    {
        let mut x = vec![0.0f32; N_EMBD];
        for k in 0..N_BANDS {
            x[k * 2] = (k as f32 * 0.1).cos() * 0.5;
            x[k * 2 + 1] = (k as f32 * 0.1).sin() * 0.5;
        }
        let weights = KerrWeights {
            gamma_raw: (0..N_BANDS).map(|k| -2.0 + k as f32 * 0.05).collect(), // softplus → ~0.1
            omega: (0..N_BANDS).map(|k| k as f32 / N_BANDS as f32).collect(),
            alpha: 0.1,
            beta: 0.1,
        };

        let cpu_y = cpu.kerr_ode(&weights, &x);
        let gpu_y = gpu.kerr_ode(&weights, &x);

        let max_diff = cpu_y.iter().zip(&gpu_y).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        if max_diff < 1e-4 {
            println!("PASS (max_diff={max_diff:.2e})");
        } else {
            println!("FAIL (max_diff={max_diff:.2e})");
        }
    }

    println!("\nGPU backend validation complete.");
}

/// Benchmark GPU vs CPU on all primitives. Runs each operation many times
/// and reports median timing. This tells us whether GPU is worth wiring
/// into the training loop at our current scale (128-dim).
pub fn benchmark_gpu_vs_cpu() {
    use crate::backend::CpuBackend;

    println!("GPU vs CPU Benchmark\n");

    let gpu = GpuBackend::new();
    let cpu = CpuBackend;

    // ─── Shared test data ───────────────────────────────────────

    // Linear 128→128 (output projection, maestro layers)
    let in_dim = 128;
    let out_dim = 128;
    let x128: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let w128: Vec<Vec<f32>> = (0..out_dim).map(|i| {
        (0..in_dim).map(|j| ((i * in_dim + j) as f32 * 0.001).cos()).collect()
    }).collect();
    let b128: Vec<f32> = (0..out_dim).map(|i| i as f32 * 0.01).collect();

    // Linear 384→128 (QKV attention projection)
    let qkv_out = 384;
    let w384: Vec<Vec<f32>> = (0..qkv_out).map(|i| {
        (0..in_dim).map(|j| ((i * in_dim + j) as f32 * 0.0007).sin()).collect()
    }).collect();
    let b384: Vec<f32> = (0..qkv_out).map(|i| i as f32 * 0.005).collect();

    // Layer norm
    let ln_x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin() * 2.0).collect();
    let ln_w: Vec<f32> = (0..128).map(|i| 1.0 + (i as f32 * 0.01).cos() * 0.1).collect();
    let ln_b: Vec<f32> = (0..128).map(|i| (i as f32 * 0.02).sin() * 0.05).collect();

    // Kerr-ODE
    let mut kerr_x = vec![0.0f32; N_EMBD];
    for k in 0..N_BANDS {
        kerr_x[k * 2] = (k as f32 * 0.1).cos() * 0.5;
        kerr_x[k * 2 + 1] = (k as f32 * 0.1).sin() * 0.5;
    }
    let kerr_w = KerrWeights {
        gamma_raw: (0..N_BANDS).map(|k| -2.0 + k as f32 * 0.05).collect(),
        omega: (0..N_BANDS).map(|k| k as f32 / N_BANDS as f32).collect(),
        alpha: 0.1,
        beta: 0.1,
    };

    // Maestro
    let maestro_w = MaestroWeights {
        squeeze: LinearWeights {
            w: (0..MAESTRO_DIM).map(|i| {
                (0..N_EMBD).map(|j| ((i * N_EMBD + j) as f32 * 0.002).sin()).collect()
            }).collect(),
            b: vec![0.01; MAESTRO_DIM],
        },
        process_1: LinearWeights {
            w: (0..N_EMBD).map(|i| {
                (0..MAESTRO_DIM).map(|j| ((i * MAESTRO_DIM + j) as f32 * 0.003).cos()).collect()
            }).collect(),
            b: vec![0.01; N_EMBD],
        },
    };

    // ─── Warmup (GPU shader compilation, buffer caching) ────────

    print!("  Warmup...");
    for _ in 0..5 {
        let _ = gpu.linear(&w128, &b128, &x128);
        let _ = gpu.layer_norm(&ln_x, &ln_w, &ln_b);
    }
    println!(" done\n");

    let n_iters = 200;

    println!("  {:>30} {:>12} {:>12} {:>8}", "Operation", "CPU (us)", "GPU (us)", "Ratio");
    println!("  {}", "-".repeat(66));

    // ─── Benchmark each primitive ───────────────────────────────

    // Linear 128→128
    let cpu_us = bench(n_iters, || { let _ = cpu.linear(&w128, &b128, &x128); });
    let gpu_us = bench(n_iters, || { let _ = gpu.linear(&w128, &b128, &x128); });
    print_row("linear (128→128, bias)", cpu_us, gpu_us);

    // Linear 384→128 (QKV)
    let cpu_us = bench(n_iters, || { let _ = cpu.linear(&w384, &b384, &x128); });
    let gpu_us = bench(n_iters, || { let _ = gpu.linear(&w384, &b384, &x128); });
    print_row("linear (384→128, QKV)", cpu_us, gpu_us);

    // Linear no bias 128→128
    let cpu_us = bench(n_iters, || { let _ = cpu.linear_no_bias(&w128, &x128); });
    let gpu_us = bench(n_iters, || { let _ = gpu.linear_no_bias(&w128, &x128); });
    print_row("linear_no_bias (128→128)", cpu_us, gpu_us);

    // Layer norm
    let cpu_us = bench(n_iters, || { let _ = cpu.layer_norm(&ln_x, &ln_w, &ln_b); });
    let gpu_us = bench(n_iters, || { let _ = gpu.layer_norm(&ln_x, &ln_w, &ln_b); });
    print_row("layer_norm (dim=128)", cpu_us, gpu_us);

    // Kerr-ODE (full RK4, 8 steps × 4 derivative evals = 32 dispatches)
    let cpu_us = bench(n_iters / 4, || { let _ = cpu.kerr_ode(&kerr_w, &kerr_x); });
    let gpu_us = bench(n_iters / 4, || { let _ = gpu.kerr_ode(&kerr_w, &kerr_x); });
    print_row("kerr_ode (RK4, 8 steps)", cpu_us, gpu_us);

    // Maestro (squeeze + GELU + process = 2 linear + activation)
    let cpu_us = bench(n_iters, || { let _ = cpu.maestro(&maestro_w, &kerr_x); });
    let gpu_us = bench(n_iters, || { let _ = gpu.maestro(&maestro_w, &kerr_x); });
    print_row("maestro (128→16→128)", cpu_us, gpu_us);

    // ─── Composite: simulate one forward position ───────────────
    // One position through block 1-3: layer_norm + attn_proj(QKV) + attn_proj(out) +
    //   layer_norm + kerr_ode + maestro + out_proj
    // That's: 2 layer_norms + 3 linears (QKV, out_proj, kerr out_proj) + kerr + maestro

    println!("\n  {:>30} {:>12} {:>12} {:>8}", "Composite", "CPU (us)", "GPU (us)", "Ratio");
    println!("  {}", "-".repeat(66));

    let cpu_us = bench(n_iters / 4, || {
        let _ = cpu.layer_norm(&ln_x, &ln_w, &ln_b);
        let _ = cpu.linear(&w384, &b384, &x128);   // QKV
        let _ = cpu.linear(&w128, &b128, &x128);   // attn out_proj
        let _ = cpu.layer_norm(&ln_x, &ln_w, &ln_b);
        let _ = cpu.kerr_ode(&kerr_w, &kerr_x);
        let _ = cpu.maestro(&maestro_w, &kerr_x);
        let _ = cpu.linear(&w128, &b128, &x128);   // block out_proj
    });
    let gpu_us = bench(n_iters / 4, || {
        let _ = gpu.layer_norm(&ln_x, &ln_w, &ln_b);
        let _ = gpu.linear(&w384, &b384, &x128);
        let _ = gpu.linear(&w128, &b128, &x128);
        let _ = gpu.layer_norm(&ln_x, &ln_w, &ln_b);
        let _ = gpu.kerr_ode(&kerr_w, &kerr_x);
        let _ = gpu.maestro(&maestro_w, &kerr_x);
        let _ = gpu.linear(&w128, &b128, &x128);
    });
    print_row("one block (1 position)", cpu_us, gpu_us);

    println!();
    println!("  Ratio < 1.0 = GPU wins, > 1.0 = CPU wins at this scale.");
    println!("  GPU dispatch overhead is fixed ~50-200us per call.");
    println!("  At 128-dim, CPU compute per call is comparable to dispatch cost.");
}

fn bench(n: usize, mut f: impl FnMut()) -> f64 {
    let mut times = Vec::with_capacity(n);
    for _ in 0..n {
        let start = std::time::Instant::now();
        f();
        times.push(start.elapsed().as_micros() as f64);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Median
    times[times.len() / 2]
}

fn print_row(name: &str, cpu_us: f64, gpu_us: f64) {
    let ratio = gpu_us / cpu_us;
    let marker = if ratio < 1.0 { " <GPU" } else { "" };
    println!("  {:>30} {:>10.0} {:>10.0} {:>7.2}x{}", name, cpu_us, gpu_us, ratio, marker);
}
