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
    // Backward shaders
    matvec_bwd_pipeline: wgpu::ComputePipeline,
    matvec_bwd_layout: wgpu::BindGroupLayout,
    layer_norm_bwd_pipeline: wgpu::ComputePipeline,
    layer_norm_bwd_layout: wgpu::BindGroupLayout,
    #[allow(dead_code)] // Ready for FFN backward restructuring
    gelu_bwd_pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    gelu_bwd_layout: wgpu::BindGroupLayout,
    // Attention backward (two dispatches)
    attn_bwd_scores_pipeline: wgpu::ComputePipeline,
    attn_bwd_scores_layout: wgpu::BindGroupLayout,
    attn_bwd_dkv_pipeline: wgpu::ComputePipeline,
    attn_bwd_dkv_layout: wgpu::BindGroupLayout,
    // Batched outer product: d_w = D_Y^T @ X
    outer_product_pipeline: wgpu::ComputePipeline,
    outer_product_layout: wgpu::BindGroupLayout,
    // Batched matvec backward: d_x[pos] = W^T @ d_y[pos] for all positions
    matvec_bwd_batch_pipeline: wgpu::ComputePipeline,
    matvec_bwd_batch_layout: wgpu::BindGroupLayout,
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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatvecBwdParams {
    out_dim: u32,
    in_dim: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GeluBwdParams {
    len: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnBwdParams {
    seq_len: u32,
    n_head: u32,
    head_dim: u32,
    n_embd: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct OuterProductParams {
    out_dim: u32,
    in_dim: u32,
    n_pos: u32,
    compute_bias: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatvecBwdBatchParams {
    out_dim: u32,
    in_dim: u32,
    n_pos: u32,
    _pad: u32,
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

        Self::from_adapter(adapter)
    }

    /// Initialize GPU with a specific adapter selected by index.
    pub fn with_device_index(idx: usize) -> Self {
        let instance = wgpu::Instance::default();
        let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all()).into_iter().collect();
        assert!(
            idx < adapters.len(),
            "GPU device index {idx} out of range ({} adapters available)",
            adapters.len()
        );
        // enumerate_adapters returns owned adapters — take the one we want
        let mut adapters = adapters;
        let adapter = adapters.swap_remove(idx);

        Self::from_adapter(adapter)
    }

    /// Shared constructor: compile shaders and build pipelines from a chosen adapter.
    fn from_adapter(adapter: wgpu::Adapter) -> Self {
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

        // ─── Compile backward shaders ──────────────────────────────

        // matvec_backward: d_x = W^T @ d_y
        let mvb_src = include_str!("../shaders/matvec_backward.wgsl");
        let mvb_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matvec_backward"),
            source: wgpu::ShaderSource::Wgsl(mvb_src.into()),
        });
        // bindings: 0=w(ro), 1=d_y(ro), 2=d_x(rw), 3=params(uniform)
        let matvec_bwd_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matvec_bwd_layout"),
            entries: &[storage_ro(0), storage_ro(1), storage_rw(2), uniform_entry(3)],
        });
        let mvb_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matvec_bwd_pl"),
            bind_group_layouts: &[&matvec_bwd_layout],
            push_constant_ranges: &[],
        });
        let matvec_bwd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matvec_bwd_pipeline"),
            layout: Some(&mvb_pl),
            module: &mvb_module,
            entry_point: Some("matvec_backward"),
            compilation_options: Default::default(),
            cache: None,
        });

        // layer_norm_backward: d_x, d_weight, d_bias from d_y
        let lnb_src = include_str!("../shaders/layer_norm_backward.wgsl");
        let lnb_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("layer_norm_backward"),
            source: wgpu::ShaderSource::Wgsl(lnb_src.into()),
        });
        // bindings: 0=d_y(ro), 1=x(ro), 2=weight(ro), 3=out(rw), 4=params(uniform)
        let layer_norm_bwd_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layer_norm_bwd_layout"),
            entries: &[storage_ro(0), storage_ro(1), storage_ro(2), storage_rw(3), uniform_entry(4)],
        });
        let lnb_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layer_norm_bwd_pl"),
            bind_group_layouts: &[&layer_norm_bwd_layout],
            push_constant_ranges: &[],
        });
        let layer_norm_bwd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("layer_norm_bwd_pipeline"),
            layout: Some(&lnb_pl),
            module: &lnb_module,
            entry_point: Some("layer_norm_backward"),
            compilation_options: Default::default(),
            cache: None,
        });

        // gelu_backward: d_x = d_y * gelu'(x)
        let gb_src = include_str!("../shaders/gelu_backward.wgsl");
        let gb_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gelu_backward"),
            source: wgpu::ShaderSource::Wgsl(gb_src.into()),
        });
        // bindings: 0=d_y(ro), 1=x(ro), 2=d_x(rw), 3=params(uniform)
        let gelu_bwd_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gelu_bwd_layout"),
            entries: &[storage_ro(0), storage_ro(1), storage_rw(2), uniform_entry(3)],
        });
        let gb_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gelu_bwd_pl"),
            bind_group_layouts: &[&gelu_bwd_layout],
            push_constant_ranges: &[],
        });
        let gelu_bwd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gelu_bwd_pipeline"),
            layout: Some(&gb_pl),
            module: &gb_module,
            entry_point: Some("gelu_backward"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ─── Compile attention backward shaders ─────────────────────
        // Dispatch 1: attn_backward_scores — one workgroup per (pos, head)
        let abs_src = include_str!("../shaders/attn_backward_scores.wgsl");
        let abs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("attn_backward_scores"),
            source: wgpu::ShaderSource::Wgsl(abs_src.into()),
        });
        // bindings: 0=d_out(ro), 1=q(ro), 2=k(ro), 3=v(ro), 4=att_weights(ro),
        //           5=d_q(rw), 6=d_score_buf(rw), 7=params(uniform)
        let attn_bwd_scores_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("attn_bwd_scores_layout"),
            entries: &[
                storage_ro(0), storage_ro(1), storage_ro(2), storage_ro(3), storage_ro(4),
                storage_rw(5), storage_rw(6), uniform_entry(7),
            ],
        });
        let abs_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("attn_bwd_scores_pl"),
            bind_group_layouts: &[&attn_bwd_scores_layout],
            push_constant_ranges: &[],
        });
        let attn_bwd_scores_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("attn_bwd_scores_pipeline"),
            layout: Some(&abs_pl),
            module: &abs_module,
            entry_point: Some("attn_backward_scores"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Dispatch 2: attn_backward_dkv — one thread per (ki, d_global)
        let abdkv_src = include_str!("../shaders/attn_backward_dkv.wgsl");
        let abdkv_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("attn_backward_dkv"),
            source: wgpu::ShaderSource::Wgsl(abdkv_src.into()),
        });
        // bindings: 0=q(ro), 1=d_out(ro), 2=att_weights(ro), 3=d_score_buf(ro),
        //           4=d_k(rw), 5=d_v(rw), 6=params(uniform)
        let attn_bwd_dkv_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("attn_bwd_dkv_layout"),
            entries: &[
                storage_ro(0), storage_ro(1), storage_ro(2), storage_ro(3),
                storage_rw(4), storage_rw(5), uniform_entry(6),
            ],
        });
        let abdkv_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("attn_bwd_dkv_pl"),
            bind_group_layouts: &[&attn_bwd_dkv_layout],
            push_constant_ranges: &[],
        });
        let attn_bwd_dkv_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("attn_bwd_dkv_pipeline"),
            layout: Some(&abdkv_pl),
            module: &abdkv_module,
            entry_point: Some("attn_backward_dkv"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ─── Compile outer product shader ────────────────────────────
        let op_src = include_str!("../shaders/outer_product.wgsl");
        let op_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("outer_product"),
            source: wgpu::ShaderSource::Wgsl(op_src.into()),
        });
        // bindings: 0=d_y(ro), 1=x(ro), 2=d_w(rw), 3=d_b(rw), 4=params(uniform)
        let outer_product_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("outer_product_layout"),
            entries: &[storage_ro(0), storage_ro(1), storage_rw(2), storage_rw(3), uniform_entry(4)],
        });
        let op_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("outer_product_pl"),
            bind_group_layouts: &[&outer_product_layout],
            push_constant_ranges: &[],
        });
        let outer_product_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("outer_product_pipeline"),
            layout: Some(&op_pl),
            module: &op_module,
            entry_point: Some("outer_product"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ─── Compile batched matvec backward shader ─────────────────
        let mvbb_src = include_str!("../shaders/matvec_backward_batch.wgsl");
        let mvbb_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matvec_backward_batch"),
            source: wgpu::ShaderSource::Wgsl(mvbb_src.into()),
        });
        // bindings: 0=w(ro), 1=d_y(ro), 2=d_x(rw), 3=params(uniform)
        let matvec_bwd_batch_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matvec_bwd_batch_layout"),
            entries: &[storage_ro(0), storage_ro(1), storage_rw(2), uniform_entry(3)],
        });
        let mvbb_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matvec_bwd_batch_pl"),
            bind_group_layouts: &[&matvec_bwd_batch_layout],
            push_constant_ranges: &[],
        });
        let matvec_bwd_batch_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matvec_bwd_batch_pipeline"),
            layout: Some(&mvbb_pl),
            module: &mvbb_module,
            entry_point: Some("matvec_backward_batch"),
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
            matvec_bwd_pipeline,
            matvec_bwd_layout,
            layer_norm_bwd_pipeline,
            layer_norm_bwd_layout,
            gelu_bwd_pipeline,
            gelu_bwd_layout,
            attn_bwd_scores_pipeline,
            attn_bwd_scores_layout,
            attn_bwd_dkv_pipeline,
            attn_bwd_dkv_layout,
            outer_product_pipeline,
            outer_product_layout,
            matvec_bwd_batch_pipeline,
            matvec_bwd_batch_layout,
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
        let n_bands = weights.gamma_raw.len();
        let n_embd = n_bands * 2;
        let n_steps = weights.rk4_n_steps;
        let dt = 1.0 / n_steps as f32;

        // Unpack interleaved [r0, s0, r1, s1, ...] into separate r, s arrays
        let mut r = vec![0.0f32; n_bands];
        let mut s = vec![0.0f32; n_bands];
        for k in 0..n_bands {
            r[k] = x[k * 2];
            s[k] = x[k * 2 + 1];
        }

        // Pre-compute softplus(gamma_raw) on CPU (trivial)
        fn softplus(x: f32) -> f32 {
            if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
        }
        let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();

        // RK4 integration
        for _ in 0..n_steps {
            let (k1r, k1s) = self.gpu_kerr_derivative(&r, &s, &gamma, &weights.omega, weights.alpha, weights.beta);

            let r2: Vec<f32> = r.iter().zip(&k1r).map(|(&ri, &k)| ri + 0.5 * dt * k).collect();
            let s2: Vec<f32> = s.iter().zip(&k1s).map(|(&si, &k)| si + 0.5 * dt * k).collect();
            let (k2r, k2s) = self.gpu_kerr_derivative(&r2, &s2, &gamma, &weights.omega, weights.alpha, weights.beta);

            let r3: Vec<f32> = r.iter().zip(&k2r).map(|(&ri, &k)| ri + 0.5 * dt * k).collect();
            let s3: Vec<f32> = s.iter().zip(&k2s).map(|(&si, &k)| si + 0.5 * dt * k).collect();
            let (k3r, k3s) = self.gpu_kerr_derivative(&r3, &s3, &gamma, &weights.omega, weights.alpha, weights.beta);

            let r4: Vec<f32> = r.iter().zip(&k3r).map(|(&ri, &k)| ri + dt * k).collect();
            let s4: Vec<f32> = s.iter().zip(&k3s).map(|(&si, &k)| si + dt * k).collect();
            let (k4r, k4s) = self.gpu_kerr_derivative(&r4, &s4, &gamma, &weights.omega, weights.alpha, weights.beta);

            for k in 0..n_bands {
                r[k] += dt / 6.0 * (k1r[k] + 2.0 * k2r[k] + 2.0 * k3r[k] + k4r[k]);
                s[k] += dt / 6.0 * (k1s[k] + 2.0 * k2s[k] + 2.0 * k3s[k] + k4s[k]);
            }
        }

        // Re-interleave
        let mut out = vec![0.0f32; n_embd];
        for k in 0..n_bands {
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
        let t = x.len();
        let n_embd = x[0].len();
        let n_head = weights.n_head;
        let head_dim = n_embd / n_head;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut q_all = vec![vec![0.0f32; n_embd]; t];
        let mut k_all = vec![vec![0.0f32; n_embd]; t];
        let mut v_all = vec![vec![0.0f32; n_embd]; t];

        for pos in 0..t {
            let qkv = self.linear(&weights.c_attn.w, &weights.c_attn.b, &x[pos]);
            for i in 0..n_embd {
                q_all[pos][i] = qkv[i];
                k_all[pos][i] = qkv[n_embd + i];
                v_all[pos][i] = qkv[2 * n_embd + i];
            }
        }

        let mut out = vec![vec![0.0f32; n_embd]; t];
        for head in 0..n_head {
            let offset = head * head_dim;
            for qi in 0..t {
                let mut att = vec![f32::NEG_INFINITY; t];
                for ki in 0..=qi {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
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

                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for ki in 0..=qi {
                        sum += att[ki] * v_all[ki][offset + d];
                    }
                    out[qi][offset + d] = sum;
                }
            }
        }

        out.iter()
            .map(|o| self.linear(&weights.c_proj.w, &weights.c_proj.b, o))
            .collect()
    }

    fn per_band_linear(&self, weights: &PerBandLinearWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let t = x.len();
        let n_bands = weights.band_w.len();
        let n_embd = n_bands * 2;
        let mut result = Vec::with_capacity(t);

        for pos in 0..t {
            let mut bands_out = vec![0.0f32; n_embd];
            for band in 0..n_bands {
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
        let n_embd = x[0].len();
        let mut result = Vec::with_capacity(t);

        for pos in 0..t {
            let kerr_out = self.kerr_ode(&weights.kerr, &x[pos]);
            let maestro_out = self.maestro(&weights.maestro, &x[pos]);
            let mut combined = vec![0.0f32; n_embd];
            for i in 0..n_embd {
                combined[i] = kerr_out[i] + maestro_out[i];
            }
            let projected = self.linear(&weights.out_proj.w, &weights.out_proj.b, &combined);
            result.push(projected);
        }

        result
    }

    fn linear_backward_dx(&self, d_y: &[f32], w: &[Vec<f32>]) -> Vec<f32> {
        let out_dim = w.len();
        let in_dim = if out_dim > 0 { w[0].len() } else { return vec![] };
        let mut w_flat = Vec::with_capacity(out_dim * in_dim);
        for row in w { w_flat.extend_from_slice(row); }

        let w_buf = self.storage_buf("bwd_w", &w_flat);
        let dy_buf = self.storage_buf("bwd_dy", d_y);
        let dx_buf = self.output_buf("bwd_dx", in_dim);
        let params = MatvecBwdParams {
            out_dim: out_dim as u32, in_dim: in_dim as u32, _pad1: 0, _pad2: 0,
        };
        let params_buf = self.uniform_buf("bwd_params", &params);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matvec_bwd_bg"),
            layout: &self.matvec_bwd_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dy_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dx_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.matvec_bwd_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (in_dim as u32 + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.readback(&dx_buf, in_dim)
    }

    fn layer_norm_backward(&self, d_y: &[f32], x: &[f32], weight: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let dim = x.len();
        let dy_buf = self.storage_buf("lnb_dy", d_y);
        let x_buf = self.storage_buf("lnb_x", x);
        let w_buf = self.storage_buf("lnb_w", weight);
        let out_buf = self.output_buf("lnb_out", dim * 3); // d_x, d_weight, d_bias concatenated
        let params = LayerNormParams { dim: dim as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
        let params_buf = self.uniform_buf("lnb_params", &params);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer_norm_bwd_bg"),
            layout: &self.layer_norm_bwd_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dy_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.layer_norm_bwd_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1); // single workgroup handles all
        }
        self.queue.submit(Some(encoder.finish()));

        let result = self.readback(&out_buf, dim * 3);
        let d_x = result[..dim].to_vec();
        let d_weight = result[dim..dim * 2].to_vec();
        let d_bias = result[dim * 2..].to_vec();
        (d_x, d_weight, d_bias)
    }

    fn gelu_backward(&self, d_y: &[f32], x: &[f32]) -> Vec<f32> {
        let n = x.len();
        let dy_buf = self.storage_buf("gb_dy", d_y);
        let x_buf = self.storage_buf("gb_x", x);
        let dx_buf = self.output_buf("gb_dx", n);
        let params = GeluBwdParams { len: n as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
        let params_buf = self.uniform_buf("gb_params", &params);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gelu_bwd_bg"),
            layout: &self.gelu_bwd_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dy_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dx_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.gelu_bwd_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n as u32 + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.readback(&dx_buf, n)
    }

    fn linear_backward_dx_batch(&self, d_y: &[Vec<f32>], w: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n_pos = d_y.len();
        let out_dim = w.len();
        let in_dim = if out_dim > 0 { w[0].len() } else { return vec![vec![]; n_pos] };

        // Flatten W and d_y
        let mut w_flat = Vec::with_capacity(out_dim * in_dim);
        for row in w { w_flat.extend_from_slice(row); }
        let dy_flat: Vec<f32> = d_y.iter().flat_map(|v| v.iter().copied()).collect();

        let w_buf = self.storage_buf("mvbb_w", &w_flat);
        let dy_buf = self.storage_buf("mvbb_dy", &dy_flat);
        let dx_buf = self.output_buf("mvbb_dx", n_pos * in_dim);
        let params = MatvecBwdBatchParams {
            out_dim: out_dim as u32, in_dim: in_dim as u32,
            n_pos: n_pos as u32, _pad: 0,
        };
        let params_buf = self.uniform_buf("mvbb_params", &params);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mvbb_bg"),
            layout: &self.matvec_bwd_batch_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dy_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dx_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.matvec_bwd_batch_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let total = (n_pos * in_dim) as u32;
            pass.dispatch_workgroups((total + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        let dx_flat = self.readback(&dx_buf, n_pos * in_dim);
        dx_flat.chunks(in_dim).map(|c| c.to_vec()).collect()
    }

    fn outer_product_accum(
        &self,
        d_y: &[Vec<f32>],
        x: &[Vec<f32>],
        compute_bias: bool,
    ) -> (Vec<Vec<f32>>, Vec<f32>) {
        let n_pos = d_y.len();
        let out_dim = d_y[0].len();
        let in_dim = x[0].len();

        // Flatten inputs: d_y[pos][i] → d_y_flat[pos * out_dim + i]
        let d_y_flat: Vec<f32> = d_y.iter().flat_map(|v| v.iter().copied()).collect();
        let x_flat: Vec<f32> = x.iter().flat_map(|v| v.iter().copied()).collect();

        let dy_buf = self.storage_buf("op_dy", &d_y_flat);
        let x_buf = self.storage_buf("op_x", &x_flat);
        let dw_buf = self.output_buf("op_dw", out_dim * in_dim);
        // d_b buffer — always create it (even if not used) for binding
        let db_buf = self.output_buf("op_db", out_dim);
        let params = OuterProductParams {
            out_dim: out_dim as u32,
            in_dim: in_dim as u32,
            n_pos: n_pos as u32,
            compute_bias: if compute_bias { 1 } else { 0 },
        };
        let params_buf = self.uniform_buf("op_params", &params);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outer_product_bg"),
            layout: &self.outer_product_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dy_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dw_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: db_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.outer_product_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per output row
            pass.dispatch_workgroups(out_dim as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        // Readback and unflatten
        let dw_flat = self.readback(&dw_buf, out_dim * in_dim);
        let d_w: Vec<Vec<f32>> = dw_flat.chunks(in_dim).map(|c| c.to_vec()).collect();
        let d_b = if compute_bias {
            self.readback(&db_buf, out_dim)
        } else {
            vec![0.0f32; out_dim]
        };

        (d_w, d_b)
    }

    fn attention_backward(
        &self,
        d_pre_proj: &[Vec<f32>],
        q_all: &[Vec<f32>],
        k_all: &[Vec<f32>],
        v_all: &[Vec<f32>],
        att_weights: &[Vec<Vec<f32>>],
        n_head: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let t = d_pre_proj.len();
        let n_embd = d_pre_proj[0].len();
        let head_dim = n_embd / n_head;

        // Flatten inputs to contiguous f32 arrays
        let d_out_flat: Vec<f32> = d_pre_proj.iter().flat_map(|v| v.iter().copied()).collect();
        let q_flat: Vec<f32> = q_all.iter().flat_map(|v| v.iter().copied()).collect();
        let k_flat: Vec<f32> = k_all.iter().flat_map(|v| v.iter().copied()).collect();
        let v_flat: Vec<f32> = v_all.iter().flat_map(|v| v.iter().copied()).collect();

        // att_weights: [n_head][T][T] → flatten
        // att_weights[head][pos][ki] → att_flat[head * T * T + pos * T + ki]
        let mut att_flat = vec![0.0f32; n_head * t * t];
        for (head, head_weights) in att_weights.iter().enumerate() {
            for (pos, pos_weights) in head_weights.iter().enumerate() {
                for (ki, &w) in pos_weights.iter().enumerate() {
                    att_flat[head * t * t + pos * t + ki] = w;
                }
            }
        }

        // Create GPU buffers
        let d_out_buf = self.storage_buf("ab_d_out", &d_out_flat);
        let q_buf = self.storage_buf("ab_q", &q_flat);
        let k_buf = self.storage_buf("ab_k", &k_flat);
        let v_buf = self.storage_buf("ab_v", &v_flat);
        let att_buf = self.storage_buf("ab_att", &att_flat);
        let dq_buf = self.output_buf("ab_dq", t * n_embd);
        let dk_buf = self.output_buf("ab_dk", t * n_embd);
        let dv_buf = self.output_buf("ab_dv", t * n_embd);
        let dscore_buf = self.output_buf("ab_dscore", t * n_head * t);

        let params = AttnBwdParams {
            seq_len: t as u32,
            n_head: n_head as u32,
            head_dim: head_dim as u32,
            n_embd: n_embd as u32,
        };
        let params_buf = self.uniform_buf("ab_params", &params);
        // Dispatch 2 needs its own uniform buffer (same data, different bind group)
        let params_buf2 = self.uniform_buf("ab_params2", &params);

        // ─── Dispatch 1: attn_backward_scores ──────────────────────
        // One workgroup per (pos, head)
        let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attn_bwd_scores_bg"),
            layout: &self.attn_bwd_scores_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: d_out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: att_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: dq_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: dscore_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: params_buf.as_entire_binding() },
            ],
        });

        // ─── Dispatch 2: attn_backward_dkv ─────────────────────────
        // One thread per (ki, d_global)
        let bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attn_bwd_dkv_bg"),
            layout: &self.attn_bwd_dkv_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: d_out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: att_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dscore_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dk_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: dv_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: params_buf2.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            // Dispatch 1: (T, n_head, 1) workgroups of 64 threads each
            pass.set_pipeline(&self.attn_bwd_scores_pipeline);
            pass.set_bind_group(0, &bg1, &[]);
            pass.dispatch_workgroups(t as u32, n_head as u32, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            // Dispatch 2: ceil(T * n_embd / 64) workgroups of 64 threads
            pass.set_pipeline(&self.attn_bwd_dkv_pipeline);
            pass.set_bind_group(0, &bg2, &[]);
            let total_threads = (t * n_embd) as u32;
            pass.dispatch_workgroups((total_threads + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        // Readback d_q, d_k, d_v and unflatten
        let dq_flat = self.readback(&dq_buf, t * n_embd);
        let dk_flat = self.readback(&dk_buf, t * n_embd);
        let dv_flat = self.readback(&dv_buf, t * n_embd);

        let unflatten = |flat: Vec<f32>| -> Vec<Vec<f32>> {
            flat.chunks(n_embd).map(|c| c.to_vec()).collect()
        };

        (unflatten(dq_flat), unflatten(dk_flat), unflatten(dv_flat))
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
            rk4_n_steps: RK4_N_STEPS,
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
        rk4_n_steps: RK4_N_STEPS,
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
