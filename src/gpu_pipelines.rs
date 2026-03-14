//! GPU pipeline setup — struct definition, shader compilation, and dispatch helpers.
//!
//! This module contains GpuBackend's struct, constructors (new, with_device_index,
//! from_adapter), uniform param structs, bind group layout helpers, and low-level
//! GPU dispatch methods (gpu_matvec, gpu_layer_norm, gpu_kerr_derivative, gpu_kerr_ode).

use wgpu::util::DeviceExt;

use crate::model::*;

/// GPU backend — dispatches to WGSL compute shaders.
pub struct GpuBackend {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) matvec_pipeline: wgpu::ComputePipeline,
    pub(crate) matvec_layout: wgpu::BindGroupLayout,
    pub(crate) layer_norm_pipeline: wgpu::ComputePipeline,
    pub(crate) layer_norm_layout: wgpu::BindGroupLayout,
    pub(crate) kerr_deriv_pipeline: wgpu::ComputePipeline,
    pub(crate) kerr_deriv_layout: wgpu::BindGroupLayout,
    // Backward shaders
    pub(crate) matvec_bwd_pipeline: wgpu::ComputePipeline,
    pub(crate) matvec_bwd_layout: wgpu::BindGroupLayout,
    pub(crate) layer_norm_bwd_pipeline: wgpu::ComputePipeline,
    pub(crate) layer_norm_bwd_layout: wgpu::BindGroupLayout,
    #[allow(dead_code)] // Ready for FFN backward restructuring
    pub(crate) gelu_bwd_pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    pub(crate) gelu_bwd_layout: wgpu::BindGroupLayout,
    // Attention backward (two dispatches)
    pub(crate) attn_bwd_scores_pipeline: wgpu::ComputePipeline,
    pub(crate) attn_bwd_scores_layout: wgpu::BindGroupLayout,
    pub(crate) attn_bwd_dkv_pipeline: wgpu::ComputePipeline,
    pub(crate) attn_bwd_dkv_layout: wgpu::BindGroupLayout,
    // Batched outer product: d_w = D_Y^T @ X
    pub(crate) outer_product_pipeline: wgpu::ComputePipeline,
    pub(crate) outer_product_layout: wgpu::BindGroupLayout,
    // Batched matvec backward: d_x[pos] = W^T @ d_y[pos] for all positions
    pub(crate) matvec_bwd_batch_pipeline: wgpu::ComputePipeline,
    pub(crate) matvec_bwd_batch_layout: wgpu::BindGroupLayout,
    // Batched matvec forward: y[pos] = W @ x[pos] + b for all positions
    pub(crate) matvec_batch_pipeline: wgpu::ComputePipeline,
    pub(crate) matvec_batch_layout: wgpu::BindGroupLayout,
    // Batched layer norm: one workgroup per position
    pub(crate) layer_norm_batch_pipeline: wgpu::ComputePipeline,
    pub(crate) layer_norm_batch_layout: wgpu::BindGroupLayout,
    // Batched Kerr derivative: all positions in one dispatch
    pub(crate) kerr_deriv_batch_pipeline: wgpu::ComputePipeline,
    pub(crate) kerr_deriv_batch_layout: wgpu::BindGroupLayout,
}

// ─── Uniform param structs ──────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MatvecParams {
    pub out_dim: u32,
    pub in_dim: u32,
    pub use_bias: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LayerNormParams {
    pub dim: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct KerrDerivParams {
    pub n_bands: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MatvecBwdParams {
    pub out_dim: u32,
    pub in_dim: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GeluBwdParams {
    pub len: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct AttnBwdParams {
    pub seq_len: u32,
    pub n_head: u32,
    pub head_dim: u32,
    pub n_embd: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OuterProductParams {
    pub out_dim: u32,
    pub in_dim: u32,
    pub n_pos: u32,
    pub compute_bias: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MatvecBwdBatchParams {
    pub out_dim: u32,
    pub in_dim: u32,
    pub n_pos: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MatvecBatchParams {
    pub out_dim: u32,
    pub in_dim: u32,
    pub n_pos: u32,
    pub use_bias: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LayerNormBatchParams {
    pub dim: u32,
    pub n_pos: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct KerrDerivBatchParams {
    pub n_bands: u32,
    pub n_pos: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

// ─── Helper: build bind group layout entries ────────────────────

pub(crate) fn storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

pub(crate) fn storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

pub(crate) fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

        // ─── Compile batched matvec forward shader ─────────────────
        let mvb_fwd_src = include_str!("../shaders/matvec_batch.wgsl");
        let mvb_fwd_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matvec_batch"),
            source: wgpu::ShaderSource::Wgsl(mvb_fwd_src.into()),
        });
        // bindings: 0=w(ro), 1=x(ro), 2=b(ro), 3=y(rw), 4=params(uniform)
        let matvec_batch_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matvec_batch_layout"),
            entries: &[storage_ro(0), storage_ro(1), storage_ro(2), storage_rw(3), uniform_entry(4)],
        });
        let mvb_fwd_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matvec_batch_pl"),
            bind_group_layouts: &[&matvec_batch_layout],
            push_constant_ranges: &[],
        });
        let matvec_batch_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matvec_batch_pipeline"),
            layout: Some(&mvb_fwd_pl),
            module: &mvb_fwd_module,
            entry_point: Some("matvec_batch"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ─── Compile batched layer norm shader ──────────────────────
        let lnb_fwd_src = include_str!("../shaders/layer_norm_batch.wgsl");
        let lnb_fwd_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("layer_norm_batch"),
            source: wgpu::ShaderSource::Wgsl(lnb_fwd_src.into()),
        });
        // bindings: 0=x(ro), 1=weight(ro), 2=bias(ro), 3=y(rw), 4=params(uniform)
        let layer_norm_batch_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layer_norm_batch_layout"),
            entries: &[storage_ro(0), storage_ro(1), storage_ro(2), storage_rw(3), uniform_entry(4)],
        });
        let lnb_fwd_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layer_norm_batch_pl"),
            bind_group_layouts: &[&layer_norm_batch_layout],
            push_constant_ranges: &[],
        });
        let layer_norm_batch_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("layer_norm_batch_pipeline"),
            layout: Some(&lnb_fwd_pl),
            module: &lnb_fwd_module,
            entry_point: Some("layer_norm_batch"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ─── Compile batched Kerr derivative shader ─────────────────
        let kdb_src = include_str!("../shaders/kerr_step_batch.wgsl");
        let kdb_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kerr_derivative_batch"),
            source: wgpu::ShaderSource::Wgsl(kdb_src.into()),
        });
        // bindings: same as kerr_step — 0=r(ro), 1=s(ro), 2=dr(rw), 3=ds(rw),
        //           4=gamma(ro), 5=omega(ro), 6=params(uniform), 7=alpha_beta(ro)
        let kerr_deriv_batch_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("kerr_deriv_batch_layout"),
            entries: &[
                storage_ro(0), storage_ro(1), storage_rw(2), storage_rw(3),
                storage_ro(4), storage_ro(5), uniform_entry(6), storage_ro(7),
            ],
        });
        let kdb_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("kerr_deriv_batch_pl"),
            bind_group_layouts: &[&kerr_deriv_batch_layout],
            push_constant_ranges: &[],
        });
        let kerr_deriv_batch_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("kerr_deriv_batch_pipeline"),
            layout: Some(&kdb_pl),
            module: &kdb_module,
            entry_point: Some("kerr_derivative_batch"),
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
            matvec_batch_pipeline,
            matvec_batch_layout,
            layer_norm_batch_pipeline,
            layer_norm_batch_layout,
            kerr_deriv_batch_pipeline,
            kerr_deriv_batch_layout,
        }
    }

    // ─── GPU dispatch helpers ───────────────────────────────────

    /// Create a read-only storage buffer from f32 data.
    pub(crate) fn storage_buf(&self, label: &str, data: &[f32]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    /// Create a read-write storage buffer of given f32 count.
    pub(crate) fn output_buf(&self, label: &str, n_floats: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (n_floats * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Create a uniform buffer from a Pod struct.
    pub(crate) fn uniform_buf<T: bytemuck::Pod>(&self, label: &str, data: &T) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(data),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    /// Read back f32 data from a GPU buffer.
    pub(crate) fn readback(&self, buf: &wgpu::Buffer, n_floats: usize) -> Vec<f32> {
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
    pub(crate) fn gpu_matvec(&self, w_flat: &[f32], x: &[f32], b: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
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
    pub(crate) fn gpu_layer_norm(&self, x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
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
    pub(crate) fn gpu_kerr_derivative(&self, r: &[f32], s: &[f32], gamma: &[f32], omega: &[f32], alpha: f32, beta: f32) -> (Vec<f32>, Vec<f32>) {
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
    pub(crate) fn gpu_kerr_ode(&self, weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
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

    /// Batched matvec forward: y[pos] = W @ x[pos] + b for all positions in one dispatch.
    pub(crate) fn gpu_matvec_batch(&self, w_flat: &[f32], x_flat: &[f32], b: &[f32], out_dim: usize, in_dim: usize, n_pos: usize) -> Vec<f32> {
        let use_bias = if b.is_empty() { 0u32 } else { 1u32 };

        let w_buf = self.storage_buf("mvb_w", w_flat);
        let x_buf = self.storage_buf("mvb_x", x_flat);
        let b_data = if b.is_empty() { &[0.0f32][..] } else { b };
        let b_buf = self.storage_buf("mvb_b", b_data);
        let y_buf = self.output_buf("mvb_y", n_pos * out_dim);
        let params = MatvecBatchParams {
            out_dim: out_dim as u32, in_dim: in_dim as u32,
            n_pos: n_pos as u32, use_bias,
        };
        let params_buf = self.uniform_buf("mvb_params", &params);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matvec_batch_bg"),
            layout: &self.matvec_batch_layout,
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
            pass.set_pipeline(&self.matvec_batch_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let total = (n_pos * out_dim) as u32;
            pass.dispatch_workgroups((total + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.readback(&y_buf, n_pos * out_dim)
    }

    /// Batched layer norm: one workgroup per position, all positions in one dispatch.
    pub(crate) fn gpu_layer_norm_batch(&self, x_flat: &[f32], weight: &[f32], bias: &[f32], dim: usize, n_pos: usize) -> Vec<f32> {
        let x_buf = self.storage_buf("lnb_x", x_flat);
        let w_buf = self.storage_buf("lnb_w", weight);
        let b_buf = self.storage_buf("lnb_b", bias);
        let y_buf = self.output_buf("lnb_y", n_pos * dim);
        let params = LayerNormBatchParams {
            dim: dim as u32, n_pos: n_pos as u32, _pad1: 0, _pad2: 0,
        };
        let params_buf = self.uniform_buf("lnb_params", &params);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer_norm_batch_bg"),
            layout: &self.layer_norm_batch_layout,
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
            pass.set_pipeline(&self.layer_norm_batch_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_pos as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.readback(&y_buf, n_pos * dim)
    }

    /// Batched Kerr derivative: compute dr/ds for all positions in one dispatch.
    /// r_flat/s_flat are [n_pos * n_bands]. Returns (dr_flat, ds_flat).
    pub(crate) fn gpu_kerr_derivative_batch(
        &self, r_flat: &[f32], s_flat: &[f32],
        gamma: &[f32], omega: &[f32], alpha: f32, beta: f32,
        n_bands: usize, n_pos: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let r_buf = self.storage_buf("kdb_r", r_flat);
        let s_buf = self.storage_buf("kdb_s", s_flat);
        let dr_buf = self.output_buf("kdb_dr", n_pos * n_bands);
        let ds_buf = self.output_buf("kdb_ds", n_pos * n_bands);
        let gamma_buf = self.storage_buf("kdb_gamma", gamma);
        let omega_buf = self.storage_buf("kdb_omega", omega);
        let params = KerrDerivBatchParams {
            n_bands: n_bands as u32, n_pos: n_pos as u32, _pad1: 0, _pad2: 0,
        };
        let params_buf = self.uniform_buf("kdb_params", &params);
        let ab = [alpha, beta];
        let ab_buf = self.storage_buf("kdb_ab", &ab);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("kerr_deriv_batch_bg"),
            layout: &self.kerr_deriv_batch_layout,
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
            pass.set_pipeline(&self.kerr_deriv_batch_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let total = (n_pos * n_bands) as u32;
            pass.dispatch_workgroups((total + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        let dr = self.readback(&dr_buf, n_pos * n_bands);
        let ds = self.readback(&ds_buf, n_pos * n_bands);
        (dr, ds)
    }

    /// Batched Kerr-ODE forward: RK4 integration for all positions simultaneously.
    /// x is [n_pos][n_embd] interleaved. Returns [n_pos][n_embd].
    pub(crate) fn gpu_kerr_ode_batch(&self, weights: &KerrWeights, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n_pos = xs.len();
        let n_bands = weights.gamma_raw.len();
        let n_embd = n_bands * 2;
        let n_steps = weights.rk4_n_steps;
        let dt = 1.0 / n_steps as f32;

        // Unpack all positions: interleaved → separate r, s flat arrays
        let mut r = vec![0.0f32; n_pos * n_bands];
        let mut s = vec![0.0f32; n_pos * n_bands];
        for pos in 0..n_pos {
            for k in 0..n_bands {
                r[pos * n_bands + k] = xs[pos][k * 2];
                s[pos * n_bands + k] = xs[pos][k * 2 + 1];
            }
        }

        fn softplus(x: f32) -> f32 {
            if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
        }
        let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();

        let total = n_pos * n_bands;

        // RK4 integration — 4 derivative evals per step, each batched across all positions
        for _ in 0..n_steps {
            let (k1r, k1s) = self.gpu_kerr_derivative_batch(&r, &s, &gamma, &weights.omega, weights.alpha, weights.beta, n_bands, n_pos);

            let r2: Vec<f32> = r.iter().zip(&k1r).map(|(&ri, &k)| ri + 0.5 * dt * k).collect();
            let s2: Vec<f32> = s.iter().zip(&k1s).map(|(&si, &k)| si + 0.5 * dt * k).collect();
            let (k2r, k2s) = self.gpu_kerr_derivative_batch(&r2, &s2, &gamma, &weights.omega, weights.alpha, weights.beta, n_bands, n_pos);

            let r3: Vec<f32> = r.iter().zip(&k2r).map(|(&ri, &k)| ri + 0.5 * dt * k).collect();
            let s3: Vec<f32> = s.iter().zip(&k2s).map(|(&si, &k)| si + 0.5 * dt * k).collect();
            let (k3r, k3s) = self.gpu_kerr_derivative_batch(&r3, &s3, &gamma, &weights.omega, weights.alpha, weights.beta, n_bands, n_pos);

            let r4: Vec<f32> = r.iter().zip(&k3r).map(|(&ri, &k)| ri + dt * k).collect();
            let s4: Vec<f32> = s.iter().zip(&k3s).map(|(&si, &k)| si + dt * k).collect();
            let (k4r, k4s) = self.gpu_kerr_derivative_batch(&r4, &s4, &gamma, &weights.omega, weights.alpha, weights.beta, n_bands, n_pos);

            for i in 0..total {
                r[i] += dt / 6.0 * (k1r[i] + 2.0 * k2r[i] + 2.0 * k3r[i] + k4r[i]);
                s[i] += dt / 6.0 * (k1s[i] + 2.0 * k2s[i] + 2.0 * k3s[i] + k4s[i]);
            }
        }

        // Re-interleave per position
        (0..n_pos).map(|pos| {
            let mut out = vec![0.0f32; n_embd];
            for k in 0..n_bands {
                out[k * 2] = r[pos * n_bands + k];
                out[k * 2 + 1] = s[pos * n_bands + k];
            }
            out
        }).collect()
    }
}
