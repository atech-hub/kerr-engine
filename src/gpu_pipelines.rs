//! GPU pipeline setup — struct definition, shader compilation, and dispatch helpers.
//!
//! This module contains GpuBackend's struct, constructors (new, with_device_index,
//! from_adapter), uniform param structs, bind group layout helpers, and low-level
//! GPU dispatch methods (gpu_matvec, gpu_layer_norm, gpu_kerr_derivative, gpu_kerr_ode).

use wgpu::util::DeviceExt;

use std::sync::Mutex;

use crate::model::*;
use crate::gpu_buffers::GpuBufferPool;

/// GPU backend — dispatches to WGSL compute shaders.
pub struct GpuBackend {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    /// Buffer pool for reusable GPU buffers (eliminates per-dispatch allocation).
    pub(crate) pool: Mutex<GpuBufferPool>,
    /// Resident weight buffers — uploaded once, updated after Adam step.
    pub(crate) resident: Mutex<Option<crate::gpu_resident::ResidentWeightBuffers>>,
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
    // Batched Kerr derivative backward: all positions in one dispatch
    pub(crate) kerr_bwd_batch_pipeline: wgpu::ComputePipeline,
    pub(crate) kerr_bwd_batch_layout: wgpu::BindGroupLayout,
    // Fused RK4 utilities: vec_scale_add and rk4_combine for chained dispatch
    pub(crate) vec_scale_add_pipeline: wgpu::ComputePipeline,
    pub(crate) vec_scale_add_layout: wgpu::BindGroupLayout,
    pub(crate) rk4_combine_pipeline: wgpu::ComputePipeline,
    pub(crate) rk4_combine_layout: wgpu::BindGroupLayout,
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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct VecScaleAddParams {
    pub len: u32,
    pub scale: f32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Rk4CombineParams {
    pub len: u32,
    pub dt_over_6: f32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct KerrBwdBatchParams {
    pub n_bands: u32,
    pub n_pos: u32,
    pub alpha: f32,
    pub beta: f32,
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
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 12,
                    ..wgpu::Limits::default()
                },
                ..Default::default()
            },
            None,
        ))
        .expect("Failed to get GPU device");

        // ─── Compile matvec shader (tiled workgroup reduction) ──────
        // 64 threads per workgroup, each accumulates in_dim/64 terms,
        // then tree-reduce. Error O(in_dim/64 + log2(64)) vs O(in_dim).
        // One workgroup PER output row (dispatch = out_dim workgroups).
        let matvec_src = include_str!("../shaders/matvec_tiled.wgsl");
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

        // matvec_backward: d_x = W^T @ d_y (tiled reduction)
        let mvb_src = include_str!("../shaders/matvec_backward_tiled.wgsl");
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
        let mvbb_src = include_str!("../shaders/matvec_backward_batch_tiled.wgsl");
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

        // ─── Compile batched matvec forward shader (tiled reduction) ──
        println!("  Precision: tiled workgroup reduction (stable at 768-dim+)");
        let mvb_fwd_src = include_str!("../shaders/matvec_batch_tiled.wgsl");
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

        // ─── Compile batched Kerr backward shader ─────────────────
        let kbb_src = include_str!("../shaders/kerr_backward_batch.wgsl");
        let kbb_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kerr_backward_batch"),
            source: wgpu::ShaderSource::Wgsl(kbb_src.into()),
        });
        // 13 bindings: r, s, gamma, omega, d_dr, d_ds, d_r, d_s, d_gamma, d_omega, d_alpha, d_beta, params
        let kerr_bwd_batch_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("kerr_bwd_batch_layout"),
            entries: &[
                storage_ro(0), storage_ro(1), storage_ro(2), storage_ro(3),  // r, s, gamma, omega
                storage_ro(4), storage_ro(5),                                 // d_dr, d_ds
                storage_rw(6), storage_rw(7),                                 // d_r, d_s
                storage_rw(8), storage_rw(9),                                 // d_gamma, d_omega
                storage_rw(10), storage_rw(11),                               // d_alpha_partial, d_beta_partial
                uniform_entry(12),                                            // params
            ],
        });
        let kbb_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("kerr_bwd_batch_pl"),
            bind_group_layouts: &[&kerr_bwd_batch_layout],
            push_constant_ranges: &[],
        });
        let kerr_bwd_batch_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("kerr_bwd_batch_pipeline"),
            layout: Some(&kbb_pl),
            module: &kbb_module,
            entry_point: Some("kerr_backward_batch"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ─── Compile vec_scale_add shader (y = a + scale * b) ───
        let vsa_src = include_str!("../shaders/vec_scale_add.wgsl");
        let vsa_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vec_scale_add"),
            source: wgpu::ShaderSource::Wgsl(vsa_src.into()),
        });
        let vec_scale_add_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vec_scale_add_layout"),
            entries: &[storage_ro(0), storage_ro(1), storage_rw(2), uniform_entry(3)],
        });
        let vsa_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vec_scale_add_pl"),
            bind_group_layouts: &[&vec_scale_add_layout],
            push_constant_ranges: &[],
        });
        let vec_scale_add_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vec_scale_add_pipeline"),
            layout: Some(&vsa_pl),
            module: &vsa_module,
            entry_point: Some("vec_scale_add"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ─── Compile rk4_combine shader (y = base + dt/6*(k1 + 2*k2 + 2*k3 + k4)) ───
        let rc_src = include_str!("../shaders/rk4_combine.wgsl");
        let rc_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rk4_combine"),
            source: wgpu::ShaderSource::Wgsl(rc_src.into()),
        });
        let rk4_combine_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rk4_combine_layout"),
            entries: &[storage_ro(0), storage_ro(1), storage_ro(2), storage_ro(3), storage_ro(4), storage_rw(5), uniform_entry(6)],
        });
        let rc_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rk4_combine_pl"),
            bind_group_layouts: &[&rk4_combine_layout],
            push_constant_ranges: &[],
        });
        let rk4_combine_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rk4_combine_pipeline"),
            layout: Some(&rc_pl),
            module: &rc_module,
            entry_point: Some("rk4_combine"),
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
            kerr_bwd_batch_pipeline,
            kerr_bwd_batch_layout,
            vec_scale_add_pipeline,
            vec_scale_add_layout,
            rk4_combine_pipeline,
            rk4_combine_layout,
            pool: Mutex::new(GpuBufferPool::new()),
            resident: Mutex::new(None),
        }
    }

    /// Invalidate cached weight buffers. Call after optimizer step.
    pub fn invalidate_weight_cache(&self) {
        self.pool.lock().unwrap().invalidate_weights();
    }


}
