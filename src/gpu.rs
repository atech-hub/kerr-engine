//! WGPU device setup, buffer management, and kernel dispatch.

use wgpu::util::DeviceExt;

/// All GPU resources needed to run Kerr-ODE steps.
pub struct KernGpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pub n_bands: u32,
}

/// Parameters passed to the shader as a uniform buffer.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub dt: f32,
    pub gamma: f32,
    pub n_bands: u32,
    pub _pad: u32,
}

impl KernGpu {
    /// Initialize GPU device and compile the Kerr-ODE shader.
    pub fn new(n_bands: u32) -> Self {
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
                label: Some("kerr-engine"),
                ..Default::default()
            },
            None,
        ))
        .expect("Failed to get GPU device");

        // Load shader
        let shader_src = include_str!("../shaders/kerr_step.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kerr_step"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Bind group layout: input (storage read), output (storage rw), params (uniform)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("kerr_bind_group_layout"),
            entries: &[
                // binding 0: input amplitudes (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: output amplitudes (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("kerr_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("kerr_step_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("kerr_step"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            n_bands,
        }
    }

    /// Run one Euler step of the Kerr-ODE on the GPU.
    ///
    /// Takes input amplitudes (2 * n_bands f32 values: [re_0, im_0, re_1, im_1, ...])
    /// Returns output amplitudes in the same format.
    pub fn step(&self, input: &[f32], dt: f32, gamma: f32) -> Vec<f32> {
        let buf_size = (self.n_bands as usize * 2 * std::mem::size_of::<f32>()) as u64;

        // Create buffers
        let input_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = Params {
            dt,
            gamma,
            n_bands: self.n_bands,
            _pad: 0,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Staging buffer for readback
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("kerr_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Dispatch enough workgroups to cover all bands (workgroup size = 64)
            let workgroups = (self.n_bands + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy output to staging
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, buf_size);
        self.queue.submit(Some(encoder.finish()));

        // Read back
        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buf.unmap();

        result
    }
}
