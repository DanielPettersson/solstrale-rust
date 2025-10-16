use crate::geo::vec3::Vec3;
use crate::post::{pixel_colors_to_rgb_image, PostProcessor, PostProcessors};
use crate::util::gaussian::create_gaussian_blur_weights;
use bytemuck::{Pod, Zeroable};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::error::Error;
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

#[derive(Clone)]
/// Applies a bloom effect on the pixel colors
pub struct BloomPostProcessor {
    kernel_size_fraction: f64,
    threshold: f64,
    max_intensity: f64,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct BloomFilterBrightConfig {
    threshold: f32,
    max_intensity: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct BloomApplyConfig {
    width: u32,
    x_dir: i32,
    y_dir: i32,
}

impl BloomPostProcessor {
    #![allow(clippy::new_ret_no_self)]
    /// Create a new bloom post processor
    /// # Arguments
    /// * `kernel_size_fraction` Radius of the blur effect, as a fraction of the rendered image's width
    /// * `threshold` Color intensity threshold for applying bloom effect. If not specified, defaults to "white"
    /// * `max_intensity` Maximum color intensity of the bloom effect. If not specified, defaults to unlimited
    pub fn new(
        kernel_size_fraction: f64,
        threshold: Option<f64>,
        max_intensity: Option<f64>,
    ) -> Result<PostProcessors, simple_error::SimpleError> {
        if !(0. ..=0.5).contains(&kernel_size_fraction) {
            return Err(simple_error::SimpleError::new(
                "kernel_size_fraction must be between 0 and 0.5",
            ));
        }

        let threshold = threshold.unwrap_or(Vec3::new(1., 1., 1.).length());
        let max_intensity = max_intensity.unwrap_or(1000.);

        Ok(PostProcessors::from(BloomPostProcessor {
            kernel_size_fraction,
            threshold,
            max_intensity,
        }))
    }
}

impl PostProcessor for BloomPostProcessor {
    fn post_process(
        &self,
        pixel_colors: &[Vec3],
        albedo_colors: &[Vec3],
        normal_colors: &[Vec3],
        width: u32,
        height: u32,
        num_samples: u32,
    ) -> Result<image::RgbImage, Box<dyn Error>> {
        let pixel_colors = self.intermediate_post_process(
            pixel_colors,
            albedo_colors,
            normal_colors,
            width,
            height,
            num_samples,
        )?;
        Ok(pixel_colors_to_rgb_image(
            &pixel_colors,
            width,
            height,
            num_samples,
        ))
    }

    #[allow(clippy::needless_range_loop)]
    fn intermediate_post_process(
        &self,
        pixel_colors: &[Vec3],
        _albedo_colors: &[Vec3],
        _normal_colors: &[Vec3],
        width: u32,
        _height: u32,
        num_samples: u32,
    ) -> Result<Vec<Vec3>, Box<dyn Error>> {
        let threshold = self.threshold * num_samples as f64;
        let max_intensity = self.max_intensity * num_samples as f64;
        let kernel_size = (self.kernel_size_fraction * width as f64) as usize * 2 + 1;
        let weights = create_gaussian_blur_weights(kernel_size, kernel_size as f32 / 5.);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("Failed to create adapter");

        let downlevel_capabilities = adapter.get_downlevel_capabilities();
        if !downlevel_capabilities
            .flags
            .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
        {
            panic!("Adapter does not support compute shaders");
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .expect("Failed to create device");

        let bloom_apply_module = device.create_shader_module(wgpu::include_wgsl!("bloom_apply.wgsl"));
        let bloom_filter_bright_module = device.create_shader_module(wgpu::include_wgsl!("bloom_filter_bright.wgsl"));
        let bloom_add_module = device.create_shader_module(wgpu::include_wgsl!("bloom_add.wgsl"));

        let weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&weights),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let input_pixels: Vec<[f32; 4]> = pixel_colors.iter().map(|p| p.into()).collect();
        let input_pixels_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&input_pixels),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let intermediate_buffer1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: input_pixels_buffer.size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let intermediate_buffer2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: input_pixels_buffer.size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let output_pixels_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: input_pixels_buffer.size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: input_pixels_buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bloom_apply_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Config buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(NonZeroU64::new(size_of::<BloomApplyConfig>() as u64).unwrap()),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                // Weights buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                // Input pixels buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(NonZeroU64::new(16).unwrap()),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                // Output pixels buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: Some(NonZeroU64::new(16).unwrap()),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });

        let bloom_filter_bright_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Config buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(NonZeroU64::new(size_of::<BloomFilterBrightConfig>() as u64).unwrap()),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                // Input pixels buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(NonZeroU64::new(16).unwrap()),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                // Output pixels buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: Some(NonZeroU64::new(16).unwrap()),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });

        let bloom_add_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Input pixels buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(NonZeroU64::new(16).unwrap()),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                // Bloom pixels buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(NonZeroU64::new(16).unwrap()),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },

                // Output pixels buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: Some(NonZeroU64::new(16).unwrap()),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });

        let bloom_filter_bright_config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&BloomFilterBrightConfig {
                threshold: threshold as f32,
                max_intensity: max_intensity as f32,
            }),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bloom_filter_bright_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bloom_filter_bright_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bloom_filter_bright_config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_pixels_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: intermediate_buffer1.as_entire_binding(),
                },
            ],
        });

        // First pass: horizontal (x_dir = 1, y_dir = 0)
        let bloom_apply_config_buffer_x = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&BloomApplyConfig {
                width,
                x_dir: 1,
                y_dir: 0,
            }),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bloom_apply_bind_group_x = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bloom_apply_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bloom_apply_config_buffer_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: intermediate_buffer1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: intermediate_buffer2.as_entire_binding(),
                },
            ],
        });

        // Second pass: vertical (x_dir = 0, y_dir = 1)
        let bloom_apply_config_buffer_y = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&BloomApplyConfig {
                width,
                x_dir: 0,
                y_dir: 1,
            }),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bloom_apply_bind_group_y = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bloom_apply_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bloom_apply_config_buffer_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: intermediate_buffer2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: intermediate_buffer1.as_entire_binding(),
                },
            ],
        });

        let bloom_add_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bloom_add_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_pixels_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: intermediate_buffer1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_pixels_buffer.as_entire_binding(),
                },
            ],
        });

        let bloom_apply_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bloom_apply_bind_group_layout],
            push_constant_ranges: &[],
        });

        let bloom_filter_bright_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bloom_filter_bright_bind_group_layout],
            push_constant_ranges: &[],
        });

        let bloom_add_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bloom_add_bind_group_layout],
            push_constant_ranges: &[],
        });

        let bloom_apply_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&bloom_apply_pipeline_layout),
            module: &bloom_apply_module,
            entry_point: Some("bloom"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let bloom_filter_bright_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&bloom_filter_bright_pipeline_layout),
            module: &bloom_filter_bright_module,
            entry_point: Some("bloom"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let bloom_add_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&bloom_add_pipeline_layout),
            module: &bloom_add_module,
            entry_point: Some("bloom"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let workgroup_count = pixel_colors.len().div_ceil(64);

        // Filter bright-pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&bloom_filter_bright_pipeline);
            compute_pass.set_bind_group(0, &bloom_filter_bright_bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

        // First apply-pass (horizontal)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&bloom_apply_pipeline);
            compute_pass.set_bind_group(0, &bloom_apply_bind_group_x, &[]);
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

        // Second apply-pass (vertical)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&bloom_apply_pipeline);
            compute_pass.set_bind_group(0, &bloom_apply_bind_group_y, &[]);
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

        // Add-pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&bloom_add_pipeline);
            compute_pass.set_bind_group(0, &bloom_add_bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &output_pixels_buffer,
            0,
            &download_buffer,
            0,
            output_pixels_buffer.size(),
        );
        let command_buffer = encoder.finish();
        queue.submit([command_buffer]);

        let buffer_slice = download_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = buffer_slice.get_mapped_range();
        let result: &[[f32; 4]] = bytemuck::cast_slice(&data);

        Ok(result.par_iter().map(|d| d.into()).collect())
    }

    fn needs_albedo_and_normal_colors(&self) -> bool {
        false
    }
}
