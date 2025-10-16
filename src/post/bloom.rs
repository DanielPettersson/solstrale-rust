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
struct Bloom {
    width: u32,
    threshold: f32,
    max_intensity: f32,
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
        let max_intensity = max_intensity.unwrap_or(f64::MAX);

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

        // We first initialize a wgpu `Instance`, which contains any "global" state wgpu needs.
        //
        // This is what loads the vulkan/dx12/metal/opengl libraries.
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        // We then create an `Adapter` which represents a physical gpu in the system. It allows
        // us to query information about it and create a `Device` from it.
        //
        // This function is asynchronous in WebGPU, so request_adapter returns a future. On native/webgl
        // the future resolves immediately, so we can block on it without harm.
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("Failed to create adapter");

        // Print out some basic information about the adapter.
        println!("Running on Adapter: {:#?}", adapter.get_info());

        // Check to see if the adapter supports compute shaders. While WebGPU guarantees support for
        // compute shaders, wgpu supports a wider range of devices through the use of "downlevel" devices.
        let downlevel_capabilities = adapter.get_downlevel_capabilities();
        if !downlevel_capabilities
            .flags
            .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
        {
            panic!("Adapter does not support compute shaders");
        }

        // We then create a `Device` and a `Queue` from the `Adapter`.
        //
        // The `Device` is used to create and manage GPU resources.
        // The `Queue` is a queue used to submit work for the GPU to process.
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .expect("Failed to create device");

        let module = device.create_shader_module(wgpu::include_wgsl!("bloom.wgsl"));

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&Bloom {
                width,
                threshold: threshold as f32,
                max_intensity: max_intensity as f32,
            }),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&weights),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let input_pixels: Vec<[f32; 3]> = pixel_colors.iter().map(|p| p.into()).collect();
        let input_pixels_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&input_pixels),
            usage: wgpu::BufferUsages::STORAGE,
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

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Input buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(NonZeroU64::new(size_of::<Bloom>() as u64).unwrap()),
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_pixels_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_pixels_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("bloom"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_count = pixel_colors.len().div_ceil(64);
        compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);

        drop(compute_pass);

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
        let result: &[[f32; 3]] = bytemuck::cast_slice(&data);

        Ok(result.par_iter().map(|d| d.into()).collect())
    }

    fn needs_albedo_and_normal_colors(&self) -> bool {
        false
    }
}
