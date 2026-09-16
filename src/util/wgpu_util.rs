//! Utility functions for working with wgpu
use crate::util::tone_map::ToneMapper;
use bytemuck::AnyBitPattern;
use once_cell::sync::Lazy;
use simple_error::SimpleError;
use std::error::Error;
use std::num::NonZeroU64;

pub(crate) enum BindingType {
    Storage {
        read_only: bool,
        min_binding_size: u64,
    },
    Uniform {
        min_binding_size: u64,
    },
    Texture {
        view_dimension: wgpu::TextureViewDimension,
    },
    Sampler,
}

pub(crate) struct BindingInfo {
    pub binding_type: BindingType,
}

static DEVICE_AND_QUEUE: Lazy<(wgpu::Device, wgpu::Queue)> =
    Lazy::new(|| create_wgpu_device_and_queue().expect("Failed to create device and queue"));

/// Returns the global WGPU device and queue
pub fn get_wgpu_device_and_queue() -> &'static (wgpu::Device, wgpu::Queue) {
    &DEVICE_AND_QUEUE
}

fn create_wgpu_device_and_queue() -> Result<(wgpu::Device, wgpu::Queue), Box<dyn Error>> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;

    println!("Got Compute Adapter: {:#?}", adapter.get_info());

    let downlevel_capabilities = adapter.get_downlevel_capabilities();
    if !downlevel_capabilities
        .flags
        .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
    {
        return Err(SimpleError::new("Adapter does not support compute shaders").into());
    }

    // The default limits cap storage buffers at 8 per stage, which the scene
    // bindings alone now fill. Ask for what the adapter actually offers so the
    // hot/cold primitive buffers fit; native backends expose far more.
    let mut required_limits = adapter.limits();
    if required_limits.max_storage_buffers_per_shader_stage < 12 {
        return Err(SimpleError::new(format!(
            "Adapter supports only {} storage buffers per stage, 12 are required",
            required_limits.max_storage_buffers_per_shader_stage
        ))
        .into());
    }
    // Keep the rest conservative -- only the binding count needs raising.
    required_limits.max_storage_buffers_per_shader_stage =
        required_limits.max_storage_buffers_per_shader_stage.min(16);

    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        required_limits,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| SimpleError::new(format!("Failed to create device: {}", e)).into())
}

/// Reads the content of a buffer from the GPU
pub fn get_result_from_buffer<T: AnyBitPattern>(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
) -> Vec<T> {
    let buffer_slice = buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let result = {
        let data = buffer_slice.get_mapped_range();
        bytemuck::cast_slice(&data).to_vec()
    };
    buffer.unmap();
    result
}

pub(crate) fn add_compute_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroup_count_x: u32,
) {
    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(pipeline);
    compute_pass.set_bind_group(0, bind_group, &[]);
    compute_pass.dispatch_workgroups(workgroup_count_x, 1, 1);
}

/// Dispatches over a 2-D workgroup grid.
///
/// The tracer needs this both for coherence (a workgroup covering an 8x8 tile
/// of pixels traces far more similar rays than one covering 64 pixels of a
/// single scanline) and for reach: a 1-D dispatch runs into
/// `max_compute_workgroups_per_dimension` at ~4.2M pixels, so 4K was not
/// renderable at all.
pub(crate) fn add_compute_pass_2d(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroup_count_x: u32,
    workgroup_count_y: u32,
) {
    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(pipeline);
    compute_pass.set_bind_group(0, bind_group, &[]);
    compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
}

pub(crate) fn compute_pipeline<'a>(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
    constants: &'a [(&'a str, f64)],
) -> wgpu::ComputePipeline {
    compute_pipeline_with_entry(device, bind_group_layout, module, "compute", constants)
}

/// As [`compute_pipeline`], but for a module that declares more than one entry
/// point. Sharing a module lets two related passes share their helper functions,
/// which WGSL offers no other way to do -- `include_wgsl!` is `include_str!` and
/// there is no include directive.
pub(crate) fn compute_pipeline_with_entry<'a>(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    constants: &'a [(&'a str, f64)],
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout(device, bind_group_layout)),
        module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants,
            ..Default::default()
        },
        cache: None,
    })
}

pub(crate) fn bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    resources: &[wgpu::BindingResource],
) -> wgpu::BindGroup {
    let entries = resources
        .iter()
        .enumerate()
        .map(|(i, r)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: r.clone(),
        })
        .collect::<Vec<_>>();

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &entries,
    })
}

pub(crate) fn bind_group_layout(
    device: &wgpu::Device,
    entry_infos: &[BindingInfo],
) -> wgpu::BindGroupLayout {
    let entries = entry_infos
        .iter()
        .enumerate()
        .map(|(i, e)| bind_group_layout_entry0(i as u32, e))
        .collect::<Vec<_>>();

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &entries,
    })
}

pub(crate) fn storage_binding(read_only: bool, min_binding_size: u64) -> BindingInfo {
    BindingInfo {
        binding_type: BindingType::Storage {
            read_only,
            min_binding_size,
        },
    }
}

pub(crate) fn uniform_binding(min_binding_size: u64) -> BindingInfo {
    BindingInfo {
        binding_type: BindingType::Uniform { min_binding_size },
    }
}

pub(crate) fn texture_binding(view_dimension: wgpu::TextureViewDimension) -> BindingInfo {
    BindingInfo {
        binding_type: BindingType::Texture { view_dimension },
    }
}

pub(crate) fn sampler_binding() -> BindingInfo {
    BindingInfo {
        binding_type: BindingType::Sampler,
    }
}

fn bind_group_layout_entry0(binding: u32, info: &BindingInfo) -> wgpu::BindGroupLayoutEntry {
    let ty = match info.binding_type {
        BindingType::Storage {
            read_only,
            min_binding_size,
        } => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            min_binding_size: NonZeroU64::new(min_binding_size),
            has_dynamic_offset: false,
        },
        BindingType::Uniform { min_binding_size } => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            min_binding_size: NonZeroU64::new(min_binding_size),
            has_dynamic_offset: false,
        },
        BindingType::Texture { view_dimension } => wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension,
            multisampled: false,
        },
        BindingType::Sampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
    };

    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

fn pipeline_layout(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    })
}

/// Converts a wgpu buffer of linear HDR radiance to an RgbImage.
///
/// This is the display transform: `tone_mapper` brings unbounded radiance into
/// `[0, 1]`, then gamma encoding makes it displayable. Everything upstream --
/// the accumulator, bloom, the denoiser -- works on the untouched linear
/// values, so the curve chosen here changes only what is shown, never what is
/// computed.
pub fn buffer_to_image(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    width: u32,
    height: u32,
    tone_mapper: ToneMapper,
) -> image::RgbImage {
    let size = (width * height * 16) as u64;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
    queue.submit(Some(encoder.finish()));

    let result: Vec<[f32; 4]> = get_result_from_buffer(device, &staging_buffer);

    let mut img = image::RgbImage::new(width, height);
    for (i, pixel) in result.iter().enumerate() {
        let x = (i as u32) % width;
        let y = (i as u32) / width;
        if x < width && y < height {
            let mapped = tone_mapper.map([pixel[0], pixel[1], pixel[2]]);
            // Gamma 2.0, and the 0.999 ceiling so the `* 256` below cannot
            // reach 256 and wrap the cast to u8.
            let encode = |v: f32| (v.sqrt().min(0.999) * 256.0) as u8;
            img.put_pixel(
                x,
                y,
                image::Rgb([encode(mapped[0]), encode(mapped[1]), encode(mapped[2])]),
            );
        }
    }
    img
}
