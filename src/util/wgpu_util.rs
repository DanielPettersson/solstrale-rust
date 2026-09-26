//! Utility functions for working with wgpu
use crate::util::gpu_timing::GpuTimer;
use crate::util::luminance::LUMINANCE_WGSL;
use crate::util::tone_map::ToneMapper;
use bytemuck::AnyBitPattern;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use simple_error::SimpleError;
use std::collections::HashMap;
use std::error::Error;
use std::num::NonZeroU64;
use std::sync::Mutex;

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
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

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

    // Timestamp queries are a diagnostic, asked for only when the adapter has
    // them. The base feature covers `ComputePassTimestampWrites`;
    // `TIMESTAMP_QUERY_INSIDE_ENCODERS` buys `CommandEncoder::write_timestamp`,
    // the only way to time the one piece of GPU work in the render loop that is
    // not a compute pass. Asked for separately, since an adapter may have the
    // first without the second.
    let mut required_features = wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    required_features |= adapter.features()
        & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);

    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features,
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
        bytemuck::cast_slice(&data.unwrap()).to_vec()
    };
    buffer.unmap();
    result
}

/// Dispatches over a flat 1-D workgroup grid.
///
/// Test-only: at `workgroup_size(64)` this crosses
/// `max_compute_workgroups_per_dimension` (65535 on a Radeon RX 5700 XT) at
/// 4194240 elements, below 4K, so no pass over the image may use it.
#[cfg(test)]
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
/// The tracer needs this for coherence -- a workgroup covering an 8x8 tile of
/// pixels traces far more similar rays than one covering 64 pixels of a
/// scanline -- and for reach, since a 1-D dispatch runs into
/// `max_compute_workgroups_per_dimension` at ~4.2M pixels.
///
/// Bracketed by a pair of GPU timestamps when a [`GpuTimer`] is passed. `label`
/// names the pass in that report and is handed to wgpu, so the same name
/// identifies it in a graphics debugger.
pub(crate) fn add_compute_pass_2d(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroup_count_x: u32,
    workgroup_count_y: u32,
    timer: Option<&mut GpuTimer>,
    label: &'static str,
) {
    let timestamp_writes = timer.and_then(|t| t.pass_writes(label));

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes,
    });
    compute_pass.set_pipeline(pipeline);
    compute_pass.set_bind_group(0, bind_group, &[]);
    compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
}

/// Creates a shader module with the shared `luminance` declaration spliced
/// ahead of `source`. WGSL has no include directive, so the one definition of
/// the Rec. 709 weights has to be concatenated to reach a second module.
///
/// `source` rather than a path, so a caller that already splices something in
/// -- the denoiser's tone curve -- can hand over the result.
pub(crate) fn shader_module_with_luminance(
    device: &wgpu::Device,
    label: &str,
    source: &str,
) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(format!("{LUMINANCE_WGSL}\n{source}").into()),
    })
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
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    })
}

/// A compiled tone-map-and-pack pipeline and the layout its bind group needs.
#[derive(Clone)]
struct PackPipeline {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

/// The emitted curve, and the image size its override constants are baked with.
type PackKey = (String, u32, u32);

/// Cache of those pipelines. Neither half of the key can change without a
/// recompile: the first is source text, the second a pair of override
/// constants.
///
/// [`buffer_to_image`] caches nothing else, but a pipeline is a few kilobytes
/// where the staging buffer is 33 MB, and the driver's compile is ~0.4 ms
/// against the 0.64 ms an 800x600 readback costs.
static PACK_PIPELINES: Lazy<Mutex<HashMap<PackKey, PackPipeline>>> = Lazy::new(Default::default);

fn pack_pipeline(
    device: &wgpu::Device,
    tone_mapper: ToneMapper,
    width: u32,
    height: u32,
) -> PackPipeline {
    PACK_PIPELINES
        .lock()
        .unwrap()
        .entry((tone_mapper.wgsl(), width, height))
        .or_insert_with_key(|(curve, width, height)| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Tone Map Pack"),
                source: wgpu::ShaderSource::Wgsl(
                    format!("{curve}\n{}", include_str!("tone_map_pack.wgsl")).into(),
                ),
            });
            let layout = bind_group_layout(
                device,
                &[storage_binding(true, 16), storage_binding(false, 4)],
            );
            let pipeline = compute_pipeline(
                device,
                &layout,
                &module,
                &[("width", *width as f64), ("height", *height as f64)],
            );
            PackPipeline { layout, pipeline }
        })
        .clone()
}

/// Converts a wgpu buffer of linear HDR radiance to an RgbImage.
///
/// This is the display transform: `tone_mapper` brings unbounded radiance into
/// `[0, 1]`, then the gamma encode makes it displayable. Everything upstream --
/// the accumulator, bloom, the denoiser -- works on the untouched linear
/// values, so the curve chosen here changes only what is shown, never what is
/// computed.
///
/// This is the one place the image leaves the GPU, so how it is read matters
/// more than the arithmetic does. The curve and the encode run in a compute
/// pass first and pack to RGBA8, so what is copied back is 4 bytes a pixel
/// rather than 16 -- 33 MB at 4K instead of 133 MB, and 26 ms to 8.4 ms. What
/// is left on the CPU is a byte shuffle.
///
/// The GPU curve is [`ToneMapper::wgsl`], the same source the denoiser's
/// resolve pass and a desktop viewport splice in;
/// `wgsl_matches_the_cpu_curve` holds it to [`ToneMapper::map`] to 1e-4 across
/// all four curves, and `buffer_to_image_matches_the_cpu_encode` covers the
/// encode and the pack around it.
///
/// Two things it deliberately does not do:
///
/// - It does not go through [`get_result_from_buffer`], which copies the whole
///   mapped range into a `Vec` first. That range is host-visible, uncached,
///   write-combined memory that reads serially at roughly 1 GB/s, so the copy
///   cost more than everything else here together. The pixels are read once, in
///   place.
/// - It does not use `put_pixel`, whose bounds check and `%`/`/` per pixel are
///   overhead when the traversal order is already row-major. The output rows
///   are walked in step with the input, in parallel, since the read is
///   latency-bound on that write-combined memory and threads hide it.
///
/// Neither staging nor packed buffer is cached: holding 33 MB of host-visible
/// memory alive to save a few ms once at the end of a render is the wrong
/// trade. The pipeline is, because a driver compile is not proportional to the
/// image.
///
/// `buffer` is bound as a read-only storage buffer, so it needs
/// `BufferUsages::STORAGE`. Every buffer the renderer hands out on
/// [`RenderProgress`](crate::renderer::RenderProgress) has it.
///
/// WGSL does not require `sqrt` or the curve's arithmetic to round exactly as
/// Rust's do, so a channel sitting on a code-value boundary may land one value
/// either side of what the CPU produced. On a Radeon RX 5700 XT under RADV it
/// is in fact byte-identical, checked over 1e-6 to 1e6 radiance and all four
/// curves, and the goldens did not shift at all.
pub fn buffer_to_image(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    width: u32,
    height: u32,
    tone_mapper: ToneMapper,
) -> image::RgbImage {
    let size = width as u64 * height as u64 * 4;

    let packed_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Packed Image Buffer"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let pack = pack_pipeline(device, tone_mapper, width, height);
    let group = bind_group(
        device,
        &pack.layout,
        &[
            wgpu::BindingResource::Buffer(buffer.as_entire_buffer_binding()),
            wgpu::BindingResource::Buffer(packed_buffer.as_entire_buffer_binding()),
        ],
    );

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    add_compute_pass_2d(
        &mut encoder,
        &pack.pipeline,
        &group,
        width.div_ceil(8),
        height.div_ceil(8),
        None,
        "tone_map_pack",
    );
    encoder.copy_buffer_to_buffer(&packed_buffer, 0, &staging_buffer, 0, size);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let mut img = image::RgbImage::new(width, height);
    {
        let data = buffer_slice.get_mapped_range().unwrap();
        let pixels: &[u32] = bytemuck::cast_slice(&data);

        img.as_mut()
            .par_chunks_mut(3)
            .zip(pixels.par_iter())
            .for_each(|(out, pixel)| {
                out[0] = *pixel as u8;
                out[1] = (*pixel >> 8) as u8;
                out[2] = (*pixel >> 16) as u8;
            });
    }
    staging_buffer.unmap();

    img
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::util::DeviceExt;

    /// The GPU display transform against the CPU one.
    ///
    /// `wgsl_matches_the_cpu_curve` pins the curve; what this adds is what
    /// wraps it -- the gamma, the 0.999 ceiling, the truncating cast and the
    /// byte order of the pack -- plus the dispatch's bounds check, which is why
    /// the image is deliberately not a multiple of the 8x8 workgroup.
    ///
    /// One code value of slack, for a driver whose `sqrt` or rational rounds
    /// differently. This machine's does not.
    #[test]
    fn buffer_to_image_matches_the_cpu_encode() {
        let (device, queue) = get_wgpu_device_and_queue();
        let (width, height) = (13u32, 7u32);

        // The awkward inputs first, then a ramp across the shoulder fine enough
        // that consecutive pixels differ by a code value or two.
        let odd = [0., -1., f32::NAN, f32::INFINITY, f32::MAX, 1e18];
        let radiance: Vec<[f32; 4]> = (0..(width * height) as usize)
            .map(|i| {
                let v = odd
                    .get(i)
                    .copied()
                    .unwrap_or_else(|| (i - odd.len()) as f32 / 20.);
                [v, v * 0.5, v * 0.25, 0.]
            })
            .collect();

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&radiance),
            usage: wgpu::BufferUsages::STORAGE,
        });

        for mapper in [
            ToneMapper::Aces,
            ToneMapper::PbrNeutral,
            ToneMapper::Reinhard { white_point: 4. },
            ToneMapper::Clamp,
        ] {
            let image = buffer_to_image(device, queue, &buffer, width, height, mapper);
            assert_eq!(image.dimensions(), (width, height));

            for (pixel, linear) in image.pixels().zip(radiance.iter()) {
                let want = mapper.map([linear[0], linear[1], linear[2]]);
                let encode = |v: f32| (v.sqrt().min(0.999) * 256.) as u8;
                let want = [encode(want[0]), encode(want[1]), encode(want[2])];

                for ch in 0..3 {
                    assert!(
                        pixel[ch].abs_diff(want[ch]) <= 1,
                        "{:?} at {:?}: gpu {:?} vs cpu {:?}",
                        mapper,
                        &linear[..3],
                        pixel.0,
                        want
                    );
                }
            }
        }
    }
}
