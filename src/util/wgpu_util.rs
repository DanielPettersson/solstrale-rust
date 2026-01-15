//! Utility functions for working with wgpu
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
}

pub(crate) struct BindingInfo {
    pub binding_type: BindingType,
}

static DEVICE_AND_QUEUE: Lazy<(wgpu::Device, wgpu::Queue)> =
    Lazy::new(|| create_wgpu_device_and_queue().expect("Failed to create device and queue"));

pub(crate) fn get_wgpu_device_and_queue() -> &'static (wgpu::Device, wgpu::Queue) {
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

    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| SimpleError::new(format!("Failed to create device: {}", e)).into())
}

pub(crate) fn get_result_from_buffer<T: AnyBitPattern>(
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

pub(crate) fn add_buffer_copy(
    encoder: &mut wgpu::CommandEncoder,
    source: &wgpu::Buffer,
    destination: &wgpu::Buffer,
) {
    encoder.copy_buffer_to_buffer(source, 0, destination, 0, destination.size());
}

pub(crate) fn compute_pipeline<'a>(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
    constants: &'a [(&'a str, f64)],
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout(device, bind_group_layout)),
        module,
        entry_point: Some("compute"),
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
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let entries = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
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
        immediate_size: 0,
    })
}
