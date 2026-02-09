//! Post-processor for applying saturation

use crate::post::PostProcessor;
use crate::util::wgpu_util::{
    bind_group, bind_group_layout, compute_pipeline, get_wgpu_device_and_queue, storage_binding,
};
use std::error::Error;
use wgpu::BufferUsages;

#[derive(Clone)]
/// Applies a saturation effect on the pixel colors
pub struct SaturationPostProcessor {
    width: u32,
    height: u32,

    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,

    input_pixels_buffer: Option<wgpu::Buffer>,
    output_pixels_buffer: Option<wgpu::Buffer>,
    download_buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
}

impl SaturationPostProcessor {
    /// Creates new saturation post-processor
    /// # Arguments
    /// * `saturation_factor` Saturation of the image. From -1 (black and white) to 1 (fully saturated)
    pub fn new(saturation_factor: f64) -> Result<Self, simple_error::SimpleError> {
        if !(-1. ..=1.).contains(&saturation_factor) {
            return Err(simple_error::SimpleError::new(
                "saturation_factor must be between -1 and 1",
            ));
        }

        let (device, _) = get_wgpu_device_and_queue();

        let module = device.create_shader_module(wgpu::include_wgsl!("saturation.wgsl"));

        let bind_group_layout = bind_group_layout(
            device,
            &[storage_binding(true, 16), storage_binding(false, 16)],
        );

        let pipeline = compute_pipeline(
            device,
            &bind_group_layout,
            &module,
            &[("saturation_factor", saturation_factor)],
        );

        Ok(SaturationPostProcessor {
            width: 0,
            height: 0,
            bind_group_layout,
            pipeline,
            input_pixels_buffer: None,
            output_pixels_buffer: None,
            download_buffer: None,
            bind_group: None,
        })
    }
}

impl PostProcessor for SaturationPostProcessor {
    fn initialize(&mut self, device: &wgpu::Device, _queue: &wgpu::Queue, width: u32, height: u32) {
        if self.width == width && self.height == height && self.input_pixels_buffer.is_some() {
            return;
        }

        self.width = width;
        self.height = height;

        let size = (width * height * 16) as u64;

        let input_pixels_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_pixels_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = bind_group(
            device,
            &self.bind_group_layout,
            &[
                wgpu::BindingResource::Buffer(input_pixels_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(output_pixels_buffer.as_entire_buffer_binding()),
            ],
        );

        self.input_pixels_buffer = Some(input_pixels_buffer);
        self.output_pixels_buffer = Some(output_pixels_buffer);
        self.download_buffer = Some(download_buffer);
        self.bind_group = Some(bind_group);
    }

    #[allow(clippy::needless_range_loop)]
    fn post_process(
        &self,
        _encoder: &mut wgpu::CommandEncoder,
        _buffer: &wgpu::Buffer,
        _num_samples: u32,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}
