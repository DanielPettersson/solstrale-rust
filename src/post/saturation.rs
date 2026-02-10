//! Post-processor for applying saturation

use crate::post::PostProcessor;
use crate::util::wgpu_util::{
    bind_group, bind_group_layout, compute_pipeline, get_wgpu_device_and_queue, storage_binding,
};
use std::error::Error;

#[derive(Clone)]
/// Applies a saturation effect on the pixel colors
pub struct SaturationPostProcessor {
    width: u32,
    height: u32,

    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
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

        let bind_group_layout = bind_group_layout(device, &[storage_binding(false, 16)]);

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
        })
    }
}

impl PostProcessor for SaturationPostProcessor {
    fn initialize(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        width: u32,
        height: u32,
    ) {
        self.width = width;
        self.height = height;
    }

    fn post_process(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        buffer: &wgpu::Buffer,
        _num_samples: u32,
    ) -> Result<(), Box<dyn Error>> {
        let (device, _) = get_wgpu_device_and_queue();

        let bind_group = bind_group(
            device,
            &self.bind_group_layout,
            &[wgpu::BindingResource::Buffer(
                buffer.as_entire_buffer_binding(),
            )],
        );

        let workgroup_count = (self.width * self.height).div_ceil(64);
        crate::util::wgpu_util::add_compute_pass(
            encoder,
            &self.pipeline,
            &bind_group,
            workgroup_count,
        );

        Ok(())
    }
}
