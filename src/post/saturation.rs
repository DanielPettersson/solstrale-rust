//! Post-processor for applying saturation

use crate::post::{PostProcessContext, PostProcessor};
use crate::util::wgpu_util::{bind_group, bind_group_layout, compute_pipeline, storage_binding};
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
    pub fn new(
        saturation_factor: f64,
        device: &wgpu::Device,
    ) -> Result<Self, simple_error::SimpleError> {
        if !(-1. ..=1.).contains(&saturation_factor) {
            return Err(simple_error::SimpleError::new(
                "saturation_factor must be between -1 and 1",
            ));
        }

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

    fn post_process(&self, ctx: &mut PostProcessContext) -> Result<(), Box<dyn Error>> {
        let bind_group = bind_group(
            ctx.device,
            &self.bind_group_layout,
            &[wgpu::BindingResource::Buffer(
                ctx.buffer.as_entire_buffer_binding(),
            )],
        );

        let workgroup_count = (self.width * self.height).div_ceil(64);
        crate::util::wgpu_util::add_compute_pass(
            ctx.encoder,
            &self.pipeline,
            &bind_group,
            workgroup_count,
        );

        Ok(())
    }
}
