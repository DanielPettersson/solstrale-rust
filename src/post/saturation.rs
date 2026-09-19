//! Post-processor for applying saturation

use crate::post::{PostProcessContext, PostProcessor};
use crate::util::wgpu_util::{
    add_compute_pass_2d, bind_group, bind_group_layout, compute_pipeline, storage_binding,
};
use std::error::Error;

#[derive(Clone)]
/// Applies a saturation effect on the pixel colors
pub struct SaturationPostProcessor {
    width: u32,
    height: u32,
    saturation_factor: f64,
    module: wgpu::ShaderModule,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Built in `initialize` rather than in `new`, because the image dimensions
    /// are override constants of the shader. See there for why they have to be.
    pipeline: Option<wgpu::ComputePipeline>,
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

        Ok(SaturationPostProcessor {
            width: 0,
            height: 0,
            saturation_factor,
            module,
            bind_group_layout,
            pipeline: None,
        })
    }
}

impl PostProcessor for SaturationPostProcessor {
    fn initialize(&mut self, device: &wgpu::Device, _queue: &wgpu::Queue, width: u32, height: u32) {
        if self.width == width && self.height == height && self.pipeline.is_some() {
            return;
        }

        self.width = width;
        self.height = height;

        // The shader dispatches over a 2-D grid and so needs the row stride and
        // the bounds as constants; it can no longer recover either from
        // `arrayLength`, which only ever gave it a pixel count.
        self.pipeline = Some(compute_pipeline(
            device,
            &self.bind_group_layout,
            &self.module,
            &[
                ("width", width as f64),
                ("height", height as f64),
                ("saturation_factor", self.saturation_factor),
            ],
        ));
    }

    fn post_process(&self, ctx: &mut PostProcessContext) -> Result<(), Box<dyn Error>> {
        let pipeline = self.pipeline.as_ref().ok_or("Not initialized")?;

        let bind_group = bind_group(
            ctx.device,
            &self.bind_group_layout,
            &[wgpu::BindingResource::Buffer(
                ctx.buffer.as_entire_buffer_binding(),
            )],
        );

        add_compute_pass_2d(
            ctx.encoder,
            pipeline,
            &bind_group,
            self.width.div_ceil(8),
            self.height.div_ceil(8),
            ctx.timer.as_deref_mut(),
            "saturation",
        );

        Ok(())
    }
}
