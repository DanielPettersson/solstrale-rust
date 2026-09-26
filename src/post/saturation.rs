//! Post-processor for applying saturation

use crate::post::{PostProcessContext, PostProcessor};
use crate::util::wgpu_util::{
    add_compute_pass_2d, bind_group, bind_group_layout, compute_pipeline,
    shader_module_with_luminance, storage_binding,
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

        let module = shader_module_with_luminance(
            device,
            "saturation.wgsl",
            include_str!("saturation.wgsl"),
        );

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

        // The shader dispatches over a 2-D grid, so it needs the row stride and
        // the bounds as constants: `arrayLength` only gives it a pixel count.
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

    /// Yes: one dispatch of one pass, and a grade the preview would otherwise
    /// be shown without and then have applied under it on the last batch.
    fn preview(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::vec3::Vec3;
    use crate::post::PostProcessContext;
    use crate::util::luminance::luminance;
    use crate::util::wgpu_util::{get_result_from_buffer, get_wgpu_device_and_queue};
    use wgpu::util::DeviceExt;

    /// The pass pivots around Rec. 709 luminance. The golden tests run the
    /// whole filter at a 0.95 structural threshold, which is loose enough to
    /// miss a wrong set of weights; this checks the arithmetic itself against
    /// the definition the rest of the renderer uses.
    #[test]
    fn pivots_around_rec_709_luminance() {
        let (device, queue) = get_wgpu_device_and_queue();

        // Primaries, because that is where two sets of weights disagree most,
        // plus a value above the display range: the pass runs on linear HDR.
        let pixels: [[f32; 4]; 8] = [
            [0., 0., 0., 1.],
            [1., 1., 1., 1.],
            [1., 0., 0., 1.],
            [0., 1., 0., 1.],
            [0., 0., 1., 1.],
            [0.2, 0.6, 0.9, 1.],
            [0.8, 0.1, 0.35, 1.],
            [4., 0.5, 0.25, 1.],
        ];
        let width = pixels.len() as u32;
        let size = size_of_val(&pixels) as u64;

        for factor in [-1., -0.7, 0., 0.5, 1.] {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&pixels),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            // Bound by nothing: the fields the saturation pass never reads.
            let unused = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            let mut processor = SaturationPostProcessor::new(factor, device).unwrap();
            processor.initialize(device, queue, width, 1);

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            processor
                .post_process(&mut PostProcessContext {
                    encoder: &mut encoder,
                    buffer: &buffer,
                    accumulator: &unused,
                    sample_count_buffer: &unused,
                    gbuffer: &unused,
                    samples_completed: 1,
                    device,
                    timer: None,
                })
                .unwrap();

            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&buffer, 0, &staging, 0, size);
            queue.submit(Some(encoder.finish()));

            let got: Vec<[f32; 4]> = get_result_from_buffer(device, &staging);

            for (input, got) in pixels.iter().zip(got.iter()) {
                let gray =
                    luminance(Vec3::new(input[0] as f64, input[1] as f64, input[2] as f64)) as f32;
                for ch in 0..3 {
                    let want = -gray * factor as f32 + input[ch] * (1. + factor as f32);
                    assert!(
                        (got[ch] - want).abs() <= 1e-5 * want.abs().max(1.),
                        "factor {factor} channel {ch} of {:?}: got {:?} want {want}",
                        &input[..3],
                        got[ch]
                    );
                }
            }
        }
    }
}
