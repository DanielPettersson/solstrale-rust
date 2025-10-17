use rayon::iter::ParallelIterator;
use crate::geo::vec3::Vec3;
use crate::post::{PostProcessor, PostProcessors};
use std::error::Error;
use rayon::iter::IntoParallelRefIterator;
use wgpu::BufferUsages;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use crate::util::wgpu_util::{add_buffer_copy, add_compute_pass, bind_group, bind_group_layout, bind_group_layout_entry, compute_pipeline, get_result_from_buffer, get_wgpu_device_and_queue};

#[derive(Clone)]
/// Applies a saturation effect on the pixel colors
pub struct SaturationPostProcessor {
    saturation_factor: f64,
}

impl SaturationPostProcessor {
    #![allow(clippy::new_ret_no_self)]
    /// Create a new saturation post-processor
    /// # Arguments
    /// * `saturation_factor` Saturation of the image. From -1 (black and white) to 1 (fully saturated)
    pub fn new(saturation_factor: f64) -> Result<PostProcessors, simple_error::SimpleError> {
        if !(-1. ..=1.).contains(&saturation_factor) {
            return Err(simple_error::SimpleError::new(
                "saturation_factor must be between -1 and 1",
            ));
        }

        Ok(PostProcessors::from(SaturationPostProcessor {
            saturation_factor,
        }))
    }
}

impl PostProcessor for SaturationPostProcessor {
    #[allow(clippy::needless_range_loop)]
    fn intermediate_post_process(
        &self,
        pixel_colors: &[Vec3],
        _albedo_colors: &[Vec3],
        _normal_colors: &[Vec3],
        _width: u32,
        _height: u32,
        _num_samples: u32,
    ) -> Result<Vec<Vec3>, Box<dyn Error>> {

        let input_pixels: Vec<[f32; 4]> = pixel_colors.par_iter().map(|p| p.into()).collect();

        let (device, queue) = get_wgpu_device_and_queue();

        let module =
            device.create_shader_module(wgpu::include_wgsl!("saturation.wgsl"));

        let input_pixels_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&input_pixels),
            usage: BufferUsages::STORAGE,
        });

        let output_pixels_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: input_pixels_buffer.size(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: input_pixels_buffer.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group_layout = bind_group_layout(
            device,
            &[
                bind_group_layout_entry(true, 16),
                bind_group_layout_entry(false, 16),
            ],
        );

        let bind_group = bind_group(
            device,
            &bind_group_layout,
            &[&input_pixels_buffer, &output_pixels_buffer],
        );

        let pipeline = compute_pipeline(
            device,
            &bind_group_layout,
            &module,
            &[("saturation_factor", self.saturation_factor)],
        );

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let workgroup_count = pixel_colors.len().div_ceil(64) as u32;
        add_compute_pass(
            &mut encoder,
            &pipeline,
            &bind_group,
            workgroup_count,
        );
        add_buffer_copy(&mut encoder, &output_pixels_buffer, &download_buffer);

        let command_buffer = encoder.finish();
        queue.submit([command_buffer]);

        let result = get_result_from_buffer::<[f32; 4]>(device, &download_buffer);
        Ok(result.par_iter().map(|d| d.into()).collect())

    }

    fn needs_albedo_and_normal_colors(&self) -> bool {
        false
    }
}
