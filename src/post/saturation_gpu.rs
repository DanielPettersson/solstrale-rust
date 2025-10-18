//! Post-processor for applying saturation
#![cfg(feature = "gpu")]

use crate::geo::vec3::Vec3;
use crate::post::{PostProcessor, PostProcessors};
use crate::util::wgpu_util::{
    add_buffer_copy, add_compute_pass, bind_group, bind_group_layout, bind_group_layout_entry,
    compute_pipeline, get_result_from_buffer, get_wgpu_device_and_queue,
};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::error::Error;
use std::time::Instant;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::BufferUsages;

#[derive(Clone)]
/// Applies a saturation effect on the pixel colors
pub struct SaturationPostProcessor {
    width: u32,
    height: u32,

    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
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

        let (device, _) = get_wgpu_device_and_queue();

        let module = device.create_shader_module(wgpu::include_wgsl!("saturation.wgsl"));

        let bind_group_layout = bind_group_layout(
            device,
            &[
                bind_group_layout_entry(true, 16),
                bind_group_layout_entry(false, 16),
            ],
        );

        let pipeline = compute_pipeline(
            device,
            &bind_group_layout,
            &module,
            &[("saturation_factor", saturation_factor)],
        );

        Ok(PostProcessors::from(SaturationPostProcessor {
            width: 0,
            height: 0,
            bind_group_layout,
            pipeline,
        }))
    }
}

impl PostProcessor for SaturationPostProcessor {
    fn initialize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    #[allow(clippy::needless_range_loop)]
    fn intermediate_post_process(
        &self,
        pixel_colors: &[Vec3],
        _albedo_colors: &[Vec3],
        _normal_colors: &[Vec3],
        _num_samples: u32,
    ) -> Result<Vec<Vec3>, Box<dyn Error>> {
        let now = Instant::now();

        let input_pixels: Vec<[f32; 4]> = pixel_colors.par_iter().map(|p| p.into()).collect();

        let (device, queue) = get_wgpu_device_and_queue();

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

        let bind_group = bind_group(
            device,
            &self.bind_group_layout,
            &[&input_pixels_buffer, &output_pixels_buffer],
        );

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let workgroup_count = pixel_colors.len().div_ceil(64) as u32;
        add_compute_pass(&mut encoder, &self.pipeline, &bind_group, workgroup_count);
        add_buffer_copy(&mut encoder, &output_pixels_buffer, &download_buffer);

        let command_buffer = encoder.finish();
        queue.submit([command_buffer]);

        let result = get_result_from_buffer::<[f32; 4]>(device, &download_buffer);

        println!("Saturation done after {}ms", now.elapsed().as_millis());

        Ok(result.par_iter().map(|d| d.into()).collect())
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }
}