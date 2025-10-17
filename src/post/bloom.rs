use crate::geo::vec3::Vec3;
use crate::post::{pixel_colors_to_rgb_image, PostProcessor, PostProcessors};
use crate::util::gaussian::create_gaussian_blur_weights;
use crate::util::wgpu_util::{
    add_buffer_copy, add_compute_pass, bind_group, bind_group_layout, bind_group_layout_entry,
    compute_pipeline, get_result_from_buffer, get_wgpu_device_and_queue,
};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::error::Error;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::BufferUsages;

#[derive(Clone)]
/// Applies a bloom effect on the pixel colors
pub struct BloomPostProcessor {
    kernel_size_fraction: f64,
    threshold: f64,
    max_intensity: f64,
}

impl BloomPostProcessor {
    #![allow(clippy::new_ret_no_self)]
    /// Create a new bloom post processor
    /// # Arguments
    /// * `kernel_size_fraction` Radius of the blur effect, as a fraction of the rendered image's width
    /// * `threshold` Color intensity threshold for applying bloom effect. If not specified, defaults to "white"
    /// * `max_intensity` Maximum color intensity of the bloom effect. If not specified, defaults to unlimited
    pub fn new(
        kernel_size_fraction: f64,
        threshold: Option<f64>,
        max_intensity: Option<f64>,
    ) -> Result<PostProcessors, simple_error::SimpleError> {
        if !(0. ..=0.5).contains(&kernel_size_fraction) {
            return Err(simple_error::SimpleError::new(
                "kernel_size_fraction must be between 0 and 0.5",
            ));
        }

        let threshold = threshold.unwrap_or(Vec3::new(1., 1., 1.).length());
        let max_intensity = max_intensity.unwrap_or(1000.);

        Ok(PostProcessors::from(BloomPostProcessor {
            kernel_size_fraction,
            threshold,
            max_intensity,
        }))
    }
}

impl PostProcessor for BloomPostProcessor {
    fn post_process(
        &self,
        pixel_colors: &[Vec3],
        albedo_colors: &[Vec3],
        normal_colors: &[Vec3],
        width: u32,
        height: u32,
        num_samples: u32,
    ) -> Result<image::RgbImage, Box<dyn Error>> {
        let pixel_colors = self.intermediate_post_process(
            pixel_colors,
            albedo_colors,
            normal_colors,
            width,
            height,
            num_samples,
        )?;
        Ok(pixel_colors_to_rgb_image(
            &pixel_colors,
            width,
            height,
            num_samples,
        ))
    }

    #[allow(clippy::needless_range_loop)]
    fn intermediate_post_process(
        &self,
        pixel_colors: &[Vec3],
        _albedo_colors: &[Vec3],
        _normal_colors: &[Vec3],
        width: u32,
        _height: u32,
        num_samples: u32,
    ) -> Result<Vec<Vec3>, Box<dyn Error>> {
        let threshold = self.threshold * num_samples as f64;
        let max_intensity = self.max_intensity * num_samples as f64;
        let kernel_size = (self.kernel_size_fraction * width as f64) as usize * 2 + 1;
        let weights = create_gaussian_blur_weights(kernel_size, kernel_size as f32 / 5.);

        let input_pixels: Vec<[f32; 4]> = pixel_colors.par_iter().map(|p| p.into()).collect();

        let (device, queue) = get_wgpu_device_and_queue()?;

        let filter_bright_module =
            device.create_shader_module(wgpu::include_wgsl!("bloom_filter_bright.wgsl"));
        let apply_module = device.create_shader_module(wgpu::include_wgsl!("bloom_apply.wgsl"));
        let add_module = device.create_shader_module(wgpu::include_wgsl!("bloom_add.wgsl"));

        let weights_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&weights),
            usage: BufferUsages::STORAGE,
        });

        let input_pixels_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&input_pixels),
            usage: BufferUsages::STORAGE,
        });

        let intermediate_buffer1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: input_pixels_buffer.size(),
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let intermediate_buffer2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: input_pixels_buffer.size(),
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
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

        let filter_bright_bind_group_layout = bind_group_layout(
            &device,
            &[
                bind_group_layout_entry(true, 16),
                bind_group_layout_entry(false, 16),
            ],
        );

        let apply_bind_group_layout = bind_group_layout(
            &device,
            &[
                bind_group_layout_entry(true, 4),
                bind_group_layout_entry(true, 16),
                bind_group_layout_entry(false, 16),
            ],
        );

        let add_bind_group_layout = bind_group_layout(
            &device,
            &[
                bind_group_layout_entry(true, 16),
                bind_group_layout_entry(true, 16),
                bind_group_layout_entry(false, 16),
            ],
        );

        let filter_bright_bind_group = bind_group(
            &device,
            &filter_bright_bind_group_layout,
            &[&input_pixels_buffer, &intermediate_buffer1],
        );

        let apply_bind_group_x = bind_group(
            &device,
            &apply_bind_group_layout,
            &[
                &weights_buffer,
                &intermediate_buffer1,
                &intermediate_buffer2,
            ],
        );

        let apply_bind_group_y = bind_group(
            &device,
            &apply_bind_group_layout,
            &[
                &weights_buffer,
                &intermediate_buffer2,
                &intermediate_buffer1,
            ],
        );

        let add_bind_group = bind_group(
            &device,
            &add_bind_group_layout,
            &[
                &input_pixels_buffer,
                &intermediate_buffer1,
                &output_pixels_buffer,
            ],
        );

        let filter_bright_pipeline = compute_pipeline(
            &device,
            &filter_bright_bind_group_layout,
            &filter_bright_module,
            &[("threshold", threshold), ("max_intensity", max_intensity)],
        );
        let apply_pipeline_x = compute_pipeline(
            &device,
            &apply_bind_group_layout,
            &apply_module,
            &[("width", width as f64), ("x_dir", 1.), ("y_dir", 0.)],
        );
        let apply_pipeline_y = compute_pipeline(
            &device,
            &apply_bind_group_layout,
            &apply_module,
            &[("width", width as f64), ("x_dir", 0.), ("y_dir", 1.)],
        );
        let add_pipeline = compute_pipeline(&device, &add_bind_group_layout, &add_module, &[]);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let workgroup_count = pixel_colors.len().div_ceil(64) as u32;
        add_compute_pass(
            &mut encoder,
            &filter_bright_pipeline,
            &filter_bright_bind_group,
            workgroup_count,
        );
        add_compute_pass(
            &mut encoder,
            &apply_pipeline_x,
            &apply_bind_group_x,
            workgroup_count,
        );
        add_compute_pass(
            &mut encoder,
            &apply_pipeline_y,
            &apply_bind_group_y,
            workgroup_count,
        );
        add_compute_pass(
            &mut encoder,
            &add_pipeline,
            &add_bind_group,
            workgroup_count,
        );
        add_buffer_copy(&mut encoder, &output_pixels_buffer, &download_buffer);

        let command_buffer = encoder.finish();
        queue.submit([command_buffer]);

        let result = get_result_from_buffer::<[f32; 4]>(&device, &download_buffer);
        Ok(result.par_iter().map(|d| d.into()).collect())
    }

    fn needs_albedo_and_normal_colors(&self) -> bool {
        false
    }
}
