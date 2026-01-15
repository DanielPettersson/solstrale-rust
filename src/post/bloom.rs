//! Post-processor for applying bloom effect

use crate::geo::vec3::Vec3;
use crate::post::PostProcessor;
use crate::util::gaussian::create_gaussian_blur_weights;
use crate::util::wgpu_util::{
    add_buffer_copy, add_compute_pass, bind_group, bind_group_layout,
    compute_pipeline, get_result_from_buffer, get_wgpu_device_and_queue,
    storage_binding,
};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::error::Error;
use wgpu::BufferUsages;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use std::sync::{Arc, Mutex};

#[derive(Clone)]
/// Applies a bloom effect on the pixel colors
pub struct BloomPostProcessor {
    width: u32,
    height: u32,

    kernel_size_fraction: f64,
    threshold: f64,
    max_intensity: f64,

    filter_bright_module: wgpu::ShaderModule,
    apply_module: wgpu::ShaderModule,

    filter_bright_bind_group_layout: wgpu::BindGroupLayout,
    apply_bind_group_layout: wgpu::BindGroupLayout,
    add_bind_group_layout: wgpu::BindGroupLayout,

    apply_pipeline_x: Option<wgpu::ComputePipeline>,
    apply_pipeline_y: Option<wgpu::ComputePipeline>,
    add_pipeline: wgpu::ComputePipeline,

    weights_buffer: Option<wgpu::Buffer>,
    input_pixels_buffer: Option<wgpu::Buffer>,
    intermediate_buffer1: Option<wgpu::Buffer>,
    intermediate_buffer2: Option<wgpu::Buffer>,
    output_pixels_buffer: Option<wgpu::Buffer>,
    download_buffer: Option<wgpu::Buffer>,

    filter_bright_bind_group: Option<wgpu::BindGroup>,
    apply_bind_group_x: Option<wgpu::BindGroup>,
    apply_bind_group_y: Option<wgpu::BindGroup>,
    add_bind_group: Option<wgpu::BindGroup>,

    filter_bright_pipeline_cache: Arc<Mutex<Option<(u32, wgpu::ComputePipeline)>>>,
}

impl BloomPostProcessor {
    /// Create a new bloom post-processor
    /// # Arguments
    /// * `kernel_size_fraction` Radius of the blur effect, as a fraction of the rendered image's width
    /// * `threshold` Color intensity threshold for applying bloom effect. If not specified, defaults to "white"
    /// * `max_intensity` Maximum color intensity of the bloom effect. If not specified, defaults to unlimited
    pub fn new(
        kernel_size_fraction: f64,
        threshold: Option<f64>,
        max_intensity: Option<f64>,
    ) -> Result<Self, simple_error::SimpleError> {
        if !(0. ..=0.5).contains(&kernel_size_fraction) {
            return Err(simple_error::SimpleError::new(
                "kernel_size_fraction must be between 0 and 0.5",
            ));
        }

        let threshold = threshold.unwrap_or(Vec3::new(1., 1., 1.).length());
        let max_intensity = max_intensity.unwrap_or(1000.);

        let (device, _) = get_wgpu_device_and_queue();

        let filter_bright_module =
            device.create_shader_module(wgpu::include_wgsl!("bloom_filter_bright.wgsl"));
        let apply_module = device.create_shader_module(wgpu::include_wgsl!("bloom_apply.wgsl"));
        let add_module = device.create_shader_module(wgpu::include_wgsl!("bloom_add.wgsl"));

        let filter_bright_bind_group_layout = bind_group_layout(
            device,
            &[
                storage_binding(true, 16),
                storage_binding(false, 16),
            ],
        );

        let apply_bind_group_layout = bind_group_layout(
            device,
            &[
                storage_binding(true, 4),
                storage_binding(true, 16),
                storage_binding(false, 16),
            ],
        );

        let add_bind_group_layout = bind_group_layout(
            device,
            &[
                storage_binding(true, 16),
                storage_binding(true, 16),
                storage_binding(false, 16),
            ],
        );

        let add_pipeline = compute_pipeline(device, &add_bind_group_layout, &add_module, &[]);

        Ok(BloomPostProcessor {
            width: 0,
            height: 0,
            kernel_size_fraction,
            threshold,
            max_intensity,
            filter_bright_module,
            apply_module,
            filter_bright_bind_group_layout,
            apply_bind_group_layout,
            add_bind_group_layout,
            apply_pipeline_x: None,
            apply_pipeline_y: None,
            add_pipeline,
            weights_buffer: None,
            input_pixels_buffer: None,
            intermediate_buffer1: None,
            intermediate_buffer2: None,
            output_pixels_buffer: None,
            download_buffer: None,
            filter_bright_bind_group: None,
            apply_bind_group_x: Option::None,
            apply_bind_group_y: Option::None,
            add_bind_group: None,
            filter_bright_pipeline_cache: Arc::new(Mutex::new(None)),
        })
    }
}

impl PostProcessor for BloomPostProcessor {
    fn initialize(&mut self, width: u32, height: u32) {
        if self.width == width && self.height == height && self.weights_buffer.is_some() {
            return;
        }

        self.width = width;
        self.height = height;

        let (device, _) = get_wgpu_device_and_queue();

        self.apply_pipeline_x = Some(compute_pipeline(
            device,
            &self.apply_bind_group_layout,
            &self.apply_module,
            &[("width", width as f64), ("x_dir", 1.), ("y_dir", 0.)],
        ));
        self.apply_pipeline_y = Some(compute_pipeline(
            device,
            &self.apply_bind_group_layout,
            &self.apply_module,
            &[("width", width as f64), ("x_dir", 0.), ("y_dir", 1.)],
        ));

        let kernel_size = (self.kernel_size_fraction * width as f64) as usize * 2 + 1;
        let weights = create_gaussian_blur_weights(kernel_size, kernel_size as f32 / 5.);
        let size = (width * height * 16) as u64;

        let weights_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&weights),
            usage: BufferUsages::STORAGE,
        });

        let input_pixels_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let intermediate_buffer1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let intermediate_buffer2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE,
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

        let filter_bright_bind_group = bind_group(
            device,
            &self.filter_bright_bind_group_layout,
            &[&input_pixels_buffer, &intermediate_buffer1],
        );

        let apply_bind_group_x = bind_group(
            device,
            &self.apply_bind_group_layout,
            &[
                &weights_buffer,
                &intermediate_buffer1,
                &intermediate_buffer2,
            ],
        );

        let apply_bind_group_y = bind_group(
            device,
            &self.apply_bind_group_layout,
            &[
                &weights_buffer,
                &intermediate_buffer2,
                &intermediate_buffer1,
            ],
        );

        let add_bind_group = bind_group(
            device,
            &self.add_bind_group_layout,
            &[
                &input_pixels_buffer,
                &intermediate_buffer1,
                &output_pixels_buffer,
            ],
        );

        self.weights_buffer = Some(weights_buffer);
        self.input_pixels_buffer = Some(input_pixels_buffer);
        self.intermediate_buffer1 = Some(intermediate_buffer1);
        self.intermediate_buffer2 = Some(intermediate_buffer2);
        self.output_pixels_buffer = Some(output_pixels_buffer);
        self.download_buffer = Some(download_buffer);
        self.filter_bright_bind_group = Some(filter_bright_bind_group);
        self.apply_bind_group_x = Some(apply_bind_group_x);
        self.apply_bind_group_y = Some(apply_bind_group_y);
        self.add_bind_group = Some(add_bind_group);
        *self.filter_bright_pipeline_cache.lock().unwrap() = None;
    }

    #[allow(clippy::needless_range_loop)]
    fn intermediate_post_process(
        &self,
        pixel_colors: &[Vec3],
        _albedo_colors: &[Vec3],
        _normal_colors: &[Vec3],
        num_samples: u32,
    ) -> Result<Vec<Vec3>, Box<dyn Error>> {
        let input_pixels: Vec<[f32; 4]> = pixel_colors.par_iter().map(|p| p.into()).collect();

        let (device, queue) = get_wgpu_device_and_queue();

        let input_pixels_buffer = self.input_pixels_buffer.as_ref().ok_or("Not initialized")?;
        let output_pixels_buffer = self
            .output_pixels_buffer
            .as_ref()
            .ok_or("Not initialized")?;
        let download_buffer = self.download_buffer.as_ref().ok_or("Not initialized")?;
        let filter_bright_bind_group = self
            .filter_bright_bind_group
            .as_ref()
            .ok_or("Not initialized")?;
        let apply_bind_group_x = self.apply_bind_group_x.as_ref().ok_or("Not initialized")?;
        let apply_bind_group_y = self.apply_bind_group_y.as_ref().ok_or("Not initialized")?;
        let add_bind_group = self.add_bind_group.as_ref().ok_or("Not initialized")?;

        queue.write_buffer(input_pixels_buffer, 0, bytemuck::cast_slice(&input_pixels));

        let mut cache = self.filter_bright_pipeline_cache.lock().unwrap();
        if cache
            .as_ref()
            .map(|(n, _)| *n != num_samples)
            .unwrap_or(true)
        {
            let threshold = self.threshold * num_samples as f64;
            let max_intensity = self.max_intensity * num_samples as f64;
            let pipeline = compute_pipeline(
                device,
                &self.filter_bright_bind_group_layout,
                &self.filter_bright_module,
                &[("threshold", threshold), ("max_intensity", max_intensity)],
            );
            *cache = Some((num_samples, pipeline));
        }
        let filter_bright_pipeline = &cache.as_ref().unwrap().1;

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let workgroup_count = pixel_colors.len().div_ceil(64) as u32;
        add_compute_pass(
            &mut encoder,
            filter_bright_pipeline,
            filter_bright_bind_group,
            workgroup_count,
        );
        add_compute_pass(
            &mut encoder,
            self.apply_pipeline_x.as_ref().unwrap(),
            apply_bind_group_x,
            workgroup_count,
        );
        add_compute_pass(
            &mut encoder,
            self.apply_pipeline_y.as_ref().unwrap(),
            apply_bind_group_y,
            workgroup_count,
        );
        add_compute_pass(
            &mut encoder,
            &self.add_pipeline,
            add_bind_group,
            workgroup_count,
        );
        add_buffer_copy(&mut encoder, output_pixels_buffer, download_buffer);

        let command_buffer = encoder.finish();
        queue.submit([command_buffer]);

        let result = get_result_from_buffer::<[f32; 4]>(device, download_buffer);

        Ok(result.par_iter().map(|d| d.into()).collect())
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }
}