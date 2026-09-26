//! Post-processor for applying bloom effect

use crate::geo::vec3::Vec3;
use crate::post::{PostProcessContext, PostProcessor};
use crate::util::gaussian::create_gaussian_blur_weights;
use crate::util::wgpu_util::{
    add_compute_pass_2d, bind_group, bind_group_layout, compute_pipeline, storage_binding,
};
use std::error::Error;
use wgpu::BufferUsages;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

#[derive(Clone)]
/// Applies a bloom effect on the pixel colors
///
/// Not a preview processor (see [`PostProcessor::preview`]): a full-resolution
/// separable Gaussian is the most expensive pass in the crate, and the cost of
/// leaving it out of the preview is that the glow appears on the last batch --
/// a pop, on an effect that is a look rather than information.
pub struct BloomPostProcessor {
    width: u32,
    height: u32,

    kernel_size_fraction: f64,
    threshold: f64,
    max_intensity: f64,

    filter_bright_module: wgpu::ShaderModule,
    apply_module: wgpu::ShaderModule,
    add_module: wgpu::ShaderModule,

    filter_bright_bind_group_layout: wgpu::BindGroupLayout,
    apply_bind_group_layout: wgpu::BindGroupLayout,
    add_bind_group_layout: wgpu::BindGroupLayout,

    // All four pipelines carry the image dimensions as override constants, so
    // none of them can be built before `initialize`.
    apply_pipeline_x: Option<wgpu::ComputePipeline>,
    apply_pipeline_y: Option<wgpu::ComputePipeline>,
    add_pipeline: Option<wgpu::ComputePipeline>,
    filter_bright_pipeline: Option<wgpu::ComputePipeline>,

    weights_buffer: Option<wgpu::Buffer>,
    intermediate_buffer1: Option<wgpu::Buffer>,
    intermediate_buffer2: Option<wgpu::Buffer>,

    apply_bind_group_x: Option<wgpu::BindGroup>,
    apply_bind_group_y: Option<wgpu::BindGroup>,
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
        device: &wgpu::Device,
    ) -> Result<Self, simple_error::SimpleError> {
        if !(0. ..=0.5).contains(&kernel_size_fraction) {
            return Err(simple_error::SimpleError::new(
                "kernel_size_fraction must be between 0 and 0.5",
            ));
        }

        let threshold = threshold.unwrap_or(Vec3::new(1., 1., 1.).length());
        let max_intensity = max_intensity.unwrap_or(1000.);

        let filter_bright_module =
            device.create_shader_module(wgpu::include_wgsl!("bloom_filter_bright.wgsl"));
        let apply_module = device.create_shader_module(wgpu::include_wgsl!("bloom_apply.wgsl"));
        let add_module = device.create_shader_module(wgpu::include_wgsl!("bloom_add.wgsl"));

        let filter_bright_bind_group_layout = bind_group_layout(
            device,
            &[storage_binding(true, 16), storage_binding(false, 16)],
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
            &[storage_binding(false, 16), storage_binding(true, 16)],
        );

        Ok(BloomPostProcessor {
            width: 0,
            height: 0,
            kernel_size_fraction,
            threshold,
            max_intensity,
            filter_bright_module,
            apply_module,
            add_module,
            filter_bright_bind_group_layout,
            apply_bind_group_layout,
            add_bind_group_layout,
            apply_pipeline_x: None,
            apply_pipeline_y: None,
            add_pipeline: None,
            filter_bright_pipeline: None,
            weights_buffer: None,
            intermediate_buffer1: None,
            intermediate_buffer2: None,
            apply_bind_group_x: None,
            apply_bind_group_y: None,
        })
    }
}

impl PostProcessor for BloomPostProcessor {
    fn initialize(&mut self, device: &wgpu::Device, _queue: &wgpu::Queue, width: u32, height: u32) {
        if self.width == width && self.height == height && self.weights_buffer.is_some() {
            return;
        }

        self.width = width;
        self.height = height;

        // Every pass dispatches over a 2-D grid, so all four need the row
        // stride and the bounds as constants. A 1-D dispatch would cross
        // `max_compute_workgroups_per_dimension` (65535 on a Radeon RX 5700 XT)
        // at 4194240 pixels, so 4K would fail validation outright.
        let dimensions = [("width", width as f64), ("height", height as f64)];

        self.filter_bright_pipeline = Some(compute_pipeline(
            device,
            &self.filter_bright_bind_group_layout,
            &self.filter_bright_module,
            &[
                ("width", width as f64),
                ("height", height as f64),
                ("threshold", self.threshold),
                ("max_intensity", self.max_intensity),
            ],
        ));

        self.add_pipeline = Some(compute_pipeline(
            device,
            &self.add_bind_group_layout,
            &self.add_module,
            &dimensions,
        ));

        self.apply_pipeline_x = Some(compute_pipeline(
            device,
            &self.apply_bind_group_layout,
            &self.apply_module,
            &[
                ("width", width as f64),
                ("height", height as f64),
                ("x_dir", 1.),
                ("y_dir", 0.),
            ],
        ));
        self.apply_pipeline_y = Some(compute_pipeline(
            device,
            &self.apply_bind_group_layout,
            &self.apply_module,
            &[
                ("width", width as f64),
                ("height", height as f64),
                ("x_dir", 0.),
                ("y_dir", 1.),
            ],
        ));

        let kernel_size = (self.kernel_size_fraction * width as f64) as usize * 2 + 1;
        let weights = create_gaussian_blur_weights(kernel_size, kernel_size as f32 / 5.);
        let size = (width * height * 16) as u64;

        let weights_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&weights),
            usage: BufferUsages::STORAGE,
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

        let apply_bind_group_x = bind_group(
            device,
            &self.apply_bind_group_layout,
            &[
                wgpu::BindingResource::Buffer(weights_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(intermediate_buffer1.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(intermediate_buffer2.as_entire_buffer_binding()),
            ],
        );

        let apply_bind_group_y = bind_group(
            device,
            &self.apply_bind_group_layout,
            &[
                wgpu::BindingResource::Buffer(weights_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(intermediate_buffer2.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(intermediate_buffer1.as_entire_buffer_binding()),
            ],
        );

        self.weights_buffer = Some(weights_buffer);
        self.intermediate_buffer1 = Some(intermediate_buffer1);
        self.intermediate_buffer2 = Some(intermediate_buffer2);
        self.apply_bind_group_x = Some(apply_bind_group_x);
        self.apply_bind_group_y = Some(apply_bind_group_y);
    }

    fn post_process(&self, ctx: &mut PostProcessContext) -> Result<(), Box<dyn Error>> {
        let device = ctx.device;
        let buffer = ctx.buffer;

        let intermediate_buffer1 = self
            .intermediate_buffer1
            .as_ref()
            .ok_or("Not initialized")?;
        let apply_bind_group_x = self.apply_bind_group_x.as_ref().ok_or("Not initialized")?;
        let apply_bind_group_y = self.apply_bind_group_y.as_ref().ok_or("Not initialized")?;
        let filter_bright_pipeline = self
            .filter_bright_pipeline
            .as_ref()
            .ok_or("Not initialized")?;
        let apply_pipeline_x = self.apply_pipeline_x.as_ref().ok_or("Not initialized")?;
        let apply_pipeline_y = self.apply_pipeline_y.as_ref().ok_or("Not initialized")?;
        let add_pipeline = self.add_pipeline.as_ref().ok_or("Not initialized")?;

        let filter_bright_bind_group = bind_group(
            device,
            &self.filter_bright_bind_group_layout,
            &[
                wgpu::BindingResource::Buffer(buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(intermediate_buffer1.as_entire_buffer_binding()),
            ],
        );

        let add_bind_group = bind_group(
            device,
            &self.add_bind_group_layout,
            &[
                wgpu::BindingResource::Buffer(buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(intermediate_buffer1.as_entire_buffer_binding()),
            ],
        );

        let groups_x = self.width.div_ceil(8);
        let groups_y = self.height.div_ceil(8);

        add_compute_pass_2d(
            ctx.encoder,
            filter_bright_pipeline,
            &filter_bright_bind_group,
            groups_x,
            groups_y,
            ctx.timer.as_deref_mut(),
            "bloom_filter_bright",
        );
        // The two blur directions are timed apart: they run the same shader
        // over the same pixels, so a difference between them is a memory access
        // pattern rather than arithmetic.
        add_compute_pass_2d(
            ctx.encoder,
            apply_pipeline_x,
            apply_bind_group_x,
            groups_x,
            groups_y,
            ctx.timer.as_deref_mut(),
            "bloom_blur_x",
        );
        add_compute_pass_2d(
            ctx.encoder,
            apply_pipeline_y,
            apply_bind_group_y,
            groups_x,
            groups_y,
            ctx.timer.as_deref_mut(),
            "bloom_blur_y",
        );
        add_compute_pass_2d(
            ctx.encoder,
            add_pipeline,
            &add_bind_group,
            groups_x,
            groups_y,
            ctx.timer.as_deref_mut(),
            "bloom_add",
        );

        Ok(())
    }
}
