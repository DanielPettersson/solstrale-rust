//! The renderer takes a [`Scene`] as input, renders it and reports [`RenderProgress`]

use crate::hittable::Hittable;
use crate::post::PostProcessor;
use std::error::Error;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use crate::camera::{Camera, CameraConfig};
use crate::geo::vec3::Vec3;
use crate::hittable::Hittables;
use crate::post::PostProcessors;
use crate::renderer::gpu_data::{GpuCamera, GpuRenderConfig};
use crate::renderer::scene_flattener::flatten_scene;
use crate::util::wgpu_util::{
    add_compute_pass_2d, bind_group, bind_group_layout, compute_pipeline, sampler_binding,
    storage_binding, texture_binding, uniform_binding,
};
use image::{DynamicImage, Rgb, RgbImage};
use simple_error::SimpleError;
use wgpu::BufferUsages;

pub mod gpu_data;
pub mod scene_flattener;

///Input to the ray tracer for how the image should be rendered
#[derive(Clone)]
pub struct RenderConfig {
    /// Width in pixels of the rendered image
    pub width: usize,
    /// Height in pixels of the rendered image
    pub height: usize,
    /// Number of times each pixel should be sampled
    pub samples_per_pixel: u32,
    /// Maximum number of ray bounces before a path is cut off.
    pub max_depth: u32,
    /// Samples traced per GPU dispatch, used as the starting point and the
    /// upper bound for the batch size.
    ///
    /// Larger batches amortise dispatch overhead and collapse the per-sample
    /// read-modify-write of the accumulation buffer, at the cost of coarser
    /// progress reporting and slower response to camera changes. The renderer
    /// tunes the actual size down from here to keep a single dispatch within
    /// [`TARGET_DISPATCH`].
    pub samples_per_batch: u32,
    /// Post processor to apply to the rendered image
    pub post_processors: Vec<PostProcessors>,
}

impl Default for RenderConfig {
    fn default() -> Self {
        RenderConfig {
            width: 300,
            height: 200,
            samples_per_pixel: 50,
            max_depth: 10,
            samples_per_batch: 4,
            post_processors: vec![],
        }
    }
}

/// Contains all information needed to render an image
pub struct Scene {
    /// World is the hittable objects in the scene
    pub world: Hittables,
    /// A camera for defining the view of the world
    pub camera: CameraConfig,
    /// Background color of the scene
    pub background_color: Vec3,
    /// Render configuration
    pub render_config: RenderConfig,
}

/// Progress reported back to the caller of the raytrace function
pub struct RenderProgress {
    /// progress is reported between 0 -> 1 and represents a percentage of completion
    pub progress: f64,
    /// Current speed of rendering in number of frames per second
    pub fps: Option<f64>,
    /// Estimated time left until rendering is complete
    pub estimated_time_left: Duration,
    /// Output buffer containing the image data
    pub output_buffer: wgpu::Buffer,
}

/// Wall clock budget for a single dispatch.
///
/// A caller that renders interactively hands us the same device and queue its
/// user interface draws on, so a dispatch that overruns a display frame is a
/// dropped frame for whoever is waiting behind us. Roughly one vsync interval.
const TARGET_DISPATCH: Duration = Duration::from_millis(12);

/// Ceiling on the adaptive batch size, so a very cheap scene does not end up
/// reporting progress once a second.
const MAX_BATCH: u32 = 64;

/// Ray depth used while the camera is being moved.
///
/// Mid-drag the thing that matters is seeing where the camera now points, and
/// losing most of the indirect light for a moment is far less distracting than
/// losing the frame rate. Ignored for a scene whose own depth is already this
/// shallow, so no restart is paid for nothing.
const INTERACTIVE_MAX_DEPTH: u32 = 3;

/// How long after the last camera update the full ray depth is restored.
const INTERACTION_TIMEOUT: Duration = Duration::from_millis(200);

/// Length of a single wait slice.
///
/// Waits are sliced rather than indefinite so that `abort` is still observed
/// while a dispatch is in flight, and so a wedged submission cannot hang the
/// render thread forever. Doubles as the idle poll interval.
const POLL_SLICE: Duration = Duration::from_millis(50);

/// Blocks until `index` has finished executing on the GPU.
///
/// Returns `Ok(false)` if an abort arrived while waiting.
///
/// Note this is a native-only guarantee: on WebGPU `PollType::Wait` is a no-op
/// and callbacks are driven by the window event loop instead.
fn wait_for_submission(
    device: &wgpu::Device,
    index: &wgpu::SubmissionIndex,
    abort: &Receiver<bool>,
) -> Result<bool, Box<dyn Error>> {
    loop {
        match device.poll(wgpu::PollType::Wait {
            submission_index: Some(index.clone()),
            timeout: Some(POLL_SLICE),
        }) {
            Ok(_) => return Ok(true),
            Err(wgpu::PollError::Timeout) => {
                if abort.try_recv().is_ok() {
                    return Ok(false);
                }
                // Still running. Recording the next dispatch here would defeat
                // the point of waiting, so go around again.
            }
            Err(e) => {
                return Err(SimpleError::new(format!("Failed to poll device: {}", e)).into());
            }
        }
    }
}

/// Renderer is a central part of the raytracer responsible for controlling the
/// process reporting back progress to the caller
pub struct Renderer<'a> {
    /// Only the sample count is retained from the scene. Holding the whole
    /// `Scene` kept the entire CPU scene graph and every decoded texture
    /// resident for the life of the render -- on an integrated GPU that is
    /// the GPU's memory too.
    samples_per_pixel: u32,
    samples_per_batch: u32,
    width: u32,
    height: u32,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    output_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    nodes_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    spheres_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    triangle_pos_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    triangle_attr_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    quad_pos_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    quad_attr_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    materials_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    camera_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    config_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    lights_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    prim_refs_buffer: wgpu::Buffer,
    post_processors: Vec<PostProcessors>,
    render_config: GpuRenderConfig,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
}

impl<'a> Renderer<'a> {
    /// Creates a new GPU renderer given a scene
    pub fn new(
        scene: Scene,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
    ) -> Result<Self, Box<dyn Error>> {
        if !scene.world.has_lights() {
            return Err(Box::new(SimpleError::new(
                "Scene should have at least one light",
            )));
        }

        let width = scene.render_config.width as u32;
        let height = scene.render_config.height as u32;

        let module = device.create_shader_module(wgpu::include_wgsl!("ray_trace.wgsl"));

        // Flatten scene
        let scene_data = flatten_scene(&scene);

        // Create buffers
        let nodes_buffer = create_and_upload_buffer(
            device,
            queue,
            "Nodes Buffer",
            &scene_data.nodes,
            BufferUsages::STORAGE,
        );
        let prim_refs_buffer = create_and_upload_buffer(
            device,
            queue,
            "Prim Refs Buffer",
            &scene_data.prim_refs,
            BufferUsages::STORAGE,
        );
        let spheres_buffer = create_and_upload_buffer(
            device,
            queue,
            "Spheres Buffer",
            &scene_data.spheres,
            BufferUsages::STORAGE,
        );
        let triangle_pos_buffer = create_and_upload_buffer(
            device,
            queue,
            "Triangle Positions Buffer",
            &scene_data.triangle_pos,
            BufferUsages::STORAGE,
        );
        let triangle_attr_buffer = create_and_upload_buffer(
            device,
            queue,
            "Triangle Attributes Buffer",
            &scene_data.triangle_attr,
            BufferUsages::STORAGE,
        );
        let quad_pos_buffer = create_and_upload_buffer(
            device,
            queue,
            "Quad Positions Buffer",
            &scene_data.quad_pos,
            BufferUsages::STORAGE,
        );
        let quad_attr_buffer = create_and_upload_buffer(
            device,
            queue,
            "Quad Attributes Buffer",
            &scene_data.quad_attr,
            BufferUsages::STORAGE,
        );
        let materials_buffer = create_and_upload_buffer(
            device,
            queue,
            "Materials Buffer",
            &scene_data.materials,
            BufferUsages::STORAGE,
        );
        let lights_buffer = create_and_upload_buffer(
            device,
            queue,
            "Lights Buffer",
            &scene_data.lights,
            BufferUsages::STORAGE,
        );

        // Blit the atlas using the layout `flatten_scene` already computed.
        // This used to re-run the identical packing here and rely on it being
        // deterministic.
        let mut atlas_image;

        if let Some(layout) = scene_data.atlas_layout.as_ref() {
            atlas_image = RgbImage::new(layout.width, layout.height);

            for placement in layout.placements.iter() {
                let texture = &scene_data.textures[placement.original_index];
                image::imageops::replace(
                    &mut atlas_image,
                    texture.as_ref(),
                    placement.x as i64,
                    placement.y as i64,
                );
            }
        } else {
            // Create a 1x1 white pixel if no textures, just to have valid binding
            atlas_image = RgbImage::from_pixel(1, 1, Rgb([255, 255, 255]));
        }

        let texture_extent = wgpu::Extent3d {
            width: atlas_image.width(),
            height: atlas_image.height(),
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture Atlas"),
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let atlas_rgba = DynamicImage::ImageRgb8(atlas_image).to_rgba8();

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * atlas_rgba.width()),
                rows_per_image: Some(atlas_rgba.height()),
            },
            texture_extent,
        );

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let camera_inst = Camera::new(width as usize, height as usize, &scene.camera);

        let gpu_camera = camera_to_gpu(&camera_inst);
        let camera_buffer = create_and_upload_buffer(
            device,
            queue,
            "Camera Buffer",
            &[gpu_camera],
            BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
        );

        let render_config = GpuRenderConfig {
            width,
            height,
            sample_count: 0,
            max_depth: scene.render_config.max_depth.max(1),
            background_color: [
                scene.background_color.x as f32,
                scene.background_color.y as f32,
                scene.background_color.z as f32,
            ],
            light_count: scene_data.lights.len() as u32,
            samples_per_batch: scene.render_config.samples_per_batch.max(1),
            _pad: [0; 3],
        };
        let config_buffer = create_and_upload_buffer(
            device,
            queue,
            "Config Buffer",
            &[render_config],
            BufferUsages::UNIFORM,
        );

        let bind_group_layout = bind_group_layout(
            device,
            &[
                storage_binding(false, 0), // 0: output buffer
                storage_binding(true, 0),  // 1: nodes
                storage_binding(true, 0),  // 2: spheres
                storage_binding(true, 0),  // 3: triangles
                storage_binding(true, 0),  // 4: quads
                storage_binding(true, 0),  // 5: materials
                uniform_binding(std::mem::size_of::<GpuCamera>() as u64), // 6: camera
                uniform_binding(std::mem::size_of::<GpuRenderConfig>() as u64), // 7: config
                texture_binding(wgpu::TextureViewDimension::D2), // 8: texture array
                sampler_binding(),         // 9: sampler
                storage_binding(true, 0),  // 10: lights
                storage_binding(true, 0),  // 11: primitive references
                storage_binding(true, 0),  // 12: triangle attributes
                storage_binding(true, 0),  // 13: quad attributes
            ],
        );

        let pipeline = compute_pipeline(device, &bind_group_layout, &module, &[]);

        let size = (width * height * 16) as u64; // vec3 is 16 bytes aligned (as vec4 effectively)

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = bind_group(
            device,
            &bind_group_layout,
            &[
                wgpu::BindingResource::Buffer(output_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(nodes_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(spheres_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(triangle_pos_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(quad_pos_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(materials_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(camera_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(config_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::TextureView(&texture_view),
                wgpu::BindingResource::Sampler(&sampler),
                wgpu::BindingResource::Buffer(lights_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(prim_refs_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(triangle_attr_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(quad_attr_buffer.as_entire_buffer_binding()),
            ],
        );

        let mut post_processors = scene.render_config.post_processors.clone();
        for p in &mut post_processors {
            p.initialize(device, queue, width, height);
        }

        Ok(Renderer {
            width,
            height,
            bind_group_layout,
            samples_per_pixel: scene.render_config.samples_per_pixel,
            samples_per_batch: scene.render_config.samples_per_batch.max(1),
            pipeline,
            output_buffer,
            bind_group,
            nodes_buffer,
            spheres_buffer,
            triangle_pos_buffer,
            triangle_attr_buffer,
            quad_pos_buffer,
            quad_attr_buffer,
            materials_buffer,
            camera_buffer,
            config_buffer,
            lights_buffer,
            prim_refs_buffer,
            post_processors,
            render_config,
            device,
            queue,
        })
    }

    /// Updates the camera buffer with a new camera configuration
    pub fn update_camera(&mut self, camera_config: &CameraConfig) {
        let camera_inst = Camera::new(self.width as usize, self.height as usize, camera_config);
        let gpu_camera = camera_to_gpu(&camera_inst);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[gpu_camera]));
    }

    /// Executes the rendering of the image on the GPU
    pub fn render(
        &mut self,
        output: &Sender<RenderProgress>,
        camera_config: &Receiver<CameraConfig>,
        abort: &Receiver<bool>,
        idle_after_rendering: bool,
    ) -> Result<(), Box<dyn Error>> {
        let samples_per_pixel = self.samples_per_pixel;
        let workgroup_count_x = self.width.div_ceil(8);
        let workgroup_count_y = self.height.div_ceil(8);

        // Number of samples already accumulated into the output buffer.
        let mut completed = 0;
        // Seeded from the configured batch size, then continuously re-tuned to
        // keep a single dispatch inside TARGET_DISPATCH.
        let mut batch_size = self.samples_per_batch.max(1);
        // Moving average of what one sample costs. Dominated by the scene
        // rather than by the view, so it deliberately survives camera changes
        // and only has to re-converge when the view changes character.
        let mut ms_per_sample: Option<f64> = None;
        // A camera config picked up while idling, handled at the top of the
        // next iteration together with any that arrived after it.
        let mut idle_camera_config = None;
        // Full ray depth, and the reduced one used while the camera moves.
        let full_max_depth = self.render_config.max_depth;
        let interactive_max_depth = INTERACTIVE_MAX_DEPTH.min(full_max_depth);
        let mut last_camera_update: Option<Instant> = None;

        loop {
            if abort.try_recv().is_ok() {
                return Ok(());
            }

            let mut latest_camera_config = idle_camera_config.take();
            while let Ok(config) = camera_config.try_recv() {
                latest_camera_config = Some(config);
            }

            if let Some(config) = latest_camera_config {
                self.update_camera(&config);
                // Restart the accumulation. The output buffer deliberately is
                // not cleared: the shader overwrites every pixel it covers when
                // sample_count is zero, so clearing only costs a dispatch and
                // leaves a window in which a caller blitting the buffer sees
                // black.
                completed = 0;
                // Get an image of the new view out as fast as possible, then
                // grow back into the budget. Carrying a large batch across the
                // restart would spend a whole dispatch before showing anything
                // of where the camera now points.
                batch_size = 1;
                last_camera_update = Some(Instant::now());
            }

            // Trade ray depth for responsiveness while the camera is moving,
            // and restore it once the view has settled. Both directions change
            // what a sample means, so the accumulation restarts either way --
            // on the way in it has restarted already.
            let interacting =
                last_camera_update.is_some_and(|at| at.elapsed() < INTERACTION_TIMEOUT);
            if !interacting {
                last_camera_update = None;
            }

            let wanted_max_depth = if interacting {
                interactive_max_depth
            } else {
                full_max_depth
            };
            if self.render_config.max_depth != wanted_max_depth {
                self.render_config.max_depth = wanted_max_depth;
                completed = 0;
            }

            if completed >= samples_per_pixel {
                if !idle_after_rendering {
                    break;
                }
                // Block on the channel rather than polling it, so a converged
                // image costs nothing and reacts the moment a camera update
                // arrives.
                match camera_config.recv_timeout(POLL_SLICE) {
                    Ok(config) => idle_camera_config = Some(config),
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return Ok(()),
                }
                continue;
            }

            // Never overshoot the requested sample count.
            let batch = batch_size.min(samples_per_pixel - completed);
            self.render_config.sample_count = completed;
            self.render_config.samples_per_batch = batch;
            self.queue.write_buffer(
                &self.config_buffer,
                0,
                bytemuck::cast_slice(&[self.render_config]),
            );

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            add_compute_pass_2d(
                &mut encoder,
                &self.pipeline,
                &self.bind_group,
                workgroup_count_x,
                workgroup_count_y,
            );

            if completed + batch >= samples_per_pixel {
                for p in &self.post_processors {
                    p.post_process(&mut encoder, &self.output_buffer, self.device)?;
                }
            }

            let command_buffer = encoder.finish();
            let dispatch_start = Instant::now();
            let submission = self.queue.submit([command_buffer]);

            // Back-pressure. Submissions on a queue execute in order, so
            // anything left queued here is added directly to the frame latency
            // of a caller sharing the device -- and without a wait this loop
            // runs thousands of iterations ahead of the GPU. Waiting also means
            // the progress reported below describes work that has actually
            // completed.
            if !wait_for_submission(self.device, &submission, abort)? {
                return Ok(());
            }
            let dispatch_time = dispatch_start.elapsed();

            completed += batch;

            let sample_ms = dispatch_time.as_secs_f64() * 1000. / batch as f64;
            let ema = ms_per_sample.map_or(sample_ms, |prev| prev * 0.8 + sample_ms * 0.2);
            ms_per_sample = Some(ema);

            let target = (TARGET_DISPATCH.as_secs_f64() * 1000. / ema.max(1e-3)) as u32;
            // Grow at most by doubling, so a single anomalously cheap dispatch
            // cannot blow the batch size up and stall the next frame.
            batch_size = target.clamp(1, (batch_size * 2).min(MAX_BATCH));

            output.send(RenderProgress {
                progress: completed as f64 / samples_per_pixel as f64,
                fps: Some(1000. / ema),
                estimated_time_left: calculate_estimated_time_left(
                    ema,
                    samples_per_pixel - completed,
                ),
                output_buffer: self.output_buffer.clone(),
            })?;
        }

        Ok(())
    }
}

/// Time left, from the measured cost of a sample rather than from the elapsed
/// time of the run. Restarting the accumulation on a camera change therefore
/// does not throw the estimate off.
fn calculate_estimated_time_left(ms_per_sample: f64, samples_left: u32) -> Duration {
    Duration::from_secs_f64(ms_per_sample / 1000. * samples_left as f64)
}

fn camera_to_gpu(camera_inst: &Camera) -> GpuCamera {
    GpuCamera {
        origin: [
            camera_inst.origin.x as f32,
            camera_inst.origin.y as f32,
            camera_inst.origin.z as f32,
        ],
        lens_radius: camera_inst.lens_radius as f32,
        lower_left_corner: [
            camera_inst.lower_left_corner.x as f32,
            camera_inst.lower_left_corner.y as f32,
            camera_inst.lower_left_corner.z as f32,
        ],
        _pad1: 0.0,
        horizontal: [
            camera_inst.horizontal.x as f32,
            camera_inst.horizontal.y as f32,
            camera_inst.horizontal.z as f32,
        ],
        _pad2: 0.0,
        vertical: [
            camera_inst.vertical.x as f32,
            camera_inst.vertical.y as f32,
            camera_inst.vertical.z as f32,
        ],
        _pad3: 0.0,
        u: [
            camera_inst.u.x as f32,
            camera_inst.u.y as f32,
            camera_inst.u.z as f32,
        ],
        _pad4: 0.0,
        v: [
            camera_inst.v.x as f32,
            camera_inst.v.y as f32,
            camera_inst.v.z as f32,
        ],
        _pad5: 0.0,
    }
}

fn create_and_upload_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &[T],
    usage: BufferUsages,
) -> wgpu::Buffer {
    let size_bytes = size_of_val(data) as u64;

    // Ensure the minimum size for a valid buffer and pad to 16 bytes for WGSL array compatibility
    let mut effective_size = if size_bytes == 0 {
        size_of::<T>() as u64
    } else {
        size_bytes
    };

    if effective_size % 16 != 0 {
        effective_size = ((effective_size / 16) + 1) * 16;
    }

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: effective_size,
        usage: usage | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    if size_bytes > 0 {
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));
    }

    buffer
}

#[cfg(test)]
mod test {
    use crate::renderer::calculate_estimated_time_left;
    use std::time::Duration;

    #[test]
    fn test_calculate_estimated_time_left() {
        let mut time_left = calculate_estimated_time_left(1000., 99);
        assert_eq!(time_left, Duration::from_secs(99));

        time_left = calculate_estimated_time_left(20., 50);
        assert_eq!(time_left, Duration::from_secs(1));

        time_left = calculate_estimated_time_left(20., 0);
        assert_eq!(time_left, Duration::from_secs(0));
    }

    #[test]
    fn test_update_camera() {
        use crate::camera::CameraConfig;
        use crate::geo::vec3::Vec3;
        use crate::hittable::{Bvh, Sphere};
        use crate::material::DiffuseLight;
        use crate::renderer::gpu_data::GpuCamera;
        use crate::renderer::{RenderConfig, Renderer, Scene};
        use crate::util::wgpu_util::{get_result_from_buffer, get_wgpu_device_and_queue};

        let (device, queue) = get_wgpu_device_and_queue();
        let render_config = RenderConfig {
            width: 10,
            height: 10,
            ..Default::default()
        };
        let mut world = Vec::new();
        world.push(
            Sphere::new(
                Vec3::new(0., 10., 0.),
                1.,
                DiffuseLight::new(1., 1., 1., None).into(),
            )
            .into(),
        );

        let scene = Scene {
            world: Bvh::new(world).into(),
            camera: CameraConfig {
                look_from: Vec3::new(0., 0., 1.),
                ..Default::default()
            },
            background_color: Vec3::new(0., 0., 0.),
            render_config,
        };

        let mut renderer = Renderer::new(scene, device, queue).unwrap();

        let new_config = CameraConfig {
            look_from: Vec3::new(0., 0., 10.),
            ..Default::default()
        };

        renderer.update_camera(&new_config);

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<GpuCamera>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            &renderer.camera_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of::<GpuCamera>() as u64,
        );
        queue.submit(Some(encoder.finish()));

        let camera_data: Vec<GpuCamera> = get_result_from_buffer(device, &staging_buffer);
        assert_eq!(camera_data[0].origin, [0., 0., 10.]);
    }
}
