//! The renderer takes a [`Scene`] as input, renders it and reports [`RenderProgress`]

use crate::hittable::Hittable;
use crate::post::PostProcessor;
use std::error::Error;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, SystemTime};

use crate::camera::{Camera, CameraConfig};
use crate::geo::vec3::Vec3;
use crate::hittable::Hittables;
use crate::post::PostProcessors;
use crate::renderer::gpu_data::{GpuCamera, GpuRenderConfig};
use crate::renderer::scene_flattener::flatten_scene;
use crate::util::wgpu_util::{
    add_compute_pass, bind_group, bind_group_layout, compute_pipeline, sampler_binding,
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
    /// Post processor to apply to the rendered image
    pub post_processors: Vec<PostProcessors>,
    /// Describes at which points in time the render progress should contain an image
    pub render_image_strategy: RenderImageStrategy,
}

impl Default for RenderConfig {
    fn default() -> Self {
        RenderConfig {
            width: 300,
            height: 200,
            samples_per_pixel: 50,
            post_processors: vec![],
            render_image_strategy: RenderImageStrategy::OnlyFinal,
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

#[derive(Copy, Clone)]
/// When should [`RenderProgress`] contain an image of the rendering
pub enum RenderImageStrategy {
    /// Every sample should contain an image
    EverySample,
    /// Only include an image if at least "duration" has elapsed since last time
    /// Plus always include the final image
    Interval(Duration),
    /// Only include image in last rendered sample
    OnlyFinal,
}

impl RenderImageStrategy {
    /// Is it time to generate a new render image for the output channel?
    pub fn should_generate_image(
        &self,
        sample: u32,
        total_samples: u32,
        now: SystemTime,
        last_image_generated_time: SystemTime,
    ) -> bool {
        match self {
            RenderImageStrategy::EverySample => true,
            RenderImageStrategy::Interval(d) => {
                sample == total_samples
                    || now
                        .duration_since(last_image_generated_time)
                        .unwrap_or(Duration::from_millis(0))
                        > *d
            }
            RenderImageStrategy::OnlyFinal => sample == total_samples,
        }
    }
}

/// Renderer is a central part of the raytracer responsible for controlling the
/// process reporting back progress to the caller
pub struct Renderer<'a> {
    #[allow(dead_code)]
    scene: Scene,
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
    triangles_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    quads_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    materials_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    camera_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    config_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    lights_buffer: wgpu::Buffer,
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
        if scene.world.get_lights().is_empty() {
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
        let spheres_buffer = create_and_upload_buffer(
            device,
            queue,
            "Spheres Buffer",
            &scene_data.spheres,
            BufferUsages::STORAGE,
        );
        let triangles_buffer = create_and_upload_buffer(
            device,
            queue,
            "Triangles Buffer",
            &scene_data.triangles,
            BufferUsages::STORAGE,
        );
        let quads_buffer = create_and_upload_buffer(
            device,
            queue,
            "Quads Buffer",
            &scene_data.quads,
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

        // Create texture atlas
        let (max_atlas_width, max_atlas_height) = (8192, 8192);
        let mut atlas_image;

        if !scene_data.textures.is_empty() {
            let packer = crate::util::texture_processing::TexturePacker::new(
                max_atlas_width,
                max_atlas_height,
            );
            let dims: Vec<(u32, u32)> = scene_data
                .textures
                .iter()
                .map(|img| (img.width(), img.height()))
                .collect();
            let layout = packer.pack(&dims).expect("Failed to pack textures in renderer - this should have been caught in flatten_scene");

            atlas_image = RgbImage::new(layout.width, layout.height);

            for placement in layout.placements.iter() {
                let texture = &scene_data.textures[placement.original_index];
                image::imageops::replace(
                    &mut atlas_image,
                    texture,
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
            max_depth: 50,
            background_color: [
                scene.background_color.x as f32,
                scene.background_color.y as f32,
                scene.background_color.z as f32,
            ],
            light_count: scene_data.lights.len() as u32,
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
                wgpu::BindingResource::Buffer(triangles_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(quads_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(materials_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(camera_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(config_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::TextureView(&texture_view),
                wgpu::BindingResource::Sampler(&sampler),
                wgpu::BindingResource::Buffer(lights_buffer.as_entire_buffer_binding()),
            ],
        );

        let mut post_processors = scene.render_config.post_processors.clone();
        for p in &mut post_processors {
            p.initialize(device, queue, width, height);
        }

        Ok(Renderer {
            scene,
            width,
            height,
            bind_group_layout,
            pipeline,
            output_buffer,
            bind_group,
            nodes_buffer,
            spheres_buffer,
            triangles_buffer,
            quads_buffer,
            materials_buffer,
            camera_buffer,
            config_buffer,
            lights_buffer,
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
        let mut render_start_time = SystemTime::now();
        let samples_per_pixel = self.scene.render_config.samples_per_pixel;
        let pixel_count = self.width * self.height;
        let workgroup_count = pixel_count.div_ceil(64);

        let mut sample = 1;
        loop {
            let sample_start_time = SystemTime::now();
            if abort.try_recv().is_ok() {
                return Ok(());
            }

            let mut latest_camera_config = None;
            while let Ok(config) = camera_config.try_recv() {
                latest_camera_config = Some(config);
            }

            if let Some(config) = latest_camera_config {
                self.update_camera(&config);
                sample = 1;
                render_start_time = SystemTime::now();

                // Clear the output buffer when the camera moves
                let mut encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                encoder.clear_buffer(&self.output_buffer, 0, None);
                self.queue.submit([encoder.finish()]);
            }

            if sample > samples_per_pixel {
                if idle_after_rendering {
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                } else {
                    break;
                }
            }

            self.render_config.sample_count = sample;
            self.queue.write_buffer(
                &self.config_buffer,
                0,
                bytemuck::cast_slice(&[self.render_config]),
            );

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            add_compute_pass(
                &mut encoder,
                &self.pipeline,
                &self.bind_group,
                workgroup_count,
            );

            if sample == samples_per_pixel {
                for p in &self.post_processors {
                    p.post_process(&mut encoder, &self.output_buffer, self.device)?;
                }
            }

            let command_buffer = encoder.finish();
            self.queue.submit([command_buffer]);

            let now = SystemTime::now();

            output.send(RenderProgress {
                progress: sample as f64 / samples_per_pixel as f64,
                fps: Some(calculate_fps(render_start_time, now, sample)),
                estimated_time_left: calculate_estimated_time_left(
                    render_start_time,
                    now,
                    sample,
                    samples_per_pixel,
                ),
                output_buffer: self.output_buffer.clone(),
            })?;

            if now
                .duration_since(sample_start_time)
                .unwrap_or(Duration::from_millis(0))
                > Duration::from_millis(10)
            {
                std::thread::sleep(Duration::from_millis(1));
            }

            sample += 1;
        }

        Ok(())
    }
}

fn calculate_fps(render_start_time: SystemTime, now: SystemTime, samples_done: u32) -> f64 {
    let time_since_start = now
        .duration_since(render_start_time)
        .unwrap_or(Duration::from_millis(1));

    samples_done as f64 / time_since_start.as_secs_f64()
}

fn calculate_estimated_time_left(
    render_start_time: SystemTime,
    now: SystemTime,
    samples_done: u32,
    total_samples: u32,
) -> Duration {
    let time_since_start = now
        .duration_since(render_start_time)
        .unwrap_or(Duration::from_millis(1));
    let samples_left = total_samples - samples_done;

    time_since_start
        .div_f32(samples_done as f32)
        .mul_f32(samples_left as f32)
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
    use crate::renderer::{calculate_estimated_time_left, calculate_fps};
    use std::time::{Duration, SystemTime};
    #[test]
    fn test_calculate_fps() {
        let render_start = SystemTime::UNIX_EPOCH + Duration::from_millis(900);
        let now = SystemTime::UNIX_EPOCH + Duration::from_millis(1000);

        let fps = calculate_fps(render_start, now, 5);
        assert_eq!(fps, 50.);
    }

    #[test]
    fn test_calculate_estimated_time_left() {
        let render_start = SystemTime::UNIX_EPOCH;
        let now = SystemTime::UNIX_EPOCH + Duration::from_millis(1000);

        let mut time_left = calculate_estimated_time_left(render_start, now, 1, 100);
        assert_eq!(time_left, Duration::from_secs(99));

        time_left = calculate_estimated_time_left(render_start, now, 50, 100);
        assert_eq!(time_left, Duration::from_secs(1));

        time_left = calculate_estimated_time_left(render_start, now, 100, 100);
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
