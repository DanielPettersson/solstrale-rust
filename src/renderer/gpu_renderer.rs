//! GPU-based renderer implementation using WGPU

use crate::camera::Camera;
use crate::renderer::gpu_data::{GpuCamera, GpuRenderConfig};
use crate::renderer::scene_flattener::flatten_scene;
use crate::renderer::{RenderProgress, Scene};
use crate::util::wgpu_util::{
    add_buffer_copy, add_compute_pass, bind_group, bind_group_layout,
    compute_pipeline, get_result_from_buffer, get_wgpu_device_and_queue,
    storage_binding, uniform_binding,
};
use std::error::Error;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, SystemTime};
use image::{RgbImage, Rgb};
use wgpu::BufferUsages;

/// Renderer that uses the GPU to render the scene
pub struct GpuRenderer {
    #[allow(dead_code)]
    scene: Scene,
    width: u32,
    height: u32,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    output_buffer: wgpu::Buffer,
    download_buffer: wgpu::Buffer,
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
}

impl GpuRenderer {
    /// Creates a new GPU renderer given a scene
    pub fn new(scene: Scene) -> Result<Self, Box<dyn Error>> {
        let width = scene.render_config.width as u32;
        let height = scene.render_config.height as u32;
        let (device, queue) = get_wgpu_device_and_queue();

        let module = device.create_shader_module(wgpu::include_wgsl!("ray_trace.wgsl"));

        // Flatten scene
        let scene_data = flatten_scene(&scene);

        // Create buffers
        let nodes_buffer = create_and_upload_buffer(device, queue, "Nodes Buffer", &scene_data.nodes, BufferUsages::STORAGE);
        let spheres_buffer = create_and_upload_buffer(device, queue, "Spheres Buffer", &scene_data.spheres, BufferUsages::STORAGE);
        let triangles_buffer = create_and_upload_buffer(device, queue, "Triangles Buffer", &scene_data.triangles, BufferUsages::STORAGE);
        let quads_buffer = create_and_upload_buffer(device, queue, "Quads Buffer", &scene_data.quads, BufferUsages::STORAGE);
        let materials_buffer = create_and_upload_buffer(device, queue, "Materials Buffer", &scene_data.materials, BufferUsages::STORAGE);

        let camera = Camera::new(width as usize, height as usize, &scene.camera);
        // We need to access private fields of Camera. Let's check Camera struct in camera.rs
        // origin, lower_left_corner, horizontal, vertical, lens_radius
        // They are private. I should make them pub(crate) as well.
        
        let gpu_camera = GpuCamera {
            origin: [camera.origin.x as f32, camera.origin.y as f32, camera.origin.z as f32],
            _pad0: 0.0,
            lower_left_corner: [camera.lower_left_corner.x as f32, camera.lower_left_corner.y as f32, camera.lower_left_corner.z as f32],
            _pad1: 0.0,
            horizontal: [camera.horizontal.x as f32, camera.horizontal.y as f32, camera.horizontal.z as f32],
            _pad2: 0.0,
            vertical: [camera.vertical.x as f32, camera.vertical.y as f32, camera.vertical.z as f32],
            lens_radius: camera.lens_radius as f32,
        };
        let camera_buffer = create_and_upload_buffer(device, queue, "Camera Buffer", &[gpu_camera], BufferUsages::UNIFORM);

        let gpu_config = GpuRenderConfig { width, height };
        let config_buffer = create_and_upload_buffer(device, queue, "Config Buffer", &[gpu_config], BufferUsages::UNIFORM);

        let bind_group_layout = bind_group_layout(
            device,
            &[
                storage_binding(false, 16), // 0: output buffer
                storage_binding(true, 32),  // 1: nodes
                storage_binding(true, 32),  // 2: spheres
                storage_binding(true, 64),  // 3: triangles
                storage_binding(true, 96),  // 4: quads
                storage_binding(true, 48),  // 5: materials
                uniform_binding(64),        // 6: camera
                uniform_binding(8),         // 7: config
            ],
        );

        let pipeline = compute_pipeline(
            device,
            &bind_group_layout,
            &module,
            &[],
        );

        let size = (width * height * 16) as u64; // vec3 is 16 bytes aligned (as vec4 effectively)

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Download Buffer"),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = bind_group(
            device,
            &bind_group_layout,
            &[
                &output_buffer,
                &nodes_buffer,
                &spheres_buffer,
                &triangles_buffer,
                &quads_buffer,
                &materials_buffer,
                &camera_buffer,
                &config_buffer,
            ],
        );

        Ok(GpuRenderer {
            scene,
            width,
            height,
            bind_group_layout,
            pipeline,
            output_buffer,
            download_buffer,
            bind_group,
            nodes_buffer,
            spheres_buffer,
            triangles_buffer,
            quads_buffer,
            materials_buffer,
            camera_buffer,
            config_buffer,
        })
    }

    /// Executes the rendering of the image on the GPU
    pub fn render(
        &self,
        output: &Sender<RenderProgress>,
        abort: &Receiver<bool>,
    ) -> Result<(), Box<dyn Error>> {
        let (device, queue) = get_wgpu_device_and_queue();
        let render_start_time = SystemTime::now();

        // One pass "rendering" for now
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let pixel_count = self.width * self.height;
        let workgroup_count = pixel_count.div_ceil(64);

        add_compute_pass(&mut encoder, &self.pipeline, &self.bind_group, workgroup_count);
        add_buffer_copy(&mut encoder, &self.output_buffer, &self.download_buffer);

        let command_buffer = encoder.finish();
        queue.submit([command_buffer]);

        // Read back
        let result: Vec<[f32; 4]> = get_result_from_buffer(device, &self.download_buffer);
        
        if abort.try_recv().is_ok() {
             return Ok(());
        }

        // Convert to Image
        let mut img = RgbImage::new(self.width, self.height);
        for (i, pixel) in result.iter().enumerate() {
            let x = (i as u32) % self.width;
            let y = (i as u32) / self.width;
            if x < self.width && y < self.height {
                 // Simple tone mapping (clip)
                 let r = (pixel[0] * 255.0).clamp(0.0, 255.0) as u8;
                 let g = (pixel[1] * 255.0).clamp(0.0, 255.0) as u8;
                 let b = (pixel[2] * 255.0).clamp(0.0, 255.0) as u8;
                 img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }
        
        let now = SystemTime::now();
        let time_since_start = now
            .duration_since(render_start_time)
            .unwrap_or(Duration::from_millis(1));

        output.send(RenderProgress {
            progress: 1.0,
            fps: Some(1.0 / time_since_start.as_secs_f64()), 
            estimated_time_left: Duration::from_secs(0),
            render_image: Some(img),
        })?;

        Ok(())
    }
}

fn create_and_upload_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &[T],
    usage: BufferUsages,
) -> wgpu::Buffer {
    let size_bytes = (data.len() * std::mem::size_of::<T>()) as u64;
    let effective_size = if size_bytes == 0 {
        std::mem::size_of::<T>() as u64
    } else {
        size_bytes
    };

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
