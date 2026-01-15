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

        let camera_inst = Camera::new(width as usize, height as usize, &scene.camera);
        
        let gpu_camera = GpuCamera {
            origin: [camera_inst.origin.x as f32, camera_inst.origin.y as f32, camera_inst.origin.z as f32],
            _pad0: 0.0,
            lower_left_corner: [camera_inst.lower_left_corner.x as f32, camera_inst.lower_left_corner.y as f32, camera_inst.lower_left_corner.z as f32],
            _pad1: 0.0,
            horizontal: [camera_inst.horizontal.x as f32, camera_inst.horizontal.y as f32, camera_inst.horizontal.z as f32],
            _pad2: 0.0,
            vertical: [camera_inst.vertical.x as f32, camera_inst.vertical.y as f32, camera_inst.vertical.z as f32],
            lens_radius: camera_inst.lens_radius as f32,
        };
        let camera_buffer = create_and_upload_buffer(device, queue, "Camera Buffer", &[gpu_camera], BufferUsages::UNIFORM);

        let gpu_config = GpuRenderConfig { width, height, sample_count: 0, _pad: 0 };
        let config_buffer = create_and_upload_buffer(device, queue, "Config Buffer", &[gpu_config], BufferUsages::UNIFORM);

        let bind_group_layout = bind_group_layout(
            device,
            &[
                storage_binding(false, 0), // 0: output buffer
                storage_binding(true, 0),  // 1: nodes
                storage_binding(true, 0),  // 2: spheres
                storage_binding(true, 0),  // 3: triangles
                storage_binding(true, 0),  // 4: quads
                storage_binding(true, 0),  // 5: materials
                uniform_binding(std::mem::size_of::<GpuCamera>() as u64),      // 6: camera
                uniform_binding(std::mem::size_of::<GpuRenderConfig>() as u64), // 7: config
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
        let mut last_image_generated_time = SystemTime::UNIX_EPOCH;
        
        let samples_per_pixel = self.scene.render_config.samples_per_pixel;

        for sample in 1..=samples_per_pixel {
            if abort.try_recv().is_ok() {
                return Ok(());
            }

            // Update config buffer with current sample count
            let gpu_config = GpuRenderConfig { 
                width: self.width, 
                height: self.height, 
                sample_count: sample,
                _pad: 0 
            };
            queue.write_buffer(&self.config_buffer, 0, bytemuck::cast_slice(&[gpu_config]));

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            let pixel_count = self.width * self.height;
            let workgroup_count = pixel_count.div_ceil(64);

            add_compute_pass(&mut encoder, &self.pipeline, &self.bind_group, workgroup_count);
            
            let now = SystemTime::now();
            let should_generate_image = self.scene.render_config.render_image_strategy.should_generate_image(
                sample,
                samples_per_pixel,
                now,
                last_image_generated_time,
            );

            if should_generate_image {
                add_buffer_copy(&mut encoder, &self.output_buffer, &self.download_buffer);
            }

            let command_buffer = encoder.finish();
            queue.submit([command_buffer]);

            if should_generate_image {
                last_image_generated_time = now;
                
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
                         // Very simple accumulation visualization: divide by sample count
                         // In future we should accumulate in f32 buffer on GPU
                         let r = (pixel[0] * 255.0).clamp(0.0, 255.0) as u8;
                         let g = (pixel[1] * 255.0).clamp(0.0, 255.0) as u8;
                         let b = (pixel[2] * 255.0).clamp(0.0, 255.0) as u8;
                         img.put_pixel(x, y, Rgb([r, g, b]));
                    }
                }

                output.send(RenderProgress {
                    progress: sample as f64 / samples_per_pixel as f64,
                    fps: Some(sample as f64 / now.duration_since(render_start_time).unwrap_or(Duration::from_millis(1)).as_secs_f64()), 
                    estimated_time_left: Duration::from_secs(0), // TODO calculate
                    render_image: Some(img),
                })?;
            } else if sample == samples_per_pixel || sample % 10 == 0 {
                 output.send(RenderProgress {
                    progress: sample as f64 / samples_per_pixel as f64,
                    fps: Some(sample as f64 / now.duration_since(render_start_time).unwrap_or(Duration::from_millis(1)).as_secs_f64()), 
                    estimated_time_left: Duration::from_secs(0),
                    render_image: None,
                })?;
            }
        }

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
    
    // Ensure minimum size for valid buffer and pad to 16 bytes for WGSL array compatibility
    let mut effective_size = if size_bytes == 0 {
        std::mem::size_of::<T>() as u64
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