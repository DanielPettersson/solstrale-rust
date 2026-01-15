//! GPU-based renderer implementation using WGPU

use crate::renderer::gpu_data::{BvhNode, Material, Quad, Sphere, Triangle};
use crate::renderer::scene_flattener::flatten_scene;
use crate::renderer::{RenderProgress, Scene};
use crate::util::wgpu_util::{
    add_buffer_copy, add_compute_pass, bind_group, bind_group_layout, bind_group_layout_entry,
    compute_pipeline, get_result_from_buffer, get_wgpu_device_and_queue,
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
        let nodes_buffer = create_and_upload_buffer(device, queue, "Nodes Buffer", &scene_data.nodes);
        let spheres_buffer = create_and_upload_buffer(device, queue, "Spheres Buffer", &scene_data.spheres);
        let triangles_buffer = create_and_upload_buffer(device, queue, "Triangles Buffer", &scene_data.triangles);
        let quads_buffer = create_and_upload_buffer(device, queue, "Quads Buffer", &scene_data.quads);
        let materials_buffer = create_and_upload_buffer(device, queue, "Materials Buffer", &scene_data.materials);

        let bind_group_layout = bind_group_layout(
            device,
            &[
                bind_group_layout_entry(false, 16), // 0: output buffer
                bind_group_layout_entry(true, 32),  // 1: nodes
                bind_group_layout_entry(true, 32),  // 2: spheres
                bind_group_layout_entry(true, 64),  // 3: triangles
                bind_group_layout_entry(true, 96),  // 4: quads
                bind_group_layout_entry(true, 48),  // 5: materials
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
) -> wgpu::Buffer {
    let size_bytes = (data.len() * std::mem::size_of::<T>()) as u64;
    // Ensure minimum size for valid buffer (e.g., 4 bytes? or align to struct size?)
    // Structs are aligned to 16/32 etc.
    // If empty, creating 0 size buffer is problematic for binding.
    // create a dummy buffer with size of 1 element if empty.
    let effective_size = if size_bytes == 0 {
        std::mem::size_of::<T>() as u64
    } else {
        size_bytes
    };

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: effective_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    if size_bytes > 0 {
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));
    }

    buffer
}