#[cfg(test)]
mod tests {
    use solstrale::renderer::gpu_renderer::GpuRenderer;
    use solstrale::renderer::{Scene};
    use solstrale::camera::{CameraConfig};
    use solstrale::geo::vec3::Vec3;
    use solstrale::hittable::{Bvh, Hittables};
    use std::sync::mpsc::channel;
    use std::time::Duration;

    #[test]
    fn test_gpu_renderer_initialization_and_basic_render() {
        let scene = Scene {
            world: Hittables::Bvh(Bvh::new(vec![])), // Empty world for now
            camera: CameraConfig {
                look_from: Vec3::new(0., 0., 0.),
                look_at: Vec3::new(0., 0., -1.),
                vertical_fov_degrees: 90.,
                up: Vec3::new(0., 1., 0.),
                aperture_size: 0.,
            },
            background_color: Vec3::new(0., 0., 0.),
            render_config: Default::default(),
        };

        // This might fail if WGPU is not available (e.g. CI without GPU), 
        // but we assume the environment supports it or we handle it gracefully.
        // For this "Hello World" test, we expect it to succeed.
        let renderer = GpuRenderer::new(scene);
        
        if let Err(e) = renderer {
            println!("Skipping test: WGPU initialization failed: {}", e);
            return;
        }
        let renderer = renderer.unwrap();

        let (tx, rx) = channel();
        let (_abort_tx, abort_rx) = channel();

        // Spawn render in a separate thread or just run it if it's blocking but fast enough
        // The render method is blocking in the current CPU implementation.
        
        let result = renderer.render(&tx, &abort_rx);
        assert!(result.is_ok());

        // We expect at least one progress report with an image
        let mut received_image = false;
        while let Ok(progress) = rx.recv_timeout(Duration::from_secs(5)) {
            if let Some(image) = progress.render_image {
                received_image = true;
                // Check if image has some content (e.g. not all black, or specific color from shader)
                // For the "Hello World" shader, we might just write Red (255, 0, 0)
                let pixel = image.get_pixel(0, 0);
                // We'll define the expected behavior: The Hello World shader should write RED.
                // Note: The CPU renderer writes background color if no hit. 
                // But here we are testing the compute pipeline specifically.
                assert_eq!(pixel.0, [255, 0, 0]);
                
                assert!(image.width() > 0);
                break;
            }
            if progress.progress >= 1.0 {
                break;
            }
        }

        assert!(received_image, "Did not receive any image from GpuRenderer");
    }
}
