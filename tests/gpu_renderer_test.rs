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
        // ... rest of test
        let (tx, rx) = channel();
        let (_abort_tx, abort_rx) = channel();
        
        let result = renderer.render(&tx, &abort_rx);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gpu_renderer_with_scene_objects() {
        use solstrale::hittable::Sphere;
        use solstrale::material::{Lambertian, Materials};
        use solstrale::material::texture::SolidColor;
        
        let mat = Materials::Lambertian(Lambertian::new(
            SolidColor::new(1.0, 0.0, 0.0).into(),
            None
        ));
        
        let sphere = Sphere::new(Vec3::new(0., 0., -2.), 1.0, mat);
        
        let scene = Scene {
            world: Hittables::Bvh(Bvh::new(vec![Hittables::Sphere(sphere)])),
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

        let renderer = GpuRenderer::new(scene);
        
        if let Err(e) = renderer {
            println!("Skipping test: WGPU initialization failed: {}", e);
            return;
        }
        let renderer = renderer.unwrap();
        
        let (tx, _rx) = channel();
        let (_abort_tx, abort_rx) = channel();
        
        let result = renderer.render(&tx, &abort_rx);
        assert!(result.is_ok());
    }
}
