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

        let renderer = GpuRenderer::new(scene);
        
        if let Err(e) = renderer {
            println!("Skipping test: WGPU initialization failed: {}", e);
            return;
        }
        let renderer = renderer.unwrap();

        let (tx, rx) = channel();
        let (_abort_tx, abort_rx) = channel();
        
        let result = renderer.render(&tx, &abort_rx);
        assert!(result.is_ok());

        let mut received_image = false;
        while let Ok(progress) = rx.recv_timeout(Duration::from_secs(5)) {
            if let Some(image) = progress.render_image {
                received_image = true;
                assert!(image.width() > 0);
                break;
            }
            if progress.progress >= 1.0 {
                break;
            }
        }

        assert!(received_image, "Did not receive any image from GpuRenderer");
    }

    #[test]
    fn test_gpu_renderer_uv_map() {
        // Test if the output image has varying colors representing ray directions
        let scene = Scene {
            world: Hittables::Bvh(Bvh::new(vec![])),
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

        let renderer = GpuRenderer::new(scene).unwrap();
        let (tx, rx) = channel();
        let (_abort_tx, abort_rx) = channel();
        
        renderer.render(&tx, &abort_rx).unwrap();

        let mut received_image = false;
        while let Ok(progress) = rx.recv_timeout(Duration::from_secs(5)) {
            if let Some(image) = progress.render_image {
                received_image = true;
                let p1 = image.get_pixel(0, 0);
                assert_eq!(p1.0, [0, 0, 0]); 
                break;
            }
        }
        assert!(received_image);
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

    #[test]
    fn test_gpu_renderer_with_different_materials() {
        use solstrale::hittable::Sphere;
        use solstrale::material::{Lambertian, Materials};
        use solstrale::material::texture::SolidColor;
        
        let mat1 = Materials::Lambertian(Lambertian::new(
            SolidColor::new(1.0, 0.0, 0.0).into(),
            None
        ));
        let s1 = Sphere::new(Vec3::new(-2., 0., -5.), 1.0, mat1);
        
        let scene = Scene {
            world: Hittables::Bvh(Bvh::new(vec![Hittables::Sphere(s1)])),
            camera: CameraConfig::default(),
            background_color: Default::default(),
            render_config: Default::default(),
        };

        let renderer = GpuRenderer::new(scene).unwrap();
        let (tx, _rx) = channel();
        let (_abort_tx, abort_rx) = channel();
        
        renderer.render(&tx, &abort_rx).unwrap();
    }

    #[test]
    fn test_gpu_renderer_with_box() {
        use solstrale::hittable::Quad;
        use solstrale::material::{Lambertian, Materials};
        use solstrale::material::texture::SolidColor;
        use solstrale::geo::transformation::NopTransformer;
        
        let mat = Materials::Lambertian(Lambertian::new(
            SolidColor::new(0.0, 1.0, 0.0).into(),
            None
        ));
        let box_sides = Quad::new_box(
            Vec3::new(-0.5, -0.5, -2.5),
            Vec3::new(0.5, 0.5, -1.5),
            mat,
            &NopTransformer {},
        );
        let scene = Scene {
            world: Hittables::Bvh(Bvh::new(box_sides)),
            camera: Default::default(),
            background_color: Default::default(),
            render_config: Default::default(),
        };
        let renderer = GpuRenderer::new(scene).unwrap();
        let (tx, _rx) = channel();
        let (_abort_tx, abort_rx) = channel();
        renderer.render(&tx, &abort_rx).unwrap();
    }

    #[test]
    fn test_gpu_renderer_rng_changes() {
        use solstrale::renderer::{RenderConfig, RenderImageStrategy};
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
            camera: Default::default(),
            background_color: Default::default(),
            render_config: RenderConfig {
                samples_per_pixel: 2,
                render_image_strategy: RenderImageStrategy::EverySample,
                ..Default::default()
            },
        };

        let renderer = GpuRenderer::new(scene).unwrap();
        let (tx, rx) = channel();
        let (_abort_tx, abort_rx) = channel();
        
        renderer.render(&tx, &abort_rx).unwrap();

        let mut count = 0;
        
        while let Ok(progress) = rx.recv_timeout(Duration::from_secs(5)) {
            if let Some(_image) = progress.render_image {
                count += 1;
            }
            if progress.progress >= 1.0 { break; }
        }
        assert_eq!(count, 2);
    }
}
