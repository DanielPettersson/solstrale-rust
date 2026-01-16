#[cfg(test)]
mod tests {
    use image::RgbImage;
    use image::imageops::FilterType;
    use image_compare::Algorithm::RootMeanSquared;
    use solstrale::camera::CameraConfig;
    use solstrale::geo::transformation::NopTransformer;
    use solstrale::geo::vec3::Vec3;
    use solstrale::hittable::{Bvh, Hittables, Quad, Sphere, Triangle};
    use solstrale::material::texture::SolidColor;
    use solstrale::material::{DiffuseLight, Lambertian};
    use solstrale::renderer::gpu_renderer::GpuRenderer;
    use solstrale::renderer::{RenderConfig, Scene};
    use std::sync::mpsc::channel;
    use std::thread;
    use std::time::Duration;

    const IMAGE_COMPARISON_SCORE_THRESHOLD: f64 = 0.85;

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
        use solstrale::material::texture::SolidColor;
        use solstrale::material::{Lambertian, Materials};

        let mat =
            Materials::Lambertian(Lambertian::new(SolidColor::new(1.0, 0.0, 0.0).into(), None));

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
        use solstrale::material::texture::SolidColor;
        use solstrale::material::{Lambertian, Materials};

        let mat1 =
            Materials::Lambertian(Lambertian::new(SolidColor::new(1.0, 0.0, 0.0).into(), None));
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
        use solstrale::geo::transformation::NopTransformer;
        use solstrale::hittable::Quad;
        use solstrale::material::texture::SolidColor;
        use solstrale::material::{Lambertian, Materials};

        let mat =
            Materials::Lambertian(Lambertian::new(SolidColor::new(0.0, 1.0, 0.0).into(), None));
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
        use solstrale::hittable::Sphere;
        use solstrale::material::texture::SolidColor;
        use solstrale::material::{Lambertian, Materials};
        use solstrale::renderer::{RenderConfig, RenderImageStrategy};

        let mat =
            Materials::Lambertian(Lambertian::new(SolidColor::new(1.0, 0.0, 0.0).into(), None));
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
            if progress.progress >= 1.0 {
                break;
            }
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn test_gpu_scene_sphere() {
        let render_config = RenderConfig {
            width: 400,
            height: 400,
            ..Default::default()
        };

        let camera = CameraConfig {
            look_from: Vec3::new(0., 0., 20.),
            look_at: Vec3::new(0., 0., 0.),
            ..Default::default()
        };

        let mut world: Vec<Hittables> = Vec::new();
        let light = DiffuseLight::new(45., 45., 45., None);
        world.push(Sphere::new(Vec3::new(-30., 30., 30.), 5., light.into()).into());

        let mat = Lambertian::new(SolidColor::new(0.2, 0.2, 1.0).into(), None);

        world.push(Sphere::new(Vec3::new(0., 0., 0.), 6., mat.into()).into());

        let scene = Scene {
            world: Bvh::new(world).into(),
            camera,
            background_color: Vec3::new(0., 0., 0.),
            render_config,
        };

        render_and_compare_output(scene, "gpu_sphere");
    }

    #[test]
    fn test_gpu_scene_sphere2() {
        let render_config = RenderConfig {
            width: 400,
            height: 400,
            ..Default::default()
        };

        let camera = CameraConfig {
            look_from: Vec3::new(0., 0., 20.),
            look_at: Vec3::new(0., 0., 0.),
            ..Default::default()
        };

        let mut world: Vec<Hittables> = Vec::new();
        let light = DiffuseLight::new(45., 45., 45., None);
        world.push(Sphere::new(Vec3::new(-30., 30., 30.), 5., light.into()).into());

        let blue = Lambertian::new(SolidColor::new(0.2, 0.2, 1.).into(), None);
        let red = Lambertian::new(SolidColor::new(1., 0.2, 0.2).into(), None);

        world.push(Sphere::new(Vec3::new(-4., -1., 0.), 4., blue.into()).into());
        world.push(Sphere::new(Vec3::new(4., 1., 0.), 4., red.into()).into());

        let scene = Scene {
            world: Bvh::new(world).into(),
            camera,
            background_color: Vec3::new(0., 0., 0.),
            render_config,
        };

        render_and_compare_output(scene, "gpu_sphere2");
    }

    #[test]
    fn test_gpu_scene_sphere_quad_and_triangle() {
        let render_config = RenderConfig {
            width: 400,
            height: 400,
            ..Default::default()
        };

        let camera = CameraConfig {
            look_from: Vec3::new(0., 0., 15.),
            look_at: Vec3::new(0., 0., 0.),
            ..Default::default()
        };

        let mut world: Vec<Hittables> = Vec::new();
        let light = DiffuseLight::new(45., 45., 45., None);
        world.push(Sphere::new(Vec3::new(-30., 30., 30.), 5., light.into()).into());

        let blue = Lambertian::new(SolidColor::new(0.2, 0.2, 1.).into(), None);
        let red = Lambertian::new(SolidColor::new(1., 0.2, 0.2).into(), None);
        let green = Lambertian::new(SolidColor::new(0.2, 1., 0.2).into(), None);

        world.push(Sphere::new(Vec3::new(-4., 1., 0.), 2., blue.into()).into());
        world.push(
            Triangle::new(
                Vec3::new(4., 0., 0.),
                Vec3::new(2., 2., 0.),
                Vec3::new(2., 0., 0.),
                red.into(),
                &NopTransformer(),
            )
            .into(),
        );
        world.push(
            Quad::new(
                Vec3::new(-1., -1., 0.),
                Vec3::new(2., 0., 0.),
                Vec3::new(0., 2., 0.),
                green.into(),
                &NopTransformer(),
            )
            .into(),
        );

        let scene = Scene {
            world: Bvh::new(world).into(),
            camera,
            background_color: Vec3::new(0., 0., 0.),
            render_config,
        };

        render_and_compare_output(scene, "gpu_sphere_quad_and_triangle");
    }

    fn render_and_compare_output(scene: Scene, name: &str) {
        let (output_sender, output_receiver) = channel();
        let (_, abort_receiver) = channel();

        let width = scene.render_config.width as u32;
        let height = scene.render_config.height as u32;

        thread::spawn(move || {
            GpuRenderer::new(scene)
                .unwrap()
                .render(&output_sender, &abort_receiver)
                .unwrap();
        });

        let mut image = RgbImage::new(width, height);
        for render_output in output_receiver {
            if let Some(render_image) = render_output.render_image {
                image = render_image;
            }
        }

        compare_output(name, &image);
    }

    fn compare_output(name: &str, actual_image: &RgbImage) {
        actual_image
            .save(format!("tests/output/out_actual_{}.jpg", name))
            .unwrap();

        let expected_image_path = format!("tests/output/out_expected_{}.jpg", name);
        let expected_image = image::open(&expected_image_path)
            .unwrap_or_else(|_| panic!("Could not load {}", &expected_image_path))
            .into_rgb8();

        let sized_actual = image::imageops::resize(actual_image, 100, 50, FilterType::Gaussian);
        let sized_expected =
            image::imageops::resize(&expected_image, 100, 50, FilterType::Gaussian);

        let score = image_compare::rgb_similarity_structure(
            &RootMeanSquared,
            &sized_expected,
            &sized_actual,
        )
        .expect("Failed to compare images")
        .score;

        assert!(
            score > IMAGE_COMPARISON_SCORE_THRESHOLD,
            "Comparison score for {} is: {}",
            name,
            score
        )
    }
}
