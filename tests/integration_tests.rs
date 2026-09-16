use std::default::Default;
use std::ops::Deref;
use std::sync::mpsc::channel;
use std::thread;

use image::RgbImage;
use image::imageops::FilterType;
use image_compare::Algorithm::RootMeanSquared;
use solstrale::camera::CameraConfig;
use solstrale::geo::transformation::{
    NopTransformer, RotationX, RotationY, RotationZ, Transformations, Transformer,
};
use solstrale::geo::vec3::Vec3;
use solstrale::hittable::{Bvh, Hittables, Quad, Sphere, Triangle};
use solstrale::material::texture::SolidColor;
use solstrale::material::{DiffuseLight, Lambertian};
use solstrale::post::{BloomPostProcessor, DenoisePostProcessor, SaturationPostProcessor};
use solstrale::ray_trace;
use solstrale::renderer::{RenderConfig, Scene};

use crate::scenes::{
    create_blend_material_scene, create_light_attenuation_scene, create_normal_mapping_scene,
    create_normal_mapping_sphere_scene, create_obj_scene, create_obj_with_box,
    create_obj_with_triangle, create_quad_rotation_scene, create_simple_test_scene,
    create_specular_scene, create_test_scene, create_texture_mapping_scene, create_uv_scene,
};

mod scenes;

#[test]
fn test_scene() {
    let render_config = RenderConfig {
        width: 200,
        height: 100,
        samples_per_pixel: 100,
        ..Default::default()
    };
    let scene = create_test_scene(render_config);

    render_and_compare_output(scene, "test_scene", 0.9)
}

#[test]
fn test_scene_bloom() {
    let (device, _) = get_wgpu_device_and_queue();
    let render_config = RenderConfig {
        width: 200,
        height: 100,
        samples_per_pixel: 100,
        post_processors: vec![
            BloomPostProcessor::new(0.1, None, Some(3.0), &device)
                .unwrap()
                .into(),
        ],
        ..Default::default()
    };
    let scene = create_test_scene(render_config);

    render_and_compare_output(scene, "test_scene_bloom", 0.95)
}

#[test]
fn test_scene_saturation() {
    let (device, _) = get_wgpu_device_and_queue();
    let render_config = RenderConfig {
        width: 200,
        height: 100,
        samples_per_pixel: 100,
        post_processors: vec![SaturationPostProcessor::new(-0.7, &device).unwrap().into()],
        ..Default::default()
    };
    let scene = create_test_scene(render_config);

    render_and_compare_output(scene, "test_scene_sat", 0.95)
}

#[test]
fn test_scene_bloom_and_saturation() {
    let (device, _) = get_wgpu_device_and_queue();
    let render_config = RenderConfig {
        width: 200,
        height: 100,
        samples_per_pixel: 100,
        post_processors: vec![
            SaturationPostProcessor::new(-0.7, &device).unwrap().into(),
            BloomPostProcessor::new(0.1, None, Some(3.0), &device)
                .unwrap()
                .into(),
        ],
        ..Default::default()
    };
    let scene = create_test_scene(render_config);

    render_and_compare_output(scene, "test_scene_bloom_sat", 0.95)
}

#[test]
fn test_render_obj_with_textures() {
    let render_config = RenderConfig {
        width: 200,
        height: 100,
        ..Default::default()
    };
    let scene = create_obj_scene(render_config);

    render_and_compare_output(scene, "obj", 0.95);
}

#[test]
fn test_render_obj_with_default_material() {
    let render_config = RenderConfig {
        width: 200,
        height: 100,
        ..Default::default()
    };
    let scene = create_obj_with_box(render_config, "resources/obj/", "box.obj");

    render_and_compare_output(scene, "obj_default", 0.95);
}

#[test]
fn test_render_obj_with_diffuse_material() {
    let render_config = RenderConfig {
        width: 200,
        height: 100,
        ..Default::default()
    };
    let scene = create_obj_with_box(render_config, "resources/obj/", "boxWithMat.obj");

    render_and_compare_output(scene, "obj_diffuse", 0.95);
}

#[test]
fn test_render_uv_mapping() {
    let render_config = RenderConfig {
        width: 200,
        height: 200,
        ..Default::default()
    };
    let scene = create_uv_scene(render_config);

    render_and_compare_output(scene, "uv", 0.95);
}

#[test]
fn test_render_normal_mapping_disabled() {
    let render_config = RenderConfig {
        width: 300,
        height: 300,
        ..Default::default()
    };

    let scene = create_normal_mapping_scene(render_config, Vec3::new(30., 30., 30.), false);
    render_and_compare_output(scene, "normal_mapping_disabled", 0.95);
}

#[test]
fn test_render_normal_mapping_1() {
    let render_config = RenderConfig {
        width: 300,
        height: 300,
        ..Default::default()
    };

    let scene = create_normal_mapping_scene(render_config, Vec3::new(30., 30., 30.), true);
    render_and_compare_output(scene, "normal_mapping_1", 0.95);
}

#[test]
fn test_render_normal_mapping_2() {
    let render_config = RenderConfig {
        width: 300,
        height: 300,
        ..Default::default()
    };

    let scene = create_normal_mapping_scene(render_config, Vec3::new(-30., 30., 30.), true);
    render_and_compare_output(scene, "normal_mapping_2", 0.95);
}

#[test]
fn test_render_normal_mapping_sphere_1() {
    let render_config = RenderConfig {
        width: 300,
        height: 300,
        ..Default::default()
    };
    let scene = create_normal_mapping_sphere_scene(render_config, Vec3::new(-30., 30., 30.));
    render_and_compare_output(scene, "normal_mapping_sphere_1", 0.97);
}

#[test]
fn test_render_normal_mapping_sphere_2() {
    let render_config = RenderConfig {
        width: 300,
        height: 300,
        ..Default::default()
    };
    let scene = create_normal_mapping_sphere_scene(render_config, Vec3::new(30., 30., 30.));
    render_and_compare_output(scene, "normal_mapping_sphere_2", 0.97);
}

#[test]
fn test_render_scene_without_light() {
    let (device, queue) = get_wgpu_device_and_queue();
    let render_config = RenderConfig {
        width: 20,
        height: 10,
        ..Default::default()
    };
    let scene = create_simple_test_scene(render_config, false);

    let (output_sender, _) = channel();
    let (_, camera_config_receiver) = channel();
    let (_, abort_receiver) = channel();

    let res = ray_trace(
        scene,
        &output_sender,
        &camera_config_receiver,
        &abort_receiver,
        device,
        queue,
        false,
    );

    match res {
        Ok(_) => panic!("There should be an error"),
        Err(e) => assert_eq!("Scene should have at least one light", e.to_string()),
    }
}

#[test]
fn test_render_obj_with_normal_map() {
    let render_config = RenderConfig {
        width: 300,
        height: 300,
        ..Default::default()
    };
    let scene = create_obj_with_triangle(render_config, "resources/obj/", "triWithNormalMap.obj");

    render_and_compare_output(scene, "obj_normal_map", 0.95);
}

#[test]
fn test_render_obj_with_height_map() {
    let render_config = RenderConfig {
        width: 300,
        height: 300,
        ..Default::default()
    };
    let scene = create_obj_with_triangle(render_config, "resources/obj/", "triWithHeightMap.obj");

    render_and_compare_output(scene, "obj_height_map", 0.95);
}

#[test]
fn test_render_light_attenuation() {
    for attenuation_half_length in [Some(0.1), Some(0.8), None] {
        let render_config = RenderConfig {
            width: 300,
            height: 300,
            ..Default::default()
        };
        let scene = create_light_attenuation_scene(render_config, attenuation_half_length);

        render_and_compare_output(
            scene,
            &format!(
                "light_attenuation_{}",
                attenuation_half_length.map_or(-1., |a| a)
            ),
            0.95,
        );
    }
}

#[test]
fn test_aabb_of_rotated_quad() {
    let mut rotations: Vec<Box<dyn Transformer>> = Vec::new();
    rotations.push(Box::new(RotationX::new(40.)));
    rotations.push(Box::new(RotationY::new(40.)));
    rotations.push(Box::new(RotationZ::new(40.)));

    for (i, rotation) in rotations.iter().enumerate() {
        let scene = create_quad_rotation_scene(
            RenderConfig {
                width: 300,
                height: 300,
                ..RenderConfig::default()
            },
            rotation.deref(),
        );

        render_and_compare_output(scene, &format!("quad_rotated{}", i), 0.95);
    }
}

#[test]
fn test_blended_materials() {
    for blend_factor in [0., 0.5, 1.] {
        let scene = create_blend_material_scene(
            RenderConfig {
                width: 300,
                height: 300,
                ..RenderConfig::default()
            },
            blend_factor,
        );

        render_and_compare_output(scene, &format!("blended_materials_{}", blend_factor), 0.95);
    }
}

#[test]
fn test_texture_map() {
    let scene = create_texture_mapping_scene(RenderConfig {
        width: 300,
        height: 300,
        ..RenderConfig::default()
    });

    render_and_compare_output(scene, "texture_map", 0.95);
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

    render_and_compare_output(scene, "gpu_sphere", 0.95);
}

#[test]
fn test_gpu_scene_box() {
    let render_config = RenderConfig {
        width: 400,
        height: 400,
        ..Default::default()
    };

    let camera = CameraConfig {
        look_from: Vec3::new(0., 0., 12.),
        look_at: Vec3::new(0., 0., 0.),
        ..Default::default()
    };

    let mut world: Vec<Hittables> = Vec::new();
    let light = DiffuseLight::new(45., 45., 45., None);
    world.push(Sphere::new(Vec3::new(-10., 20., 30.), 5., light.into()).into());

    let mat = Lambertian::new(SolidColor::new(0.2, 0.2, 1.0).into(), None);

    let box_transformations = Transformations::new(vec![
        Box::new(RotationY::new(25.)),
        Box::new(RotationX::new(45.)),
    ]);

    world.append(&mut Quad::new_box(
        Vec3::new(-2.5, -2.5, -2.5),
        Vec3::new(2.5, 2.5, 2.5),
        mat.into(),
        &box_transformations,
    ));

    let scene = Scene {
        world: Bvh::new(world).into(),
        camera,
        background_color: Vec3::new(0., 0., 0.),
        render_config,
    };

    render_and_compare_output(scene, "gpu_box", 0.95);
}

#[test]
fn test_gpu_scene_quad() {
    let render_config = RenderConfig {
        width: 400,
        height: 400,
        ..Default::default()
    };

    let camera = CameraConfig {
        look_from: Vec3::new(278., 278., -800.),
        look_at: Vec3::new(278., 278., 0.),
        vertical_fov_degrees: 35.,
        ..Default::default()
    };

    let mut world: Vec<Hittables> = Vec::new();
    let light = DiffuseLight::new(45., 45., 45., None);
    world.push(
        Quad::new(
            Vec3::new(408., 554., 383.),
            Vec3::new(-260., 0., 0.),
            Vec3::new(0., 0., -210.),
            light.into(),
            &NopTransformer(),
        )
        .into(),
    );

    let mat = Lambertian::new(SolidColor::new(0.2, 0.2, 1.0).into(), None);

    world.push(
        Quad::new(
            Vec3::new(0., 0., 555.),
            Vec3::new(555., 0., 0.),
            Vec3::new(0., 555., 0.),
            mat.into(),
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

    render_and_compare_output(scene, "gpu_quad", 0.95);
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

    render_and_compare_output(scene, "gpu_sphere2", 0.95);
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

    render_and_compare_output(scene, "gpu_sphere_quad_and_triangle", 0.95);
}

#[test]
fn test_gpu_scene_triangle3() {
    let render_config = RenderConfig {
        width: 400,
        height: 400,
        ..Default::default()
    };

    let camera = CameraConfig {
        look_from: Vec3::new(0., 0., 10.),
        look_at: Vec3::new(0., 0., 0.),
        ..Default::default()
    };

    let mut world: Vec<Hittables> = Vec::new();
    let light = DiffuseLight::new(45., 45., 45., None);
    world.push(Sphere::new(Vec3::new(-30., 30., 30.), 5., light.into()).into());

    let blue = Lambertian::new(SolidColor::new(0.2, 0.2, 1.).into(), None);
    let red = Lambertian::new(SolidColor::new(1., 0.2, 0.2).into(), None);
    let green = Lambertian::new(SolidColor::new(0.2, 1., 0.2).into(), None);

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
        Triangle::new(
            Vec3::new(2., -2., 1.),
            Vec3::new(0., 0., 1.),
            Vec3::new(0., -2., 1.),
            blue.into(),
            &NopTransformer(),
        )
        .into(),
    );
    world.push(
        Triangle::new(
            Vec3::new(3., -1., 1.),
            Vec3::new(1., 1., 1.),
            Vec3::new(1., -1., 1.),
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

    render_and_compare_output(scene, "gpu_triangle3", 0.95);
}

#[test]
fn test_gpu_scene_nested_bvh() {
    let render_config = RenderConfig {
        width: 400,
        height: 400,
        ..Default::default()
    };

    let camera = CameraConfig {
        look_from: Vec3::new(0., 0., 10.),
        look_at: Vec3::new(0., 0., 0.),
        ..Default::default()
    };

    let mut world: Vec<Hittables> = Vec::new();
    let light = DiffuseLight::new(45., 45., 45., None);
    world.push(Sphere::new(Vec3::new(-30., 30., 30.), 5., light.into()).into());

    let blue = Lambertian::new(SolidColor::new(0.2, 0.2, 1.).into(), None);
    let red = Lambertian::new(SolidColor::new(1., 0.2, 0.2).into(), None);
    let green = Lambertian::new(SolidColor::new(0.2, 1., 0.2).into(), None);

    world.push(Sphere::new(Vec3::new(-4., -1., 0.), 2., blue.into()).into());

    let mut sub_world: Vec<Hittables> = Vec::new();
    sub_world.push(Sphere::new(Vec3::new(0., -1., 0.), 2., red.into()).into());
    sub_world.push(Sphere::new(Vec3::new(4., -1., 0.), 2., green.into()).into());

    let bvh = Bvh::new(sub_world);
    world.push(bvh.into());

    let scene = Scene {
        world: Bvh::new(world).into(),
        camera,
        background_color: Vec3::new(0., 0., 0.),
        render_config,
    };

    render_and_compare_output(scene, "gpu_nested_bvh", 0.95);
}

use solstrale::util::tone_map::ToneMapper;
use solstrale::util::wgpu_util::{buffer_to_image, get_result_from_buffer, get_wgpu_device_and_queue};

fn render_and_compare_output(scene: Scene, name: &str, comparison_threshold: f64) {
    let (device, queue) = get_wgpu_device_and_queue();
    let (output_sender, output_receiver) = channel();
    let (_, camera_config_receiver) = channel();
    let (_, abort_receiver) = channel();

    let width = scene.render_config.width as u32;
    let height = scene.render_config.height as u32;

    thread::spawn(move || {
        ray_trace(
            scene,
            &output_sender,
            &camera_config_receiver,
            &abort_receiver,
            device,
            queue,
            false,
        )
        .unwrap();
    });

    let mut output_buffer = None;
    for render_output in output_receiver {
        output_buffer = Some(render_output.output_buffer);
    }

    let image = buffer_to_image(
        device,
        queue,
        &output_buffer.unwrap(),
        width,
        height,
        ToneMapper::default(),
    );

    compare_output(name, &image, comparison_threshold);
}

fn compare_output(name: &str, actual_image: &RgbImage, comparison_threshold: f64) {
    actual_image
        .save(format!("tests/output/out_actual_{}.jpg", name))
        .unwrap();

    let expected_image_path = format!("tests/output/out_expected_{}.jpg", name);
    let expected_image = image::open(&expected_image_path)
        .unwrap_or_else(|_| panic!("Could not load {}", &expected_image_path))
        .into_rgb8();

    let sized_actual = image::imageops::resize(actual_image, 100, 50, FilterType::Gaussian);
    let sized_expected = image::imageops::resize(&expected_image, 100, 50, FilterType::Gaussian);

    let score =
        image_compare::rgb_similarity_structure(&RootMeanSquared, &sized_expected, &sized_actual)
            .expect("Failed to compare images")
            .score;

    assert!(
        score > comparison_threshold,
        "Comparison score for {} is: {}",
        name,
        score
    )
}

// Adaptive sampling changes how many samples a pixel gets once it has
// converged, not what it converges to. This is the check TODO.md asks for:
// unlike the gamma-mapped RMS golden tests above (which tolerate a biased but
// structurally similar image), comparing the linear-radiance mean across spp
// levels would catch an energy bias introduced by the convergence heuristic.
#[test]
fn test_adaptive_sampling_convergence() {
    let (device, queue) = get_wgpu_device_and_queue();

    let spp_levels = [50u32, 200u32, 2000u32];
    let means: Vec<f64> = spp_levels
        .iter()
        .map(|&samples_per_pixel| {
            let render_config = RenderConfig {
                width: 100,
                height: 60,
                samples_per_pixel,
                ..Default::default()
            };
            let scene = create_test_scene(render_config);
            mean_linear_radiance(scene, device, queue)
        })
        .collect();

    let baseline = means[0];
    for (spp, mean) in spp_levels.iter().zip(means.iter()) {
        // Absolute, not just relative: the assert below only checks that the
        // mean is flat across spp within one build, which says nothing about
        // where the whole curve sits. Anything that changes how much energy
        // reaches the image -- the firefly clamp above all -- moves all three
        // means together and is invisible to the assert. Printing them is what
        // makes a before/after comparison across builds possible at all.
        println!("mean linear radiance at {} spp: {}", spp, mean);

        assert!(
            (mean - baseline).abs() / baseline < 0.05,
            "mean linear radiance at {} spp ({}) diverges from the {} spp baseline ({}) by more than 5%",
            spp,
            mean,
            spp_levels[0],
            baseline
        );
    }
}

fn mean_linear_radiance(
    scene: Scene,
    device: &'static wgpu::Device,
    queue: &'static wgpu::Queue,
) -> f64 {
    let (output_sender, output_receiver) = channel();
    let (_, camera_config_receiver) = channel();
    let (_, abort_receiver) = channel();

    let width = scene.render_config.width as u32;
    let height = scene.render_config.height as u32;

    thread::spawn(move || {
        ray_trace(
            scene,
            &output_sender,
            &camera_config_receiver,
            &abort_receiver,
            device,
            queue,
            false,
        )
        .unwrap();
    });

    let mut output_buffer = None;
    for render_output in output_receiver {
        output_buffer = Some(render_output.output_buffer);
    }
    let output_buffer = output_buffer.unwrap();

    let size = (width * height * 16) as u64;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, size);
    queue.submit(Some(encoder.finish()));

    let result: Vec<[f32; 4]> = get_result_from_buffer(device, &staging_buffer);
    let sum: f64 = result
        .iter()
        .map(|p| (p[0] + p[1] + p[2]) as f64 / 3.0)
        .sum();
    sum / result.len() as f64
}

/// Renders a scene and returns its raw linear pixels, before gamma.
fn render_linear(
    scene: Scene,
    device: &'static wgpu::Device,
    queue: &'static wgpu::Queue,
) -> Vec<[f32; 4]> {
    let (output_sender, output_receiver) = channel();
    let (_, camera_config_receiver) = channel();
    let (_, abort_receiver) = channel();

    let width = scene.render_config.width as u32;
    let height = scene.render_config.height as u32;

    thread::spawn(move || {
        ray_trace(
            scene,
            &output_sender,
            &camera_config_receiver,
            &abort_receiver,
            device,
            queue,
            false,
        )
        .unwrap();
    });

    let mut output_buffer = None;
    for render_output in output_receiver {
        output_buffer = Some(render_output.output_buffer);
    }

    let size = (width * height * 16) as u64;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&output_buffer.unwrap(), 0, &staging_buffer, 0, size);
    queue.submit(Some(encoder.finish()));

    get_result_from_buffer(device, &staging_buffer)
}

/// Root mean squared error between two linear images, over the colour channels.
fn linear_rmse(a: &[[f32; 4]], b: &[[f32; 4]]) -> f64 {
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(p, q)| {
            (0..3)
                .map(|c| {
                    let d = (p[c] - q[c]) as f64;
                    d * d
                })
                .sum::<f64>()
        })
        .sum();
    (sum / (a.len() * 3) as f64).sqrt()
}

/// Every post-processor has to survive an image larger than a flat 1-D dispatch
/// can address.
///
/// A `workgroup_size(64)` pass dispatched as `(width * height) / 64` workgroups
/// crosses `max_compute_workgroups_per_dimension` -- 65535 on a Radeon RX 5700
/// XT, so 4194240 pixels -- and fails wgpu validation outright, which surfaces
/// as a panic from the default uncaptured-error handler rather than as a
/// `Result`. Bloom and saturation were both built that way, and both broke below
/// 4K, on the same limit the tracer's dispatch was converted to a 2-D grid for.
///
/// 2732x1536 is the smallest 16:9 size past the ceiling, and is deliberately
/// only just past it: the point is the dispatch shape, not the resolution, and
/// the whole chain holds seven full-frame buffers at 16 bytes a pixel. One
/// sample per pixel, since nothing here looks at the image.
#[test]
fn test_post_processors_above_the_1d_dispatch_limit() {
    let (device, queue) = get_wgpu_device_and_queue();
    let (width, height) = (2732usize, 1536usize);
    assert!(
        width * height > device.limits().max_compute_workgroups_per_dimension as usize * 64,
        "test resolution no longer exceeds the 1-D dispatch ceiling on this adapter"
    );

    let render_config = RenderConfig {
        width,
        height,
        samples_per_pixel: 1,
        post_processors: vec![
            DenoisePostProcessor::new(1., Some(1), None, device)
                .unwrap()
                .into(),
            SaturationPostProcessor::new(-0.5, device).unwrap().into(),
            BloomPostProcessor::new(0.002, None, Some(3.0), device)
                .unwrap()
                .into(),
        ],
        ..Default::default()
    };

    let (output_sender, output_receiver) = channel();
    let (_c, camera_config_receiver) = channel();
    let (_a, abort_receiver) = channel();
    ray_trace(
        create_test_scene(render_config),
        &output_sender,
        &camera_config_receiver,
        &abort_receiver,
        device,
        queue,
        false,
    )
    .unwrap();
    drop(output_sender);

    let output_buffer = output_receiver
        .into_iter()
        .last()
        .expect("no render progress reported")
        .output_buffer;

    // Force the queue to drain, so a validation error in any recorded pass has
    // been raised by the time the test returns.
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    assert_eq!(
        output_buffer.size(),
        (width * height * 16) as u64,
        "post buffer is not the full image"
    );
}

/// Regression net for the denoise chain end to end, and for the documented
/// ordering: the denoiser runs first so bloom lands on a clean image rather than
/// being blurred by it.
#[test]
fn test_scene_denoise() {
    let (device, _) = get_wgpu_device_and_queue();
    let render_config = RenderConfig {
        width: 200,
        height: 100,
        samples_per_pixel: 16,
        post_processors: vec![
            DenoisePostProcessor::new(1., None, None, device)
                .unwrap()
                .into(),
        ],
        ..Default::default()
    };
    render_and_compare_output(create_test_scene(render_config), "denoise", 0.95);
}

#[test]
fn test_scene_denoise_and_bloom() {
    let (device, _) = get_wgpu_device_and_queue();
    let render_config = RenderConfig {
        width: 200,
        height: 100,
        samples_per_pixel: 16,
        post_processors: vec![
            DenoisePostProcessor::new(1., None, None, device)
                .unwrap()
                .into(),
            BloomPostProcessor::new(0.1, None, None, device)
                .unwrap()
                .into(),
        ],
        ..Default::default()
    };
    render_and_compare_output(create_test_scene(render_config), "denoise_and_bloom", 0.95);
}

/// Regression net for the specular guide: most of this frame is a mirror or a
/// lens, so a guide that described the specular surface instead of what it
/// shows would smear the reflected and refracted floor texture sideways and
/// move this image.
#[test]
fn test_scene_specular_denoise() {
    let (device, _) = get_wgpu_device_and_queue();
    let render_config = RenderConfig {
        width: 200,
        height: 100,
        samples_per_pixel: 16,
        post_processors: vec![
            DenoisePostProcessor::new(1., None, None, device)
                .unwrap()
                .into(),
        ],
        ..Default::default()
    };
    render_and_compare_output(
        create_specular_scene(render_config),
        "specular_denoise",
        0.95,
    );
}

/// A denoise test scene. `strength` of `None` leaves the chain empty, so the
/// same scene builder produces the control and the treatment.
fn denoise_scene(samples_per_pixel: u32, strength: Option<f64>, adaptive: bool) -> Scene {
    let (device, _) = get_wgpu_device_and_queue();
    let post_processors = match strength {
        Some(s) => vec![
            DenoisePostProcessor::new(s, None, None, device)
                .unwrap()
                .into(),
        ],
        None => vec![],
    };
    create_test_scene(RenderConfig {
        width: 200,
        height: 100,
        samples_per_pixel,
        // Above samples_per_pixel disables adaptive sampling, as the benches do.
        min_samples_per_pixel: if adaptive { 32 } else { u32::MAX },
        post_processors,
        ..Default::default()
    })
}

/// `denoise_scene`'s counterpart on the specular scene. Adaptive sampling is
/// always on here, as in the default configuration.
fn specular_denoise_scene(samples_per_pixel: u32, strength: Option<f64>) -> Scene {
    let (device, _) = get_wgpu_device_and_queue();
    let post_processors = match strength {
        Some(s) => vec![
            DenoisePostProcessor::new(s, None, None, device)
                .unwrap()
                .into(),
        ],
        None => vec![],
    };
    create_specular_scene(RenderConfig {
        width: 200,
        height: 100,
        samples_per_pixel,
        min_samples_per_pixel: 32,
        post_processors,
        ..Default::default()
    })
}

/// The objective gate: denoising a low-sample render must land it measurably
/// closer to a converged reference than the noisy render it started from.
///
/// This is a controlled experiment rather than a noise comparison. The RNG seed
/// in `trace_sample` is a pure function of pixel, sample index and restart
/// index; `restart_index` is zero for a non-interactive render; and adaptive
/// sampling cannot fire at 8 spp because `min_samples_per_pixel` defaults to 32.
/// So the noisy and denoised renders trace bit-identical sample streams and the
/// only difference between the two images is the filter.
///
/// Measured in linear space at full resolution. Deliberately *not* through
/// `compare_output`, whose 100x50 Gaussian downsample is itself a denoiser and
/// would wash out most of the effect being measured.
#[test]
fn test_denoise_improves_low_sample_image() {
    let (device, queue) = get_wgpu_device_and_queue();

    let reference = render_linear(denoise_scene(2000, None, true), device, queue);
    let noisy = render_linear(denoise_scene(8, None, true), device, queue);
    let denoised = render_linear(denoise_scene(8, Some(1.), true), device, queue);

    let rmse_noisy = linear_rmse(&noisy, &reference);
    let rmse_denoised = linear_rmse(&denoised, &reference);

    println!("linear RMSE against 2000 spp reference:");
    println!("  8 spp, no denoiser: {}", rmse_noisy);
    println!("  8 spp, denoised:    {}", rmse_denoised);
    println!("  ratio:              {}", rmse_denoised / rmse_noisy);

    assert!(
        rmse_denoised < rmse_noisy * 0.7,
        "denoising 8 spp should cut linear RMSE against the reference by at least 30%, \
         was {} against {}",
        rmse_denoised,
        rmse_noisy
    );
}

/// The same controlled experiment as above, on a scene that is mostly mirror
/// and glass: the filter has to still be a net win where most of the frame is
/// seen in a reflection or through a lens.
///
/// Deliberately not the gate on the specular guide itself. Global RMSE at a low
/// sample count rewards blurring, so a guide that smears the reflected image
/// along the mirror scores about the same here -- measured, the primary-hit
/// guide gives 0.71 against this guide's 0.72. What pins the guide is
/// `renderer::test::test_gbuffer_follows_specular_chain`, which checks it
/// analytically, and `test_scene_specular_denoise`, which would move if the
/// reflections changed.
#[test]
fn test_denoise_improves_specular_image() {
    let (device, queue) = get_wgpu_device_and_queue();

    let reference = render_linear(specular_denoise_scene(2000, None), device, queue);
    let noisy = render_linear(specular_denoise_scene(8, None), device, queue);
    let denoised = render_linear(specular_denoise_scene(8, Some(1.)), device, queue);

    let rmse_noisy = linear_rmse(&noisy, &reference);
    let rmse_denoised = linear_rmse(&denoised, &reference);

    println!("specular scene, linear RMSE against 2000 spp reference:");
    println!("  8 spp, no denoiser: {}", rmse_noisy);
    println!("  8 spp, denoised:    {}", rmse_denoised);
    println!("  ratio:              {}", rmse_denoised / rmse_noisy);

    // 0.72 measured. The margin is wider than the 0.7 the diffuse scene asserts
    // because the 2000 spp reference is itself adaptively sampled and moves a
    // little run to run.
    assert!(
        rmse_denoised < rmse_noisy * 0.8,
        "denoising 8 spp of a specular scene should cut linear RMSE against the \
         reference by at least 20%, was {} against {}",
        rmse_denoised,
        rmse_noisy
    );
}

/// The filter has to fade itself out as the render converges, because its
/// luminance tolerance is set by the variance of the pixel mean and that falls
/// as 1/n. Without this the filter would keep a fixed blur floor that a
/// single-spp golden test cannot see.
#[test]
fn test_denoise_is_near_identity_at_high_samples() {
    let (device, queue) = get_wgpu_device_and_queue();

    // Adaptive sampling off, so every pixel really has 2000 samples. With it on,
    // pixels retire at 5% relative standard error and the image is not converged
    // in the sense this test is about.
    let plain = render_linear(denoise_scene(2000, None, false), device, queue);
    let denoised = render_linear(denoise_scene(2000, Some(1.), false), device, queue);

    let mean: f64 = plain
        .iter()
        .map(|p| (p[0] + p[1] + p[2]) as f64 / 3.0)
        .sum::<f64>()
        / plain.len() as f64;
    let relative = linear_rmse(&denoised, &plain) / mean;

    println!(
        "relative linear RMSE of the denoiser at 2000 spp: {}",
        relative
    );

    assert!(
        relative < 0.02,
        "at 2000 spp the denoiser should be close to the identity, relative RMSE was {}",
        relative
    );
}

/// Diagnostic, not a gate: prints linear RMSE against a converged reference for
/// a range of strengths at a range of sample counts. This is how the defaults in
/// `DenoisePostProcessor::initialize` were chosen, and re-running it is how to
/// re-choose them. `cargo test --release denoise_strength_sweep -- --ignored --nocapture`
#[test]
#[ignore]
fn denoise_strength_sweep() {
    let (device, queue) = get_wgpu_device_and_queue();
    let reference = render_linear(denoise_scene(4000, None, false), device, queue);

    for adaptive in [false, true] {
        for spp in [8u32, 64, 2000] {
            let plain = render_linear(denoise_scene(spp, None, adaptive), device, queue);
            let base = linear_rmse(&plain, &reference);
            let mut line = format!("adaptive={} spp={:<5} none={:.4}", adaptive, spp, base);
            for s in [0.25f64, 0.5, 1.0, 2.0] {
                let d = render_linear(denoise_scene(spp, Some(s), adaptive), device, queue);
                let rmse = linear_rmse(&d, &reference);
                line += &format!("  s={}: {:.4} ({:.2})", s, rmse, rmse / base);
            }
            println!("{}", line);
        }
    }
}

/// Diagnostic, not a gate: linear RMSE against a converged reference on the
/// specular scene, across sample counts and strengths.
///
/// Read it knowing what it cannot see. Global RMSE is dominated by the pixels
/// that are neither mirror nor lens, and at a low sample count it rewards
/// blurring -- including blurring the reflected image along the mirror. Run
/// against the primary-hit guide (`GUIDE_MAX_SPECULAR` set to 0 in
/// ray_trace.wgsl) the two agree to within a couple of percent at every sample
/// count, in both directions. So this is a net-effect check, not a measurement
/// of the specular guide; for that, look at the images `denoise_visual_pair`
/// saves.
/// `cargo test --release specular_denoise_sweep -- --ignored --nocapture`
#[test]
#[ignore]
fn specular_denoise_sweep() {
    let (device, queue) = get_wgpu_device_and_queue();
    let reference = render_linear(specular_denoise_scene(4000, None), device, queue);

    for spp in [8u32, 32, 64, 200, 800] {
        let plain = render_linear(specular_denoise_scene(spp, None), device, queue);
        let base = linear_rmse(&plain, &reference);
        let mut line = format!("spp={:<5} none={:.4}", spp, base);
        for s in [0.5f64, 1.0, 2.0] {
            let d = render_linear(specular_denoise_scene(spp, Some(s)), device, queue);
            let rmse = linear_rmse(&d, &reference);
            line += &format!("  s={}: {:.4} ({:.3})", s, rmse, rmse / base);
        }
        println!("{}", line);
    }
}

/// Diagnostic, not a gate: saves a noisy/denoised pair for visual inspection,
/// which is how the golden images above were vetted before being promoted.
/// The specular pair is where the guide's specular chain shows: compare the
/// mirror and the glass sphere against a build with `GUIDE_MAX_SPECULAR` set
/// to 0 in ray_trace.wgsl, which is the old primary-hit guide.
/// `cargo test --release denoise_visual_pair -- --ignored`
#[test]
#[ignore]
fn denoise_visual_pair() {
    let (device, queue) = get_wgpu_device_and_queue();
    let save = |name: String, scene: Scene| {
        let (width, height) = (
            scene.render_config.width as u32,
            scene.render_config.height as u32,
        );
        let pixels = render_linear(scene, device, queue);
        // Same display transform as buffer_to_image, so what lands here is what
        // the renderer actually shows rather than the pre-tone-mapping clip.
        let img = encode(&pixels, width, height, ToneMapper::default());
        img.save(format!("tests/output/out_actual_visual_{}.png", name))
            .unwrap();
    };
    for (name, strength) in [("noisy", None), ("denoised", Some(1.))] {
        save(name.to_string(), denoise_scene(16, strength, true));
        save(
            format!("specular_{}", name),
            specular_denoise_scene(16, strength),
        );
    }
}

/// Applies the display transform to a linear buffer, exactly as
/// [`buffer_to_image`] does, but from pixels already read back.
fn encode(pixels: &[[f32; 4]], width: u32, height: u32, tone_mapper: ToneMapper) -> RgbImage {
    let mut img = RgbImage::new(width, height);
    for (i, p) in pixels.iter().enumerate() {
        let m = tone_mapper.map([p[0], p[1], p[2]]);
        let c = |v: f32| (v.sqrt().min(0.999) * 256.) as u8;
        img.put_pixel(
            i as u32 % width,
            i as u32 / width,
            image::Rgb([c(m[0]), c(m[1]), c(m[2])]),
        );
    }
    img
}

/// Renders one scene through every tone mapper, so the curves can be compared
/// on identical radiance rather than on separate renders.
///
/// The specular scene is the interesting one: it has a caustic under the glass
/// sphere and a bright horizon, which is exactly the range the old 1.0 clip
/// flattened into a single white.
/// `cargo test --release tone_map_visual_comparison -- --ignored`
#[test]
#[ignore]
fn tone_map_visual_comparison() {
    let (device, queue) = get_wgpu_device_and_queue();

    for (scene_name, scene_of) in [
        (
            "specular",
            (|| specular_denoise_scene(64, None)) as fn() -> Scene,
        ),
        ("test_scene", (|| denoise_scene(64, None, true)) as fn() -> Scene),
    ] {
        let scene = scene_of();
        let (width, height) = (
            scene.render_config.width as u32,
            scene.render_config.height as u32,
        );
        let pixels = render_linear(scene, device, queue);

        for (name, mapper) in [
            ("clamp", ToneMapper::Clamp),
            ("aces", ToneMapper::Aces),
            ("pbr_neutral", ToneMapper::PbrNeutral),
            ("reinhard", ToneMapper::Reinhard { white_point: 4. }),
        ] {
            encode(&pixels, width, height, mapper)
                .save(format!(
                    "tests/output/out_actual_tonemap_{}_{}.png",
                    scene_name, name
                ))
                .unwrap();
        }
    }
}
