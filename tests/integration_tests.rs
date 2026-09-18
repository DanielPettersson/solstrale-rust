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
    create_blend_material_scene, create_cornell_scene, create_light_attenuation_scene,
    create_normal_mapping_scene, create_normal_mapping_sphere_scene, create_obj_scene,
    create_obj_with_box, create_obj_with_triangle, create_quad_rotation_scene,
    create_simple_test_scene, create_specular_scene, create_test_scene,
    create_texture_mapping_scene, create_uv_scene,
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
use solstrale::util::wgpu_util::{
    buffer_to_image, get_result_from_buffer, get_wgpu_device_and_queue,
};

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
/// Root mean square of each pixel's luminance departure from its 3x3
/// neighbourhood, relative to the image's mean luminance: how grainy the image
/// looks, in one number.
///
/// Unlike RMSE against a reference this needs no reference, and it does not
/// reward blur -- which matters for the property below, where the failure being
/// guarded against is an image that is *less* blurred and more speckled.
/// It does count real pixel-scale detail, so it is only meaningful compared
/// between renders of the same scene.
fn grain(pixels: &[[f32; 4]], width: usize, height: usize) -> f64 {
    let lum = |p: &[f32; 4]| (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]) as f64;
    let mut sum_sq = 0.0;
    let mut sum_lum = 0.0;

    for y in 0..height {
        for x in 0..width {
            let mut local = 0.0;
            let mut taps = 0.0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;
                    let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as usize;
                    local += lum(&pixels[ny * width + nx]);
                    taps += 1.0;
                }
            }
            let d = lum(&pixels[y * width + x]) - local / taps;
            sum_sq += d * d;
            sum_lum += lum(&pixels[y * width + x]);
        }
    }

    let pixels_count = (width * height) as f64;
    (sum_sq / pixels_count).sqrt() / (sum_lum / pixels_count)
}

/// How many pixels read as an isolated bright speck: a count of those whose
/// displayed luminance exceeds the brightest of their four neighbours by more
/// than `excess`, on the 0-255 scale the image is finally written on.
///
/// Measured *through the display transform*, and that is the whole point. The
/// first version of this counted in linear radiance, and it lied: it scored a
/// change that removed 100% of the outliers it could see, while the rendered
/// image looked all but unchanged. ACES plus gamma 2.0 compresses highlights
/// hard, so a pixel pulled from twenty times its neighbourhood's brightness down
/// to three times has lost 85% of its excess radiance and almost none of its
/// visibility -- it is still a white dot on a dark ceiling. A metric in linear
/// space rewards the first 85% and says nothing about the part a viewer sees.
///
/// Reference-free, like [`grain`], and for the same reason: it must not reward
/// blur, since blur is what the thing being measured would otherwise hide
/// behind. Unlike `grain`, which is an RMS over every pixel and so is dominated
/// by the many small departures, this is a count -- a handful of pixels at ten
/// times the local level barely moves an RMS and is the only thing the eye
/// picks out.
#[allow(dead_code)]
fn fireflies(pixels: &[[f32; 4]], width: usize, height: usize, excess: f64) -> usize {
    // The same display transform buffer_to_image applies, so what is counted is
    // what is looked at.
    let displayed: Vec<f64> = pixels
        .iter()
        .map(|p| {
            let m = ToneMapper::default().map([p[0], p[1], p[2]]);
            let encode = |v: f32| (v.sqrt().min(0.999) * 256.) as f64;
            0.2126 * encode(m[0]) + 0.7152 * encode(m[1]) + 0.0722 * encode(m[2])
        })
        .collect();

    let mut count = 0;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let at = |dx: usize, dy: usize| displayed[dy * width + dx];
            let brightest_neighbour = at(x - 1, y)
                .max(at(x + 1, y))
                .max(at(x, y - 1))
                .max(at(x, y + 1));

            if at(x, y) - brightest_neighbour > excess {
                count += 1;
            }
        }
    }

    count
}

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

/// `denoise_scene`'s counterpart on the Cornell box, which is the scene that
/// actually produces fireflies. Adaptive sampling is off, as in `denoise_scene`,
/// so every pixel really has the sample count asked for and the comparison
/// across sample counts means what it says.
fn cornell_denoise_scene(samples_per_pixel: u32, strength: Option<f64>) -> Scene {
    cornell_denoise_scene_at(samples_per_pixel, strength, 200, 200)
}

fn cornell_denoise_scene_at(
    samples_per_pixel: u32,
    strength: Option<f64>,
    width: usize,
    height: usize,
) -> Scene {
    let (device, _) = get_wgpu_device_and_queue();
    let post_processors = match strength {
        Some(s) => vec![
            DenoisePostProcessor::new(s, None, None, device)
                .unwrap()
                .into(),
        ],
        None => vec![],
    };
    create_cornell_scene(RenderConfig {
        width,
        height,
        samples_per_pixel,
        min_samples_per_pixel: u32::MAX,
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

    // 0.56 measured.
    assert!(
        rmse_denoised < rmse_noisy * 0.65,
        "denoising 8 spp should cut linear RMSE against the reference by at least 35%, \
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

    // 0.68 measured. The margin is wider than the one the diffuse scene asserts
    // because the 2000 spp reference is itself adaptively sampled and moves a
    // little run to run.
    assert!(
        rmse_denoised < rmse_noisy * 0.75,
        "denoising 8 spp of a specular scene should cut linear RMSE against the \
         reference by at least 25%, was {} against {}",
        rmse_denoised,
        rmse_noisy
    );
}

/// Spending more samples must not make the denoised image grainier. Obvious,
/// and it did not hold: at 1 sample per pixel the chain has no Welford variance
/// to work from and falls back on estimates that cannot collapse, while from 2
/// samples up it trusted each pixel's own M2 -- one degree of freedom, reading
/// near zero for every pixel whose two samples happened to agree, which in a
/// path tracer means every pixel that missed the light twice. Each of those
/// declared itself converged and kept its raw value, so a 2 spp render came out
/// of the denoiser covered in speckle that a 1 spp render did not have.
///
/// Measured at strength 4, where the effect was reported and where it is
/// largest: a strong filter has the most to undo when a pixel opts out of it.
/// Before the fix the 2 spp image was 30% grainier than the 1 spp one.
#[test]
fn test_denoise_grain_does_not_grow_with_samples() {
    let (device, queue) = get_wgpu_device_and_queue();
    let (width, height) = (200, 100);

    let grain_at = |spp| {
        let pixels = render_linear(denoise_scene(spp, Some(4.), false), device, queue);
        grain(&pixels, width, height)
    };
    let (one, two, four) = (grain_at(1), grain_at(2), grain_at(4));

    println!(
        "denoised grain: 1 spp {:.4}, 2 spp {:.4}, 4 spp {:.4}",
        one, two, four
    );

    // 2 spp is allowed to be a shade grainier than 1: it is also less blurred,
    // and this metric counts the detail that buys as grain. 1.04 measured.
    assert!(
        two < one * 1.1,
        "denoising 2 spp came out grainier than 1 spp, {} against {}",
        two,
        one
    );
    // By 4 samples there is no excuse left. 0.92 measured.
    assert!(
        four < one,
        "denoising 4 spp came out grainier than 1 spp, {} against {}",
        four,
        one
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
        (
            "test_scene",
            (|| denoise_scene(64, None, true)) as fn() -> Scene,
        ),
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

/// The gate this whole change exists for: a denoised image must not keep
/// fireflies, at any sample count.
///
/// Before the despeckle stage in `prefilter_variance` the denoiser removed
/// between 61% and 91% of them depending on the sample count, which sounds
/// respectable and looks terrible -- what is left is a scatter of single pixels
/// tens of times brighter than the surface they sit on, and the eye finds every
/// one. It also got *worse* with more samples in the sense that mattered: 1 spp
/// at strength 5 was the one clean configuration in the whole grid, because it
/// is the only one where `denoise_resolve.wgsl` takes the filtered result whole
/// instead of blending a fraction of the raw outlier back in.
///
/// So the assertion is a proportion rather than an absolute count -- the scene
/// produces a different number of outliers at every sample count and there is
/// no point pinning that -- with a small absolute allowance so a configuration
/// that produces almost none cannot fail on integer noise.
#[test]
fn test_denoise_removes_fireflies_at_every_sample_count() {
    let (device, queue) = get_wgpu_device_and_queue();
    let (width, height) = (400, 400);

    let count = |spp, strength| {
        let pixels = render_linear(
            cornell_denoise_scene_at(spp, strength, width, height),
            device,
            queue,
        );
        // 20 of 255 above every neighbour. Below about 10 the count is
        // dominated by ordinary grain, which is not what this is measuring;
        // above about 40 only the very worst pixels are counted and the
        // denoiser scored well there even before any of this work.
        fireflies(&pixels, width, height, 20.)
    };

    for spp in [1, 2, 5, 10, 16] {
        let noisy = count(spp, None);

        // Both ends of the documented strength range. That strength barely
        // moved the firefly count was half the original symptom: the stage that
        // decided their fate read no sigma at all, so there was nothing for the
        // knob to do.
        for strength in [1., 5.] {
            let denoised = count(spp, Some(strength));
            println!(
                "{:>2} spp, strength {}: {:>5} fireflies -> {:>4}",
                spp, strength, noisy, denoised
            );

            let allowed = (noisy as f64 * 0.08) as usize + 4;
            assert!(
                denoised <= allowed,
                "denoising {} spp at strength {} left {} fireflies of the {} the \
                 raw render had, and at most {} is allowed",
                spp,
                strength,
                denoised,
                noisy,
                allowed
            );
        }
    }
}

/// The positive control for the test above, and the risk it introduces.
///
/// Clamping a pixel against its neighbours is a good way to remove fireflies
/// and an excellent way to remove a caustic, which is the same shape --
/// legitimately bright, high variance, and sitting on a surface whose guide
/// matches its neighbours'. The specular scene's glass sphere puts one directly
/// under it.
///
/// The 99.9th percentile rather than the maximum: the maximum is a light source
/// seen directly, which the guide protects trivially and which would pass this
/// however wrong the filter was.
#[test]
fn test_denoise_preserves_bright_detail_when_converged() {
    let (device, queue) = get_wgpu_device_and_queue();

    let percentile_999 = |pixels: &[[f32; 4]]| {
        let mut lums: Vec<f64> = pixels
            .iter()
            .map(|p| (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]) as f64)
            .collect();
        lums.sort_by(|a, b| a.partial_cmp(b).unwrap());
        lums[lums.len() * 999 / 1000]
    };

    for (name, plain, denoised) in [
        (
            "specular",
            specular_denoise_scene(2000, None),
            specular_denoise_scene(2000, Some(1.)),
        ),
        (
            "cornell",
            cornell_denoise_scene(2000, None),
            cornell_denoise_scene(2000, Some(1.)),
        ),
    ] {
        let before = percentile_999(&render_linear(plain, device, queue));
        let after = percentile_999(&render_linear(denoised, device, queue));

        println!(
            "{}: 99.9th percentile luminance {} -> {}",
            name, before, after
        );
        assert!(
            after > before * 0.97,
            "denoising a converged {} scene dimmed its brightest detail, 99.9th \
             percentile went {} -> {}",
            name,
            before,
            after
        );
    }
}

/// Saves the same 1/2/5/10 spp by strength 1/5 grid as `denoise_examples/`,
/// through the same display transform the renderer uses, so the change can be
/// looked at rather than only measured.
/// `cargo test cornell_visual_grid -- --ignored`
#[test]
#[ignore]
fn cornell_visual_grid() {
    let (device, queue) = get_wgpu_device_and_queue();
    let tag = std::env::var("VISUAL_TAG").unwrap_or_else(|_| "actual".into());
    let (width, height) = (500, 500);

    for spp in [1, 2, 5, 10] {
        for strength in [1., 5.] {
            let pixels = render_linear(
                cornell_denoise_scene_at(spp, Some(strength), width, height),
                device,
                queue,
            );
            encode(&pixels, width as u32, height as u32, ToneMapper::default())
                .save(format!(
                    "tests/output/out_actual_cornell_{}_{}spp_str{}.png",
                    tag, spp, strength as u32
                ))
                .unwrap();
        }
    }
}

/// Diagnostic, not a gate: firefly counts across the sample-count and strength
/// grid, reference-free so it needs no converged render. This is how
/// `despeckle_k`, `despeckle_floor` and `despeckle_min_weight` in
/// `denoise_atrous.wgsl` were chosen, and re-running it is how to re-choose
/// them. `cargo test cornell_firefly_sweep -- --ignored --nocapture`
#[test]
#[ignore]
fn cornell_firefly_sweep() {
    let (device, queue) = get_wgpu_device_and_queue();
    let (width, height) = (500, 500);

    // Rendered once per configuration and counted at every threshold, because
    // where the cut is drawn decides how many marginal pixels are counted and
    // the conclusion should not depend on it.
    let thresholds = [10., 20., 40., 60.];
    let counts = |spp, strength| {
        let pixels = render_linear(
            cornell_denoise_scene_at(spp, strength, width, height),
            device,
            queue,
        );
        thresholds
            .iter()
            .map(|&e| fireflies(&pixels, width, height, e))
            .collect::<Vec<_>>()
    };

    for spp in [1, 2, 5, 10, 16, 64] {
        let rows = [
            ("raw", counts(spp, None)),
            ("strength 1", counts(spp, Some(1.))),
            ("strength 5", counts(spp, Some(5.))),
        ];
        println!("--- {} spp, {}x{} ---", spp, width, height);
        print!("{:>12}", "excess >");
        for e in thresholds {
            print!("{:>8}", e as u32);
        }
        println!();
        for (name, row) in rows {
            print!("{:>12}", name);
            for c in row {
                print!("{:>8}", c);
            }
            println!();
        }
    }
}

/// A strength of 0 is the bottom of the documented range, where `sigma_colour`
/// collapses the luminance tolerance to 1e-8 and every non-centre tap goes to
/// zero weight. That is as close to the identity as this filter gets.
///
/// Outlier rejection does not read `sigma_colour`, and deliberately so: that
/// their fate was decided by a stage reading no sigma at all is exactly why
/// `strength` used to make so little difference to fireflies. So it has to be
/// switched off explicitly at zero, or "off" would quietly stop meaning off,
/// and this is the test that says so. Measured, the bound below has a hundred-
/// fold margin when the switch is wired up and fails by three hundredfold when
/// it is not.
#[test]
fn test_denoise_strength_zero_is_the_identity() {
    let (device, queue) = get_wgpu_device_and_queue();

    // A sample count low enough that the despeckle would certainly fire if it
    // were running at all: at 2 spp it clamps a few hundred pixels of this
    // scene.
    let plain = render_linear(cornell_denoise_scene(2, None), device, queue);
    let denoised = render_linear(cornell_denoise_scene(2, Some(0.)), device, queue);

    let mean: f64 = plain
        .iter()
        .map(|p| (p[0] + p[1] + p[2]) as f64 / 3.0)
        .sum::<f64>()
        / plain.len() as f64;
    let relative = linear_rmse(&denoised, &plain) / mean;

    println!(
        "relative linear RMSE of the denoiser at strength 0: {}",
        relative
    );

    // Not bit-exact, and never was: a tap whose luminance matches the centre's
    // exactly still passes the collapsed tolerance, and the resolve pass still
    // mixes. What the bound is here for is the despeckle -- left wired up at
    // strength 0 it clamps a few hundred pixels of this scene and this number
    // goes up by two orders of magnitude.
    assert!(
        relative < 0.001,
        "a denoiser at strength 0 should be as close to the identity as the \
         filter can be, relative RMSE was {}",
        relative
    );
}
