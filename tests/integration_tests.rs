use std::default::Default;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::mpsc::channel;
use std::thread;

use image::Rgb;
use image::RgbImage;
use image::imageops::FilterType;
use image_compare::Algorithm::RootMeanSquared;
use solstrale::camera::CameraConfig;
use solstrale::geo::transformation::{
    NopTransformer, RotationX, RotationY, RotationZ, Transformations, Transformer,
};
use solstrale::geo::vec3::Vec3;
use solstrale::hittable::{Bvh, Hittables, Quad, Sphere, Triangle};
use solstrale::material::texture::{ImageMap, SolidColor, Textures};
use solstrale::material::{DiffuseLight, Lambertian, Metal};
use solstrale::post::{
    BloomPostProcessor, DenoisePostProcessor, PostProcessors, SaturationPostProcessor,
};
use solstrale::ray_trace;
use solstrale::renderer::{RenderConfig, Renderer, Scene};
use solstrale::util::rgb_color::linear_to_srgb;

use crate::scenes::{
    FURNACE_SPHERE_CENTER, FURNACE_SPHERE_RADIUS, ROUGH_GLASS_RADIUS, ROUGH_GLASS_ROUGHNESS,
    ROUGH_METAL_FUZZ, ROUGH_METAL_RADIUS, create_blend_material_scene, create_cornell_scene,
    create_cornell_stacked_lights_scene, create_furnace_scene, create_glass_slab_scene,
    create_light_attenuation_scene, create_many_lights_scene, create_normal_mapping_scene,
    create_normal_mapping_sphere_scene, create_obj_scene, create_obj_with_box,
    create_obj_with_triangle, create_quad_rotation_scene, create_rough_glass_scene,
    create_rough_metal_scene, create_simple_test_scene, create_smooth_vs_flat_scene,
    create_specular_scene, create_srgb_decode_scene, create_test_scene,
    create_texture_mapping_scene, create_uv_scene, create_wide_light_scene,
    rough_glass_sphere_center, rough_metal_sphere_center,
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

/// The only test that exercises the GPU half of the octahedral normal path:
/// `pack_oct_round_trip` pins the Rust encoder against a transcription of the
/// shader's decoder, and only this render puts the shader's own `oct_decode`
/// and the interpolation in `resolve_hit` in front of anything that can tell.
///
/// The threshold is 0.99 rather than the suite's usual 0.95, chosen by
/// measuring what it has to separate:
///
///     smooth, as written                  0.9964
///     smooth sphere regressed to flat     0.9703
///     pack_oct's fold polarity flipped    0.7691
///
/// At 0.95 the first two are indistinguishable and the test is decorative.
/// 0.99 leaves 0.0064 of headroom above and 0.0197 of margin below.
#[test]
fn test_render_smooth_vs_flat() {
    let render_config = RenderConfig {
        width: 200,
        height: 100,
        samples_per_pixel: 200,
        ..Default::default()
    };
    let scene = create_smooth_vs_flat_scene(render_config);

    render_and_compare_output(scene, "smooth_vs_flat", 0.99);
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

/// The roughness range as an image: five metals from mirror to nearly diffuse,
/// over a textured floor, lit by one small light. A regression net for the
/// gross appearance of the lobe -- what it cannot see is noise, which
/// `test_rough_metal_converges_with_nee` measures instead.
#[test]
fn test_scene_rough_metal() {
    let render_config = RenderConfig {
        width: 200,
        height: 100,
        samples_per_pixel: 200,
        ..Default::default()
    };
    render_and_compare_output(create_rough_metal_scene(render_config), "rough_metal", 0.95);
}

/// A scene with nothing to light it at all -- no emitter and a black
/// background -- can only render black, and is rejected. A scene with no
/// emitter but a lit background is not: that is the furnace test's
/// configuration, and it renders fine with next-event estimation simply having
/// nothing to sample.
#[test]
fn test_render_scene_without_light() {
    let (device, queue) = get_wgpu_device_and_queue();
    let render_config = RenderConfig {
        width: 20,
        height: 10,
        ..Default::default()
    };
    let mut scene = create_simple_test_scene(render_config, false);
    scene.background_color = Vec3::new(0., 0., 0.);

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
        Err(e) => assert_eq!(
            "Scene should have at least one light or a non-black background",
            e.to_string()
        ),
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

use solstrale::util::gpu_timing::{PassTiming, TIMING_ENV, drain_totals};
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
// converged, not what it converges to. This is the check LIMITATIONS.md asks for:
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

/// The MIS weights and the light PDFs, checked against the one estimator in
/// the crate that uses neither.
///
/// [`Renderer::bsdf_only_reference`] is a plain BSDF path tracer with the
/// firefly clamp lifted: no light is ever sampled or weighed, and every
/// emitter it lands on is taken at weight 1. It estimates the same integral as
/// the shipped renderer, so converged, the two have to agree. Nothing else in
/// the suite can say that -- the gamma-mapped RMS goldens are scored at 0.9 on
/// a 100x50 downsample, which a uniform brightness shift walks straight
/// through.
///
/// The six scenes are six ways a light PDF can be wrong. Cornell has no
/// background at all, so a total loss of direct lighting has nowhere to hide;
/// the stacked-lights Cornell is the one configuration where a direction
/// reaches two emitters at once; the specular scene drives the weight-1 path
/// through metal and glass; the test scene's radius-10 sphere light is crossed
/// by most of the rays in the frame; the many-lights room is where the
/// selection probability stops being the constant `1 / light_count` -- a
/// `select_pdf` that disagreed with the alias table the sampler draws from
/// would land here as a shifted mean and nowhere else.
///
/// Rough glass is the sixth, and it is here for the BSDF rather than the light.
/// It is the strongest available check on the transmission lobe, and the only
/// one that does not go through the pdf it is checking: the two arms are
/// different estimators of the same integral, so agreeing to half a percent is
/// what says the microfacet refraction Jacobian and the MIS algebra are right.
/// A pdf wrong by a constant factor shows up here and in nothing else in the
/// suite -- `test_rough_glass_converges_with_nee` would still pass, because
/// both of its arms would be wrong together.
///
/// Its light is 60x dimmer than the metal twin's for a reason that belongs to
/// this test: a bright pinpoint seen through glass is clamped in one arm and
/// not the other, which reads as a 5% disagreement that has nothing to do with
/// a pdf. `create_rough_glass_scene` records the rest.
///
/// The reference lifts the clamp because the shipped renderer's clamp is the
/// one thing here that is deliberately biased, and on a BSDF-only path it bites
/// far harder -- every emitter hit arrives as an unweighted indirect sample.
/// Clamped both ways, Cornell's two arms sit 15% apart and the gate would have
/// to be loose enough to be worthless. Unclamped on the reference side, the
/// four scenes come in at 0.07%, 0.11%, 0.00% and 0.13%. So a failure here that
/// *shrinks* when `clamping_threshold` in `ray_trace.wgsl` is raised is the
/// clamp costing the shipped renderer energy, not the MIS weights being wrong.
#[test]
fn test_bsdf_only_sampling_converges_to_the_same_image() {
    let (device, queue) = get_wgpu_device_and_queue();

    let config = || RenderConfig {
        width: 100,
        height: 60,
        samples_per_pixel: 2000,
        samples_per_batch: 100,
        // Off. A retired pixel holds whatever mean it had when it retired, and
        // the two arms retire different pixels at different times, which would
        // put a difference into the comparison that is not the estimator's.
        min_samples_per_pixel: u32::MAX,
        ..Default::default()
    };

    // At 2000 spp the sampling error in a whole-image mean is far below a tenth
    // of a percent, so this is roughly four times the widest measured gap and
    // still tight enough to catch a PDF that is wrong by a constant factor.
    const TOLERANCE: f64 = 0.005;

    let scenes: [OracleScene; 6] = [
        ("cornell", create_cornell_scene),
        (
            "cornell_stacked_lights",
            create_cornell_stacked_lights_scene,
        ),
        ("specular", create_specular_scene),
        ("test_scene", create_test_scene),
        ("many_lights", create_many_lights_scene),
        ("rough_glass", create_rough_glass_scene),
    ];

    for (name, build) in scenes {
        let shipped = converged_mean(build(config()), true, device, queue);
        let reference = converged_mean(build(config()), false, device, queue);
        let relative = (shipped - reference).abs() / reference;

        println!(
            "{}: mean linear radiance {} with NEE against {} without, {:.2}% apart",
            name,
            shipped,
            reference,
            relative * 100.
        );

        assert!(
            relative < TOLERANCE,
            "{}: next-event estimation converges to {} where BSDF sampling alone converges to {}, \
             {:.2}% apart against a {:.1}% tolerance",
            name,
            shipped,
            reference,
            relative * 100.,
            TOLERANCE * 100.
        );
    }
}

/// One scene for the comparison above: a name for the report, and the builder
/// that makes it at a given config.
type OracleScene = (&'static str, fn(RenderConfig) -> Scene);

/// As [`mean_linear_radiance`], but choosing which estimator traces the scene.
fn converged_mean(
    scene: Scene,
    shipped_estimator: bool,
    device: &'static wgpu::Device,
    queue: &'static wgpu::Queue,
) -> f64 {
    let (progress_sender, progress_receiver) = channel();
    let (_camera_sender, camera_receiver) = channel();
    let (_abort_sender, abort_receiver) = channel();

    let pixels = (scene.render_config.width * scene.render_config.height) as u32;
    let mut renderer = if shipped_estimator {
        Renderer::new(scene, device, queue)
    } else {
        Renderer::bsdf_only_reference(scene, device, queue)
    }
    .unwrap();
    renderer
        .render(&progress_sender, &camera_receiver, &abort_receiver, false)
        .unwrap();
    // The channel is unbounded, so the render above ran to completion without
    // anyone draining it; dropping the sender is what ends the iteration below.
    drop(progress_sender);

    let output_buffer = progress_receiver
        .into_iter()
        .last()
        .expect("render reported no progress")
        .output_buffer;

    mean_of_buffer(&output_buffer, pixels, device, queue)
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

    mean_of_buffer(&output_buffer.unwrap(), width * height, device, queue)
}

/// Mean of the three colour channels over every pixel of a linear output
/// buffer, which is the whole image's radiance in one number.
fn mean_of_buffer(
    output_buffer: &wgpu::Buffer,
    pixels: u32,
    device: &'static wgpu::Device,
    queue: &'static wgpu::Queue,
) -> f64 {
    let size = (pixels * 16) as u64;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(output_buffer, 0, &staging_buffer, 0, size);
    queue.submit(Some(encoder.finish()));

    let result: Vec<[f32; 4]> = get_result_from_buffer(device, &staging_buffer);
    let sum: f64 = result
        .iter()
        .map(|p| (p[0] + p[1] + p[2]) as f64 / 3.0)
        .sum();
    sum / result.len() as f64
}

/// Renders a scene and returns its raw linear pixels, before gamma.
/// Marks a scene as the converged reference an error metric is measured
/// against, by moving it off the seed every other render uses.
///
/// Without this the reference shares its sample-stream prefix with the render
/// being measured, so the error between them is correlated and biased low. With
/// white noise that was about 0.2% at 8 spp against 4000 and ignorable; with
/// the low-discrepancy sampler it is structural, because an 8-sample render is
/// literally a sub-net of the 4000-sample reference.
fn as_reference(mut scene: Scene) -> Scene {
    scene.render_config.seed = 1;
    scene
}

/// Picks the sampler backing for one arm of a comparison.
fn with_sampler(mut scene: Scene, low_discrepancy: bool) -> Scene {
    scene.render_config.low_discrepancy = low_discrepancy;
    scene
}

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

/// How much grain a viewer would actually see, in code values of the 0-255
/// scale the image is written on.
///
/// [`grain`]'s sibling, and it exists because `grain` cannot see the thing this
/// change is about. That one is an RMS in linear radiance, normalised by the
/// mean; this one measures through the display transform, for the reason spelt
/// out on [`fireflies`], and reports an absolute number of code values so it
/// can be compared against what the eye resolves -- about one.
///
/// The median rather than an RMS, scaled by 1.4826 so it reads as a standard
/// deviation on the Gaussian the flat regions are. Every high-pass of a render
/// has object silhouettes in it, and those are real detail, arbitrarily large,
/// and present in exactly the same places whether the image is denoised or not.
/// They put a floor under any RMS that is independent of the noise -- which is
/// precisely the floor that would hide the improvement being measured. The
/// median is owned by the flat majority of the frame, which is where grain
/// lives and where the complaint came from.
#[allow(dead_code)]
fn displayed_grain(pixels: &[[f32; 4]], width: usize, height: usize) -> f64 {
    let displayed: Vec<f64> = pixels
        .iter()
        .map(|p| {
            let m = ToneMapper::default().map([p[0], p[1], p[2]]);
            let encode = |v: f32| (linear_to_srgb(v).min(0.999) * 256.) as f64;
            0.2126 * encode(m[0]) + 0.7152 * encode(m[1]) + 0.0722 * encode(m[2])
        })
        .collect();

    let mut departures = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let mut local = 0.0;
            let mut taps = 0.0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;
                    let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as usize;
                    local += displayed[ny * width + nx];
                    taps += 1.0;
                }
            }
            departures.push((displayed[y * width + x] - local / taps).abs());
        }
    }

    departures.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // 1.4826 is the reciprocal of the standard normal's interquartile
    // half-width, so the result is on the same scale a standard deviation
    // would be and can be read directly as "code values of grain".
    1.4826 * departures[departures.len() / 2]
}

/// RMS difference between two images in code values, after the display
/// transform -- [`linear_rmse`]'s counterpart in the units a viewer sees.
///
/// An RMS here rather than a median, unlike [`displayed_difference`]: this one
/// is measured against a converged reference, so the large departures it is
/// dominated by are error rather than the denoiser doing its job.
#[allow(dead_code)]
fn displayed_rmse(a: &[[f32; 4]], b: &[[f32; 4]]) -> f64 {
    let encode = |p: &[f32; 4]| {
        let m = ToneMapper::default().map([p[0], p[1], p[2]]);
        m.map(|v| (linear_to_srgb(v).min(0.999) * 256.) as f64)
    };

    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(p, q)| {
            let (p, q) = (encode(p), encode(q));
            (0..3).map(|c| (p[c] - q[c]).powi(2)).sum::<f64>()
        })
        .sum();
    (sum / (a.len() * 3) as f64).sqrt()
}

/// The median per-pixel, per-channel difference between two images, in code
/// values, after the display transform.
///
/// Median for the same reason [`displayed_grain`] uses one: an RMS is owned by
/// the few percent of pixels that are genuinely noisy, where a denoiser both
/// does and should change a lot, and a bound loose enough to accommodate those
/// says nothing about the rest of the frame.
#[allow(dead_code)]
fn displayed_difference(a: &[[f32; 4]], b: &[[f32; 4]]) -> f64 {
    let encode = |p: &[f32; 4]| {
        let m = ToneMapper::default().map([p[0], p[1], p[2]]);
        m.map(|v| (linear_to_srgb(v).min(0.999) * 256.) as f64)
    };

    let mut diffs: Vec<f64> = a
        .iter()
        .zip(b.iter())
        .map(|(p, q)| {
            let (p, q) = (encode(p), encode(q));
            (0..3).map(|c| (p[c] - q[c]).abs()).fold(0.0, f64::max)
        })
        .collect();

    diffs.sort_by(|x, y| x.partial_cmp(y).unwrap());
    diffs[diffs.len() / 2]
}

/// How many pixels read as an isolated bright speck: a count of those whose
/// displayed luminance exceeds the brightest of their four neighbours by more
/// than `excess`, on the 0-255 scale the image is finally written on.
///
/// Measured *through the display transform*, and that is the whole point. The
/// first version of this counted in linear radiance, and it lied: it scored a
/// change that removed 100% of the outliers it could see, while the rendered
/// image looked all but unchanged. ACES plus the sRGB encode compresses highlights
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
            let encode = |v: f32| (linear_to_srgb(v).min(0.999) * 256.) as f64;
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

/// [`linear_rmse`] with the worst 0.1% of pixels dropped.
///
/// RMSE is a mean of squares, so a single firefly dominates it: one pixel off
/// by 40 on a 200x200 image is 0.16 of RMSE on its own, which is the whole
/// figure at any sample count worth measuring. That makes the plain number the
/// right one for "how wrong is this image" and the wrong one for "how well is
/// the estimator converging on the smooth part". Read the pair. On the Cornell
/// box the two disagree by a factor of four, because white noise's error there
/// is mostly a handful of fireflies.
fn trimmed_linear_rmse(a: &[[f32; 4]], b: &[[f32; 4]]) -> f64 {
    let mut squared: Vec<f64> = a
        .iter()
        .zip(b.iter())
        .map(|(p, q)| (0..3).map(|c| ((p[c] - q[c]) as f64).powi(2)).sum::<f64>())
        .collect();
    squared.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let keep = squared.len() * 999 / 1000;
    (squared[..keep].iter().sum::<f64>() / (keep * 3) as f64).sqrt()
}

/// [`linear_rmse`] over a rectangle of the frame, as `(x0, y0, x1, y1)` with
/// the upper bounds exclusive and row 0 at the top.
///
/// A whole-image figure is the wrong instrument for a change whose effect is
/// concentrated: the win from better light sampling lives on the surfaces near
/// the light, and averaging it over the rest of the frame divides it by the
/// fraction of pixels that saw it.
fn cropped_linear_rmse(
    a: &[[f32; 4]],
    b: &[[f32; 4]],
    width: usize,
    crop: (usize, usize, usize, usize),
) -> f64 {
    let (x0, y0, x1, y1) = crop;
    let mut sum = 0.0;
    for y in y0..y1 {
        for x in x0..x1 {
            let (p, q) = (a[y * width + x], b[y * width + x]);
            for c in 0..3 {
                let d = (p[c] - q[c]) as f64;
                sum += d * d;
            }
        }
    }
    (sum / ((x1 - x0) * (y1 - y0) * 3) as f64).sqrt()
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

    let reference = render_linear(as_reference(denoise_scene(2000, None, true)), device, queue);
    let noisy = render_linear(denoise_scene(8, None, true), device, queue);
    let denoised = render_linear(denoise_scene(8, Some(1.), true), device, queue);

    let rmse_noisy = linear_rmse(&noisy, &reference);
    let rmse_denoised = linear_rmse(&denoised, &reference);

    println!("linear RMSE against 2000 spp reference:");
    println!("  8 spp, no denoiser: {}", rmse_noisy);
    println!("  8 spp, denoised:    {}", rmse_denoised);
    println!("  ratio:              {}", rmse_denoised / rmse_noisy);

    // Two gates, because the ratio alone is misleading here.
    //
    // The ratio was 0.56 against white noise, 0.62 against the low-discrepancy
    // sampler, 0.71 once image textures were sRGB decoded, and is 0.82 now that
    // light selection is power-proportional. Every loosening is the same
    // effect: the denominator falls faster than the numerator. The sampler
    // removed noise before the filter could; the decode darkened this scene's
    // textured floor, which is the easy, flat majority of the frame, leaving
    // the RMSE dominated by the untextured reds and the glass the filter was
    // always going to struggle with; and power-proportional selection stopped
    // this scene's three very differently powered lights being sampled equally
    // often, which took 15% off the 8 spp input (0.232 -> 0.199) against 2% off
    // the denoised image (0.166 -> 0.162).
    //
    // The absolute assertion is what stops each such change having to make that
    // argument again: in all three the thing anyone actually looks at improved
    // while the ratio got worse.
    assert!(
        rmse_denoised < rmse_noisy * 0.85,
        "denoising 8 spp should cut linear RMSE against the reference by at least 15%, \
         was {} against {}",
        rmse_denoised,
        rmse_noisy
    );
    // 0.162 measured.
    assert!(
        rmse_denoised < 0.2,
        "denoised 8 spp should land within 0.2 linear RMSE of the reference, was {}",
        rmse_denoised
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

    let reference = render_linear(
        as_reference(specular_denoise_scene(2000, None)),
        device,
        queue,
    );
    let noisy = render_linear(specular_denoise_scene(8, None), device, queue);
    let denoised = render_linear(specular_denoise_scene(8, Some(1.)), device, queue);

    let rmse_noisy = linear_rmse(&noisy, &reference);
    let rmse_denoised = linear_rmse(&denoised, &reference);

    println!("specular scene, linear RMSE against 2000 spp reference:");
    println!("  8 spp, no denoiser: {}", rmse_noisy);
    println!("  8 spp, denoised:    {}", rmse_denoised);
    println!("  ratio:              {}", rmse_denoised / rmse_noisy);

    // 0.84 measured, from 0.68 against white noise and 0.78 before image
    // textures were sRGB decoded, and re-baselined each time for the same
    // reason as the diffuse gate above: the denominator falls faster than the
    // numerator. 0.115 -> 0.083 -> 0.063 against 0.083 -> 0.065 -> 0.053. The
    // absolute figure is the one that improved, so it is asserted too.
    //
    // The margin is wider than the diffuse scene's because the 2000 spp
    // reference is itself adaptively sampled and moves a little run to run.
    assert!(
        rmse_denoised < rmse_noisy * 0.88,
        "denoising 8 spp of a specular scene should cut linear RMSE against the \
         reference by at least 12%, was {} against {}",
        rmse_denoised,
        rmse_noisy
    );
    // 0.053 measured.
    assert!(
        rmse_denoised < 0.063,
        "denoised 8 spp of a specular scene should land within 0.063 linear RMSE of \
         the reference, was {}",
        rmse_denoised
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

/// The gate this change exists for: spending more samples has to buy a
/// visibly smoother denoised image, and it did not.
///
/// The old resolve fade set `blend` to the pixel's relative standard error over
/// a `full_strength_error` of 0.4, so the residual it left was
/// `sigma * (1 - sigma / (0.4 * L))` -- a downward parabola in sigma, peaking
/// at `sigma = 0.2 L`. Any two noise levels symmetric about that peak leave
/// *identical* grain, and a Cornell wall at 10 spp (sigma ~ 0.30 L) and at 100
/// spp (sigma ~ 0.095 L) are almost exactly that pair. The fade handed noise
/// back at the same rate the sampler removed it, so the two images were
/// indistinguishable. Measured on a 1303x964 Cornell render, ten times the
/// samples bought 19% less grain.
///
/// Measured through the display transform, in code values, because the old
/// criterion's defect was precisely that it was not perceptual: 0.4 relative
/// linear error is 13 to 30 code values of visible grain depending on
/// brightness, and the eye sees grain at about one.
///
/// Three assertions, and all three fail on the code this replaced, where the
/// denoised sequence ran 3.009, 3.566, 3.512, 2.632 -- rising at first, flat
/// through the middle, and ending at 0.84 of the raw render.
///
/// What is deliberately *not* asserted is a per-step ratio near the 0.5 that
/// 1/sqrt(n) would suggest, because this metric has a floor that has nothing to
/// do with noise. A 4000 spp render of this scene measures 0.628 code values of
/// "grain" that are its silhouettes and shading gradients -- real detail, which
/// no amount of filtering should remove. The denoised figures below are within a
/// factor of two of that floor, so the achievable per-step ratio is well above
/// 0.5 and rises as the floor is approached. `denoise_display_sweep` prints the
/// floor alongside the sweep for exactly this reason, and 128 is where this
/// stops because beyond it the ratio is measuring the scene rather than the
/// filter.
#[test]
fn test_denoised_grain_keeps_falling_with_samples() {
    let (device, queue) = get_wgpu_device_and_queue();
    let (width, height) = (200, 200);

    let grain_at = |spp, strength| {
        let pixels = render_linear(cornell_denoise_scene(spp, strength), device, queue);
        displayed_grain(&pixels, width, height)
    };

    let counts = [2u32, 8, 32, 128];
    let mut raw = Vec::new();
    let mut denoised = Vec::new();
    for spp in counts {
        raw.push(grain_at(spp, None));
        denoised.push(grain_at(spp, Some(1.)));
    }

    println!("displayed grain, code values:");
    for (i, spp) in counts.iter().enumerate() {
        println!(
            "  {:>4} spp: raw {:.3}, denoised {:.3} ({:.2} of raw)",
            spp,
            raw[i],
            denoised[i],
            denoised[i] / raw[i]
        );
    }

    // The filter has to be doing most of the work at every sample count, not
    // just where the image is obviously broken. 0.13 to 0.34 measured.
    for (i, spp) in counts.iter().enumerate() {
        assert!(
            denoised[i] < raw[i] * 0.4,
            "at {} spp the denoiser left {:.3} code values of grain against the raw render's {:.3}",
            spp,
            denoised[i],
            raw[i]
        );
    }

    // Monotone. This is the assertion the complaint was about, and the one the
    // old fade broke outright: it made 8 spp grainier than 2.
    for i in 1..counts.len() {
        assert!(
            denoised[i] < denoised[i - 1],
            "spending {} samples instead of {} made the denoised image grainier, {:.3} against {:.3}",
            counts[i],
            counts[i - 1],
            denoised[i],
            denoised[i - 1]
        );
    }

    // And falling by enough to be worth the samples, across the range as a
    // whole rather than step by step. 0.52 measured against 0.88 before.
    assert!(
        *denoised.last().unwrap() < denoised[0] * 0.6,
        "over the whole range denoised grain only went {:.3} -> {:.3}",
        denoised[0],
        denoised.last().unwrap()
    );
}

/// The filter has to fade itself out as the render converges, because its
/// luminance tolerance is set by the variance of the pixel mean and that falls
/// as 1/n. Without this the filter would keep a fixed blur floor that a
/// single-spp golden test cannot see.
///
/// This used to bound the relative linear RMSE between the denoised and the
/// plain 2000 spp render at 0.02, and measured 0.0045. It now measures 0.023 and
/// that is correct rather than a regression: the test scene at 2000 spp with
/// adaptive sampling off still carries about two code values of grain, which is
/// visible, and the fade is now built to remove visible grain rather than to
/// retire on a relative error. The old bound *was* the premise being replaced --
/// "2000 samples means converged, so do nothing" is the same linear-relative
/// notion of convergence that let 13 to 30 code values of grain through at lower
/// sample counts.
///
/// So the property is restated in the units the filter now works in, and it is
/// two properties rather than one, because "near identity" was doing both jobs:
///
/// 1. The image a viewer sees must be all but unchanged -- half the frame moving
///    by less than one code value is what that should always have meant.
/// 2. The fade must actually keep fading. This is the part the old bound really
///    protected, and the part a fixed blur floor would violate: the criterion is
///    quadratic in sigma while the blend is unsaturated and linear once it
///    saturates, so four times the samples has to cut the change by at least
///    two. A filter that had stopped fading would hold it flat.
///
/// Measured with the median rather than an RMS, for the reason on
/// [`displayed_difference`]: the few percent of pixels that are genuinely still
/// noisy are pixels the denoiser both does and should change a lot, and a bound
/// loose enough to admit them says nothing about the rest of the frame.
#[test]
fn test_denoise_is_near_identity_at_high_samples() {
    let (device, queue) = get_wgpu_device_and_queue();

    // Adaptive sampling off, so every pixel really has the samples asked for.
    // With it on, pixels retire at the variance threshold and the image is not
    // converged in the sense this test is about.
    let plain = render_linear(denoise_scene(2000, None, false), device, queue);
    let denoised = render_linear(denoise_scene(2000, Some(1.), false), device, queue);
    let plain_500 = render_linear(denoise_scene(500, None, false), device, queue);
    let denoised_500 = render_linear(denoise_scene(500, Some(1.), false), device, queue);

    let change = displayed_difference(&denoised, &plain);
    let change_500 = displayed_difference(&denoised_500, &plain_500);

    // Kept and printed rather than asserted, as `test_adaptive_sampling_convergence`
    // does, so a before-and-after across builds is still possible in the old
    // units.
    let mean: f64 = plain
        .iter()
        .map(|p| (p[0] + p[1] + p[2]) as f64 / 3.0)
        .sum::<f64>()
        / plain.len() as f64;
    println!(
        "relative linear RMSE of the denoiser at 2000 spp: {} (unasserted)",
        linear_rmse(&denoised, &plain) / mean
    );
    println!(
        "median displayed change:  500 spp {:.3} cv, 2000 spp {:.3} cv, ratio {:.2}",
        change_500,
        change,
        change / change_500
    );

    assert!(
        change < 1.0,
        "at 2000 spp the denoiser should be close to the identity, but half the \
         image moved by {:.3} code values or more",
        change
    );

    assert!(
        change < change_500 * 0.5,
        "the fade has stopped fading: four times the samples took the denoiser's \
         visible change only from {:.3} to {:.3} code values",
        change_500,
        change
    );
}

/// Diagnostic, not a gate: prints linear RMSE against a converged reference for
/// a range of strengths at a range of sample counts. This is how the defaults in
/// `DenoisePostProcessor::initialize` were chosen, and re-running it is how to
/// re-choose them. `cargo test denoise_strength_sweep -- --ignored --nocapture`
///
/// A `grain` column beside each RMSE, because RMSE alone cannot see the failure
/// this is most often run to rule out. Against a converged reference it rewards
/// blur at every sample count where noise still dominates, so a filter that is
/// flattening real detail scores *better* until the noise runs out. The pair is
/// what has to be read: grain falling while RMSE rises is over-blur.
///
/// The strength range runs the full documented 0 to 10 rather than stopping at
/// 2, because `strength` now scales the resolve fade as well as `sigma_colour`
/// and the top of the range is no longer a mild variation on the middle.
#[test]
#[ignore]
fn denoise_strength_sweep() {
    let (device, queue) = get_wgpu_device_and_queue();
    let reference = render_linear(
        as_reference(denoise_scene(4000, None, false)),
        device,
        queue,
    );

    for adaptive in [false, true] {
        for spp in [8u32, 64, 2000] {
            let plain = render_linear(denoise_scene(spp, None, adaptive), device, queue);
            let base = linear_rmse(&plain, &reference);
            let mut line = format!(
                "adaptive={} spp={:<5} none={:.4}/{:.3}",
                adaptive,
                spp,
                base,
                grain(&plain, 200, 100)
            );
            for s in [0., 0.25, 0.5, 1., 2., 5., 10.] {
                let d = render_linear(denoise_scene(spp, Some(s), adaptive), device, queue);
                let rmse = linear_rmse(&d, &reference);
                line += &format!(
                    "  s={}: {:.4} ({:.2})/{:.3}",
                    s,
                    rmse,
                    rmse / base,
                    grain(&d, 200, 100)
                );
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
/// `cargo test specular_denoise_sweep -- --ignored --nocapture`
#[test]
#[ignore]
fn specular_denoise_sweep() {
    let (device, queue) = get_wgpu_device_and_queue();
    let reference = render_linear(
        as_reference(specular_denoise_scene(4000, None)),
        device,
        queue,
    );

    for spp in [8u32, 32, 64, 200, 800] {
        let plain = render_linear(specular_denoise_scene(spp, None), device, queue);
        let base = linear_rmse(&plain, &reference);
        let mut line = format!(
            "spp={:<5} none={:.4}/{:.3}",
            spp,
            base,
            grain(&plain, 200, 100)
        );
        for s in [0., 0.25, 0.5, 1., 2., 5., 10.] {
            let d = render_linear(specular_denoise_scene(spp, Some(s)), device, queue);
            let rmse = linear_rmse(&d, &reference);
            line += &format!(
                "  s={}: {:.4} ({:.3})/{:.3}",
                s,
                rmse,
                rmse / base,
                grain(&d, 200, 100)
            );
        }
        println!("{}", line);
    }
}

/// Diagnostic, not a gate: saves a noisy/denoised pair for visual inspection,
/// which is how the golden images above were vetted before being promoted.
/// The specular pair is where the guide's specular chain shows: compare the
/// mirror and the glass sphere against a build with `GUIDE_MAX_SPECULAR` set
/// to 0 in ray_trace.wgsl, which is the old primary-hit guide.
/// `cargo test denoise_visual_pair -- --ignored`
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
        let c = |v: f32| (linear_to_srgb(v).min(0.999) * 256.) as u8;
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
/// `cargo test tone_map_visual_comparison -- --ignored`
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

/// Saves a sample-count by strength grid through the same display transform the
/// renderer uses, so the change can be looked at rather than only measured.
/// `cargo test cornell_visual_grid -- --ignored`
///
/// Strength 10 is in the grid because it is the end of the documented range and
/// it is where the fade saturates -- `FULL_STRENGTH_GRAIN / 10` is 0.2 code
/// values, below the quantisation step of the image, so the blend is 1 almost
/// everywhere and what lands on disk is the a-trous filter's own output with
/// nothing held back. That is the case a number cannot settle.
#[test]
#[ignore]
fn cornell_visual_grid() {
    let (device, queue) = get_wgpu_device_and_queue();
    let tag = std::env::var("VISUAL_TAG").unwrap_or_else(|_| "actual".into());
    let (width, height) = (500, 500);

    for spp in [1, 2, 5, 10, 100] {
        for strength in [1., 5., 10.] {
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

/// Diagnostic, not a gate: what the low-discrepancy sampler buys, as linear
/// RMSE against a converged reference, on all three scenes across sample
/// counts. This is the measurement the sampler was landed on, and re-running it
/// is how to re-judge it.
/// `cargo test sampler_convergence_sweep -- --ignored --nocapture`
///
/// Adaptive sampling is off in every arm. With it on, the two arms spend
/// different numbers of samples -- low-discrepancy samples are negatively
/// correlated, so Welford's marginal variance overstates the variance of the
/// mean and retires each pixel later -- and the comparison stops meaning
/// anything.
///
/// RMSE rather than `grain`, deliberately. `grain` is reference-free and a 3x3
/// high-pass, so it cannot tell "less error" from "error moved to a different
/// spatial frequency", and it would under-report a technique whose whole effect
/// is on the smooth part of the integrand. RMSE's usual objection -- that it
/// rewards blur at a low sample count -- does not apply, because nothing here
/// blurs.
///
/// Every reference is rendered at a different seed from the arms measured
/// against it; see [`as_reference`] for why that stops mattering the moment the
/// sampler is low-discrepancy.
#[test]
#[ignore]
fn sampler_convergence_sweep() {
    let (device, queue) = get_wgpu_device_and_queue();

    /// A scene at a sample count, with a sampler picked.
    type Arm = fn(u32, bool) -> Scene;

    let scenes: [(&str, Arm); 3] = [
        ("test_scene", |spp, ld| {
            with_sampler(denoise_scene(spp, None, false), ld)
        }),
        ("specular", |spp, ld| {
            let mut scene = specular_denoise_scene(spp, None);
            scene.render_config.min_samples_per_pixel = u32::MAX;
            with_sampler(scene, ld)
        }),
        ("cornell", |spp, ld| {
            with_sampler(cornell_denoise_scene(spp, None), ld)
        }),
    ];

    for (name, scene_of) in scenes {
        let reference = render_linear(as_reference(scene_of(4000, true)), device, queue);
        for spp in [8u32, 32, 64, 256] {
            let white = render_linear(scene_of(spp, false), device, queue);
            let sobol = render_linear(scene_of(spp, true), device, queue);
            let (w, wt) = (
                linear_rmse(&white, &reference),
                trimmed_linear_rmse(&white, &reference),
            );
            let (s, st) = (
                linear_rmse(&sobol, &reference),
                trimmed_linear_rmse(&sobol, &reference),
            );
            println!(
                "{:<10} spp={:<4} rmse {:.5} -> {:.5} ({:+.0}%)   trimmed {:.5} -> {:.5} ({:+.0}%)",
                name,
                spp,
                w,
                s,
                (s / w - 1.) * 100.,
                wt,
                st,
                (st / wt - 1.) * 100.
            );
        }
    }
}

/// Diagnostic, not a gate: how noisy the direct-lighting estimator is, as
/// linear RMSE against a converged reference, whole-frame and on the crop
/// where a change to it can actually show.
/// `cargo test light_sampling_sweep -- --ignored --nocapture`
///
/// The crop column is the point of this sweep. A light sampling change moves
/// the surfaces near the light and nothing else, so a whole-image figure
/// divides its effect by the fraction of the frame that saw it -- and `grain`
/// would be worse still, being a whole-image RMS over a whole-image mean. On
/// Cornell the emitter is in the ceiling, so the crop is the upper third: the
/// ceiling itself, the top of the tall box and the upper walls.
///
/// The test scene's crop is where its triangle light sits, which is the only
/// triangle emitter in the suite and therefore the only place the spherical
/// triangle question (issue #49) could be answered from. `wide_light`'s is its
/// walls, which run up to the emitter's own plane and are where area sampling's
/// `d^2 / cos` has no bound at all.
///
/// Adaptive sampling is off in every arm, so every pixel really has the sample
/// count in the column and the comparison across builds means what it says. References are rendered at a different seed; see [`as_reference`].
#[test]
#[ignore]
fn light_sampling_sweep() {
    let (device, queue) = get_wgpu_device_and_queue();

    // Name, builder, dimensions, and the crop with its label.
    type Arm = (
        &'static str,
        fn(u32) -> Scene,
        usize,
        usize,
        &'static str,
        (usize, usize, usize, usize),
    );
    let arms: [Arm; 3] = [
        (
            "cornell",
            |spp| cornell_denoise_scene(spp, None),
            200,
            200,
            "upper third",
            (0, 0, 200, 66),
        ),
        (
            "test_scene",
            |spp| denoise_scene(spp, None, false),
            200,
            100,
            "triangle",
            (0, 4, 48, 50),
        ),
        (
            "wide_light",
            |spp| {
                create_wide_light_scene(RenderConfig {
                    width: 200,
                    height: 100,
                    samples_per_pixel: spp,
                    min_samples_per_pixel: u32::MAX,
                    ..Default::default()
                })
            },
            200,
            100,
            "walls",
            (0, 18, 200, 50),
        ),
    ];

    for (name, build, width, _height, crop_name, crop) in arms {
        let reference = render_linear(as_reference(build(4000)), device, queue);
        for spp in [8u32, 32, 128, 512] {
            let image = render_linear(build(spp), device, queue);
            println!(
                "{:<10} spp={:<4} rmse {:.5}   {} {:.5}",
                name,
                spp,
                linear_rmse(&image, &reference),
                crop_name,
                cropped_linear_rmse(&image, &reference, width, crop),
            );
        }
    }
}

/// Diagnostic, not a gate: what the denoiser leaves on screen, across the
/// sample-count and strength grid. This is how `FULL_STRENGTH_GRAIN` in
/// `post/denoise.rs` was chosen, and re-running it is how to re-choose it.
/// `cargo test denoise_display_sweep -- --ignored --nocapture`
///
/// Two columns, because neither is enough on its own. Grain alone rewards blur,
/// and so does reference RMSE at a low sample count -- the two only disagree
/// where over-blur lives, which is the whole question this has to answer. Read
/// them as a pair: grain falling while RMSE rises is the filter flattening real
/// detail.
///
/// `detail` is the raw render's grain at the reference sample count. It is the
/// floor under the denoised column -- object silhouettes and shading gradients
/// that no amount of filtering should remove -- so a denoised figure near it
/// means the filter has run out of noise to find, not that it has stopped
/// working.
///
/// Note that `strength` scales the fade and `sigma_colour` together, so a row of
/// this sweep moves both. To move the fade's threshold alone, edit
/// `FULL_STRENGTH_GRAIN`.
#[test]
#[ignore]
fn denoise_display_sweep() {
    let (device, queue) = get_wgpu_device_and_queue();
    let (width, height) = (200, 200);

    let reference = render_linear(
        as_reference(cornell_denoise_scene(4000, None)),
        device,
        queue,
    );
    println!(
        "detail floor (4000 spp, no denoiser): {:.3} cv",
        displayed_grain(&reference, width, height)
    );

    for spp in [2u32, 8, 32, 128, 512] {
        let plain = render_linear(cornell_denoise_scene(spp, None), device, queue);
        let mut line = format!(
            "spp={:<4} raw: grain {:6.3}  rmse {:6.3}",
            spp,
            displayed_grain(&plain, width, height),
            displayed_rmse(&plain, &reference)
        );
        for strength in [0.5f64, 1., 2., 5., 10.] {
            let d = render_linear(cornell_denoise_scene(spp, Some(strength)), device, queue);
            line += &format!(
                "   s={}: {:6.3}/{:6.3}",
                strength,
                displayed_grain(&d, width, height),
                displayed_rmse(&d, &reference)
            );
        }
        println!("{}", line);
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

/// Diagnostic, not a gate: what each GPU pass of the bench scenes actually
/// costs, measured with timestamp queries rather than with a CPU clock around
/// submit-plus-poll. This is the table to paste into a commit message that
/// claims a pass got faster.
///
/// `SOLSTRALE_GPU_TIMING` has to come from the environment rather than from the
/// test: setting a variable from inside a test process is unsound while the
/// other test threads are running, and `cargo test -- --ignored` runs the
/// sweeps in parallel.
/// `SOLSTRALE_GPU_TIMING=1 cargo test gpu_pass_timings -- --ignored --nocapture`
///
/// Two blocks, matching the two bench points the cost model was fitted against:
/// `render/test_scene_800x600_64spp` and the 16 spp `denoise` / `post` arms.
/// The per-dispatch lines the environment variable also switches on are
/// interleaved with these tables, and are the finer-grained view of the same
/// numbers -- read them for the wall-clock-versus-GPU column, which says how
/// much of a dispatch is submission latency rather than work.
#[test]
#[ignore]
fn gpu_pass_timings() {
    assert!(
        std::env::var_os(TIMING_ENV).is_some(),
        "run as: {}=1 cargo test gpu_pass_timings -- --ignored --nocapture",
        TIMING_ENV
    );

    let (device, queue) = get_wgpu_device_and_queue();

    let arms: Vec<(u32, &str, Vec<PostProcessors>)> = vec![
        (64, "none", vec![]),
        (16, "none", vec![]),
        (
            16,
            "bloom_0.1",
            vec![
                BloomPostProcessor::new(0.1, None, Some(3.0), device)
                    .unwrap()
                    .into(),
            ],
        ),
        (
            16,
            "bloom_0.002",
            vec![
                BloomPostProcessor::new(0.002, None, Some(3.0), device)
                    .unwrap()
                    .into(),
            ],
        ),
        (
            16,
            "saturation",
            vec![SaturationPostProcessor::new(-0.7, device).unwrap().into()],
        ),
        (
            16,
            "denoise_iterations_5",
            vec![
                DenoisePostProcessor::new(1., Some(5), None, device)
                    .unwrap()
                    .into(),
            ],
        ),
    ];

    for (spp, name, post_processors) in arms {
        let scene = create_test_scene(RenderConfig {
            width: 800,
            height: 600,
            samples_per_pixel: spp,
            post_processors,
            ..Default::default()
        });

        // Whatever the arm before this one left behind.
        drain_totals();
        render_linear(scene, device, queue);
        print_pass_table(&format!("800x600 {} spp, chain: {}", spp, name));
    }
}

/// Prints the accumulated per-pass totals as a table and clears them.
fn print_pass_table(title: &str) {
    let passes: Vec<PassTiming> = drain_totals();
    let total: f64 = passes.iter().map(|p| p.ms).sum();

    println!("\n--- {} ---", title);
    println!(
        "{:<22}{:>8}{:>12}{:>12}{:>9}",
        "pass", "passes", "ms", "ms/pass", "share"
    );
    for p in &passes {
        println!(
            "{:<22}{:>8}{:>12.3}{:>12.4}{:>8.1}%",
            p.label,
            p.count,
            p.ms,
            p.ms / p.count as f64,
            100. * p.ms / total
        );
    }
    println!("{:<22}{:>8}{:>12.3}", "total", "", total);
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

/// The linear value the sRGB decode turns byte 188 into. Spelled out rather
/// than computed, so the test is a statement about the transfer function and
/// not a restatement of the code under test.
const BYTE_188_LINEAR: f64 = 0.502_886_6;

/// A solid `size` x `size` image, for use as an albedo or a normal map.
fn solid_image(size: u32, rgb: [u8; 3]) -> Textures {
    ImageMap::new(Arc::new(RgbImage::from_pixel(size, size, Rgb(rgb)))).into()
}

fn srgb_decode_config() -> RenderConfig {
    RenderConfig {
        width: 80,
        height: 60,
        samples_per_pixel: 256,
        // Three bounces is enough for the albedo to enter the product more than
        // once, and short enough that Russian roulette -- which starts at depth
        // 3 -- never fires. Without that, a path in one scene can survive while
        // its twin in the other terminates, and the two buffers stop being
        // comparable pixel by pixel.
        max_depth: 3,
        // Adaptive sampling retires pixels on their own variance, so two scenes
        // could end up with different sample counts on the same pixel.
        min_samples_per_pixel: u32::MAX,
        ..Default::default()
    }
}

fn max_channel_difference(a: &[[f32; 4]], b: &[[f32; 4]]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .flat_map(|(x, y)| (0..3).map(move |c| (x[c] - y[c]).abs() as f64))
        .fold(0.0, f64::max)
}

/// Where the sRGB decode of an image texture lands, pinned without reference to
/// any golden image.
///
/// Two renders of the same box: one whose walls are textured with a solid image
/// of byte 188, one whose walls are the [`SolidColor`] that byte decodes to. If
/// the decode is missing the textured arm renders at linear 0.737 instead of
/// 0.503, and three bounces off the walls turn that 47% error into a much
/// larger one. If it were applied twice, or to the value after the light's
/// contribution rather than to the texel, the two arms disagree the same way.
///
/// Both arms carry the same normal map, so the atlas holds two textures in both
/// and the albedo lookup has to pick the right region of it.
#[test]
fn test_srgb_decode_matches_equivalent_solid_color() {
    let (device, queue) = get_wgpu_device_and_queue();

    let textured = render_linear(
        create_srgb_decode_scene(
            srgb_decode_config(),
            solid_image(4, [188, 188, 188]),
            Some(solid_image(4, [128, 128, 255])),
        ),
        device,
        queue,
    );
    let solid = render_linear(
        create_srgb_decode_scene(
            srgb_decode_config(),
            SolidColor::new(BYTE_188_LINEAR, BYTE_188_LINEAR, BYTE_188_LINEAR).into(),
            Some(solid_image(4, [128, 128, 255])),
        ),
        device,
        queue,
    );

    let difference = max_channel_difference(&textured, &solid);
    println!("max linear channel difference, textured vs solid: {difference}");

    assert!(
        difference < 1e-3,
        "an image texture of byte 188 should render as the linear colour it \
         decodes to, largest channel difference was {difference}"
    );
}

/// The other half of the placement: the decode must not reach the normal map.
///
/// The atlas is shared between albedo and normal maps, so decoding it as a
/// whole -- by giving the texture an `Rgba8UnormSrgb` format, or by moving the
/// call above the branch in `surface_at` -- would compile and would look right
/// on the albedo. Byte 128 is the neutral axis of a normal map: read raw it is
/// the surface's own normal, read as sRGB it is 0.216, which tilts the shading
/// normal by 39 degrees. So a render through a neutral normal map has to match
/// one with no normal map at all.
#[test]
fn test_srgb_decode_is_not_applied_to_normal_maps() {
    let (device, queue) = get_wgpu_device_and_queue();

    let albedo = || SolidColor::new(0.6, 0.6, 0.6).into();

    let normal_mapped = render_linear(
        create_srgb_decode_scene(
            srgb_decode_config(),
            albedo(),
            Some(solid_image(4, [128, 128, 255])),
        ),
        device,
        queue,
    );
    let plain = render_linear(
        create_srgb_decode_scene(srgb_decode_config(), albedo(), None),
        device,
        queue,
    );

    let difference = max_channel_difference(&normal_mapped, &plain);
    println!("max linear channel difference, neutral normal map vs none: {difference}");

    // Not zero: byte 128 is 0.5020, not 0.5, so the map tilts the normal by
    // 0.22 degrees and the scattered directions shift with it. That residue
    // measures 0.013 here and shrinks with samples; decoding the normal map
    // instead puts it at 0.73, which is what the bound is placed between.
    assert!(
        difference < 0.1,
        "a neutral normal map should be close to no normal map, largest \
         channel difference was {difference}"
    );
}

// ---------------------------------------------------------------------------
// #43: the conductor is a GGX microfacet lobe
// ---------------------------------------------------------------------------

/// Per-pixel centre-ray directions, reproducing `Camera::new` exactly.
///
/// This is what lets a measurement below be restricted to the pixels that look
/// at a particular sphere, rather than to a rectangle guessed from the image.
fn center_rays(camera: &CameraConfig, width: usize, height: usize) -> Vec<Vec3> {
    let aspect_ratio = width as f64 / height as f64;
    let h = (camera.vertical_fov_degrees.to_radians() / 2.).tan();
    let view_port_height = 2. * h;
    let view_port_width = aspect_ratio * view_port_height;

    let look_v = camera.look_from - camera.look_at;
    let focus_distance = look_v.length();
    let w = look_v.unit();
    let u = camera.up.unit().cross(w).unit();
    let v = w.cross(u);

    let horizontal = u * view_port_width * focus_distance;
    let vertical = v * view_port_height * focus_distance;
    let lower_left_corner = camera.look_from - horizontal / 2. - vertical / 2. - w * focus_distance;

    (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                let s = (x as f64 + 0.5) / width as f64;
                let t = 1. - (y as f64 + 0.5) / height as f64;
                lower_left_corner + horizontal * s + vertical * t - camera.look_from
            })
        })
        .collect()
}

/// The pixels whose centre ray passes within `shrink` of the way from the
/// sphere's centre to its silhouette.
///
/// `shrink` is load-bearing rather than cosmetic: a pixel on the silhouette is
/// part sphere and part background, and the background in these scenes is
/// exactly the quantity being compared against.
fn sphere_disc_mask(
    camera: &CameraConfig,
    width: usize,
    height: usize,
    center: Vec3,
    radius: f64,
    shrink: f64,
) -> Vec<bool> {
    let to_center = center - camera.look_from;
    center_rays(camera, width, height)
        .iter()
        // Perpendicular distance from the sphere's centre to the ray's line.
        .map(|d| to_center.cross(*d).length() / d.length() < radius * shrink)
        .collect()
}

fn mask_union(masks: &[Vec<bool>]) -> Vec<bool> {
    let mut out = vec![false; masks[0].len()];
    for mask in masks {
        for (o, m) in out.iter_mut().zip(mask) {
            *o |= m;
        }
    }
    out
}

/// Mean linear radiance over the masked pixels, averaged over the three
/// channels.
fn masked_mean(pixels: &[[f32; 4]], mask: &[bool]) -> f64 {
    let (sum, count) = pixels
        .iter()
        .zip(mask)
        .filter(|(_, m)| **m)
        .fold((0., 0), |(sum, count), (p, _)| {
            (sum + (p[0] + p[1] + p[2]) as f64 / 3., count + 1)
        });
    assert!(count > 0, "the mask selected no pixels");
    sum / count as f64
}

/// `linear_rmse` over the masked pixels only.
fn masked_linear_rmse(a: &[[f32; 4]], b: &[[f32; 4]], mask: &[bool]) -> f64 {
    let (sum_sq, count) = a.iter().zip(b).zip(mask).filter(|(_, m)| **m).fold(
        (0., 0),
        |(sum_sq, count), ((p, q), _)| {
            let d: f64 = (0..3).map(|c| ((p[c] - q[c]) as f64).powi(2)).sum();
            (sum_sq + d, count + 3)
        },
    );
    assert!(count > 0, "the mask selected no pixels");
    (sum_sq / count as f64).sqrt()
}

/// A white metal in a white furnace, which is the only way to see what a
/// microfacet BRDF does with the energy it is given.
///
/// Under a uniform environment of radiance 1 an energy-conserving BRDF reflects
/// exactly 1 at every angle and every roughness, so whatever the sphere reads
/// below 1 is energy the BRDF lost. With `albedo = 1` the Fresnel term is 1
/// everywhere too, so what is left is the masking-shadowing term, the samples
/// GGX reflects below the horizon, and how much of both the multiple-scattering
/// factor puts back.
///
/// Three ways the obvious version of this measurement lies:
///
/// - the whole-image mean is dominated by background pixels, which equal 1.0
///   whatever the BRDF does, so the disc mask is doing the work;
/// - the saved JPEG has been through ACES and a transfer function, so the
///   measurement has to be of the linear buffer;
/// - `CLAMPING_THRESHOLD` and Russian roulette both touch the estimator. RR is
///   unbiased so it would be fine either way, and it never runs here: a sphere
///   is convex, so every path is one bounce and the background, and RR starts
///   at depth 3. The clamp is inert because a unit environment never comes near
///   10. Neither is disabled for the test -- they are arranged to be inert, so
///   what is measured is the shipping estimator.
///
/// The Lambertian control costs one more render and catches a wrong mask, a
/// wrong stride, or a tone mapper leaking into the readback -- all of which
/// would otherwise be read as a GGX defect.
#[test]
fn test_ggx_metal_is_energy_conserving_in_a_furnace() {
    let (device, queue) = get_wgpu_device_and_queue();

    const SIZE: usize = 160;
    let config = || RenderConfig {
        width: SIZE,
        height: SIZE,
        samples_per_pixel: 512,
        // Off: a retired pixel holds whatever mean it had when it retired, and
        // this test is a measurement of the mean.
        min_samples_per_pixel: u32::MAX,
        ..Default::default()
    };

    let white = || SolidColor::new(1., 1., 1.).into();
    let scene = create_furnace_scene(config(), Lambertian::new(white(), None).into());
    let mask = sphere_disc_mask(
        &scene.camera,
        SIZE,
        SIZE,
        FURNACE_SPHERE_CENTER,
        FURNACE_SPHERE_RADIUS,
        0.8,
    );

    let lambertian = masked_mean(&render_linear(scene, device, queue), &mask);
    println!("furnace, albedo-1 Lambertian: {lambertian:.4}");
    assert!(
        (lambertian - 1.).abs() < 0.005,
        "an albedo-1 Lambertian sphere in a unit furnace must read 1, not {lambertian}; \
         the BRDF is not what is wrong here, the harness is"
    );

    for (fuzz, expected) in FURNACE_EXPECTED {
        let scene = create_furnace_scene(config(), Metal::new(white(), None, fuzz).into());
        let measured = masked_mean(&render_linear(scene, device, queue), &mask);
        println!(
            "furnace, metal fuzz {fuzz} (alpha {:.3}): {measured:.4}",
            fuzz * fuzz
        );

        assert!(
            (measured - expected).abs() < 0.01,
            "furnace reading for fuzz {fuzz} moved: {measured:.4} against the pinned {expected:.4}"
        );
        // The physical criterion, separate from the pin above: a compensated
        // conductor conserves energy, and 2% is the room the fit is allowed.
        assert!(
            measured > 0.98,
            "a compensated metal at fuzz {fuzz} loses {:.1}% of the energy it is given",
            (1. - measured) * 100.
        );
    }
}

/// What the furnace reads per roughness. Pinned rather than bounded, so that
/// anything which moves them shows up in a diff.
///
/// Uncompensated, the same five readings were 1.0000 / 0.9942 / 0.8976 /
/// 0.6318 / 0.3503 -- a fully rough metal kept a third of the light it was
/// given. A CPU quadrature of the BRDF's definition agreed with every one of
/// those to 0.0015, so that table pinned the single-scatter model rather than
/// this implementation of it, and this one pins how much of the rest Turquin's
/// factor puts back.
///
/// `albedo = 1` is the hardest case for it, not the easiest: with f0 = 1 the
/// factor is exactly `1 / E`, so the compensated reading is `E_true / E_fit`
/// and the fit's error appears undiluted. A coloured metal is compensated less,
/// by design -- light that bounces twice between microfacets is tinted twice.
const FURNACE_EXPECTED: [(f64, f64); 5] = [
    (0., 1.0000),
    (0.25, 0.9970),
    (0.5, 0.9983),
    (0.75, 0.9980),
    (1., 0.9975),
];

/// Next-event estimation now reaches a rough metal, and this is the measurement
/// that says so: the same scene at 64 and 1024 spp, compared over the spheres
/// alone.
///
/// Relative to the reference's own mean, not absolute. The pre-GGX metal lost
/// most of its energy to the below-horizon `break`, so its image was six times
/// darker and an absolute error would have scored it *better* for being dark.
/// Measured on this scene, at 64 spp against each model's own 1024 spp
/// reference: the fuzz-sphere metal with no next-event estimation ran 0.886 of
/// its own mean, the single-scatter GGX metal with it ran 0.346, and with the
/// multiple-scattering factor -- which brightens the rough spheres and so
/// raises the mean the error is taken against -- it runs 0.256.
///
/// The golden harness provably cannot capture this. It resizes to 100x50 before
/// comparing, which averages the noise away -- the noisy before and the clean
/// after score almost identically through it. **The golden harness averages away
/// exactly the defect being fixed**, which is why this test measures per-pixel
/// error at full resolution instead.
#[test]
fn test_rough_metal_converges_with_nee() {
    let (device, queue) = get_wgpu_device_and_queue();

    const WIDTH: usize = 200;
    const HEIGHT: usize = 100;
    let config = |samples_per_pixel| RenderConfig {
        width: WIDTH,
        height: HEIGHT,
        samples_per_pixel,
        // Off in both arms: adaptive sampling would stop sampling the very
        // pixels whose noise is being measured.
        min_samples_per_pixel: u32::MAX,
        ..Default::default()
    };

    let scene = create_rough_metal_scene(config(64));
    let mask = mask_union(
        &(0..ROUGH_METAL_FUZZ.len())
            .map(|i| {
                sphere_disc_mask(
                    &scene.camera,
                    WIDTH,
                    HEIGHT,
                    rough_metal_sphere_center(i),
                    ROUGH_METAL_RADIUS,
                    0.9,
                )
            })
            .collect::<Vec<_>>(),
    );

    let noisy = render_linear(scene, device, queue);
    let reference = render_linear(
        as_reference(create_rough_metal_scene(config(1024))),
        device,
        queue,
    );

    let relative = masked_linear_rmse(&noisy, &reference, &mask) / masked_mean(&reference, &mask);
    println!("rough metal, 64 spp against 1024 spp, over the spheres: {relative:.4} of the mean");

    assert!(
        relative < ROUGH_METAL_RMSE_BOUND,
        "64 spp of the rough metal scene is at {relative:.4} relative RMSE against its own \
         1024 spp reference, past the pinned {ROUGH_METAL_RMSE_BOUND}"
    );
}

/// Pinned above the 0.256 measured, with room for a different driver's
/// arithmetic. Well below the 0.886 the same scene scored before metal could
/// take a shadow ray, which is the point of the bound rather than its exact
/// value.
const ROUGH_METAL_RMSE_BOUND: f64 = 0.35;

// ---------------------------------------------------------------------------
// #82: the dielectric is a GGX microfacet lobe
// ---------------------------------------------------------------------------

/// The same measurement as `test_rough_metal_converges_with_nee`, on the twin
/// scene in glass: 64 spp against its own 1024 spp reference, over the rough
/// spheres.
///
/// Relative to the reference's own mean rather than absolute, for the reason
/// the metal test gives -- an absolute figure rewards an arm that is merely
/// darker, and the single-scatter dielectric does darken as roughness rises.
///
/// Measured here, against a build with `bsdf_is_specular` forced true for the
/// dielectric and its sampled pdf forced to 0 -- the lobe still has its width,
/// it simply has no density for a light to land on, which is exactly what a
/// rough dielectric bolted onto the old RTIOW arm would have been:
///
/// ```text
/// roughness      0.15    0.3    0.5    0.8   all four
/// width, no pdf  4.588  4.212  5.114  8.342    5.250
/// with pdf       0.667  1.080  0.764  0.627    0.877
/// ```
///
/// Six times less error for the same 64 samples, and the gap widens with
/// roughness, which is the shape the claim predicts: the wider the lobe, the
/// smaller the chance of stumbling onto a light a fraction of a degree across.
///
/// The smooth sphere is deliberately outside the mask. It is a Dirac lobe that
/// this change does not touch -- it scores 4.036 before and 4.074 after, the
/// same number twice -- and what it is noisy *about* is a caustic: a glass ball
/// is a lens, and a lens focusing a pinpoint is the one thing next-event
/// estimation cannot help with, because a shadow ray has no chance of landing
/// on a delta. That is issue #82's second commit, not this one.
///
/// **The golden harness cannot see this.** It downscales to 100x50 before
/// comparing, which averages away exactly the defect being measured -- the
/// noisy before and the clean after score almost identically through it. Hence
/// a per-pixel error at full resolution, as the metal test does.
#[test]
fn test_rough_glass_converges_with_nee() {
    let (device, queue) = get_wgpu_device_and_queue();

    const WIDTH: usize = 200;
    const HEIGHT: usize = 100;
    let config = |samples_per_pixel| RenderConfig {
        width: WIDTH,
        height: HEIGHT,
        samples_per_pixel,
        // Off in both arms: adaptive sampling would stop sampling the very
        // pixels whose noise is being measured.
        min_samples_per_pixel: u32::MAX,
        ..Default::default()
    };

    let scene = create_rough_glass_scene(config(64));
    // From 1: the roughness-0 sphere is the Dirac lobe this measurement is not
    // about. See the note above.
    let mask = mask_union(
        &(1..ROUGH_GLASS_ROUGHNESS.len())
            .map(|i| {
                sphere_disc_mask(
                    &scene.camera,
                    WIDTH,
                    HEIGHT,
                    rough_glass_sphere_center(i),
                    ROUGH_GLASS_RADIUS,
                    0.9,
                )
            })
            .collect::<Vec<_>>(),
    );

    let noisy = render_linear(scene, device, queue);
    let reference = render_linear(
        as_reference(create_rough_glass_scene(config(1024))),
        device,
        queue,
    );

    let relative = masked_linear_rmse(&noisy, &reference, &mask) / masked_mean(&reference, &mask);
    println!(
        "rough glass, 64 spp against 1024 spp, over the rough spheres: {relative:.4} of the mean"
    );

    assert!(
        relative < ROUGH_GLASS_RMSE_BOUND,
        "64 spp of the rough glass scene is at {relative:.4} relative RMSE against its own \
         1024 spp reference, past the pinned {ROUGH_GLASS_RMSE_BOUND}"
    );
}

/// Pinned above the 0.877 measured, with the same room for a different driver's
/// arithmetic that `ROUGH_METAL_RMSE_BOUND` leaves. Far below the 5.250 the
/// same scene scores with a lobe that has width but no density, which is the
/// point of the bound rather than its exact value.
const ROUGH_GLASS_RMSE_BOUND: f64 = 1.2;

/// The regression net under solid-angle sampling of a quad light, on the one
/// scene in the suite whose light is large enough for it to matter.
///
/// Nothing else could hold it. On Cornell the technique is worth between half a
/// percent and three percent of RMSE -- real, and far inside the noise of any
/// gate that could be written around it -- because its emitter subtends a small
/// enough solid angle that area sampling's `d^2 / cos` barely varies across it.
/// Here that factor varies by 49 across the floor and without bound on the
/// walls, and the same measurement separates the two samplers by 36%.
///
/// It catches variance, not bias: the reference is rendered by the same code,
/// so a sampler that lost energy would move both arms together and read as
/// unchanged. `test_bsdf_only_sampling_converges_to_the_same_image` is what
/// holds that end, against an estimator that samples no light at all.
///
/// Relative to the reference's own mean, as `test_rough_metal_converges_with_nee`
/// is and for the same reason: an absolute figure rewards an arm that is simply
/// darker.
#[test]
fn test_wide_quad_light_is_solid_angle_sampled() {
    let (device, queue) = get_wgpu_device_and_queue();

    let config = |samples_per_pixel| RenderConfig {
        width: 200,
        height: 100,
        samples_per_pixel,
        // Off: adaptive sampling would stop spending samples on the very pixels
        // whose noise is the measurement.
        min_samples_per_pixel: u32::MAX,
        ..Default::default()
    };

    let noisy = render_linear(create_wide_light_scene(config(16)), device, queue);
    let reference = render_linear(
        as_reference(create_wide_light_scene(config(2000))),
        device,
        queue,
    );

    let mean: f64 = reference
        .iter()
        .map(|p| (p[0] + p[1] + p[2]) as f64 / 3.)
        .sum::<f64>()
        / reference.len() as f64;
    let relative = linear_rmse(&noisy, &reference) / mean;
    println!("wide quad light, 16 spp against 2000 spp: {relative:.4} of the mean");

    assert!(
        relative < WIDE_LIGHT_RMSE_BOUND,
        "16 spp of the wide-light scene is at {relative:.4} relative RMSE against its own \
         2000 spp reference, past the pinned {WIDE_LIGHT_RMSE_BOUND}"
    );
}

/// Pinned between the two samplers, with room for a different driver's
/// arithmetic on the side that has to pass. Sampling the light by solid angle
/// measures 0.0283; sampling it by area measures 0.0444, which is what the
/// bound is really placed against.
const WIDE_LIGHT_RMSE_BOUND: f64 = 0.035;

/// Beer-Lambert absorption inside a dielectric, measured as channel ratios
/// rather than as brightness.
///
/// A glass slab of known thickness in front of a unit emitter, with the
/// albedo's red channel at 1 so red is the unattenuated control. Dividing green
/// and blue by red cancels both Fresnel interfaces, both refractions and any
/// residual throughput scaling, leaving `albedo^thickness` as the only term the
/// ratio can still see.
///
/// The obvious metric lies. "The image got darker" passes for a wrong sign, a
/// wrong exponent, absorption applied on entry instead of on exit, or applied
/// once per bounce instead of per unit length -- every one of those merely
/// dims. None of them survives a ratio.
///
/// Two thicknesses, and neither of them 1: at unit thickness `a^t` and a flat
/// `a` per traversal are the same number, so a single slab cannot tell an
/// exponential from a constant.
///
/// With the albedo ignored, as it was, both ratios read 1.
#[test]
fn test_beer_lambert_absorption_follows_the_exponential() {
    let (device, queue) = get_wgpu_device_and_queue();

    const SIZE: usize = 200;
    const HALF: usize = 5;
    let config = RenderConfig {
        width: SIZE,
        height: SIZE,
        samples_per_pixel: 64,
        // Off: a retired pixel holds whatever mean it had when it retired, and
        // this is a measurement of the mean.
        min_samples_per_pixel: u32::MAX,
        ..Default::default()
    };
    let albedo = Vec3::new(1., 0.5, 0.25);

    for thickness in [0.5, 2.] {
        let pixels = render_linear(
            create_glass_slab_scene(config.clone(), thickness, albedo),
            device,
            queue,
        );

        // The central 11x11, which is where the slab is crossed at normal
        // incidence and the path through it really is `thickness` long.
        let mut sum = [0f64; 3];
        for y in SIZE / 2 - HALF..=SIZE / 2 + HALF {
            for x in SIZE / 2 - HALF..=SIZE / 2 + HALF {
                let p = pixels[y * SIZE + x];
                (0..3).for_each(|c| sum[c] += p[c] as f64);
            }
        }
        let n = ((2 * HALF + 1) * (2 * HALF + 1)) as f64;
        let (r, g, b) = (sum[0] / n, sum[1] / n, sum[2] / n);
        println!("glass slab {thickness} units: r {r:.4} g {g:.4} b {b:.4}");

        // Catches the harness rather than the absorption: two normal-incidence
        // interfaces pass 92% of the emitter, and a red channel far off that
        // means the centre rays are not going through the slab at all.
        assert!(
            (r - 0.923).abs() < 0.02,
            "red should be the unattenuated control at 0.923, not {r:.4}"
        );

        for (name, measured, expected) in [
            ("green", g / r, 0.5f64.powf(thickness)),
            ("blue", b / r, 0.25f64.powf(thickness)),
        ] {
            let relative = (measured / expected - 1.).abs();
            assert!(
                relative < 0.02,
                "{name} over red through {thickness} units of glass is {measured:.4}, \
                 against the {expected:.4} that albedo^thickness asks for -- {:.1}% off",
                relative * 100.
            );
        }
    }
}
