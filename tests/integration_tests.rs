use std::collections::HashMap;
use std::default::Default;
use std::error::Error;
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
use solstrale::geo::vec3::{Vec3, ZERO_VECTOR};
use solstrale::hittable::{Bvh, ConstantMedium, Hittables, Quad, Sphere, Triangle};
use solstrale::material::texture::SolidColor;
use solstrale::material::{DiffuseLight, Lambertian};
use solstrale::post::{
    BloomPostProcessor, OidnPostProcessor, PostProcessor, SaturationPostProcessor,
};
use solstrale::ray_trace;
use solstrale::renderer::gpu_renderer::GpuRenderer;
use solstrale::renderer::shader::{PathTracingShader, Shaders, SimpleShader};
use solstrale::renderer::{RenderConfig, Renderer, Scene};
use solstrale::util::rgb_color::rgb_to_vec3;

use crate::scenes::{
    create_blend_material_scene, create_light_attenuation_scene, create_normal_mapping_scene,
    create_normal_mapping_sphere_scene, create_obj_scene, create_obj_with_box,
    create_obj_with_triangle, create_quad_rotation_scene, create_simple_test_scene,
    create_test_scene, create_texture_mapping_scene, create_uv_scene,
};

mod scenes;

#[test]
fn test_render_scene() {
    let shaders: HashMap<&str, Shaders> = HashMap::from([
        ("pathTracing", PathTracingShader::new(50).into()),
        ("simple", SimpleShader::new().into()),
    ]);

    for (shader_name, shader) in shaders {
        let render_config = RenderConfig {
            width: 200,
            height: 100,
            samples_per_pixel: 25,
            shader,
            ..Default::default()
        };
        let scene = create_test_scene(render_config);

        render_and_compare_output(scene, shader_name, 0.95, false)
    }
}

#[test]
#[cfg(feature = "oidn-postprocessor")]
fn test_render_scene_with_oidn() {
    let render_config = RenderConfig {
        width: 200,
        height: 100,
        samples_per_pixel: 20,
        shader: PathTracingShader::new(50),
        post_processors: vec![OidnPostProcessor::new()],
        ..Default::default()
    };

    let scene = create_simple_test_scene(render_config, true);
    render_and_compare_output(scene, "oidn", true)
}

#[test]
fn test_render_obj_with_textures() {
    let render_config = RenderConfig {
        width: 200,
        height: 100,
        ..Default::default()
    };
    let scene = create_obj_scene(render_config);

    render_and_compare_output(scene, "obj", 0.95, true);
}

#[test]
fn test_render_obj_with_default_material() {
    let render_config = RenderConfig {
        width: 200,
        height: 100,
        ..Default::default()
    };
    let scene = create_obj_with_box(render_config, "resources/obj/", "box.obj");

    render_and_compare_output(scene, "obj_default", 0.95, true);
}

#[test]
fn test_render_obj_with_diffuse_material() {
    let render_config = RenderConfig {
        width: 200,
        height: 100,
        ..Default::default()
    };
    let scene = create_obj_with_box(render_config, "resources/obj/", "boxWithMat.obj");

    render_and_compare_output(scene, "obj_diffuse", 0.95, true);
}

#[test]
fn test_render_uv_mapping() {
    let render_config = RenderConfig {
        width: 200,
        height: 200,
        ..Default::default()
    };
    let scene = create_uv_scene(render_config);

    render_and_compare_output(scene, "uv", 0.95, true);
}

#[test]
fn test_render_normal_mapping_disabled() {
    let render_config = RenderConfig {
        width: 300,
        height: 300,
        post_processors: vec![OidnPostProcessor::new().into()],
        ..Default::default()
    };

    let scene = create_normal_mapping_scene(render_config, Vec3::new(30., 30., 30.), false);
    render_and_compare_output(scene, "normal_mapping_disabled", 0.95, true);
}

#[test]
fn test_render_normal_mapping_1() {
    let render_config = RenderConfig {
        width: 300,
        height: 300,
        post_processors: vec![OidnPostProcessor::new().into()],
        ..Default::default()
    };

    let scene = create_normal_mapping_scene(render_config, Vec3::new(30., 30., 30.), true);
    render_and_compare_output(scene, "normal_mapping_1", 0.95, true);
}

#[test]
fn test_render_normal_mapping_2() {
    let render_config = RenderConfig {
        width: 300,
        height: 300,
        post_processors: vec![OidnPostProcessor::new().into()],
        ..Default::default()
    };

    let scene = create_normal_mapping_scene(render_config, Vec3::new(-30., 30., 30.), true);
    render_and_compare_output(scene, "normal_mapping_2", 0.95, true);
}

#[test]
fn test_render_normal_mapping_sphere_1() {
    let render_config = RenderConfig {
        width: 300,
        height: 300,
        ..Default::default()
    };
    let scene = create_normal_mapping_sphere_scene(render_config, Vec3::new(-30., 30., 30.));
    render_and_compare_output(scene, "normal_mapping_sphere_1", 0.97, true);
}

#[test]
fn test_render_normal_mapping_sphere_2() {
    let render_config = RenderConfig {
        width: 300,
        height: 300,
        ..Default::default()
    };
    let scene = create_normal_mapping_sphere_scene(render_config, Vec3::new(30., 30., 30.));
    render_and_compare_output(scene, "normal_mapping_sphere_2", 0.97, true);
}

#[test]
fn test_render_scene_without_light() {
    let render_config = RenderConfig {
        width: 20,
        height: 10,
        ..Default::default()
    };
    let scene = create_simple_test_scene(render_config, false);

    let (output_sender, _) = channel();
    let (_, abort_receiver) = channel();

    let res = ray_trace(scene, &output_sender, &abort_receiver);

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

    render_and_compare_output(scene, "obj_normal_map", 0.95, true);
}

#[test]
fn test_render_obj_with_height_map() {
    let render_config = RenderConfig {
        width: 300,
        height: 300,
        ..Default::default()
    };
    let scene = create_obj_with_triangle(render_config, "resources/obj/", "triWithHeightMap.obj");

    render_and_compare_output(scene, "obj_height_map", 0.95, true);
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
            true,
        );
    }
}

#[test]
fn test_bloom() -> Result<(), Box<dyn Error>> {
    let mut post = BloomPostProcessor::new(0.2, None, None)?;
    let bloom_image = image::open("resources/textures/bloom.png")?.into_rgb8();
    let w = bloom_image.width();
    let h = bloom_image.height();
    let pixel_colors = image_to_vec3(bloom_image);

    post.initialize(w, h);

    let res = post.post_process(&pixel_colors, &[ZERO_VECTOR; 0], &[ZERO_VECTOR; 0], 1)?;

    compare_output("bloom", &res, 0.95);

    let res = post.post_process(&pixel_colors, &[ZERO_VECTOR; 0], &[ZERO_VECTOR; 0], 1)?;

    compare_output("bloom", &res, 0.95);

    Ok(())
}

#[test]
fn test_saturation() -> Result<(), Box<dyn Error>> {
    for saturation_factor in [-1., 0., 1.] {
        let mut post = SaturationPostProcessor::new(saturation_factor)?;
        let saturation_image = image::open("resources/textures/bloom.png")?.into_rgb8();
        let w = saturation_image.width();
        let h = saturation_image.height();
        let pixel_colors = image_to_vec3(saturation_image);

        post.initialize(w, h);

        let res = post.post_process(&pixel_colors, &[ZERO_VECTOR; 0], &[ZERO_VECTOR; 0], 1)?;

        compare_output(&format!("saturation_{}", saturation_factor), &res, 0.95);

        let res = post.post_process(&pixel_colors, &[ZERO_VECTOR; 0], &[ZERO_VECTOR; 0], 1)?;

        compare_output(&format!("saturation_{}", saturation_factor), &res, 0.95);
    }

    Ok(())
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

        render_and_compare_output(scene, &format!("quad_rotated{}", i), 0.95, true);
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

        render_and_compare_output(
            scene,
            &format!("blended_materials_{}", blend_factor),
            0.95,
            true,
        );
    }
}

#[test]
fn test_texture_map() {
    let scene = create_texture_mapping_scene(RenderConfig {
        width: 300,
        height: 300,
        ..RenderConfig::default()
    });

    render_and_compare_output(scene, "texture_map", 0.95, true);
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

    render_and_compare_output(scene, "gpu_sphere", 0.95, true);
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

    render_and_compare_output(scene, "gpu_box", 0.95, true);
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

    render_and_compare_output(scene, "gpu_sphere2", 0.95, true);
}

#[test]
fn test_gpu_scene_spheres_and_fog() {
    let render_config = RenderConfig {
        width: 400,
        height: 400,
        samples_per_pixel: 200,
        ..Default::default()
    };

    let camera = CameraConfig {
        look_from: Vec3::new(0., 0., 20.),
        look_at: Vec3::new(0., 0., 0.),
        ..Default::default()
    };

    let mut world: Vec<Hittables> = Vec::new();
    let light = DiffuseLight::new(45., 45., 45., None);
    world.push(Sphere::new(Vec3::new(-30., 30., 30.), 10., light.into()).into());

    let blue = Lambertian::new(SolidColor::new(0.2, 0.2, 1.).into(), None);
    let red = Lambertian::new(SolidColor::new(1., 0.2, 0.2).into(), None);

    world.push(Sphere::new(Vec3::new(-4., -1., 0.), 4., blue.into()).into());
    world.push(Sphere::new(Vec3::new(4., 1., 0.), 2., red.clone().into()).into());
    world.push(
        ConstantMedium::new(
            Sphere::new(Vec3::new(4., 1., 0.), 8., red.into()).into(),
            0.03,
            Vec3::new(1.0, 1.0, 1.0),
        )
        .into(),
    );

    let scene = Scene {
        world: Bvh::new(world).into(),
        camera,
        background_color: Vec3::new(0., 0., 0.),
        render_config,
    };

    render_and_compare_output(scene, "gpu_spheres_and_fog", 0.95, true);
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

    render_and_compare_output(scene, "gpu_sphere_quad_and_triangle", 0.95, true);
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

    render_and_compare_output(scene, "gpu_triangle3", 0.95, true);
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

    render_and_compare_output(scene, "gpu_nested_bvh", 0.95, true);
}

fn image_to_vec3(image: RgbImage) -> Vec<Vec3> {
    let mut ret = Vec::with_capacity((image.width() * image.height()) as usize);
    for y in 0..image.height() {
        for x in 0..image.width() {
            ret.push(rgb_to_vec3(image.get_pixel(x, y)));
        }
    }
    ret
}

fn render_and_compare_output(scene: Scene, name: &str, comparison_threshold: f64, gpu: bool) {
    let (output_sender, output_receiver) = channel();
    let (_, abort_receiver) = channel();

    let width = scene.render_config.width as u32;
    let height = scene.render_config.height as u32;

    thread::spawn(move || {
        if gpu {
            GpuRenderer::new(scene)
                .unwrap()
                .render(&output_sender, &abort_receiver)
                .unwrap();
        } else {
            Renderer::new(scene)
                .unwrap()
                .render(&output_sender, &abort_receiver)
                .unwrap();
        }
    });

    let mut image = RgbImage::new(width, height);
    for render_output in output_receiver {
        if let Some(render_image) = render_output.render_image {
            image = render_image;
        }
    }

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
