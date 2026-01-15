use std::collections::HashMap;
use std::default::Default;
use std::error::Error;
use std::ops::Deref;
use std::sync::mpsc::channel;
use std::thread;

use image::RgbImage;
use image::imageops::FilterType;
use image_compare::Algorithm::RootMeanSquared;

use solstrale::geo::transformation::{RotationX, RotationY, RotationZ, Transformer};
use solstrale::geo::vec3::{Vec3, ZERO_VECTOR};
use solstrale::post::{
    BloomPostProcessor, OidnPostProcessor, PostProcessor, SaturationPostProcessor,
};
use solstrale::ray_trace;
use solstrale::renderer::shader::{PathTracingShader, Shaders, SimpleShader};
use solstrale::renderer::{RenderConfig, Scene};
use solstrale::util::rgb_color::rgb_to_vec3;

use crate::scenes::{
    create_blend_material_scene, create_light_attenuation_scene, create_normal_mapping_scene,
    create_normal_mapping_sphere_scene, create_obj_scene, create_obj_with_box,
    create_obj_with_triangle, create_quad_rotation_scene, create_simple_test_scene,
    create_test_scene, create_uv_scene,
};

mod scenes;

const IMAGE_COMPARISON_SCORE_THRESHOLD: f64 = 0.95;

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

        render_and_compare_output(scene, shader_name)
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
    render_and_compare_output(scene, "oidn")
}

#[test]
fn test_render_obj_with_textures() {
    let render_config = RenderConfig {
        width: 200,
        height: 100,
        samples_per_pixel: 20,
        ..Default::default()
    };
    let scene = create_obj_scene(render_config);

    render_and_compare_output(scene, "obj");
}

#[test]
fn test_render_obj_with_default_material() {
    let render_config = RenderConfig {
        width: 200,
        height: 100,
        ..Default::default()
    };
    let scene = create_obj_with_box(render_config, "resources/obj/", "box.obj");

    render_and_compare_output(scene, "obj_default");
}

#[test]
fn test_render_obj_with_diffuse_material() {
    let render_config = RenderConfig {
        width: 200,
        height: 100,
        ..Default::default()
    };
    let scene = create_obj_with_box(render_config, "resources/obj/", "boxWithMat.obj");

    render_and_compare_output(scene, "obj_diffuse");
}

#[test]
fn test_render_uv_mapping() {
    let render_config = RenderConfig {
        width: 200,
        height: 200,
        samples_per_pixel: 5,
        ..Default::default()
    };
    let scene = create_uv_scene(render_config);

    render_and_compare_output(scene, "uv");
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
    render_and_compare_output(scene, "normal_mapping_disabled");
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
    render_and_compare_output(scene, "normal_mapping_1");
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
    render_and_compare_output(scene, "normal_mapping_2");
}

#[test]
fn test_render_normal_mapping_sphere_1() {
    let render_config = RenderConfig {
        width: 300,
        height: 300,
        ..Default::default()
    };
    let scene = create_normal_mapping_sphere_scene(render_config, Vec3::new(-30., 30., 30.));
    render_and_compare_output(scene, "normal_mapping_sphere_1");
}

#[test]
fn test_render_normal_mapping_sphere_2() {
    let render_config = RenderConfig {
        width: 300,
        height: 300,
        ..Default::default()
    };
    let scene = create_normal_mapping_sphere_scene(render_config, Vec3::new(30., 30., 30.));
    render_and_compare_output(scene, "normal_mapping_sphere_2");
}

#[test]
fn test_render_scene_without_light() {
    let render_config = RenderConfig {
        width: 20,
        height: 10,
        samples_per_pixel: 100,
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

    render_and_compare_output(scene, "obj_normal_map");
}

#[test]
fn test_render_obj_with_height_map() {
    let render_config = RenderConfig {
        width: 300,
        height: 300,
        ..Default::default()
    };
    let scene = create_obj_with_triangle(render_config, "resources/obj/", "triWithHeightMap.obj");

    render_and_compare_output(scene, "obj_height_map");
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
        );
    }
}

#[test]
fn test_bloom() -> Result<(), Box<dyn Error>> {
    let mut post = BloomPostProcessor::new(0.2, None, None)?;
    let bloom_image = image::open("resources/textures/bloom.png")
        .unwrap()
        .into_rgb8();
    let w = bloom_image.width();
    let h = bloom_image.height();
    let pixel_colors = image_to_vec3(bloom_image);

    post.initialize(w, h);

    let res = post.post_process(&pixel_colors, &[ZERO_VECTOR; 0], &[ZERO_VECTOR; 0], 1)?;

    compare_output("bloom", &res);

    let res = post.post_process(&pixel_colors, &[ZERO_VECTOR; 0], &[ZERO_VECTOR; 0], 1)?;

    compare_output("bloom", &res);

    Ok(())
}

#[test]
fn test_saturation() -> Result<(), Box<dyn Error>> {
    for saturation_factor in [-1., 0., 1.] {
        let mut post = SaturationPostProcessor::new(saturation_factor)?;
        let saturation_image = image::open("resources/textures/bloom.png")
            .unwrap()
            .into_rgb8();
        let w = saturation_image.width();
        let h = saturation_image.height();
        let pixel_colors = image_to_vec3(saturation_image);

        post.initialize(w, h);

        let res = post.post_process(&pixel_colors, &[ZERO_VECTOR; 0], &[ZERO_VECTOR; 0], 1)?;

        compare_output(&format!("saturation_{}", saturation_factor), &res);

        let res = post.post_process(&pixel_colors, &[ZERO_VECTOR; 0], &[ZERO_VECTOR; 0], 1)?;

        compare_output(&format!("saturation_{}", saturation_factor), &res);
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
                shader: SimpleShader::new().into(),
                samples_per_pixel: 1,
                ..RenderConfig::default()
            },
            rotation.deref(),
        );

        render_and_compare_output(scene, &format!("quad_rotated{}", i));
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

        render_and_compare_output(scene, &format!("blended_materials_{}", blend_factor));
    }
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

fn render_and_compare_output(scene: Scene, name: &str) {
    let (output_sender, output_receiver) = channel();
    let (_, abort_receiver) = channel();

    let width = scene.render_config.width as u32;
    let height = scene.render_config.height as u32;

    thread::spawn(move || {
        ray_trace(scene, &output_sender, &abort_receiver).unwrap();
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
    let sized_expected = image::imageops::resize(&expected_image, 100, 50, FilterType::Gaussian);

    let score =
        image_compare::rgb_similarity_structure(&RootMeanSquared, &sized_expected, &sized_actual)
            .expect("Failed to compare images")
            .score;

    assert!(
        score > IMAGE_COMPARISON_SCORE_THRESHOLD,
        "Comparison score for {} is: {}",
        name,
        score
    )
}
