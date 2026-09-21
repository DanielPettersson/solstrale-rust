//! Checks that what the tracer is compiled against matches the scene it is
//! compiled for.
//!
//! Every flag in [`Specialisation`] removes a branch the scene could never have
//! taken, so a correctly wired one is invisible in the image and a mis-wired one
//! is not: the wrong primitive buffer gets indexed, or a material loses the lobe
//! that shades it. That is what makes the render below a test at all -- on a
//! scene that uses every feature the two pipelines are compiled from identical
//! constants, so anything but bit-identical output means a flag was derived
//! wrongly.

use std::sync::Arc;
use std::sync::mpsc::channel;

use image::{Rgb, RgbImage};

use crate::camera::CameraConfig;
use crate::geo::transformation::NopTransformer;
use crate::geo::vec3::Vec3;
use crate::hittable::{Bvh, Hittables, Quad, Sphere, Triangle};
use crate::material::texture::{ImageMap, SolidColor, Textures};
use crate::material::{Blend, Dielectric, DiffuseLight, Lambertian, Metal};
use crate::renderer::scene_flattener::flatten_scene;
use crate::renderer::{RenderConfig, Renderer, Scene, Specialisation};
use crate::util::wgpu_util::{get_result_from_buffer, get_wgpu_device_and_queue};

/// Small and cheap, and with the batch size equal to the sample count so the
/// whole render is a single dispatch -- the Welford merge across batches rounds
/// differently for different groupings, and the renderer tunes the grouping
/// from wall-clock time.
fn test_render_config() -> RenderConfig {
    RenderConfig {
        width: 64,
        height: 48,
        samples_per_pixel: 8,
        samples_per_batch: 8,
        ..Default::default()
    }
}

fn solid_image(rgb: [u8; 3]) -> Arc<RgbImage> {
    Arc::new(RgbImage::from_pixel(4, 4, Rgb(rgb)))
}

/// A scene that uses every feature the tracer specialises on, so
/// `Specialisation::from_scene_data` has to report all of them.
fn every_feature_scene() -> Scene {
    let nop = NopTransformer();

    let albedo: Textures = ImageMap::new(solid_image([200, 180, 160])).into();
    // A flat tangent-space normal, which leaves the shading normal where it was
    // while still taking the normal-map branch.
    let normal: Textures = ImageMap::new(solid_image([128, 128, 255])).into();

    let world: Vec<Hittables> = vec![
        Quad::new(
            Vec3::new(-4., 0., -4.),
            Vec3::new(8., 0., 0.),
            Vec3::new(0., 0., 8.),
            Lambertian::new(albedo, Some(normal)).into(),
            &nop,
        )
        .into(),
        Quad::new(
            Vec3::new(-1., 3., -1.),
            Vec3::new(2., 0., 0.),
            Vec3::new(0., 0., 2.),
            DiffuseLight::new(8., 8., 8., None).into(),
            &nop,
        )
        .into(),
        Sphere::new(
            Vec3::new(-0.8, 0.6, 0.),
            0.6,
            Metal::new(SolidColor::new(0.9, 0.8, 0.7).into(), None, 0.2).into(),
        )
        .into(),
        Sphere::new(
            Vec3::new(0.8, 0.6, 0.),
            0.6,
            Dielectric::new(SolidColor::new(1., 1., 1.).into(), None, 1.5, 0.).into(),
        )
        .into(),
        // Rough glass as well as smooth, so the bit-identical invariant covers
        // the microfacet dielectric arm and not only the Dirac one -- they are
        // separate branches with separate sampler draws.
        Sphere::new(
            Vec3::new(1.9, 0.45, 0.6),
            0.45,
            Dielectric::new(SolidColor::new(1., 1., 1.).into(), None, 1.5, 0.35).into(),
        )
        .into(),
        Triangle::new(
            Vec3::new(-0.8, 0., -1.5),
            Vec3::new(0.8, 0., -1.5),
            Vec3::new(0., 1.6, -1.5),
            Blend::new(
                Lambertian::new(SolidColor::new(0.8, 0.1, 0.1).into(), None).into(),
                Lambertian::new(SolidColor::new(0.1, 0.1, 0.8).into(), None).into(),
                0.5,
            )
            .into(),
            &nop,
        )
        .into(),
    ];

    Scene {
        world: Bvh::new(world).into(),
        camera: CameraConfig {
            vertical_fov_degrees: 45.,
            aperture_size: 0.,
            look_from: Vec3::new(0., 1.2, 4.),
            look_at: Vec3::new(0., 0.8, 0.),
            up: Vec3::new(0., 1., 0.),
        },
        background_color: Vec3::new(0.1, 0.15, 0.25),
        render_config: test_render_config(),
    }
}

/// The other end of the range: nothing but untextured triangles, which is what
/// `identity_prim_refs` exists for.
fn triangles_only_scene() -> Scene {
    let nop = NopTransformer();
    let yellow = Lambertian::new(SolidColor::new(1., 1., 0.).into(), None);

    let mut world: Vec<Hittables> = Vec::new();
    for i in 0..8 {
        let x = i as f64 - 4.;
        world.push(
            Triangle::new(
                Vec3::new(x, -0.5, 0.),
                Vec3::new(x + 1., -0.5, 0.),
                Vec3::new(x + 0.5, 0.5, 0.),
                yellow.clone().into(),
                &nop,
            )
            .into(),
        );
    }
    world.push(
        Triangle::new(
            Vec3::new(-2., 4., 2.),
            Vec3::new(2., 4., 2.),
            Vec3::new(0., 4., -2.),
            DiffuseLight::new(10., 10., 10., None).into(),
            &nop,
        )
        .into(),
    );

    Scene {
        world: Bvh::new(world).into(),
        camera: CameraConfig {
            vertical_fov_degrees: 30.,
            aperture_size: 0.,
            look_from: Vec3::new(0., 0., 6.),
            look_at: Vec3::new(0., 0., 0.),
            up: Vec3::new(0., 1., 0.),
        },
        background_color: Vec3::new(0.2, 0.3, 0.5),
        render_config: test_render_config(),
    }
}

#[test]
fn a_scene_using_everything_strips_nothing() {
    let data = flatten_scene(&every_feature_scene());

    assert_eq!(1, data.lights.len());
    assert_eq!(
        Specialisation::unspecialised(1),
        Specialisation::from_scene_data(&data)
    );
}

#[test]
fn an_all_triangle_scene_strips_the_other_primitives() {
    let data = flatten_scene(&triangles_only_scene());
    let specialisation = Specialisation::from_scene_data(&data);

    assert!(specialisation.identity_prim_refs);
    assert!(!specialisation.has_spheres);
    assert!(!specialisation.has_quads);
    assert!(!specialisation.has_blends);
    assert!(!specialisation.has_metal);
    assert!(!specialisation.has_dielectrics);
    assert!(!specialisation.has_rough_dielectrics);
    assert!(!specialisation.has_textures);
    assert!(!specialisation.has_normal_maps);
    assert_eq!(1, specialisation.light_count);
}

/// On a scene that uses everything, the derived constants *are* the
/// unspecialised ones, so the two pipelines are the same shader and the images
/// have to match. Any difference means a flag was derived from the wrong thing.
#[test]
fn specialised_matches_unspecialised_when_nothing_is_stripped() {
    assert_renders_match(every_feature_scene);
}

/// The same comparison where the flags genuinely differ: the all-triangle scene
/// strips both other primitive arms, the blend walk, three material arms, both
/// texture fetches and the `prim_refs` load. Every one of those is a branch the
/// scene could never have taken, so removing them must leave the sample stream
/// alone -- not approximately, exactly.
///
/// This is the one that can catch a specialisation that changes what the
/// surviving path computes: an inverted gate, a fallback arm reading the wrong
/// primitive buffer, an identity map that is not the identity.
#[test]
fn stripping_branches_leaves_the_image_alone() {
    assert_renders_match(triangles_only_scene);
}

/// Renders `scene` twice -- once compiled against what the scene says, once
/// with every branch left in -- and insists the two agree bit for bit.
fn assert_renders_match(scene: fn() -> Scene) {
    let (device, queue) = get_wgpu_device_and_queue();
    let light_count = flatten_scene(&scene()).lights.len() as u32;

    let specialised = render_pixels(scene(), None, device, queue);
    let everything_on = render_pixels(
        scene(),
        Some(Specialisation::unspecialised(light_count)),
        device,
        queue,
    );

    assert_eq!(specialised.len(), everything_on.len());
    // Compared as bit patterns and reported by index: the buffers are twelve
    // thousand floats, and an assert_eq! on the pair would print all of them.
    let differing = specialised
        .iter()
        .zip(everything_on.iter())
        .position(|(a, b)| a.to_bits() != b.to_bits());
    if let Some(i) = differing {
        panic!(
            "specialised and unspecialised renders differ from component {} on \
             ({} against {})",
            i, specialised[i], everything_on[i]
        );
    }
}

fn render_pixels(
    scene: Scene,
    specialisation: Option<Specialisation>,
    device: &'static wgpu::Device,
    queue: &'static wgpu::Queue,
) -> Vec<f32> {
    let width = scene.render_config.width as u32;
    let height = scene.render_config.height as u32;

    let (progress_sender, progress_receiver) = channel();
    let (_camera_sender, camera_receiver) = channel();
    let (_abort_sender, abort_receiver) = channel();

    let mut renderer = Renderer::with_specialisation(scene, device, queue, specialisation).unwrap();
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

    let size = (width * height * 16) as u64;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Specialisation Test Staging Buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, size);
    queue.submit(Some(encoder.finish()));

    get_result_from_buffer::<f32>(device, &staging_buffer)
}
