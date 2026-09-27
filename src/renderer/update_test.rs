//! What a [`SceneUpdate`] does to a running render.
//!
//! The strongest check is the first: an update from one scene to another has
//! to render, bit for bit, what a renderer built for the second one would.
//! That holds only at one dispatch per render -- the Welford merge across
//! batches rounds differently for different groupings, and the renderer tunes
//! the grouping from wall-clock time -- so every render here is one sample in
//! one batch, as in `specialisation_test`.
//!
//! Every update below changes the seed as well as the part under test. A
//! restart otherwise moves `restart_index` on by one, which is the point of it
//! during a drag and would make the two images differ by design.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;

use crate::camera::CameraConfig;
use crate::geo::transformation::{NopTransformer, RotationY, Transformations, Translation};
use crate::geo::vec3::Vec3;
use crate::hittable::{Bvh, Hittables, Quad, Sphere, Triangle};
use crate::material::texture::SolidColor;
use crate::material::{DiffuseLight, Lambertian, Metal};
use crate::post::{DenoisePostProcessor, PostProcessors, SaturationPostProcessor};
use crate::renderer::{RenderConfig, RenderProgress, Renderer, Scene, SceneUpdate};
use crate::util::wgpu_util::{get_result_from_buffer, get_wgpu_device_and_queue};

fn one_sample() -> RenderConfig {
    RenderConfig {
        width: 64,
        height: 48,
        samples_per_pixel: 1,
        samples_per_batch: 1,
        ..Default::default()
    }
}

/// A mesh big enough to be kept apart as a subtree: a sphere of
/// `2 * rings * rings` triangles.
fn mesh(rings: usize, centre: Vec3, radius: f64) -> Bvh {
    let point = |i: usize, j: usize| {
        let theta = std::f64::consts::PI * i as f64 / rings as f64;
        let phi = 2. * std::f64::consts::PI * j as f64 / rings as f64;
        centre
            + Vec3::new(
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            ) * radius
    };
    let mat = Metal::new(SolidColor::new(0.8, 0.7, 0.6).into(), None, 0.3);
    let mut tris: Vec<Hittables> = Vec::new();
    for i in 0..rings {
        for j in 0..rings {
            let (a, b, c, d) = (
                point(i, j),
                point(i + 1, j),
                point(i + 1, j + 1),
                point(i, j + 1),
            );
            tris.push(Triangle::new(a, b, c, mat.clone().into(), &NopTransformer()).into());
            tris.push(Triangle::new(a, c, d, mat.clone().into(), &NopTransformer()).into());
        }
    }
    Bvh::new(tris)
}

fn world(albedo: f64, mesh: Option<Bvh>) -> Hittables {
    let nop = NopTransformer();
    let mut world: Vec<Hittables> = vec![
        Quad::new(
            Vec3::new(-4., 0., -4.),
            Vec3::new(8., 0., 0.),
            Vec3::new(0., 0., 8.),
            Lambertian::new(SolidColor::new(0.7, 0.7, 0.7).into(), None).into(),
            &nop,
        )
        .into(),
        Sphere::new(
            Vec3::new(-1., 0.6, 0.),
            0.6,
            Lambertian::new(SolidColor::new(albedo, 0.2, 0.2).into(), None).into(),
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
    ];
    world.extend(mesh.map(Hittables::from));
    Bvh::new(world).into()
}

fn camera(x: f64) -> CameraConfig {
    CameraConfig {
        vertical_fov_degrees: 45.,
        aperture_size: 0.,
        look_from: Vec3::new(x, 1.2, 4.),
        look_at: Vec3::new(0., 0.8, 0.),
        up: Vec3::new(0., 1., 0.),
    }
}

fn scene() -> Scene {
    Scene {
        world: world(0.8, None),
        camera: camera(0.),
        background_color: Vec3::new(0.1, 0.15, 0.25),
        render_config: one_sample(),
        post_processors: vec![],
    }
}

fn reseeded(seed: u32) -> RenderConfig {
    RenderConfig {
        seed,
        ..one_sample()
    }
}

fn read(progress: &RenderProgress) -> Vec<f32> {
    let (device, queue) = get_wgpu_device_and_queue();
    let size = progress.output_buffer.size();
    assert_eq!(
        size,
        (progress.width * progress.height) as u64 * 16,
        "the published buffer is not the size the progress says"
    );
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&progress.output_buffer, 0, &staging, 0, size);
    queue.submit(Some(encoder.finish()));
    get_result_from_buffer::<f32>(device, &staging)
}

fn fresh(scene: Scene) -> Vec<f32> {
    let (device, queue) = get_wgpu_device_and_queue();
    let (tx, rx) = channel();
    let (_u, updates) = channel();
    let (_a, abort) = channel();
    Renderer::new(scene, device, queue)
        .unwrap()
        .render(&tx, &updates, &abort, false)
        .unwrap();
    drop(tx);
    read(&rx.into_iter().last().unwrap())
}

/// A render running on its own thread, idling once it is done, as an
/// interactive caller runs one.
struct Live {
    progress: Receiver<RenderProgress>,
    updates: Sender<SceneUpdate>,
    handle: thread::JoinHandle<Renderer<'static>>,
}

impl Live {
    fn start(scene: Scene) -> Live {
        let (device, queue) = get_wgpu_device_and_queue();
        let mut renderer = Renderer::new(scene, device, queue).unwrap();
        let (tx, progress) = channel();
        let (updates, rx) = channel();
        let handle = thread::spawn(move || {
            let (_a, abort) = channel();
            renderer.render(&tx, &rx, &abort, true).unwrap();
            renderer
        });
        Live {
            progress,
            updates,
            handle,
        }
    }

    /// Every progress report up to and including the first one that is done.
    fn until_done(&self) -> Vec<RenderProgress> {
        let mut reports = Vec::new();
        loop {
            let p = self.progress.recv().unwrap();
            let done = p.progress >= 1.;
            reports.push(p);
            if done {
                return reports;
            }
        }
    }

    fn send(&self, update: SceneUpdate) {
        self.updates.send(update).unwrap();
    }

    /// Nothing more is reported within a generous wait.
    fn is_quiet(&self) -> bool {
        self.progress
            .recv_timeout(std::time::Duration::from_millis(300))
            .is_err()
    }

    fn stop(self) -> Renderer<'static> {
        drop(self.updates);
        self.handle.join().unwrap()
    }
}

fn assert_bit_identical(name: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{}: the images differ in size", name);
    if let Some(i) = a
        .iter()
        .zip(b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
    {
        panic!(
            "{}: the update renders differently from a fresh build from component {} on \
             ({} against {})",
            name, i, a[i], b[i]
        );
    }
}

/// Renders `from`, updates it with `update`, and holds the result to what a
/// renderer built for `to` renders.
fn assert_update_matches(name: &str, from: Scene, update: SceneUpdate, to: Scene) {
    let expected = fresh(to);
    let live = Live::start(from);
    live.until_done();
    live.send(update);
    let got = read(live.until_done().last().unwrap());
    live.stop();
    assert_bit_identical(name, &got, &expected);
}

fn update(f: impl FnOnce(&mut SceneUpdate)) -> SceneUpdate {
    let mut u = SceneUpdate::default();
    f(&mut u);
    u
}

#[test]
fn a_camera_update_renders_what_a_fresh_build_does() {
    assert_update_matches(
        "camera",
        scene(),
        update(|u| {
            u.camera = Some(camera(1.5));
            u.render_config = Some(reseeded(7));
        }),
        Scene {
            camera: camera(1.5),
            render_config: reseeded(7),
            ..scene()
        },
    );
}

#[test]
fn a_background_update_renders_what_a_fresh_build_does() {
    let background = Vec3::new(0.6, 0.3, 0.1);
    assert_update_matches(
        "background",
        scene(),
        update(|u| {
            u.background_color = Some(background);
            u.render_config = Some(reseeded(7));
        }),
        Scene {
            background_color: background,
            render_config: reseeded(7),
            ..scene()
        },
    );
}

#[test]
fn a_render_config_update_renders_what_a_fresh_build_does() {
    let configs = [
        (
            "max_depth",
            RenderConfig {
                max_depth: 3,
                ..reseeded(7)
            },
        ),
        (
            "low_discrepancy",
            RenderConfig {
                low_discrepancy: false,
                ..reseeded(7)
            },
        ),
        ("seed", reseeded(7)),
        (
            "size",
            RenderConfig {
                width: 80,
                height: 30,
                ..reseeded(7)
            },
        ),
    ];
    for (name, config) in configs {
        assert_update_matches(
            name,
            scene(),
            update(|u| u.render_config = Some(config.clone())),
            Scene {
                render_config: config,
                ..scene()
            },
        );
    }
}

#[test]
fn a_post_processor_update_renders_what_a_fresh_build_does() {
    let (device, _) = get_wgpu_device_and_queue();
    type Chain = fn(&wgpu::Device) -> Vec<PostProcessors>;
    let chains: [(&str, Chain); 2] = [
        ("saturation", |device| {
            vec![SaturationPostProcessor::new(-0.5, device).unwrap().into()]
        }),
        // Reads the guide, which is only traced on a restart.
        ("denoise", |device| {
            vec![
                DenoisePostProcessor::new(1., None, None, device)
                    .unwrap()
                    .into(),
            ]
        }),
    ];
    for (name, chain) in chains {
        assert_update_matches(
            name,
            scene(),
            update(|u| {
                u.post_processors = Some(chain(device));
                u.render_config = Some(reseeded(7));
            }),
            Scene {
                post_processors: chain(device),
                render_config: reseeded(7),
                ..scene()
            },
        );
    }
}

#[test]
fn a_world_update_renders_what_a_fresh_build_does() {
    assert_update_matches(
        "world",
        scene(),
        update(|u| {
            u.world = Some(world(0.1, None));
            u.render_config = Some(reseeded(7));
        }),
        Scene {
            world: world(0.1, None),
            render_config: reseeded(7),
            ..scene()
        },
    );
}

/// The case the arena exists for: a mesh kept apart as a subtree, moved.
/// The update writes the moved mesh over its old segment; the fresh build
/// lays it out from nothing.
#[test]
fn a_moved_subtree_renders_what_a_fresh_build_does() {
    let m = mesh(72, Vec3::new(1., 0.8, 0.), 0.7);
    assert!(m.primitive_count() >= crate::hittable::SUBTREE_MIN_PRIMS);
    let moved = || {
        m.transformed(Transformations::new(vec![
            Box::new(RotationY::new(30.)),
            Box::new(Translation::new(Vec3::new(-0.3, 0.1, 0.2))),
        ]))
    };
    let with_mesh = |mesh: Bvh| Scene {
        world: world(0.8, Some(mesh)),
        ..scene()
    };
    assert_update_matches(
        "moved subtree",
        with_mesh(m.clone()),
        update(|u| {
            u.world = Some(world(0.8, Some(moved())));
            u.render_config = Some(reseeded(7));
        }),
        Scene {
            render_config: reseeded(7),
            ..with_mesh(moved())
        },
    );
}

/// A new subtree lands in whatever the arena has free, which is not where a
/// fresh build puts it; the image must not care.
#[test]
fn a_replaced_subtree_renders_what_a_fresh_build_does() {
    let a = mesh(72, Vec3::new(1., 0.8, 0.), 0.7);
    let b = || mesh(70, Vec3::new(0.8, 0.9, 0.3), 0.6);
    assert_update_matches(
        "replaced subtree",
        Scene {
            world: world(0.8, Some(a)),
            ..scene()
        },
        update(|u| {
            u.world = Some(world(0.8, Some(b())));
            u.render_config = Some(reseeded(7));
        }),
        Scene {
            world: world(0.8, Some(b())),
            render_config: reseeded(7),
            ..scene()
        },
    );
}

/// A tree rebuilt from a bake has the bake's source and none of its leaf
/// order, so it must not be written over the bake's segment with the bake's
/// material indices. Two materials, so a mix-up shows.
#[test]
fn a_rebuilt_subtree_renders_what_a_fresh_build_does() {
    let two_materials = {
        let m = mesh(72, Vec3::new(1., 0.8, 0.), 0.7);
        let red = Lambertian::new(SolidColor::new(0.9, 0.1, 0.1).into(), None);
        let blue = Lambertian::new(SolidColor::new(0.1, 0.1, 0.9).into(), None);
        Bvh::new(
            m.prims()
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let Hittables::Triangle(t) = p else {
                        unreachable!()
                    };
                    let mat = if (i / 64) % 2 == 0 { &red } else { &blue };
                    let mut t = t.clone();
                    t.mat = mat.clone().into();
                    t.into()
                })
                .collect(),
        )
    };
    let baked = two_materials.transformed(RotationY::new(40.));
    let rebuilt = baked.rebuilt();
    assert_update_matches(
        "rebuilt subtree",
        Scene {
            world: world(0.8, Some(baked)),
            ..scene()
        },
        update(|u| {
            u.world = Some(world(0.8, Some(rebuilt.clone())));
            u.render_config = Some(reseeded(7));
        }),
        Scene {
            world: world(0.8, Some(rebuilt)),
            render_config: reseeded(7),
            ..scene()
        },
    );
}

/// Raising the sample count on a finished render keeps what it has and traces
/// only the rest, and the result is what asking for the higher count from the
/// start renders: the same two batches of four, at the same seed.
#[test]
fn raising_the_sample_count_continues() {
    let config = |spp| RenderConfig {
        samples_per_pixel: spp,
        samples_per_batch: 4,
        ..one_sample()
    };
    let expected = fresh(Scene {
        render_config: config(8),
        ..scene()
    });

    let live = Live::start(Scene {
        render_config: config(4),
        ..scene()
    });
    live.until_done();
    live.send(update(|u| u.render_config = Some(config(8))));
    let reports = live.until_done();
    live.stop();

    assert_eq!(
        reports.len(),
        1,
        "a continued render restarted: it took {} batches to finish",
        reports.len()
    );
    assert_bit_identical("continued", &read(&reports[0]), &expected);
}

/// Lowering the sample count below what is done leaves nothing to trace, and a
/// finished render still owes the caller its last-batch chain. Exactly once:
/// before the change it went idle without it.
#[test]
fn lowering_the_sample_count_runs_the_chain_once() {
    let (device, _) = get_wgpu_device_and_queue();
    let config = |spp| RenderConfig {
        samples_per_pixel: spp,
        samples_per_batch: 8,
        ..one_sample()
    };
    let live = Live::start(Scene {
        render_config: config(8),
        post_processors: vec![SaturationPostProcessor::new(-1., device).unwrap().into()],
        ..scene()
    });
    live.until_done();
    live.send(update(|u| u.render_config = Some(config(4))));
    let republished = live.until_done();
    assert!(live.is_quiet(), "the render did more than republish");
    let renderer = live.stop();

    assert_eq!(1, republished.len());
    assert_eq!(1., republished[0].progress);
    assert_eq!(
        2, renderer.chain_runs,
        "the chain should run once for the render and once for the republish"
    );

    // Saturation at -1 greys every pixel, so the republished image is the
    // processed one rather than the raw accumulator.
    let pixels = read(&republished[0]);
    for px in pixels.chunks(4) {
        assert!(
            (px[0] - px[1]).abs() < 1e-5 && (px[1] - px[2]).abs() < 1e-5,
            "the republished image is not the chain's output: {:?}",
            px
        );
    }
}

/// Lowering it to a count not yet reached is just a shorter render.
#[test]
fn lowering_the_sample_count_above_what_is_done_changes_nothing_else() {
    let (device, queue) = get_wgpu_device_and_queue();
    let config = |spp| RenderConfig {
        samples_per_pixel: spp,
        ..one_sample()
    };
    let mut renderer = Renderer::new(
        Scene {
            render_config: config(50),
            ..scene()
        },
        device,
        queue,
    )
    .unwrap();
    let outcome = renderer.apply(update(|u| u.render_config = Some(config(20))), 10);
    assert!(!outcome.restart && !outcome.republish && !outcome.reset_cost);
}

/// The table in [`SceneUpdate`]'s documentation, row by row.
#[test]
fn each_part_lands_on_its_outcome() {
    let (device, queue) = get_wgpu_device_and_queue();
    let base = RenderConfig::default();
    let config = |f: fn(&mut RenderConfig)| {
        let mut c = RenderConfig::default();
        f(&mut c);
        c
    };
    // (name, update, restart, reset_cost, republish), applied to a render
    // that has done 10 of its 50 samples.
    let rows: Vec<(&str, SceneUpdate, bool, bool, bool)> = vec![
        (
            "samples_per_batch",
            update(|u| u.render_config = Some(config(|c| c.samples_per_batch = 9))),
            false,
            false,
            false,
        ),
        (
            "preview",
            update(|u| u.render_config = Some(config(|c| c.preview = true))),
            false,
            false,
            false,
        ),
        (
            "samples_per_pixel raised",
            update(|u| u.render_config = Some(config(|c| c.samples_per_pixel = 500))),
            false,
            false,
            false,
        ),
        (
            "min_samples_per_pixel",
            update(|u| u.render_config = Some(config(|c| c.min_samples_per_pixel = 3))),
            false,
            false,
            false,
        ),
        (
            "variance_threshold",
            update(|u| u.render_config = Some(config(|c| c.variance_threshold = 0.5))),
            false,
            false,
            false,
        ),
        (
            "samples_per_pixel lowered to what is done",
            update(|u| u.render_config = Some(config(|c| c.samples_per_pixel = 10))),
            false,
            false,
            true,
        ),
        (
            "post_processors without the guide",
            update(|u| {
                u.post_processors = Some(vec![
                    SaturationPostProcessor::new(0.5, device).unwrap().into(),
                ])
            }),
            false,
            false,
            true,
        ),
        (
            "post_processors with the guide",
            update(|u| {
                u.post_processors = Some(vec![
                    DenoisePostProcessor::new(1., None, None, device)
                        .unwrap()
                        .into(),
                ])
            }),
            true,
            true,
            true,
        ),
        (
            "camera",
            update(|u| u.camera = Some(camera(2.))),
            true,
            false,
            false,
        ),
        (
            "background_color",
            update(|u| u.background_color = Some(Vec3::new(1., 0., 0.))),
            true,
            false,
            false,
        ),
        (
            "max_depth",
            update(|u| u.render_config = Some(config(|c| c.max_depth = 2))),
            true,
            true,
            false,
        ),
        (
            "low_discrepancy",
            update(|u| u.render_config = Some(config(|c| c.low_discrepancy = false))),
            true,
            true,
            false,
        ),
        (
            "seed",
            update(|u| u.render_config = Some(config(|c| c.seed = 3))),
            true,
            false,
            false,
        ),
        (
            "size",
            update(|u| u.render_config = Some(config(|c| c.width = 100))),
            true,
            true,
            false,
        ),
        (
            "world",
            update(|u| u.world = Some(world(0.3, None))),
            true,
            true,
            false,
        ),
    ];
    for (name, u, restart, reset_cost, republish) in rows {
        let mut renderer = Renderer::new(
            Scene {
                render_config: base.clone(),
                ..scene()
            },
            device,
            queue,
        )
        .unwrap();
        let o = renderer.apply(u, 10);
        assert_eq!(
            (restart, reset_cost, republish),
            (o.restart, o.reset_cost, o.republish),
            "{}: (restart, reset_cost, republish)",
            name
        );
    }
}

/// The light count is a uniform, so a light added to a lit scene takes the
/// pipeline it already has.
#[test]
fn adding_a_light_to_a_lit_scene_compiles_nothing() {
    let (device, queue) = get_wgpu_device_and_queue();
    let mut renderer = Renderer::new(scene(), device, queue).unwrap();
    let key = renderer.pipeline_key.clone();
    let more_lights = {
        let mut w = vec![world(0.8, None)];
        w.push(
            Sphere::new(
                Vec3::new(2., 2., 0.),
                0.3,
                DiffuseLight::new(4., 4., 4., None).into(),
                &NopTransformer(),
            )
            .into(),
        );
        Bvh::new(w).into()
    };
    let o = renderer.apply(update(|u| u.world = Some(more_lights)), 0);
    assert_eq!(2, renderer.render_config.light_count);
    assert_eq!(
        key, renderer.pipeline_key,
        "a new light recompiled the tracer"
    );
    assert!(o.reset_cost, "a world change always resets the cost model");
}

/// A world with no light at all, under a black background, is refused by
/// `Renderer::new` but rendered by an update -- black.
#[test]
fn an_update_to_a_black_scene_renders_black() {
    let dark: Hittables = Bvh::new(vec![
        Sphere::new(
            Vec3::new(0., 0.6, 0.),
            0.6,
            Lambertian::new(SolidColor::new(0.8, 0.8, 0.8).into(), None).into(),
            &NopTransformer(),
        )
        .into(),
    ])
    .into();

    let live = Live::start(scene());
    live.until_done();
    live.send(update(|u| {
        u.world = Some(dark);
        u.background_color = Some(Vec3::new(0., 0., 0.));
    }));
    let pixels = read(live.until_done().last().unwrap());
    live.stop();

    assert!(
        pixels.chunks(4).all(|px| px[..3] == [0., 0., 0.]),
        "a scene with nothing to light it rendered something"
    );
}

/// The published buffer, and the size it is published with, follow a resize.
#[test]
fn a_resize_publishes_the_new_size() {
    let live = Live::start(scene());
    let before = live.until_done().pop().unwrap();
    live.send(update(|u| {
        u.render_config = Some(RenderConfig {
            width: 33,
            height: 17,
            ..one_sample()
        })
    }));
    let after = live.until_done().pop().unwrap();
    live.stop();

    assert_eq!((64, 48), (before.width, before.height));
    assert_eq!((33, 17), (after.width, after.height));
    assert_eq!(33 * 17 * 16, after.output_buffer.size());
}
