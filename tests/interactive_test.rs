use solstrale::camera::CameraConfig;
use solstrale::geo::transformation::NopTransformer;
use solstrale::geo::vec3::Vec3;
use solstrale::hittable::{Bvh, Sphere};
use solstrale::material::DiffuseLight;
use solstrale::ray_trace;
use solstrale::renderer::{RenderConfig, Scene};
use solstrale::util::wgpu_util::get_wgpu_device_and_queue;
use std::sync::mpsc::channel;
use std::thread;

#[test]
fn test_interactive_camera_restart() {
    let (device, queue) = get_wgpu_device_and_queue();
    let render_config = RenderConfig {
        width: 10,
        height: 10,
        samples_per_pixel: 100,
        ..Default::default()
    };
    let mut world = Vec::new();
    world.push(
        Sphere::new(
            Vec3::new(0., 10., 0.),
            1.,
            DiffuseLight::new(1., 1., 1., None).into(),
            &NopTransformer(),
        )
        .into(),
    );

    let scene = Scene {
        world: Bvh::new(world).into(),
        camera: CameraConfig {
            look_from: Vec3::new(0., 0., 1.),
            ..Default::default()
        },
        background_color: Vec3::new(0., 0., 0.),
        render_config,
        post_processors: vec![],
    };

    let (output_sender, output_receiver) = channel();
    let (camera_config_sender, camera_config_receiver) = channel();
    let (abort_sender, abort_receiver) = channel();

    thread::spawn(move || {
        ray_trace(
            scene,
            &output_sender,
            &camera_config_receiver,
            &abort_receiver,
            device,
            queue,
            true,
        )
        .unwrap();
    });

    // Wait for some progress
    let mut first_progress = 0.0;
    for progress in &output_receiver {
        if progress.progress > 0.1 {
            first_progress = progress.progress;
            break;
        }
    }
    assert!(first_progress > 0.1);

    // Update camera
    camera_config_sender
        .send(
            CameraConfig {
                look_from: Vec3::new(0., 0., 2.),
                ..Default::default()
            }
            .into(),
        )
        .unwrap();

    // Verify progress restarts
    let mut restarted = false;
    for progress in &output_receiver {
        if progress.progress < first_progress {
            restarted = true;
            break;
        }
    }
    assert!(restarted);

    // Abort
    abort_sender.send(true).unwrap();
}

/// Diagnostic: what one frame of a camera drag costs, and what the two things
/// this crate does only for a live viewer cost inside it.
///
/// `cargo test --release --test interactive_test -- --ignored --nocapture`
///
/// Nothing else measures this. Every render benchmark traces one accumulation
/// to the end, so it sees exactly one restart against thousands of sample
/// paths -- under 0.2% on `render/test_scene_800x600_64spp`, which is the same
/// scene and the same size as this. A drag is the opposite regime: every frame
/// restarts the accumulation at one sample per pixel, so everything charged per
/// restart is charged per frame.
///
/// The three arms are that difference, each read against the one above it:
///
/// * **no chain** -- no post-processor, so `writes_guide` is off and the tracer
///   is compiled without `trace_guide`. The floor.
/// * **denoise, preview off** -- the guide ray is back: one deterministic ray
///   per pixel per restart, up to `GUIDE_MAX_SPECULAR` more on a specular first
///   hit. Plus the copy into the post buffer. The filter itself never runs,
///   because a drag never reaches a last batch.
/// * **denoise, preview on** -- what a drag looks like with
///   [`RenderConfig::preview`] on: the filter runs on every frame, which is the
///   whole point of it.
///
/// Driven the way a viewport drives it -- a new camera on every frame -- so
/// every dispatch is a restart. A frame that turns out not to be one is still
/// timed, to keep the clock honest, but not recorded.
#[test]
#[ignore]
fn interactive_restart_frame_cost() {
    use crate::scenes::create_test_scene;
    use solstrale::post::DenoisePostProcessor;
    use std::time::{Duration, Instant};

    /// Never reached: the point is that the accumulation restarts long before
    /// it gets there, which is what a drag does.
    const SAMPLES_PER_PIXEL: u32 = 100_000;
    const FRAMES: usize = 60;
    const WARMUP: usize = 20;

    fn drag(name: &str, denoise: bool, preview: bool) {
        let (device, queue) = get_wgpu_device_and_queue();
        let post_processors = if denoise {
            vec![
                DenoisePostProcessor::new(1., None, None, device)
                    .unwrap()
                    .into(),
            ]
        } else {
            vec![]
        };
        let scene = Scene {
            post_processors,
            ..create_test_scene(RenderConfig {
                width: 800,
                height: 600,
                samples_per_pixel: SAMPLES_PER_PIXEL,
                preview,
                ..Default::default()
            })
        };
        let look_at = scene.camera.look_at;

        let (output_sender, output_receiver) = channel();
        let (camera_config_sender, camera_config_receiver) = channel();
        let (abort_sender, abort_receiver) = channel();

        let handle = thread::spawn(move || {
            ray_trace(
                scene,
                &output_sender,
                &camera_config_receiver,
                &abort_receiver,
                device,
                queue,
                true,
            )
            .unwrap();
        });

        // A restart's first batch is one sample, so this is what tells a frame
        // that actually restarted from one that slipped through between our
        // send and the renderer draining the channel.
        let restart_progress = 1. / SAMPLES_PER_PIXEL as f64;

        let mut frames = Vec::with_capacity(FRAMES);
        let mut last = Instant::now();
        for i in 0..WARMUP + FRAMES {
            // Orbit the scene, so every frame is a genuinely different view and
            // no two share a BVH traversal pattern.
            let angle = i as f64 * 0.02;
            camera_config_sender
                .send(
                    CameraConfig {
                        vertical_fov_degrees: 20.,
                        aperture_size: 0.1,
                        look_from: Vec3::new(angle.sin() * 8. - 5., 3., angle.cos() * 8.),
                        look_at,
                        up: Vec3::new(0., 1., 0.),
                    }
                    .into(),
                )
                .unwrap();
            let progress = output_receiver.recv().unwrap();
            let now = Instant::now();
            if i >= WARMUP && (progress.progress - restart_progress).abs() < restart_progress / 2. {
                frames.push(now - last);
            }
            last = now;
        }

        abort_sender.send(true).unwrap();
        handle.join().unwrap();

        assert!(!frames.is_empty(), "{}: no frame was a restart", name);
        frames.sort();
        let ms = |d: Duration| d.as_secs_f64() * 1000.;
        println!(
            "{:<22} {:3} restart frames   median {:7.2} ms   min {:7.2} ms   max {:7.2} ms",
            name,
            frames.len(),
            ms(frames[frames.len() / 2]),
            ms(frames[0]),
            ms(frames[frames.len() - 1]),
        );
    }

    println!("create_test_scene at 800x600, a new camera every frame:");
    drag("no chain", false, false);
    drag("denoise, preview off", true, false);
    drag("denoise, preview on", true, true);
}

/// Diagnostic: what one frame of an edit costs, against what a camera frame
/// does, which is the floor. Each arm makes one edit per frame to a live
/// 800x600 render of `create_test_scene` with a mesh added to it, and times it
/// from building the edit to the image of it arriving.
///
/// `cargo test --release --test interactive_test interactive_edit_frame_cost -- --ignored --nocapture`
///
/// * **camera** -- a uniform write and a restart.
/// * **background** -- the same, through a different field.
/// * **albedo** -- a sphere's colour: a new world, rebuilt around the mesh,
///   which it holds as the same subtree.
/// * **translate** -- the mesh moved: `Bvh::transformed`, a new world, and the
///   mesh's segment rewritten in place.
/// * **rebuild** -- the translate edit the way it had to be made before
///   `SceneUpdate`: a new renderer for every frame.
///
/// The mesh is `SOLSTRALE_BENCH_OBJ` when that names one, and a 10k-triangle
/// sphere otherwise, which is just past `SUBTREE_MIN_PRIMS`.
#[test]
#[ignore]
fn interactive_edit_frame_cost() {
    use crate::scenes::{create_test_scene, tessellated_sphere};
    use solstrale::geo::transformation::{Scale, Transformations, Translation};
    use solstrale::hittable::Hittables;
    use solstrale::loader::Loader;
    use solstrale::loader::obj::Obj;
    use solstrale::material::texture::SolidColor;
    use solstrale::material::{Lambertian, Metal};
    use solstrale::renderer::SceneUpdate;
    use std::time::{Duration, Instant};

    const FRAMES: usize = 60;
    const WARMUP: usize = 10;

    let (device, queue) = get_wgpu_device_and_queue();
    let config = || RenderConfig {
        width: 800,
        height: 600,
        samples_per_pixel: 1,
        ..Default::default()
    };
    let base = create_test_scene(config());
    let room = base.world.clone();

    // In object space, with the scale that fits it in the test scene: the
    // transformation `Bvh::transformed` takes is absolute, so every move
    // carries the scale along.
    let (mesh, scale) = match std::env::var("SOLSTRALE_BENCH_OBJ") {
        Ok(path) => {
            let path = std::path::PathBuf::from(path);
            let dir = format!("{}/", path.parent().unwrap().display());
            let file = path.file_name().unwrap().to_str().unwrap().to_string();
            let mesh = Obj::new(&dir, &file).load(&NopTransformer(), None).unwrap();
            let b = solstrale::hittable::Hittable::bounding_box(&mesh).clone();
            let size = (b.x.max - b.x.min)
                .max(b.y.max - b.y.min)
                .max(b.z.max - b.z.min);
            (mesh, 1.5 / size)
        }
        Err(_) => (
            Bvh::new(tessellated_sphere(
                Vec3::new(0., 0., 0.),
                0.6,
                72,
                &Metal::new(SolidColor::new(0.9, 0.8, 0.6).into(), None, 0.2).into(),
            )),
            1.,
        ),
    };
    println!(
        "create_test_scene at 800x600 with a {}-triangle mesh, one edit every frame:",
        mesh.primitive_count()
    );

    let world = |albedo: f64, mesh: Bvh| -> Hittables {
        Bvh::new(vec![
            room.clone(),
            mesh.into(),
            Sphere::new(
                Vec3::new(1.5, 0.3, 1.),
                0.3,
                Lambertian::new(SolidColor::new(albedo, 0.3, 0.2).into(), None).into(),
                &NopTransformer(),
            )
            .into(),
        ])
        .into()
    };
    let moved = |i: usize| {
        mesh.transformed(Transformations::new(vec![
            Box::new(Scale::new(scale)),
            Box::new(Translation::new(Vec3::new(
                1.5 + (i as f64 * 0.1).sin() * 0.5,
                0.6,
                -1.,
            ))),
        ]))
    };
    let resting = moved(0);

    let report = |name: &str, mut frames: Vec<Duration>| {
        frames.sort();
        let ms = |d: Duration| d.as_secs_f64() * 1000.;
        println!(
            "{:<12} median {:8.2} ms   min {:8.2} ms   max {:8.2} ms",
            name,
            ms(frames[frames.len() / 2]),
            ms(frames[0]),
            ms(frames[frames.len() - 1]),
        );
    };

    type Edit<'e> = Box<dyn Fn(usize) -> SceneUpdate + 'e>;
    let arms: Vec<(&str, Edit)> = vec![
        (
            "camera",
            Box::new(|i| {
                let angle = i as f64 * 0.02;
                CameraConfig {
                    vertical_fov_degrees: 20.,
                    aperture_size: 0.1,
                    look_from: Vec3::new(angle.sin() * 8. - 5., 3., angle.cos() * 8.),
                    look_at: base.camera.look_at,
                    up: Vec3::new(0., 1., 0.),
                }
                .into()
            }),
        ),
        (
            "background",
            Box::new(|i| {
                let mut u = SceneUpdate::default();
                u.background_color = Some(Vec3::new(0.2, 0.3, 0.3 + (i % 10) as f64 * 0.02));
                u
            }),
        ),
        (
            "albedo",
            Box::new(|i| {
                let mut u = SceneUpdate::default();
                u.world = Some(world((i % 10) as f64 * 0.1, resting.clone()));
                u
            }),
        ),
        (
            "translate",
            Box::new(|i| {
                let mut u = SceneUpdate::default();
                u.world = Some(world(0.5, moved(i)));
                u
            }),
        ),
    ];

    for (name, edit) in arms {
        let (output_sender, output_receiver) = channel();
        let (update_sender, update_receiver) = channel();
        let (_abort_sender, abort_receiver) = channel();
        let scene = Scene {
            world: world(0.5, resting.clone()),
            ..create_test_scene(config())
        };
        let handle = thread::spawn(move || {
            ray_trace(
                scene,
                &output_sender,
                &update_receiver,
                &abort_receiver,
                device,
                queue,
                true,
            )
            .unwrap();
        });
        output_receiver.recv().unwrap();

        let mut frames = Vec::with_capacity(FRAMES);
        for i in 0..WARMUP + FRAMES {
            let start = Instant::now();
            update_sender.send(edit(i)).unwrap();
            output_receiver.recv().unwrap();
            if i >= WARMUP {
                frames.push(start.elapsed());
            }
        }
        drop(update_sender);
        handle.join().unwrap();
        report(name, frames);
    }

    let mut frames = Vec::with_capacity(FRAMES);
    for i in 0..WARMUP + FRAMES {
        let start = Instant::now();
        // Not through `create_test_scene`, which decodes its texture again.
        let scene = Scene {
            world: world(0.5, moved(i)),
            camera: base.camera,
            background_color: base.background_color,
            render_config: config(),
            post_processors: vec![],
        };
        let (output_sender, output_receiver) = channel();
        let (_u, update_receiver) = channel();
        let (_a, abort_receiver) = channel();
        ray_trace(
            scene,
            &output_sender,
            &update_receiver,
            &abort_receiver,
            device,
            queue,
            false,
        )
        .unwrap();
        output_receiver.recv().unwrap();
        if i >= WARMUP {
            frames.push(start.elapsed());
        }
    }
    report("rebuild", frames);
}

mod scenes;
