use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::mpsc::channel;
use std::thread;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use derive_more::{Constructor, Display};

use crate::scenes::{
    create_many_lights_scene, create_rough_metal_scene, create_test_scene, new_bvh_test_scene,
};
use solstrale::camera::CameraConfig;
use solstrale::geo::transformation::NopTransformer;
use solstrale::geo::vec3::Vec3;
use solstrale::hittable::{Bvh, Hittables, Triangle};
use solstrale::loader::Loader;
use solstrale::loader::obj::Obj;
use solstrale::material::texture::SolidColor;
use solstrale::material::{DiffuseLight, Lambertian};
use solstrale::post::{
    BloomPostProcessor, DenoisePostProcessor, PostProcessors, SaturationPostProcessor,
};
use solstrale::ray_trace;
use solstrale::renderer::scene_flattener::flatten_scene;
use solstrale::renderer::{RenderConfig, Renderer, Scene};
use solstrale::util::tone_map::ToneMapper;
use solstrale::util::wgpu_util::{buffer_to_image, get_wgpu_device_and_queue};

#[path = "../tests/scenes.rs"]
mod scenes;

/// Builds a spatially spread pseudo-random triangle cloud.
///
/// Deliberately not the colinear strip `new_bvh_test_scene` uses: a degenerate
/// 1-D layout makes every split heuristic look identical, so it cannot show the
/// difference between a median split and SAH.
fn triangle_cloud(n: u32) -> Vec<Hittables> {
    let mat = Lambertian::new(SolidColor::new(1., 1., 0.).into(), None);
    let nop = NopTransformer();

    // Cheap deterministic LCG -- reproducible across runs and platforms.
    let mut state: u32 = 0x9E3779B9;
    let mut next = move || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state >> 8) as f64 / 16777216.0
    };

    let extent = (n as f64).cbrt() * 2.0;
    (0..n)
        .map(|_| {
            let cx = next() * extent;
            let cy = next() * extent;
            let cz = next() * extent;
            Triangle::new(
                Vec3::new(cx, cy, cz),
                Vec3::new(cx + 0.6, cy + 0.1, cz + 0.2),
                Vec3::new(cx + 0.2, cy + 0.7, cz - 0.1),
                mat.clone().into(),
                &nop,
            )
            .into()
        })
        .collect()
}

/// Wraps [`triangle_cloud`] in a renderable scene lit by a *triangle* light.
///
/// The light is the point. Every other `bvh_traversal` arm carries a sphere
/// light, so a scene with no spheres in it at all is the only one on which a
/// future `has_spheres` / `has_quads` pipeline specialisation could show
/// anything -- and, since the cloud is spatially spread rather than colinear,
/// the only one where a change to BVH quality is visible against a realistic
/// distribution.
fn triangle_cloud_scene(render_config: RenderConfig, n: u32) -> Scene {
    let extent = (n as f64).cbrt() * 2.0;
    let centre = extent / 2.;

    let mut world = triangle_cloud(n);
    world.push(
        Triangle::new(
            Vec3::new(centre - extent, extent * 1.5, centre - extent),
            Vec3::new(centre + extent, extent * 1.5, centre - extent),
            Vec3::new(centre, extent * 1.5, centre + extent),
            DiffuseLight::new(10., 10., 10., None).into(),
            &NopTransformer(),
        )
        .into(),
    );

    Scene {
        world: Bvh::new(world).into(),
        camera: CameraConfig {
            vertical_fov_degrees: 40.,
            aperture_size: 0.,
            look_from: Vec3::new(centre, centre, centre + extent * 1.5),
            look_at: Vec3::new(centre, centre, centre),
            up: Vec3::new(0., 1., 0.),
        },
        background_color: Vec3::new(0.2, 0.3, 0.5),
        render_config,
    }
}

/// Times BVH construction in isolation.
///
/// The old `bvh_benchmark` built its scene inside `iter_with_setup`'s setup
/// closure, which criterion does not time -- so BVH construction was never
/// actually measured.
pub fn bvh_build_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvh_build");
    for n in [10_000u32, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.sample_size(10);

        // What sizes MAX_TRAVERSAL_DEPTH. The GPU stack has to hold one entry
        // per level, and the build asserts against it, so the constant should
        // track a measured number rather than a guess.
        println!("bvh_build/{}: {}", n, Bvh::new(triangle_cloud(n)));

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_with_setup(|| triangle_cloud(n), |tris| black_box(Bvh::new(tris)));
        });
    }
    group.finish();
}

/// Writes a deterministic grid mesh to a temp file and returns its path.
///
/// A grid rather than a triangle soup because the loader's cost depends on vertex
/// *sharing*: every interior vertex here is used by six faces, the way a real scanned
/// mesh behaves, which is what makes transforming per vertex rather than per corner
/// worth anything. `vt` lines are emitted so `mesh.texcoords` is non-empty and the
/// loader takes its textured branch.
///
/// The file is written once per size and reused, so generation never lands inside a
/// timed closure.
fn grid_obj(faces: usize) -> &'static PathBuf {
    /// One `OnceLock` per size the benches ask for.
    static FIXTURES: [(usize, OnceLock<PathBuf>); 2] =
        [(50_000, OnceLock::new()), (250_000, OnceLock::new())];

    let cell = &FIXTURES
        .iter()
        .find(|(n, _)| *n == faces)
        .expect("no fixture slot for this size")
        .1;

    cell.get_or_init(|| {
        // k*k cells, two triangles each.
        let k = ((faces as f64 / 2.).sqrt()).ceil() as usize;
        let path = std::env::temp_dir().join(format!("solstrale_bench_grid_{}.obj", faces));

        let file = std::fs::File::create(&path).expect("failed to create bench fixture");
        let mut w = BufWriter::new(file);

        // Same cheap deterministic LCG as `triangle_cloud`, used to give the grid some
        // height so it is not a degenerate plane for the BVH build underneath.
        let mut state: u32 = 0x9E37_79B9;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f64 / 16777216.0
        };

        for z in 0..=k {
            for x in 0..=k {
                writeln!(w, "v {} {} {}", x as f64 * 0.1, next() * 2., z as f64 * 0.1)
                    .expect("failed to write bench fixture");
            }
        }
        for z in 0..=k {
            for x in 0..=k {
                writeln!(w, "vt {} {}", x as f64 / k as f64, z as f64 / k as f64)
                    .expect("failed to write bench fixture");
            }
        }
        let vert = |x: usize, z: usize| z * (k + 1) + x + 1; // OBJ indices are 1-based
        for z in 0..k {
            for x in 0..k {
                let (a, b, c, d) = (
                    vert(x, z),
                    vert(x + 1, z),
                    vert(x + 1, z + 1),
                    vert(x, z + 1),
                );
                writeln!(w, "f {}/{} {}/{} {}/{}", a, a, b, b, c, c)
                    .expect("failed to write bench fixture");
                writeln!(w, "f {}/{} {}/{} {}/{}", a, a, c, c, d, d)
                    .expect("failed to write bench fixture");
            }
        }
        w.flush().expect("failed to flush bench fixture");
        path
    })
}

/// Splits a path into the (directory, filename) pair `Obj::new` wants.
fn split_obj_path(path: &std::path::Path) -> (String, String) {
    let dir = path.parent().unwrap().to_str().unwrap();
    let file = path.file_name().unwrap().to_str().unwrap();
    (format!("{}/", dir), file.to_string())
}

fn load_options() -> tobj::LoadOptions {
    tobj::LoadOptions {
        triangulate: true,
        ..Default::default()
    }
}

/// Times OBJ loading, split so the parse is visible on its own.
///
/// `total` minus `parse`, minus the `bvh_build` group at the matching size, is what the
/// triangle-construction loop actually costs. That subtraction is the point: tobj's
/// parser is single-threaded and allocates a `String` per line, so it is the serial
/// floor no amount of rayon in the construction loop can get under.
///
/// Set `SOLSTRALE_BENCH_OBJ` to a path to measure a real mesh instead of the synthetic
/// grid, e.g. `SOLSTRALE_BENCH_OBJ=/path/to/dragon.obj cargo bench -- obj_load`.
pub fn obj_load_benchmark(c: &mut Criterion) {
    let mut inputs: Vec<(String, PathBuf)> = [50_000usize, 250_000]
        .iter()
        .map(|&n| (n.to_string(), grid_obj(n).clone()))
        .collect();

    if let Ok(path) = std::env::var("SOLSTRALE_BENCH_OBJ") {
        let path = PathBuf::from(path);
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("external")
            .to_string();
        inputs.push((name, path));
    }

    let mut group = c.benchmark_group("obj_load");
    group.sample_size(10);

    for (name, path) in &inputs {
        let (dir, file) = split_obj_path(path);

        group.bench_with_input(BenchmarkId::new("parse", name), path, |b, path| {
            // `iter_with_large_drop` throughout: freeing ~80 MB of triangles and their
            // Arcs is real work, but it is not load time. (`bvh_build` above does
            // currently charge itself for that drop -- pre-existing, left alone.)
            b.iter_with_large_drop(|| tobj::load_obj(path, &load_options()).unwrap());
        });

        group.bench_with_input(
            BenchmarkId::new("total", name),
            &(dir, file),
            |b, (d, f)| {
                b.iter_with_large_drop(|| Obj::new(d, f).load(&NopTransformer(), None).unwrap());
            },
        );
    }
    group.finish();
}

/// Times the scene-graph -> flat GPU buffer conversion in isolation.
pub fn flatten_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("scene_flatten");
    for n in [10_000u32, 100_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.sample_size(10);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_with_setup(
                || Scene {
                    world: Bvh::new(triangle_cloud(n)).into(),
                    camera: create_test_scene(RenderConfig::default()).camera,
                    background_color: Vec3::new(0.2, 0.3, 0.5),
                    render_config: RenderConfig::default(),
                },
                |scene| black_box(flatten_scene(&scene)),
            );
        });
    }
    group.finish();
}

/// Renders a scene to completion and blocks until the GPU has actually finished.
///
/// `Renderer::render` never syncs -- it submits and returns -- so timing around
/// the render loop alone measures queue submission, not tracing. The `poll` is
/// what makes this a real measurement.
fn render_and_sync(scene: Scene) {
    let (device, queue) = get_wgpu_device_and_queue();

    let (output_sender, output_receiver) = channel();
    let (_abort_sender, abort_receiver) = channel();
    let (_camera_sender, camera_config_receiver) = channel();

    let handle = thread::spawn(move || {
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

    for progress in output_receiver {
        black_box(progress.progress);
    }
    handle.join().unwrap();

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

/// As [`render_and_sync`], but keeps the finished image buffer.
///
/// `readback_benchmark` needs a rendered buffer to read back, and needs it
/// built outside the timed closure.
fn render_to_buffer(scene: Scene) -> wgpu::Buffer {
    let (device, queue) = get_wgpu_device_and_queue();

    let (output_sender, output_receiver) = channel();
    let (_abort_sender, abort_receiver) = channel();
    let (_camera_sender, camera_config_receiver) = channel();

    let handle = thread::spawn(move || {
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

    let mut buffer = None;
    for progress in output_receiver {
        buffer = Some(progress.output_buffer);
    }
    handle.join().unwrap();

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    buffer.expect("render reported no progress")
}

/// End-to-end trace throughput at a realistic resolution and sample count.
///
/// This includes pipeline creation and buffer upload, which `ray_trace` does
/// internally. At this size and sample count tracing dominates them heavily.
pub fn render_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("render");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("test_scene_800x600_64spp", |b| {
        b.iter_with_setup(
            || {
                create_test_scene(RenderConfig {
                    samples_per_pixel: 64,
                    width: 800,
                    height: 600,
                    ..RenderConfig::default()
                })
            },
            render_and_sync,
        )
    });

    // The fixed per-render cost the arm above pays before it traces anything,
    // on its own: scene flatten, buffer upload and shader compilation. Fitting
    // the two bench points apart puts that cost at around 8 ms, which is worth
    // knowing exactly rather than by subtraction -- pipeline specialisation
    // would move the compile into it, and there is no measuring that against a
    // number derived from two other numbers.
    let (device, queue) = get_wgpu_device_and_queue();
    group.bench_function("renderer_setup", |b| {
        b.iter_with_setup(
            || {
                create_test_scene(RenderConfig {
                    samples_per_pixel: 64,
                    width: 800,
                    height: 600,
                    ..RenderConfig::default()
                })
            },
            |scene| black_box(Renderer::new(scene, device, queue).unwrap()),
        )
    });

    // `create_test_scene` contains no metal at all, so the first arm would show
    // nothing if the conductor lobe changed in cost. This one is five metal
    // spheres over the whole roughness range, each taking a shadow ray the
    // fuzz-sphere metal never took.
    group.bench_function("rough_metal_800x600_64spp", |b| {
        b.iter_with_setup(
            || {
                create_rough_metal_scene(RenderConfig {
                    samples_per_pixel: 64,
                    width: 800,
                    height: 600,
                    ..RenderConfig::default()
                })
            },
            render_and_sync,
        )
    });

    // 120 emitters spanning three decades of radiance: the only arm where the
    // cost of *selecting* a light is on the critical path at all. The others
    // have one to three, where the alias table is a constant multiply and the
    // binary search behind the MIS weight is two iterations.
    group.bench_function("many_lights_400x300_256spp", |b| {
        b.iter_with_setup(
            || {
                create_many_lights_scene(RenderConfig {
                    samples_per_pixel: 256,
                    width: 400,
                    height: 300,
                    ..RenderConfig::default()
                })
            },
            render_and_sync,
        )
    });

    group.finish();
}

/// Adaptive sampling on vs. effectively disabled, at the same total sample
/// budget -- isolates the win LIMITATIONS.md's "per-pixel adaptive sampling" item
/// describes from everything else `render_benchmark` already measures.
/// Disabling is done by setting `min_samples_per_pixel` above the render's
/// total sample count, so no pixel is ever eligible to be skipped, rather
/// than adding a dedicated on/off flag to `RenderConfig`.
pub fn adaptive_sampling_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_sampling");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for adaptive in [true, false] {
        group.bench_with_input(
            BenchmarkId::from_parameter(if adaptive { "adaptive" } else { "forced_off" }),
            &adaptive,
            |b, &adaptive| {
                b.iter_with_setup(
                    || {
                        create_test_scene(RenderConfig {
                            // A larger sample budget than `render_benchmark`'s:
                            // the win comes from quiet pixels converging long
                            // before noisy ones, which only shows up once
                            // there's enough headroom past convergence for it
                            // to matter.
                            samples_per_pixel: 2000,
                            width: 400,
                            height: 300,
                            min_samples_per_pixel: if adaptive { 16 } else { u32::MAX },
                            ..RenderConfig::default()
                        })
                    },
                    render_and_sync,
                )
            },
        );
    }
    group.finish();
}

/// What the denoise chain costs on top of a render, and how that scales with the
/// iteration count.
///
/// Held at a low sample count, which is the regime a denoiser exists for: the
/// filter is a fixed per-dispatch cost and would be lost under the tracing at
/// high spp. The `none` arm is what makes the numbers mean anything, since
/// `render_and_sync` also times scene upload and readback.
///
/// The processors are built once, outside the setup closure. `PostProcessors` is
/// `Clone` and wgpu handles are refcounted, so cloning reuses the pipelines --
/// rebuilding them per iteration would measure shader compilation instead.
pub fn denoise_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("denoise");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let (device, _) = get_wgpu_device_and_queue();
    let denoisers: Vec<(&str, Option<PostProcessors>)> = vec![
        ("none", None),
        (
            "iterations_3",
            Some(
                DenoisePostProcessor::new(1., Some(3), None, device)
                    .unwrap()
                    .into(),
            ),
        ),
        (
            "iterations_5",
            Some(
                DenoisePostProcessor::new(1., Some(5), None, device)
                    .unwrap()
                    .into(),
            ),
        ),
    ];

    for (name, denoiser) in &denoisers {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            denoiser,
            |b, denoiser| {
                b.iter_with_setup(
                    || {
                        create_test_scene(RenderConfig {
                            samples_per_pixel: 16,
                            width: 800,
                            height: 600,
                            post_processors: denoiser.clone().into_iter().collect(),
                            ..RenderConfig::default()
                        })
                    },
                    render_and_sync,
                )
            },
        );
    }
    group.finish();
}

/// Trace throughput against triangle count and BVH shape.
///
/// The arms are `nested` and `flat`, not "BVH" and "no BVH": the world is
/// always wrapped in a top-level [`Bvh`], so the flag only decides whether the
/// triangles get a sub-BVH of their own. See [`new_bvh_test_scene`].
pub fn bvh_traversal_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvh_traversal");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let render_config = || RenderConfig {
        samples_per_pixel: 16,
        width: 400,
        height: 200,
        ..RenderConfig::default()
    };

    for input_param in [
        BvhInput::new(10, true),
        BvhInput::new(10000, true),
        BvhInput::new(10000, false),
    ]
    .iter()
    {
        group.throughput(Throughput::Elements(input_param.num_triangles as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(input_param),
            input_param,
            |b, bvh_input| {
                b.iter_with_setup(
                    || {
                        new_bvh_test_scene(
                            render_config(),
                            bvh_input.nested,
                            bvh_input.num_triangles,
                        )
                    },
                    render_and_sync,
                );
            },
        );
    }

    // The arms above all trace a colinear strip against a sphere light, which
    // makes them a poor control for anything but the nesting. This one is a
    // spread cloud of triangles and nothing else -- see `triangle_cloud_scene`.
    let n = 10000u32;
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("triangles_only", |b| {
        b.iter_with_setup(|| triangle_cloud_scene(render_config(), n), render_and_sync);
    });

    group.finish();
}

#[derive(Constructor, Display)]
#[display("{} {}", num_triangles, if *nested { "nested" } else { "flat" })]
struct BvhInput {
    num_triangles: u32,
    /// Whether the triangles get a sub-BVH inside the world's own BVH.
    nested: bool,
}

/// What bloom and saturation cost on top of a render.
///
/// Built like `denoise_benchmark`, down to the `none` control arm and the
/// processors being constructed outside the setup closure -- `PostProcessors`
/// is `Clone` and wgpu handles are refcounted, so cloning reuses the pipelines
/// rather than recompiling them inside the measurement.
///
/// Both bloom radii are here because the blur is separable and its cost is
/// linear in the kernel, so the default the goldens use and the degenerate
/// three-tap case are different measurements: 0.1 of an 800-pixel width is a
/// 161-tap kernel, 0.002 is three taps. Without the pair, a change that only
/// helps long kernels looks like a change that helps bloom.
pub fn post_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("post");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let (device, _) = get_wgpu_device_and_queue();
    let processors: Vec<(&str, Option<PostProcessors>)> = vec![
        ("none", None),
        (
            "bloom_0.1",
            Some(
                BloomPostProcessor::new(0.1, None, Some(3.0), device)
                    .unwrap()
                    .into(),
            ),
        ),
        (
            "bloom_0.002",
            Some(
                BloomPostProcessor::new(0.002, None, Some(3.0), device)
                    .unwrap()
                    .into(),
            ),
        ),
        (
            "saturation",
            Some(SaturationPostProcessor::new(-0.7, device).unwrap().into()),
        ),
    ];

    for (name, processor) in &processors {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            processor,
            |b, processor| {
                b.iter_with_setup(
                    || {
                        // Same size and sample count as `denoise_benchmark`, so
                        // the two groups' `none` arms are the same measurement.
                        create_test_scene(RenderConfig {
                            samples_per_pixel: 16,
                            width: 800,
                            height: 600,
                            post_processors: processor.clone().into_iter().collect(),
                            ..RenderConfig::default()
                        })
                    },
                    render_and_sync,
                )
            },
        );
    }
    group.finish();
}

/// What it costs to get the finished image off the GPU.
///
/// On no other bench path at all: `render_and_sync` consumes progress and polls
/// but never reads back, so the copy, the map and the tone-mapped encode in
/// `buffer_to_image` were unmeasured -- including the 198 ms -> 25 ms at 4K that
/// its own doc comment claims. Two sizes an order of magnitude apart, because
/// the staging allocation is a fixed cost and the 133 MB read at 4K is not.
///
/// The scene is rendered once per size, outside the timed closure: what is
/// being measured is the readback, not the trace.
pub fn readback_benchmark(c: &mut Criterion) {
    let (device, queue) = get_wgpu_device_and_queue();

    let mut group = c.benchmark_group("readback");
    group.sample_size(10);

    for (name, width, height) in [("800x600", 800u32, 600u32), ("4K", 3840, 2160)] {
        let buffer = render_to_buffer(create_test_scene(RenderConfig {
            samples_per_pixel: 1,
            width: width as usize,
            height: height as usize,
            ..RenderConfig::default()
        }));

        group.throughput(Throughput::Bytes(width as u64 * height as u64 * 16));
        group.bench_function(name, |b| {
            b.iter_with_large_drop(|| {
                black_box(buffer_to_image(
                    device,
                    queue,
                    &buffer,
                    width,
                    height,
                    ToneMapper::default(),
                ))
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    obj_load_benchmark,
    bvh_build_benchmark,
    flatten_benchmark,
    bvh_traversal_benchmark,
    render_benchmark,
    adaptive_sampling_benchmark,
    denoise_benchmark,
    post_benchmark,
    readback_benchmark
);
criterion_main!(benches);
