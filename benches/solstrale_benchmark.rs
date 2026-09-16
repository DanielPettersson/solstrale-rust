use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::mpsc::channel;
use std::thread;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use derive_more::{Constructor, Display};

use crate::scenes::{create_test_scene, new_bvh_test_scene};
use solstrale::geo::transformation::NopTransformer;
use solstrale::geo::vec3::Vec3;
use solstrale::hittable::{Bvh, Hittables, Triangle};
use solstrale::loader::Loader;
use solstrale::loader::obj::Obj;
use solstrale::material::Lambertian;
use solstrale::material::texture::SolidColor;
use solstrale::post::{DenoisePostProcessor, PostProcessors};
use solstrale::ray_trace;
use solstrale::renderer::scene_flattener::flatten_scene;
use solstrale::renderer::{RenderConfig, Scene};
use solstrale::util::wgpu_util::get_wgpu_device_and_queue;

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

    group.finish();
}

/// Adaptive sampling on vs. effectively disabled, at the same total sample
/// budget -- isolates the win TODO.md's "per-pixel adaptive sampling" item
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

/// Trace throughput against triangle count, with and without a BVH.
pub fn bvh_traversal_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvh_traversal");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

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
                        let render_config = RenderConfig {
                            samples_per_pixel: 16,
                            width: 400,
                            height: 200,
                            ..RenderConfig::default()
                        };
                        new_bvh_test_scene(
                            render_config,
                            bvh_input.use_bvh,
                            bvh_input.num_triangles,
                        )
                    },
                    render_and_sync,
                );
            },
        );
    }
    group.finish();
}

#[derive(Constructor, Display)]
#[display("{} {}", num_triangles, use_bvh)]
struct BvhInput {
    num_triangles: u32,
    use_bvh: bool,
}

criterion_group!(
    benches,
    obj_load_benchmark,
    bvh_build_benchmark,
    flatten_benchmark,
    bvh_traversal_benchmark,
    render_benchmark,
    adaptive_sampling_benchmark,
    denoise_benchmark
);
criterion_main!(benches);
