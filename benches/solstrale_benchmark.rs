use std::hint::black_box;
use std::sync::mpsc::channel;
use std::thread;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use derive_more::{Constructor, Display};

use crate::scenes::{create_test_scene, new_bvh_test_scene};
use solstrale::geo::transformation::NopTransformer;
use solstrale::geo::vec3::Vec3;
use solstrale::hittable::{Bvh, Hittables, Triangle};
use solstrale::material::Lambertian;
use solstrale::material::texture::SolidColor;
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
    bvh_build_benchmark,
    flatten_benchmark,
    bvh_traversal_benchmark,
    render_benchmark
);
criterion_main!(benches);
