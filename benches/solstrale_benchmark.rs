use std::hint::black_box;
use std::sync::mpsc::channel;
use std::thread;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use derive_more::{Constructor, Display};

use crate::scenes::{create_test_scene, new_bvh_test_scene};
use solstrale::ray_trace;
use solstrale::renderer::RenderConfig;
use solstrale::util::wgpu_util::get_wgpu_device_and_queue;

#[path = "../tests/scenes.rs"]
mod scenes;

pub fn bvh_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvh_benchmark");
    for input_param in [
        BvhInput::new(10, true),
        BvhInput::new(10, false),
        BvhInput::new(10000, true),
        BvhInput::new(10000, false),
    ]
    .iter()
    {
        group.throughput(Throughput::Bytes(input_param.num_triangles as u64));
        group.sample_size(25);
        group.measurement_time(Duration::from_secs(10));
        group.bench_with_input(
            BenchmarkId::from_parameter(input_param),
            input_param,
            |b, bvh_input| {
                b.iter_with_setup(
                    || {
                        let render_config = RenderConfig {
                            samples_per_pixel: 1,
                            width: black_box(20),
                            height: black_box(10),
                            ..RenderConfig::default()
                        };
                        new_bvh_test_scene(
                            render_config,
                            bvh_input.use_bvh,
                            bvh_input.num_triangles,
                        )
                    },
                    |scene| {
                        let (device, queue) = get_wgpu_device_and_queue();

                        let (output_sender, output_receiver) = channel();
                        let (_, abort_receiver) = channel();

                        thread::spawn(move || {
                            ray_trace(scene, &output_sender, &abort_receiver, device, queue)
                                .unwrap();
                        });

                        for _ in output_receiver {}
                    },
                );
            },
        );
    }
    group.finish();
}

pub fn scene_benchmark(c: &mut Criterion) {
    c.bench_function("scene_benchmark", |b| {
        b.iter_with_setup(
            || {
                let render_config = RenderConfig {
                    samples_per_pixel: 1,
                    width: black_box(100),
                    height: black_box(50),
                    ..RenderConfig::default()
                };
                create_test_scene(render_config)
            },
            |scene| {
                let (device, queue) = get_wgpu_device_and_queue();

                let (output_sender, output_receiver) = channel();
                let (_, abort_receiver) = channel();

                thread::spawn(move || {
                    ray_trace(scene, &output_sender, &abort_receiver, device, queue).unwrap();
                });

                for _ in output_receiver {}
            },
        )
    });
}

#[derive(Constructor, Display)]
#[display("{} {}", num_triangles, use_bvh)]
struct BvhInput {
    num_triangles: u32,
    use_bvh: bool,
}

criterion_group!(benches, bvh_benchmark, scene_benchmark);
criterion_main!(benches);
