use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use solstrale::geo::vec3::Vec3;
use solstrale::post::{BloomPostProcessor, PostProcessor, SaturationPostProcessor};

pub fn bloom_benchmark(c: &mut Criterion) {
    let width = 800;
    let height = 600;
    let pixel_colors = vec![Vec3::new(0.5, 0.5, 0.5); (width * height) as usize];
    let albedo_colors = vec![Vec3::new(0., 0., 0.); 0];
    let normal_colors = vec![Vec3::new(0., 0., 0.); 0];

    c.bench_function("bloom_post_process", |b| {
        b.iter_with_setup(
            || {
                let mut post = BloomPostProcessor::new(0.1, None, None).unwrap();
                post.initialize(width, height);
                post
            },
            |post| {
                post.intermediate_post_process(
                    black_box(&pixel_colors),
                    black_box(&albedo_colors),
                    black_box(&normal_colors),
                    1,
                )
                .unwrap();
            },
        )
    });
}

pub fn saturation_benchmark(c: &mut Criterion) {
    let width = 800;
    let height = 600;
    let pixel_colors = vec![Vec3::new(0.5, 0.5, 0.5); (width * height) as usize];
    let albedo_colors = vec![Vec3::new(0., 0., 0.); 0];
    let normal_colors = vec![Vec3::new(0., 0., 0.); 0];

    c.bench_function("saturation_post_process", |b| {
        b.iter_with_setup(
            || {
                let mut post = SaturationPostProcessor::new(0.5).unwrap();
                post.initialize(width, height);
                post
            },
            |post| {
                post.intermediate_post_process(
                    black_box(&pixel_colors),
                    black_box(&albedo_colors),
                    black_box(&normal_colors),
                    1,
                )
                .unwrap();
            },
        )
    });
}

criterion_group!(benches, bloom_benchmark, saturation_benchmark);
criterion_main!(benches);
