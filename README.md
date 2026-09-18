[![license](https://img.shields.io/github/license/DanielPettersson/solstrale-rust.svg)](https://www.tldrlegal.com/license/apache-license-2-0-apache-2-0)
[![CI](https://github.com/DanielPettersson/solstrale-rust/workflows/CI/badge.svg)](https://github.com/DanielPettersson/solstrale-rust/actions/workflows/ci.yaml)
[![Crates.io](https://img.shields.io/crates/d/solstrale?color=green&label=crates.io)](https://crates.io/crates/solstrale)
[![docs.rs](https://img.shields.io/docsrs/solstrale)](https://docs.rs/solstrale)
------
# Solstrale
A WGPU-based GPU Monte Carlo path tracing library, with features like:

### Core Engine
* Global illumination
* Caustics
* Reflection
* Refraction
* Soft shadows
* Bump mapping
* Light attenuation
* Next-event estimation with multiple importance sampling, for much lower noise
  per sample on scenes lit by discrete lights
* Russian roulette path termination

### Performance & Loading
* Loading of obj models with included materials
* Multithreaded BVH construction using Rayon, with a binned SAH split heuristic
  and front-to-back ordered GPU traversal

### Post-Processing
Custom GPU-accelerated filters implemented as compute shaders via [WGPU](https://wgpu.rs/):
* Bloom filter
* Saturation filter
* Denoiser: an edge-avoiding a-trous wavelet filter guided by the per-pixel
  variance the sample loop already tracks, which cuts error against a converged
  reference by about a third at 8 samples per pixel and leaves an already
  converged image essentially untouched. Fireflies do not survive it: the filter
  is faded back in against each pixel's neighbourhood rather than against its own
  noise-inflated brightness, and outlier rejection runs alongside the filter's
  variance pre-pass for the tail an edge-avoiding filter cannot reach

### Display
* Tone mapping: ACES filmic by default, with Khronos PBR Neutral, extended
  Reinhard and a plain clamp selectable. Applied in `buffer_to_image` as a
  display transform, so highlights roll off instead of clipping at linear 1.0
  while everything upstream keeps working on real radiance

## Requirements
A GPU adapter supporting compute shaders and at least 12 storage buffers per
shader stage. Native backends (Vulkan, Metal, DX12) exceed this comfortably; the
WebGPU downlevel default of 8 does not.

## Installation
To build the library, ensure you have Rust installed and run:
```bash
cargo build --release
```

## Usage
### Running Tests
To run the project's test suite:
```bash
cargo test
```

### Library Example
Add `solstrale` to your `Cargo.toml`. Below is a basic example of how to set up a scene and start rendering:

```rust
use std::sync::mpsc::channel;
use std::thread;
use solstrale::camera::CameraConfig;
use solstrale::geo::vec3::Vec3;
use solstrale::hittable::{Bvh, Sphere};
use solstrale::material::Lambertian;
use solstrale::material::texture::SolidColor;
use solstrale::ray_trace;
use solstrale::renderer::{RenderConfig, Scene};
use solstrale::util::wgpu_util::get_wgpu_device_and_queue;

fn main() {
    let (device, queue) = get_wgpu_device_and_queue();

    let scene = Scene {
        world: Bvh::new(vec![
            Sphere::new(
                Vec3::new(0., 0., 0.), 
                0.5, 
                Lambertian::new(SolidColor::new(1., 1., 0.).into(), None).into()
            ).into()
        ]).into(),
        camera: CameraConfig::default(),
        background_color: Vec3::new(0.2, 0.3, 0.5),
        render_config: RenderConfig::default(),
    };

    let (output_sender, output_receiver) = channel();
    let (_, camera_config_receiver) = channel();
    let (_, abort_receiver) = channel();

    thread::spawn(move || {
        ray_trace(scene, &output_sender, &camera_config_receiver, &abort_receiver, &device, &queue, false).unwrap();
    });

    for render_output in output_receiver {
        // Handle render progress and output buffer
        let _progress = render_output.progress;
    }
}
```

## Configuration
`RenderConfig` controls how a scene is rendered:

| Field | Default | Description |
|---|---|---|
| `width`, `height` | 300 x 200 | Output resolution in pixels |
| `samples_per_pixel` | 50 | Paths traced per pixel |
| `max_depth` | 10 | Maximum ray bounces before a path is cut off |
| `samples_per_batch` | 4 | Samples traced per GPU dispatch |
| `post_processors` | none | Filters applied to the final image |

Order matters in `post_processors`: put the denoiser first. Denoising a bloomed
image blurs the bloom, while bloom applied to a denoised image is what you want.

`samples_per_batch` trades reporting granularity for throughput: larger batches
amortise dispatch overhead and collapse the per-sample read-modify-write of the
accumulation buffer, but render progress is reported less often and camera
changes take longer to take effect.

## Upgrading to 0.4
`PostProcessor::post_process` now takes a single `PostProcessContext` instead of
an encoder, buffer and device. The context also carries the accumulation buffer,
the per-pixel sample counts and the primary-hit guide buffer, which is what a
denoiser needs and what the old signature could not express. `PostProcessors`
gained a `DenoisePostProcessor` variant and is now `#[non_exhaustive]`, so future
post-processors are not themselves breaking changes.

The post-processing chain now runs on a copy of the accumulation buffer rather
than in place, so `RenderProgress::output_buffer` is that copy whenever any
post-processor is configured. It remains a stable handle for the life of a
render.

## Upgrading to 0.3
`RenderConfig` gained `max_depth` and `samples_per_batch`, so code constructing it
with an exhaustive struct literal needs updating. Spreading the defaults keeps it
working across future additions:

```rust
RenderConfig {
    width: 800,
    height: 600,
    samples_per_pixel: 200,
    ..Default::default()
}
```

## Example output
![bedroom2](https://github.com/DanielPettersson/solstrale-rust/assets/3603911/a78e4a85-2acb-409f-b7f4-4f6c5afb797e)
![conference](https://github.com/DanielPettersson/solstrale-rust/assets/3603911/8c88c777-0b85-4854-bd14-10a999bb3f78)
![happy](https://github.com/DanielPettersson/solstrale-rust/assets/3603911/c5357792-a3dc-42f9-8230-320140f9c30e)
![sponza-bump2](https://github.com/DanielPettersson/solstrale-rust/assets/3603911/0ab79ed9-cddf-46b1-84e7-03cef35f5600)

## Credits
The ray tracing is inspired by the excellent [Ray Tracing in One Weekend Book Series](https://github.com/RayTracing/raytracing.github.io) by Peter Shirley
