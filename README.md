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

### Performance & Loading
* Loading of obj models with included materials
* Multithreaded BVH creation using Rayon to greatly speed up rendering

### Post-Processing
Custom GPU-accelerated filters implemented as compute shaders via [WGPU](https://wgpu.rs/):
* Bloom filter
* Saturation filter

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

## Example output
![bedroom2](https://github.com/DanielPettersson/solstrale-rust/assets/3603911/a78e4a85-2acb-409f-b7f4-4f6c5afb797e)
![conference](https://github.com/DanielPettersson/solstrale-rust/assets/3603911/8c88c777-0b85-4854-bd14-10a999bb3f78)
![happy](https://github.com/DanielPettersson/solstrale-rust/assets/3603911/c5357792-a3dc-42f9-8230-320140f9c30e)
![sponza-bump2](https://github.com/DanielPettersson/solstrale-rust/assets/3603911/0ab79ed9-cddf-46b1-84e7-03cef35f5600)

## Credits
The ray tracing is inspired by the excellent [Ray Tracing in One Weekend Book Series](https://github.com/RayTracing/raytracing.github.io) by Peter Shirley
