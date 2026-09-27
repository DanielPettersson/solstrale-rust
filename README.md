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
* Smooth shading: per-vertex normals are interpolated across a triangle, taken
  from the model's own `vn` records or generated at a default crease angle when
  it has none, so hard edges stay hard. `with_flat_shading` turns generation off
* GGX microfacet metal and glass, both with Turquin multiple-scattering energy
  compensation, so a rough surface does not darken as its roughness rises.
  `Metal::albedo` is f0 and `fuzz` is a perceptual roughness; `Dielectric` takes
  a roughness of its own, plus a per-world-unit Beer-Lambert absorption
* Next-event estimation with multiple importance sampling, for much lower noise
  per sample on scenes lit by discrete lights. Lights are picked in proportion
  to their emitted power through an alias table, and a large quad light is
  sampled by solid angle rather than by area
* Owen-scrambled Sobol sampling, hash-based and table-free, padded into
  independent 2D sequences per draw. Cuts error against a converged reference by
  25-40% on the test scenes at every sample count, for the same work -- it does
  not change what the image converges to, only how fast it gets there. Owen
  rather than plain stratification because adaptive sampling retires each pixel
  at a sample count nobody knows in advance, and only Owen scrambling keeps a
  truncated sequence unbiased
* Per-pixel adaptive sampling: a pixel stops being traced once the relative
  standard error of its own estimate falls below `variance_threshold`
* Russian roulette path termination

### Performance & Loading
* Loading of obj models with included materials and per-vertex normals, with MTL
  materials mapped onto the crate's material types
* Multithreaded BVH construction using Rayon, with a binned SAH sweep over all
  three axes and front-to-back ordered GPU traversal
* The tracer is compiled against the scene it is tracing, so a branch the scene
  could never have taken is not in the shader at all
* A running render takes scene updates and applies each for what it costs: a new
  sample count continues the render, a new camera or colour restarts it, and a
  moved mesh is re-baked and re-uploaded on its own, with nothing else rebuilt.
  See [Updating a running render](#updating-a-running-render)

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
* With `RenderConfig::preview`, the denoiser and the saturation grade also run
  on the unfinished image, so an interactive viewport is filtered while the
  camera is moving rather than only once it stops

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
use solstrale::geo::transformation::NopTransformer;
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
                Lambertian::new(SolidColor::new(1., 1., 0.).into(), None).into(),
                &NopTransformer()
            ).into()
        ]).into(),
        camera: CameraConfig::default(),
        background_color: Vec3::new(0.2, 0.3, 0.5),
        render_config: RenderConfig::default(),
        post_processors: vec![],
    };

    let (output_sender, output_receiver) = channel();
    let (_, update_receiver) = channel();
    let (_, abort_receiver) = channel();

    thread::spawn(move || {
        ray_trace(scene, &output_sender, &update_receiver, &abort_receiver, &device, &queue, false).unwrap();
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
| `min_samples_per_pixel` | 32 | Samples a pixel must reach before adaptive sampling may retire it |
| `variance_threshold` | 0.01 | Relative standard error below which a pixel is retired |
| `low_discrepancy` | `true` | Draw samples from an Owen-scrambled Sobol sequence rather than white noise |
| `seed` | 0 | Distinguishes two otherwise identical renders |
| `preview` | `false` | Run the preview post-processors on every batch, not just the last |

The filters applied to the final image are `Scene::post_processors`, beside the
render configuration rather than in it. Order matters there: put the denoiser
first. Denoising a bloomed image blurs the bloom, while bloom applied to a
denoised image is what you want.

`samples_per_batch` trades reporting granularity for throughput: larger batches
amortise dispatch overhead and collapse the per-sample read-modify-write of the
accumulation buffer, but render progress is reported less often and camera
changes take longer to take effect. The renderer tunes the actual size down from
here to keep a single dispatch inside one vsync interval.

`variance_threshold` is a noise floor, not only a speed knob: retiring is
permanent, so a pixel keeps whatever error it had when it passed the test and no
amount of `samples_per_pixel` pushes it lower. Set `min_samples_per_pixel` above
`samples_per_pixel` to disable adaptive sampling entirely.

`preview` is what an interactive viewport wants. A camera drag restarts the
accumulation on every frame and so never reaches a last batch, which is the one
regime a denoiser is built for; the cost is a full run of those processors per
batch whether or not anyone is watching, hence the default.

`seed` matters when a render is being measured against another one. Two renders
that share a seed trace the same sample stream, so a low-sample render and the
converged reference it is compared with will agree more closely than they should
-- with `low_discrepancy` on, the short render is literally a subset of the long
one. Give the reference a different seed.

## Updating a running render
`ray_trace` takes a channel of `SceneUpdate`s, which a running render applies at
its next batch. Each part costs what changing it has to cost, not a rebuild:

| outcome | parts |
|---|---|
| nothing | `samples_per_batch`, `preview` |
| the accumulation continues | `samples_per_pixel` raised, `min_samples_per_pixel`, `variance_threshold` |
| the finished image goes through the full post chain again | `samples_per_pixel` lowered to what is done, `post_processors` |
| the accumulation restarts | camera, background, `max_depth`, `low_discrepancy`, `seed`, size, world |

A camera update is one `.into()` away: `sender.send(camera.into())`.

A new world is uploaded for what changed in it. A nested `Bvh` of at least
`SUBTREE_MIN_PRIMS` (8192) primitives is kept apart as a subtree rather than
dissolved into the world's tree, and the renderer keeps each subtree it has
uploaded: sending a world that holds the same `Bvh` again costs nothing for it,
and `Bvh::transformed` moves one without reloading or rebuilding it, written
over its old segment in place. `Bvh` clones are reference counts, so a model
cache can hand the same one out as often as it likes.

```rust
let mesh = Obj::new("models/", "dragon.obj").load(&NopTransformer(), None)?;
// Absolute, from the mesh as loaded, however often it has moved since.
let moved = mesh.transformed(Translation::new(Vec3::new(1., 0., 0.)));

let mut update = SceneUpdate::default();
update.world = Some(Bvh::new(vec![moved.into(), floor.clone()]).into());
update_sender.send(update)?;
```

A moved mesh keeps its tree, refitted, and a rotation loosens it: up to 23%
slower to render at 45 degrees. `Bvh::rebuilt` builds it again for where it is,
on whatever thread can afford it once a drag ends, and a world update swaps it
in.

## Known limitations

[`LIMITATIONS.md`](LIMITATIONS.md) records what is true of the renderer on
purpose, what was tried and measured and rejected, and the numbers those
decisions rest on. It is worth reading before changing anything in the
integrator or the denoiser -- several of the obvious metrics for this kind of
work are recorded there as having been measured and found to lie.

Outstanding work is tracked in
[GitHub issues](https://github.com/DanielPettersson/solstrale-rust/issues),
prioritised `P1`/`P2`/`P3`.

## Upgrading to 0.6
**The camera channel is now an update channel.** `ray_trace` and
`Renderer::render` take a `&Receiver<SceneUpdate>` where they took a
`&Receiver<CameraConfig>`. A camera converts into an update, so a caller that
only ever sends cameras changes one call:

```rust
// 0.5
camera_sender.send(camera)?;
// 0.6
update_sender.send(camera.into())?;
```

Anything else about the scene can now be sent the same way instead of aborting
the render and building a new one; see
[Updating a running render](#updating-a-running-render).

**Post-processors moved from `RenderConfig` to `Scene`.** A processor's
parameters are baked into its pipelines when it is built, so a chain can only be
replaced, never compared, and it no longer rides along with every change to the
sample count:

```rust
// 0.5
Scene { world, camera, background_color, render_config: RenderConfig { post_processors, ..Default::default() } }
// 0.6
Scene { world, camera, background_color, render_config: RenderConfig::default(), post_processors }
```

`RenderConfig` is now `PartialEq`, and `CameraConfig` is `Clone`, `Copy`,
`Debug` and `PartialEq`.

**`RenderProgress` gained `width` and `height`.** Take a blit's or a saved
image's stride from them rather than from the size that was asked for: a render
updated to a new size publishes a new buffer, and so does one whose chain
becomes empty or stops being so.

**A large nested `Bvh` stays a tree of its own.** One of `SUBTREE_MIN_PRIMS`
(8192) primitives or more is kept apart as a subtree instead of being dissolved
into the tree it is nested in, which is what lets an update move it without
touching the rest. `Bvh::clone` is now a reference count rather than a copy,
`Bvh::primitive_count` counts through subtrees, and `Bvh::transformed` and
`Bvh::rebuilt` are new.

**`Transformer` requires `Send + Sync`**, so a mesh can be re-baked in
parallel. Every transformer in the crate already is, and `Box` and `Arc` of one
are transformers too.

**`SceneData` is laid out as an arena.** `flatten_scene` still returns one, but
the triangle arrays are indexed like `prim_refs` -- a sphere or a quad leaves a
hole in them -- and an empty world gets a root node over two empty leaves.
`GpuRenderConfig` carries the light count, which is no longer an override.

**A world update never ends the render.** `Renderer::new` still refuses a scene
with no light and a black background; an update to one renders it black.

## Upgrading to 0.5
`Metal` is now a GGX microfacet conductor. `albedo` means f0, the reflectance at
normal incidence, rather than a flat multiplier, and `fuzz` is a perceptual
roughness used as `alpha = fuzz * fuzz`. Numbers are not comparable with the old
fuzz parameter: the same value reads noticeably sharper.

`Dielectric::new` takes two more arguments, `albedo` and `roughness`. `albedo`
is the fraction transmitted per world unit travelled inside the glass, so
`SolidColor::new(1., 1., 1.)` is clear glass and exactly a no-op; `roughness` is
the same perceptual roughness `Metal` takes, and `0.` is the smooth glass of
earlier versions.

`Sphere::new` takes a `&dyn Transformer` as its fourth argument, like `Quad::new`
and `Triangle::new`. Pass `&NopTransformer()` to keep the old behaviour.

`buffer_to_image` takes a `ToneMapper`. `ToneMapper::Clamp` reproduces what
earlier versions did; `ToneMapper::default()` is ACES filmic.

`RenderConfig` gained `preview`, and `Hittable::get_lights` and `GpuRay` are
gone. `AttenuatedColor` is deprecated and will be removed.

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
<img width="1303" height="964" alt="gallery" src="https://github.com/user-attachments/assets/50d1fc18-ddcb-4a04-819f-ecb8b6e99e40" />
<img width="1303" height="964" alt="happy" src="https://github.com/user-attachments/assets/1c264ddc-1612-41b0-8036-4339d2df13f8" />

