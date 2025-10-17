[![license](https://img.shields.io/github/license/DanielPettersson/solstrale-rust.svg)](https://www.tldrlegal.com/license/apache-license-2-0-apache-2-0)
[![CI](https://github.com/DanielPettersson/solstrale-rust/workflows/CI/badge.svg)](https://github.com/DanielPettersson/solstrale-rust/actions/workflows/ci.yaml)
[![Crates.io](https://img.shields.io/crates/d/solstrale?color=green&label=crates.io)](https://crates.io/crates/solstrale)
[![docs.rs](https://img.shields.io/docsrs/solstrale)](https://docs.rs/solstrale)
------
# Solstrale
A multithreaded Monte Carlo path tracing library, that as such has features like:
* Global illumination
* Caustics
* Reflection
* Refraction
* Soft shadows

Additionally, the library has:
* Loading of obj models with included materials
* Multithreaded Bvh creation to greatly speed up rendering
* Post-processing of rendered images by:
  * [Open Image Denoise](https://www.openimagedenoise.org/)
  * Bloom filter, GPU compute implementation using [WGPU](https://wgpu.rs/)
  * Saturation filter, GPU compute implementation using [WGPU](https://wgpu.rs/)
* Bump mapping
* Light attenuation

## Example output
![bedroom2](https://github.com/DanielPettersson/solstrale-rust/assets/3603911/a78e4a85-2acb-409f-b7f4-4f6c5afb797e)
![conference](https://github.com/DanielPettersson/solstrale-rust/assets/3603911/8c88c777-0b85-4854-bd14-10a999bb3f78)
![happy](https://github.com/DanielPettersson/solstrale-rust/assets/3603911/c5357792-a3dc-42f9-8230-320140f9c30e)
![sponza-bump2](https://github.com/DanielPettersson/solstrale-rust/assets/3603911/0ab79ed9-cddf-46b1-84e7-03cef35f5600)

## Credits
The ray tracing is inspired by the excellent [Ray Tracing in One Weekend Book Series](https://github.com/RayTracing/raytracing.github.io) by Peter Shirley
