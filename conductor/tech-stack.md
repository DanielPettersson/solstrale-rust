# Tech Stack

## Core Language & Runtime
- **Rust (Edition 2021):** The primary language chosen for memory safety, performance, and its rich ecosystem for graphics and systems programming.

## Graphics & Rendering
- **Path Tracing Engine:** Custom-built Monte Carlo path tracer.
- **WGPU:** Used for GPU-accelerated post-processing effects (Bloom, Saturation) via compute shaders.
- **Open Image Denoise (OIDN):** Integration for high-quality AI denoising of rendered frames.

## Performance & Concurrency
- **Rayon:** Utilized for data-parallelism, specifically in multithreaded BVH construction and CPU-based rendering tasks.
- **Fastrand:** Low-overhead random number generation critical for Monte Carlo simulations.

## Data & I/O
- **image:** Crate for handling image encoding and decoding (PNG, JPG).
- **tobj:** For loading and parsing OBJ models and MTL materials.
- **bytemuck:** For safe casting of data structures, particularly when interfacing with the GPU (WGPU).

## Utilities
- **derive_more:** To reduce boilerplate for common trait implementations.
- **once_cell:** For thread-safe initialization of global or shared resources.
