# Specification: GPU Path Tracing Migration

## Overview
Move the entire path tracing engine from a CPU-based implementation to a GPU-resident system using WGPU compute shaders. This is a complete replacement of the current CPU renderer, aimed at leveraging the parallel compute capabilities of modern GPUs for significant performance gains while maintaining visual parity.

## Goals
- Transition from CPU rendering (using Rayon for parallelism) to GPU compute (using WGSL).
- Maintain the same high visual fidelity and feature set (Global Illumination, BVH, OBJ support, Materials).
- Optimize data structures for GPU memory layouts and access patterns.

## Functional Requirements
- **WGSL Path Tracer:** Implement a full path tracing loop in WGSL, including:
    - Ray Generation (Camera logic).
    - Efficient BVH Traversal (Iterative traversal, as recursion is not supported in WGSL).
    - Intersection logic for primitives (Spheres, Quads, Triangles).
    - Shading and Material evaluation (Lambertian, Metal, Dielectric, Emissive).
    - Sample Accumulation over multiple frames/passes.
- **GPU Data Management:**
    - Develop a buffer management system to upload scene data (BVH nodes, Triangles, Spheres, Materials, Textures) to the GPU.
    - Implement a "Mega-Kernel" or a multi-pass compute pipeline that handles the tracing logic.
    - Use `bytemuck` for safe and efficient data uploading.
- **Randomness:** Implement a fast, GPU-compatible random number generator (e.g., PCG or Xorshift) within WGSL.
- **Integration:** Update the `Renderer` interface (or replace it) to drive the WGPU pipeline instead of the Rayon-based loop.

## Non-Functional Requirements
- **Visual Parity:** Rendered output must match the existing integration test "expected" images (within a small floating-point tolerance).
- **Performance:** Significant reduction in render time for complex scenes compared to the CPU implementation.
- **Memory Efficiency:** Manage GPU VRAM usage carefully, especially for large models and high-resolution textures.

## Acceptance Criteria
1.  All existing integration tests in `tests/scenes.rs` pass using the new GPU renderer.
2.  The CPU-based path tracing logic in `src/renderer/mod.rs` (and associated modules) is removed or fully bypassed.
3.  The application correctly handles cases where WGPU initialization fails or required features are missing (graceful exit or error message).
4.  No regression in visual features (Bump mapping, textures, light attenuation must still work).

## Out of Scope
- Support for multiple GPUs.
- Migrating the OBJ loader itself to the GPU (loading stays on CPU).
- Real-time denoising (OIDN remains a post-processing step on CPU/GPU as currently implemented).
