# Plan: GPU Path Tracing Migration

## Phase 1: WGPU Compute Foundation & Data Transfer
Establish the basic compute pipeline and implement the robust transfer of complex scene data (BVH, primitives, materials) from CPU to GPU memory.

- [x] Task: Initialize WGPU Compute Pipeline for Ray Tracing d88b119
    - [x] Create a new `GpuRenderer` struct/module using the same API as the CPU renderer.
    - [x] Set up the WGPU infrastructure to render to a texture using wgpu_util in much the same way as the GPU-based post-processing pipeline.
    - [x] Create a "Hello World" compute shader that simply writes a solid color to the output texture.
    - [x] Test: Write a test that initializes the renderer and checks if the output texture has the expected color.
- [x] Task: Implement GPU Data Structures (WGSL & Rust) 1fcb5b2
    - [x] Define WGSL structs for `Ray`, `HitRecord`, `Sphere`, `Triangle`, `Quad`, `Material`, and `BvhNode` ensuring strict alignment rules (std140/std430).
    - [x] Create corresponding `repr(C)` Rust structs in `src/renderer/gpu_data.rs` implementing `bytemuck::Pod`.
    - [x] Test: Write unit tests to assert the size and alignment of Rust structs match WGSL expectations.
- [ ] Task: Scene Data Upload System
    - [ ] Implement logic to flatten the CPU BVH tree and primitive lists into linear `Vec` buffers.
    - [ ] Create Storage Buffers for Nodes, Primitives, and Materials on the GPU.
    - [ ] Upload the linear buffers to the GPU.
    - [ ] Test: Create a small mock scene, upload it, and read back the buffers to verify data integrity (staging buffer readback).
- [ ] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

## Phase 2: Ray Generation & Intersection
Implement the primary visibility rays. Instead of full path tracing, we will first render a "Normal Buffer" or "Depth Buffer" to visually verify that geometry and BVH traversal are working correctly.

- [ ] Task: Implement Camera Uniforms & Ray Generation
    - [ ] Map the `Camera` struct to a Uniform Buffer.
    - [ ] Write the WGSL `ray_generation` function to spawn rays based on pixel coordinates.
    - [ ] Test: Render a "UV map" image where pixel color corresponds to ray direction, verify against CPU equivalent.
- [ ] Task: Implement Iterative BVH Traversal in WGSL
    - [ ] Write the stack-based traversal algorithm in WGSL (replacing CPU recursion).
    - [ ] Implement intersection functions for Sphere, Triangle, and Quad in WGSL.
    - [ ] Connect the traversal loop to the intersection tests.
    - [ ] Test: Render a "Depth Map" (grayscale based on distance). Verify a simple scene with 1 sphere and 1 box.
- [ ] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: Shading & Path Tracing Loop
Implement the core Monte Carlo integration, material evaluation, and sample accumulation.

- [ ] Task: Implement Random Number Generation in WGSL
    - [ ] Port a lightweight PRNG (e.g., PCG Hash or Xorshift) to WGSL.
    - [ ] manage per-pixel RNG state.
    - [ ] Test: Render a visual noise image to verify distribution quality.
- [ ] Task: Port Material System to WGSL
    - [ ] Implement `scatter` functions for Lambertian, Metal, and Dielectric materials in WGSL.
    - [ ] Implement texture sampling for material properties.
    - [ ] Test: Render a scene with one of each material type (no global illumination yet, just direct hit color).
- [ ] Task: Implement the Path Tracing Loop
    - [ ] Write the iterative bounce loop (max depth limit).
    - [ ] Accumulate emitted light and attenuation at each bounce.
    - [ ] Implement progressive accumulation (blending new samples with the existing image buffer).
    - [ ] Test: Render the standard `scenes::simple` scene and compare with `out_expected_simple.jpg`.
- [ ] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)

## Phase 4: Integration & Cleanup
Finalize the replacement of the CPU renderer.

- [ ] Task: Switch Main Entry Point
    - [ ] Update `lib.rs` and the public API to use `GpuRenderer` by default.
    - [ ] Ensure all configuration options (samples per pixel, max depth) are respected.
    - [ ] Test: Run the full integration test suite (`tests/scenes.rs`).
- [ ] Task: Remove CPU Rendering Logic
    - [ ] Delete `src/renderer/cpu_impl.rs` (or equivalent old modules).
    - [ ] Clean up unused dependencies (e.g., `rayon` usage for rendering, though it might stay for BVH building).
    - [ ] Test: Ensure project compiles and runs without the old code.
- [ ] Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)
