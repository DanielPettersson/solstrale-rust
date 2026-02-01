# Plan: GPU Imagemap Texture Support

## Phase 1: Core Implementation [checkpoint: 20c5d7d]
Goal: Update GPU data structures, shader, and host code to support texture mapping.

- [x] Task: Update GPU Data Structures 1f8c052
    - [x] Update `Triangle` struct in `src/renderer/gpu_data.rs` to include `uv0`, `uv1`, `uv2` (each [f32; 2]).
    - [x] Update `Material` struct in `src/renderer/gpu_data.rs` to include `texture_index` (i32).
- [x] Task: Update Shader Structs & Bindings a7d6866
    - [x] Update `Triangle` and `Material` structs in `src/renderer/ray_trace.wgsl` to match Rust changes.
    - [x] Add `texture_2d_array<f32>` and `sampler` bindings to `ray_trace.wgsl`.
- [x] Task: Implement UV Calculation & Sampling in Shader a7d6866
    - [x] Update `Quad` hit logic in `ray_trace.wgsl` to calculate and return UVs.
    - [x] Update `Sphere` hit logic in `ray_trace.wgsl` to calculate and return UVs.
    - [x] Update `Triangle` hit logic in `ray_trace.wgsl` to interpolate per-vertex UVs.
    - [x] Modify `scatter` function in `ray_trace.wgsl` to sample the texture array if `texture_index >= 0`.
- [x] Task: Implement Texture Processing Utility d3861cf
    - [x] Create a utility to resize an `RgbImage` to 1024x1024 using a high-quality filter.
- [x] Task: Update GPU Renderer for Texture Bindings e86883a
    - [x] Update `GpuRenderer` in `src/renderer/gpu_renderer.rs` to create the `texture_2d_array` and `sampler`.
    - [x] Update bind group creation to include the new texture array and sampler.
- [x] Task: Conductor - User Manual Verification 'Core Implementation' (Protocol in workflow.md) 20c5d7d

## Phase 2: Scene Flattening Updates
Goal: Connect the CPU scene representation to the new GPU structures.

- [~] Task: Update Scene Flattener
    - [ ] Modify `flatten_scene` in `src/renderer/scene_flattener.rs` to collect all unique `ImageMap` textures.
    - [ ] Update `add_primitive` to populate per-vertex UVs for Triangles.
    - [ ] Update `add_material` to assign the correct `texture_index` based on the collected textures.
- [ ] Task: Conductor - User Manual Verification 'Scene Flattening Updates' (Protocol in workflow.md)

## Phase 3: Integration & Verification
Goal: Finalize implementation and ensure all tests pass.

- [ ] Task: Verify Integration Tests
    - [ ] Run `test_texture_map` and ensure it passes on the GPU.
    - [ ] Run other GPU integration tests (`test_gpu_scene_sphere`, etc.) to ensure no regressions.
- [ ] Task: Performance & Quality Check
    - [ ] Verify visual parity between CPU and GPU renders.
    - [ ] Check for any significant performance bottlenecks introduced by texture sampling.
- [ ] Task: Conductor - User Manual Verification 'Integration & Verification' (Protocol in workflow.md)
