# Implementation Plan: Efficient Texture Handling via Atlasing

This plan replaces the fixed 1024x1024 texture resizing with a single Texture Atlas using a shelf packing algorithm to improve memory efficiency and loading performance.

## Phase 1: Texture Packing Utility [checkpoint: 143c014]
Implement the core packing algorithm and data structures to manage the atlas layout.

- [x] Task: Implement `TexturePacker` utility 87fad38
    - [x] Define `TextureRect` and `AtlasLayout` structs in `src/util/texture_processing.rs`
    - [x] Implement Shelf Packing algorithm logic
- [x] Task: TDD - `TexturePacker` unit tests 87fad38
    - [x] Write tests in `tests/texture_processing_test.rs` to verify non-overlapping placement
    - [x] Write tests to verify the error condition when textures exceed max atlas size
- [x] Task: Conductor - User Manual Verification 'Phase 1: Texture Packing Utility' (Protocol in workflow.md) 143c014

## Phase 2: GpuData and Scene Flattener Refactor [checkpoint: 173a4ab]
Update the data structures passed to the GPU to include atlas coordinates and scales.

- [x] Task: Update `GpuData` structures 455ba32
    - [x] Modify `Material` struct in `src/renderer/gpu_data.rs` to include `uv_offset` and `uv_scale`
- [x] Task: Update `SceneFlattener` to calculate these values during scene processing 455ba32
- [x] Task: TDD - Scene Flattener verification 455ba32
    - [x] Write unit tests to ensure `SceneFlattener` correctly assigns atlas coordinates to materials
- [x] Task: Conductor - User Manual Verification 'Phase 2: GpuData and Scene Flattener Refactor' (Protocol in workflow.md) 173a4ab

## Phase 3: Texture Upload and Shader Updates
Modify the renderer to upload a single atlas and update the WGSL shader to use it.

- [x] Task: Refactor texture upload in `src/renderer/mod.rs` 5a52a8b
    - [x] Change `texture_2d_array` to a single `texture_2d` for the atlas
    - [x] Implement the logic to blit individual textures into the atlas buffer before upload
- [x] Task: Update `ray_trace.wgsl` 5a52a8b
    - [x] Modify the texture sampling logic to apply the `uv_offset` and `uv_scale` to mesh UVs
- [x] Task: TDD - Shader and Upload verification 5a52a8b
    - [x] Update integration tests in `tests/integration_tests.rs` to verify visual correctness with different texture sizes
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Texture Upload and Shader Updates' (Protocol in workflow.md)

## Phase 4: Final Integration and Optimization
Clean up old resizing logic and perform final verification.

- [ ] Task: Remove obsolete resizing logic
    - [ ] Remove forced 1024x1024 resizing from `src/util/texture_processing.rs`
- [ ] Task: Final quality gate and coverage check
    - [ ] Run `./coverage.sh` and ensure >90% coverage for new logic
    - [ ] Run `cargo clippy` and `cargo fmt`
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Final Integration and Optimization' (Protocol in workflow.md)
