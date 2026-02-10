# Implementation Plan: Efficient Texture Handling via Atlasing

This plan replaces the fixed 1024x1024 texture resizing with a single Texture Atlas using a shelf packing algorithm to improve memory efficiency and loading performance.

## Phase 1: Texture Packing Utility
Implement the core packing algorithm and data structures to manage the atlas layout.

- [x] Task: Implement `TexturePacker` utility 87fad38
    - [ ] Define `TextureRect` and `AtlasLayout` structs in `src/util/texture_processing.rs`
    - [ ] Implement Shelf Packing algorithm logic
- [ ] Task: TDD - `TexturePacker` unit tests
    - [ ] Write tests in `tests/texture_processing_test.rs` to verify non-overlapping placement
    - [ ] Write tests to verify the error condition when textures exceed max atlas size
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Texture Packing Utility' (Protocol in workflow.md)

## Phase 2: GpuData and Scene Flattener Refactor
Update the data structures passed to the GPU to include atlas coordinates and scales.

- [ ] Task: Update `GpuData` structures
    - [ ] Modify `Material` struct in `src/renderer/gpu_data.rs` to include `uv_offset` and `uv_scale`
    - [ ] Update `SceneFlattener` to calculate these values during scene processing
- [ ] Task: TDD - Scene Flattener verification
    - [ ] Write unit tests to ensure `SceneFlattener` correctly assigns atlas coordinates to materials
- [ ] Task: Conductor - User Manual Verification 'Phase 2: GpuData and Scene Flattener Refactor' (Protocol in workflow.md)

## Phase 3: Texture Upload and Shader Updates
Modify the renderer to upload a single atlas and update the WGSL shader to use it.

- [ ] Task: Refactor texture upload in `src/renderer/mod.rs`
    - [ ] Change `texture_2d_array` to a single `texture_2d` for the atlas
    - [ ] Implement the logic to blit individual textures into the atlas buffer before upload
- [ ] Task: Update `ray_trace.wgsl`
    - [ ] Modify the texture sampling logic to apply the `uv_offset` and `uv_scale` to mesh UVs
- [ ] Task: TDD - Shader and Upload verification
    - [ ] Update integration tests in `tests/integration_tests.rs` to verify visual correctness with different texture sizes
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Texture Upload and Shader Updates' (Protocol in workflow.md)

## Phase 4: Final Integration and Optimization
Clean up old resizing logic and perform final verification.

- [ ] Task: Remove obsolete resizing logic
    - [ ] Remove forced 1024x1024 resizing from `src/util/texture_processing.rs`
- [ ] Task: Final quality gate and coverage check
    - [ ] Run `./coverage.sh` and ensure >90% coverage for new logic
    - [ ] Run `cargo clippy` and `cargo fmt`
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Final Integration and Optimization' (Protocol in workflow.md)
