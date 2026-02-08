# Implementation Plan: GPU Blend Material

This plan covers the implementation of the `Blend` material for the GPU renderer, ensuring parity with the CPU's probabilistic selection logic and supporting nested materials.

## Phase 1: Host Data Structures and Scene Flattening [checkpoint: 321ecfd]

In this phase, we update the material data structures shared between the CPU and GPU and extend the scene flattener to support `Blend` materials.

- [x] Task: Update `GpuMaterial` struct in `src/renderer/gpu_data.rs` 321ecfd
    - [x] Rename `_padding2` to `blend_factor`
    - [x] Rename `_padding4` to `blend_indices`
- [x] Task: Update `add_material` in `src/renderer/scene_flattener.rs` 321ecfd
    - [x] Add `Materials::Blend` to the match arm in `add_material`
    - [x] Implement recursive addition of child materials and population of `blend_factor` and `blend_indices`
- [x] Task: Write failing unit test for `flatten_scene` with `Blend` material (Red Phase) 321ecfd
    - [x] Create a test in `src/renderer/scene_flattener.rs` (or update existing) that flattens a scene with nested `Blend` materials
- [x] Task: Implement flattener changes to pass the test (Green Phase) 321ecfd
- [x] Task: Conductor - User Manual Verification 'Phase 1: Host Data Structures and Scene Flattening' (Protocol in workflow.md) 321ecfd

## Phase 2: Shader Implementation [checkpoint: ae3b2da]

In this phase, we update the WGSL shader to recognize the `Blend` material type and implement the probabilistic resolution loop.

- [x] Task: Update `Material` struct in `src/renderer/ray_trace.wgsl` ae3b2da
    - [x] Match the field changes made in Phase 1
- [x] Task: Implement material resolution logic in `ray_trace.wgsl` ae3b2da
    - [x] Define `mat_type` constant for `Blend` (e.g., 4u)
    - [x] Update `compute` or `scatter` to resolve the final material index by looping while the material type is `Blend`
- [x] Task: Update `scatter` function in `ray_trace.wgsl` to handle the resolved material ae3b2da
- [x] Task: Conductor - User Manual Verification 'Phase 2: Shader Implementation' (Protocol in workflow.md) ae3b2da

## Phase 3: Integration and Final Verification [checkpoint: ae3b2da]

Final phase to verify the end-to-end rendering of blended materials on the GPU.

- [x] Task: Run integration tests for `Blend` material on GPU ae3b2da
    - [x] Execute `cargo test --test integration_tests test_blended_materials`
    - [x] Verify that GPU output matches expected images in `tests/output/`
- [x] Task: Conductor - User Manual Verification 'Phase 3: Integration and Final Verification' (Protocol in workflow.md) ae3b2da
