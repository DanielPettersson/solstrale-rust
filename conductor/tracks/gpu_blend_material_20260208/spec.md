# Specification: GPU Blend Material Implementation

## Overview
This track implements the `Blend` material for the GPU renderer. The `Blend` material allows for combining two materials based on a `blend_factor` (0.0 to 1.0). To maintain consistency with the existing CPU implementation, the GPU version will use probabilistic selection: for each ray interaction, one of the two materials is chosen based on the blend factor.

## Functional Requirements
- **Probabilistic Selection:** The GPU shader must use a random number to select between `material_1` and `material_2` at each hit point.
- **Support for Nesting:** The implementation must support nested `Blend` materials (e.g., a `Blend` material containing another `Blend` material) by resolving the final material in a loop within the shader.
- **Scene Flattening:** The `scene_flattener` must be updated to correctly process `Blend` materials and their children, populating the new fields in the GPU material buffer.

## Technical Requirements
- **Data Structure Update:**
    - Repurpose `_padding2` (f32) in `Material` struct to `blend_factor`.
    - Repurpose `_padding4` ([u32; 2]) in `Material` struct to `blend_indices`.
- **Shader Updates:**
    - Update `Material` struct definition in `ray_trace.wgsl`.
    - Implement a material resolution loop in `compute` or `scatter` to handle nested `Blend` materials until a terminal material type is reached.
- **Host Updates:**
    - Update `GpuMaterial` in `src/renderer/gpu_data.rs`.
    - Update `add_material` in `src/renderer/scene_flattener.rs` to handle `Materials::Blend`.

## Acceptance Criteria
- [ ] Scenes using `Blend` materials render correctly on the GPU.
- [ ] GPU output matches CPU output (statistically, due to probabilistic nature).
- [ ] Nested `Blend` materials work as expected.
- [ ] No regression in performance for non-blended materials.
- [ ] Integration test `test_blended_materials` in `tests/integration_tests.rs` passes with the GPU renderer.

## Out of Scope
- Linear interpolation (lerp) of material properties (this is a separate technique not used in the CPU implementation).
- Non-probabilistic blending for the GPU renderer.
