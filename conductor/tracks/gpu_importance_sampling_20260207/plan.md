# Implementation Plan: GPU Importance Sampling for Lambertian Material

This plan outlines the steps to implement importance sampling for Lambertian materials in the GPU renderer, matching the CPU's mixture PDF approach.

## Phase 1: Preparation and Analysis [checkpoint: 6235bcf]
Understand the current CPU implementation and prepare the GPU environment.

- [x] Task: Analyze CPU PDF implementation. Study `src/pdf.rs`, `src/material/mod.rs` (Lambertian scattering), and how lights are collected. 82b4292
- [x] Task: Identify failing tests. Run integration tests to pinpoint which ones specifically require importance sampling to pass (or converge faster). 82b4292
- [x] Task: Plan WGSL data structures. Determine how to pass light indices (for the light PDF) to the GPU. 82b4292
- [ ] Task: Conductor - User Manual Verification 'Preparation and Analysis' (Protocol in workflow.md)

## Phase 2: Shader Infrastructure for PDFs [checkpoint: 12873a4]
Implement the necessary PDF functions and data structures in the WGSL shader.

- [x] Task: Implement `OrthonormalBasis` struct and functions in `src/renderer/ray_trace.wgsl` (if not already present) for Cosine PDF. dda8bc3
- [x] Task: Implement `CosinePdf` logic in WGSL (generate random direction, calculate PDF value). dda8bc3
- [x] Task: Implement `HittablePdf` logic in WGSL for lights (Sphere, Triangle, Quad PDF values and random generation). dda8bc3
- [x] Task: Implement `MixturePdf` logic in WGSL to combine material and light sampling. dda8bc3
- [ ] Task: Conductor - User Manual Verification 'Shader Infrastructure for PDFs' (Protocol in workflow.md)

## Phase 3: Integration and Rendering [checkpoint: 588da58]
Integrate the PDF logic into the main rendering loop and verify results.

- [x] Task: Update `SceneData` and `GpuRenderer` to collect and pass a list of light indices to the GPU. 12873a4
- [x] Task: Modify the `scatter` function in `src/renderer/ray_trace.wgsl` for Lambertian materials to use the `MixturePdf`. 9d6bc1f
- [x] Task: Update the ray generation and color accumulation logic to account for the PDF probability (weighting the sample). 9d6bc1f
- [x] Task: Verify with tests. Run the identified integration tests and compare results. 9d6bc1f
- [ ] Task: Conductor - User Manual Verification 'Integration and Rendering' (Protocol in workflow.md)

## Phase 4: Final Verification and Cleanup
Ensure code quality and performance.

- [ ] Task: Run full integration test suite.
- [ ] Task: Compare performance and visual quality with CPU renderer.
- [ ] Task: Conductor - User Manual Verification 'Final Verification and Cleanup' (Protocol in workflow.md)