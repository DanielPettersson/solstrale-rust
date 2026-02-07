# Implementation Plan: Fix Triangle UV Mapping in GPU Renderer

This plan addresses the inconsistency in triangle UV mapping between the CPU and GPU renderers, specifically targeting the logic in the GPU shader.

## Phase 1: Diagnosis and Investigation
Reproduction of the issue and analysis of the root cause in the GPU shader logic.

- [x] Task: Confirm test failure on GPU. Run `cargo test test_render_uv_mapping` and examine the output `tests/output/uv.jpg`. Compare it with `tests/output/out_expected_uv.jpg`. 9ba78cd
- [x] Task: Analyze barycentric coordinate and UV interpolation logic in `src/renderer/ray_trace.wgsl` and compare with `src/hittable/triangle.rs`. 9ba78cd
- [ ] Task: Conductor - User Manual Verification 'Diagnosis and Investigation' (Protocol in workflow.md)

## Phase 2: Implementation
Correcting the GPU shader logic to match the CPU renderer's UV mapping.

- [x] Task: Fix UV mapping in `src/renderer/ray_trace.wgsl`. Possible causes include incorrect barycentric weight assignment or mismatch in vertex/UV order. 9ba78cd
- [x] Task: Verify fix with `test_render_uv_mapping` on GPU. Ensure the output matches expectations and the test passes. 9ba78cd
- [ ] Task: Conductor - User Manual Verification 'Implementation' (Protocol in workflow.md)

## Phase 3: Final Verification and Cleanup
Ensuring no regressions and overall system stability.

- [ ] Task: Run all integration tests using `cargo test` to ensure the fix doesn't negatively impact other scenes or primitives.
- [ ] Task: Run `./coverage.sh` to ensure high test coverage is maintained.
- [ ] Task: Conductor - User Manual Verification 'Final Verification and Cleanup' (Protocol in workflow.md)