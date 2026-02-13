# Implementation Plan: Fix Zero-Thickness Hittable Rendering Bug

## Phase 1: Investigation & Reproduce
- [ ] Task: Confirm failure of `test_gpu_scene_quad` in `tests/integration_tests.rs`.
- [ ] Task: Inspect `src/hittable/quad.rs` and `src/hittable/triangle.rs` bounding box (AABB) generation logic.
- [ ] Task: Inspect `src/hittable/bvh.rs` to see how it handles AABBs with zero width/height/depth.
- [ ] Task: Inspect `src/renderer/ray_trace.wgsl` intersection logic for potential division by zero or precision issues with axis-aligned surfaces.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Investigation & Reproduce' (Protocol in workflow.md)

## Phase 2: Fix & Verify (TDD)
- [ ] Task: Create new unit tests for AABB generation of axis-aligned quads and triangles to verify they provide sufficient volume for the BVH.
- [ ] Task: Implement "padding" for AABBs in `src/hittable` to ensure all bounding boxes have a minimum thickness in all dimensions.
- [ ] Task: Update `src/renderer/ray_trace.wgsl` or BVH data structures if necessary to ensure robust intersection with axis-aligned surfaces.
- [ ] Task: Verify all existing tests pass, specifically `test_gpu_scene_quad`.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Fix & Verify (TDD)' (Protocol in workflow.md)

## Phase 3: Quality Assurance
- [ ] Task: Run `./coverage.sh` and ensure coverage for `src/hittable` and `src/renderer` remains > 90%.
- [ ] Task: Run `cargo clippy` and `cargo fmt` to ensure code quality.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Quality Assurance' (Protocol in workflow.md)
