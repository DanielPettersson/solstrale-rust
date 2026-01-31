# Specification: Fix GpuRenderer Nested BVH Bug

## Overview
A bug exists in the `GpuRenderer` where objects contained within nested Bounding Volume Hierarchies (BVHs) are not visible in the rendered output. This is suspected to be caused by the scene flattening logic in `scene_flattener.rs`, which fails to recursively process `Hittables::Bvh` variants when they are encountered as items within another BVH.

## Goals
- Ensure that nested `Bvh` structures are correctly flattened into the linear GPU buffer.
- Restore visibility of objects inside nested BVHs in the `GpuRenderer`.
- Ensure the existing regression test `test_gpu_scene_nested_bvh` passes.

## Functional Requirements
- Modify `scene_flattener.rs` to detect and recursively process `Hittables::Bvh` when encountered in `add_primitive` or `process_item`.
- The flattening should be transparent, meaning the nested BVH nodes should be integrated into the main `nodes` vector in `SceneData`.
- Maintain the existing logic for spheres, triangles, and quads.

## Non-Functional Requirements
- Performance: The recursion should not significantly impact scene flattening time for typical scenes.
- Robustness: The implementation should handle arbitrary levels of BVH nesting (within stack limits).

## Acceptance Criteria
- The test `test_gpu_scene_nested_bvh` in `tests/gpu_renderer_test.rs` passes with a similarity score above the threshold.
- No regressions in other `GpuRenderer` tests.
- Code follows project style guidelines.

## Out of Scope
- Support for `Hittables::ConstantMedium` nesting (to be addressed in a separate track if needed).
- Complex shader changes to support hierarchical BVH traversal; sticking to transparent flattening.
