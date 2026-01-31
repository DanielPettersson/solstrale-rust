# Implementation Plan - Fix GpuRenderer Nested BVH Bug

## Phase 1: Fix Implementation [checkpoint: 8ee97d5]
- [x] Task: Create a regression test case for nested BVH flattening in `scene_flattener.rs`. [d54000a]
    - [x] Create a new test `test_flatten_scene_nested_bvh` in `src/renderer/scene_flattener.rs` that explicitly constructs a nested BVH structure and asserts that the flattened output contains the expected number of nodes and primitives.
    - [x] Run the test and confirm it fails or produces incorrect output (e.g., missing nodes).
- [x] Task: Implement recursive BVH flattening in `scene_flattener.rs`. [2820e4d]
    - [x] Modify `process_item` (or `add_primitive`) to detect `Hittables::Bvh`.
    - [x] When a `Bvh` is encountered, recursively call `process_node` instead of ignoring it or treating it as an invalid primitive.
    - [x] Ensure the return index from the recursive call is correctly linked in the parent node.
- [x] Task: Verify fix with tests. [2820e4d]
    - [x] Run the new unit test `test_flatten_scene_nested_bvh` and ensure it passes.
    - [x] Run the existing integration test `test_gpu_scene_nested_bvh` in `tests/gpu_renderer_test.rs` and ensure it passes.
    - [x] Run all other tests to ensure no regressions.
- [x] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)
