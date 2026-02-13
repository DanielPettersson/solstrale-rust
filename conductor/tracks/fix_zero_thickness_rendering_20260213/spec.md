# Specification: Fix Zero-Thickness Hittable Rendering Bug

## Overview
A bug was identified where quads (and potentially other hittables) that are perfectly flat in one axis (e.g., Z-plane) fail to render in the GPU renderer. This is evidenced by the failing `test_gpu_scene_quad` integration test. Shifting the flat coordinate by a tiny amount (0.1) resolves the issue, suggesting the problem lies in how zero-thickness volumes are handled in the BVH or the intersection shaders.

## Functional Requirements
- **Fix Rendering:** Ensure quads and triangles perfectly aligned with any axis (XY, YZ, or XZ planes) render correctly in the GPU renderer.
- **Generic Solution:** The fix should address the root cause in a way that applies to all hittable types (Quads, Triangles, Spheres) to prevent similar regressions.
- **Pass Existing Tests:** `test_gpu_scene_quad` and all other integration tests must pass.

## Non-Functional Requirements
- **Performance:** The fix must not introduce significant performance regressions in BVH traversal or shader intersection logic.
- **Maintainability:** Adhere to existing TDD and code style guidelines.

## Acceptance Criteria
- [ ] `test_gpu_scene_quad` passes without modifying the test scene's coordinates.
- [ ] All integration tests in `tests/integration_tests.rs` pass.
- [ ] Unit tests for "flat" quads and triangles in various orientations are added and passing.
- [ ] Code coverage for the affected modules remains above 90%.

## Out of Scope
- Optimizing general path tracing performance beyond what's necessary for this fix.
- Refactoring the entire GPU renderer architecture.
