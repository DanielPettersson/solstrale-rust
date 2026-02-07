# Specification: Fix Triangle UV Mapping in GPU Renderer

## Overview
The GPU renderer currently produces incorrect UV mapping for `Triangle` primitives, leading to visual inconsistencies and a failing integration test (`test_render_uv_mapping`). This track aims to identify and fix the discrepancy in how barycentric coordinates or vertex UVs are interpolated on the GPU.

## Functional Requirements
- **Correct Triangle UV Interpolation**: Update the GPU shader logic (likely in `ray_trace.wgsl`) to correctly interpolate UV coordinates across triangle surfaces based on vertex UV data.
- **Consistency with Expectations**: Ensure the resulting UV mapping is mathematically correct and consistent with standard UV mapping principles.
- **Fix Failing Test**: The `test_render_uv_mapping` test case must pass when executed using the GPU renderer.

## Non-Functional Requirements
- **Performance**: The fix should not introduce significant performance regressions in the GPU rendering pipeline.
- **Maintainability**: The implementation should remain clean and consistent with the existing GPU renderer architecture.

## Acceptance Criteria
- `test_render_uv_mapping` passes consistently on the GPU.
- Visual inspection of rendered triangles with textures (e.g., checkers) confirms correct alignment and scaling as defined in the scene.

## Out of Scope
- Pixel-perfect matching with the CPU renderer (slight variations due to floating-point precision are acceptable).
- Fixing UV mapping for other primitives (Sphere, Quad, etc.) unless they share the same underlying logic that is broken.