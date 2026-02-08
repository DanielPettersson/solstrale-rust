# Specification: GPU Implementation of Simple, Normal, and Albedo Shaders

Implement the missing shader types (`SimpleShader`, `NormalShader`, and `AlbedoShader`) in the GPU renderer to achieve parity with the CPU implementation and pass existing integration tests.

## Overview
Currently, `GpuRenderer` only supports `PathTracingShader`. The other shaders defined in `src/renderer/shader.rs` need to be implemented in `ray_trace.wgsl` and correctly dispatched from the Rust side.

## Functional Requirements
1.  **GpuRenderConfig Update:** Add a `shader_type` field to `GpuRenderConfig` in `src/renderer/gpu_data.rs` to communicate the active shader to the GPU.
2.  **Shader Mapping:** The `shader_type` will follow the order of the `Shaders` enum in `src/renderer/shader.rs`:
    *   0: `PathTracingShader`
    *   1: `AlbedoShader`
    *   2: `NormalShader`
    *   3: `SimpleShader`
3.  **WGSL Implementation:**
    *   Implement `shade_albedo`, `shade_normal`, and `shade_simple` functions in `ray_trace.wgsl`.
    *   `shade_albedo`: Returns the material's albedo color (from texture or base color).
    *   `shade_normal`: Returns the surface normal at the hit point, mapped to a color (same as CPU).
    *   `shade_simple`: Implements basic Lambertian-style shading using a fixed light direction `(1, 1, -1)`.
4.  **Shader Dispatch:** Use a `switch` statement in the `compute` function in `ray_trace.wgsl` to call the appropriate shader function based on `config.shader_type`.
5.  **Sampling Behavior:**
    *   `PathTracingShader`: Continues to use multi-sample accumulation.
    *   `Albedo`, `Normal`, `Simple`: Perform only 1 sample (no accumulation) for efficiency, matching their "quick rendering" or debug purpose.

## Non-Functional Requirements
*   **Performance:** Debug shaders should be significantly faster than path tracing on the GPU.
*   **Parity:** Output images should match the CPU-rendered counterparts within reasonable floating-point tolerances.

## Acceptance Criteria
*   The `test_shaders` integration test in `tests/integration_tests.rs` passes when run with the GPU renderer.
*   GPU-rendered images for these shaders match the `out_expected_*.jpg` files in `tests/output/`.
