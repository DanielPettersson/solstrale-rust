# Implementation Plan - GPU Shaders (Simple, Normal, Albedo)

Implement `SimpleShader`, `NormalShader`, and `AlbedoShader` in the GPU renderer to match CPU functionality and pass integration tests.

## Phase 1: GPU Data and Renderer Infrastructure [checkpoint: 3e431a3]
- [x] Task: Update `GpuRenderConfig` and `GpuRenderer` to support `shader_type` c89e884
    - [ ] Add `shader_type: u32` to `GpuRenderConfig` in `src/renderer/gpu_data.rs`
    - [ ] Update `GpuRenderer::new` in `src/renderer/gpu_renderer.rs` to initialize `GpuRenderConfig` with correct `shader_type` based on `scene.render_config.shader`
    - [ ] Update `GpuRenderer::render` in `src/renderer/gpu_renderer.rs` to update `GpuRenderConfig` with correct `max_depth` and `shader_type` during the render loop
- [x] Task: Conductor - User Manual Verification 'Phase 1: GPU Data and Renderer Infrastructure' (Protocol in workflow.md) 3e431a3

## Phase 2: WGSL Shader Implementation [checkpoint: ad0ce10]
- [x] Task: Add shader functions and dispatch logic to `ray_trace.wgsl` 3fef898
    - [ ] Implement `shade_albedo` function in `src/renderer/ray_trace.wgsl`
    - [ ] Implement `shade_normal` function in `src/renderer/ray_trace.wgsl`
    - [ ] Implement `shade_simple` function in `src/renderer/ray_trace.wgsl`
    - [ ] Update `compute` function in `src/renderer/ray_trace.wgsl` to use `switch` for shader selection and enforce 1-sample limit for non-path-tracing shaders
- [x] Task: Conductor - User Manual Verification 'Phase 2: WGSL Shader Implementation' (Protocol in workflow.md) ad0ce10

## Phase 3: Integration and Verification [checkpoint: 8df3336]
- [x] Task: Verify implementation with `test_shaders` d4ab6ee
    - [ ] Run `cargo test --test integration_tests test_shaders` and confirm all shaders pass comparison
- [x] Task: Conductor - User Manual Verification 'Phase 3: Integration and Verification' (Protocol in workflow.md) 8df3336
