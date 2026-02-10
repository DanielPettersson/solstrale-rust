# Implementation Plan: GPU Buffer Output and Render Loop Optimization

## Phase 1: Preparation and Utilities
- [x] Task: Create a public utility function `buffer_to_image` in `src/util/mod.rs` (or a suitable utility module) that converts a `wgpu::Buffer` to an `RgbImage`. This logic should be extracted from the current `Renderer::render` implementation. ec0269d
- [x] Task: Update `tests/integration_tests.rs` to use the new `buffer_to_image` utility function. Since `RenderProgress` will no longer contain the image, the tests will need to convert the returned buffer at the end of rendering. ec0269d
- [x] Task: Conductor - User Manual Verification 'Preparation and Utilities' (Protocol in workflow.md) ec0269d

## Phase 2: Core API and Renderer Refactoring
- [x] Task: Update `RenderProgress` struct in `src/renderer/mod.rs`:
    - Remove `render_image: Option<RgbImage>`.
    - Add `output_buffer: wgpu::Buffer`. ec0269d
- [x] Task: Update `Renderer::render` in `src/renderer/mod.rs`:
    - Remove `download_buffer` usage and the image conversion logic from the main render loop.
    - Pass `self.output_buffer.clone()` into the `RenderProgress` struct for all updates.
    - Update the loop to only perform buffer copies and post-processing on the *final* sample. ec0269d
- [x] Task: Modify the `ray_trace` function in `src/lib.rs` and `Renderer::render` to return the final `wgpu::Buffer` upon successful completion. ec0269d
- [x] Task: Conductor - User Manual Verification 'Core API and Renderer Refactoring' (Protocol in workflow.md) ec0269d

## Phase 3: Verification and Finalization
- [x] Task: Run all integration tests using `./coverage.sh` or `cargo test` to ensure visual parity is maintained and no regressions are introduced. ec0269d
- [x] Task: Perform a final code review to ensure adherence to the project's code style and quality gates. ec0269d
- [x] Task: Conductor - User Manual Verification 'Verification and Finalization' (Protocol in workflow.md) ec0269d
