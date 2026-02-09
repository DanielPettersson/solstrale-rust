# Implementation Plan - GPU Post-Processing Migration

Migrate post-processing effects (Bloom, Saturation) to run directly on the GPU using `wgpu::Buffer`, refactor the `PostProcessor` trait, and remove post-processing from the CPU renderer.

## Phase 1: CPU Renderer Cleanup [checkpoint: 0977744]
- [x] Task: Remove post-processing logic from `src/renderer/mod.rs`. aa2fa9a
    - [x] Update `Renderer::new` to remove post-processor initialization.
    - [x] Update `Renderer::render` to remove the post-processing loop.
    - [ ] Note: The CPU renderer will now only output raw path-traced images.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: CPU Renderer Cleanup' (Protocol in workflow.md)

## Phase 2: Trait Refactoring [checkpoint: 4e01828]
- [x] Task: Update `PostProcessor` trait in `src/post/mod.rs` 207b407
    - [x] Change `initialize` to: `fn initialize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32)`
    - [x] Change `post_process` to: `fn post_process(&self, encoder: &mut wgpu::CommandEncoder, buffer: &wgpu::Buffer, num_samples: u32) -> Result<(), Box<dyn Error>>`
    - [x] Remove `width()` and `height()` from the trait as they are no longer needed by callers.
- [x] Task: Update `PostProcessors` enum and `enum_dispatch` in `src/post/mod.rs` 207b407
- [x] Task: Stub implementations in `src/post/bloom.rs`, `src/post/saturation.rs`, and `src/post/nop.rs` to fix compilation errors. 207b407
- [x] Task: Conductor - User Manual Verification 'Phase 2: Trait Refactoring' (Protocol in workflow.md)

## Phase 3: GPU Renderer Integration [checkpoint: fe79503]
- [x] Task: Update `GpuRenderer` struct in `src/renderer/gpu_renderer.rs` to store initialized post-processors. a1497fc
- [x] Task: Update `GpuRenderer::new` to initialize post-processors using the provided `device` and `queue`. a1497fc
- [x] Task: Update `GpuRenderer::render` to execute the post-processing chain. a1497fc
    - [x] Call `post_process` on each configured post-processor before copying to the `download_buffer`.
    - [x] Note: Pass `1` as `num_samples` to post-processors because the GPU shader already averages colors.
- [x] Task: Conductor - User Manual Verification 'Phase 3: GPU Renderer Integration' (Protocol in workflow.md)

## Phase 4: GPU Post-Processor Implementations [checkpoint: eca129a]
- [x] Task: Implement `NopPostProcessor` in `src/post/nop.rs` with the new trait. 0f5f7ff
- [x] Task: Implement `SaturationPostProcessor` in `src/post/saturation.rs`. 0f5f7ff
    - [x] Remove CPU round-trip (no `pixel_colors` slice conversion).
    - [x] Record compute pass directly into the provided `encoder`.
- [x] Task: Implement `BloomPostProcessor` in `src/post/bloom.rs`. 0f5f7ff
    - [x] Remove CPU round-trip.
    - [x] Record all compute passes (filter, blur X/Y, add) into the provided `encoder`.
    - [x] Ensure threshold/intensity calculations account for the fact that `GpuRenderer` provides averaged colors.
- [x] Task: Conductor - User Manual Verification 'Phase 4: GPU Post-Processor Implementations' (Protocol in workflow.md)
- [ ] Task: Conductor - User Manual Verification 'Phase 4: GPU Post-Processor Implementations' (Protocol in workflow.md)

## Phase 5: Verification & Testing [checkpoint: 27b3b89]
- [x] Task: Run integration tests in `tests/integration_tests.rs` and ensure they pass. 0f5f7ff
- [x] Task: Verify code coverage and perform final cleanup. 0f5f7ff
- [x] Task: Conductor - User Manual Verification 'Phase 5: Verification & Testing' (Protocol in workflow.md)
