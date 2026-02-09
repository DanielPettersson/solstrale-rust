# Implementation Plan - GPU Post-Processing Migration

Migrate post-processing effects (Bloom, Saturation) to run directly on the GPU using `wgpu::Buffer`, refactor the `PostProcessor` trait, and remove post-processing from the CPU renderer.

## Phase 1: CPU Renderer Cleanup [checkpoint: 0977744]
- [x] Task: Remove post-processing logic from `src/renderer/mod.rs`. aa2fa9a
    - [x] Update `Renderer::new` to remove post-processor initialization.
    - [x] Update `Renderer::render` to remove the post-processing loop.
    - [ ] Note: The CPU renderer will now only output raw path-traced images.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: CPU Renderer Cleanup' (Protocol in workflow.md)

## Phase 2: Trait Refactoring
- [ ] Task: Update `PostProcessor` trait in `src/post/mod.rs`
    - [ ] Change `initialize` to: `fn initialize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32)`
    - [ ] Change `post_process` to: `fn post_process(&self, encoder: &mut wgpu::CommandEncoder, buffer: &wgpu::Buffer, num_samples: u32) -> Result<(), Box<dyn Error>>`
    - [ ] Remove `width()` and `height()` from the trait as they are no longer needed by callers.
- [ ] Task: Update `PostProcessors` enum and `enum_dispatch` in `src/post/mod.rs`
- [ ] Task: Stub implementations in `src/post/bloom.rs`, `src/post/saturation.rs`, and `src/post/nop.rs` to fix compilation errors.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Trait Refactoring' (Protocol in workflow.md)

## Phase 3: GPU Renderer Integration
- [ ] Task: Update `GpuRenderer` struct in `src/renderer/gpu_renderer.rs` to store initialized post-processors.
- [ ] Task: Update `GpuRenderer::new` to initialize post-processors using the provided `device` and `queue`.
- [ ] Task: Update `GpuRenderer::render` to execute the post-processing chain.
    - [ ] Call `post_process` on each configured post-processor before copying to the `download_buffer`.
    - [ ] Note: Pass `1` as `num_samples` to post-processors because the GPU shader already averages colors.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: GPU Renderer Integration' (Protocol in workflow.md)

## Phase 4: GPU Post-Processor Implementations
- [ ] Task: Implement `NopPostProcessor` in `src/post/nop.rs` with the new trait.
- [ ] Task: Implement `SaturationPostProcessor` in `src/post/saturation.rs`.
    - [ ] Remove CPU round-trip (no `pixel_colors` slice conversion).
    - [ ] Record compute pass directly into the provided `encoder`.
- [ ] Task: Implement `BloomPostProcessor` in `src/post/bloom.rs`.
    - [ ] Remove CPU round-trip.
    - [ ] Record all compute passes (filter, blur X/Y, add) into the provided `encoder`.
    - [ ] Ensure threshold/intensity calculations account for the fact that `GpuRenderer` provides averaged colors.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: GPU Post-Processor Implementations' (Protocol in workflow.md)

## Phase 5: Verification & Testing
- [ ] Task: Run integration tests in `tests/integration_tests.rs` and ensure they pass.
- [ ] Task: Verify code coverage and perform final cleanup.
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Verification & Testing' (Protocol in workflow.md)
