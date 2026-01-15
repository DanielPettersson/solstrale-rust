# Plan: Remove `gpu` Feature Flag

## Phase 1: Configuration & CI [checkpoint: f98b144]
Update the project configuration to make WGPU standard and ensure CI verifies it.
- [x] Task: Update `Cargo.toml`.
    - [x] Move `wgpu`, `pollster`, and `bytemuck` from optional to standard dependencies.
    - [x] Remove the `gpu` feature definition.
    - [x] Remove `gpu` from default features.
- [x] Task: Update `ci.yaml`.
    - [x] Remove `--no-default-features` flag from the coverage generation step to ensure all code is tested.
- [x] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

## Phase 2: Refactor Saturation Post-Processing [checkpoint: 9f1c44e]
Remove the CPU fallback for saturation and enforce the GPU implementation.
- [x] Task: Remove `src/post/saturation_cpu.rs`. 87130e0
- [x] Task: Update `src/post/saturation_gpu.rs`.
    - [x] Remove `#![cfg(feature = "gpu")]`.
- [x] Task: Update `src/post/mod.rs`.
    - [x] Remove `cfg` attributes for saturation modules.
    - [x] Unconditionally import and use the GPU saturation implementation.
- [x] Task: Verify tests.
    - [x] Ensure `post_processing_benchmark` and integration tests still pass.
- [x] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: Refactor Bloom Post-Processing [checkpoint: 7d9eeb2]
Remove the CPU fallback for bloom and enforce the GPU implementation.
- [x] Task: Remove `src/post/bloom_cpu.rs`. 8f136fa
- [x] Task: Update `src/post/bloom_gpu.rs`.
    - [x] Remove `#![cfg(feature = "gpu")]`.
- [x] Task: Update `src/post/mod.rs`.
    - [x] Remove `cfg` attributes for bloom modules.
    - [x] Unconditionally import and use the GPU bloom implementation.
- [x] Task: Verify tests.
    - [x] Ensure `post_processing_benchmark` and integration tests still pass.
- [x] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)

## Phase 4: Final Cleanup & Verification [checkpoint: b4e46db]
Clean up any remaining feature flags and verify the entire system.
- [x] Task: Remove remaining feature guards.
    - [x] Update `src/util/mod.rs` (unguard `wgpu_util`).
    - [x] Search for and remove any other lingering `feature = "gpu"` checks.
- [x] Task: Run full integration suite.
    - [x] Execute `cargo test` to ensure stability.
- [x] Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)
