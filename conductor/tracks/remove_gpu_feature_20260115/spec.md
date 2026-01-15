# Specification: Remove `gpu` Feature Flag

## Overview
The `gpu` feature flag was originally used to make WGPU-related functionality optional. However, since `wgpu` supports software rendering (running on the CPU), this optionality is no longer necessary and adds complexity to the codebase and build process. This track aims to make WGPU a standard, required dependency and remove redundant CPU-only implementations.

## Functional Requirements
- Remove the `gpu` feature from `Cargo.toml`.
- Make `wgpu`, `pollster`, and `bytemuck` required dependencies (non-optional).
- Remove all `#[cfg(feature = "gpu")]` attributes related to WGPU across the codebase.
- **Remove CPU-specific implementations that are currently guarded by `#[cfg(not(feature = "gpu"))]`.** These fallbacks are no longer needed as WGPU will be available universally.
- Ensure that WGPU-related code is always compiled and available.

## Non-Functional Requirements
- **CI Integration:** CI must now build and test WGPU-related code.
- **Code Cleanliness:** Simplify the codebase by removing conditional compilation boilerplate and redundant fallback implementations.

## Acceptance Criteria
- [ ] `Cargo.toml` no longer contains the `gpu` feature.
- [ ] `wgpu`, `pollster`, and `bytemuck` are standard dependencies.
- [ ] All WGPU-related `#[cfg(feature = "gpu")]` guards are removed from `src/`.
- [ ] Code guarded by `#[cfg(not(feature = "gpu"))]` (CPU fallbacks) is deleted.
- [ ] Project builds successfully without any feature flags.
- [ ] Integration tests pass in the current environment.

## Out of Scope
- Optimizing WGPU performance for CPU rendering.
- Removing other features like `oidn-postprocessor`.
