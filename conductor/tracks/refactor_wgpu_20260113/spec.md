# Track Spec: Refactor GPU Post-Processing

## Goal
Improve the efficiency and modularity of the GPU-accelerated post-processing effects (Bloom and Saturation) by optimizing resource management (specifically WGPU buffers and pipelines) and enhancing code structure.

## Current Issues
1. **Redundant Buffer Creation:** GPU buffers (weights, input pixels, intermediate buffers) are created on every call to `intermediate_post_process`.
2. **Pipeline Creation:** Pipelines are created in `initialize`, which is better than in every process call, but could still be more centralized.
3. **Resource Leakage/Overhead:** Frequent allocation/deallocation of GPU memory causes unnecessary overhead.

## Requirements
1. **Resource Reuse:** GPU buffers should be allocated once (or when dimensions change) and reused across multiple frames/calls.
2. **Encapsulation:** WGPU-related logic should be cleaner and more modular.
3. **Maintainability:** Ensure that adding new GPU post-processors is straightforward and follows a consistent pattern.
4. **Efficiency:** Minimize data transfer between CPU and GPU.

## Proposed Changes
1. **Buffer Management:** Modify `BloomPostProcessor` and `SaturationPostProcessor` to store and reuse `wgpu::Buffer` objects. Re-allocate them only if the image dimensions change in `initialize`.
2. **Common GPU Utilities:** Enhance `src/util/wgpu_util.rs` if needed to support better resource lifecycle management.
3. **Testing:** Ensure that the refactored post-processors produce identical output to the current versions (using existing integration tests).
