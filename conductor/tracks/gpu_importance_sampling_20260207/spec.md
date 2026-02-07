# Specification: GPU Importance Sampling for Lambertian Material

## Overview
Implement importance sampling in the GPU renderer for Lambertian materials, mirroring the functionality of the CPU renderer. This involves updating the GPU shader logic to use a mixture PDF approach, sampling both the material's cosine distribution and the explicit light sources in the scene.

## Functional Requirements
- **Mixture PDF Implementation**: Implement a strategy in the GPU shader (`ray_trace.wgsl`) to sample rays based on a mixture of:
    - The material's PDF (Cosine-weighted for Lambertian).
    - The lights' PDF (sampling emissive objects like Spheres, Triangles, Quads).
- **Light Sampling**: Ensure the shader can identify and sample light sources (emissive materials) in the scene.
- **Scattering Logic Update**: Refactor the Lambertian scattering logic in the shader to calculate the scattered ray direction and probability based on the mixture PDF.
- **Match CPU Behavior**: The visual result and convergence behavior should closely match the CPU renderer.

## Non-Functional Requirements
- **Performance**: The sampling logic should be efficient to minimize the impact on rendering performance.
- **Code Consistency**: The shader implementation should be structurally similar to the CPU implementation where possible to aid in maintainability.

## Acceptance Criteria
- The failing tests in `integration_tests.rs` (likely `test_render_light_attenuation` or similar, if they are currently failing due to this missing feature) must pass.
- Visual comparison between CPU and GPU renders for scenes with Lambertian materials and lights shows consistent lighting and noise characteristics.

## Out of Scope
- Implementing importance sampling for materials other than Lambertian (unless they share the exact same logic).
- Advanced PDF optimizations beyond the standard mixture model used in the CPU renderer.