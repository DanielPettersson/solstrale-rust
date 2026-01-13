# Initial Concept
To learn about Rust and WPGU by implementing a modern path tracing library

# Product Guide: Solstrale

## Target Audience
- Graphics enthusiasts and developers specifically interested in learning modern path tracing techniques and GPU integration using Rust and WGPU.

## Project Goals
- Serve as a robust educational resource for learning ray tracing concepts.
- Provide a platform for experimenting with advanced rendering techniques, specifically Open Image Denoise (OIDN) and WGPU-based post-processing.
- Deepen understanding of Rust's performance capabilities and the WGPU graphics API.

## Key Features
- **Core Engine:** Multithreaded Monte Carlo path tracing implementing global illumination, caustics, reflections, and refractions.
- **Model Support:** Efficient loading of OBJ models with full material support.
- **Acceleration:** Multithreaded BVH (Bounding Volume Hierarchy) creation for optimized rendering performance.
- **Advanced Post-Processing:**
    - AI-powered denoising via Open Image Denoise (OIDN).
    - Custom GPU-accelerated Bloom and Saturation filters implemented with WGPU compute shaders.
- **Visual Fidelity:** Support for bump mapping and realistic light attenuation.

## Long-Term Vision
- To become the premier educational resource and reference implementation for Rust developers learning modern, feature-rich path tracing and GPU compute integration.
