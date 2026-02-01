# Specification: GPU Imagemap Texture Support

## Overview
This track implements support for image-based textures (imagemaps) in the GPU-accelerated path tracer. Currently, the GPU renderer only supports solid colors. This enhancement will allow complex materials using textures (e.g., JPEG/PNG maps) to be rendered on the GPU, matching the capabilities of the CPU renderer and fixing the currently failing `test_texture_map` integration test.

## Functional Requirements
1.  **Texture Array Integration:** Implement a `texture_2d_array` in the GPU shader (`ray_trace.wgsl`) to store all scene textures.
2.  **Standardized Resolution:** All textures will be automatically resized to a fixed resolution of 1024x1024 pixels before being uploaded to the GPU to satisfy `texture_2d_array` requirements.
3.  **UV Mapping:**
    *   **Triangles:** Update the GPU `Triangle` data structure to include per-vertex UV coordinates.
    *   **Quads:** Calculate UV coordinates dynamically in the shader based on hit position, matching CPU logic.
    *   **Spheres:** Calculate UV coordinates dynamically in the shader using spherical mapping, matching CPU logic.
4.  **Material System Expansion:** Add a `texture_index` field to the GPU `Material` struct.
    *   `texture_index >= 0`: Sample the texture array at the given index.
    *   `texture_index == -1`: Use the material's albedo (solid color).
5.  **Scene Flattening:** Update the scene flattener to identify all unique textures, build the GPU texture array, and map material texture indices correctly.

## Non-Functional Requirements
- **Performance:** Texture sampling should be efficient within the compute shader.
- **Compatibility:** Maintain existing solid-color material support.

## Acceptance Criteria
- The `test_texture_map` integration test passes when running on the GPU.
- Visual parity (within reasonable floating-point/sampling noise) between CPU and GPU renders for textured scenes.
- Successful rendering of scenes containing both textured and non-textured materials.

## Out of Scope
- Support for textures larger or smaller than the standardized 1024x1024 without resizing.
- Procedural textures (other than solid color).
- Mipmapping or advanced texture filtering (linear filtering is sufficient for this stage).
