//! GPU data structures matching WGSL layout

use bytemuck::{Pod, Zeroable};
use std::fmt::Debug;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
/// Ray structure matching WGSL layout
pub struct Ray {
    /// Origin of the ray
    pub origin: [f32; 3],
    /// Padding to align to 16 bytes
    pub _padding1: f32,
    /// Direction of the ray
    pub direction: [f32; 3],
    /// Padding to align to 16 bytes
    pub _padding2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
/// Sphere structure matching WGSL layout
pub struct Sphere {
    /// Center of the sphere + radius in w
    pub center_and_radius: [f32; 4],
    /// Index of the material in the materials buffer
    pub material_index: u32,
    /// Padding to align to 32 bytes
    pub _padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
/// Triangle structure matching WGSL layout
pub struct Triangle {
    /// First vertex
    pub v0: [f32; 3],
    /// Padding
    pub _pad0: f32,
    /// Second vertex
    pub v1: [f32; 3],
    /// Padding
    pub _pad1: f32,
    /// Third vertex
    pub v2: [f32; 3],
    /// Padding
    pub _pad2: f32,
    /// Normal vector (precomputed)
    pub normal: [f32; 3],
    /// Index of the material
    pub material_index: u32,
    /// Padding to 80 bytes
    pub _pad3: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
/// Quad structure matching WGSL layout
pub struct Quad {
    /// Starting corner
    pub q: [f32; 3],
    /// Padding
    pub _pad0: f32,
    /// U vector
    pub u: [f32; 3],
    /// Padding
    pub _pad1: f32,
    /// V vector
    pub v: [f32; 3],
    /// Padding
    pub _pad2: f32,
    /// Normal vector
    pub normal: [f32; 3],
    /// Padding
    pub _pad3: f32,
    /// w vector = n / dot(n, n)
    pub w: [f32; 3],
    /// d = dot(normal, Q)
    pub d: f32,
    /// Index of the material
    pub material_index: u32,
    /// Padding to 112 bytes
    pub _pad4: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
/// BVH Node structure matching WGSL layout
pub struct BvhNode {
    /// Minimum point of the AABB (as u32 bits) + Left child index
    pub min_and_left: [u32; 4],
    /// Maximum point of the AABB (as u32 bits) + Right child index
    pub max_and_right: [u32; 4],
}

impl Debug for BvhNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let leaf = &self.max_and_right[3] & 0x80000000 != 0;
        let prim_type = self.max_and_right[3] & 0x7FFFFFFF;
        f.debug_struct("BvhNode")
            .field("leaf", &leaf)
            .field("idx", if leaf { &self.min_and_left[3] } else { &-1 })
            .field("type", if leaf { &prim_type } else { &-1 })
            .field("left_idx", if leaf { &-1 } else { &self.min_and_left[3] })
            .field("right_idx", if leaf { &-1 } else { &self.max_and_right[3] })
            .finish()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
/// Material structure matching WGSL layout
pub struct Material {
    /// Albedo color
    pub albedo: [f32; 3],
    /// Padding
    pub _padding1: f32,
    /// Emission color
    pub emission: [f32; 3],
    /// Padding
    pub _padding2: f32,
    /// Fuzziness (for metal)
    pub fuzz: f32,
    /// Refraction index (for dielectric)
    pub refraction_index: f32,
    /// Material type identifier
    pub mat_type: u32,
    /// Padding
    pub _padding3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
/// Camera structure matching WGSL layout
pub struct GpuCamera {
    /// Origin of the camera
    pub origin: [f32; 3],
    /// Padding
    pub _pad0: f32,
    /// Lower left corner of the viewport
    pub lower_left_corner: [f32; 3],
    /// Padding
    pub _pad1: f32,
    /// Horizontal viewport vector
    pub horizontal: [f32; 3],
    /// Padding
    pub _pad2: f32,
    /// Vertical viewport vector
    pub vertical: [f32; 3],
    /// lens radius
    pub lens_radius: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
/// Render configuration matching WGSL layout
pub struct GpuRenderConfig {
    /// Width of the image
    pub width: u32,
    /// Height of the image
    pub height: u32,
    /// Number of samples taken so far (used for RNG seed)
    pub sample_count: u32,
    /// Padding to align to 16 bytes
    pub _pad: u32,
}
