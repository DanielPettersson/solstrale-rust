use bytemuck::{Pod, Zeroable};

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
    /// Center of the sphere
    pub center: [f32; 3],
    /// Radius of the sphere
    pub radius: f32,
    /// Index of the material in the materials buffer
    pub material_index: u32,
    /// Padding to align to 32 bytes (16 for vec3/radius + 16 for mat_idx/pad)
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
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
/// BVH Node structure matching WGSL layout
pub struct BvhNode {
    /// Minimum point of the AABB
    pub min: [f32; 3],
    /// Left child index (or primitive index if leaf)
    pub left_child_index: u32,
    /// Maximum point of the AABB
    pub max: [f32; 3],
    /// Right child index (or primitive count/type if leaf)
    pub right_child_index: u32,
    /// Padding to 48 bytes
    pub _pad: [u32; 2],
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
}