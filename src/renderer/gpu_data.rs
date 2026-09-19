//! GPU data structures matching WGSL layout

use crate::hittable::LEAF_FLAG;
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
/// Triangle geometry touched by the traversal inner loop.
///
/// Split out from the shading attributes: the ray/triangle test needs only
/// these 48 bytes, so keeping the 80 bytes of UVs, normal and tangent frame in
/// a separate buffer cuts the bandwidth the inner loop pulls by roughly 3x.
/// Edges are stored rather than absolute vertices because the CPU already has
/// them and Moeller-Trumbore wants them directly.
pub struct TrianglePos {
    /// First vertex
    pub v0: [f32; 3],
    /// Padding to keep vec3 alignment
    pub _pad0: f32,
    /// v1 - v0
    pub e1: [f32; 3],
    /// Padding to keep vec3 alignment
    pub _pad1: f32,
    /// v2 - v0
    pub e2: [f32; 3],
    /// Padding to keep vec3 alignment
    pub _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
/// Triangle shading attributes, fetched once per ray after traversal settles.
///
/// The three shading normals ride in what used to be padding -- one word at
/// offset 44 and two at 72 -- so smooth shading costs no extra buffer, binding
/// or traversal bandwidth. `tests/gpu_data_test.rs` pins the offsets.
pub struct TriangleAttr {
    /// Geometric normal, derived from the winding
    pub normal: [f32; 3],
    /// Index of the material in the materials buffer
    pub material_index: u32,
    /// Tangent for normal mapping
    pub tangent: [f32; 3],
    /// Surface area, used for light sampling
    pub area: f32,
    /// Bi-tangent for normal mapping
    pub bi_tangent: [f32; 3],
    /// Shading normal at v0, octahedral, two snorm16 (see `pack_oct`)
    pub n0_oct: u32,
    /// Texture coordinate at v0
    pub uv0: [f32; 2],
    /// Texture coordinate at v1
    pub uv1: [f32; 2],
    /// Texture coordinate at v2
    pub uv2: [f32; 2],
    /// Shading normal at v1, octahedral, two snorm16
    pub n1_oct: u32,
    /// Shading normal at v2, octahedral, two snorm16
    pub n2_oct: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
/// Quad geometry touched by the traversal inner loop.
pub struct QuadPos {
    /// Starting corner
    pub q: [f32; 3],
    /// Plane offset along the normal
    pub d: f32,
    /// First edge vector
    pub u: [f32; 3],
    /// Padding to keep vec3 alignment
    pub _pad0: f32,
    /// Second edge vector
    pub v: [f32; 3],
    /// Padding to keep vec3 alignment
    pub _pad1: f32,
    /// Plane normal
    pub normal: [f32; 3],
    /// Padding to keep vec3 alignment
    pub _pad2: f32,
    /// Precomputed planar basis constant
    pub w: [f32; 3],
    /// Padding to keep vec3 alignment
    pub _pad3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
/// Quad shading attributes, fetched once per ray after traversal settles.
pub struct QuadAttr {
    /// Tangent for normal mapping
    pub tangent: [f32; 3],
    /// Surface area, used for light sampling
    pub area: f32,
    /// Bi-tangent for normal mapping
    pub bi_tangent: [f32; 3],
    /// Index of the material in the materials buffer
    pub material_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
/// BVH node matching the WGSL layout: a two-child node carrying *both*
/// children's bounding boxes.
///
/// Holding both boxes here is what lets the shader test the two children up
/// front, descend into the nearer one without touching the stack, and push the
/// farther one only when it can still contain a closer hit.
///
/// Each `*_meta` word is either a node index, or -- when `LEAF_FLAG` is set --
/// an inline leaf: primitive count in bits 30..24, offset into `prim_refs` in
/// bits 23..0. A leaf with count 0 is the empty child of a single-leaf root and
/// intersects nothing.
pub struct BvhNode {
    /// Left child AABB minimum
    pub left_min: [f32; 3],
    /// Left child: node index, or packed inline leaf
    pub left_meta: u32,
    /// Left child AABB maximum
    pub left_max: [f32; 3],
    /// Right child: node index, or packed inline leaf
    pub right_meta: u32,
    /// Right child AABB minimum
    pub right_min: [f32; 3],
    /// Padding to keep vec3 alignment
    pub _pad0: u32,
    /// Right child AABB maximum
    pub right_max: [f32; 3],
    /// Padding to keep vec3 alignment
    pub _pad1: u32,
}

impl Debug for BvhNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn child(meta: u32) -> String {
            if meta & LEAF_FLAG != 0 {
                format!(
                    "leaf(offset: {}, count: {})",
                    meta & 0x00FF_FFFF,
                    (meta >> 24) & 0x7F
                )
            } else {
                format!("node({})", meta)
            }
        }

        f.debug_struct("BvhNode")
            .field("left", &child(self.left_meta))
            .field("right", &child(self.right_meta))
            .finish()
    }
}

/// Reference to a primitive from a BVH leaf: type in bits 31..30, index into
/// the per-type array in bits 29..0.
pub const PRIM_TYPE_SHIFT: u32 = 30;
/// Mask for the primitive index within a [`PRIM_TYPE_SHIFT`]-tagged reference.
pub const PRIM_INDEX_MASK: u32 = (1 << PRIM_TYPE_SHIFT) - 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
/// Material structure matching WGSL layout
pub struct Material {
    /// Albedo color
    pub albedo: [f32; 3],
    /// Attenuation factor
    pub attenuation_factor: f32,
    /// Emission color
    pub emission: [f32; 3],
    /// Blend factor (0.0 - 1.0)
    pub blend_factor: f32,
    /// Fuzziness (for metal)
    pub fuzz: f32,
    /// Refraction index (for dielectric)
    pub refraction_index: f32,
    /// Material type identifier
    pub mat_type: u32,
    /// Padding
    pub _padding3: u32,
    /// Texture index (-1 for none)
    pub texture_index: i32,
    /// Normal map texture index (-1 for none)
    pub normal_texture_index: i32,
    /// Indices of the two materials to blend
    pub blend_indices: [u32; 2],
    /// UV offset for the albedo texture in the atlas
    pub albedo_offset: [f32; 2],
    /// UV scale for the albedo texture in the atlas
    pub albedo_scale: [f32; 2],
    /// UV offset for the normal texture in the atlas
    pub normal_offset: [f32; 2],
    /// UV scale for the normal texture in the atlas
    pub normal_scale: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
/// Camera structure matching WGSL layout
pub struct GpuCamera {
    /// Origin of the camera
    pub origin: [f32; 3],
    /// lens radius
    pub lens_radius: f32,
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
    /// Padding
    pub _pad3: f32,
    /// Camera u vector
    pub u: [f32; 3],
    /// Padding
    pub _pad4: f32,
    /// Camera v vector
    pub v: [f32; 3],
    /// Padding
    pub _pad5: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
/// Reference to a light source
pub struct LightRef {
    /// Type of the primitive (0=Sphere, 1=Triangle, 2=Quad)
    pub prim_type: u32,
    /// Index of the primitive in its respective buffer
    pub prim_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
/// Render configuration matching WGSL layout
pub struct GpuRenderConfig {
    /// Width of the image
    pub width: u32,
    /// Height of the image
    pub height: u32,
    /// Number of samples already accumulated into the output buffer before
    /// this dispatch. Also seeds the RNG.
    pub sample_count: u32,
    /// Maximum number of ray bounces
    pub max_depth: u32,
    /// Background color
    pub background_color: [f32; 3],
    /// Number of light sources in the scene
    pub light_count: u32,
    /// Samples traced per dispatch, accumulated in-shader
    pub samples_per_batch: u32,
    /// Minimum samples a pixel must have before adaptive sampling may skip it
    pub min_samples_per_pixel: u32,
    /// Relative standard-error threshold below which a pixel is converged
    pub variance_threshold: f32,
    /// Distinguishes successive accumulation restarts, mixed into every
    /// pixel's sampler seed, and seeded from `RenderConfig::seed`. Occupies
    /// what was padding, so the 48-byte layout is unchanged.
    pub restart_index: u32,
}
