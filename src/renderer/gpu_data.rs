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
///
/// The two padding words are free space, not waste that has to stay: a parent
/// index and a which-child-am-I bit fit in them at the same 64 bytes, which is
/// what a stackless Laine-style restart trail would need. Worth knowing if the
/// traversal stack ever shows up as scratch traffic; not a reason to rewrite
/// the loop before it does.
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

/// Sphere, as tagged in bits 31..30 of a `prim_refs` entry and in [`LightRef`].
pub const PRIM_TYPE_SPHERE: u32 = 0;
/// Triangle, as tagged in bits 31..30 of a `prim_refs` entry and in [`LightRef`].
pub const PRIM_TYPE_TRIANGLE: u32 = 1;
/// Quad, as tagged in bits 31..30 of a `prim_refs` entry and in [`LightRef`].
pub const PRIM_TYPE_QUAD: u32 = 2;

/// [`Material::mat_type`] values. Mirrored by the `MAT_*` constants in
/// `renderer/ray_trace.wgsl`.
pub const MAT_LAMBERTIAN: u32 = 0;
/// See [`MAT_LAMBERTIAN`].
pub const MAT_METAL: u32 = 1;
/// See [`MAT_LAMBERTIAN`].
pub const MAT_DIELECTRIC: u32 = 2;
/// See [`MAT_LAMBERTIAN`].
pub const MAT_DIFFUSE_LIGHT: u32 = 3;
/// See [`MAT_LAMBERTIAN`].
pub const MAT_BLEND: u32 = 4;

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
    /// Where this material's single-scatter albedo table starts in
    /// `dielectric_energy`, entering half first. Only a dielectric that can be
    /// rough has one; everything else leaves it 0 and never reads it.
    ///
    /// Reuses what was padding, so the struct is the same 96 bytes it was and
    /// the byte-image interning in `add_material` still works unchanged.
    pub energy_offset: u32,
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
/// Reference to a light source, plus its share of the scene's emitted power.
///
/// The sampler and the MIS weight both need the probability of picking this
/// light. They read it from `select_pdf` -- the same field of the same entry --
/// rather than each computing it, which is what keeps a non-uniform selector
/// from silently biasing the image: there is no second expression to drift.
///
/// `alias_prob` and `alias_index` are Vose's alias table over `select_pdf`, so
/// selection is one 1D draw regardless of how many emitters there are. See
/// `build_alias_table` in `scene_flattener`.
pub struct LightRef {
    /// `(prim_type << PRIM_TYPE_SHIFT) | prim_index`, the packing `prim_refs`
    /// uses. Lights are sorted by it so the tracer can binary-search a hit
    /// primitive back to its entry here.
    pub prim: u32,
    /// Probability that `sample_light` picks this light: its emitted power over
    /// the scene's total, or `1 / light_count` when no light emits anything.
    pub select_pdf: f32,
    /// Chance of keeping this slot rather than jumping to `alias_index`.
    pub alias_prob: f32,
    /// The slot this one's leftover probability was filled from.
    pub alias_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
/// Render configuration matching WGSL layout.
///
/// What is *not* here is as deliberate as what is. The image size and the light
/// count are fixed for the life of a `Renderer`, so they reach the shader as
/// override constants instead -- see `Specialisation` in `renderer`. Only the
/// fields a dispatch actually varies are left.
pub struct GpuRenderConfig {
    /// Number of samples already accumulated into the output buffer before
    /// this dispatch. Also seeds the RNG.
    pub sample_count: u32,
    /// Maximum number of ray bounces
    pub max_depth: u32,
    /// Samples traced per dispatch, accumulated in-shader
    pub samples_per_batch: u32,
    /// Minimum samples a pixel must have before adaptive sampling may skip it
    pub min_samples_per_pixel: u32,
    /// Background color
    pub background_color: [f32; 3],
    /// Relative standard-error threshold below which a pixel is converged
    pub variance_threshold: f32,
    /// Distinguishes successive accumulation restarts, mixed into every
    /// pixel's sampler seed, and seeded from `RenderConfig::seed`.
    pub restart_index: u32,
    /// WGSL rounds a struct containing a `vec3<f32>` up to its 16-byte
    /// alignment, so the uniform is 48 bytes whether these three words are
    /// spelled out or not. Spelled out, `size_of` agrees with the binding size
    /// the layout asks for.
    pub _padding: [u32; 3],
}
