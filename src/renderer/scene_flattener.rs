//! Utilities for flattening the scene graph into linear buffers for the GPU

use crate::geo::Aabb;
use crate::geo::Uv;
use crate::geo::vec3::{Vec3, ZERO_VECTOR};
use crate::hittable::{Bvh, Hittable, Hittables, LEAF_FLAG};
use crate::material::texture::{Texture, Textures};
use crate::material::{Material, Materials};
use crate::renderer::Scene;
use crate::renderer::gpu_data::{
    BvhNode as GpuBvhNode, LightRef, Material as GpuMaterial, PRIM_TYPE_SHIFT, QuadAttr, QuadPos,
    Sphere as GpuSphere, TriangleAttr, TrianglePos,
};
use crate::util::texture_processing::{AtlasLayout, TexturePacker};
use image::RgbImage;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

/// Interning caches used while flattening.
///
/// Without these the flattener emits one `GpuMaterial` per *primitive* and
/// re-scans the texture list linearly per material, so a large mesh sharing a
/// single material produced megabytes of duplicate material records that
/// thrashed the material fetch in `scatter`.
#[derive(Default)]
struct FlattenCaches {
    /// Byte image of an already-emitted `GpuMaterial` -> its index.
    material_ids: HashMap<Vec<u8>, u32, BuildHasherDefault<FxHasher>>,
    /// `Arc::as_ptr` of a decoded texture -> its index in `unique_textures`.
    texture_ids: HashMap<usize, usize, BuildHasherDefault<FxHasher>>,
}

/// The multiply-xor hash rustc uses internally, over 64-bit words.
///
/// These two maps are probed once per *primitive* -- 348k times for a scene with a
/// couple of imported meshes -- and the material key is the 96-byte image of a
/// `GpuMaterial`. SipHash's per-byte work is the wrong trade for interning our own bytes
/// into a table that never leaves this function, so it is not exposed to anything that
/// could choose keys adversarially.
#[derive(Default)]
struct FxHasher {
    hash: u64,
}

impl FxHasher {
    const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(Self::SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let chunks = bytes.as_chunks::<8>();
        for chunk in chunks.0 {
            self.add(u64::from_le_bytes(*chunk));
        }
        let remainder = chunks.1;
        if !remainder.is_empty() {
            let mut buf = [0u8; 8];
            buf[..remainder.len()].copy_from_slice(remainder);
            self.add(u64::from_le_bytes(buf));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

/// Container for all scene data flattened for the GPU
pub struct SceneData {
    /// Flattened BVH nodes
    pub nodes: Vec<GpuBvhNode>,
    /// Primitive references indexed by BVH leaves: type in bits 31..30, index in bits 29..0
    pub prim_refs: Vec<u32>,
    /// Spheres
    pub spheres: Vec<GpuSphere>,
    /// Triangle geometry, read during BVH traversal
    pub triangle_pos: Vec<TrianglePos>,
    /// Triangle shading attributes, read once per ray after traversal
    pub triangle_attr: Vec<TriangleAttr>,
    /// Quad geometry, read during BVH traversal
    pub quad_pos: Vec<QuadPos>,
    /// Quad shading attributes, read once per ray after traversal
    pub quad_attr: Vec<QuadAttr>,
    /// Materials
    pub materials: Vec<GpuMaterial>,
    /// Unique decoded textures, shared with the scene (not copied).
    pub textures: Vec<Arc<RgbImage>>,
    /// Atlas placement for `textures`, computed once here and reused by the renderer.
    pub atlas_layout: Option<AtlasLayout>,
    /// Light sources
    pub lights: Vec<LightRef>,
}

/// Flattens the scene into linear buffers
pub fn flatten_scene(scene: &Scene) -> SceneData {
    let mut data = SceneData {
        nodes: Vec::new(),
        prim_refs: Vec::new(),
        spheres: Vec::new(),
        triangle_pos: Vec::new(),
        triangle_attr: Vec::new(),
        quad_pos: Vec::new(),
        quad_attr: Vec::new(),
        materials: Vec::new(),
        textures: Vec::new(),
        atlas_layout: None,
        lights: Vec::new(),
    };

    let mut caches = FlattenCaches::default();
    let mut unique_textures: Vec<Arc<RgbImage>> = Vec::new();
    collect_unique_textures(&scene.world, &mut unique_textures);

    let atlas_layout = if !unique_textures.is_empty() {
        let packer = TexturePacker::new(8192, 8192);
        let dims: Vec<(u32, u32)> = unique_textures
            .iter()
            .map(|img| (img.width(), img.height()))
            .collect();
        Some(
            packer
                .pack(&dims)
                .expect("Failed to pack textures into atlas"),
        )
    } else {
        None
    };

    // Process world
    match &scene.world {
        Hittables::Bvh(bvh) => {
            emit_bvh(
                bvh,
                &mut data,
                &unique_textures,
                atlas_layout.as_ref(),
                &mut caches,
            );
        }
        world => {
            // A bare primitive as the whole scene: emit a root whose left child
            // is a one-primitive leaf and whose right child is an empty leaf.
            let (prim_index, prim_type) = add_primitive(
                world,
                &mut data,
                &unique_textures,
                atlas_layout.as_ref(),
                &mut caches,
            );
            data.prim_refs
                .push((prim_type << PRIM_TYPE_SHIFT) | prim_index);

            let bbox = world.bounding_box();
            data.nodes.push(GpuBvhNode {
                left_min: aabb_min(bbox),
                left_meta: leaf_meta(0, 1),
                left_max: aabb_max(bbox),
                right_meta: leaf_meta(0, 0),
                right_min: [0.0; 3],
                _pad0: 0,
                right_max: [0.0; 3],
                _pad1: 0,
            });
        }
    }

    // Hand the renderer the shared Arcs and the layout we already computed above;
    // it used to deep-copy every decoded image and re-run the identical packing.
    data.textures = unique_textures;
    data.atlas_layout = atlas_layout;

    data
}

fn collect_unique_textures(hittable: &Hittables, unique_textures: &mut Vec<Arc<RgbImage>>) {
    match hittable {
        Hittables::Sphere(s) => collect_material_textures(&s.mat, unique_textures),
        Hittables::Triangle(t) => collect_material_textures(&t.mat, unique_textures),
        Hittables::Quad(q) => collect_material_textures(&q.mat, unique_textures),
        Hittables::Bvh(bvh) => {
            for prim in &bvh.prims {
                collect_unique_textures(prim, unique_textures);
            }
        }
    }
}

fn collect_material_textures(material: &Materials, unique_textures: &mut Vec<Arc<RgbImage>>) {
    match material {
        Materials::Lambertian(m) => {
            collect_texture(&m.albedo, unique_textures);
            if let Some(n) = &m.normal {
                collect_texture(n, unique_textures);
            }
        }
        Materials::Metal(m) => {
            collect_texture(&m.albedo, unique_textures);
            if let Some(n) = &m.normal {
                collect_texture(n, unique_textures);
            }
        }
        Materials::Dielectric(m) => {
            collect_texture(&m.albedo, unique_textures);
            if let Some(n) = &m.normal {
                collect_texture(n, unique_textures);
            }
        }
        Materials::DiffuseLight(m) => {
            collect_texture(&m.tex, unique_textures);
        }
        Materials::Blend(b) => {
            collect_material_textures(&b.material_1, unique_textures);
            collect_material_textures(&b.material_2, unique_textures);
        }
    }
}

fn collect_texture(tex: &Textures, unique_textures: &mut Vec<Arc<RgbImage>>) {
    if let Textures::ImageMap(im) = tex {
        let img = im.get_image();
        if !unique_textures
            .iter()
            .any(|existing| Arc::ptr_eq(existing, &img))
        {
            unique_textures.push(img.clone());
        }
    }
}

/// Copies a built [`Bvh`] into the flat GPU buffers.
///
/// `bvh.prims` is already in leaf order, so emitting them in order makes
/// `prim_refs` line up exactly with the leaf offsets the builder baked into the
/// node metas -- no index rewriting is needed.
fn emit_bvh(
    bvh: &Bvh,
    data: &mut SceneData,
    unique_textures: &[Arc<RgbImage>],
    atlas_layout: Option<&AtlasLayout>,
    caches: &mut FlattenCaches,
) {
    data.prim_refs.reserve(bvh.prims.len());

    // The per-type buffers were growing by doubling from zero, which for a 250k-triangle
    // mesh is ~32 MB of geometry copied about twice over. One pass over the discriminants
    // is far cheaper than that, and the loop below walks the same memory anyway.
    let (mut triangles, mut quads, mut spheres) = (0, 0, 0);
    for prim in &bvh.prims {
        match prim {
            Hittables::Triangle(_) => triangles += 1,
            Hittables::Quad(_) => quads += 1,
            Hittables::Sphere(_) => spheres += 1,
            Hittables::Bvh(_) => {}
        }
    }
    data.triangle_pos.reserve(triangles);
    data.triangle_attr.reserve(triangles);
    data.quad_pos.reserve(quads);
    data.quad_attr.reserve(quads);
    data.spheres.reserve(spheres);

    for prim in &bvh.prims {
        let (prim_index, prim_type) =
            add_primitive(prim, data, unique_textures, atlas_layout, caches);
        data.prim_refs
            .push((prim_type << PRIM_TYPE_SHIFT) | prim_index);
    }

    data.nodes.reserve(bvh.nodes.len());
    for node in &bvh.nodes {
        data.nodes.push(GpuBvhNode {
            left_min: aabb_min(&node.left_box),
            left_meta: node.left_meta,
            left_max: aabb_max(&node.left_box),
            right_meta: node.right_meta,
            right_min: aabb_min(&node.right_box),
            _pad0: 0,
            right_max: aabb_max(&node.right_box),
            _pad1: 0,
        });
    }
}

fn aabb_min(a: &Aabb) -> [f32; 3] {
    [a.x.min as f32, a.y.min as f32, a.z.min as f32]
}

fn aabb_max(a: &Aabb) -> [f32; 3] {
    [a.x.max as f32, a.y.max as f32, a.z.max as f32]
}

/// Packs an inline leaf the same way the builder does.
fn leaf_meta(offset: u32, count: u32) -> u32 {
    LEAF_FLAG | (count << 24) | (offset & 0x00FF_FFFF)
}

fn add_primitive(
    hittable: &Hittables,
    data: &mut SceneData,
    unique_textures: &[Arc<RgbImage>],
    atlas_layout: Option<&AtlasLayout>,
    caches: &mut FlattenCaches,
) -> (u32, u32) {
    match hittable {
        Hittables::Sphere(s) => {
            let index = data.spheres.len() as u32;
            let mat_idx = add_material(&s.mat, data, unique_textures, atlas_layout, caches);
            data.spheres.push(GpuSphere {
                center_and_radius: [
                    s.center.x as f32,
                    s.center.y as f32,
                    s.center.z as f32,
                    s.radius as f32,
                ],
                material_index: mat_idx,
                _padding: [0; 3],
            });
            if s.mat.is_light() {
                data.lights.push(LightRef {
                    prim_type: 0,
                    prim_index: index,
                });
            }
            (index, 0) // Type 0 = Sphere
        }
        Hittables::Triangle(t) => {
            let index = data.triangle_pos.len() as u32;
            let mat_idx = add_material(&t.mat, data, unique_textures, atlas_layout, caches);
            // Edges go to the GPU as-is; the shader used to re-derive them from
            // absolute vertices that this function had reconstructed from edges.
            data.triangle_pos.push(TrianglePos {
                v0: to_array(t.v0),
                _pad0: 0.0,
                e1: to_array(t.v0v1),
                _pad1: 0.0,
                e2: to_array(t.v0v2),
                _pad2: 0.0,
            });
            data.triangle_attr.push(TriangleAttr {
                normal: to_array(t.normal),
                material_index: mat_idx,
                tangent: to_array(t.tangent),
                area: t.area as f32,
                bi_tangent: to_array(t.bi_tangent),
                _pad0: 0.0,
                uv0: [t.uv0.u, t.uv0.v],
                uv1: [t.uv1.u, t.uv1.v],
                uv2: [t.uv2.u, t.uv2.v],
                _pad1: [0.0; 2],
            });
            if t.mat.is_light() {
                data.lights.push(LightRef {
                    prim_type: 1,
                    prim_index: index,
                });
            }
            (index, 1) // Type 1 = Triangle
        }
        Hittables::Quad(q) => {
            let index = data.quad_pos.len() as u32;
            let mat_idx = add_material(&q.mat, data, unique_textures, atlas_layout, caches);
            data.quad_pos.push(QuadPos {
                q: to_array(q.q),
                d: q.d as f32,
                u: to_array(q.u),
                _pad0: 0.0,
                v: to_array(q.v),
                _pad1: 0.0,
                normal: to_array(q.normal),
                _pad2: 0.0,
                w: to_array(q.w),
                _pad3: 0.0,
            });
            data.quad_attr.push(QuadAttr {
                tangent: to_array(q.u.unit()),
                area: q.area as f32,
                bi_tangent: to_array(q.v.unit()),
                material_index: mat_idx,
            });
            if q.mat.is_light() {
                data.lights.push(LightRef {
                    prim_type: 2,
                    prim_index: index,
                });
            }
            (index, 2) // Type 2 = Quad
        }
        Hittables::Bvh(_) => (0xFFFFFFFF, 0),
    }
}

fn add_material(
    material: &Materials,
    data: &mut SceneData,
    unique_textures: &[Arc<RgbImage>],
    atlas_layout: Option<&AtlasLayout>,
    caches: &mut FlattenCaches,
) -> u32 {
    let (
        albedo_tex,
        emission_tex,
        normal_tex,
        fuzz,
        ref_idx,
        mat_type,
        attenuation_factor,
        blend_indices,
        blend_factor,
    ) = match material {
        Materials::Lambertian(m) => (
            Some(&m.albedo),
            None,
            m.normal.as_ref(),
            0.0,
            0.0,
            0,
            0.0,
            [0, 0],
            0.0,
        ),
        Materials::Metal(m) => (
            Some(&m.albedo),
            None,
            m.normal.as_ref(),
            m.fuzz as f32,
            0.0,
            1,
            0.0,
            [0, 0],
            0.0,
        ),
        Materials::Dielectric(m) => (
            Some(&m.albedo),
            None,
            m.normal.as_ref(),
            0.0,
            m.index_of_refraction as f32,
            2,
            0.0,
            [0, 0],
            0.0,
        ),
        Materials::DiffuseLight(m) => (
            None,
            Some(&m.tex),
            None,
            0.0,
            0.0,
            3,
            m.attenuation_factor.unwrap_or(0.0) as f32,
            [0, 0],
            0.0,
        ),
        Materials::Blend(b) => {
            let idx1 = add_material(&b.material_1, data, unique_textures, atlas_layout, caches);
            let idx2 = add_material(&b.material_2, data, unique_textures, atlas_layout, caches);
            (
                None,
                None,
                None,
                0.0,
                0.0,
                4,
                0.0,
                [idx1, idx2],
                b.blend_factor as f32,
            )
        }
    };

    let albedo = albedo_tex.map(sample_texture).unwrap_or(ZERO_VECTOR);
    let emission = emission_tex.map(sample_texture).unwrap_or(ZERO_VECTOR);

    let (texture_index, albedo_offset, albedo_scale) = albedo_tex
        .or(emission_tex)
        .map(|t| get_texture_info(t, unique_textures, atlas_layout, caches))
        .unwrap_or((-1, [0.0; 2], [1.0; 2]));

    let (normal_texture_index, normal_offset, normal_scale) = normal_tex
        .map(|t| get_texture_info(t, unique_textures, atlas_layout, caches))
        .unwrap_or((-1, [0.0; 2], [1.0; 2]));

    let gpu_material = GpuMaterial {
        albedo: to_array(albedo),
        attenuation_factor,
        emission: to_array(emission),
        blend_factor,
        fuzz,
        refraction_index: ref_idx,
        mat_type,
        _padding3: 0,
        texture_index,
        normal_texture_index,
        blend_indices,
        albedo_offset,
        albedo_scale,
        normal_offset,
        normal_scale,
    };

    // Intern on the exact byte image. `GpuMaterial` is `Pod` (no uninit padding),
    // so bytewise equality is exactly "identical to the GPU". Blend children are
    // resolved above, so their indices are already interned and stable by now.
    let key = bytemuck::bytes_of(&gpu_material);
    if let Some(&existing) = caches.material_ids.get(key) {
        return existing;
    }

    let index = data.materials.len() as u32;
    caches.material_ids.insert(key.to_vec(), index);
    data.materials.push(gpu_material);

    index
}

fn get_texture_info(
    tex: &Textures,
    unique_textures: &[Arc<RgbImage>],
    atlas_layout: Option<&AtlasLayout>,
    caches: &mut FlattenCaches,
) -> (i32, [f32; 2], [f32; 2]) {
    if let Textures::ImageMap(im) = tex {
        let img = im.get_image();
        let found = match caches.texture_ids.get(&(Arc::as_ptr(&img) as usize)) {
            Some(&i) => Some(i),
            None => {
                let i = unique_textures
                    .iter()
                    .position(|existing| Arc::ptr_eq(existing, &img));
                if let Some(i) = i {
                    caches.texture_ids.insert(Arc::as_ptr(&img) as usize, i);
                }
                i
            }
        };
        if let Some(i) = found {
            {
                if let Some(layout) = atlas_layout {
                    let rect = &layout.placements[i];
                    return (
                        i as i32,
                        [
                            rect.x as f32 / layout.width as f32,
                            rect.y as f32 / layout.height as f32,
                        ],
                        [
                            rect.width as f32 / layout.width as f32,
                            rect.height as f32 / layout.height as f32,
                        ],
                    );
                }
                return (i as i32, [0.0; 2], [1.0; 2]);
            }
        }
    }
    (-1, [0.0; 2], [1.0; 2])
}

fn sample_texture(tex: &Textures) -> Vec3 {
    tex.color(Uv::default())
}

fn to_array(v: Vec3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

#[cfg(test)]
mod tests {
    use crate::geo::vec3::Vec3;
    use crate::hittable::LEAF_FLAG;
    use crate::hittable::{Bvh, Hittables, Sphere};
    use crate::material::texture::SolidColor;
    use crate::material::{Lambertian, Materials};
    use crate::renderer::scene_flattener::flatten_scene;
    use crate::renderer::{RenderConfig, Scene};

    #[test]
    fn test_flatten_scene_simple() {
        let mat =
            Materials::Lambertian(Lambertian::new(SolidColor::new(1.0, 0.0, 0.0).into(), None));
        let sphere = Sphere::new(Vec3::new(0., 0., -2.), 1.0, mat);
        let scene = Scene {
            world: Hittables::Bvh(Bvh::new(vec![Hittables::Sphere(sphere)])),
            camera: Default::default(),
            background_color: Default::default(),
            render_config: RenderConfig::default(),
        };

        let data = flatten_scene(&scene);

        assert_eq!(data.spheres.len(), 1);
        assert_eq!(data.materials.len(), 1);
        // A single primitive fits in one leaf, so there is exactly one node.
        assert_eq!(data.nodes.len(), 1);
        assert_eq!(data.prim_refs.len(), 1);

        // Check sphere data
        let s = &data.spheres[0];
        assert_eq!(s.center_and_radius, [0.0, 0.0, -2.0, 1.0]);
        assert_eq!(s.material_index, 0);

        // The one prim_ref points at sphere 0, type 0.
        assert_eq!(data.prim_refs[0], 0);

        // Root: left child is a one-primitive leaf at offset 0, right is empty.
        let n0 = &data.nodes[0];
        assert_eq!(n0.left_meta, LEAF_FLAG | (1 << 24));
        assert_eq!(n0.right_meta, LEAF_FLAG);
    }

    #[test]
    fn test_flatten_scene_nested_bvh() {
        let mat =
            Materials::Lambertian(Lambertian::new(SolidColor::new(1.0, 0.0, 0.0).into(), None));
        let sphere = Sphere::new(Vec3::new(0., 0., -2.), 1.0, mat.clone());

        let mut sub_world: Vec<Hittables> = Vec::new();
        sub_world.push(Hittables::Sphere(sphere.clone()));
        let bvh = Bvh::new(sub_world);

        let scene = Scene {
            world: Hittables::Bvh(Bvh::new(vec![Hittables::Bvh(bvh)])),
            camera: Default::default(),
            background_color: Default::default(),
            render_config: RenderConfig::default(),
        };

        let data = flatten_scene(&scene);

        // Nested BVHs are expanded into one global tree, so the nesting leaves
        // no trace: this is the same single-sphere scene as above.
        assert_eq!(data.spheres.len(), 1, "Should have 1 sphere");
        assert_eq!(data.nodes.len(), 1, "Nested BVH should be flattened away");
        assert_eq!(data.prim_refs.len(), 1);
    }

    #[test]
    fn test_flatten_scene_blend() {
        use crate::material::Blend;

        let mat1 =
            Materials::Lambertian(Lambertian::new(SolidColor::new(1.0, 0.0, 0.0).into(), None));
        let mat2 =
            Materials::Lambertian(Lambertian::new(SolidColor::new(0.0, 0.0, 1.0).into(), None));
        let blend_mat = Materials::Blend(Blend::new(mat1, mat2, 0.5));

        let sphere = Sphere::new(Vec3::new(0., 0., -2.), 1.0, blend_mat);
        let scene = Scene {
            world: Hittables::Bvh(Bvh::new(vec![Hittables::Sphere(sphere)])),
            camera: Default::default(),
            background_color: Default::default(),
            render_config: RenderConfig::default(),
        };

        let data = flatten_scene(&scene);

        // We expect:
        // 1 Sphere
        // 3 Materials: Blend, Mat1, Mat2 (order depends on implementation, but Blend is the root)

        assert_eq!(data.spheres.len(), 1);

        // Ensure we have at least 3 materials
        assert!(data.materials.len() >= 3);

        // The sphere points to the blend material.
        // NOTE: In current implementation add_material returns the index of the added material.
        // If recursive, children added first?
        let sphere_mat_idx = data.spheres[0].material_index as usize;
        let blend_gpu_mat = &data.materials[sphere_mat_idx];

        // Check properties
        assert_eq!(blend_gpu_mat.mat_type, 4, "Blend material type should be 4"); // 4 is new type for Blend
        assert_eq!(blend_gpu_mat.blend_factor, 0.5);

        let child1_idx = blend_gpu_mat.blend_indices[0];
        let child2_idx = blend_gpu_mat.blend_indices[1];

        // Verify children are valid indices
        assert!(child1_idx < data.materials.len() as u32);
        assert!(child2_idx < data.materials.len() as u32);
        assert_ne!(child1_idx, child2_idx);
        assert_ne!(child1_idx, sphere_mat_idx as u32);
        assert_ne!(child2_idx, sphere_mat_idx as u32);
    }

    #[test]
    fn test_flatten_scene_with_texture_atlas() {
        use crate::material::texture::ImageMap;
        use image::RgbImage;
        use std::sync::Arc;

        let img = Arc::new(RgbImage::new(100, 100));
        let mat = Materials::Lambertian(Lambertian::new(ImageMap::new(img).into(), None));

        let sphere = Sphere::new(Vec3::new(0., 0., -2.), 1.0, mat);
        let scene = Scene {
            world: Hittables::Bvh(Bvh::new(vec![Hittables::Sphere(sphere)])),
            camera: Default::default(),
            background_color: Default::default(),
            render_config: RenderConfig::default(),
        };

        let data = flatten_scene(&scene);

        assert_eq!(data.materials.len(), 1);
        let m = &data.materials[0];

        // With one 100x100 texture:
        // width aligned to 64 is 128. height is 100.
        // offset should be [0, 0]
        // scale should be [100/128, 100/100]
        assert_eq!(m.albedo_offset, [0.0, 0.0]);
        assert_eq!(m.albedo_scale, [100.0 / 128.0, 100.0 / 100.0]);
    }
}
