//! Utilities for flattening the scene graph into linear buffers for the GPU

use crate::geo::Aabb;
use crate::geo::Uv;
use crate::geo::vec3::{Vec3, ZERO_VECTOR};
use crate::hittable::{Bvh, Hittable, Hittables, LEAF_FLAG};
use crate::material::texture::{Texture, Textures};
use crate::material::{Material, Materials};
use crate::renderer::Scene;
use crate::renderer::dielectric_energy::build_table;
use crate::renderer::gpu_data::{
    BvhNode as GpuBvhNode, LightRef, MAT_BLEND, MAT_DIELECTRIC, MAT_DIFFUSE_LIGHT, MAT_LAMBERTIAN,
    MAT_METAL, Material as GpuMaterial, PRIM_TYPE_QUAD, PRIM_TYPE_SHIFT, PRIM_TYPE_SPHERE,
    PRIM_TYPE_TRIANGLE, QuadAttr, QuadPos, Sphere as GpuSphere, TriangleAttr, TrianglePos,
};
use crate::util::rgb_color::srgb_to_vec3;
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
    /// `Arc::as_ptr` of a decoded texture -> the mean of its texels, for
    /// emitters that carry an image. An emissive mesh shares one texture across
    /// hundreds of triangles, and scanning it per triangle is not affordable.
    texture_means: HashMap<usize, Vec3, BuildHasherDefault<FxHasher>>,
    /// An index of refraction's bit pattern -> where its energy table starts.
    ///
    /// Keyed on the index alone because that is all the table depends on:
    /// roughness and angle are its two axes, so five glass spheres across the
    /// roughness range at 1.5 share one table and build it once.
    energy_offsets: HashMap<u64, u32, BuildHasherDefault<FxHasher>>,
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

/// Side of the square texture atlas. 8192 is the `max_texture_dimension_2d`
/// every backend we target guarantees.
const ATLAS_SIZE: u32 = 8192;

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
    /// Unique decoded textures at atlas size, shared with the scene unless they
    /// had to be scaled down to fit.
    pub textures: Vec<Arc<RgbImage>>,
    /// Atlas placement for `textures`, computed once here and reused by the renderer.
    pub atlas_layout: Option<AtlasLayout>,
    /// Light sources, sorted by `LightRef::prim` so the tracer can
    /// binary-search a primitive back to the lights array, and carrying the
    /// power-proportional selection distribution `build_alias_table` computed.
    pub lights: Vec<LightRef>,
    /// Single-scatter albedo tables for the scene's dielectrics, concatenated
    /// and indexed by `Material::energy_offset`. Empty when nothing in the
    /// scene can present a rough dielectric.
    pub dielectric_energy: Vec<f32>,
    /// Whether `prim_refs[k]` is `(PRIM_TYPE_TRIANGLE << 30) | k` at every `k`.
    ///
    /// True exactly when the scene is nothing but triangles: every primitive
    /// contributes one reference and one entry to its own type's array, and
    /// both are appended in the same pass, so equal lengths leave no room for a
    /// sphere or a quad. The tracer is compiled against this, and then the
    /// innermost traversal loop reads the triangle's address straight out of
    /// the leaf slot instead of chasing a reference to it.
    pub prim_refs_are_identity: bool,
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
        dielectric_energy: Vec::new(),
        prim_refs_are_identity: false,
    };

    let mut caches = FlattenCaches::default();
    let mut unique_textures: Vec<Arc<RgbImage>> = Vec::new();
    collect_unique_textures(&scene.world, &mut unique_textures);

    // `unique_textures` stays the identity list the material lookup matches the
    // scene's `Arc`s against; `atlas_textures` is what gets blitted, which is the
    // same images unless they had to shrink to fit.
    let (atlas_layout, atlas_textures) = if !unique_textures.is_empty() {
        let packer = TexturePacker::new(ATLAS_SIZE, ATLAS_SIZE);
        let (layout, placed) = packer
            .pack_images(&unique_textures)
            .expect("Failed to pack textures into atlas");
        (Some(layout), placed)
    } else {
        (None, Vec::new())
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
    data.textures = atlas_textures;
    data.atlas_layout = atlas_layout;

    // Emitters are appended as their primitives are, which interleaves the
    // types. Sorting them by the same packed key `prim_refs` uses is what lets
    // the tracer binary-search a hit primitive back to "is this an emitter?"
    // instead of asking every light.
    data.lights.sort_unstable_by_key(|light| light.prim);
    build_alias_table(&mut data.lights);

    data.prim_refs_are_identity = data.triangle_pos.len() == data.prim_refs.len();
    debug_assert!(
        !data.prim_refs_are_identity
            || data
                .prim_refs
                .iter()
                .enumerate()
                .all(|(k, &r)| r == (PRIM_TYPE_TRIANGLE << PRIM_TYPE_SHIFT) | k as u32),
        "prim_refs was reported as the identity map but is not"
    );

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

/// A light's entry, with `select_pdf` holding its raw emitted power until
/// [`build_alias_table`] normalises the set.
///
/// Power for a one-sided diffuse emitter is `pi * A * luminance(L)`. A sphere
/// emits over its whole surface, hence `A = 4 pi r^2` at the call site.
fn light_ref(
    prim_type: u32,
    prim_index: u32,
    material: &Materials,
    area: f64,
    caches: &mut FlattenCaches,
) -> LightRef {
    // Only a DiffuseLight answers true to `is_light`, which is what gates every
    // call here; a Blend holding one is not in `lights` at all.
    let radiance = match material {
        Materials::DiffuseLight(m) => mean_color(&m.tex, caches),
        _ => ZERO_VECTOR,
    };

    LightRef {
        prim: (prim_type << PRIM_TYPE_SHIFT) | prim_index,
        select_pdf: (std::f64::consts::PI * area * luminance(radiance)) as f32,
        alias_prob: 0.,
        alias_index: 0,
    }
}

/// Rec. 709 luma: a light's colour reduced to the one number its selection
/// probability ranks it by.
fn luminance(c: Vec3) -> f64 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

/// The average colour a texture emits.
///
/// `add_material` flattens a texture to the single texel at UV (0, 0) for the
/// GPU's `emission` slot, and ranking a textured emitter by a corner pixel is
/// the kind of silent wrongness that never shows up as a crash -- so an image
/// is averaged over every texel instead, sRGB-decoded per texel to match
/// `ImageMap::color`.
fn mean_color(tex: &Textures, caches: &mut FlattenCaches) -> Vec3 {
    let Textures::ImageMap(im) = tex else {
        return sample_texture(tex);
    };

    let image = im.get_image();
    let key = Arc::as_ptr(&image) as usize;
    if let Some(&mean) = caches.texture_means.get(&key) {
        return mean;
    }

    let texels = (image.width() * image.height()).max(1) as f64;
    let mean = image
        .pixels()
        .fold(ZERO_VECTOR, |acc, p| acc + srgb_to_vec3(p))
        / texels;
    caches.texture_means.insert(key, mean);
    mean
}

/// Turns the raw powers sitting in `select_pdf` into the normalised selection
/// distribution and Vose's alias table over it, in O(L).
///
/// The alias table is what keeps selection at one 1D draw as the light count
/// grows: `sample_light` scales its draw by `L`, takes the integer part as a
/// slot and the fraction as the coin that decides between that slot and its
/// alias. A linear CDF scan would cost the dimension budget nothing either, but
/// it costs O(L) per bounce, which is the thing an emissive mesh makes matter.
///
/// Falls back to uniform when nothing emits -- a `DiffuseLight` may be black,
/// and is still a light as far as `is_light` is concerned. A distribution of
/// zeroes has no normalisation, and every direction to such a light carries no
/// radiance anyway.
fn build_alias_table(lights: &mut [LightRef]) {
    let n = lights.len();
    if n == 0 {
        return;
    }

    let total: f64 = lights.iter().map(|l| l.select_pdf as f64).sum();
    let uniform = !total.is_finite() || total <= 0.;

    // Probabilities scaled by `n`, so a slot is under- or over-full against 1
    // rather than against 1/n. This is the array Vose consumes.
    let mut scaled: Vec<f64> = if uniform {
        vec![1.; n]
    } else {
        lights
            .iter()
            .map(|l| l.select_pdf as f64 / total * n as f64)
            .collect()
    };

    // Defaults first: every slot keeps itself. Vose's leftovers -- slots whose
    // scaled probability is 1 up to rounding -- are meant to end up exactly
    // here, so the loop below simply never touches them.
    for (i, light) in lights.iter_mut().enumerate() {
        light.select_pdf = (scaled[i] / n as f64) as f32;
        light.alias_prob = 1.;
        light.alias_index = i as u32;
    }

    let (mut small, mut large): (Vec<usize>, Vec<usize>) = (0..n).partition(|&i| scaled[i] < 1.);

    while let (Some(&s), Some(&l)) = (small.last(), large.last()) {
        small.pop();
        large.pop();

        lights[s].alias_prob = scaled[s] as f32;
        lights[s].alias_index = l as u32;

        // `l` donated what `s` was short of, and goes back on whichever list it
        // now belongs to.
        scaled[l] -= 1. - scaled[s];
        if scaled[l] < 1. {
            small.push(l);
        } else {
            large.push(l);
        }
    }
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
                let area = 4. * std::f64::consts::PI * s.radius * s.radius;
                data.lights
                    .push(light_ref(PRIM_TYPE_SPHERE, index, &s.mat, area, caches));
            }
            (index, PRIM_TYPE_SPHERE)
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
                n0_oct: pack_oct(t.n0),
                uv0: [t.uv0.u, t.uv0.v],
                uv1: [t.uv1.u, t.uv1.v],
                uv2: [t.uv2.u, t.uv2.v],
                n1_oct: pack_oct(t.n1),
                n2_oct: pack_oct(t.n2),
            });
            if t.mat.is_light() {
                data.lights
                    .push(light_ref(PRIM_TYPE_TRIANGLE, index, &t.mat, t.area, caches));
            }
            (index, PRIM_TYPE_TRIANGLE)
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
                data.lights
                    .push(light_ref(PRIM_TYPE_QUAD, index, &q.mat, q.area, caches));
            }
            (index, PRIM_TYPE_QUAD)
        }
        Hittables::Bvh(_) => (0xFFFFFFFF, PRIM_TYPE_SPHERE),
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
            MAT_LAMBERTIAN,
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
            MAT_METAL,
            0.0,
            [0, 0],
            0.0,
        ),
        Materials::Dielectric(m) => (
            Some(&m.albedo),
            None,
            m.normal.as_ref(),
            m.roughness as f32,
            m.index_of_refraction as f32,
            MAT_DIELECTRIC,
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
            MAT_DIFFUSE_LIGHT,
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
                MAT_BLEND,
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

    // Built once per distinct index of refraction, and only when something can
    // make the lobe rough. Interned before the material is, so two dielectrics
    // of the same index stay one material as well as one table.
    let energy_offset = if mat_type == MAT_DIELECTRIC && fuzz > 0. {
        let key = (ref_idx as f64).to_bits();
        match caches.energy_offsets.get(&key) {
            Some(&offset) => offset,
            None => {
                let offset = data.dielectric_energy.len() as u32;
                data.dielectric_energy.extend(build_table(ref_idx as f64));
                caches.energy_offsets.insert(key, offset);
                offset
            }
        }
    } else {
        0
    };

    let gpu_material = GpuMaterial {
        albedo: to_array(albedo),
        attenuation_factor,
        emission: to_array(emission),
        blend_factor,
        fuzz,
        refraction_index: ref_idx,
        mat_type,
        energy_offset,
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

/// Octahedral-encodes a unit vector into two snorm16, the layout WGSL
/// `unpack2x16snorm` reads back.
///
/// Reproduces `oct_encode` then `pack2x16snorm` in `renderer/ray_trace.wgsl`
/// bit for bit, including the `n.z <= 0.0` polarity of the fold. The shader
/// only decodes, so a disagreement would mirror normals on the hemisphere
/// straddling the branch with nothing on the GPU side to catch it.
///
/// `snorm` rather than the G-buffer's `pack2x16float`: the parameters live in
/// exactly `[-1, 1]`, where a half-float's exponent range is wasted. Worth
/// 0.0036 degrees at worst, pinned by `pack_oct_round_trip`.
pub(crate) fn pack_oct(n: Vec3) -> u32 {
    // f32 throughout: the precision the shader decodes in.
    let (x, y, z) = (n.x as f32, n.y as f32, n.z as f32);
    let l = x.abs() + y.abs() + z.abs();
    let (mut px, mut py) = (x / l, y / l);
    if z <= 0.0 {
        let sx = if px >= 0.0 { 1.0 } else { -1.0 };
        let sy = if py >= 0.0 { 1.0 } else { -1.0 };
        let (ax, ay) = (px.abs(), py.abs());
        px = (1.0 - ay) * sx;
        py = (1.0 - ax) * sy;
    }

    // WGSL pack2x16snorm: floor(0.5 + 32767 * clamp(e, -1, 1)), as a two's
    // complement 16-bit value.
    let q = |e: f32| ((0.5 + 32767.0 * e.clamp(-1.0, 1.0)).floor() as i32 as i16) as u16 as u32;
    q(px) | (q(py) << 16)
}

#[cfg(test)]
mod tests {
    use crate::geo::vec3::Vec3;
    use crate::hittable::LEAF_FLAG;
    use crate::hittable::{Bvh, Hittables, Sphere};
    use crate::material::texture::SolidColor;
    use crate::material::{Dielectric, Lambertian, Materials};
    use crate::renderer::dielectric_energy::ENERGY_TABLE_LEN;
    use crate::renderer::gpu_data::LightRef;
    use crate::renderer::scene_flattener::{SceneData, build_alias_table, flatten_scene, pack_oct};
    use crate::renderer::{RenderConfig, Scene};

    /// WGSL `unpack2x16snorm` then `oct_decode`, in Rust. Only the encoder
    /// ships; this exists to measure the round-trip.
    ///
    /// That leaves a gap: it pins `pack_oct` against a *transcription* of the
    /// shader's decoder, not the shader's own. The GPU half is covered only by
    /// the `smooth_vs_flat` golden.
    fn unpack_oct(packed: u32) -> Vec3 {
        let d = |bits: u16| (bits as i16 as f32 / 32767.0).max(-1.0);
        let (ex, ey) = (d(packed as u16), d((packed >> 16) as u16));

        let (mut x, mut y) = (ex, ey);
        let z = 1.0 - ex.abs() - ey.abs();
        if z < 0.0 {
            let sx = if x >= 0.0 { 1.0 } else { -1.0 };
            let sy = if y >= 0.0 { 1.0 } else { -1.0 };
            let (ax, ay) = (x.abs(), y.abs());
            x = (1.0 - ay) * sx;
            y = (1.0 - ax) * sy;
        }
        Vec3::new(x as f64, y as f64, z as f64).unit()
    }

    /// A scene of glass spheres, one per (index, roughness) pair, over nothing
    /// else. Only the material table is of interest here.
    fn flatten_glass(roughness: f64) -> SceneData {
        flatten_glass_spheres(&[(1.5, roughness)])
    }

    fn flatten_glass_pair(n1: f64, r1: f64, n2: f64, r2: f64) -> SceneData {
        flatten_glass_spheres(&[(n1, r1), (n2, r2)])
    }

    fn flatten_glass_spheres(glass: &[(f64, f64)]) -> SceneData {
        let world: Vec<Hittables> = glass
            .iter()
            .enumerate()
            .map(|(i, &(ior, roughness))| {
                Sphere::new(
                    Vec3::new(i as f64 * 3., 0., 0.),
                    1.,
                    Dielectric::new(SolidColor::new(1., 1., 1.).into(), None, ior, roughness)
                        .into(),
                )
                .into()
            })
            .collect();

        flatten_scene(&Scene {
            world: Bvh::new(world).into(),
            camera: crate::camera::CameraConfig {
                vertical_fov_degrees: 40.,
                aperture_size: 0.,
                look_from: Vec3::new(0., 0., 6.),
                look_at: Vec3::new(0., 0., 0.),
                up: Vec3::new(0., 1., 0.),
            },
            background_color: Vec3::new(1., 1., 1.),
            render_config: RenderConfig::default(),
        })
    }

    /// Who gets an energy table and who does not.
    ///
    /// Building one is the most expensive thing the flattener does per
    /// material -- half a million evaluations of the microfacet model -- and it
    /// is wasted on a scene whose glass is smooth, which is every scene that
    /// predates the microfacet dielectric. The shader agrees:
    /// `has_rough_dielectrics` is false under the same condition and compiles
    /// out the arm that would read it.
    #[test]
    fn a_table_is_built_only_for_a_rough_dielectric() {
        assert!(
            flatten_glass(0.).dielectric_energy.is_empty(),
            "smooth glass built a table it can never read"
        );
        assert_eq!(
            ENERGY_TABLE_LEN,
            flatten_glass(0.3).dielectric_energy.len(),
            "a rough dielectric needs both halves of one table"
        );
    }

    /// Two dielectrics of the same index share one table, whatever their
    /// roughness: roughness is an *axis* of the table rather than a parameter
    /// of it, which is what keeps a five-sphere roughness sweep to one build.
    #[test]
    fn one_table_per_index_of_refraction() {
        assert_eq!(
            ENERGY_TABLE_LEN,
            flatten_glass_pair(1.5, 0.2, 1.5, 0.8)
                .dielectric_energy
                .len()
        );
        assert_eq!(
            2 * ENERGY_TABLE_LEN,
            flatten_glass_pair(1.5, 0.2, 1.8, 0.2)
                .dielectric_energy
                .len(),
            "two indices of refraction are two different tables"
        );
    }

    #[test]
    fn pack_oct_round_trip() {
        // A deterministic spiral rather than a random sample: roughly uniform
        // in area, and it lands on both sides of the `n.z <= 0.0` fold and
        // close to it, where a polarity disagreement shows.
        let n = 128;
        let mut worst_degrees: f64 = 0.;
        for i in 0..n {
            let z = 1. - 2. * (i as f64 + 0.5) / n as f64;
            let r = (1. - z * z).max(0.).sqrt();
            // The golden angle, so successive points do not line up.
            let phi = i as f64 * std::f64::consts::PI * (3. - 5f64.sqrt());
            let dir = Vec3::new(r * phi.cos(), r * phi.sin(), z).unit();

            let round_tripped = unpack_oct(pack_oct(dir));
            let degrees = dir.dot(round_tripped).clamp(-1., 1.).acos().to_degrees();
            worst_degrees = worst_degrees.max(degrees);
        }

        assert!(
            worst_degrees < 0.01,
            "worst octahedral round-trip error was {} degrees",
            worst_degrees
        );
    }

    #[test]
    fn pack_oct_handles_the_poles() {
        // +Z is the one direction the fold does not touch, -Z the one it maps
        // to all four corners at once.
        for dir in [Vec3::new(0., 0., 1.), Vec3::new(0., 0., -1.)] {
            let round_tripped = unpack_oct(pack_oct(dir));
            assert!(
                dir.dot(round_tripped) > 0.9999999,
                "{:?} round-tripped to {:?}",
                dir,
                round_tripped
            );
        }
    }

    /// Draws a light the way `sample_light` does: one uniform scalar, split
    /// into a slot and the coin that decides between that slot and its alias.
    ///
    /// A transcription of the shader, which is the point -- the GPU side is
    /// three lines and untestable without a device, so the thing worth pinning
    /// is that the table those three lines read produces the distribution the
    /// MIS weight is told it produces.
    fn alias_draw(lights: &[LightRef], x: f64) -> usize {
        let scaled = x * lights.len() as f64;
        let slot = (scaled as usize).min(lights.len() - 1);
        if scaled - slot as f64 >= lights[slot].alias_prob as f64 {
            lights[slot].alias_index as usize
        } else {
            slot
        }
    }

    fn lights_with_powers(powers: &[f32]) -> Vec<LightRef> {
        powers
            .iter()
            .enumerate()
            .map(|(i, &power)| LightRef {
                prim: i as u32,
                select_pdf: power,
                alias_prob: 0.,
                alias_index: 0,
            })
            .collect()
    }

    /// The alias table against the `select_pdf` the tracer weights by.
    ///
    /// These two are the whole bias risk in power-proportional selection: the
    /// sampler draws from the table, while the PDF it is divided by is
    /// `select_pdf` times the density of the point on the light. A Vose bug
    /// that leaves the two disagreeing is an image that is quietly wrong rather
    /// than one that crashes, and nothing on the GPU can see it -- a wrong
    /// distribution and a wrong weight make a plausible picture together.
    #[test]
    fn alias_table_matches_its_pdf() {
        // Three orders of magnitude, the range an emissive mesh spans, with the
        // bright light first so most slots are under-full and the table's
        // donation loop actually runs.
        let powers = [1000., 0.5, 12., 3., 200., 1., 0.25, 40.];
        let mut lights = lights_with_powers(&powers);
        build_alias_table(&mut lights);

        let total: f64 = powers.iter().map(|&p| p as f64).sum();
        let sum: f64 = lights.iter().map(|l| l.select_pdf as f64).sum();
        assert!(
            (sum - 1.).abs() < 1e-6,
            "select_pdf sums to {} rather than 1",
            sum
        );
        for (i, light) in lights.iter().enumerate() {
            let expected = powers[i] as f64 / total;
            assert!(
                (light.select_pdf as f64 - expected).abs() < 1e-6,
                "light {} has select_pdf {} against a power share of {}",
                i,
                light.select_pdf,
                expected
            );
        }

        const DRAWS: usize = 1_000_000;
        let mut counts = vec![0usize; lights.len()];
        let mut state: u32 = 0x9E37_79B9;
        for _ in 0..DRAWS {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let x = (state >> 8) as f64 / 16777216.;
            counts[alias_draw(&lights, x)] += 1;
        }

        for (i, &count) in counts.iter().enumerate() {
            let p = lights[i].select_pdf as f64;
            let mean = DRAWS as f64 * p;
            let sigma = (DRAWS as f64 * p * (1. - p)).sqrt();
            assert!(
                (count as f64 - mean).abs() <= 3. * sigma,
                "light {} was drawn {} times against an expected {} +/- {} (3 sigma)",
                i,
                count,
                mean,
                3. * sigma
            );
        }
    }

    /// A `DiffuseLight` may be black, and `is_light` still calls it a light.
    /// Power-proportional selection has nothing to divide by then, so the table
    /// has to come out uniform rather than NaN.
    #[test]
    fn alias_table_falls_back_to_uniform_when_nothing_emits() {
        let mut lights = lights_with_powers(&[0., 0., 0., 0.]);
        build_alias_table(&mut lights);

        let mut counts = vec![0usize; lights.len()];
        for (i, light) in lights.iter().enumerate() {
            assert_eq!(light.select_pdf, 0.25, "light {} is not uniform", i);
            counts[alias_draw(&lights, (i as f64 + 0.5) / lights.len() as f64)] += 1;
        }
        assert_eq!(counts, vec![1; lights.len()], "a slot is unreachable");
    }

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
