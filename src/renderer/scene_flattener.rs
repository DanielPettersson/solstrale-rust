//! Utilities for flattening the scene graph into linear buffers for the GPU
//!
//! The buffers are laid out as an arena, so a renderer can update them in
//! place rather than build them again. The world's own tree -- its loose
//! primitives and the nodes over them -- sits at node 0 with room to grow, and
//! every subtree a [`Bvh`] keeps apart is a segment of its own at a stable
//! offset. Indices stay absolute, which is what keeps the shader unchanged: a
//! subtree is reached through an ordinary node index.
//!
//! Triangles share an allocator with `prim_refs`, so a triangle's slot is its
//! reference's slot. That is what keeps `prim_refs` the identity map in a scene
//! of nothing but triangles however its segments come and go. The price is a
//! hole in the triangle arrays wherever a sphere or a quad sits.

use crate::geo::Aabb;
use crate::geo::Uv;
use crate::geo::vec3::{Vec3, ZERO_VECTOR};
use crate::hittable::Geometry;
use crate::hittable::{Bvh, BvhData, Child, Hittable, Hittables, LEAF_FLAG, LeafPrims, child_of};
use crate::material::texture::{Texture, Textures};
use crate::material::{Material, Materials};
use crate::renderer::Scene;
use crate::renderer::dielectric_energy::{ENERGY_TABLE_LEN, build_table};
use crate::renderer::gpu_data::{
    BvhNode as GpuBvhNode, LightRef, MAT_BLEND, MAT_DIELECTRIC, MAT_DIFFUSE_LIGHT, MAT_LAMBERTIAN,
    MAT_METAL, Material as GpuMaterial, PRIM_TYPE_QUAD, PRIM_TYPE_SHIFT, PRIM_TYPE_SPHERE,
    PRIM_TYPE_TRIANGLE, QuadAttr, QuadPos, Sphere as GpuSphere, TriangleAttr, TrianglePos,
};
use crate::util::luminance::luminance;
use crate::util::rgb_color::srgb_to_vec3;
use crate::util::texture_processing::{AtlasLayout, TexturePacker};
use bytemuck::Zeroable;
use image::RgbImage;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::ops::Range;
use std::sync::{Arc, Weak};

type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// The multiply-xor hash rustc uses internally, over 64-bit words.
///
/// These maps are probed once per material -- a mesh interns its one material
/// once, but a scene of many small objects probes once per primitive -- on a
/// 96-byte key. SipHash's per-byte work is the wrong trade for a table whose
/// keys nothing external can choose.
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

/// What the tracer can be specialised on, as far as the world decides it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Features(u16);

impl Features {
    pub(crate) const SPHERES: u16 = 1;
    pub(crate) const QUADS: u16 = 1 << 1;
    pub(crate) const BLENDS: u16 = 1 << 2;
    pub(crate) const METAL: u16 = 1 << 3;
    pub(crate) const DIELECTRICS: u16 = 1 << 4;
    pub(crate) const ROUGH_DIELECTRICS: u16 = 1 << 5;
    pub(crate) const TEXTURES: u16 = 1 << 6;
    pub(crate) const NORMAL_MAPS: u16 = 1 << 7;

    pub(crate) fn has(self, flag: u16) -> bool {
        self.0 & flag != 0
    }
}

impl std::ops::BitOr for Features {
    type Output = Features;
    fn bitor(self, rhs: Features) -> Features {
        Features(self.0 | rhs.0)
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
    /// Triangle geometry, read during BVH traversal. Indexed like
    /// `prim_refs`, so empty in a scene without triangles and with holes
    /// where a sphere or a quad sits in one with them.
    pub triangle_pos: Vec<TrianglePos>,
    /// Triangle shading attributes, read once per ray after traversal.
    /// Indexed like `triangle_pos`.
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
    /// Whether `prim_refs[k]` is `(PRIM_TYPE_TRIANGLE << 30) | k` at every `k`
    /// a leaf reaches.
    ///
    /// True exactly when the scene is nothing but triangles, since a triangle
    /// takes the slot of its own reference. The tracer is compiled against
    /// this, and the innermost traversal loop then reads the triangle's address
    /// straight out of the leaf slot.
    pub prim_refs_are_identity: bool,
    /// What the tracer is specialised on; read by `specialisation_test`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) features: Features,
}

/// Flattens the scene into linear buffers
pub fn flatten_scene(scene: &Scene) -> SceneData {
    WorldLayout::new(false)
        .update(&scene.world)
        .into_scene_data()
}

/// Everything one [`WorldLayout::update`] asks of the GPU buffers: writes at
/// element offsets, and the length each array must have.
#[derive(Default)]
pub(crate) struct WorldWrites {
    /// The whole arena was laid out again, so nothing already in the buffers
    /// is worth keeping.
    pub(crate) full: bool,
    pub(crate) nodes: Vec<(u32, Vec<GpuBvhNode>)>,
    pub(crate) prim_refs: Vec<(u32, Vec<u32>)>,
    pub(crate) triangle_pos: Vec<(u32, Vec<TrianglePos>)>,
    pub(crate) triangle_attr: Vec<(u32, Vec<TriangleAttr>)>,
    pub(crate) spheres: Vec<(u32, Vec<GpuSphere>)>,
    pub(crate) quad_pos: Vec<(u32, Vec<QuadPos>)>,
    pub(crate) quad_attr: Vec<(u32, Vec<QuadAttr>)>,
    /// The whole table, every time: it holds hundreds of records, not
    /// hundreds of thousands.
    pub(crate) materials: Vec<GpuMaterial>,
    pub(crate) dielectric_energy: Vec<f32>,
    pub(crate) lights: Vec<LightRef>,
    pub(crate) atlas: AtlasChange,
    pub(crate) extents: Extents,
    pub(crate) features: Features,
}

/// The length each arena array has to be, in elements.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Extents {
    pub(crate) nodes: u32,
    pub(crate) prims: u32,
    /// Both triangle arrays, which share the `prims` allocator: the end of the
    /// last region holding a triangle, so a scene of spheres pays nothing.
    pub(crate) triangles: u32,
    pub(crate) spheres: u32,
    pub(crate) quads: u32,
}

/// What became of the texture atlas.
#[derive(Default)]
pub(crate) enum AtlasChange {
    /// The one already uploaded still holds every image the world samples.
    #[default]
    Keep,
    /// Nothing samples an image any more.
    Empty,
    /// Packed again, for these images at the size they were placed at.
    Packed(AtlasLayout, Vec<Arc<RgbImage>>),
}

impl WorldWrites {
    fn into_scene_data(self) -> SceneData {
        let tri_len = self.extents.triangles;
        let (textures, atlas_layout) = match self.atlas {
            AtlasChange::Packed(layout, textures) => (textures, Some(layout)),
            _ => (Vec::new(), None),
        };
        SceneData {
            nodes: assemble(self.nodes, self.extents.nodes),
            prim_refs: assemble(self.prim_refs, self.extents.prims),
            spheres: assemble(self.spheres, self.extents.spheres),
            triangle_pos: assemble(self.triangle_pos, tri_len),
            triangle_attr: assemble(self.triangle_attr, tri_len),
            quad_pos: assemble(self.quad_pos, self.extents.quads),
            quad_attr: assemble(self.quad_attr, self.extents.quads),
            materials: self.materials,
            textures,
            atlas_layout,
            lights: self.lights,
            dielectric_energy: self.dielectric_energy,
            prim_refs_are_identity: !self.features.has(Features::SPHERES)
                && !self.features.has(Features::QUADS),
            features: self.features,
        }
    }
}

fn assemble<T: Zeroable + Copy>(chunks: Vec<(u32, Vec<T>)>, len: u32) -> Vec<T> {
    let mut out = vec![T::zeroed(); len as usize];
    for (offset, chunk) in chunks {
        out[offset as usize..offset as usize + chunk.len()].copy_from_slice(&chunk);
    }
    out
}

/// A first-fit allocator over one arena array.
#[derive(Default)]
struct RangeAlloc {
    end: u32,
    /// Sorted, disjoint and never adjacent.
    free: Vec<Range<u32>>,
}

impl RangeAlloc {
    fn alloc(&mut self, n: u32) -> Range<u32> {
        if n == 0 {
            return 0..0;
        }
        if let Some(i) = self.free.iter().position(|r| r.len() as u32 >= n) {
            let start = self.free[i].start;
            self.free[i].start += n;
            if self.free[i].is_empty() {
                self.free.remove(i);
            }
            return start..start + n;
        }
        let start = self.end;
        self.end += n;
        start..start + n
    }

    fn free(&mut self, r: Range<u32>) {
        if r.is_empty() {
            return;
        }
        let i = self.free.partition_point(|f| f.start < r.start);
        self.free.insert(i, r);
        if i + 1 < self.free.len() && self.free[i].end == self.free[i + 1].start {
            self.free[i].end = self.free[i + 1].end;
            self.free.remove(i + 1);
        }
        if i > 0 && self.free[i - 1].end == self.free[i].start {
            self.free[i - 1].end = self.free[i].end;
            self.free.remove(i);
        }
        if let Some(last) = self.free.last()
            && last.end == self.end
        {
            self.end = last.start;
            self.free.pop();
        }
    }
}

/// Where a region of the arena sits.
#[derive(Clone, Debug, Default)]
struct Ranges {
    nodes: Range<u32>,
    prims: Range<u32>,
    spheres: Range<u32>,
    quads: Range<u32>,
}

/// How much of the arena a tree needs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Shape {
    nodes: u32,
    prims: u32,
    spheres: u32,
    quads: u32,
}

impl Shape {
    fn of(nodes: &[crate::hittable::BvhNode], prims: &[Hittables]) -> Shape {
        let (mut spheres, mut quads) = (0, 0);
        for p in prims {
            match p {
                Hittables::Sphere(_) => spheres += 1,
                Hittables::Quad(_) => quads += 1,
                _ => {}
            }
        }
        Shape {
            nodes: nodes.len() as u32,
            prims: prims.len() as u32,
            spheres,
            quads,
        }
    }

    fn fits(&self, ranges: &Ranges) -> bool {
        self.nodes <= ranges.nodes.len() as u32
            && self.prims <= ranges.prims.len() as u32
            && self.spheres <= ranges.spheres.len() as u32
            && self.quads <= ranges.quads.len() as u32
    }
}

/// One tree's part of the arena: a kept-apart subtree, or the world's own.
struct Region {
    /// The tree this was laid out from, held weakly so a batch render can
    /// still drop the CPU scene once it is uploaded.
    id: Weak<BvhData>,
    /// That tree's topology: a tree of the same one is written over this
    /// region in place, with the material indices it had.
    topology: Weak<BvhData>,
    ranges: Ranges,
    shape: Shape,
    /// Per primitive, in leaf order, so a re-bake of the same source can skip
    /// interning: its materials are the same ones.
    material_indices: Vec<u32>,
    /// The distinct materials above, each holding one reference.
    materials: Vec<u32>,
    /// With the raw emitted power in `select_pdf`, and absolute keys.
    emitters: Vec<LightRef>,
    features: Features,
    has_triangles: bool,
    /// Nested subtrees, as slots in [`WorldLayout::segments`].
    children: Vec<usize>,
}

/// How a tree of the new world gets its segment.
enum Plan {
    /// Already uploaded, and still is.
    Exact(usize),
    /// Baked from the same source as an uploaded segment, so it is written over
    /// that one in place.
    Rebake(usize, Bvh, Vec<Plan>),
    New(Bvh, Vec<Plan>),
}

impl Plan {
    fn slot(&self) -> Option<usize> {
        match self {
            Plan::Exact(s) | Plan::Rebake(s, ..) => Some(*s),
            Plan::New(..) => None,
        }
    }
}

/// The materials the arena's segments point into, reference-counted so a
/// record no segment uses any more is reused rather than appended after.
///
/// A record's index is baked into every primitive that uses it, so a record
/// can never move while one does -- which is why this is a free list and not
/// something compacted.
#[derive(Default)]
struct MaterialTable {
    /// Without their atlas placement, which is filled in on upload so a
    /// repacked atlas moves no index.
    records: Vec<GpuMaterial>,
    bits: Vec<u16>,
    refs: Vec<u32>,
    free: Vec<u32>,
    ids: FxMap<Vec<u8>, u32>,
    energy: EnergyTables,
}

impl MaterialTable {
    fn intern(&mut self, record: GpuMaterial, bits: u16) -> u32 {
        let key = bytemuck::bytes_of(&record);
        if let Some(&existing) = self.ids.get(key) {
            return existing;
        }
        let index = match self.free.pop() {
            Some(i) => {
                self.records[i as usize] = record;
                self.bits[i as usize] = bits;
                i
            }
            None => {
                self.records.push(record);
                self.bits.push(bits);
                self.refs.push(0);
                self.records.len() as u32 - 1
            }
        };
        self.ids.insert(key.to_vec(), index);
        index
    }

    fn acquire(&mut self, m: u32) {
        let i = m as usize;
        self.refs[i] += 1;
        if self.refs[i] == 1 {
            let record = self.records[i];
            if record.mat_type == MAT_BLEND {
                self.acquire(record.blend_indices[0]);
                self.acquire(record.blend_indices[1]);
            }
            if let Some(slot) = energy_slot(&record) {
                self.energy.refs[slot] += 1;
            }
        }
    }

    fn release(&mut self, m: u32) {
        let i = m as usize;
        self.refs[i] -= 1;
        if self.refs[i] == 0 {
            let record = self.records[i];
            if record.mat_type == MAT_BLEND {
                self.release(record.blend_indices[0]);
                self.release(record.blend_indices[1]);
            }
            if let Some(slot) = energy_slot(&record) {
                self.energy.release(slot);
            }
            self.ids.remove(bytemuck::bytes_of(&record));
            self.free.push(m);
        }
    }
}

fn energy_slot(record: &GpuMaterial) -> Option<usize> {
    (record.mat_type == MAT_DIELECTRIC && record.fuzz > 0.)
        .then_some(record.energy_offset as usize / ENERGY_TABLE_LEN)
}

/// Single-scatter albedo tables, one per index of refraction some rough
/// dielectric uses, in fixed-size slots.
#[derive(Default)]
struct EnergyTables {
    data: Vec<f32>,
    refs: Vec<u32>,
    free: Vec<usize>,
    /// An index of refraction's bit pattern -> its slot.
    ///
    /// Keyed on the index alone because that is all the table depends on:
    /// roughness and angle are its two axes, so five glass spheres across the
    /// roughness range at 1.5 share one table and build it once.
    slots: FxMap<u64, usize>,
    slot_keys: Vec<u64>,
}

impl EnergyTables {
    /// Where the table for `ior` starts, built if no slot holds it.
    fn offset(&mut self, ior: f32) -> u32 {
        let key = (ior as f64).to_bits();
        let slot = match self.slots.get(&key) {
            Some(&slot) => slot,
            None => {
                let table = build_table(ior as f64);
                let slot = match self.free.pop() {
                    Some(slot) => {
                        self.data[slot * ENERGY_TABLE_LEN..(slot + 1) * ENERGY_TABLE_LEN]
                            .copy_from_slice(&table);
                        self.slot_keys[slot] = key;
                        slot
                    }
                    None => {
                        self.data.extend(table);
                        self.refs.push(0);
                        self.slot_keys.push(key);
                        self.refs.len() - 1
                    }
                };
                self.slots.insert(key, slot);
                slot
            }
        };
        (slot * ENERGY_TABLE_LEN) as u32
    }

    fn release(&mut self, slot: usize) {
        self.refs[slot] -= 1;
        if self.refs[slot] == 0 {
            self.slots.remove(&self.slot_keys[slot]);
            self.free.push(slot);
        }
    }
}

type Placement = ([f32; 2], [f32; 2]);

/// Every image the arena's materials sample, under an id that outlives any
/// one packing of the atlas.
///
/// A material names its image by that id, and its atlas placement is filled in
/// on upload, so repacking the atlas rewrites the material table and nothing
/// else. The `Weak`s pin each image's allocation, so a freed image's address
/// cannot come back as another image's key.
#[derive(Default)]
struct TextureTable {
    ids: FxMap<usize, (Weak<RgbImage>, u32)>,
    /// By id; `None` for an image the current atlas does not hold.
    placements: Vec<Option<Placement>>,
    packed: usize,
    downscaled: bool,
    /// The mean of an emissive texture's texels. An emissive mesh shares one
    /// texture across hundreds of triangles, and scanning it per triangle, or
    /// per update, is not affordable.
    means: FxMap<usize, (Weak<RgbImage>, Vec3)>,
}

impl TextureTable {
    fn id(&mut self, image: &Arc<RgbImage>) -> u32 {
        let next = self.placements.len() as u32;
        let (_, id) = *self
            .ids
            .entry(Arc::as_ptr(image) as usize)
            .or_insert_with(|| (Arc::downgrade(image), next));
        if id == next {
            self.placements.push(None);
        }
        id
    }

    fn placement(&self, id: i32) -> Option<Placement> {
        usize::try_from(id)
            .ok()
            .and_then(|id| self.placements.get(id).copied().flatten())
    }

    /// Whether the uploaded atlas can stand for one packed from `textures`.
    ///
    /// Only if it holds every one of them, and, when it had to shrink them to
    /// fit, only if it holds nothing else: a packing of fewer might not have
    /// had to.
    fn covers(&self, textures: &[Arc<RgbImage>]) -> bool {
        let all_placed = textures.iter().all(|t| {
            self.ids
                .get(&(Arc::as_ptr(t) as usize))
                .is_some_and(|(_, id)| self.placements[*id as usize].is_some())
        });
        all_placed && (!self.downscaled || textures.len() == self.packed)
    }

    fn repack(&mut self, textures: &[Arc<RgbImage>]) -> AtlasChange {
        self.placements.iter_mut().for_each(|p| *p = None);
        self.packed = textures.len();
        self.downscaled = false;
        if textures.is_empty() {
            return AtlasChange::Empty;
        }
        let (layout, placed) = TexturePacker::new(ATLAS_SIZE, ATLAS_SIZE)
            .pack_images(textures)
            .expect("Failed to pack textures into atlas");
        for (i, texture) in textures.iter().enumerate() {
            let rect = &layout.placements[i];
            let id = self.id(texture) as usize;
            self.placements[id] = Some((
                [
                    rect.x as f32 / layout.width as f32,
                    rect.y as f32 / layout.height as f32,
                ],
                [
                    rect.width as f32 / layout.width as f32,
                    rect.height as f32 / layout.height as f32,
                ],
            ));
        }
        self.downscaled = textures
            .iter()
            .zip(&placed)
            .any(|(a, b)| !Arc::ptr_eq(a, b));
        AtlasChange::Packed(layout, placed)
    }

    /// Drops what the cache holds for images nothing holds any more.
    fn prune(&mut self) {
        self.means.retain(|_, (w, _)| w.strong_count() > 0);
    }
}

/// The arena behind a renderer's world buffers: where every tree sits, and the
/// tables its records point into.
pub(crate) struct WorldLayout {
    /// Whether the world's own region gets room to grow into. Off for a scene
    /// flattened once and never updated.
    spare: bool,
    materials: MaterialTable,
    textures: TextureTable,
    nodes: RangeAlloc,
    prims: RangeAlloc,
    spheres: RangeAlloc,
    quads: RangeAlloc,
    /// The world's own tree, at node 0.
    top: Option<Region>,
    segments: Vec<Option<Region>>,
}

impl WorldLayout {
    pub(crate) fn new(spare: bool) -> Self {
        WorldLayout {
            spare,
            materials: MaterialTable::default(),
            textures: TextureTable::default(),
            nodes: RangeAlloc::default(),
            prims: RangeAlloc::default(),
            spheres: RangeAlloc::default(),
            quads: RangeAlloc::default(),
            top: None,
            segments: Vec::new(),
        }
    }

    /// Lays `world` out, reusing whatever of the last world it still holds,
    /// and says what the buffers need written.
    ///
    /// O(loose primitives + subtrees + new subtrees): a subtree already
    /// uploaded costs nothing, and one baked from the same source as an
    /// uploaded one is written over it in place with the materials it had.
    pub(crate) fn update(&mut self, world: &Hittables) -> WorldWrites {
        let mut writes = WorldWrites::default();

        let world_textures = match world {
            Hittables::Bvh(b) => b.textures().to_vec(),
            other => {
                let mut v = Vec::new();
                if let Some(mat) = other.material() {
                    mat.for_each_atlas_texture(&mut |t: &Arc<RgbImage>| {
                        if !v.iter().any(|e| Arc::ptr_eq(e, t)) {
                            v.push(t.clone());
                        }
                    });
                }
                v
            }
        };
        if !self.textures.covers(&world_textures) {
            writes.atlas = self.textures.repack(&world_textures);
        }

        if !self.try_update(world, &mut writes) {
            self.relayout(world, &mut writes);
        }

        writes.materials = self
            .materials
            .records
            .iter()
            .map(|r| self.with_placement(*r))
            .collect();
        writes.dielectric_energy = self.materials.energy.data.clone();

        let top = self.top.as_ref().unwrap();
        let live = std::iter::once(top).chain(self.segments.iter().flatten());
        let mut features = Features::default();
        let mut lights = Vec::new();
        let mut triangles = 0;
        for region in live {
            features = features | region.features;
            lights.extend_from_slice(&region.emitters);
            if region.has_triangles {
                triangles = triangles.max(region.ranges.prims.end);
            }
        }
        // Emitters are appended as their primitives are, which interleaves the
        // types. Sorting them by the same packed key `prim_refs` uses is what
        // lets the tracer binary-search a hit primitive back to "is this an
        // emitter?" instead of asking every light.
        lights.sort_unstable_by_key(|light| light.prim);
        build_alias_table(&mut lights);
        writes.lights = lights;
        writes.features = features;
        writes.extents = Extents {
            nodes: self.nodes.end,
            prims: self.prims.end,
            triangles,
            spheres: self.spheres.end,
            quads: self.quads.end,
        };
        writes
    }

    /// The incremental update. False, having written nothing, when the world's
    /// own tree has outgrown its region or the arena has outgrown the leaf
    /// encoding, which only laying everything out again can fix.
    fn try_update(&mut self, world: &Hittables, writes: &mut WorldWrites) -> bool {
        let Some(top) = self.top.as_ref() else {
            return false;
        };
        let (top_nodes, top_prims, top_subtrees) = top_parts(world);
        let top_shape = Shape::of(&top_nodes, top_prims.leaf().source());
        if !top_shape.fits(&top.ranges) {
            return false;
        }
        let top_unchanged = matches!(world, Hittables::Bvh(b) if b.is(&top.id));

        // Exact matches first, so a baked copy cannot take the segment the
        // tree it was baked from is still using.
        let mut claimed = vec![false; self.segments.len()];
        let mut exact = self.match_exact(top_subtrees, &mut claimed);
        let plans: Vec<Plan> = top_subtrees
            .iter()
            .map(|s| self.plan(s, &mut claimed, &mut exact))
            .collect();

        // The old world's segments that the new one does not use.
        let unclaimed: Vec<usize> = (0..self.segments.len())
            .filter(|&s| !claimed[s] && self.segments[s].is_some())
            .collect();
        let freed: Vec<Region> = unclaimed
            .into_iter()
            .map(|s| self.segments[s].take().unwrap())
            .collect();
        for region in &freed {
            self.free_ranges(&region.ranges);
        }

        let mut ranges = Vec::new();
        for plan in &plans {
            self.allocate(plan, &mut ranges);
        }
        if self.prims.end as usize > crate::hittable::MAX_PRIMITIVES {
            // Holes, most likely: laying out again packs them.
            return false;
        }

        let mut ranges = ranges.into_iter();
        let roots: Vec<u32> = plans
            .into_iter()
            .map(|plan| self.place(plan, &mut ranges, writes))
            .collect();

        if !top_unchanged {
            let top = self.top.take().unwrap();
            let (id, topology) = world_identity(world);
            let cached = (!topology.ptr_eq(&Weak::new()) && topology.ptr_eq(&top.topology))
                .then_some((&top.material_indices[..], &top.materials[..]));
            let region = self.emit(
                &top_nodes,
                top_prims.leaf(),
                &roots,
                top.ranges.clone(),
                cached,
                id,
                topology,
                Vec::new(),
                writes,
            );
            self.release(&top);
            self.top = Some(region);
        }
        for region in freed {
            self.release(&region);
        }
        true
    }

    /// The uploaded segment each of `trees`, or anything nested in them, can
    /// keep as it is, as the slots found for each tree's identity. Claims them
    /// and everything nested in them.
    fn match_exact(&self, trees: &[Bvh], claimed: &mut [bool]) -> FxMap<usize, Vec<usize>> {
        let mut by_id: FxMap<usize, Vec<usize>> = FxMap::default();
        for (slot, region) in self.segments.iter().enumerate() {
            if let Some(region) = region {
                by_id
                    .entry(region.id.as_ptr() as usize)
                    .or_default()
                    .push(slot);
            }
        }
        let mut found: FxMap<usize, Vec<usize>> = FxMap::default();
        let mut stack: Vec<&Bvh> = trees.iter().rev().collect();
        while let Some(bvh) = stack.pop() {
            let slot = by_id.get_mut(&bvh.id_ptr()).and_then(|slots| {
                slots
                    .iter()
                    .position(|&s| !claimed[s])
                    .map(|i| slots.remove(i))
            });
            match slot {
                Some(slot) => {
                    self.claim(slot, claimed);
                    found.entry(bvh.id_ptr()).or_default().push(slot);
                }
                None => stack.extend(bvh.subtrees().iter().rev()),
            }
        }
        // Popped in the order `plan` meets the trees.
        found.values_mut().for_each(|slots| slots.reverse());
        found
    }

    /// How `bvh` gets its segment, claiming the segments it will reuse.
    /// `exact` is what [`Self::match_exact`] found, handed out one slot per
    /// occurrence of a tree.
    fn plan(&self, bvh: &Bvh, claimed: &mut [bool], exact: &mut FxMap<usize, Vec<usize>>) -> Plan {
        if let Some(slot) = exact.get_mut(&bvh.id_ptr()).and_then(Vec::pop) {
            return Plan::Exact(slot);
        }
        let children: Vec<Plan> = bvh
            .subtrees()
            .iter()
            .map(|s| self.plan(s, claimed, exact))
            .collect();
        let topology = bvh.topology();
        let mut shape = None;
        let rebake = self.segments.iter().enumerate().position(|(slot, r)| {
            !claimed[slot]
                && r.as_ref().is_some_and(|r| {
                    r.topology.ptr_eq(&topology)
                        && r.shape
                            == *shape.get_or_insert_with(|| {
                                Shape::of(bvh.nodes(), bvh.leaf_prims().source())
                            })
                })
        });
        match rebake {
            Some(slot) => {
                claimed[slot] = true;
                Plan::Rebake(slot, bvh.clone(), children)
            }
            None => Plan::New(bvh.clone(), children),
        }
    }

    /// Claims an exact match and every segment nested in it, which are exact
    /// too: a tree cannot change what it holds.
    fn claim(&self, slot: usize, claimed: &mut [bool]) {
        claimed[slot] = true;
        for &child in &self.segments[slot].as_ref().unwrap().children {
            self.claim(child, claimed);
        }
    }

    /// Allocates, in the order [`Self::place`] consumes them, the ranges of
    /// every new segment under `plan`.
    fn allocate(&mut self, plan: &Plan, out: &mut Vec<Ranges>) {
        match plan {
            Plan::Exact(_) => {}
            Plan::Rebake(_, _, children) => {
                for c in children {
                    self.allocate(c, out);
                }
            }
            Plan::New(bvh, children) => {
                for c in children {
                    self.allocate(c, out);
                }
                let shape = Shape::of(bvh.nodes(), bvh.leaf_prims().source());
                out.push(self.alloc_ranges(shape, 0));
            }
        }
    }

    /// Allocates a tree's ranges. `spare` is how many primitives of room it
    /// may take on top of what it holds, which only the world's own region
    /// gets, and which the leaf encoding bounds.
    fn alloc_ranges(&mut self, shape: Shape, spare: u32) -> Ranges {
        let grow = |n: u32, cap: u32| {
            if spare > 0 {
                n + ((n / 4).min(16384) + 64).min(cap)
            } else {
                n
            }
        };
        Ranges {
            nodes: self.nodes.alloc(grow(shape.nodes, u32::MAX)),
            prims: self.prims.alloc(grow(shape.prims, spare)),
            spheres: self.spheres.alloc(grow(shape.spheres, u32::MAX)),
            quads: self.quads.alloc(grow(shape.quads, u32::MAX)),
        }
    }

    fn free_ranges(&mut self, r: &Ranges) {
        self.nodes.free(r.nodes.clone());
        self.prims.free(r.prims.clone());
        self.spheres.free(r.spheres.clone());
        self.quads.free(r.quads.clone());
    }

    /// Emits a planned tree into its segment and returns its root node.
    fn place(
        &mut self,
        plan: Plan,
        ranges: &mut impl Iterator<Item = Ranges>,
        writes: &mut WorldWrites,
    ) -> u32 {
        match plan {
            Plan::Exact(slot) => self.segments[slot].as_ref().unwrap().ranges.nodes.start,
            Plan::Rebake(slot, bvh, children) => {
                let child_slots: Vec<Option<usize>> = children.iter().map(Plan::slot).collect();
                let roots: Vec<u32> = children
                    .into_iter()
                    .map(|c| self.place(c, ranges, writes))
                    .collect();
                let old = self.segments[slot].take().unwrap();
                let region = self.emit(
                    bvh.nodes(),
                    bvh.leaf_prims(),
                    &roots,
                    old.ranges.clone(),
                    Some((&old.material_indices, &old.materials)),
                    bvh.downgrade(),
                    bvh.topology(),
                    self.child_slots(&child_slots, &roots),
                    writes,
                );
                self.release(&old);
                self.segments[slot] = Some(region);
                region_root(&self.segments[slot])
            }
            Plan::New(bvh, children) => {
                let child_slots: Vec<Option<usize>> = children.iter().map(Plan::slot).collect();
                let roots: Vec<u32> = children
                    .into_iter()
                    .map(|c| self.place(c, ranges, writes))
                    .collect();
                let r = ranges.next().unwrap();
                let region = self.emit(
                    bvh.nodes(),
                    bvh.leaf_prims(),
                    &roots,
                    r,
                    None,
                    bvh.downgrade(),
                    bvh.topology(),
                    self.child_slots(&child_slots, &roots),
                    writes,
                );
                let slot = match self.segments.iter().position(Option::is_none) {
                    Some(slot) => slot,
                    None => {
                        self.segments.push(None);
                        self.segments.len() - 1
                    }
                };
                self.segments[slot] = Some(region);
                region_root(&self.segments[slot])
            }
        }
    }

    /// The slots of a tree's children, found by the root each was placed at
    /// when its plan did not know the slot yet.
    fn child_slots(&self, planned: &[Option<usize>], roots: &[u32]) -> Vec<usize> {
        planned
            .iter()
            .zip(roots)
            .map(|(slot, &root)| {
                slot.unwrap_or_else(|| {
                    self.segments
                        .iter()
                        .position(|r| r.as_ref().is_some_and(|r| r.ranges.nodes.start == root))
                        .unwrap()
                })
            })
            .collect()
    }

    /// Lays everything out again from nothing, the world's own tree first so
    /// it sits at node 0. What a new renderer does, and what an update falls
    /// back to when the incremental path cannot place the world.
    fn relayout(&mut self, world: &Hittables, writes: &mut WorldWrites) {
        let atlas = std::mem::take(&mut writes.atlas);
        *writes = WorldWrites {
            full: true,
            atlas,
            ..Default::default()
        };

        let mut textures = std::mem::take(&mut self.textures);
        textures.prune();
        *self = WorldLayout {
            textures,
            ..WorldLayout::new(self.spare)
        };

        let (top_nodes, top_prims, top_subtrees) = top_parts(world);
        // Room for the world's own region to grow into, as long as the whole
        // arena still fits the leaf encoding.
        let prims = match world {
            Hittables::Bvh(b) => b.primitive_count(),
            _ => 1,
        };
        let room = if self.spare {
            crate::hittable::MAX_PRIMITIVES.saturating_sub(prims) as u32
        } else {
            0
        };
        let top_ranges = self.alloc_ranges(Shape::of(&top_nodes, top_prims.leaf().source()), room);

        let plans: Vec<Plan> = top_subtrees
            .iter()
            .map(|s| self.plan(s, &mut [], &mut FxMap::default()))
            .collect();
        let mut ranges = Vec::new();
        for plan in &plans {
            self.allocate(plan, &mut ranges);
        }
        let mut ranges = ranges.into_iter();
        let roots: Vec<u32> = plans
            .into_iter()
            .map(|plan| self.place(plan, &mut ranges, writes))
            .collect();

        let (id, topology) = world_identity(world);
        let top = self.emit(
            &top_nodes,
            top_prims.leaf(),
            &roots,
            top_ranges,
            None,
            id,
            topology,
            Vec::new(),
            writes,
        );
        self.top = Some(top);
    }

    fn release(&mut self, region: &Region) {
        for &m in &region.materials {
            self.materials.release(m);
        }
    }

    /// Writes one tree's nodes and primitives into `ranges`.
    ///
    /// `cached` is the material indices of an earlier tree baked from the same
    /// source, and the references it held, which this one takes over.
    #[allow(clippy::too_many_arguments)]
    fn emit(
        &mut self,
        nodes: &[crate::hittable::BvhNode],
        leaf_prims: LeafPrims,
        subtree_roots: &[u32],
        ranges: Ranges,
        cached: Option<(&[u32], &[u32])>,
        id: Weak<BvhData>,
        topology: Weak<BvhData>,
        children: Vec<usize>,
        writes: &mut WorldWrites,
    ) -> Region {
        // Materials, kinds and order are the source's, baked or not.
        let prims = leaf_prims.source();
        let baked = match leaf_prims {
            LeafPrims::Baked(_, t) => Some(t),
            LeafPrims::Plain(_) => None,
        };
        let (material_indices, materials) = match cached {
            Some((indices, distinct)) => (indices.to_vec(), distinct.to_vec()),
            None => self.intern_all(prims),
        };
        for &m in &materials {
            self.materials.acquire(m);
        }

        let prim_base = ranges.prims.start;
        let sphere_base = ranges.spheres.start;
        let quad_base = ranges.quads.start;

        let mut prim_refs = Vec::with_capacity(prims.len());
        let mut spheres = Vec::new();
        let mut quad_pos = Vec::new();
        let mut quad_attr = Vec::new();
        let mut emitters = Vec::new();
        let mut has_triangles = false;
        for (i, prim) in prims.iter().enumerate() {
            let mat_idx = material_indices[i];
            // A sphere or a quad is cheap to bake whole, and rare in a mesh.
            let owned;
            let prim = match (baked, prim) {
                (Some(t), Hittables::Sphere(_) | Hittables::Quad(_)) => {
                    owned = prim.transformed(t);
                    &owned
                }
                _ => prim,
            };
            let (prim_type, index) = match prim {
                Hittables::Sphere(s) => {
                    let index = sphere_base + spheres.len() as u32;
                    spheres.push(GpuSphere {
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
                        emitters.push(self.light_ref(PRIM_TYPE_SPHERE, index, &s.mat, area));
                    }
                    (PRIM_TYPE_SPHERE, index)
                }
                Hittables::Triangle(t) => {
                    has_triangles = true;
                    let index = prim_base + i as u32;
                    if t.mat.is_light() {
                        let area = match baked {
                            Some(b) => t.transformed_geometry(b).area,
                            None => t.area,
                        };
                        emitters.push(self.light_ref(PRIM_TYPE_TRIANGLE, index, &t.mat, area));
                    }
                    (PRIM_TYPE_TRIANGLE, index)
                }
                Hittables::Quad(q) => {
                    let index = quad_base + quad_pos.len() as u32;
                    quad_pos.push(QuadPos {
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
                    quad_attr.push(QuadAttr {
                        tangent: to_array(q.u.unit()),
                        area: q.area as f32,
                        bi_tangent: to_array(q.v.unit()),
                        material_index: mat_idx,
                    });
                    if q.mat.is_light() {
                        emitters.push(self.light_ref(PRIM_TYPE_QUAD, index, &q.mat, q.area));
                    }
                    (PRIM_TYPE_QUAD, index)
                }
                Hittables::Bvh(_) => unreachable!("a Bvh's primitives hold no Bvh"),
            };
            prim_refs.push((prim_type << PRIM_TYPE_SHIFT) | index);
        }

        if has_triangles {
            let (pos, attr): (Vec<TrianglePos>, Vec<TriangleAttr>) = prims
                .par_iter()
                .with_min_len(4096)
                .zip(material_indices.par_iter())
                .map(|(prim, &mat_idx)| match (prim, baked) {
                    (Hittables::Triangle(t), None) => triangle_records(t, mat_idx),
                    (Hittables::Triangle(t), Some(b)) => {
                        geometry_records(&t.transformed_geometry(b), t.uv(), mat_idx)
                    }
                    _ => (TrianglePos::zeroed(), TriangleAttr::zeroed()),
                })
                .unzip();
            writes.triangle_pos.push((prim_base, pos));
            writes.triangle_attr.push((prim_base, attr));
        }

        let node_base = ranges.nodes.start;
        let rebase = |meta: u32| match child_of(meta) {
            Child::Node(n) => n + node_base,
            Child::Leaf { count: 0, .. } => LEAF_FLAG,
            Child::Leaf { offset, count } => leaf_meta(offset + prim_base, count),
            Child::Subtree(i) => subtree_roots[i as usize],
        };
        let gpu_nodes: Vec<GpuBvhNode> = nodes
            .par_iter()
            .with_min_len(4096)
            .map(|node| GpuBvhNode {
                left_min: aabb_min(&node.left_box),
                left_meta: rebase(node.left_meta),
                left_max: aabb_max(&node.left_box),
                right_meta: rebase(node.right_meta),
                right_min: aabb_min(&node.right_box),
                _pad0: 0,
                right_max: aabb_max(&node.right_box),
                _pad1: 0,
            })
            .collect();

        let mut features = Features::default();
        for &m in &materials {
            features.0 |= self.materials.bits[m as usize];
        }
        if !spheres.is_empty() {
            features.0 |= Features::SPHERES;
        }
        if !quad_pos.is_empty() {
            features.0 |= Features::QUADS;
        }

        let shape = Shape {
            nodes: gpu_nodes.len() as u32,
            prims: prims.len() as u32,
            spheres: spheres.len() as u32,
            quads: quad_pos.len() as u32,
        };
        writes.nodes.push((node_base, gpu_nodes));
        writes.prim_refs.push((prim_base, prim_refs));
        if !spheres.is_empty() {
            writes.spheres.push((sphere_base, spheres));
        }
        if !quad_pos.is_empty() {
            writes.quad_pos.push((quad_base, quad_pos));
            writes.quad_attr.push((quad_base, quad_attr));
        }

        Region {
            id,
            topology,
            ranges,
            shape,
            material_indices,
            materials,
            emitters,
            features,
            has_triangles,
            children,
        }
    }

    /// Every primitive's material index, and the distinct ones among them.
    fn intern_all(&mut self, prims: &[Hittables]) -> (Vec<u32>, Vec<u32>) {
        let mut indices = Vec::with_capacity(prims.len());
        let mut distinct = Vec::new();
        let mut seen: HashSet<u32, BuildHasherDefault<FxHasher>> = HashSet::default();
        // A mesh shares one material across every triangle, so nearly every
        // primitive's material is the one before it.
        let mut last: Option<(&Materials, u32)> = None;
        for prim in prims {
            let mat = prim.material().expect("a Bvh's primitives hold no Bvh");
            let index = match last {
                Some((prev, index)) if prev.same_as(mat) => index,
                _ => {
                    let index = self.add_material(mat);
                    if seen.insert(index) {
                        distinct.push(index);
                    }
                    index
                }
            };
            last = Some((mat, index));
            indices.push(index);
        }
        (indices, distinct)
    }

    fn with_placement(&self, mut m: GpuMaterial) -> GpuMaterial {
        if let Some((offset, scale)) = self.textures.placement(m.texture_index) {
            m.albedo_offset = offset;
            m.albedo_scale = scale;
        }
        if let Some((offset, scale)) = self.textures.placement(m.normal_texture_index) {
            m.normal_offset = offset;
            m.normal_scale = scale;
        }
        m
    }

    /// A light's entry, with `select_pdf` holding its raw emitted power until
    /// [`build_alias_table`] normalises the set.
    ///
    /// Power for a one-sided diffuse emitter is `pi * A * luminance(L)`. A sphere
    /// emits over its whole surface, hence `A = 4 pi r^2` at the call site.
    fn light_ref(
        &mut self,
        prim_type: u32,
        prim_index: u32,
        material: &Materials,
        area: f64,
    ) -> LightRef {
        // Only a DiffuseLight answers true to `is_light`, which is what gates every
        // call here; a Blend holding one is not in `lights` at all.
        let radiance = match material {
            Materials::DiffuseLight(m) => mean_color(&m.tex, &mut self.textures.means),
            _ => ZERO_VECTOR,
        };

        LightRef {
            prim: (prim_type << PRIM_TYPE_SHIFT) | prim_index,
            select_pdf: (std::f64::consts::PI * area * luminance(radiance)) as f32,
            alias_prob: 0.,
            alias_index: 0,
        }
    }

    fn add_material(&mut self, material: &Materials) -> u32 {
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
                let idx1 = self.add_material(&b.material_1);
                let idx2 = self.add_material(&b.material_2);
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

        // The mean of the texture, not the texel at UV (0, 0). `emission` is all
        // the shader knows about a light's radiance -- neither `surface_at` nor
        // `sample_light` samples a texture for it -- so a textured emitter is a
        // flat light of its average colour, and `light_ref` ranks its power by the
        // same number.
        let emission = emission_tex
            .map(|t| mean_color(t, &mut self.textures.means))
            .unwrap_or(ZERO_VECTOR);

        // Albedo only, deliberately: falling back to the emission texture would put
        // an emitter's image in the albedo slot, which no emitter arm reads.
        let texture_index = albedo_tex.map_or(-1, |t| self.texture_id(t));
        let normal_texture_index = normal_tex.map_or(-1, |t| self.texture_id(t));

        // Built once per distinct index of refraction, and only when something can
        // make the lobe rough. Looked up before the material is interned, so two
        // dielectrics of the same index stay one material as well as one table.
        let energy_offset = if mat_type == MAT_DIELECTRIC && fuzz > 0. {
            self.materials.energy.offset(ref_idx)
        } else {
            0
        };

        // The atlas placement stays out of the record, and so out of the key:
        // it is filled in on upload, from `texture_index`.
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
            albedo_offset: [0.0; 2],
            albedo_scale: [1.0; 2],
            normal_offset: [0.0; 2],
            normal_scale: [1.0; 2],
        };

        let mut bits = 0;
        match mat_type {
            MAT_BLEND => {
                bits |= Features::BLENDS
                    | self.materials.bits[blend_indices[0] as usize]
                    | self.materials.bits[blend_indices[1] as usize]
            }
            MAT_METAL => bits |= Features::METAL,
            MAT_DIELECTRIC => {
                bits |= Features::DIELECTRICS;
                // Any roughness at all, not a threshold: the shader still routes
                // a roughness below `GGX_ALPHA_MIN` down the Dirac path, so this
                // only has to be conservative.
                if fuzz > 0. {
                    bits |= Features::ROUGH_DIELECTRICS;
                }
            }
            _ => {}
        }
        if texture_index >= 0 {
            bits |= Features::TEXTURES;
        }
        if normal_texture_index >= 0 {
            bits |= Features::NORMAL_MAPS;
        }

        // Intern on the exact byte image. `GpuMaterial` is `Pod` (no uninit padding),
        // so bytewise equality is exactly "identical to the GPU". Blend children are
        // resolved above, so their indices are already interned and stable by now.
        self.materials.intern(gpu_material, bits)
    }

    fn texture_id(&mut self, tex: &Textures) -> i32 {
        match tex {
            Textures::ImageMap(im) => self.textures.id(im.image()) as i32,
            _ => -1,
        }
    }
}

/// The world's own primitives: its tree's, or the one primitive it is.
enum TopPrims<'a> {
    Tree(LeafPrims<'a>),
    Bare(Vec<Hittables>),
}

impl TopPrims<'_> {
    fn leaf(&self) -> LeafPrims<'_> {
        match self {
            TopPrims::Tree(LeafPrims::Plain(p)) => LeafPrims::Plain(p),
            TopPrims::Tree(LeafPrims::Baked(p, t)) => LeafPrims::Baked(p, *t),
            TopPrims::Bare(p) => LeafPrims::Plain(p),
        }
    }
}

/// The world's own tree as nodes, primitives and subtrees, whatever the world
/// is: a bare primitive gets a root over a one-primitive leaf, and an empty
/// tree a root over two empty ones, so node 0 is always a node.
fn top_parts(
    world: &Hittables,
) -> (
    std::borrow::Cow<'_, [crate::hittable::BvhNode]>,
    TopPrims<'_>,
    &[Bvh],
) {
    use crate::hittable::BvhNode;
    use std::borrow::Cow;
    match world {
        Hittables::Bvh(b) if b.nodes().is_empty() => (
            Cow::Owned(vec![BvhNode {
                left_box: Aabb::default(),
                right_box: Aabb::default(),
                left_meta: LEAF_FLAG,
                right_meta: LEAF_FLAG,
            }]),
            TopPrims::Bare(Vec::new()),
            &[],
        ),
        Hittables::Bvh(b) => (
            Cow::Borrowed(b.nodes()),
            TopPrims::Tree(b.leaf_prims()),
            b.subtrees(),
        ),
        prim => (
            Cow::Owned(vec![BvhNode {
                left_box: prim.bounding_box().clone(),
                right_box: Aabb::default(),
                left_meta: leaf_meta(0, 1),
                right_meta: LEAF_FLAG,
            }]),
            TopPrims::Bare(vec![prim.clone()]),
            &[],
        ),
    }
}

fn world_identity(world: &Hittables) -> (Weak<BvhData>, Weak<BvhData>) {
    match world {
        Hittables::Bvh(b) => (b.downgrade(), b.topology()),
        _ => (Weak::new(), Weak::new()),
    }
}

fn region_root(region: &Option<Region>) -> u32 {
    region.as_ref().unwrap().ranges.nodes.start
}

fn triangle_records(t: &crate::hittable::Triangle, mat_idx: u32) -> (TrianglePos, TriangleAttr) {
    let g = Geometry {
        v0: t.v0,
        v0v1: t.v0v1,
        v0v2: t.v0v2,
        normal: t.normal,
        n0: t.n0,
        n1: t.n1,
        n2: t.n2,
        tangent: t.tangent,
        bi_tangent: t.bi_tangent,
        b_box: Aabb::default(),
        area: t.area,
    };
    geometry_records(&g, t.uv(), mat_idx)
}

fn geometry_records(
    g: &Geometry,
    [uv0, uv1, uv2]: [Uv; 3],
    mat_idx: u32,
) -> (TrianglePos, TriangleAttr) {
    // Edges go to the GPU as-is, which is what Moeller-Trumbore wants.
    (
        TrianglePos {
            v0: to_array(g.v0),
            _pad0: 0.0,
            e1: to_array(g.v0v1),
            _pad1: 0.0,
            e2: to_array(g.v0v2),
            _pad2: 0.0,
        },
        TriangleAttr {
            normal: to_array(g.normal),
            material_index: mat_idx,
            tangent: to_array(g.tangent),
            area: g.area as f32,
            bi_tangent: to_array(g.bi_tangent),
            n0_oct: pack_oct(g.n0),
            uv0: [uv0.u, uv0.v],
            uv1: [uv1.u, uv1.v],
            uv2: [uv2.u, uv2.v],
            n1_oct: pack_oct(g.n1),
            n2_oct: pack_oct(g.n2),
        },
    )
}

fn aabb_min(a: &Aabb) -> [f32; 3] {
    [a.x.min as f32, a.y.min as f32, a.z.min as f32]
}

fn aabb_max(a: &Aabb) -> [f32; 3] {
    [a.x.max as f32, a.y.max as f32, a.z.max as f32]
}

/// The average colour a texture emits.
///
/// The one radiance a textured emitter gets: `add_material` puts it in the
/// GPU's `emission` slot and `light_ref` ranks the light's power by it. Every
/// texel is averaged, sRGB-decoded per texel to match `ImageMap::color`, rather
/// than the texture being flattened to whichever texel sits at UV (0, 0).
fn mean_color(tex: &Textures, means: &mut FxMap<usize, (Weak<RgbImage>, Vec3)>) -> Vec3 {
    let Textures::ImageMap(im) = tex else {
        return sample_texture(tex);
    };

    let image = im.image();
    let key = Arc::as_ptr(image) as usize;
    if let Some((_, mean)) = means.get(&key) {
        return *mean;
    }

    let texels = (image.width() * image.height()).max(1) as f64;
    let mean = image
        .pixels()
        .fold(ZERO_VECTOR, |acc, p| acc + srgb_to_vec3(p))
        / texels;
    means.insert(key, (Arc::downgrade(image), mean));
    mean
}

/// Turns the raw powers sitting in `select_pdf` into the normalised selection
/// distribution and Vose's alias table over it, in O(L).
///
/// The alias table keeps selection at one 1D draw as the light count grows:
/// `sample_light` scales its draw by `L`, takes the integer part as a slot and
/// the fraction as the coin between that slot and its alias. A linear CDF scan
/// would cost O(L) per bounce, which is what an emissive mesh makes matter.
///
/// Falls back to uniform when nothing emits -- a black `DiffuseLight` is still
/// a light as far as `is_light` is concerned, and a distribution of zeroes has
/// no normalisation.
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
/// `snorm` rather than the G-buffer's `pack2x16float`, since the parameters
/// live in exactly `[-1, 1]`. Worth 0.0036 degrees at worst, pinned by
/// `pack_oct_round_trip`.
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
    use crate::geo::transformation::NopTransformer;
    use crate::geo::vec3::Vec3;
    use crate::hittable::LEAF_FLAG;
    use crate::hittable::{Bvh, Hittables, Sphere};
    use crate::material::texture::{SolidColor, Textures};
    use crate::material::{Dielectric, DiffuseLight, Lambertian, Materials};
    use crate::renderer::dielectric_energy::ENERGY_TABLE_LEN;
    use crate::renderer::gpu_data::{LightRef, PRIM_TYPE_SHIFT, PRIM_TYPE_TRIANGLE};
    use crate::renderer::scene_flattener::{
        SceneData, WorldLayout, build_alias_table, flatten_scene, pack_oct, sample_texture,
    };
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
                    &NopTransformer(),
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
            post_processors: vec![],
        })
    }

    /// Who gets an energy table and who does not.
    ///
    /// Building one is the most expensive thing the flattener does per material
    /// -- half a million evaluations of the microfacet model -- and it is
    /// wasted on a scene whose glass is smooth. The shader agrees:
    /// `has_rough_dielectrics` is false under the same condition.
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
    /// A transcription of the shader, which is the point: what is worth pinning
    /// is that the table produces the distribution the MIS weight is told it
    /// produces.
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
    /// that leaves them disagreeing is a quietly wrong image, and nothing on
    /// the GPU can see it.
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
        let sphere = Sphere::new(Vec3::new(0., 0., -2.), 1.0, mat, &NopTransformer());
        let scene = Scene {
            world: Hittables::Bvh(Bvh::new(vec![Hittables::Sphere(sphere)])),
            camera: Default::default(),
            background_color: Default::default(),
            render_config: RenderConfig::default(),
            post_processors: vec![],
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
        let sphere = Sphere::new(Vec3::new(0., 0., -2.), 1.0, mat.clone(), &NopTransformer());

        let mut sub_world: Vec<Hittables> = Vec::new();
        sub_world.push(Hittables::Sphere(sphere.clone()));
        let bvh = Bvh::new(sub_world);

        let scene = Scene {
            world: Hittables::Bvh(Bvh::new(vec![Hittables::Bvh(bvh)])),
            camera: Default::default(),
            background_color: Default::default(),
            render_config: RenderConfig::default(),
            post_processors: vec![],
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

        let sphere = Sphere::new(Vec3::new(0., 0., -2.), 1.0, blend_mat, &NopTransformer());
        let scene = Scene {
            world: Hittables::Bvh(Bvh::new(vec![Hittables::Sphere(sphere)])),
            camera: Default::default(),
            background_color: Default::default(),
            render_config: RenderConfig::default(),
            post_processors: vec![],
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

        let sphere = Sphere::new(Vec3::new(0., 0., -2.), 1.0, mat, &NopTransformer());
        let scene = Scene {
            world: Hittables::Bvh(Bvh::new(vec![Hittables::Sphere(sphere)])),
            camera: Default::default(),
            background_color: Default::default(),
            render_config: RenderConfig::default(),
            post_processors: vec![],
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

    /// A two-texel image, one white and one black, so its mean is nothing like
    /// the texel a UV of (0, 0) lands on.
    fn half_white_image() -> std::sync::Arc<image::RgbImage> {
        let mut img = image::RgbImage::new(2, 1);
        img.put_pixel(0, 0, image::Rgb([255, 255, 255]));
        img.put_pixel(1, 0, image::Rgb([0, 0, 0]));
        std::sync::Arc::new(img)
    }

    fn scene_of(mat: Materials) -> Scene {
        Scene {
            world: Hittables::Bvh(Bvh::new(vec![Hittables::Sphere(Sphere::new(
                Vec3::new(0., 0., -2.),
                1.0,
                mat,
                &NopTransformer(),
            ))])),
            camera: Default::default(),
            background_color: Default::default(),
            render_config: RenderConfig::default(),
            post_processors: vec![],
        }
    }

    /// The shader has one radiance per light and no way to sample a texture for
    /// it, so a textured emitter is a flat light of its texture's mean. It used
    /// to be a flat light of whatever texel sat at UV (0, 0), which is an
    /// arbitrary pixel of the image (#59).
    #[test]
    fn a_textured_emitter_emits_its_texture_mean() {
        use crate::material::texture::ImageMap;

        let tex: Textures = ImageMap::new(half_white_image()).into();
        // What makes this more than a tautology: the mean has to differ from
        // the texel at UV (0, 0).
        assert_eq!(Vec3::new(1., 1., 1.), sample_texture(&tex));

        let data = flatten_scene(&scene_of(
            DiffuseLight {
                tex,
                attenuation_factor: None,
            }
            .into(),
        ));

        assert_eq!(1, data.materials.len());
        assert_eq!([0.5, 0.5, 0.5], data.materials[0].emission);
    }

    /// An emission image is never sampled on the device, so it must take no
    /// texture slot and no atlas area. The trap is `texture_index` falling back
    /// to the emission texture, which is the albedo slot.
    #[test]
    fn an_emission_texture_reaches_neither_the_albedo_slot_nor_the_atlas() {
        use crate::material::texture::ImageMap;

        let data = flatten_scene(&scene_of(
            DiffuseLight {
                tex: ImageMap::new(half_white_image()).into(),
                attenuation_factor: None,
            }
            .into(),
        ));

        assert_eq!(-1, data.materials[0].texture_index);
        assert!(
            data.textures.is_empty(),
            "the atlas holds an emitter's image"
        );
        assert!(data.atlas_layout.is_none());
    }

    /// The albedo of a lit surface still reaches the atlas, so the arm above
    /// removed an emitter's texture rather than texture collection as such.
    #[test]
    fn an_albedo_texture_still_reaches_the_atlas() {
        use crate::material::texture::ImageMap;

        let world = vec![
            Hittables::Sphere(Sphere::new(
                Vec3::new(0., 0., -2.),
                1.,
                Lambertian::new(ImageMap::new(half_white_image()).into(), None).into(),
                &NopTransformer(),
            )),
            Hittables::Sphere(Sphere::new(
                Vec3::new(3., 0., -2.),
                1.,
                DiffuseLight {
                    tex: ImageMap::new(half_white_image()).into(),
                    attenuation_factor: None,
                }
                .into(),
                &NopTransformer(),
            )),
        ];
        let data = flatten_scene(&Scene {
            world: Hittables::Bvh(Bvh::new(world)),
            camera: Default::default(),
            background_color: Default::default(),
            render_config: RenderConfig::default(),
            post_processors: vec![],
        });

        assert_eq!(
            1,
            data.textures.len(),
            "only the albedo belongs in the atlas"
        );
    }

    /// A grid of `n` triangles, enough past `SUBTREE_MIN_PRIMS` to be kept
    /// apart.
    fn grid(n: usize, y: f64) -> Bvh {
        use crate::hittable::Triangle;
        let mat = Lambertian::new(SolidColor::new(0.5, 0.5, 0.5).into(), None);
        Bvh::new(
            (0..n)
                .map(|i| {
                    let (x, z) = ((i % 100) as f64, (i / 100) as f64);
                    Triangle::new(
                        Vec3::new(x, y, z),
                        Vec3::new(x + 1., y, z),
                        Vec3::new(x, y, z + 1.),
                        mat.clone().into(),
                        &NopTransformer(),
                    )
                    .into()
                })
                .collect(),
        )
    }

    fn ball(albedo: f64) -> Hittables {
        Sphere::new(
            Vec3::new(0., 5., 0.),
            1.,
            Lambertian::new(SolidColor::new(albedo, 0.5, 0.5).into(), None).into(),
            &NopTransformer(),
        )
        .into()
    }

    fn lamp() -> Hittables {
        Sphere::new(
            Vec3::new(0., 20., 0.),
            2.,
            DiffuseLight::new(4., 4., 4., None).into(),
            &NopTransformer(),
        )
        .into()
    }

    fn world(items: Vec<Hittables>) -> Hittables {
        Bvh::new(items).into()
    }

    fn offsets<T>(chunks: &[(u32, Vec<T>)]) -> Vec<u32> {
        chunks.iter().map(|(o, _)| *o).collect()
    }

    /// A subtree the arena already holds costs an update nothing: only the
    /// world's own region, at node 0, is written again.
    #[test]
    fn an_unchanged_subtree_is_not_written_again() {
        let mesh = grid(9000, 0.);
        let mut layout = WorldLayout::new(true);
        let first = layout.update(&world(vec![mesh.clone().into(), ball(0.8), lamp()]));
        assert_eq!(2, first.nodes.len(), "the world's region and the mesh's");

        let second = layout.update(&world(vec![mesh.clone().into(), ball(0.2), lamp()]));
        assert!(!second.full);
        assert_eq!(vec![0], offsets(&second.nodes));
        assert!(second.triangle_pos.is_empty(), "the mesh was written again");
        assert_eq!(first.extents.prims, second.extents.prims);
    }

    /// A subtree baked from the same source is written over the old one, at
    /// the same offsets and in the same room.
    #[test]
    fn a_moved_subtree_is_written_over_itself() {
        use crate::geo::transformation::Translation;
        let mesh = grid(9000, 0.);
        let mut layout = WorldLayout::new(true);
        let first = layout.update(&world(vec![mesh.clone().into(), ball(0.8), lamp()]));
        let mut nodes_at = offsets(&first.nodes);
        nodes_at.sort();
        let tris_at = offsets(&first.triangle_pos)[0];

        for y in 1..4 {
            let moved = mesh.transformed(Translation::new(Vec3::new(0., y as f64, 0.)));
            let w = layout.update(&world(vec![moved.into(), ball(0.8), lamp()]));
            assert!(!w.full);
            let mut at = offsets(&w.nodes);
            at.sort();
            assert_eq!(nodes_at, at);
            assert_eq!(vec![tris_at], offsets(&w.triangle_pos));
            assert_eq!(9000, w.triangle_pos[0].1.len());
            assert_eq!(first.extents.nodes, w.extents.nodes);
            assert_eq!(first.extents.prims, w.extents.prims);
        }
    }

    /// A removed subtree frees its room, and the next one of that size takes it.
    #[test]
    fn a_removed_subtree_leaves_room_for_the_next() {
        let mut layout = WorldLayout::new(true);
        let first = layout.update(&world(vec![grid(9000, 0.).into(), lamp()]));
        let tris_at = offsets(&first.triangle_pos)[0];

        let emptied = layout.update(&world(vec![ball(0.5), lamp()]));
        assert!(emptied.extents.prims < first.extents.prims);

        let refilled = layout.update(&world(vec![grid(9000, 3.).into(), lamp()]));
        assert!(!refilled.full);
        assert_eq!(vec![tris_at], offsets(&refilled.triangle_pos));
        assert_eq!(first.extents.prims, refilled.extents.prims);
    }

    /// Dragging a colour slider makes a new material every frame. The ones no
    /// segment uses any more are handed out again, so the table stays the size
    /// of the scene rather than of the drag.
    #[test]
    fn materials_nothing_uses_are_reused() {
        let mut layout = WorldLayout::new(true);
        let mut sizes = Vec::new();
        for i in 0..50 {
            let w = layout.update(&world(vec![ball(i as f64 / 50.), lamp()]));
            sizes.push(w.materials.len());
        }
        assert!(
            sizes.iter().all(|&n| n <= 3),
            "the table grew with the drag: {:?}",
            sizes
        );
    }

    /// Likewise the energy tables, one per index of refraction, when a rough
    /// glass's index is what is being dragged. Two at most: the new one is
    /// built before the old one is let go.
    #[test]
    fn energy_tables_nothing_uses_are_reused() {
        let mut layout = WorldLayout::new(true);
        for i in 0..8 {
            let glass: Hittables = Sphere::new(
                Vec3::new(0., 0., 0.),
                1.,
                Dielectric::new(
                    SolidColor::new(1., 1., 1.).into(),
                    None,
                    1.3 + i as f64 * 0.1,
                    0.4,
                )
                .into(),
                &NopTransformer(),
            )
            .into();
            let w = layout.update(&world(vec![glass, lamp()]));
            assert!(
                w.dielectric_energy.len() <= 2 * ENERGY_TABLE_LEN,
                "update {} holds {} tables",
                i,
                w.dielectric_energy.len() / ENERGY_TABLE_LEN
            );
        }
    }

    /// The world's own region is at node 0, so it cannot move: outgrowing it
    /// lays everything out again.
    #[test]
    fn a_world_that_outgrows_its_region_is_laid_out_again() {
        let mut layout = WorldLayout::new(true);
        let small = layout.update(&world(vec![ball(0.5), lamp()]));
        assert!(small.full);

        let same = layout.update(&world(vec![ball(0.6), lamp()]));
        assert!(!same.full);

        let many: Vec<Hittables> = (0..500)
            .map(|i| {
                Sphere::new(
                    Vec3::new(i as f64, 0., 0.),
                    0.4,
                    Lambertian::new(SolidColor::new(0.5, 0.5, 0.5).into(), None).into(),
                    &NopTransformer(),
                )
                .into()
            })
            .chain([lamp()])
            .collect();
        let grown = layout.update(&world(many));
        assert!(grown.full);
        assert!(grown.extents.spheres >= 501);
    }

    /// Node 0 is always a node, so the traversal needs no special case for an
    /// empty world or one that is a single primitive.
    #[test]
    fn an_empty_world_still_has_a_root() {
        let data = flatten_scene(&Scene {
            world: Bvh::new(vec![]).into(),
            camera: Default::default(),
            background_color: Vec3::new(1., 1., 1.),
            render_config: RenderConfig::default(),
            post_processors: vec![],
        });
        assert_eq!(1, data.nodes.len());
        assert_eq!(LEAF_FLAG, data.nodes[0].left_meta);
        assert_eq!(LEAF_FLAG, data.nodes[0].right_meta);
        assert!(data.prim_refs.is_empty());
    }

    /// A triangle takes the slot of its own reference, so the map stays the
    /// identity in an all-triangle scene however its segments are laid out.
    #[test]
    fn prim_refs_stay_the_identity_across_segments() {
        use crate::hittable::Triangle;
        let lamp: Hittables = Triangle::new(
            Vec3::new(0., 30., 0.),
            Vec3::new(10., 30., 0.),
            Vec3::new(0., 30., 10.),
            DiffuseLight::new(4., 4., 4., None).into(),
            &NopTransformer(),
        )
        .into();
        let data = flatten_scene(&Scene {
            world: world(vec![grid(9000, 0.).into(), grid(9000, 2.).into(), lamp]),
            camera: Default::default(),
            background_color: Vec3::new(1., 1., 1.),
            render_config: RenderConfig::default(),
            post_processors: vec![],
        });
        assert!(data.prim_refs_are_identity);
        for (k, &r) in data.prim_refs.iter().enumerate() {
            assert_eq!((PRIM_TYPE_TRIANGLE << PRIM_TYPE_SHIFT) | k as u32, r);
        }
    }

    /// A baked tree is uploaded straight from its source, without making its
    /// primitives. What it uploads must be what they would have been.
    #[test]
    fn a_baked_tree_uploads_what_its_primitives_would() {
        use crate::geo::transformation::{RotationX, Transformations, Translation};
        use crate::loader::Loader;
        use crate::loader::obj::Obj;
        // Smooth normals and texture coordinates, so every derived field of
        // a triangle is in the comparison.
        let spider = Obj::new("resources/spider/", "spider.obj")
            .load(&NopTransformer(), None)
            .unwrap();
        let moved = spider.transformed(Transformations::new(vec![
            Box::new(RotationX::new(33.)),
            Box::new(Translation::new(Vec3::new(1., 2., 3.))),
        ]));
        let flatten = || {
            flatten_scene(&Scene {
                world: moved.clone().into(),
                camera: Default::default(),
                background_color: Vec3::new(1., 1., 1.),
                render_config: RenderConfig::default(),
                post_processors: vec![],
            })
        };
        let lazy = flatten();
        assert!(matches!(
            moved.leaf_prims(),
            crate::hittable::LeafPrims::Baked(..)
        ));
        moved.prims();
        let eager = flatten();
        assert!(matches!(
            moved.leaf_prims(),
            crate::hittable::LeafPrims::Plain(..)
        ));

        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&eager.triangle_pos),
            bytemuck::cast_slice::<_, u8>(&lazy.triangle_pos)
        );
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&eager.triangle_attr),
            bytemuck::cast_slice::<_, u8>(&lazy.triangle_attr)
        );
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&eager.nodes),
            bytemuck::cast_slice::<_, u8>(&lazy.nodes)
        );
    }
}
