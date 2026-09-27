//! Bounding Volume Hierarchy.
//!
//! The tree is built over an index array (primitives never move during the
//! build) using a binned SAH sweep, and is stored as a flat array of two-child
//! nodes. Each node carries *both* of its children's bounding boxes, which is
//! what lets the GPU traversal in `ray_trace.wgsl` test both children up front,
//! descend into the nearer one immediately, and push only the farther one.
//!
//! Leaves are not nodes of their own: a child slot either points at another
//! node or encodes a contiguous run of primitives inline.
//!
//! A child slot can also hold a whole nested [`Bvh`] of at least
//! [`SUBTREE_MIN_PRIMS`] primitives, kept apart as a subtree rather than
//! dissolved into this tree. That is what makes an edit to a large mesh
//! cheap: the renderer uploads each subtree as its own segment, and a moved
//! mesh ([`Bvh::transformed`]) is written over its old segment in place. The
//! GPU never knows: a subtree's root is spliced in as an ordinary node index.

use std::collections::HashSet;
use std::fmt;
use std::fmt::Display;
use std::sync::{Arc, OnceLock, Weak};

use image::RgbImage;
use rayon::prelude::*;

use crate::geo::Aabb;
use crate::geo::transformation::Transformer;
use crate::hittable::{Hittable, Hittables};

/// Maximum primitives in a single leaf.
pub(crate) const MAX_LEAF_PRIMS: usize = 4;

/// Buckets used by the SAH sweep.
const SAH_BUCKETS: usize = 16;

/// Below this many primitives a subtree is built serially -- forking a rayon
/// task per level costs more than it saves once the subtree is small.
const PARALLEL_CUTOFF: usize = 8192;

/// A nested [`Bvh`] of at least this many primitives is kept apart as a
/// subtree; a smaller one is dissolved into the tree it is nested in.
///
/// Dissolving is what gives the better tree, since a parent that treats a
/// whole subtree as one box cannot interleave it with its neighbours. Keeping
/// apart is what makes an edit cheap. Below this size a rebuild costs about
/// 1.5 ms (`bvh_build/100000` is 18.3 ms), so dissolving wins. The largest
/// nested trees in the test suite are 125 spheres and the 1368-triangle
/// spider, so every golden stays on the dissolving path.
pub const SUBTREE_MIN_PRIMS: usize = 8192;

/// Set on a child meta word to mark it as an inline leaf rather than a node index.
pub(crate) const LEAF_FLAG: u32 = 0x8000_0000;
/// Leaf primitive count, bits 30..24.
pub(crate) const LEAF_COUNT_SHIFT: u32 = 24;
/// Mask for the leaf primitive count once shifted down.
pub(crate) const LEAF_COUNT_MASK: u32 = 0x7F;
/// Leaf primitive offset, bits 23..0.
pub(crate) const LEAF_OFFSET_MASK: u32 = 0x00FF_FFFF;

/// Leaf count that marks a child as a kept-apart subtree, whose index into
/// [`Bvh::subtrees`] is in the offset bits. CPU-side only: the flattener
/// replaces it with the subtree's absolute root node index.
pub(crate) const SUBTREE_COUNT: u32 = LEAF_COUNT_MASK;

/// Largest scene the leaf encoding can address. The renderer lays every
/// segment out in one primitive array, so this bounds the whole scene, holes
/// included, and not just one tree.
pub const MAX_PRIMITIVES: usize = LEAF_OFFSET_MASK as usize;

/// A leaf's count has to fit the seven bits the encoding gives it, below the
/// value that marks a subtree.
const _: () = assert!((MAX_LEAF_PRIMS as u32) < SUBTREE_COUNT);

/// Entries in the GPU traversal stack (`traversal_stack` in `ray_trace.wgsl`).
///
/// Traversal only ever pushes the farther child, so a tree of depth `d` --
/// counting levels of internal nodes -- needs at most `d - 1` entries.
///
/// The bound has to be checked here because overflowing it on the GPU is
/// silent: WGSL's bounds-checking policy clamps the out-of-range store, the
/// deferred subtree is never visited, and geometry disappears with no error.
/// Nothing makes the tree balanced, so `split` can legitimately return 1/N-1.
///
/// The slack is thinner than the synthetic cloud suggests. `bvh_build` reaches
/// 14 / 17 / 21 at 10k / 100k / 1M; clustered real geometry is what sets the
/// bound, and per `bvh_tree_quality` sponza (262k triangles) reaches **27**,
/// conference 25 and a 1M-triangle gallery 24, against a balanced 18-20. So a
/// change that deepens the tree has to be checked against real meshes.
///
/// A kept-apart subtree counts at the depth it hangs from: the bound is on the
/// composed tree the GPU walks. Sponza as a subtree leaves five levels above it.
pub const MAX_TRAVERSAL_DEPTH: u32 = 32;

/// Packs an inline leaf: `count` primitives starting at `offset`.
fn leaf_meta(offset: u32, count: usize) -> u32 {
    debug_assert!(count <= MAX_LEAF_PRIMS);
    debug_assert!(offset as usize <= MAX_PRIMITIVES);
    LEAF_FLAG | ((count as u32) << LEAF_COUNT_SHIFT) | (offset & LEAF_OFFSET_MASK)
}

fn subtree_meta(index: u32) -> u32 {
    LEAF_FLAG | (SUBTREE_COUNT << LEAF_COUNT_SHIFT) | (index & LEAF_OFFSET_MASK)
}

/// What a child meta word points at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Child {
    Node(u32),
    /// `count` primitives from `offset`. A count of 0 is the empty child of a
    /// single-leaf root and intersects nothing.
    Leaf {
        offset: u32,
        count: u32,
    },
    Subtree(u32),
}

pub(crate) fn child_of(meta: u32) -> Child {
    if meta & LEAF_FLAG == 0 {
        return Child::Node(meta);
    }
    let count = (meta >> LEAF_COUNT_SHIFT) & LEAF_COUNT_MASK;
    let offset = meta & LEAF_OFFSET_MASK;
    if count == SUBTREE_COUNT {
        Child::Subtree(offset)
    } else {
        Child::Leaf { offset, count }
    }
}

/// A two-child BVH node. Both child boxes live here; a child is either a node
/// index or an inline leaf, distinguished by [`LEAF_FLAG`] on the meta word.
#[derive(Debug, Clone)]
pub(crate) struct BvhNode {
    pub(crate) left_box: Aabb,
    pub(crate) right_box: Aabb,
    pub(crate) left_meta: u32,
    pub(crate) right_meta: u32,
}

/// Bounding Volume Hierarchy over a set of hittables.
///
/// Immutable once built, and shared: `Clone` is a reference count rather than
/// a copy, and two clones are the same tree to the renderer, which is how it
/// knows a subtree it has already uploaded when it sees it again.
#[derive(Clone)]
pub struct Bvh {
    data: Arc<BvhData>,
}

pub(crate) struct BvhData {
    /// Flat node array; node 0 is the root. Empty when the tree holds no primitives.
    nodes: Vec<BvhNode>,
    /// Primitives in leaf order, so a leaf is a contiguous range.
    ///
    /// Made on first use in a tree [`Bvh::transformed`] baked, from `source`'s
    /// under `transformation`. A mesh kept apart as a subtree is uploaded
    /// straight from its source and never needs them, which is most of what a
    /// move costs: a 250k-triangle mesh is 94 MB of `Triangle` to write and to
    /// free again.
    prims: OnceLock<Vec<Hittables>>,
    /// Kept-apart subtrees, in the order their child slots reach them.
    subtrees: Vec<Bvh>,
    b_box: Aabb,
    /// Levels of internal nodes on the longest root-to-leaf path, through any
    /// subtree; 0 when empty.
    depth: u32,
    /// Primitives here and in every subtree.
    prim_count: usize,
    has_lights: bool,
    /// Every image an albedo or normal map samples, here and in every subtree,
    /// in the order the flattener meets them.
    textures: Vec<Arc<RgbImage>>,
    /// What [`Bvh::transformed`] baked this from, so the next transform starts
    /// from the same geometry rather than from this one. `None` when this tree
    /// is its own source.
    source: Option<Arc<BvhData>>,
    /// What `source` was baked under, when this tree is a bake of it: `None`
    /// for one [`Bvh::rebuilt`] made, which has a source to bake from later
    /// but a topology of its own.
    transformation: Option<Arc<dyn Transformer>>,
}

impl BvhData {
    fn prims(&self) -> &[Hittables] {
        self.prims
            .get_or_init(|| match (&self.source, &self.transformation) {
                (Some(source), Some(t)) => source
                    .prims()
                    .par_iter()
                    .with_min_len(1024)
                    .map(|p| p.transformed(t.as_ref()))
                    .collect(),
                _ => Vec::new(),
            })
    }

    fn into_prims(self) -> Vec<Hittables> {
        if self.prims.get().is_none() {
            self.prims();
        }
        self.prims.into_inner().unwrap()
    }
}

/// A tree's primitives, or how to make them.
pub(crate) enum LeafPrims<'a> {
    Plain(&'a [Hittables]),
    /// Leaf `i` of the tree is `source[i]` under the transformation.
    Baked(&'a [Hittables], &'a dyn Transformer),
}

impl<'a> LeafPrims<'a> {
    /// The primitives, or the ones they are baked from, which are of the same
    /// kinds and materials in the same order.
    pub(crate) fn source(&self) -> &'a [Hittables] {
        match self {
            LeafPrims::Plain(p) | LeafPrims::Baked(p, _) => p,
        }
    }
}

/// `first`, then `then`.
struct Compose(Arc<dyn Transformer>, Arc<dyn Transformer>);

impl Transformer for Compose {
    fn transform(
        &self,
        vec: crate::geo::vec3::Vec3,
        skip_translation: bool,
    ) -> crate::geo::vec3::Vec3 {
        self.1
            .transform(self.0.transform(vec, skip_translation), skip_translation)
    }
}

impl fmt::Debug for Bvh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bvh")
            .field("nodes", &self.data.nodes.len())
            .field("primitives", &self.data.prim_count)
            .field("subtrees", &self.data.subtrees.len())
            .field("depth", &self.data.depth)
            .finish()
    }
}

impl Display for Bvh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{\"nodes\": {}, \"primitives\": {}, \"subtrees\": {}, \"depth\": {}}}",
            self.data.nodes.len(),
            self.data.prim_count,
            self.data.subtrees.len(),
            self.data.depth
        )
    }
}

impl Bvh {
    /// Creates a new hittable object from the given hittable list.
    ///
    /// A nested [`Bvh`] in `list` of fewer than [`SUBTREE_MIN_PRIMS`]
    /// primitives is expanded into its primitives: one global tree beats a
    /// parent that treats a whole subtree as one box. A larger one is kept
    /// apart as a subtree, so an edit to it does not rebuild this tree over
    /// every primitive it holds. Nothing is attached to a nested `Bvh` either
    /// way -- transformations are baked into vertices at construction.
    pub fn new(list: Vec<Hittables>) -> Bvh {
        if !list.iter().any(|h| matches!(h, Hittables::Bvh(_))) {
            // The common case: a loader hands us a flat list of primitives. Moving all of
            // them into a second vector just to discover there was nothing to flatten
            // costs a full copy -- ~78 MB for a 250k-triangle mesh. The discriminant scan
            // that avoids it walks memory the very next statement walks anyway.
            return Bvh::build(list, Vec::new());
        }

        let mut prims = Vec::with_capacity(list.len());
        let mut subtrees = Vec::new();
        for h in list {
            match h {
                Hittables::Bvh(bvh) if bvh.data.prim_count >= SUBTREE_MIN_PRIMS => {
                    subtrees.push(bvh)
                }
                Hittables::Bvh(bvh) => bvh.dissolve_into(&mut prims, &mut subtrees),
                other => prims.push(other),
            }
        }
        Bvh::build(prims, subtrees)
    }

    /// Hands this tree's primitives and subtrees to the tree it is being
    /// dissolved into. Moved when nothing else holds this tree, cloned when
    /// something does.
    fn dissolve_into(self, prims: &mut Vec<Hittables>, subtrees: &mut Vec<Bvh>) {
        match Arc::try_unwrap(self.data) {
            Ok(data) => {
                subtrees.extend(data.subtrees.iter().cloned());
                prims.extend(data.into_prims());
            }
            Err(data) => {
                prims.extend(data.prims().iter().cloned());
                subtrees.extend(data.subtrees.iter().cloned());
            }
        }
    }

    fn build(mut prims: Vec<Hittables>, mut subtrees: Vec<Bvh>) -> Bvh {
        let prim_count = prims.len() + subtrees.iter().map(|s| s.data.prim_count).sum::<usize>();
        assert!(
            prim_count <= MAX_PRIMITIVES,
            "BVH supports at most {} primitives, got {}",
            MAX_PRIMITIVES,
            prim_count
        );

        loop {
            let n_items = prims.len() + subtrees.len();
            if n_items == 0 {
                return Bvh::from_data(BvhData {
                    nodes: Vec::new(),
                    prims: OnceLock::from(Vec::new()),
                    subtrees: Vec::new(),
                    b_box: Default::default(),
                    depth: 0,
                    prim_count: 0,
                    has_lights: false,
                    textures: Vec::new(),
                    source: None,
                    transformation: None,
                });
            }

            // Bounding boxes and centroids are derived once here. The old build
            // re-derived `bounding_box().center()` inside the sort comparator, so
            // it recomputed them O(N log N) times per level.
            let boxes: Vec<Aabb32> = prims
                .iter()
                .map(|p| Aabb32::from(p.bounding_box()))
                .chain(subtrees.iter().map(|s| Aabb32::from(&s.data.b_box)))
                .collect();
            let centroids: Vec<[f32; 3]> = boxes.iter().map(|b| b.center()).collect();

            let mut b_box = match prims.first() {
                Some(p) => p.bounding_box().clone(),
                None => Aabb::default(),
            };
            for p in prims.iter().skip(1) {
                b_box = b_box.combine(p.bounding_box());
            }
            for s in &subtrees {
                b_box = b_box.combine(&s.data.b_box);
            }

            let ctx = BuildCtx {
                boxes: &boxes,
                centroids: &centroids,
                n_loose: prims.len() as u32,
                subtree_depths: subtrees.iter().map(|s| s.data.depth).collect(),
            };
            let mut indices: Vec<u32> = (0..n_items as u32).collect();

            let (mut nodes, depth) = if subtrees.is_empty() && prims.len() <= MAX_LEAF_PRIMS {
                // Everything fits in one leaf. Emit a root whose left child is that
                // leaf and whose right child is an empty leaf: a zero-count leaf
                // intersects nothing, so its box never needs to fail the slab test.
                (
                    vec![BvhNode {
                        left_box: b_box.clone(),
                        right_box: Aabb::default(),
                        left_meta: leaf_meta(0, prims.len()),
                        right_meta: leaf_meta(0, 0),
                    }],
                    1,
                )
            } else if n_items == 1 {
                // One subtree and nothing else, which cannot share a leaf.
                (
                    vec![BvhNode {
                        left_box: boxes[0].to_aabb(),
                        right_box: Aabb::default(),
                        left_meta: subtree_meta(0),
                        right_meta: leaf_meta(0, 0),
                    }],
                    1 + ctx.subtree_depths[0],
                )
            } else {
                let mut nodes = Vec::with_capacity(n_items / MAX_LEAF_PRIMS);
                let depth = build_parallel(&mut nodes, &mut indices, 0, &ctx);
                (nodes, depth)
            };

            if depth > MAX_TRAVERSAL_DEPTH && !subtrees.is_empty() {
                // Deeper than the GPU stack, but only through a subtree. Taking
                // the deepest one back in costs its edits their speed and keeps
                // the geometry, which a panic or a silent GPU overflow would not.
                let deepest = deepest_subtree(&nodes, &indices, &ctx);
                let removed = subtrees.remove(deepest);
                removed.dissolve_into(&mut prims, &mut subtrees);
                continue;
            }

            assert!(
                depth <= MAX_TRAVERSAL_DEPTH,
                "BVH depth {} exceeds the {} levels the GPU traversal stack can \
                 hold, which would silently drop geometry from the image",
                depth,
                MAX_TRAVERSAL_DEPTH
            );

            if subtrees.is_empty() {
                permute_in_place(&mut prims, &indices);
            } else {
                // The build numbered leaves and subtrees by position in one
                // order over both. Split it back into an order over each.
                let n_loose = ctx.n_loose;
                let mut subtrees_before = Vec::with_capacity(indices.len());
                let mut count = 0u32;
                for &i in &indices {
                    subtrees_before.push(count);
                    if i >= n_loose {
                        count += 1;
                    }
                }
                let remap = |meta: &mut u32| match child_of(*meta) {
                    Child::Subtree(position) => {
                        *meta = subtree_meta(subtrees_before[position as usize])
                    }
                    Child::Leaf { offset, count } if count > 0 => {
                        *meta = leaf_meta(offset - subtrees_before[offset as usize], count as usize)
                    }
                    _ => {}
                };
                for node in &mut nodes {
                    remap(&mut node.left_meta);
                    remap(&mut node.right_meta);
                }

                let loose_order: Vec<u32> =
                    indices.iter().copied().filter(|&i| i < n_loose).collect();
                permute_in_place(&mut prims, &loose_order);
                subtrees = indices
                    .iter()
                    .filter(|&&i| i >= n_loose)
                    .map(|&i| subtrees[(i - n_loose) as usize].clone())
                    .collect();
            }

            let has_lights = prims.par_iter().any(|p| p.has_lights())
                || subtrees.iter().any(|s| s.data.has_lights);
            let textures = collect_textures(&prims, &subtrees);

            return Bvh::from_data(BvhData {
                nodes,
                prims: OnceLock::from(prims),
                subtrees,
                b_box,
                depth,
                prim_count,
                has_lights,
                textures,
                source: None,
                transformation: None,
            });
        }
    }

    fn from_data(data: BvhData) -> Bvh {
        Bvh {
            data: Arc::new(data),
        }
    }

    /// This tree moved, turned or scaled, without rebuilding it.
    ///
    /// The node boxes are refitted, in one reverse pass over the same
    /// topology, to the primitives baked again from the geometry this tree was
    /// built with. So the result has the same shape as `self`, which is what
    /// lets the renderer write it over the old one in place. The primitives
    /// themselves are only made if something asks for them: a renderer uploads
    /// them straight from the source.
    ///
    /// The transformation is absolute, not a step: `a.transformed(t)` and
    /// `a.transformed(s).transformed(t)` are the same tree, both baked from
    /// `a`'s geometry. Composing steps would accumulate rounding error across
    /// a drag.
    ///
    /// A translation or a uniform scale gives the tree a rebuild would, up to
    /// rounding, since the SAH cannot see either. A rotation off an axis
    /// loosens the boxes: 3 to 18% on sponza and the dragon at 15 to 45
    /// degrees, per `refit_against_rebuild`.
    ///
    /// ```
    /// # use solstrale::geo::transformation::{NopTransformer, Translation};
    /// # use solstrale::geo::vec3::Vec3;
    /// # use solstrale::loader::Loader;
    /// # use solstrale::loader::obj::Obj;
    /// let mesh = Obj::new("resources/spider/", "spider.obj")
    ///     .load(&NopTransformer(), None)
    ///     .unwrap();
    /// let moved = mesh.transformed(Translation::new(Vec3::new(0., 1., 0.)));
    /// assert_eq!(mesh.depth(), moved.depth());
    /// ```
    pub fn transformed(&self, transformation: impl Transformer + 'static) -> Bvh {
        let source = self
            .data
            .source
            .clone()
            .unwrap_or_else(|| self.data.clone());
        Bvh::bake(source, Arc::new(transformation))
    }

    /// `source` under `transformation`. A subtree is baked from itself as it
    /// sits in `source`, which is `source`'s own geometry however that subtree
    /// came to be.
    fn bake(source: Arc<BvhData>, transformation: Arc<dyn Transformer>) -> Bvh {
        let subtrees: Vec<Bvh> = source
            .subtrees
            .iter()
            .map(|s| match (&s.data.source, &s.data.transformation) {
                // Baked already: bake its source under both, rather than a
                // bake of a bake.
                (Some(inner), Some(t)) => Bvh::bake(
                    inner.clone(),
                    Arc::new(Compose(t.clone(), transformation.clone())),
                ),
                _ => Bvh::bake(s.data.clone(), transformation.clone()),
            })
            .collect();

        let prims = source.prims();
        let t = transformation.as_ref();
        let bounds = |i: usize| match &prims[i] {
            Hittables::Triangle(tri) => tri.transformed_bounds(t),
            other => other.transformed(t).bounding_box().clone(),
        };
        let (nodes, b_box) = refit(&source.nodes, prims.len(), &bounds, &subtrees);

        Bvh::from_data(BvhData {
            nodes,
            prims: OnceLock::new(),
            subtrees,
            b_box,
            depth: source.depth,
            prim_count: source.prim_count,
            has_lights: source.has_lights,
            textures: source.textures.clone(),
            source: Some(source.clone()),
            transformation: Some(transformation),
        })
    }

    /// This tree's primitives under a tree built for where they are now.
    ///
    /// A tree [`Bvh::transformed`] baked keeps the topology it was built with,
    /// and a rotation off an axis loosens it: at 64 spp, sponza and the dragon
    /// render 12 to 23% slower refitted than rebuilt at 30 to 45 degrees about
    /// Y. So a caller moving a mesh interactively can call this on a
    /// background thread once the drag ends, and swap the result in with a
    /// world update, which uploads it once as a new subtree.
    ///
    /// Makes the primitives if they were not made yet. The result still bakes
    /// a later `transformed` from the geometry `self` was baked from.
    pub fn rebuilt(&self) -> Bvh {
        let Some(source) = &self.data.source else {
            return self.clone();
        };
        let subtrees = self.data.subtrees.iter().map(Bvh::rebuilt).collect();
        let mut bvh = Bvh::build(self.data.prims().to_vec(), subtrees);
        Arc::get_mut(&mut bvh.data)
            .expect("a tree just built is not shared")
            .source = Some(source.clone());
        bvh
    }

    /// Levels of internal nodes on the longest root-to-leaf path, through any
    /// subtree.
    ///
    /// This is what [`MAX_TRAVERSAL_DEPTH`] bounds; see it for the exact
    /// relation to the GPU stack.
    pub fn depth(&self) -> u32 {
        self.data.depth
    }

    /// Primitives in this tree, counting every subtree's.
    pub fn primitive_count(&self) -> usize {
        self.data.prim_count
    }

    pub(crate) fn nodes(&self) -> &[BvhNode] {
        &self.data.nodes
    }

    /// Primitives in leaf order, not counting any subtree's. Made on first use
    /// in a tree `transformed` baked; see [`Self::leaf_prims`].
    #[cfg(test)]
    pub(crate) fn prims(&self) -> &[Hittables] {
        self.data.prims()
    }

    /// This tree's primitives if it has them, and otherwise what they are
    /// baked from, so a flattener need not make them.
    pub(crate) fn leaf_prims(&self) -> LeafPrims<'_> {
        match (&self.data.source, &self.data.transformation) {
            (Some(source), Some(t)) if self.data.prims.get().is_none() => {
                LeafPrims::Baked(source.prims(), t.as_ref())
            }
            _ => LeafPrims::Plain(self.data.prims()),
        }
    }

    pub(crate) fn subtrees(&self) -> &[Bvh] {
        &self.data.subtrees
    }

    pub(crate) fn textures(&self) -> &[Arc<RgbImage>] {
        &self.data.textures
    }

    /// Identity of this tree, as an address.
    pub(crate) fn id_ptr(&self) -> usize {
        Arc::as_ptr(&self.data) as usize
    }

    /// Identity of this tree, for a renderer that must not keep it alive.
    pub(crate) fn downgrade(&self) -> Weak<BvhData> {
        Arc::downgrade(&self.data)
    }

    /// Identity of this tree's topology: the tree it was baked from if it came
    /// out of [`Bvh::transformed`], and itself otherwise. Two trees with the
    /// same topology have the same nodes, and the same primitives in the same
    /// leaf order, under different transformations -- which is what lets a
    /// renderer write one over the other and keep its material indices.
    ///
    /// Not the same as the source a later `transformed` bakes from: a tree
    /// [`Bvh::rebuilt`] made keeps that source but has a topology of its own.
    pub(crate) fn topology(&self) -> Weak<BvhData> {
        match (&self.data.source, &self.data.transformation) {
            (Some(source), Some(_)) => Arc::downgrade(source),
            _ => Arc::downgrade(&self.data),
        }
    }

    /// Whether `weak` is this tree. A `Weak` pins its allocation, so the
    /// address cannot have been reused by another tree.
    pub(crate) fn is(&self, weak: &Weak<BvhData>) -> bool {
        std::ptr::eq(weak.as_ptr(), Arc::as_ptr(&self.data))
    }
}

/// The unique images an albedo or normal map reads, in the order the
/// flattener meets them: this tree's primitives in leaf order, then each
/// subtree's.
fn collect_textures(prims: &[Hittables], subtrees: &[Bvh]) -> Vec<Arc<RgbImage>> {
    let mut seen: HashSet<*const RgbImage> = HashSet::new();
    let mut textures = Vec::new();
    // A mesh shares one material across every triangle, so nearly every
    // lookup is the one before it.
    let mut last: *const RgbImage = std::ptr::null();
    let mut add = |image: &Arc<RgbImage>| {
        let ptr = Arc::as_ptr(image);
        if ptr != last {
            last = ptr;
            if seen.insert(ptr) {
                textures.push(image.clone());
            }
        }
    };
    for p in prims {
        if let Some(mat) = p.material() {
            mat.for_each_atlas_texture(&mut add);
        }
    }
    for s in subtrees {
        for t in &s.data.textures {
            add(t);
        }
    }
    textures
}

/// Which subtree to take back in when the composed tree is too deep: the one
/// whose own depth plus the depth it hangs from is greatest. `indices` is the
/// build order, and a subtree child still carries its position in it.
fn deepest_subtree(nodes: &[BvhNode], indices: &[u32], ctx: &BuildCtx) -> usize {
    let mut best = (0, 0);
    let mut stack = vec![(0u32, 1u32)];
    while let Some((idx, level)) = stack.pop() {
        let node = &nodes[idx as usize];
        for meta in [node.left_meta, node.right_meta] {
            match child_of(meta) {
                Child::Node(n) => stack.push((n, level + 1)),
                Child::Subtree(position) => {
                    let subtree = (indices[position as usize] - ctx.n_loose) as usize;
                    let depth = level + ctx.subtree_depths[subtree];
                    if depth > best.0 {
                        best = (depth, subtree);
                    }
                }
                Child::Leaf { .. } => {}
            }
        }
    }
    best.1
}

/// Recomputes every box of `src` over new primitives of the same topology,
/// and the box of the whole tree. `bounds(i)` is the box of leaf primitive `i`.
///
/// The builder always places a child after its parent, so one reverse pass
/// sees every child before the node that holds it. Boxes are unions of the
/// same outward-rounded f32 boxes the builder takes them from, so an unchanged
/// primitive set refits to exactly the boxes it was built with.
fn refit(
    src: &[BvhNode],
    prim_count: usize,
    bounds: &(dyn Fn(usize) -> Aabb + Sync),
    subtrees: &[Bvh],
) -> (Vec<BvhNode>, Aabb) {
    // Leaves and subtrees first, in parallel: they are the bulk of the work
    // and depend on nothing else. The tree's own box comes out of the same
    // pass, in f64 as the builder takes it, since every primitive is in
    // exactly one leaf.
    type Fixed = ([Option<Aabb32>; 2], Aabb);
    let fixed: Vec<Fixed> = src
        .par_iter()
        .with_min_len(1024)
        .map(|node| {
            let mut all = Aabb::default();
            let children = [node.left_meta, node.right_meta].map(|meta| match child_of(meta) {
                Child::Leaf { count: 0, .. } => Some(Aabb32::EMPTY),
                Child::Leaf { offset, count } => {
                    let mut acc = Aabb32::EMPTY;
                    for i in offset as usize..(offset + count) as usize {
                        let b = bounds(i);
                        acc = acc.union(&Aabb32::from(&b));
                        all = all.combine(&b);
                    }
                    Some(acc)
                }
                Child::Subtree(i) => Some(Aabb32::from(&subtrees[i as usize].data.b_box)),
                Child::Node(_) => None,
            });
            (children, all)
        })
        .collect();
    let b_box = fixed
        .iter()
        .fold(Aabb::default(), |acc, (_, all)| acc.combine(all));
    let b_box = subtrees
        .iter()
        .fold(b_box, |acc, s| acc.combine(&s.data.b_box));

    let mut nodes = src.to_vec();
    if nodes.len() == 1 && subtrees.is_empty() && prim_count <= MAX_LEAF_PRIMS {
        // The single-leaf root, whose box is the tree's own as the builder
        // wrote it.
        nodes[0].left_box = b_box.clone();
        return (nodes, b_box);
    }

    let mut unions = vec![Aabb32::EMPTY; nodes.len()];
    for i in (0..nodes.len()).rev() {
        let [left, right] = [
            (nodes[i].left_meta, fixed[i].0[0]),
            (nodes[i].right_meta, fixed[i].0[1]),
        ]
        .map(|(meta, fixed)| {
            fixed.unwrap_or_else(|| match child_of(meta) {
                Child::Node(n) => unions[n as usize],
                _ => unreachable!(),
            })
        });
        let empty_right = matches!(child_of(nodes[i].right_meta), Child::Leaf { count: 0, .. });
        nodes[i].left_box = left.to_aabb();
        nodes[i].right_box = if empty_right {
            Aabb::default()
        } else {
            right.to_aabb()
        };
        unions[i] = left.union(&right);
    }
    (nodes, b_box)
}

/// Reorders `prims` so that `prims[i]` ends up holding what `prims[indices[i]]`
/// held, in place: no second array, and every primitive moves once.
///
/// The trade against a gather through a staging array is memory for time past
/// roughly 400k primitives, where the random-access swaps stop fitting in
/// last-level cache. Measured on `bvh_build` on a 96 MB-L3 part:
///
/// | primitives | array | gather | in place |
/// |---|---|---|---|
/// | 100k | 31 MB | 28.5 ms | **18.3 ms** |
/// | 250k | 78 MB | 75.3 ms | **70.2 ms** |
/// | 500k | 156 MB | **158.7 ms** | 161.4 ms |
/// | 1M | 312 MB | **376.3 ms** | 421.7 ms |
///
/// ~11% of build time at 1M, against 303 MB of peak RSS -- build time is paid
/// once at load, where the peak is what decides whether a scene fits at all.
///
/// Note the direction: `indices[i]` is the *source* slot for destination `i`.
/// Once slot `i` has been written, a later `indices[j]` pointing at it is
/// stale, so the chase follows `indices` forward until it lands on a slot that
/// has not been overwritten (`src >= i`). Swapping `prims` and `indices`
/// together until `indices[k] == k` looks equivalent and is not -- it applies
/// the inverse permutation, which the golden-image tests would accept.
fn permute_in_place(prims: &mut [Hittables], indices: &[u32]) {
    for i in 0..prims.len() {
        let mut src = indices[i] as usize;
        while src < i {
            src = indices[src] as usize;
        }
        prims.swap(i, src);
    }
}

/// What the build reads but never writes. Items `0..n_loose` are primitives,
/// the rest subtrees.
struct BuildCtx<'a> {
    boxes: &'a [Aabb32],
    centroids: &'a [[f32; 3]],
    n_loose: u32,
    subtree_depths: Vec<u32>,
}

impl BuildCtx<'_> {
    /// The child slot for `indices`, when it needs no node of its own: a
    /// leaf of primitives, or a single subtree. A subtree never shares a leaf.
    fn terminal(&self, indices: &[u32], offset: u32) -> Option<(u32, u32)> {
        if let [i] = indices
            && *i >= self.n_loose
        {
            return Some((
                subtree_meta(offset),
                self.subtree_depths[(i - self.n_loose) as usize],
            ));
        }
        if indices.len() <= MAX_LEAF_PRIMS && indices.iter().all(|&i| i < self.n_loose) {
            return Some((leaf_meta(offset, indices.len()), 0));
        }
        None
    }
}

/// Builds a subtree, forking the two halves onto rayon while they are large
/// enough to be worth it. Subtree nodes are built into their own vectors and
/// spliced, which only happens for the handful of levels above the cutoff.
///
/// Returns the subtree's depth, in levels of internal nodes.
fn build_parallel(
    nodes: &mut Vec<BvhNode>,
    indices: &mut [u32],
    offset: u32,
    ctx: &BuildCtx,
) -> u32 {
    if indices.len() < PARALLEL_CUTOFF {
        return build_serial(nodes, indices, offset, ctx).1;
    }

    let mid = split(indices, ctx.boxes, ctx.centroids);
    let (l_idx, r_idx) = indices.split_at_mut(mid);
    let r_offset = offset + mid as u32;

    let l_box = union_of(l_idx, ctx.boxes);
    let r_box = union_of(r_idx, ctx.boxes);
    let l_terminal = ctx.terminal(l_idx, offset);
    let r_terminal = ctx.terminal(r_idx, r_offset);

    let ((l_sub, l_depth), (r_sub, r_depth)) = rayon::join(
        || {
            let mut v = Vec::new();
            let d = match l_terminal {
                Some((_, d)) => d,
                None => build_parallel(&mut v, l_idx, offset, ctx),
            };
            (v, d)
        },
        || {
            let mut v = Vec::new();
            let d = match r_terminal {
                Some((_, d)) => d,
                None => build_parallel(&mut v, r_idx, r_offset, ctx),
            };
            (v, d)
        },
    );

    // Splice both subtrees in after this node, rebasing their node references.
    let base = nodes.len() as u32;
    let base_l = base + 1;
    let base_r = base_l + l_sub.len() as u32;

    nodes.push(BvhNode {
        left_box: l_box,
        right_box: r_box,
        left_meta: l_terminal.map_or(base_l, |(meta, _)| meta),
        right_meta: r_terminal.map_or(base_r, |(meta, _)| meta),
    });

    for (sub, sub_base) in [(l_sub, base_l), (r_sub, base_r)] {
        for mut n in sub {
            if n.left_meta & LEAF_FLAG == 0 {
                n.left_meta += sub_base;
            }
            if n.right_meta & LEAF_FLAG == 0 {
                n.right_meta += sub_base;
            }
            nodes.push(n);
        }
    }

    1 + l_depth.max(r_depth)
}

/// Builds a subtree directly into `nodes`, returning its root index and its
/// depth in levels of internal nodes.
///
/// The recursion is as deep as the tree, so the depth the caller asserts
/// against [`MAX_TRAVERSAL_DEPTH`] bounds this stack too -- but only after the
/// fact: a distribution pathological enough to overflow the native stack does
/// so before there is a depth to check.
fn build_serial(
    nodes: &mut Vec<BvhNode>,
    indices: &mut [u32],
    offset: u32,
    ctx: &BuildCtx,
) -> (u32, u32) {
    let my = nodes.len() as u32;
    nodes.push(BvhNode {
        left_box: Aabb::default(),
        right_box: Aabb::default(),
        left_meta: 0,
        right_meta: 0,
    });

    let mid = split(indices, ctx.boxes, ctx.centroids);
    let (l_idx, r_idx) = indices.split_at_mut(mid);
    let r_offset = offset + mid as u32;

    let left_box = union_of(l_idx, ctx.boxes);
    let right_box = union_of(r_idx, ctx.boxes);

    let (left_meta, left_depth) = match ctx.terminal(l_idx, offset) {
        Some(terminal) => terminal,
        None => build_serial(nodes, l_idx, offset, ctx),
    };
    let (right_meta, right_depth) = match ctx.terminal(r_idx, r_offset) {
        Some(terminal) => terminal,
        None => build_serial(nodes, r_idx, r_offset, ctx),
    };

    nodes[my as usize] = BvhNode {
        left_box,
        right_box,
        left_meta,
        right_meta,
    };
    (my, 1 + left_depth.max(right_depth))
}

/// Chooses a split point with a binned SAH sweep over all three centroid axes.
///
/// Returns an index into `indices`, which is partitioned in place. Always
/// returns a value in `1..indices.len()` so neither side is empty.
///
/// All three rather than only the widest: the widest centroid axis is a proxy
/// for the axis that best separates the boxes, and a plate or a shell spread
/// along a wide axis is where the proxy is wrong. Binned in one pass, because
/// that pass is a gather through `centroids` and `boxes` and the extra bucket
/// updates run on data already in registers.
fn split(indices: &mut [u32], boxes: &[Aabb32], centroids: &[[f32; 3]]) -> usize {
    let n = indices.len();

    let mut c_min = [f32::INFINITY; 3];
    let mut c_max = [f32::NEG_INFINITY; 3];
    for &i in indices.iter() {
        let c = centroids[i as usize];
        for a in 0..3 {
            c_min[a] = c_min[a].min(c[a]);
            c_max[a] = c_max[a].max(c[a]);
        }
    }

    let mut scale = [0f32; 3];
    let mut any_axis = false;
    for a in 0..3 {
        let extent = c_max[a] - c_min[a];
        // An axis with no extent keeps scale 0, so every primitive lands in its
        // bucket 0 and its sweep finds no plane with both sides non-empty. That
        // is cheaper than branching on it once per primitive below.
        if extent.is_nan() || extent <= 0.0 {
            continue;
        }
        scale[a] = SAH_BUCKETS as f32 / extent;
        any_axis = true;
    }
    if !any_axis {
        // All centroids coincide on every axis; nothing to separate spatially.
        return n / 2;
    }

    let mut bucket_box = [[Aabb32::EMPTY; SAH_BUCKETS]; 3];
    let mut bucket_count = [[0u32; SAH_BUCKETS]; 3];
    for &i in indices.iter() {
        let b = &boxes[i as usize];
        let c = centroids[i as usize];
        for a in 0..3 {
            let k = bucket_of(c[a], c_min[a], scale[a]);
            bucket_count[a][k] += 1;
            bucket_box[a][k] = bucket_box[a][k].union(b);
        }
    }

    let mut best_cost = f32::INFINITY;
    let mut best_axis = usize::MAX;
    let mut best_k = usize::MAX;
    for a in 0..3 {
        if let Some((cost, k)) = sweep(&bucket_box[a], &bucket_count[a])
            && cost < best_cost
        {
            best_cost = cost;
            best_axis = a;
            best_k = k;
        }
    }

    if best_axis == usize::MAX {
        return n / 2;
    }

    // Partition in place on the same bucket assignment the sweep used, so the
    // split matches the cost that was evaluated. This replaces a full
    // `sort_unstable_by` per level -- O(N) instead of O(N log N).
    let mid = partition(indices, |i| {
        bucket_of(
            centroids[i as usize][best_axis],
            c_min[best_axis],
            scale[best_axis],
        ) < best_k
    });
    if mid == 0 || mid == n { n / 2 } else { mid }
}

/// Sweeps one axis's buckets and returns its cheapest plane as
/// `(cost, bucket)`, or `None` when no plane leaves both sides non-empty.
///
/// The cost is the unnormalised `A_L * N_L + A_R * N_R`. A parent-area
/// divisor and a traversal constant are the same for every candidate plane on
/// every axis, so neither could move the argmin.
fn sweep(
    bucket_box: &[Aabb32; SAH_BUCKETS],
    bucket_count: &[u32; SAH_BUCKETS],
) -> Option<(f32, usize)> {
    // Forward sweep: cost of everything left of each split plane.
    let mut left_area = [0f32; SAH_BUCKETS];
    let mut left_count = [0u32; SAH_BUCKETS];
    let mut acc = Aabb32::EMPTY;
    let mut count = 0u32;
    for k in 0..SAH_BUCKETS {
        acc = acc.union(&bucket_box[k]);
        count += bucket_count[k];
        left_area[k] = acc.half_area();
        left_count[k] = count;
    }

    // Backward sweep, picking the cheapest plane.
    let mut best = None;
    let mut best_cost = f32::INFINITY;
    let mut acc = Aabb32::EMPTY;
    let mut right_count = 0u32;
    for k in (1..SAH_BUCKETS).rev() {
        acc = acc.union(&bucket_box[k]);
        right_count += bucket_count[k];
        let lc = left_count[k - 1];
        if lc == 0 || right_count == 0 {
            continue;
        }
        let cost = left_area[k - 1] * lc as f32 + acc.half_area() * right_count as f32;
        if cost < best_cost {
            best_cost = cost;
            best = Some((cost, k));
        }
    }
    best
}

/// Which bucket a centroid coordinate falls in.
fn bucket_of(c: f32, c_min: f32, scale: f32) -> usize {
    (((c - c_min) * scale) as usize).min(SAH_BUCKETS - 1)
}

/// Stable-enough in-place partition; returns the length of the true-side prefix.
fn partition<F: Fn(u32) -> bool>(indices: &mut [u32], pred: F) -> usize {
    let mut i = 0;
    let mut j = indices.len();
    while i < j {
        if pred(indices[i]) {
            i += 1;
        } else {
            j -= 1;
            indices.swap(i, j);
        }
    }
    i
}

fn union_of(indices: &[u32], boxes: &[Aabb32]) -> Aabb {
    let mut acc = Aabb32::EMPTY;
    for &i in indices {
        acc = acc.union(&boxes[i as usize]);
    }
    acc.to_aabb()
}

/// Compact f32 bounding box used during the build.
///
/// f32 keeps the build arrays half the size of the f64 `Aabb` and matches what
/// the GPU consumes; conversion rounds outward so the box never clips geometry.
#[derive(Copy, Clone, Debug)]
struct Aabb32 {
    min: [f32; 3],
    max: [f32; 3],
}

impl Aabb32 {
    const EMPTY: Aabb32 = Aabb32 {
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
    };

    fn from(a: &Aabb) -> Aabb32 {
        Aabb32 {
            min: [
                round_down(a.x.min),
                round_down(a.y.min),
                round_down(a.z.min),
            ],
            max: [round_up(a.x.max), round_up(a.y.max), round_up(a.z.max)],
        }
    }

    fn union(&self, o: &Aabb32) -> Aabb32 {
        Aabb32 {
            min: [
                self.min[0].min(o.min[0]),
                self.min[1].min(o.min[1]),
                self.min[2].min(o.min[2]),
            ],
            max: [
                self.max[0].max(o.max[0]),
                self.max[1].max(o.max[1]),
                self.max[2].max(o.max[2]),
            ],
        }
    }

    fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Half the surface area -- the SAH only compares these, so the constant
    /// factor of 2 is dropped.
    fn half_area(&self) -> f32 {
        let d = [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ];
        if d[0] < 0.0 {
            return 0.0;
        }
        d[0] * d[1] + d[1] * d[2] + d[2] * d[0]
    }

    fn to_aabb(self) -> Aabb {
        use crate::util::interval::Interval;
        Aabb {
            x: Interval::new(self.min[0] as f64, self.max[0] as f64),
            y: Interval::new(self.min[1] as f64, self.max[1] as f64),
            z: Interval::new(self.min[2] as f64, self.max[2] as f64),
        }
    }
}

/// Narrows to f32 without ever rounding the bound inward.
fn round_down(v: f64) -> f32 {
    let f = v as f32;
    if (f as f64) > v { f.next_down() } else { f }
}

fn round_up(v: f64) -> f32 {
    let f = v as f32;
    if (f as f64) < v { f.next_up() } else { f }
}

impl Hittable for Bvh {
    fn bounding_box(&self) -> &Aabb {
        &self.data.b_box
    }

    fn has_lights(&self) -> bool {
        self.data.has_lights
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::transformation::NopTransformer;
    use crate::geo::vec3::Vec3;
    use crate::hittable::Triangle;
    use crate::loader::Loader;
    use crate::loader::obj::Obj;
    use crate::material::Lambertian;
    use crate::material::texture::SolidColor;

    /// Cost of one node traversal relative to one primitive intersection.
    const TRAVERSAL_COST: f32 = 1.0;

    /// A triangle whose `v0.x` is `i`, so a permutation is readable off the result.
    fn tagged(i: u32) -> Hittables {
        let mat = Lambertian::new(SolidColor::new(1., 1., 1.).into(), None);
        Triangle::new(
            Vec3::new(i as f64, 0., 0.),
            Vec3::new(i as f64, 1., 0.),
            Vec3::new(i as f64, 0., 1.),
            mat.into(),
            &NopTransformer(),
        )
        .into()
    }

    fn tags(prims: &[Hittables]) -> Vec<u32> {
        prims
            .iter()
            .map(|p| match p {
                Hittables::Triangle(t) => t.v0.x as u32,
                _ => unreachable!(),
            })
            .collect()
    }

    /// The direction of the permutation is the easy thing to get backwards here, and the
    /// golden-image tests would not catch it -- they compare stochastic renders at a 0.95
    /// RMS threshold, which a reordered primitive array sails through.
    #[test]
    fn permute_in_place_matches_the_gather_it_replaced() {
        // Cheap deterministic LCG, as elsewhere in this crate.
        let mut state: u32 = 0x9E37_79B9;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        };

        for n in [1usize, 2, 3, 7, 64, 257] {
            // Fisher-Yates, so every permutation is reachable.
            let mut indices: Vec<u32> = (0..n as u32).collect();
            for i in (1..n).rev() {
                indices.swap(i, next() as usize % (i + 1));
            }

            let original: Vec<Hittables> = (0..n as u32).map(tagged).collect();
            let original_tags = tags(&original);
            let expected: Vec<u32> = indices.iter().map(|&i| original_tags[i as usize]).collect();

            let mut prims = original;
            permute_in_place(&mut prims, &indices);
            assert_eq!(expected, tags(&prims), "n = {}", n);
        }
    }

    /// The worked example from the doc comment: the inverse permutation would give
    /// `[C, A, B]` here, which is exactly the mistake this guards.
    #[test]
    fn permute_in_place_applies_the_forward_permutation() {
        let mut prims: Vec<Hittables> = (0..3).map(tagged).collect();
        permute_in_place(&mut prims, &[1, 2, 0]);
        assert_eq!(vec![1, 2, 0], tags(&prims));
    }

    /// Depth read back off the finished node array, which is the only thing the
    /// GPU traversal actually walks.
    fn measured_depth(bvh: &Bvh, idx: u32) -> u32 {
        let child = |meta: u32| match child_of(meta) {
            Child::Node(n) => measured_depth(bvh, n),
            Child::Subtree(s) => measured_depth(&bvh.subtrees()[s as usize], 0),
            Child::Leaf { .. } => 0,
        };
        let node = &bvh.nodes()[idx as usize];
        1 + child(node.left_meta).max(child(node.right_meta))
    }

    /// A cloud of `n` triangles at pseudo-random positions, as in the bench.
    fn cloud(n: u32) -> Vec<Hittables> {
        let mut state: u32 = 0x9E37_79B9;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f64 / 16777216.0
        };
        let extent = (n as f64).cbrt() * 2.0;
        (0..n)
            .map(|_| {
                let c = Vec3::new(next() * extent, next() * extent, next() * extent);
                let mat = Lambertian::new(SolidColor::new(1., 1., 1.).into(), None);
                Triangle::new(
                    c,
                    c + Vec3::new(0.6, 0.1, 0.2),
                    c + Vec3::new(0.2, 0.7, -0.1),
                    mat.into(),
                    &NopTransformer(),
                )
                .into()
            })
            .collect()
    }

    /// The depth the assert trusts has to be the depth the tree actually has --
    /// the splice in `build_parallel` is where that is easiest to get wrong, so
    /// the largest size here is over `PARALLEL_CUTOFF`.
    #[test]
    fn depth_matches_the_built_tree() {
        for n in [1u32, 4, 5, 100, 9000] {
            let bvh = Bvh::new(cloud(n));
            assert_eq!(measured_depth(&bvh, 0), bvh.depth(), "n = {}", n);
            assert!(bvh.depth() <= MAX_TRAVERSAL_DEPTH, "n = {}", n);
        }

        assert_eq!(0, Bvh::new(Vec::new()).depth());
    }

    /// The constant is only worth anything while both sides agree on it.
    #[test]
    fn max_traversal_depth_matches_the_shader() {
        let wgsl = include_str!("../renderer/ray_trace.wgsl");
        let literal = format!("const MAX_TRAVERSAL_DEPTH = {}u;", MAX_TRAVERSAL_DEPTH);
        assert!(
            wgsl.contains(&literal),
            "ray_trace.wgsl no longer declares `{}`; the stack and the assert have drifted",
            literal
        );
        assert!(
            wgsl.contains("var<private> traversal_stack: array<u32, MAX_TRAVERSAL_DEPTH>;"),
            "the traversal stack is no longer sized by MAX_TRAVERSAL_DEPTH"
        );
    }

    /// Node count, depth, mean leaf size and the SAH cost of the whole tree.
    ///
    /// The cost is what the build's greedy heuristic approximates one level at
    /// a time: every internal node charges [`TRAVERSAL_COST`] for its box and
    /// every leaf one intersection per primitive, each weighted by its surface
    /// area over the root's. It is the one number that says a change to `split`
    /// built a better tree rather than a different one, and it says so
    /// deterministically on the CPU in a second.
    #[derive(Default)]
    struct TreeStats {
        nodes: usize,
        leaves: usize,
        leaf_prims: usize,
        depth: u32,
        /// Already divided by the root area, so it is comparable across scenes.
        cost: f64,
    }

    impl Display for TreeStats {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "nodes {:>7}  depth {:>2}  leaves {:>7}  mean leaf {:.2}  sah cost {:.2}",
                self.nodes,
                self.depth,
                self.leaves,
                self.leaf_prims as f64 / self.leaves.max(1) as f64,
                self.cost
            )
        }
    }

    fn tree_stats(bvh: &Bvh) -> TreeStats {
        let mut s = TreeStats::default();
        if bvh.nodes().is_empty() {
            return s;
        }
        let root_area = Aabb32::from(&bvh.data.b_box).half_area();
        let inv_root = if root_area > 0. {
            1. / root_area as f64
        } else {
            0.
        };
        accumulate(bvh.nodes(), 0, root_area, 1, inv_root, &mut s);
        s
    }

    fn accumulate(
        nodes: &[BvhNode],
        idx: u32,
        area: f32,
        depth: u32,
        inv_root: f64,
        s: &mut TreeStats,
    ) {
        let node = &nodes[idx as usize];
        s.nodes += 1;
        s.depth = s.depth.max(depth);
        s.cost += TRAVERSAL_COST as f64 * area as f64 * inv_root;

        for (meta, b_box) in [
            (node.left_meta, &node.left_box),
            (node.right_meta, &node.right_box),
        ] {
            let area = Aabb32::from(b_box).half_area();
            if meta & LEAF_FLAG == 0 {
                accumulate(nodes, meta, area, depth + 1, inv_root, s);
                continue;
            }
            let count = ((meta >> LEAF_COUNT_SHIFT) & LEAF_COUNT_MASK) as usize;
            // The empty child of a single-leaf root intersects nothing.
            if count == 0 {
                continue;
            }
            s.leaves += 1;
            s.leaf_prims += count;
            s.cost += count as f64 * area as f64 * inv_root;
        }
    }

    /// Diagnostic, not a gate.
    /// `cargo test bvh_tree_quality -- --ignored --nocapture`
    ///
    /// The cloud is the same distribution `bvh_build` benches. The spider mesh
    /// is here because clustered real geometry is where a split heuristic and a
    /// leaf rule behave differently from a uniform cloud -- it is 1368
    /// triangles in 19 shells, and its cost moves when the cloud's barely does.
    #[test]
    #[ignore]
    fn bvh_tree_quality() {
        for n in [1_000u32, 10_000, 100_000, 1_000_000] {
            let bvh = Bvh::new(cloud(n));
            println!("cloud {:>7}  {}", n, tree_stats(&bvh));
        }

        let spider = Obj::new("resources/spider/", "spider.obj")
            .load(&NopTransformer(), None)
            .unwrap();
        println!(
            "spider {:>6}  {}",
            spider.prims().len(),
            tree_stats(&spider)
        );
    }

    /// A triangle's corners as bits, which identify it across trees.
    fn key(p: &Hittables) -> [u64; 9] {
        let Hittables::Triangle(t) = p else {
            unreachable!()
        };
        let (v1, v2) = (t.v0 + t.v0v1, t.v0 + t.v0v2);
        [t.v0, v1, v2]
            .map(|v| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()])
            .concat()
            .try_into()
            .unwrap()
    }

    /// The primitives of each leaf, as sets of keys.
    fn leaf_sets(bvh: &Bvh) -> Vec<Vec<[u64; 9]>> {
        let mut leaves = Vec::new();
        for node in bvh.nodes() {
            for meta in [node.left_meta, node.right_meta] {
                if let Child::Leaf { offset, count } = child_of(meta)
                    && count > 0
                {
                    let mut leaf: Vec<[u64; 9]> = bvh.prims()
                        [offset as usize..(offset + count) as usize]
                        .iter()
                        .map(key)
                        .collect();
                    leaf.sort();
                    leaves.push(leaf);
                }
            }
        }
        leaves.sort();
        leaves
    }

    /// Every child box holds what it points at, which is all a refit has to
    /// get right for the traversal to find every primitive.
    fn assert_boxes_contain_children(bvh: &Bvh) {
        let inside = |inner: &Aabb, outer: &Aabb| {
            let (i, o) = (Aabb32::from(inner), Aabb32::from(outer));
            (0..3).all(|a| o.min[a] <= i.min[a] && i.max[a] <= o.max[a])
        };
        for (n, node) in bvh.nodes().iter().enumerate() {
            for (meta, b) in [
                (node.left_meta, &node.left_box),
                (node.right_meta, &node.right_box),
            ] {
                let ok = match child_of(meta) {
                    Child::Leaf { offset, count } => bvh.prims()
                        [offset as usize..(offset + count) as usize]
                        .iter()
                        .all(|p| inside(p.bounding_box(), b)),
                    Child::Node(c) => {
                        let c = &bvh.nodes()[c as usize];
                        inside(&c.left_box, b)
                            && (matches!(child_of(c.right_meta), Child::Leaf { count: 0, .. })
                                || inside(&c.right_box, b))
                    }
                    Child::Subtree(s) => inside(bvh.subtrees()[s as usize].bounding_box(), b),
                };
                assert!(ok, "node {} does not contain a child", n);
            }
        }
        for s in bvh.subtrees() {
            assert_boxes_contain_children(s);
        }
    }

    fn spider() -> Bvh {
        Obj::new("resources/spider/", "spider.obj")
            .load(&NopTransformer(), None)
            .unwrap()
    }

    /// A refitted tree holds exactly the primitives a rebuild over the same
    /// transformed primitives does, and every box holds what is under it.
    #[test]
    fn transformed_keeps_every_primitive_in_a_box_that_holds_it() {
        use crate::geo::transformation::{RotationX, Transformations, Translation};
        let transformation = Arc::new(Transformations::new(vec![
            Box::new(RotationX::new(37.)),
            Box::new(Translation::new(Vec3::new(3., -1., 2.))),
        ]));
        let mesh = spider();
        let moved = mesh.transformed(transformation.clone());
        let rebuilt = Bvh::new(
            mesh.prims()
                .iter()
                .map(|p| p.transformed(&transformation))
                .collect(),
        );

        let sorted = |b: &Bvh| {
            let mut keys: Vec<[u64; 9]> = b.prims().iter().map(key).collect();
            keys.sort();
            keys
        };
        assert_eq!(sorted(&rebuilt), sorted(&moved));
        assert_eq!(mesh.nodes().len(), moved.nodes().len());
        assert_eq!(mesh.depth(), moved.depth());
        assert_boxes_contain_children(&moved);
    }

    /// Unmoved, a refit is the tree it was built as, to the bit.
    #[test]
    fn transformed_by_nothing_refits_to_the_same_boxes() {
        let mesh = spider();
        let same = mesh.transformed(NopTransformer());
        for (a, b) in mesh.nodes().iter().zip(same.nodes()) {
            for (x, y) in [(&a.left_box, &b.left_box), (&a.right_box, &b.right_box)] {
                assert_eq!(
                    [x.x.min, x.x.max, x.y.min, x.y.max, x.z.min, x.z.max],
                    [y.x.min, y.x.max, y.y.min, y.y.max, y.z.min, y.z.max]
                );
            }
        }
    }

    /// The transformation is absolute: a second transform starts from the
    /// geometry the first did, not from its result.
    #[test]
    fn transformed_is_absolute_not_a_step() {
        use crate::geo::transformation::{RotationY, Translation};
        let mesh = spider();
        let t = Arc::new(Translation::new(Vec3::new(0.1, 0.2, 0.3)));
        let direct = mesh.transformed(t.clone());
        let stepped = mesh.transformed(RotationY::new(10.)).transformed(t);
        let keys = |b: &Bvh| b.prims().iter().map(key).collect::<Vec<_>>();
        assert_eq!(keys(&direct), keys(&stepped));
        assert!(Weak::ptr_eq(&direct.topology(), &mesh.downgrade()));
        assert!(Weak::ptr_eq(&stepped.topology(), &mesh.downgrade()));
    }

    /// A translation or a uniform scale moves every centroid alike, which the
    /// SAH cannot see, so the refit is the tree a rebuild gives -- up to
    /// rounding. Binning takes `(c - c_min) * scale` in f32, and a coordinate
    /// that rounds the other way lands a primitive in the next bucket: on the
    /// spider two of 455 leaves differ at (4, -8, 2), none at the other
    /// offsets, and the SAH cost agrees to nine digits either way.
    #[test]
    fn a_translation_or_scale_refits_to_the_tree_a_rebuild_gives() {
        use crate::geo::transformation::{Scale, Transformer, Translation};
        let mesh = spider();
        let transformations: [Arc<dyn Transformer>; 4] = [
            Arc::new(Translation::new(Vec3::new(4., -8., 2.))),
            Arc::new(Translation::new(Vec3::new(0.37, 0.11, -0.5))),
            Arc::new(Translation::new(Vec3::new(1000., 0., 0.))),
            Arc::new(Scale::new(2.)),
        ];
        for t in &transformations {
            let moved = mesh.transformed(t.clone());
            let rebuilt = Bvh::new(
                mesh.prims()
                    .iter()
                    .map(|p| p.transformed(t.as_ref()))
                    .collect(),
            );
            let (a, b) = (leaf_sets(&rebuilt), leaf_sets(&moved));
            let common = a.iter().filter(|l| b.contains(l)).count();
            assert!(
                common * 100 >= a.len() * 99,
                "only {} of {} leaves are the rebuild's",
                common,
                a.len()
            );
            let (refit, rebuild) = (tree_stats(&moved).cost, tree_stats(&rebuilt).cost);
            assert!(
                (refit / rebuild - 1.).abs() < 1e-6,
                "refit SAH {} against a rebuild's {}",
                refit,
                rebuild
            );
        }
    }

    /// A rebuilt tree holds the same primitives in a tree built for where they
    /// are, which is tighter than the refit after a rotation, and a later
    /// transform still starts from the geometry the refit was baked from.
    #[test]
    fn rebuilt_is_tighter_and_still_bakes_from_the_source() {
        use crate::geo::transformation::{RotationY, Translation};
        let mesh = spider();
        let refit = mesh.transformed(RotationY::new(40.));
        let rebuilt = refit.rebuilt();

        let sorted = |b: &Bvh| {
            let mut keys: Vec<[u64; 9]> = b.prims().iter().map(key).collect();
            keys.sort();
            keys
        };
        assert_eq!(sorted(&refit), sorted(&rebuilt));
        assert!(tree_stats(&rebuilt).cost < tree_stats(&refit).cost);
        assert!(
            !Weak::ptr_eq(&rebuilt.topology(), &refit.topology()),
            "a rebuilt tree must not be written over the bake it came from"
        );
        assert_boxes_contain_children(&rebuilt);

        let t = Arc::new(Translation::new(Vec3::new(1., 0., 0.)));
        let keys = |b: &Bvh| b.prims().iter().map(key).collect::<Vec<_>>();
        assert_eq!(
            keys(&mesh.transformed(t.clone())),
            keys(&rebuilt.transformed(t))
        );
    }

    /// A nested tree of at least `SUBTREE_MIN_PRIMS` is kept as a subtree,
    /// anything smaller is dissolved into the tree it is nested in.
    #[test]
    fn only_a_large_nested_tree_is_kept_apart() {
        let large = Bvh::new(cloud(SUBTREE_MIN_PRIMS as u32));
        let small = Bvh::new(cloud(100));
        let world = Bvh::new(vec![large.clone().into(), small.into(), tagged(7)]);

        assert_eq!(1, world.subtrees().len());
        assert!(world.subtrees()[0].is(&large.downgrade()));
        assert_eq!(101, world.prims().len());
        assert_eq!(SUBTREE_MIN_PRIMS + 101, world.primitive_count());
        assert_eq!(measured_depth(&world, 0), world.depth());
        assert!(world.depth() > large.depth());
        assert_boxes_contain_children(&world);
    }

    /// Subtrees nested in subtrees, each level one node over the last: every
    /// level adds one to the composed depth until it no longer fits the GPU
    /// stack, and then the deepest subtree has to go back into the tree above
    /// it rather than panic. Every primitive is kept.
    #[test]
    fn a_composed_tree_too_deep_takes_its_deepest_subtree_back_in() {
        let base = Bvh::new(cloud(SUBTREE_MIN_PRIMS as u32));
        let levels = MAX_TRAVERSAL_DEPTH - base.depth() + 3;
        let mut tree = base.clone();
        for i in 0..levels {
            tree = Bvh::new(vec![tree.into(), tagged(i)]);
            assert!(tree.depth() <= MAX_TRAVERSAL_DEPTH, "level {}", i);
            assert_eq!(measured_depth(&tree, 0), tree.depth(), "level {}", i);
        }

        let mut nested = 0;
        let mut t = &tree;
        while let [inner] = t.subtrees() {
            nested += 1;
            t = inner;
        }
        assert!(
            nested < levels,
            "every level is still its own subtree, so none was ever too deep"
        );
        assert_eq!(MAX_TRAVERSAL_DEPTH, tree.depth());
        assert_eq!(SUBTREE_MIN_PRIMS + levels as usize, tree.primitive_count());
    }

    /// Diagnostic, not a gate: what a refit costs in tree quality against a
    /// rebuild, rotated off an axis. A translation or a uniform scale costs
    /// nothing; a rotation loosens every box.
    /// `cargo test refit_against_rebuild -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn refit_against_rebuild() {
        use crate::geo::transformation::{RotationX, RotationY, Transformer};
        let mut meshes = vec![("spider", spider())];
        for (name, dir, file) in [
            ("sponza", "../sponza/", "sponza.obj"),
            ("dragon", "../", "xyzrgb_dragon.obj"),
        ] {
            if let Ok(mesh) = Obj::new(dir, file).load(&NopTransformer(), None) {
                meshes.push((name, mesh));
            }
        }
        for (name, mesh) in &meshes {
            for degrees in [15., 30., 45.] {
                let rotations: [(&str, Arc<dyn Transformer>); 2] = [
                    ("x", Arc::new(RotationX::new(degrees))),
                    ("y", Arc::new(RotationY::new(degrees))),
                ];
                for (axis, rotation) in rotations {
                    let refit = mesh.transformed(rotation.clone());
                    let rebuilt = Bvh::new(
                        mesh.prims()
                            .iter()
                            .map(|p| p.transformed(rotation.as_ref()))
                            .collect(),
                    );
                    let (r, b) = (tree_stats(&refit), tree_stats(&rebuilt));
                    println!(
                        "{:<7} {:>2} deg about {}   refit {:8.2}   rebuild {:8.2}   {:+.1}%",
                        name,
                        degrees,
                        axis,
                        r.cost,
                        b.cost,
                        (r.cost / b.cost - 1.) * 100.
                    );
                }
            }
        }
    }

    /// A unit-extent triangle in the plane `z`, at `x` along the row.
    fn row_tri(x: f64, z: f64) -> Hittables {
        let mat = Lambertian::new(SolidColor::new(1., 1., 1.).into(), None);
        Triangle::new(
            Vec3::new(x, 0., z),
            Vec3::new(x + 0.1, 0., z),
            Vec3::new(x, 0.1, z),
            mat.into(),
            &NopTransformer(),
        )
        .into()
    }

    /// A shape where the widest-centroid-axis proxy is wrong: two thin rows of
    /// triangles spread along x but separated along z. Splitting on x (extent 7
    /// against z's 4) cuts both rows and leaves both children spanning the z
    /// gap, at a swept cost of ~243; splitting on z gives two flat plates at
    /// ~11. Fails if `split` goes back to the widest axis alone.
    #[test]
    fn split_picks_the_cheapest_axis_not_the_widest() {
        let prims: Vec<Hittables> = (0..8)
            .flat_map(|i| [row_tri(i as f64, 0.), row_tri(i as f64, 4.)])
            .collect();
        let boxes: Vec<Aabb32> = prims
            .iter()
            .map(|p| Aabb32::from(p.bounding_box()))
            .collect();
        let centroids: Vec<[f32; 3]> = boxes.iter().map(|b| b.center()).collect();
        let mut indices: Vec<u32> = (0..prims.len() as u32).collect();

        let mid = split(&mut indices, &boxes, &centroids);

        let rows = |slice: &[u32]| -> Vec<f32> {
            let mut z: Vec<f32> = slice.iter().map(|&i| centroids[i as usize][2]).collect();
            z.dedup();
            z
        };
        assert_eq!(vec![0.], rows(&indices[..mid]));
        assert_eq!(vec![4.], rows(&indices[mid..]));
    }
}
