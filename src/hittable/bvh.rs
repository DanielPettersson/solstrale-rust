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

use std::fmt;
use std::fmt::Display;

use crate::geo::Aabb;
use crate::hittable::{Hittable, Hittables};

/// Maximum primitives in a single leaf.
pub(crate) const MAX_LEAF_PRIMS: usize = 4;

/// Buckets used by the SAH sweep.
const SAH_BUCKETS: usize = 16;

/// Below this many primitives a subtree is built serially -- forking a rayon
/// task per level costs more than it saves once the subtree is small.
const PARALLEL_CUTOFF: usize = 8192;

/// Set on a child meta word to mark it as an inline leaf rather than a node index.
pub(crate) const LEAF_FLAG: u32 = 0x8000_0000;
/// Leaf primitive count, bits 30..24.
const LEAF_COUNT_SHIFT: u32 = 24;
/// Mask for the leaf primitive count once shifted down.
const LEAF_COUNT_MASK: u32 = 0x7F;
/// Leaf primitive offset, bits 23..0.
const LEAF_OFFSET_MASK: u32 = 0x00FF_FFFF;

/// Largest scene the leaf encoding can address.
pub const MAX_PRIMITIVES: usize = LEAF_OFFSET_MASK as usize;

/// A leaf's count has to fit the seven bits the encoding gives it.
const _: () = assert!(MAX_LEAF_PRIMS as u32 <= LEAF_COUNT_MASK);

/// Entries in the GPU traversal stack (`traversal_stack` in `ray_trace.wgsl`).
///
/// Traversal only ever pushes the farther child, so a tree of depth `d` --
/// counting levels of internal nodes -- needs at most `d - 1` entries: the
/// deepest internal node has two leaves and pushes nothing. Asserting
/// `d <= MAX_TRAVERSAL_DEPTH` therefore leaves one slot spare.
///
/// The bound has to be checked here because overflowing it on the GPU is
/// silent: WGSL's bounds-checking policy clamps the out-of-range store, the
/// deferred subtree is never visited, and geometry disappears from the image
/// with no error at all. Nothing built a balanced tree, so nothing made this
/// true by construction -- `split` can legitimately return a 1/N-1 split.
///
/// 32 is kept because the measured excess over a balanced tree is what scales,
/// not the depth itself -- but the slack is thinner than the synthetic cloud
/// suggests, and shrinking it is not free. The `bvh_build` cloud reaches
/// 14 / 17 / 21 at 10k / 100k / 1M, two to three levels over balanced.
/// Clustered real geometry is what actually sets the bound: measured with
/// `bvh_tree_quality`, sponza (262k triangles) reaches **27**, conference 25
/// and a 1M-triangle gallery scene 24, against a balanced 18-20. Four or five
/// levels is all the margin there is, so a change that deepens the tree has to
/// be checked against real meshes and not against the cloud.
pub const MAX_TRAVERSAL_DEPTH: u32 = 32;

/// Packs an inline leaf: `count` primitives starting at `offset`.
fn leaf_meta(offset: u32, count: usize) -> u32 {
    debug_assert!(count <= MAX_LEAF_PRIMS);
    debug_assert!(offset as usize <= MAX_PRIMITIVES);
    LEAF_FLAG | ((count as u32) << LEAF_COUNT_SHIFT) | (offset & LEAF_OFFSET_MASK)
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
#[derive(Debug, Clone)]
pub struct Bvh {
    /// Flat node array; node 0 is the root. Empty when the tree holds no primitives.
    pub(crate) nodes: Vec<BvhNode>,
    /// Primitives in leaf order, so a leaf is a contiguous range.
    pub(crate) prims: Vec<Hittables>,
    b_box: Aabb,
    /// Levels of internal nodes on the longest root-to-leaf path; 0 when empty.
    depth: u32,
}

impl Display for Bvh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{\"nodes\": {}, \"primitives\": {}, \"depth\": {}}}",
            self.nodes.len(),
            self.prims.len(),
            self.depth
        )
    }
}

impl Bvh {
    /// Creates a new hittable object from the given hittable list.
    ///
    /// Nested [`Bvh`]s in `list` are expanded into their primitives and
    /// included in this build, so the result is a single tree over every
    /// primitive rather than a tree of trees. Nothing is attached to a nested
    /// `Bvh` -- transformations are baked into vertices at construction -- so
    /// there is nothing to preserve by keeping it separate, and one global tree
    /// is strictly better than a parent that treats a whole subtree as one box.
    pub fn new(list: Vec<Hittables>) -> Bvh {
        let mut prims = if list.iter().any(|h| matches!(h, Hittables::Bvh(_))) {
            let mut v = Vec::with_capacity(list.len());
            collect_primitives(list, &mut v);
            v
        } else {
            // The common case: a loader hands us a flat list of primitives. Moving all of
            // them into a second vector just to discover there was nothing to flatten
            // costs a full copy -- ~78 MB for a 250k-triangle mesh. The discriminant scan
            // that avoids it walks memory the very next statement walks anyway.
            list
        };

        if prims.is_empty() {
            return Bvh {
                nodes: Vec::new(),
                prims: Vec::new(),
                b_box: Default::default(),
                depth: 0,
            };
        }

        assert!(
            prims.len() <= MAX_PRIMITIVES,
            "BVH supports at most {} primitives, got {}",
            MAX_PRIMITIVES,
            prims.len()
        );

        // Bounding boxes and centroids are derived once here. The old build
        // re-derived `bounding_box().center()` inside the sort comparator, so
        // it recomputed them O(N log N) times per level.
        let boxes: Vec<Aabb32> = prims
            .iter()
            .map(|p| Aabb32::from(p.bounding_box()))
            .collect();
        let centroids: Vec<[f32; 3]> = boxes.iter().map(|b| b.center()).collect();

        let mut b_box = prims[0].bounding_box().clone();
        for p in prims.iter().skip(1) {
            b_box = b_box.combine(p.bounding_box());
        }

        let mut indices: Vec<u32> = (0..prims.len() as u32).collect();

        let (nodes, depth) = if prims.len() <= MAX_LEAF_PRIMS {
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
        } else {
            let mut nodes = Vec::with_capacity(prims.len() / MAX_LEAF_PRIMS);
            let depth = build_parallel(&mut nodes, &mut indices, 0, &boxes, &centroids);
            (nodes, depth)
        };

        assert!(
            depth <= MAX_TRAVERSAL_DEPTH,
            "BVH depth {} exceeds the {} levels the GPU traversal stack can \
             hold, which would silently drop geometry from the image",
            depth,
            MAX_TRAVERSAL_DEPTH
        );

        permute_in_place(&mut prims, &indices);

        Bvh {
            nodes,
            prims,
            b_box,
            depth,
        }
    }

    /// Levels of internal nodes on the longest root-to-leaf path.
    ///
    /// This is what [`MAX_TRAVERSAL_DEPTH`] bounds; see it for the exact
    /// relation to the GPU stack.
    pub fn depth(&self) -> u32 {
        self.depth
    }
}

/// Expands nested BVHs so a single tree covers every primitive.
fn collect_primitives(list: Vec<Hittables>, out: &mut Vec<Hittables>) {
    for h in list {
        match h {
            Hittables::Bvh(bvh) => collect_primitives(bvh.prims, out),
            other => out.push(other),
        }
    }
}

/// Reorders `prims` so that `prims[i]` ends up holding what `prims[indices[i]]` held.
///
/// This allocates nothing and moves every primitive once. It replaced a gather through a
/// staging `Vec<Option<Hittables>>`, which allocated a second full-size array and moved
/// everything twice. Building a 1M-primitive BVH peaked at 736.7 MB RSS that way and
/// peaks at 433.4 MB now -- 303 MB, almost exactly the one redundant copy of the
/// primitive array.
///
/// The trade is deliberate and is not a win at every size: all the work here is
/// random-access swaps, where the gather at least wrote sequentially, so this only runs
/// faster while the primitive array still fits in last-level cache. Measured on
/// `bvh_build` (a random spatial cloud, the worst case for locality) on a 96 MB-L3 part:
///
/// | primitives | array | gather | in place |
/// |---|---|---|---|
/// | 100k | 31 MB | 28.5 ms | **18.3 ms** |
/// | 250k | 78 MB | 75.3 ms | **70.2 ms** |
/// | 500k | 156 MB | **158.7 ms** | 161.4 ms |
/// | 1M | 312 MB | **376.3 ms** | 421.7 ms |
///
/// So past roughly 400k primitives on that part -- sooner on one with a smaller
/// last-level cache -- this costs ~11% of build time. That is the deliberate trade: build
/// time is paid once at load, where the peak allocation is what decides whether a large
/// scene fits at all.
///
/// Note the direction. `indices[i]` is the *source* slot for destination `i`. Once slot
/// `i` has been written, a later `indices[j]` still pointing at it is stale, so the chase
/// follows `indices` forward until it lands on a slot that has not been overwritten yet
/// (`src >= i`). Swapping `prims` and `indices` together until `indices[k] == k` looks
/// equivalent and is not -- it applies the inverse permutation, which the golden-image
/// tests would happily accept at their 0.95 RMS threshold.
fn permute_in_place(prims: &mut [Hittables], indices: &[u32]) {
    for i in 0..prims.len() {
        let mut src = indices[i] as usize;
        while src < i {
            src = indices[src] as usize;
        }
        prims.swap(i, src);
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
    boxes: &[Aabb32],
    centroids: &[[f32; 3]],
) -> u32 {
    if indices.len() < PARALLEL_CUTOFF {
        return build_serial(nodes, indices, offset, boxes, centroids).1;
    }

    let mid = split(indices, boxes, centroids);
    let (l_idx, r_idx) = indices.split_at_mut(mid);
    let r_offset = offset + mid as u32;

    let l_box = union_of(l_idx, boxes);
    let r_box = union_of(r_idx, boxes);
    let l_leaf = l_idx.len() <= MAX_LEAF_PRIMS;
    let r_leaf = r_idx.len() <= MAX_LEAF_PRIMS;

    let ((l_sub, l_depth), (r_sub, r_depth)) = rayon::join(
        || {
            let mut v = Vec::new();
            let d = if l_leaf {
                0
            } else {
                build_parallel(&mut v, l_idx, offset, boxes, centroids)
            };
            (v, d)
        },
        || {
            let mut v = Vec::new();
            let d = if r_leaf {
                0
            } else {
                build_parallel(&mut v, r_idx, r_offset, boxes, centroids)
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
        left_meta: if l_leaf {
            leaf_meta(offset, l_idx.len())
        } else {
            base_l
        },
        right_meta: if r_leaf {
            leaf_meta(r_offset, r_idx.len())
        } else {
            base_r
        },
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
/// The recursion here is as deep as the tree, so the depth the caller asserts
/// against [`MAX_TRAVERSAL_DEPTH`] also bounds this stack -- but only after the
/// fact: a distribution pathological enough to overflow the native stack does
/// so before there is a depth to check.
fn build_serial(
    nodes: &mut Vec<BvhNode>,
    indices: &mut [u32],
    offset: u32,
    boxes: &[Aabb32],
    centroids: &[[f32; 3]],
) -> (u32, u32) {
    let my = nodes.len() as u32;
    nodes.push(BvhNode {
        left_box: Aabb::default(),
        right_box: Aabb::default(),
        left_meta: 0,
        right_meta: 0,
    });

    let mid = split(indices, boxes, centroids);
    let (l_idx, r_idx) = indices.split_at_mut(mid);
    let r_offset = offset + mid as u32;

    let left_box = union_of(l_idx, boxes);
    let right_box = union_of(r_idx, boxes);

    let (left_meta, left_depth) = if l_idx.len() <= MAX_LEAF_PRIMS {
        (leaf_meta(offset, l_idx.len()), 0)
    } else {
        build_serial(nodes, l_idx, offset, boxes, centroids)
    };
    let (right_meta, right_depth) = if r_idx.len() <= MAX_LEAF_PRIMS {
        (leaf_meta(r_offset, r_idx.len()), 0)
    } else {
        build_serial(nodes, r_idx, r_offset, boxes, centroids)
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
/// Sweeping all three rather than only the widest is the textbook improvement:
/// the widest *centroid* axis is a proxy for the axis that best separates the
/// primitives' boxes, and a plate or a shell spread along a wide axis is where
/// the proxy is wrong. The three axes are binned in one pass rather than three
/// because that pass is a gather through `centroids` and `boxes`, which is what
/// the build's time actually goes on -- the two extra bucket updates per
/// primitive run on data already in registers.
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
        &self.b_box
    }

    fn get_lights(&self) -> Vec<Hittables> {
        self.prims.iter().flat_map(|p| p.get_lights()).collect()
    }

    fn has_lights(&self) -> bool {
        self.prims.iter().any(|p| p.has_lights())
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
    fn measured_depth(nodes: &[BvhNode], idx: u32) -> u32 {
        let child = |meta: u32| {
            if meta & LEAF_FLAG != 0 {
                0
            } else {
                measured_depth(nodes, meta)
            }
        };
        1 + child(nodes[idx as usize].left_meta).max(child(nodes[idx as usize].right_meta))
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
            assert_eq!(measured_depth(&bvh.nodes, 0), bvh.depth(), "n = {}", n);
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
    /// The cost is the quantity the build's greedy heuristic approximates one
    /// level at a time: every internal node charges [`TRAVERSAL_COST`] for its
    /// box and every leaf charges one intersection per primitive, each weighted
    /// by the conditional probability that a ray hitting the root box also hits
    /// that node's -- its surface area over the root's. It is the one number
    /// that says a change to `split` built a *better* tree rather than a
    /// different one, and it says so deterministically, on the CPU, in a
    /// second. Read it before spending minutes on a GPU traversal benchmark.
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
        if bvh.nodes.is_empty() {
            return s;
        }
        let root_area = Aabb32::from(&bvh.b_box).half_area();
        let inv_root = if root_area > 0. {
            1. / root_area as f64
        } else {
            0.
        };
        accumulate(&bvh.nodes, 0, root_area, 1, inv_root, &mut s);
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
        println!("spider {:>6}  {}", spider.prims.len(), tree_stats(&spider));
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

    /// The widest centroid axis is only a proxy for the axis that separates the
    /// boxes, and this is a shape where the proxy is wrong: two thin rows of
    /// triangles spread along x but separated along z. Splitting on x -- the
    /// widest centroid axis, extent 7 against z's 4 -- cuts both rows and
    /// leaves both children spanning the whole z gap, at a swept cost of ~243.
    /// Splitting on z separates the rows into two flat plates, at ~11. Only a
    /// sweep that tries all three axes finds it, so this is the test that fails
    /// if `split` goes back to the widest axis alone.
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
