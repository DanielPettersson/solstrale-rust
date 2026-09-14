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
/// Leaf primitive offset, bits 23..0.
const LEAF_OFFSET_MASK: u32 = 0x00FF_FFFF;

/// Largest scene the leaf encoding can address.
pub const MAX_PRIMITIVES: usize = LEAF_OFFSET_MASK as usize;

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
}

impl Display for Bvh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{\"nodes\": {}, \"primitives\": {}}}",
            self.nodes.len(),
            self.prims.len()
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
        let boxes: Vec<Aabb32> = prims.iter().map(|p| Aabb32::from(p.bounding_box())).collect();
        let centroids: Vec<[f32; 3]> = boxes.iter().map(|b| b.center()).collect();

        let mut b_box = prims[0].bounding_box().clone();
        for p in prims.iter().skip(1) {
            b_box = b_box.combine(p.bounding_box());
        }

        let mut indices: Vec<u32> = (0..prims.len() as u32).collect();

        let nodes = if prims.len() <= MAX_LEAF_PRIMS {
            // Everything fits in one leaf. Emit a root whose left child is that
            // leaf and whose right child is an empty leaf: a zero-count leaf
            // intersects nothing, so its box never needs to fail the slab test.
            vec![BvhNode {
                left_box: b_box.clone(),
                right_box: Aabb::default(),
                left_meta: leaf_meta(0, prims.len()),
                right_meta: leaf_meta(0, 0),
            }]
        } else {
            let mut nodes = Vec::with_capacity(prims.len() / MAX_LEAF_PRIMS);
            build_parallel(&mut nodes, &mut indices, 0, &boxes, &centroids);
            nodes
        };

        // Permute primitives into leaf order by moving, not cloning.
        let mut slots: Vec<Option<Hittables>> = prims.into_iter().map(Some).collect();
        let prims: Vec<Hittables> = indices
            .iter()
            .map(|&i| slots[i as usize].take().expect("index visited twice"))
            .collect();

        Bvh {
            nodes,
            prims,
            b_box,
        }
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

/// Builds a subtree, forking the two halves onto rayon while they are large
/// enough to be worth it. Subtree nodes are built into their own vectors and
/// spliced, which only happens for the handful of levels above the cutoff.
fn build_parallel(
    nodes: &mut Vec<BvhNode>,
    indices: &mut [u32],
    offset: u32,
    boxes: &[Aabb32],
    centroids: &[[f32; 3]],
) {
    if indices.len() < PARALLEL_CUTOFF {
        build_serial(nodes, indices, offset, boxes, centroids);
        return;
    }

    let mid = split(indices, boxes, centroids);
    let (l_idx, r_idx) = indices.split_at_mut(mid);
    let r_offset = offset + mid as u32;

    let l_box = union_of(l_idx, boxes);
    let r_box = union_of(r_idx, boxes);
    let l_leaf = l_idx.len() <= MAX_LEAF_PRIMS;
    let r_leaf = r_idx.len() <= MAX_LEAF_PRIMS;

    let (l_sub, r_sub) = rayon::join(
        || {
            let mut v = Vec::new();
            if !l_leaf {
                build_parallel(&mut v, l_idx, offset, boxes, centroids);
            }
            v
        },
        || {
            let mut v = Vec::new();
            if !r_leaf {
                build_parallel(&mut v, r_idx, r_offset, boxes, centroids);
            }
            v
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
}

/// Builds a subtree directly into `nodes`, returning this subtree's root index.
fn build_serial(
    nodes: &mut Vec<BvhNode>,
    indices: &mut [u32],
    offset: u32,
    boxes: &[Aabb32],
    centroids: &[[f32; 3]],
) -> u32 {
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

    let left_meta = if l_idx.len() <= MAX_LEAF_PRIMS {
        leaf_meta(offset, l_idx.len())
    } else {
        build_serial(nodes, l_idx, offset, boxes, centroids)
    };
    let right_meta = if r_idx.len() <= MAX_LEAF_PRIMS {
        leaf_meta(r_offset, r_idx.len())
    } else {
        build_serial(nodes, r_idx, r_offset, boxes, centroids)
    };

    nodes[my as usize] = BvhNode {
        left_box,
        right_box,
        left_meta,
        right_meta,
    };
    my
}

/// Chooses a split point with a binned SAH sweep over the widest centroid axis.
///
/// Returns an index into `indices`, which is partitioned in place. Always
/// returns a value in `1..indices.len()` so neither side is empty.
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

    let mut axis = 0;
    for a in 1..3 {
        if c_max[a] - c_min[a] > c_max[axis] - c_min[axis] {
            axis = a;
        }
    }
    let extent = c_max[axis] - c_min[axis];
    if extent.is_nan() || extent <= 0.0 {
        // All centroids coincide on every axis; nothing to separate spatially.
        return n / 2;
    }

    let scale = SAH_BUCKETS as f32 / extent;
    let bucket_of = |i: u32| -> usize {
        (((centroids[i as usize][axis] - c_min[axis]) * scale) as usize).min(SAH_BUCKETS - 1)
    };

    let mut bucket_box = [Aabb32::EMPTY; SAH_BUCKETS];
    let mut bucket_count = [0u32; SAH_BUCKETS];
    for &i in indices.iter() {
        let b = bucket_of(i);
        bucket_count[b] += 1;
        bucket_box[b] = bucket_box[b].union(&boxes[i as usize]);
    }

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
    let mut best_cost = f32::INFINITY;
    let mut best_k = usize::MAX;
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
            best_k = k;
        }
    }

    if best_k == usize::MAX {
        return n / 2;
    }

    // Partition in place on the same bucket assignment the sweep used, so the
    // split matches the cost that was evaluated. This replaces a full
    // `sort_unstable_by` per level -- O(N) instead of O(N log N).
    let mid = partition(indices, |i| bucket_of(i) < best_k);
    if mid == 0 || mid == n { n / 2 } else { mid }
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
            min: [round_down(a.x.min), round_down(a.y.min), round_down(a.z.min)],
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
