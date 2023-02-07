use crate::geo::aabb::Aabb;
use crate::geo::ray::Ray;
use crate::hittable::triangle::Triangle;
use crate::hittable::{Hittable, Hittables};
use crate::material::HitRecord;
use crate::util::interval::Interval;

/// Bounding Volume Hierarchy
pub struct Bvh {
    left: Box<BvhItem>,
    right: Box<BvhItem>,
    b_box: Aabb,
}

enum BvhItem {
    Node(Bvh),
    Leaf(Triangle),
}

impl Bvh {
    /// Creates a new hittable object from the given hittable list
    /// The bounding Volume Hierarchy sorts the hittables in a binary tree
    /// where each node has a bounding box.
    /// This is to optimize the ray intersection search when having many hittable objects.
    pub fn new(list: &mut [Triangle]) -> Hittables {
        if list.len() == 0 {
            panic!("Cannot create a Bvh with empty list of objects")
        }
        Hittables::Bvh(create_bvh(list))
    }

    fn left_hit(&self, r: &Ray, ray_length: &Interval) -> Option<HitRecord> {
        match self.left.as_ref() {
            BvhItem::Node(i) => i.hit(r, ray_length),
            BvhItem::Leaf(i) => i.hit(r, ray_length),
        }
    }

    fn right_hit(&self, r: &Ray, ray_length: &Interval) -> Option<HitRecord> {
        match self.right.as_ref() {
            BvhItem::Node(i) => i.hit(r, ray_length),
            BvhItem::Leaf(i) => i.hit(r, ray_length),
        }
    }
}

fn create_bvh(list: &mut [Triangle]) -> Bvh {
    let (left, right, b_box) = if list.len() == 1 {
        (
            BvhItem::Leaf(list[0].clone()),
            BvhItem::Leaf(list[0].clone()),
            list[0].bounding_box().clone(),
        )
    } else if list.len() == 2 {
        (
            BvhItem::Leaf(list[0].clone()),
            BvhItem::Leaf(list[1].clone()),
            Aabb::combine_aabbs(list[0].bounding_box(), list[1].bounding_box()),
        )
    } else {
        let mid = sort_hittables_slice_by_most_spread_axis(list);
        let left = create_bvh(&mut list[..mid]);
        let right = create_bvh(&mut list[mid..]);
        let b_box = Aabb::combine_aabbs(&left.b_box, &right.b_box);
        (BvhItem::Node(left), BvhItem::Node(right), b_box)
    };

    return Bvh {
        left: Box::new(left),
        right: Box::new(right),
        b_box,
    };
}

fn sort_hittables_slice_by_most_spread_axis(list: &mut [Triangle]) -> usize {
    let (x_spread, x_center) = bounding_box_spread(list, 0);
    let (y_spread, y_center) = bounding_box_spread(list, 1);
    let (z_spread, z_center) = bounding_box_spread(list, 2);

    let mut center = if x_spread >= y_spread && x_spread >= z_spread {
        sort_triangles_by_center(list, x_center, 0)
    } else if y_spread >= x_spread && y_spread >= z_spread {
        sort_triangles_by_center(list, y_center, 1)
    } else {
        sort_triangles_by_center(list, z_center, 2)
    };

    // Could not split with objects on both sides. Just split up the middle index
    if center == 0 || center == list.len() {
        center = list.len() / 2;
    }
    center
}

fn bounding_box_spread(list: &[Triangle], axis: u8) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for triangle in list {
        let c = triangle.center(axis);
        min = min.min(c);
        max = max.max(c);
    }
    (max - min, (min + max) * 0.5)
}

fn sort_triangles_by_center(list: &mut [Triangle], center: f64, axis: u8) -> usize {
    list.sort_unstable_by(|a, b| a.center(axis).total_cmp(&b.center(axis)));
    let mut i = 0;
    for t in list {
        if t.center(axis) >= center {
            return i;
        }
        i += 1;
    }
    i
}

impl Hittable for Bvh {
    fn hit(&self, r: &Ray, ray_length: &Interval) -> Option<HitRecord> {
        if !self.b_box.hit(r) {
            return None;
        }

        match self.left_hit(r, ray_length) {
            None => self.right_hit(r, ray_length),
            Some(rec) => {
                let new_ray_length = Interval::new(ray_length.min, rec.ray_length);
                self.right_hit(r, &new_ray_length)
            }
        }
    }

    fn bounding_box(&self) -> &Aabb {
        &self.b_box
    }

    fn is_light(&self) -> bool {
        false
    }
}