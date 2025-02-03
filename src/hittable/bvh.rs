use std::fmt;
use std::fmt::Display;
use derive_more::Display;

use crate::geo::Aabb;
use crate::geo::Ray;
use crate::hittable::{Hittable, Hittables};
use crate::material::RayHit;
use crate::util::interval::Interval;

/// Bounding Volume Hierarchy
#[derive(Display, Debug)]
#[display("{{\"left\": {}, \"right\": {}}}", left, right)]
pub struct Bvh {
    left: Box<BvhItem>,
    right: Box<BvhItem>,
    b_box: Aabb,
}

#[derive(Debug, Clone)]
enum BvhItem {
    Node(Bvh),
    Leaf(Box<Hittables>),
    None,
}

impl Display for BvhItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BvhItem::Node(b) => write!(f, "{}", b),
            BvhItem::Leaf(t) => write!(f, "{}", t.bounding_box().center()),
            BvhItem::None => write!(f, "<empty>"),
        }
    }
}

impl BvhItem {
    fn hit(&self, r: &Ray, ray_length: &Interval) -> Option<RayHit> {
        match self {
            BvhItem::Node(i) => i.hit(r, ray_length),
            BvhItem::Leaf(i) => i.hit(r, ray_length),
            BvhItem::None => None,
        }
    }

    fn get_lights(&self) -> Vec<Hittables> {
        match self {
            BvhItem::Node(b) => b.get_lights(),
            BvhItem::Leaf(l) => l.get_lights(),
            BvhItem::None => vec![],
        }
    }
}

impl Bvh {
    #![allow(clippy::new_ret_no_self)]
    /// Creates a new hittable object from the given hittable list
    /// The bounding Volume Hierarchy sorts the hittables in a binary tree
    /// where each node has a bounding box.
    /// This is to optimize the ray intersection search when having many hittable objects.
    pub fn new(list: Vec<Hittables>) -> Hittables {
        if list.is_empty() {
            Hittables::from(Bvh {
                left: Box::new(BvhItem::None),
                right: Box::new(BvhItem::None),
                b_box: Default::default(),
            })
        } else {
            Hittables::from(new_bvh(list))
        }
    }
}

impl Clone for Bvh {
    fn clone(&self) -> Self {
        Bvh {
            left: self.left.clone(),
            right: self.right.clone(),
            b_box: self.b_box.clone(),
        }
    }
}

fn new_bvh(mut list: Vec<Hittables>) -> Bvh {
    let (left, right, b_box) = if list.len() == 1 {
        (
            BvhItem::Leaf(Box::new(list[0].clone())),
            BvhItem::None,
            list[0].bounding_box().clone(),
        )
    } else if list.len() == 2 {
        (
            BvhItem::Leaf(Box::new(list[0].clone())),
            BvhItem::Leaf(Box::new(list[1].clone())),
            list[0].bounding_box().combine(list[1].bounding_box()),
        )
    } else {
        let mid = sort_hittables_slice_by_most_spread_axis(list.as_mut_slice());

        let (l, r) = rayon::join(
            || new_bvh(list[..mid].to_vec()),
            || new_bvh(list[mid..].to_vec()),
        );

        let b_box = l.b_box.combine(&r.b_box);
        (BvhItem::Node(l), BvhItem::Node(r), b_box)
    };

    Bvh {
        left: Box::new(left),
        right: Box::new(right),
        b_box,
    }
}

fn sort_hittables_slice_by_most_spread_axis(list: &mut [Hittables]) -> usize {
    let (x_spread, x_center) = bounding_box_spread(list, 0);
    let (y_spread, y_center) = bounding_box_spread(list, 1);
    let (z_spread, z_center) = bounding_box_spread(list, 2);

    let mut center = if x_spread >= y_spread && x_spread >= z_spread {
        sort_hittables_by_center(list, x_center, 0)
    } else if y_spread >= x_spread && y_spread >= z_spread {
        sort_hittables_by_center(list, y_center, 1)
    } else {
        sort_hittables_by_center(list, z_center, 2)
    };

    // Could not split with objects on both sides. Just split up the middle index
    if center == 0 || center == list.len() {
        center = list.len() / 2;
    }
    center
}

fn bounding_box_spread(list: &[Hittables], axis: u8) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for hittable in list {
        let c = hittable.bounding_box().center().axis(axis);
        min = min.min(c);
        max = max.max(c);
    }
    (max - min, (min + max) * 0.5)
}

fn sort_hittables_by_center(list: &mut [Hittables], center: f64, axis: u8) -> usize {
    list.sort_unstable_by(|a, b| {
        a.bounding_box()
            .center()
            .axis(axis)
            .total_cmp(&b.bounding_box().center().axis(axis))
    });
    let mut i = 0;
    for t in list {
        if t.bounding_box().center().axis(axis) >= center {
            return i;
        }
        i += 1;
    }
    i
}

impl Hittable for Bvh {
    fn hit(&self, r: &Ray, ray_length: &Interval) -> Option<RayHit> {
        if !self.b_box.hit(r) {
            return None;
        }

        match self.left.hit(r, ray_length) {
            None => self.right.hit(r, ray_length),
            Some(left_rec) => {
                let new_ray_length = Interval::new(ray_length.min, left_rec.ray_length);
                match self.right.hit(r, &new_ray_length) {
                    Some(right_rec) => Some(right_rec),
                    None => Some(left_rec),
                }
            }
        }
    }

    fn bounding_box(&self) -> &Aabb {
        &self.b_box
    }

    fn get_lights(&self) -> Vec<Hittables> {
        let mut ret = Vec::new();

        ret.append(&mut self.left.get_lights());
        ret.append(&mut self.right.get_lights());

        ret
    }
}
