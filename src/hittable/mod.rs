//! Objects that are hittable by rays shot by the ray tracer.
//! Some of these hittable objects are containers for other objects

mod bvh;
mod quad;
mod sphere;
mod triangle;

use crate::geo::Aabb;
use crate::geo::transformation::Transformer;
pub use crate::hittable::bvh::{Bvh, MAX_PRIMITIVES, MAX_TRAVERSAL_DEPTH, SUBTREE_MIN_PRIMS};
pub(crate) use crate::hittable::bvh::{BvhData, BvhNode, Child, LEAF_FLAG, LeafPrims, child_of};
pub use crate::hittable::quad::Quad;
pub use crate::hittable::sphere::Sphere;
pub(crate) use crate::hittable::triangle::Geometry;
pub use crate::hittable::triangle::Triangle;
use crate::material::Materials;
use enum_dispatch::enum_dispatch;

/// The common trait for all objects in the ray tracing scene
/// that can be hit by rays
#[enum_dispatch]
pub trait Hittable {
    /// Create a bounding box that contains the hittable
    fn bounding_box(&self) -> &Aabb;

    /// Does this hittable contain at least one light?
    ///
    /// Asked once, by `Renderer::new`, to reject a scene that can only render
    /// black. The lights themselves are collected per primitive by the
    /// flattener, off `Material::is_light`.
    fn has_lights(&self) -> bool;
}

#[enum_dispatch(Hittable)]
#[derive(Debug, Clone)]
/// Enum of the available hittable types
pub enum Hittables {
    /// [`Hittable`] of the type [`Sphere`]
    Sphere,
    /// [`Hittable`] of the type [`Quad`]
    Quad,
    /// [`Hittable`] of the type [`Triangle`]
    Triangle,
    /// [`Hittable`] of the type [`Bvh`]
    Bvh,
}

impl Hittables {
    /// The material of a primitive; `None` for a [`Bvh`].
    pub(crate) fn material(&self) -> Option<&Materials> {
        match self {
            Hittables::Sphere(s) => Some(&s.mat),
            Hittables::Quad(q) => Some(&q.mat),
            Hittables::Triangle(t) => Some(&t.mat),
            Hittables::Bvh(_) => None,
        }
    }

    /// This primitive baked again under `transformation`, as if it had been
    /// constructed with it on top of the one it was constructed with. A tree
    /// has [`Bvh::transformed`] instead.
    pub(crate) fn transformed(&self, transformation: &dyn Transformer) -> Hittables {
        match self {
            Hittables::Sphere(s) => s.transformed(transformation).into(),
            Hittables::Quad(q) => q.transformed(transformation).into(),
            Hittables::Triangle(t) => t.transformed(transformation).into(),
            Hittables::Bvh(_) => unreachable!("a Bvh's primitives hold no Bvh"),
        }
    }
}
