//! Objects that are hittable by rays shot by the ray tracer.
//! Some of these hittable objects are containers for other objects

mod bvh;
mod quad;
mod sphere;
mod triangle;

use crate::geo::Aabb;
pub use crate::hittable::bvh::Bvh;
pub(crate) use crate::hittable::bvh::LEAF_FLAG;
pub use crate::hittable::quad::Quad;
pub use crate::hittable::sphere::Sphere;
pub use crate::hittable::triangle::Triangle;
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
