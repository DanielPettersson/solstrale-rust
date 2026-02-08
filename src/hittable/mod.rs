//! Objects that are hittable by rays shot by the ray tracer.
//! Some of these hittable objects are containers for other objects

mod bvh;
mod quad;
mod sphere;
mod triangle;

use crate::geo::Aabb;
use crate::geo::Ray;
use crate::geo::vec3::Vec3;
pub use crate::hittable::bvh::Bvh;
pub(crate) use crate::hittable::bvh::BvhItem;
pub use crate::hittable::quad::Quad;
pub use crate::hittable::sphere::Sphere;
pub use crate::hittable::triangle::Triangle;
use crate::material::RayHit;
use crate::util::interval::Interval;
use enum_dispatch::enum_dispatch;

/// The common trait for all objects in the ray tracing scene
/// that can be hit by rays
#[enum_dispatch]
pub trait Hittable {
    /// Return the pdf value for the hittable given the origin and direction of the ray that hits
    fn pdf_value(&self, _origin: Vec3, _direction: Vec3) -> f64 {
        panic!("Should not be used for materials that can not be lights")
    }

    /// Generate a random direction from the given point on the hittable
    fn random_direction(&self, _origin: Vec3) -> Vec3 {
        panic!("Should not be used for materials that can not be lights")
    }

    /// Check if the given ray hits the hittable within the interval
    fn hit(&self, r: &Ray, ray_length: &Interval) -> Option<RayHit<'_>>;

    /// Create a bounding box that contains the hittable
    fn bounding_box(&self) -> &Aabb;

    /// Is the hittable a light? Or does it contain any lights?
    fn get_lights(&self) -> Vec<Hittables>;
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
