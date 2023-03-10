use crate::geo::aabb::Aabb;
use crate::geo::ray::Ray;
use crate::geo::vec3::Vec3;
use crate::hittable::Hittables::TranslationType;
use crate::hittable::{Hittable, Hittables};
use crate::material::HitRecord;
use crate::util::interval::Interval;

#[derive(Clone, Debug)]
pub struct Translation {
    object: Box<Hittables>,
    offset: Vec3,
    b_box: Aabb,
}

impl Translation {
    /// Creates a hittable object that translates the given hittable by the given offset vector
    pub fn create(object: Hittables, offset: Vec3) -> Hittables {
        let object_b_box = object.bounding_box().clone();
        TranslationType(Translation {
            object: Box::new(object),
            offset,
            b_box: &object_b_box + offset,
        })
    }
}

impl Hittable for Translation {
    fn pdf_value(&self, origin: Vec3, direction: Vec3) -> f64 {
        self.object.pdf_value(origin, direction)
    }

    fn random_direction(&self, origin: Vec3) -> Vec3 {
        self.object.random_direction(origin)
    }

    fn hit(&self, r: &Ray, ray_length: &Interval) -> Option<HitRecord> {
        let offset_ray = Ray::new(r.origin - self.offset, r.direction, r.time);

        self.object
            .hit(&offset_ray, ray_length)
            .map(|rec| HitRecord {
                hit_point: rec.hit_point + self.offset,
                normal: rec.normal,
                material: rec.material,
                ray_length: rec.ray_length,
                uv: rec.uv,
                front_face: rec.front_face,
            })
    }

    fn bounding_box(&self) -> &Aabb {
        &self.b_box
    }

    fn is_light(&self) -> bool {
        self.object.is_light()
    }
}
