use crate::geo::Aabb;
use crate::geo::vec3::Vec3;
use crate::hittable::{Hittable, Hittables};
use crate::material::{Material, Materials};

/// A sphere-shaped hittable object
#[derive(Debug)]
pub struct Sphere {
    pub(crate) center: Vec3,
    pub(crate) radius: f64,
    pub(crate) mat: Materials,
    b_box: Aabb,
}

impl Sphere {
    ///Creates a new sphere
    pub fn new(center: Vec3, radius: f64, mat: Materials) -> Sphere {
        let r_vec = Vec3::new(radius, radius, radius);
        let b_box = Aabb::new_from_2_points(center - r_vec, center + r_vec);

        Sphere {
            center,
            radius,
            mat,
            b_box,
        }
    }
}

impl Hittable for Sphere {
    fn bounding_box(&self) -> &Aabb {
        &self.b_box
    }

    fn get_lights(&self) -> Vec<Hittables> {
        if self.mat.is_light() {
            vec![self.clone().into()]
        } else {
            vec![]
        }
    }
}

impl Clone for Sphere {
    fn clone(&self) -> Self {
        Sphere {
            center: self.center,
            radius: self.radius,
            mat: self.mat.clone(),
            b_box: self.b_box.clone(),
        }
    }
}
