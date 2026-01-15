use crate::geo::Ray;
use crate::geo::Uv;
use crate::geo::vec3::{ONE_VECTOR, Vec3, random_unit_vector};
use crate::geo::{Aabb, Onb};
use crate::hittable::{Hittable, Hittables};
use crate::material::Materials;
use crate::material::texture::SolidColor;
use crate::material::{Isotropic, RayHit};
use crate::random::random_normal_float;
use crate::util::interval::{Interval, UNIVERSE_INTERVAL};

/// A fog type hittable object where rays not only scatter
/// at the edge of the object, but at random points inside the object
/// The material of the boundary hittable is ignored
#[derive(Clone, Debug)]
pub struct ConstantMedium {
    boundary: Box<Hittables>,
    negative_inverse_density: f64,
    phase_function: Materials,
}

impl ConstantMedium {
    /// Creates a new instance of the constant medium
    pub fn new(boundary: Hittables, density: f64, color: Vec3) -> Self {
        ConstantMedium {
            boundary: Box::new(boundary),
            negative_inverse_density: -1. / density,
            phase_function: Isotropic::new(SolidColor::new_from_vec3(color).into()).into(),
        }
    }
}

impl Hittable for ConstantMedium {
    fn hit(&self, r: &Ray, ray_length: &Interval) -> Option<RayHit<'_>> {
        match self.boundary.hit(r, &UNIVERSE_INTERVAL) {
            None => None,
            Some(rec1) => {
                let interval = Interval::new(rec1.ray_length + 0.0001, f64::INFINITY);
                match self.boundary.hit(r, &interval) {
                    None => None,
                    Some(rec2) => {
                        let mut rec1_ray_length = rec1.ray_length.max(ray_length.min);
                        let rec2_ray_length = rec2.ray_length.min(ray_length.max);

                        if rec1_ray_length >= rec2_ray_length {
                            return None;
                        }

                        rec1_ray_length = rec1_ray_length.max(0.);
                        let r_length = r.direction.length();
                        let distance_inside_boundary =
                            (rec2_ray_length - rec1_ray_length) * r_length;
                        let hit_distance =
                            self.negative_inverse_density * random_normal_float().ln();

                        if hit_distance > distance_inside_boundary {
                            return None;
                        }

                        let t = rec1_ray_length + hit_distance / r_length;

                        Some(RayHit::new(
                            r.at(t),
                            Onb {
                                tangent: ONE_VECTOR,
                                bi_tangent: ONE_VECTOR,
                                normal: random_unit_vector(),
                            },
                            &self.phase_function,
                            t,
                            Uv::default(),
                            false,
                        ))
                    }
                }
            }
        }
    }

    fn bounding_box(&self) -> &Aabb {
        self.boundary.bounding_box()
    }

    fn get_lights(&self) -> Vec<Hittables> {
        vec![]
    }
}
