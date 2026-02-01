use crate::geo::Ray;
use crate::geo::Uv;
use crate::geo::transformation::Transformer;
use crate::geo::vec3::{ALMOST_ZERO, Vec3};
use crate::geo::{Aabb, Onb};
use crate::hittable::{Hittable, Hittables};
use crate::material::{Material, Materials, RayHit};
use crate::random::random_normal_float;
use crate::util::interval::{Interval, RAY_INTERVAL};

/// A triangle-shaped hittable object
#[derive(Clone, Debug)]
pub struct Triangle {
    pub(crate) v0: Vec3,
    pub(crate) v0v1: Vec3,
    pub(crate) v0v2: Vec3,
    pub(crate) uv0: Uv,
    pub(crate) uv1: Uv,
    pub(crate) uv2: Uv,
    pub(crate) normal: Vec3,
    tangent: Vec3,
    bi_tangent: Vec3,
    pub(crate) mat: Materials,
    b_box: Aabb,
    area: f64,
}

impl Triangle {
    /// Creates a new triangle hittable object with no texture coordinates
    pub fn new(
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
        mat: Materials,
        transformation: &dyn Transformer,
    ) -> Triangle {
        Triangle::new_with_tex_coords(
            v0,
            v1,
            v2,
            Uv { u: 0.0, v: 0.0 },
            Uv { u: 0.0, v: 0.0 },
            Uv { u: 0.0, v: 0.0 },
            mat,
            transformation,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Creates a new triangle flat hittable object. A counterclockwise winding is expected
    pub fn new_with_tex_coords(
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
        uv0: Uv,
        uv1: Uv,
        uv2: Uv,
        mat: Materials,
        transformation: &dyn Transformer,
    ) -> Triangle {
        let v0 = transformation.transform(v0, false);
        let v1 = transformation.transform(v1, false);
        let v2 = transformation.transform(v2, false);

        let b_box = Aabb::new_from_3_points(v0, v1, v2).pad_if_needed();
        let v0v1 = v1 - v0;
        let v0v2 = v2 - v0;
        let n = v0v1.cross(v0v2);
        let normal = n.unit();
        let area = n.length() / 2.;

        let delta_pos_1 = v1 - v0;
        let delta_pos_2 = v2 - v0;
        let delta_uv_1 = uv1 - uv0;
        let delta_uv_2 = uv2 - uv0;
        let r = 1. / (delta_uv_1.u * delta_uv_2.v - delta_uv_1.v * delta_uv_2.u);
        let tangent = ((delta_pos_1 * delta_uv_2.v - delta_pos_2 * delta_uv_1.v) * r).unit();
        let bi_tangent = ((delta_pos_2 * delta_uv_1.u - delta_pos_1 * delta_uv_2.u) * r).unit();

        Triangle {
            v0,
            v0v1,
            v0v2,
            uv0,
            uv1,
            uv2,
            normal,
            tangent,
            bi_tangent,
            mat,
            b_box,
            area,
        }
    }
}

impl Hittable for Triangle {
    fn pdf_value(&self, origin: Vec3, direction: Vec3) -> f64 {
        let ray = Ray::new(origin, direction);

        match self.hit(&ray, &RAY_INTERVAL) {
            None => 0.,
            Some(rec) => {
                let distance_squared = rec.ray_length * rec.ray_length * direction.length_squared();
                let cosine = (direction.dot(rec.normal) / direction.length()).abs();

                distance_squared / (cosine * self.area)
            }
        }
    }

    fn random_direction(&self, origin: Vec3) -> Vec3 {
        let p = self.v0 + self.v0v1 * random_normal_float() + self.v0v2 * random_normal_float();
        p - origin
    }

    fn hit(&self, r: &Ray, ray_length: &Interval) -> Option<RayHit<'_>> {
        let p_vec = r.direction.cross(self.v0v2);
        let det = self.v0v1.dot(p_vec);

        // No hit if the ray is parallel to the plane
        if det.abs() < ALMOST_ZERO {
            return None;
        }

        let inv_det = 1. / det;
        let t_vec = r.origin - self.v0;
        let q_vec = t_vec.cross(self.v0v1);

        // Is hit point outside of primitive
        let u = (t_vec.dot(p_vec) * inv_det) as f32;
        if !(0. ..=1.).contains(&u) {
            return None;
        }
        let v = (r.direction.dot(q_vec) * inv_det) as f32;
        if v < 0. || u + v > 1. {
            return None;
        }

        let tt = self.v0v2.dot(q_vec) * inv_det;
        let intersection = r.at(tt);

        // Return false if the hit point parameter t is outside the ray length interval.
        if !ray_length.contains(tt) {
            return None;
        }

        let uv0 = 1. - u - v;
        let uv = Uv::new(
            uv0 * self.uv0.u + u * self.uv1.u + v * self.uv2.u,
            uv0 * self.uv0.v + u * self.uv1.v + v * self.uv2.v,
        );

        let mut normal = self.normal;
        let front_face = r.direction.dot(normal) < 0.;
        if !front_face {
            normal = normal.neg()
        }
        Some(RayHit::new(
            intersection,
            Onb {
                tangent: self.tangent,
                bi_tangent: self.bi_tangent,
                normal,
            },
            &self.mat,
            tt,
            uv,
            front_face,
        ))
    }

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
