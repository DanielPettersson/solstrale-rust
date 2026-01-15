use std::f64::consts::PI;

use crate::geo::Aabb;
use crate::geo::Onb;
use crate::geo::Ray;
use crate::geo::Uv;
use crate::geo::vec3::{UNIT_Y, Vec3};
use crate::hittable::{Hittable, Hittables};
use crate::material::{Material, Materials, RayHit};
use crate::random::random_normal_float;
use crate::util::interval::{Interval, RAY_INTERVAL};

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
    fn pdf_value(&self, origin: Vec3, direction: Vec3) -> f64 {
        let ray = Ray::new(origin, direction);

        let hit = self.hit(&ray, &RAY_INTERVAL);

        match hit {
            None => 0.,
            Some(_) => {
                let cos_theta_max = (1.
                    - self.radius * self.radius / (self.center - origin).length_squared())
                .sqrt();
                let solid_angle = 2. * PI * (1. - cos_theta_max);

                1. / solid_angle
            }
        }
    }

    fn random_direction(&self, origin: Vec3) -> Vec3 {
        let direction = self.center - origin;
        let uvw = Onb::new(direction);
        uvw.local(random_to_sphere(self.radius, direction.length_squared()))
    }

    fn hit(&self, r: &Ray, ray_length: &Interval) -> Option<RayHit<'_>> {
        let oc = r.origin - self.center;
        let a = r.direction.length_squared();
        let half_b = oc.dot(r.direction);
        let c = oc.length_squared() - self.radius * self.radius;

        let discriminant = half_b * half_b - a * c;
        if discriminant < 0. {
            return None;
        }
        let sqrt_d = discriminant.sqrt();

        let mut root = (-half_b - sqrt_d) / a;
        if !ray_length.contains(root) {
            root = (-half_b + sqrt_d) / a;
            if !ray_length.contains(root) {
                return None;
            }
        }

        let hit_point = r.at(root);
        let n = hit_point - self.center;
        let mut normal = n.unit();
        let uv = calculate_sphere_uv(normal);

        let tangent = UNIT_Y.cross(n).unit();
        let bi_tangent = n.cross(tangent);

        let front_face = r.direction.dot(normal) < 0.;
        if !front_face {
            normal = normal.neg();
        }
        Some(RayHit::new(
            hit_point,
            Onb {
                tangent,
                bi_tangent,
                normal,
            },
            &self.mat,
            root,
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

fn calculate_sphere_uv(point_on_sphere: Vec3) -> Uv {
    let theta = (-point_on_sphere.y).acos();
    let phi = -point_on_sphere.z.atan2(point_on_sphere.x) + PI;
    let u = phi / (2. * PI);
    let v = theta / PI;
    Uv::new(u as f32, v as f32)
}

fn random_to_sphere(radius: f64, distance_squared: f64) -> Vec3 {
    let r1 = random_normal_float();
    let r2 = random_normal_float();
    let z = 1. + r2 * ((1. - radius * radius / distance_squared).sqrt() - 1.);

    let phi = 2. * PI * r1;
    let zz = (1. - z * z).sqrt();
    let x = phi.cos() * zz;
    let y = phi.sin() * zz;

    Vec3::new(x, y, z)
}
