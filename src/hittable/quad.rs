use crate::combine_aabbs;
use crate::geo::Aabb;
use crate::geo::transformation::Transformer;
use crate::geo::vec3::Vec3;
use crate::hittable::{Hittable, Hittables};
use crate::material::{Material, Materials};

/// A rectangular flat hittable object
#[derive(Clone, Debug)]
pub struct Quad {
    pub(crate) q: Vec3,
    pub(crate) u: Vec3,
    pub(crate) v: Vec3,
    pub(crate) normal: Vec3,
    pub(crate) d: f64,
    pub(crate) w: Vec3,
    pub(crate) mat: Materials,
    b_box: Aabb,
    pub(crate) area: f64,
}

impl Quad {
    /// Creates a new quad
    pub fn new(
        q: Vec3,
        u: Vec3,
        v: Vec3,
        mat: Materials,
        transformation: &dyn Transformer,
    ) -> Self {
        let q = transformation.transform(q, false);
        let u = transformation.transform(u, true);
        let v = transformation.transform(v, true);

        let b_box = combine_aabbs!(
            &Aabb::new_from_2_points(q, q + u),
            &Aabb::new_from_2_points(q, q + v),
            &Aabb::new_from_2_points(q, q + u + v)
        )
        .pad_if_needed();

        let n = u.cross(v);
        let normal = n.unit();

        Quad {
            q,
            u,
            v,
            normal,
            d: normal.dot(q),
            w: n / n.dot(n),
            mat,
            b_box,
            area: n.length(),
        }
    }

    /// creates a new box shaped hittable object
    pub fn new_box(
        a: Vec3,
        b: Vec3,
        mat: Materials,
        transformation: &dyn Transformer,
    ) -> Vec<Hittables> {
        let mut sides: Vec<Hittables> = Vec::new();

        let min = Vec3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z));
        let max = Vec3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z));

        let dx = Vec3::new(max.x - min.x, 0., 0.);
        let dy = Vec3::new(0., max.y - min.y, 0.);
        let dz = Vec3::new(0., 0., max.z - min.z);

        sides.push(
            Quad::new(
                Vec3::new(min.x, min.y, max.z),
                dx,
                dy,
                mat.clone(),
                transformation,
            )
            .into(),
        );
        sides.push(
            Quad::new(
                Vec3::new(max.x, min.y, max.z),
                dz.neg(),
                dy,
                mat.clone(),
                transformation,
            )
            .into(),
        );
        sides.push(
            Quad::new(
                Vec3::new(max.x, min.y, min.z),
                dx.neg(),
                dy,
                mat.clone(),
                transformation,
            )
            .into(),
        );
        sides.push(
            Quad::new(
                Vec3::new(min.x, min.y, min.z),
                dz,
                dy,
                mat.clone(),
                transformation,
            )
            .into(),
        );
        sides.push(
            Quad::new(
                Vec3::new(min.x, max.y, max.z),
                dx,
                dz.neg(),
                mat.clone(),
                transformation,
            )
            .into(),
        );
        sides.push(Quad::new(Vec3::new(min.x, min.y, min.z), dx, dz, mat, transformation).into());

        sides
    }
}

impl Hittable for Quad {
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
