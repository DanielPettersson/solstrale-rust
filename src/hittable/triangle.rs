use crate::geo::Aabb;
use crate::geo::Uv;
use crate::geo::transformation::Transformer;
use crate::geo::vec3::Vec3;
use crate::hittable::{Hittable, Hittables};
use crate::material::{Material, Materials};

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
    pub(crate) tangent: Vec3,
    pub(crate) bi_tangent: Vec3,
    pub(crate) mat: Materials,
    b_box: Aabb,
    pub(crate) area: f64,
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
