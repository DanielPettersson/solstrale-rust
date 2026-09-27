use crate::geo::Aabb;
use crate::geo::Uv;
use crate::geo::transformation::Transformer;
use crate::geo::vec3::{UNIT_Y, Vec3};
use crate::hittable::Hittable;
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
    /// Shading normal at `v0`. Equal to `normal` unless the triangle was built
    /// with per-vertex normals, which is what makes the mesh smooth-shaded.
    pub(crate) n0: Vec3,
    /// Shading normal at `v1`
    pub(crate) n1: Vec3,
    /// Shading normal at `v2`
    pub(crate) n2: Vec3,
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
        Triangle::build([v0, v1, v2], None, [uv0, uv1, uv2], mat, transformation)
    }

    /// Creates a smooth-shaded triangle: the shading normal is interpolated
    /// across the face from the three supplied per-vertex normals. The winding
    /// still decides the geometric normal that facing and light transport key
    /// off.
    ///
    /// Normals are transformed with translation skipped and re-normalised,
    /// which is correct only without shear or non-uniform scale. Every
    /// [`Transformer`] here is rigid or uniform, and the trait maps a `Vec3` to
    /// a `Vec3` with no way to express the matrix an inverse transpose would
    /// need, so nothing reachable through this API can violate it.
    pub fn new_with_normals(
        v: [Vec3; 3],
        n: [Vec3; 3],
        uv: [Uv; 3],
        mat: Materials,
        transformation: &dyn Transformer,
    ) -> Triangle {
        Triangle::build(v, Some(n), uv, mat, transformation)
    }

    /// What this triangle was built from, as `build` takes it: corners, the
    /// shading normals if it is smooth, and texture coordinates.
    fn parts(&self) -> ([Vec3; 3], Option<[Vec3; 3]>) {
        let v = [self.v0, self.v0 + self.v0v1, self.v0 + self.v0v2];
        // Flat shading is all three corners at the geometric normal exactly,
        // which is what `build` gives a triangle built without normals.
        let flat = self.n0 == self.normal && self.n1 == self.normal && self.n2 == self.normal;
        (v, (!flat).then_some([self.n0, self.n1, self.n2]))
    }

    pub(crate) fn uv(&self) -> [Uv; 3] {
        [self.uv0, self.uv1, self.uv2]
    }

    /// This triangle under `transformation`, applied on top of the one it was
    /// built with.
    pub(crate) fn transformed(&self, transformation: &dyn Transformer) -> Triangle {
        let (v, normals) = self.parts();
        Triangle::build(v, normals, self.uv(), self.mat.clone(), transformation)
    }

    /// What [`Self::transformed`] would hold, without building the triangle:
    /// what a renderer uploads for one it never needs on the CPU.
    pub(crate) fn transformed_geometry(&self, transformation: &dyn Transformer) -> Geometry {
        let (v, normals) = self.parts();
        geometry(v, normals, self.uv(), transformation)
    }

    /// The bounding box of [`Self::transformed`], and nothing else of it.
    pub(crate) fn transformed_bounds(&self, transformation: &dyn Transformer) -> Aabb {
        let (v, _) = self.parts();
        bounds(v.map(|v| transformation.transform(v, false)))
    }

    /// The one constructor. `normals` of `None` is flat shading, expressed by
    /// giving all three corners the geometric normal rather than by a flag, so
    /// the shader interpolates unconditionally and needs no branch.
    fn build(
        v: [Vec3; 3],
        normals: Option<[Vec3; 3]>,
        uv: [Uv; 3],
        mat: Materials,
        transformation: &dyn Transformer,
    ) -> Triangle {
        let [uv0, uv1, uv2] = uv;
        let g = geometry(v, normals, uv, transformation);
        Triangle {
            v0: g.v0,
            v0v1: g.v0v1,
            v0v2: g.v0v2,
            uv0,
            uv1,
            uv2,
            normal: g.normal,
            n0: g.n0,
            n1: g.n1,
            n2: g.n2,
            tangent: g.tangent,
            bi_tangent: g.bi_tangent,
            mat,
            b_box: g.b_box,
            area: g.area,
        }
    }
}

/// A triangle's geometry as `build` derives it, without its material.
pub(crate) struct Geometry {
    pub(crate) v0: Vec3,
    pub(crate) v0v1: Vec3,
    pub(crate) v0v2: Vec3,
    pub(crate) normal: Vec3,
    pub(crate) n0: Vec3,
    pub(crate) n1: Vec3,
    pub(crate) n2: Vec3,
    pub(crate) tangent: Vec3,
    pub(crate) bi_tangent: Vec3,
    pub(crate) b_box: Aabb,
    pub(crate) area: f64,
}

fn bounds([v0, v1, v2]: [Vec3; 3]) -> Aabb {
    Aabb::new_from_3_points(v0, v1, v2).pad_if_needed()
}

fn geometry(
    v: [Vec3; 3],
    normals: Option<[Vec3; 3]>,
    uv: [Uv; 3],
    transformation: &dyn Transformer,
) -> Geometry {
    let [uv0, uv1, uv2] = uv;
    let v0 = transformation.transform(v[0], false);
    let v1 = transformation.transform(v[1], false);
    let v2 = transformation.transform(v[2], false);

    let b_box = bounds([v0, v1, v2]);
    let v0v1 = v1 - v0;
    let v0v2 = v2 - v0;
    let n = v0v1.cross(v0v2);
    let normal = n.unit();
    let area = n.length() / 2.;

    let [n0, n1, n2] = match normals {
        None => [normal, normal, normal],
        Some(n) => n.map(|n| {
            let n = transformation.transform(n, true);
            // spider.obj carries a zero-length `vn`, and `unit()` of it is
            // NaN in all three components. Fall back to the geometric
            // normal, which is what that corner had before.
            if n.near_zero() { normal } else { n.unit() }
        }),
    };

    let delta_pos_1 = v1 - v0;
    let delta_pos_2 = v2 - v0;
    let delta_uv_1 = uv1 - uv0;
    let delta_uv_2 = uv2 - uv0;
    // Degenerate when the triangle has no UV area -- which is every triangle
    // of an untextured OBJ, since they all carry (0,0) coordinates. Without
    // this guard the reciprocal is infinite and both vectors come out NaN,
    // and those NaNs were being uploaded to the GPU.
    let det = delta_uv_1.u * delta_uv_2.v - delta_uv_1.v * delta_uv_2.u;
    let (tangent, bi_tangent) = if det.abs() < f32::EPSILON {
        // No UV frame to derive: fall back to an arbitrary basis orthogonal
        // to the normal. Only ever used for normal mapping, which needs a
        // UV frame to be meaningful anyway.
        let a = if normal.x.abs() > 0.9 {
            UNIT_Y
        } else {
            Vec3::new(1., 0., 0.)
        };
        let t = normal.cross(a).unit();
        (t, normal.cross(t))
    } else {
        let r = 1. / det as f64;
        (
            ((delta_pos_1 * delta_uv_2.v as f64 - delta_pos_2 * delta_uv_1.v as f64) * r).unit(),
            ((delta_pos_2 * delta_uv_1.u as f64 - delta_pos_1 * delta_uv_2.u as f64) * r).unit(),
        )
    };

    Geometry {
        v0,
        v0v1,
        v0v2,
        normal,
        n0,
        n1,
        n2,
        tangent,
        bi_tangent,
        b_box,
        area,
    }
}

impl Hittable for Triangle {
    fn bounding_box(&self) -> &Aabb {
        &self.b_box
    }

    fn has_lights(&self) -> bool {
        self.mat.is_light()
    }
}
