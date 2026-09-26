use crate::geo::Aabb;
use crate::geo::transformation::Transformer;
use crate::geo::vec3::{UNIT_Y, Vec3};
use crate::hittable::Hittable;
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
    /// Creates a new sphere, with the transform applied to it at construction
    /// as [`Quad::new`](crate::hittable::Quad::new) and
    /// [`Triangle::new`](crate::hittable::Triangle::new) do: the centre is
    /// moved and the radius scaled.
    ///
    /// Rotation carries the centre around the origin like any other point, but
    /// is *not* applied to the texture: `resolve_hit` in
    /// `renderer/ray_trace.wgsl` derives the spherical coordinates from the
    /// world-space outward normal, so turning a texture on a sphere would need
    /// the rotation stored and undone there. A rotated texture on a sphere is
    /// therefore not expressible.
    pub fn new(
        center: Vec3,
        radius: f64,
        mat: Materials,
        transformation: &dyn Transformer,
    ) -> Sphere {
        let center = transformation.transform(center, false);
        // The scale factor, read off a unit vector with translation skipped: a
        // rotation leaves its length alone and `Scale` multiplies it, so any
        // unit vector gives the same factor. `Scale` is uniform-only, so the
        // non-uniform case does not arise.
        let radius = radius * transformation.transform(UNIT_Y, true).length();

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

    fn has_lights(&self) -> bool {
        self.mat.is_light()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::transformation::{
        NopTransformer, RotationY, Scale, Transformations, Translation,
    };
    use crate::geo::vec3::ALMOST_ZERO;
    use crate::material::Lambertian;
    use crate::material::texture::SolidColor;

    fn sphere(transformation: &dyn Transformer) -> Sphere {
        Sphere::new(
            Vec3::new(1., 2., 3.),
            2.,
            Lambertian::new(SolidColor::new(1., 1., 1.).into(), None).into(),
            transformation,
        )
    }

    #[test]
    fn nop_transformer_leaves_the_sphere_alone() {
        let s = sphere(&NopTransformer());
        assert_eq!(Vec3::new(1., 2., 3.), s.center);
        assert_eq!(2., s.radius);
    }

    #[test]
    fn translation_moves_the_centre_and_leaves_the_radius() {
        let s = sphere(&Translation::new(Vec3::new(10., 0., -5.)));
        assert_eq!(Vec3::new(11., 2., -2.), s.center);
        assert_eq!(2., s.radius);
    }

    #[test]
    fn scale_takes_the_centre_and_the_radius_with_it() {
        let s = sphere(&Scale::new(3.));
        assert_eq!(Vec3::new(3., 6., 9.), s.center);
        assert_eq!(6., s.radius);
    }

    /// A sphere's own rotation is nothing, but the centre is a point like any
    /// other and goes round the origin.
    #[test]
    fn rotation_carries_the_centre_and_leaves_the_radius() {
        let s = sphere(&RotationY::new(90.));
        assert!((Vec3::new(3., 2., -1.) - s.center).length() < ALMOST_ZERO);
        assert_eq!(2., s.radius);
    }

    /// Composed, in the order the list gives: scaled about the origin first,
    /// then turned, then moved -- so the translation is not scaled and the
    /// radius is.
    #[test]
    fn transformations_compose() {
        let s = sphere(&Transformations::new(vec![
            Box::new(Scale::new(2.)),
            Box::new(RotationY::new(90.)),
            Box::new(Translation::new(Vec3::new(0., 1., 0.))),
        ]));
        assert!((Vec3::new(6., 5., -2.) - s.center).length() < ALMOST_ZERO);
        assert_eq!(4., s.radius);
    }

    /// The box has to follow the radius, or a scaled-up sphere is culled by
    /// the traversal before it is ever tested.
    #[test]
    fn the_bounding_box_follows_the_scaled_radius() {
        let s = sphere(&Scale::new(3.));
        let b = s.bounding_box();
        assert_eq!(-3., b.x.min);
        assert_eq!(9., b.x.max);
        assert_eq!(0., b.y.min);
        assert_eq!(12., b.y.max);
        assert_eq!(3., b.z.min);
        assert_eq!(15., b.z.max);
    }
}
