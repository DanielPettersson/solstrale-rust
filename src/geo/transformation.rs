//! Contains transformations that can modify [`Vec3`]
//! Used to translate and rotate hittables
use crate::geo::vec3::Vec3;
use crate::util::degrees_to_radians;
use derive_more::Constructor;

/// A trait used for different transformations on [`Vec3`]
pub trait Transformer {
    /// Applies transformation
    fn transform(&self, _vec: Vec3, _skip_translation: bool) -> Vec3;
}

/// A transformer that does nothing
/// # Examples:
/// ```
/// # use solstrale::geo::transformation::{NopTransformer, Transformer};
/// # use solstrale::geo::vec3::Vec3;
/// let res = NopTransformer().transform(Vec3::new(1., 2., 3.), false);
/// assert_eq!(Vec3::new(1., 2., 3.), res)
/// ```
pub struct NopTransformer();

impl Transformer for NopTransformer {
    fn transform(&self, vec: Vec3, _skip_translation: bool) -> Vec3 {
        vec
    }
}

/// A list of transformations to apply to a given [`Vec3`]
/// # Examples:
/// ```
/// # use solstrale::geo::transformation::{RotationY, Transformations, Transformer, Translation};
/// # use solstrale::geo::vec3::Vec3;
/// let res = Transformations::new(vec![
///     Box::new(RotationY::new(90.)),
///     Box::new(Translation::new(Vec3::new(1., 0., 0.))),
/// ]).transform(Vec3::new(1., 0., 0.), false);
/// assert_eq!(Vec3::new(1., 0., -1.), res)
/// ```
#[derive(Constructor)]
pub struct Transformations {
    transformations: Vec<Box<dyn Transformer>>,
}

impl Transformer for Transformations {
    fn transform(&self, vec: Vec3, skip_translation: bool) -> Vec3 {
        let mut v = vec;
        for t in &self.transformations {
            v = t.transform(v, skip_translation);
        }
        v
    }
}

/// Translates the position of the given [`Vec3`]
/// # Examples:
/// ```
/// # use solstrale::geo::transformation::{Translation, Transformer};
/// # use solstrale::geo::vec3::Vec3;
/// let translation = Translation::new(Vec3::new(4., 5., 6.));
/// let res = translation.transform(Vec3::new(1., 2., 3.), false);
/// assert_eq!(Vec3::new(5., 7., 9.), res);
/// let res = translation.transform(Vec3::new(1., 2., 3.), true);
/// assert_eq!(Vec3::new(1., 2., 3.), res);
/// ```
pub struct Translation {
    translation: Vec3,
}

impl Translation {
    /// Creates a new translation
    pub fn new(translation: Vec3) -> Translation {
        Translation { translation }
    }
}

impl Transformer for Translation {
    fn transform(&self, vec: Vec3, skip_translation: bool) -> Vec3 {
        if skip_translation {
            vec
        } else {
            vec + self.translation
        }
    }
}

/// Rotates the given [`Vec3`] around the global x-axis by angle degrees
/// # Examples:
/// ```
/// # use solstrale::geo::transformation::{RotationX, Transformer};
/// # use solstrale::geo::vec3::{ALMOST_ZERO, Vec3};
/// let res = RotationX::new(90.).transform(Vec3::new(2., 1., 0.), false);
/// assert!((Vec3::new(2., 0., -1.) - res).length() < ALMOST_ZERO)
/// ```
pub struct RotationX {
    sin_theta: f64,
    cos_theta: f64,
}

impl RotationX {
    /// Creates a new y-rotation
    pub fn new(angle: f64) -> RotationX {
        let radians = degrees_to_radians(angle);
        RotationX {
            sin_theta: radians.sin(),
            cos_theta: radians.cos(),
        }
    }
}

impl Transformer for RotationX {
    fn transform(&self, vec: Vec3, _skip_translation: bool) -> Vec3 {
        Vec3::new(
            vec.x,
            self.cos_theta * vec.y + self.sin_theta * vec.z,
            -self.sin_theta * vec.y + self.cos_theta * vec.z,
        )
    }
}

/// Rotates the given [`Vec3`] around the global y-axis by angle degrees
/// # Examples:
/// ```
/// # use solstrale::geo::transformation::{RotationY, Transformer};
/// # use solstrale::geo::vec3::{ALMOST_ZERO, Vec3};
/// let res = RotationY::new(90.).transform(Vec3::new(2., 1., 0.), false);
/// assert!((Vec3::new(0., 1., -2.) - res).length() < ALMOST_ZERO)
/// ```
pub struct RotationY {
    sin_theta: f64,
    cos_theta: f64,
}

impl RotationY {
    /// Creates a new y-rotation
    pub fn new(angle: f64) -> RotationY {
        let radians = degrees_to_radians(angle);
        RotationY {
            sin_theta: radians.sin(),
            cos_theta: radians.cos(),
        }
    }
}

impl Transformer for RotationY {
    fn transform(&self, vec: Vec3, _skip_translation: bool) -> Vec3 {
        Vec3::new(
            self.cos_theta * vec.x + self.sin_theta * vec.z,
            vec.y,
            -self.sin_theta * vec.x + self.cos_theta * vec.z,
        )
    }
}

/// Rotates the given [`Vec3`] around the global z-axis by angle degrees
/// # Examples:
/// ```
/// # use solstrale::geo::transformation::{RotationZ, Transformer};
/// # use solstrale::geo::vec3::{ALMOST_ZERO, Vec3};
/// let res = RotationZ::new(90.).transform(Vec3::new(1., 0., 2.), false);
/// assert!((Vec3::new(0., -1., 2.) - res).length() < ALMOST_ZERO)
/// ```
pub struct RotationZ {
    sin_theta: f64,
    cos_theta: f64,
}

impl RotationZ {
    /// Creates a new y-rotation
    pub fn new(angle: f64) -> RotationZ {
        let radians = degrees_to_radians(angle);
        RotationZ {
            sin_theta: radians.sin(),
            cos_theta: radians.cos(),
        }
    }
}

impl Transformer for RotationZ {
    fn transform(&self, vec: Vec3, _skip_translation: bool) -> Vec3 {
        Vec3::new(
            self.cos_theta * vec.x + self.sin_theta * vec.y,
            -self.sin_theta * vec.x + self.cos_theta * vec.y,
            vec.z,
        )
    }
}

/// Scales the given [`Vec3`] by the given factor
/// # Examples:
/// ```
/// # use solstrale::geo::transformation::{Scale, Transformer};
/// # use solstrale::geo::vec3::{ALMOST_ZERO, Vec3};
/// let res = Scale::new(3.).transform(Vec3::new(2., 1., 0.), false);
/// assert_eq!(Vec3::new(6., 3., 0.), res);
/// ```
#[derive(Constructor)]
pub struct Scale {
    scale: f64,
}

impl Transformer for Scale {
    fn transform(&self, vec: Vec3, _skip_translation: bool) -> Vec3 {
        vec * self.scale
    }
}
