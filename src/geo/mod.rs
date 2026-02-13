//! Basic geometric constructs
use derive_more::Constructor;
use std::ops::{Add, Sub};

use crate::geo::vec3::Vec3;
use crate::util::interval::{EMPTY_INTERVAL, Interval, combine_intervals};

pub mod transformation;
pub mod vec3;

const PAD_DELTA: f64 = 0.01;

/// Texture map coordinates
#[derive(Copy, Clone, Debug, Default, PartialEq, Constructor)]
pub struct Uv {
    /// U coordinate
    pub u: f32,
    /// V coordinate
    pub v: f32,
}

impl Sub for Uv {
    type Output = Uv;

    /// returns a Uv that has all values subtracted by corresponding value in given Uv
    /// # Examples:
    /// ```
    /// # use solstrale::geo::Uv;
    /// let res = Uv::new(1., 2.) - Uv::new(6., 5.);
    /// assert_eq!(Uv::new(-5., -3.), res)
    /// ```
    fn sub(self, rhs: Self) -> Self::Output {
        Uv::new(self.u - rhs.u, self.v - rhs.v)
    }
}

/// Axis Aligned Bounding Box
#[derive(Clone, Debug)]
pub struct Aabb {
    /// X axis interval
    pub x: Interval,
    /// y axis interval
    pub y: Interval,
    /// z axis interval
    pub z: Interval,
}

impl Default for Aabb {
    fn default() -> Self {
        Aabb {
            x: EMPTY_INTERVAL,
            y: EMPTY_INTERVAL,
            z: EMPTY_INTERVAL,
        }
    }
}

/// Combines the given [`Aabb`] arguments into a single [`Aabb`] encapsulating all
/// # Examples:
/// ```
/// # use solstrale::combine_aabbs;
/// # use solstrale::geo::Aabb;
/// # use solstrale::geo::vec3::Vec3;
/// # use solstrale::util::interval::Interval;
/// let aabb = combine_aabbs![
///     &Aabb::new_from_2_points(Vec3::new(-1., 0., 0.), Vec3::new(1., 0., 0.)),
///     &Aabb::new_from_2_points(Vec3::new(0., -2., 0.), Vec3::new(0., 2., 0.)),
///     &Aabb::new_from_2_points(Vec3::new(0., 0., -3.), Vec3::new(0., 0.,  3.))
/// ];
/// assert_eq!(aabb.x, Interval::new(-1., 1.));
/// assert_eq!(aabb.y, Interval::new(-2., 2.));
/// assert_eq!(aabb.z, Interval::new(-3., 3.));
#[macro_export]
macro_rules! combine_aabbs {
    ( $( $x:expr_2021 ),* ) => {
        {
            let mut temp_aabb = Aabb::default();
            $(
                temp_aabb = temp_aabb.combine($x);
            )*
            temp_aabb
        }
    };
}

impl Aabb {
    /// Create a new aabb exactly encapsulating the two given points
    pub fn new_from_2_points(a: Vec3, b: Vec3) -> Aabb {
        Aabb {
            x: Interval {
                min: a.x.min(b.x),
                max: a.x.max(b.x),
            },
            y: Interval {
                min: a.y.min(b.y),
                max: a.y.max(b.y),
            },
            z: Interval {
                min: a.z.min(b.z),
                max: a.z.max(b.z),
            },
        }
    }

    /// Create a new aabb exactly encapsulating the three given points
    pub fn new_from_3_points(a: Vec3, b: Vec3, c: Vec3) -> Aabb {
        Aabb {
            x: Interval {
                min: a.x.min(b.x).min(c.x),
                max: a.x.max(b.x).max(c.x),
            },
            y: Interval {
                min: a.y.min(b.y).min(c.y),
                max: a.y.max(b.y).max(c.y),
            },
            z: Interval {
                min: a.z.min(b.z).min(c.z),
                max: a.z.max(b.z).max(c.z),
            },
        }
    }

    /// Create a new aabb that is the sum of the two given aabb's
    pub fn combine(&self, a: &Aabb) -> Aabb {
        Aabb {
            x: combine_intervals(self.x, a.x),
            y: combine_intervals(self.y, a.y),
            z: combine_intervals(self.z, a.z),
        }
    }

    /// Create a new aabb the same size as self.
    /// Except for axis that are very small, these are padded a bit
    pub fn pad_if_needed(&self) -> Aabb {
        let new_x = if self.x.size() >= PAD_DELTA {
            self.x
        } else {
            self.x.expand(PAD_DELTA)
        };
        let new_y = if self.y.size() >= PAD_DELTA {
            self.y
        } else {
            self.y.expand(PAD_DELTA)
        };
        let new_z = if self.z.size() >= PAD_DELTA {
            self.z
        } else {
            self.z.expand(PAD_DELTA)
        };

        Aabb {
            x: new_x,
            y: new_y,
            z: new_z,
        }
    }

    /// return the center point of the aabb
    /// # Examples:
    /// ```
    /// # use solstrale::geo::Aabb;
    /// # use solstrale::geo::vec3::Vec3;
    /// let aabb = Aabb::new_from_2_points(Vec3::new(-5., 0., 1.), Vec3::new(5., 2., 1.));
    /// assert_eq!(aabb.center(), Vec3::new(0. , 1., 1.));
    /// ```
    pub fn center(&self) -> Vec3 {
        Vec3::new(
            (self.x.min + self.x.max) * 0.5,
            (self.y.min + self.y.max) * 0.5,
            (self.z.min + self.z.max) * 0.5,
        )
    }
}

impl Add<Vec3> for &Aabb {
    type Output = Aabb;

    fn add(self, rhs: Vec3) -> Self::Output {
        Aabb {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_padding() {
        // Flat in Z
        let aabb = Aabb::new_from_2_points(Vec3::new(0., 0., 555.), Vec3::new(555., 555., 555.));
        assert_eq!(aabb.z.size(), 0.0);

        let padded = aabb.pad_if_needed();
        assert!(padded.z.size() >= PAD_DELTA - 0.000001);
        assert_eq!(padded.x.size(), 555.0);
        assert_eq!(padded.y.size(), 555.0);

        // Check if padding is enough for f32 precision at large coordinates
        let z_min_f32 = padded.z.min as f32;
        let z_max_f32 = padded.z.max as f32;
        assert!(
            z_max_f32 > z_min_f32,
            "Padding {} is too small for f32 at 555.0: {} == {}",
            PAD_DELTA,
            z_min_f32,
            z_max_f32
        );
    }
}
