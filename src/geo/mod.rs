//! Basic geometric constructs
use derive_more::Constructor;
use std::ops::{Add, Sub};

use crate::geo::vec3::Vec3;
use crate::util::interval::{EMPTY_INTERVAL, Interval, combine_intervals};

pub mod transformation;
pub mod vec3;

const PAD_DELTA: f64 = 0.0001;

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

    /// Checks if the given ray hits the aabb
    pub fn hit(&self, r: &Ray) -> bool {
        let mut t_min = 0.;
        let mut t_max = f64::INFINITY;

        if r.direction_inverted.x.is_sign_negative() {
            t_min = ((self.x.max - r.origin.x) * r.direction_inverted.x).max(t_min);
            t_max = ((self.x.min - r.origin.x) * r.direction_inverted.x).min(t_max);
        } else {
            t_min = ((self.x.min - r.origin.x) * r.direction_inverted.x).max(t_min);
            t_max = ((self.x.max - r.origin.x) * r.direction_inverted.x).min(t_max);
        };

        if r.direction_inverted.y.is_sign_negative() {
            t_min = ((self.y.max - r.origin.y) * r.direction_inverted.y).max(t_min);
            t_max = ((self.y.min - r.origin.y) * r.direction_inverted.y).min(t_max);
        } else {
            t_min = ((self.y.min - r.origin.y) * r.direction_inverted.y).max(t_min);
            t_max = ((self.y.max - r.origin.y) * r.direction_inverted.y).min(t_max);
        };

        if r.direction_inverted.z.is_sign_negative() {
            t_min = ((self.z.max - r.origin.z) * r.direction_inverted.z).max(t_min);
            t_max = ((self.z.min - r.origin.z) * r.direction_inverted.z).min(t_max);
        } else {
            t_min = ((self.z.min - r.origin.z) * r.direction_inverted.z).max(t_min);
            t_max = ((self.z.max - r.origin.z) * r.direction_inverted.z).min(t_max);
        };

        t_min < t_max
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

    /// return the length of the aabb diagonal
    /// # Examples:
    /// ```
    /// # use solstrale::geo::Aabb;
    /// # use solstrale::geo::vec3::Vec3;
    /// let aabb = Aabb::new_from_2_points(Vec3::new(1., 1., 1.), Vec3::new(5., 4., 3.));
    /// assert_eq!(aabb.diagonal_length(), 5.385164807134504);
    /// ```
    pub fn diagonal_length(&self) -> f64 {
        Vec3::new(
            self.x.min - self.x.max,
            self.y.min - self.y.max,
            self.z.min - self.z.max,
        )
        .length()
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

/// Orthonormal Basis
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Onb {
    pub(crate) tangent: Vec3,
    pub(crate) bi_tangent: Vec3,
    pub(crate) normal: Vec3,
}

impl Onb {
    /// Creates a new Orthonormal Basis from the given vector
    pub fn new(w: Vec3) -> Onb {
        let unit_w = w.unit();

        let a = if unit_w.x.abs() > 0.9 {
            Vec3::new(0., 1., 0.)
        } else {
            Vec3::new(1., 0., 0.)
        };
        let v = unit_w.cross(a).unit();
        let u = unit_w.cross(v);

        Onb {
            tangent: u,
            bi_tangent: v,
            normal: unit_w,
        }
    }

    /// Translates the given vector to the Orthonormal Basis
    pub fn local(&self, a: Vec3) -> Vec3 {
        self.tangent * a.x + self.bi_tangent * a.y + self.normal * a.z
    }
}

/// Defines a ray of light used by the ray tracer
#[derive(PartialEq, Debug, Default)]
pub struct Ray {
    /// Point where the ray starts
    pub origin: Vec3,
    /// Direction of the ray
    pub direction: Vec3,
    direction_inverted: Vec3,
}

impl Ray {
    /// Create a new ray instance
    pub fn new(origin: Vec3, dir: Vec3) -> Ray {
        let dir_inv = Vec3::new(1. / dir.x, 1. / dir.y, 1. / dir.z);

        Ray {
            origin,
            direction: dir,
            direction_inverted: dir_inv,
        }
    }

    /// returns the position at a given length of the ray
    pub fn at(&self, distance: f64) -> Vec3 {
        self.origin + self.direction * distance
    }

    /// Calculates the shortest distance between the rays
    pub fn shortest_distance(&self, ray: &Ray) -> f64 {
        let n = self.direction.cross(ray.direction);
        let od = self.origin - ray.origin;

        // In case the lines are parallel
        if n.length() == 0. {
            self.direction.cross(od).length() / self.direction.length()
        } else {
            od.dot(n) / n.length()
        }
        .abs()
    }
}

#[cfg(test)]
mod ray_tests {
    use crate::geo::Ray;
    use crate::geo::vec3::Vec3;

    #[test]
    fn test_at() {
        let origin = Vec3::new(1., 2., 3.);
        let direction = Vec3::new(4., 5., 6.).unit();
        let l = direction.length();

        let r = Ray::new(origin, direction);

        assert_eq!(r.at(0.), origin);
        assert!((r.at(l) - origin - direction).near_zero());
        assert!((r.at(-l) - origin + direction).near_zero());
    }

    #[test]
    fn test_shortest_distance() {
        let r1 = Ray::new(Vec3::new(-1., 0., 0.), Vec3::new(2., 0., 0.));
        let r2 = Ray::new(Vec3::new(0., 2., -1.), Vec3::new(0., 0., 2.));
        assert_eq!(r1.shortest_distance(&r2), 2.);
        assert_eq!(r2.shortest_distance(&r1), 2.);
    }

    #[test]
    fn test_shortest_distance2() {
        let r1 = Ray::new(Vec3::new(-1., 0., 0.), Vec3::new(2., 0., 0.));
        let r2 = Ray::new(Vec3::new(0., 2., -1.), Vec3::new(0., 1., 2.));
        assert_eq!(r1.shortest_distance(&r2), 2.23606797749979);
        assert_eq!(r2.shortest_distance(&r1), 2.23606797749979);
    }

    #[test]
    fn test_shortest_parallel() {
        let r1 = Ray::new(Vec3::new(-1., 0., 0.), Vec3::new(4., 2., 0.));
        let r2 = Ray::new(Vec3::new(-1., 1., 0.), Vec3::new(2., 1., 0.));
        assert_eq!(r1.shortest_distance(&r2), 0.8944271909999159);
        assert_eq!(r2.shortest_distance(&r1), 0.8944271909999159);
    }

    #[test]
    fn test_shortest_parallel2() {
        let r1 = Ray::new(Vec3::new(-1., 0., 0.), Vec3::new(2., 0., 0.));
        let r2 = Ray::new(Vec3::new(-2., 3., 0.), Vec3::new(2., 0., 0.));
        assert_eq!(r1.shortest_distance(&r2), 3.);
        assert_eq!(r2.shortest_distance(&r1), 3.);
    }

    #[test]
    fn test_shortest_intersecting() {
        let r1 = Ray::new(Vec3::new(-1., 0., 0.), Vec3::new(2., 0., 0.));
        let r2 = Ray::new(Vec3::new(0., 0., -1.), Vec3::new(0., 0., 2.));
        assert_eq!(r1.shortest_distance(&r2), 0.);
        assert_eq!(r2.shortest_distance(&r1), 0.);
    }

    #[test]
    fn test_shortest_same() {
        let r1 = Ray::new(Vec3::new(-1., 0., 0.), Vec3::new(4., 2., 0.));
        assert_eq!(r1.shortest_distance(&r1), 0.);
        assert_eq!(r1.shortest_distance(&r1), 0.);
    }

    #[test]
    fn test_shortest_xxx() {
        let r1 = Ray::new(
            Vec3::new(395.8288, 170.6440, 112.1048),
            Vec3::new(-38.2351, 383.3560, 77.8286),
        );
        let r2 = Ray::new(
            Vec3::new(-3.4878, -0.0001, -95.4594),
            Vec3::new(629.3250, -0.0001, -95.4594),
        );
        assert_eq!(r1.shortest_distance(&r2), 229.4765553708466);
        assert_eq!(r2.shortest_distance(&r1), 229.4765553708466);
    }
}
