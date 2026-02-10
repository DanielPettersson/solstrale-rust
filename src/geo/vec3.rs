//! 3-dimensional vector module, used for positions and also
//! colors in the crate
use derive_more::{Constructor, Display};
use std::ops::{Add, AddAssign, Div, Mul, Sub};

/// A 3 dimensional vector
#[derive(Copy, Clone, PartialEq, Debug, Default, Constructor, Display)]
#[display("{{ x: {:.4}, y: {:.4}, z: {:.4} }}", x, y, z)]
pub struct Vec3 {
    /// x position
    pub x: f64,
    /// y position
    pub y: f64,
    /// z position
    pub z: f64,
}

/// A value that is so small as to be almost zero
pub const ALMOST_ZERO: f64 = 1e-8;

/// A value that is so small as to be almost zero
pub const ALMOST_ZERO_F32: f32 = 1e-6;

/// Vector at exactly zero in all axis
pub const ZERO_VECTOR: Vec3 = Vec3 {
    x: 0.,
    y: 0.,
    z: 0.,
};

/// Vector at exactly one in all axis
pub const ONE_VECTOR: Vec3 = Vec3 {
    x: 1.,
    y: 1.,
    z: 1.,
};

/// A unit vector pointing towards positive Y
pub const UNIT_Y: Vec3 = Vec3 {
    x: 0.,
    y: 1.,
    z: 0.,
};

impl Add for Vec3 {
    type Output = Vec3;

    /// returns a Vec3 that has all values added with corresponding value in given Vec3
    /// # Examples:
    /// ```
    /// # use solstrale::geo::vec3::Vec3;
    /// let res = Vec3::new(1., 2., 3.) + Vec3::new(4., 5., 6.);
    /// assert_eq!(Vec3::new(5., 7., 9.), res)
    /// ```
    fn add(self, v: Self) -> Self::Output {
        Vec3 {
            x: self.x + v.x,
            y: self.y + v.y,
            z: self.z + v.z,
        }
    }
}

impl Add<f64> for Vec3 {
    type Output = Vec3;

    /// returns a Vec3 that has all values added by scalar
    /// # Examples:
    /// ```
    /// # use solstrale::geo::vec3::Vec3;
    /// let res = Vec3::new(1., 2., 3.) + 0.5;
    /// assert_eq!(Vec3::new(1.5, 2.5, 3.5), res)
    /// ```
    fn add(self, v: f64) -> Self::Output {
        Vec3 {
            x: self.x + v,
            y: self.y + v,
            z: self.z + v,
        }
    }
}

impl AddAssign for Vec3 {
    /// returns a Vec3 that has all values added with corresponding value in given Vec3
    /// # Examples:
    /// ```
    /// # use solstrale::geo::vec3::Vec3;
    /// let mut res = Vec3::new(1., 2., 3.);
    /// res += Vec3::new(4., 5., 6.);
    /// assert_eq!(Vec3::new(5., 7., 9.), res)
    /// ```
    fn add_assign(&mut self, v: Self) {
        *self = Self {
            x: self.x + v.x,
            y: self.y + v.y,
            z: self.z + v.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Vec3;

    /// returns a Vec3 that has all values subtracted by corresponding value in given Vec3
    /// # Examples:
    /// ```
    /// # use solstrale::geo::vec3::Vec3;
    /// let res = Vec3::new(1., 2., 3.) - Vec3::new(6., 5., 4.);
    /// assert_eq!(Vec3::new(-5., -3., -1.), res)
    /// ```
    fn sub(self, v: Self) -> Self::Output {
        Vec3 {
            x: self.x - v.x,
            y: self.y - v.y,
            z: self.z - v.z,
        }
    }
}

impl Sub<f64> for Vec3 {
    type Output = Vec3;

    /// returns a Vec3 that has all values subtracted by scalar
    /// # Examples:
    /// ```
    /// # use solstrale::geo::vec3::Vec3;
    /// let res = Vec3::new(1., 2., 3.) - 0.5;
    /// assert_eq!(Vec3::new(0.5, 1.5, 2.5), res)
    /// ```
    fn sub(self, v: f64) -> Self::Output {
        Vec3 {
            x: self.x - v,
            y: self.y - v,
            z: self.z - v,
        }
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Vec3;

    /// returns a Vec3 that has all values multiplied with corresponding value in given Vec3
    /// # Examples:
    /// ```
    /// # use solstrale::geo::vec3::Vec3;
    /// let res = Vec3::new(1., 2., 3.) * Vec3::new(4., 5., 6.);
    /// assert_eq!(Vec3::new(4., 10., 18.), res)
    /// ```
    fn mul(self, v: Self) -> Self::Output {
        Vec3 {
            x: self.x * v.x,
            y: self.y * v.y,
            z: self.z * v.z,
        }
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;

    /// returns a Vec3 that has all values multiplied with given scalar
    /// # Examples:
    /// ```
    /// # use solstrale::geo::vec3::Vec3;
    /// let res = Vec3::new(1., 2., 3.) * 2.;
    /// assert_eq!(Vec3::new(2., 4., 6.), res)
    /// ```
    fn mul(self, t: f64) -> Self::Output {
        Vec3 {
            x: self.x * t,
            y: self.y * t,
            z: self.z * t,
        }
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;

    /// returns a Vec3 that has all values multiplied with given scalar
    /// # Examples:
    /// ```
    /// # use solstrale::geo::vec3::Vec3;
    /// let res = Vec3::new(1., 2., 3.) * 2.;
    /// assert_eq!(Vec3::new(2., 4., 6.), res)
    /// ```
    fn mul(self, t: f32) -> Self::Output {
        Vec3 {
            x: self.x * t as f64,
            y: self.y * t as f64,
            z: self.z * t as f64,
        }
    }
}

impl Div<Vec3> for Vec3 {
    type Output = Vec3;

    /// returns a Vec3 that has all values multiplied with corresponding value in given Vec3
    /// # Examples:
    /// ```
    /// # use solstrale::geo::vec3::Vec3;
    /// let res = Vec3::new(1., 2., 3.) / Vec3::new(5., 4., 3.);
    /// assert_eq!(Vec3::new(0.2, 0.5, 1.), res)
    /// ```
    fn div(self, v: Self) -> Self::Output {
        Vec3 {
            x: self.x / v.x,
            y: self.y / v.y,
            z: self.z / v.z,
        }
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;

    /// returns a Vec3 that has all values multiplied with given scalar
    /// # Examples:
    /// ```
    /// # use solstrale::geo::vec3::Vec3;
    /// let res = Vec3::new(1., 2., 3.) / 2.;
    /// assert_eq!(Vec3::new(0.5, 1., 1.5), res)
    /// ```
    fn div(self, t: f64) -> Self::Output {
        Vec3 {
            x: self.x / t,
            y: self.y / t,
            z: self.z / t,
        }
    }
}

impl From<Vec3> for [f32; 4] {
    fn from(value: Vec3) -> Self {
        [value.x as f32, value.y as f32, value.z as f32, 0.0]
    }
}

impl From<&Vec3> for [f32; 4] {
    fn from(value: &Vec3) -> Self {
        [value.x as f32, value.y as f32, value.z as f32, 0.0]
    }
}

impl From<&[f32; 4]> for Vec3 {
    fn from(value: &[f32; 4]) -> Self {
        Vec3::new(value[0] as f64, value[1] as f64, value[2] as f64)
    }
}

impl Vec3 {
    /// returns a Vec3 that has all values negated
    /// # Examples:
    /// ```
    /// # use solstrale::geo::vec3::Vec3;
    /// let res = Vec3::new(1., 2., 3.).neg();
    /// assert_eq!(Vec3::new(-1., -2., -3.), res)
    /// ```
    pub fn neg(&self) -> Vec3 {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// returns the dot product with given Vec3
    /// # Examples:
    /// ```
    /// # use solstrale::geo::vec3::Vec3;
    /// let res = Vec3::new(1., 2., 3.).dot(Vec3::new(4., 5., 6.));
    /// assert_eq!(32., res)
    /// ```
    pub fn dot(&self, v: Vec3) -> f64 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    /// returns the cross product with given Vec3
    /// # Examples:
    /// ```
    /// # use solstrale::geo::vec3::Vec3;
    /// let res = Vec3::new(2., 3., 4.).cross(Vec3::new(5., 6., 7.));
    /// assert_eq!(Vec3::new(-3., 6., -3.), res)
    /// ```
    pub fn cross(&self, v: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * v.z - self.z * v.y,
            y: self.z * v.x - self.x * v.z,
            z: self.x * v.y - self.y * v.x,
        }
    }

    /// return the squared length of the vector
    /// # Examples:
    /// ```
    /// # use solstrale::geo::vec3::Vec3;
    /// let v = Vec3::new(1., 2., 3.);
    /// assert_eq!(14., v.length_squared())
    /// ```
    pub fn length_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// return the length of the vector
    /// # Examples:
    /// ```
    /// # use solstrale::geo::vec3::Vec3;
    /// let v = Vec3::new(0., 3., 4.);
    /// assert_eq!(5., v.length())
    /// ```
    pub fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    /// returns the vector but sized to a length of 1
    pub fn unit(&self) -> Vec3 {
        *self / self.length()
    }

    /// Checks if the vectors length is near zero
    /// # Examples:
    /// ```
    /// # use solstrale::geo::vec3::{Vec3, ZERO_VECTOR};
    /// let v = Vec3::new(1., 2., 3.);
    /// assert!(!v.near_zero());
    /// assert!(ZERO_VECTOR.near_zero());
    /// ```
    pub fn near_zero(&self) -> bool {
        self.x.abs() < ALMOST_ZERO && self.y.abs() < ALMOST_ZERO && self.z.abs() < ALMOST_ZERO
    }

    /// returns the reflection vector given the normal n
    /// # Examples:
    /// ```
    /// # use solstrale::geo::vec3::Vec3;
    /// let v = Vec3::new(0., 3., 4.);
    /// assert_eq!(Vec3::new(0., -3., 4.), v.reflect(Vec3::new(0., 1., 0.)));
    /// assert_eq!(Vec3::new(0., 3., -4.), v.reflect(Vec3::new(0., 0., 1.)))
    /// ```
    pub fn reflect(&self, n: Vec3) -> Vec3 {
        self.sub(n * (self.dot(n) * 2.))
    }

    /// returns the refraction vector given the normal n and the index of refraction of the material
    /// # Examples:
    /// ```
    /// # use solstrale::geo::vec3::Vec3;
    /// let v = Vec3::new(-3., -3., 0.).unit();
    /// let ret = v.refract(Vec3::new(0., 1., 0.), 1.);
    /// assert!((ret - v).near_zero());
    /// ```
    pub fn refract(&self, n: Vec3, index_of_refraction: f64) -> Vec3 {
        let cos_theta = self.neg().dot(n).min(1.);
        let r_out_perpendicular = (n * cos_theta + *self) * index_of_refraction;
        let r_out_parallel = n * (-(1. - r_out_perpendicular.length_squared()).abs().sqrt());
        r_out_perpendicular + r_out_parallel
    }

    /// Returns value of a numbered axis, where x is 0, y is 1 and other is z
    /// # Examples:
    /// ```
    /// # use solstrale::geo::vec3::Vec3;
    /// let v = Vec3::new(1., 2., 3.);
    /// assert_eq!(1., v.axis(0));
    /// assert_eq!(2., v.axis(1));
    /// assert_eq!(3., v.axis(2));
    /// ```
    ///
    pub fn axis(&self, a: u8) -> f64 {
        if a == 0 {
            self.x
        } else if a == 1 {
            self.y
        } else {
            self.z
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from() {
        let array: [f32; 4] = Vec3::new(1., 2., 3.).into();
        assert_eq!(array, [1., 2., 3., 0.]);
    }
}
