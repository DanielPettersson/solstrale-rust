//! Defines a range between min and max inclusive
use derive_more::Constructor;
use std::ops::{Add, Sub};

/// Defines a range between min and max inclusive
#[derive(Copy, Clone, PartialEq, Debug, Default, Constructor)]
pub struct Interval {
    /// Minimum value of the interval
    pub min: f64,
    /// Maximum value of the interval
    pub max: f64,
}

/// Contains nothing
pub const EMPTY_INTERVAL: Interval = Interval {
    min: f64::INFINITY,
    max: f64::NEG_INFINITY,
};
/// Contains everything
pub const UNIVERSE_INTERVAL: Interval = Interval {
    min: f64::NEG_INFINITY,
    max: f64::INFINITY,
};
/// Default start interval for a ray
pub const RAY_INTERVAL: Interval = Interval {
    min: 0.001,
    max: f64::INFINITY,
};

/// creates a new interval that is the union of the two given.
/// If there is a gap between the intervals, that is included in the returned interval.
pub fn combine_intervals(a: Interval, b: Interval) -> Interval {
    Interval {
        min: a.min.min(b.min),
        max: a.max.max(b.max),
    }
}

impl Add<f64> for Interval {
    type Output = Interval;

    /// returns a new interval that is increased with given value.
    /// The returned interval has same size as original
    fn add(self, rhs: f64) -> Self::Output {
        Interval {
            min: self.min + rhs,
            max: self.max + rhs,
        }
    }
}

impl Sub<f64> for Interval {
    type Output = Interval;

    /// returns a new interval that is decreased with given value.
    /// The returned interval has same size as original
    fn sub(self, rhs: f64) -> Self::Output {
        Interval {
            min: self.min - rhs,
            max: self.max - rhs,
        }
    }
}

impl Interval {
    /// Checks if the interval contains a given value
    pub fn contains(&self, x: f64) -> bool {
        self.min <= x && x <= self.max
    }

    /// returns the given value clamped to the interval
    pub fn clamp(&self, x: f64) -> f64 {
        if x < self.min {
            return self.min;
        }
        if x > self.max {
            return self.max;
        }
        x
    }

    /// return the size of the interval
    pub fn size(&self) -> f64 {
        self.max - self.min
    }

    /// returns a new interval that is larger by given value delta.
    /// Delta is added equally to both sides of the interval
    pub fn expand(&self, delta: f64) -> Interval {
        let padding = delta / 2.;
        Interval {
            min: self.min - padding,
            max: self.max + padding,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::util::interval::{Interval, combine_intervals};

    #[test]
    fn test_combine_intervals() {
        let mut res = combine_intervals(Interval::new(0., 2.), Interval::new(1., 3.));
        assert_eq!(Interval::new(0., 3.), res);

        res = combine_intervals(Interval::new(0., 1.), Interval::new(2., 3.));
        assert_eq!(Interval::new(0., 3.), res);

        res = combine_intervals(Interval::new(3., 3.), Interval::new(-1., -1.));
        assert_eq!(Interval::new(-1., 3.), res);
    }

    #[test]
    fn test_contains() {
        let interval = Interval::new(-2., 2.);
        assert!(!interval.contains(-3.));
        assert!(interval.contains(-2.));
        assert!(interval.contains(2.));
        assert!(!interval.contains(3.));
    }

    #[test]
    fn test_clamp() {
        let interval = Interval::new(-2., 2.);
        assert_eq!(-2., interval.clamp(-3.));
        assert_eq!(-2., interval.clamp(-2.));
        assert_eq!(0., interval.clamp(-0.));
        assert_eq!(2., interval.clamp(2.));
        assert_eq!(2., interval.clamp(3.));
    }

    #[test]
    fn test_size() {
        assert_eq!(0., Interval::new(0., 0.).size());
        assert_eq!(2., Interval::new(-1., 1.).size());
        assert_eq!(-2., Interval::new(1., -1.).size());
    }

    #[test]
    fn test_expand() {
        let interval = Interval::new(-2., 2.);
        assert_eq!(Interval::new(-3., 3.), interval.expand(2.));
        assert_eq!(
            Interval {
                min: -3.5,
                max: 3.5
            },
            interval.expand(3.)
        );
        assert_eq!(Interval::new(-1., 1.), interval.expand(-2.));
    }

    #[test]
    fn test_sub_and_add() {
        let interval = Interval::new(-2., 2.);
        assert_eq!(Interval::new(0., 4.), interval + 2.);
        assert_eq!(Interval::new(1., 5.), interval + 3.);
        assert_eq!(Interval::new(-4., 0.), interval - 2.);
    }
}
