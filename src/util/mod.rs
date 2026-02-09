//! Various small helpers used by the raytracer

use std::f64::consts::PI;

pub mod gaussian;
pub mod height_map;
pub mod interval;
pub mod rgb_color;
pub mod texture_processing;
pub mod wgpu_util;

/// Converts an angle in degrees to radians
pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * (PI / 180.)
}

#[cfg(test)]
mod tests {
    use crate::util::degrees_to_radians;
    use std::f64::consts::PI;

    #[test]
    fn test_degrees_to_radians() {
        let mut r = degrees_to_radians(0.);
        assert_eq!(r, 0.);

        r = degrees_to_radians(180.);
        assert_eq!(r, PI);

        r = degrees_to_radians(360.);
        assert_eq!(r, PI * 2.);

        r = degrees_to_radians(-180.);
        assert_eq!(r, -PI);
    }
}
