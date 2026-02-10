//! Functions for converting between Vec3 and Rgb colors
use crate::geo::vec3::Vec3;
use image::Rgb;

const COLOR_SCALE: f64 = 1.0 / 255.;

/// Converts rgb pixel to a Vec3 color
pub fn rgb_to_vec3(pixel: &Rgb<u8>) -> Vec3 {
    Vec3::new(
        pixel[0] as f64 * COLOR_SCALE,
        pixel[1] as f64 * COLOR_SCALE,
        pixel[2] as f64 * COLOR_SCALE,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_vec3() {
        assert_eq!(
            Vec3::new(0., 0.39215686274509803, 1.),
            rgb_to_vec3(&Rgb([0, 100, 255]))
        )
    }
}
