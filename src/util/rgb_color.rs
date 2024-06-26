//! Functions for converting between Vec3 and Rgb colors
use crate::geo::vec3::Vec3;
use crate::util::interval::Interval;
use image::Rgb;

const COLOR_INTENSITY_INTERVAL: Interval = Interval {
    min: -0.999,
    max: 0.999,
};

const COLOR_SCALE: f64 = 1.0 / 255.;

/// Convert a color and a given number of samples used to generate that color to an rgb color
pub fn to_rgb_color(col: Vec3, samples_per_pixel: u32) -> Rgb<u8> {
    let c = to_float(col, samples_per_pixel);
    Rgb([(256. * c.x) as u8, (256. * c.y) as u8, (256. * c.z) as u8])
}

/// Converts a color in a Vec3 that is the sum of a given of amounts of samples
/// to a float color. Applies gamma correction to the output color.
pub fn to_float(col: Vec3, samples_per_pixel: u32) -> Vec3 {
    // Divide the color by the number of samples
    // and gamma-correct for gamma=2.0
    let scale = 1.0 / samples_per_pixel as f64;
    let r = (scale * col.x).sqrt();
    let g = (scale * col.y).sqrt();
    let b = (scale * col.z).sqrt();

    Vec3::new(
        COLOR_INTENSITY_INTERVAL.clamp(r),
        COLOR_INTENSITY_INTERVAL.clamp(g),
        COLOR_INTENSITY_INTERVAL.clamp(b),
    )
}

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

    #[test]
    fn test_to_rgb_color() {
        assert_eq!(Rgb([0, 140, 255]), to_rgb_color(Vec3::new(0., 0.3, 1.), 1));
        assert_eq!(Rgb([0, 99, 181]), to_rgb_color(Vec3::new(0., 0.3, 1.), 2));
    }
}
