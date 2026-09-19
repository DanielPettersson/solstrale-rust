//! Functions for converting between Vec3 and Rgb colors, and the sRGB transfer
//! functions that sit between encoded image data and the linear values the
//! renderer works in.
use crate::geo::vec3::Vec3;
use image::Rgb;

const COLOR_SCALE: f64 = 1.0 / 255.;

/// Converts rgb pixel to a Vec3 color, without any transfer function.
///
/// The raw `/ 255`. Use this only where the bytes are not a colour -- normal
/// maps, height maps -- and [`srgb_to_vec3`] where they are.
pub fn rgb_to_vec3(pixel: &Rgb<u8>) -> Vec3 {
    Vec3::new(
        pixel[0] as f64 * COLOR_SCALE,
        pixel[1] as f64 * COLOR_SCALE,
        pixel[2] as f64 * COLOR_SCALE,
    )
}

/// Converts an sRGB-encoded rgb pixel to a linear Vec3 color.
///
/// Computed in `f32`, the precision the shader decodes the atlas at, so the
/// CPU fallback and the GPU do not drift apart. An 8-bit input has far less
/// precision than `f32` carries anyway.
pub fn srgb_to_vec3(pixel: &Rgb<u8>) -> Vec3 {
    Vec3::new(
        srgb_to_linear(pixel[0] as f32 * (1. / 255.)) as f64,
        srgb_to_linear(pixel[1] as f32 * (1. / 255.)) as f64,
        srgb_to_linear(pixel[2] as f32 * (1. / 255.)) as f64,
    )
}

/// The sRGB EOTF: one encoded channel in `[0, 1]` to linear.
///
/// The exact piecewise function with the linear toe, not `pow(x, 2.2)`. The
/// toe is what the encoders that produced these images actually used, and it
/// is what makes the round trip with [`linear_to_srgb`] exact.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// The sRGB OETF: one linear channel to encoded `[0, 1]`. Inverse of
/// [`srgb_to_linear`].
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1. / 2.4) - 0.055
    }
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
    fn test_srgb_to_vec3() {
        let c = srgb_to_vec3(&Rgb([0, 128, 255]));
        assert_eq!(0., c.x);
        assert!((c.y - 0.215_861).abs() < 1e-5, "{}", c.y);
        assert_eq!(1., c.z);
    }

    /// Published sRGB linear values for a few bytes, including both sides of
    /// the toe/curve knee at 0.04045 (byte ~10.3).
    #[test]
    fn test_srgb_to_linear_table() {
        let expected = [
            (0u8, 0.),
            (1, 0.000_303_527),
            (10, 0.003_035_27),
            (11, 0.003_346_536),
            (128, 0.215_860_5),
            (255, 1.),
        ];
        for (byte, linear) in expected {
            let actual = srgb_to_linear(byte as f32 / 255.);
            assert!(
                (actual - linear).abs() < 1e-6,
                "byte {}: {} != {}",
                byte,
                actual,
                linear
            );
        }
    }

    #[test]
    fn test_transfer_pair_round_trips() {
        // Dense across the toe and the knee, sparse over the rest of the curve.
        let mut values: Vec<f32> = (0..=400).map(|i| i as f32 * 1e-5).collect();
        values.extend((0..=100).map(|i| i as f32 * 0.01));

        for v in values {
            let round_tripped = srgb_to_linear(linear_to_srgb(v));
            assert!(
                (round_tripped - v).abs() < 1e-6,
                "{} round tripped to {}",
                v,
                round_tripped
            );
        }
    }
}
