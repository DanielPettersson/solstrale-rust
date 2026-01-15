//! Contains Gaussian-related functions

fn gaussian(x: f32, mean: f32, std_dev: f32) -> f32 {
    let a = (x - mean) / std_dev;
    (-0.5 * a * a).exp()
}

/// Creates a [`Vec`] containing weights distributed according to a gaussian distribution
/// The sum of all weights is ~1
#[allow(clippy::needless_range_loop)]
pub fn create_gaussian_blur_weights(kernel_size: usize, std_dev: f32) -> Vec<f32> {
    let mut gaussian_blur_weights = Vec::with_capacity(kernel_size);
    let kernel_mean = (kernel_size - 1) as f32 / 2.0;
    for i in 0..kernel_size {
        gaussian_blur_weights.push(gaussian(i as f32, kernel_mean, std_dev))
    }

    let weight_sum: f32 = gaussian_blur_weights.iter().sum();

    for i in 0..kernel_size {
        gaussian_blur_weights[i] /= weight_sum
    }

    gaussian_blur_weights
}

#[cfg(test)]
mod tests {
    use crate::geo::vec3::ALMOST_ZERO_F32;
    use crate::util::gaussian::create_gaussian_blur_weights;

    #[test]
    fn test_create_gaussian_blur_weights() {
        let weights = create_gaussian_blur_weights(5, 1.);
        assert_eq!(
            weights,
            vec![0.05448869, 0.24420136, 0.40261996, 0.24420136, 0.05448869]
        );

        let total_weight: f32 = weights.iter().sum();
        assert!((1. - total_weight).abs() < ALMOST_ZERO_F32);
    }
}
