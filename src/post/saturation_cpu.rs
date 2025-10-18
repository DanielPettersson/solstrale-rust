//! Post-processor for applying saturation
#![cfg(not(feature = "gpu"))]

use crate::geo::vec3::Vec3;
use crate::post::{PostProcessor, PostProcessors};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::error::Error;

#[derive(Clone)]
/// Applies a saturation effect on the pixel colors
pub struct SaturationPostProcessor {
    width: u32,
    height: u32,
    saturation_factor: f64,
}

impl SaturationPostProcessor {
    #![allow(clippy::new_ret_no_self)]
    /// Create a new saturation post-processor
    /// # Arguments
    /// * `saturation_factor` Saturation of the image. From -1 (black and white) to 1 (fully saturated)
    pub fn new(saturation_factor: f64) -> Result<PostProcessors, simple_error::SimpleError> {
        if !(-1. ..=1.).contains(&saturation_factor) {
            return Err(simple_error::SimpleError::new(
                "saturation_factor must be between -1 and 1",
            ));
        }

        Ok(PostProcessors::from(SaturationPostProcessor {
            width: 0,
            height: 0,
            saturation_factor,
        }))
    }
}

impl PostProcessor for SaturationPostProcessor {
    fn initialize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    fn intermediate_post_process(
        &self,
        pixel_colors: &[Vec3],
        _albedo_colors: &[Vec3],
        _normal_colors: &[Vec3],
        _num_samples: u32,
    ) -> Result<Vec<Vec3>, Box<dyn Error>> {

        let ret = pixel_colors.par_iter().map(|p| {
            let gray = 0.2989 * p.x + 0.587 * p.y + 0.114 * p.z;
            let g = -gray * self.saturation_factor;
            let gg = 1. + self.saturation_factor;
            Vec3::new(
                g + p.x * gg,
                g + p.y * gg,
                g + p.z * gg
            )
        }).collect();

        Ok(ret)
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }
}
