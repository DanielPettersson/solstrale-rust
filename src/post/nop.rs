use crate::geo::vec3::Vec3;
use crate::post::{PostProcessor, PostProcessors};
use std::error::Error;

#[derive(Clone)]
/// A post-processor that does nothing
pub struct NopPostProcessor();

impl NopPostProcessor {
    #![allow(clippy::new_ret_no_self)]
    /// Create a new nop post-processor
    pub fn new() -> PostProcessors {
        PostProcessors::from(NopPostProcessor())
    }
}

impl PostProcessor for NopPostProcessor {
    fn intermediate_post_process(
        &self,
        pixel_colors: &[Vec3],
        _albedo_colors: &[Vec3],
        _normal_colors: &[Vec3],
        _width: u32,
        _height: u32,
        _num_samples: u32,
    ) -> Result<Vec<Vec3>, Box<dyn Error>> {
        Ok(Vec::from(pixel_colors))
    }

    fn needs_albedo_and_normal_colors(&self) -> bool {
        false
    }
}
