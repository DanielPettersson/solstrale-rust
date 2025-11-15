use crate::geo::vec3::Vec3;
use crate::post::PostProcessor;
use std::error::Error;

#[derive(Clone, Default)]
/// A post-processor that does nothing
pub struct NopPostProcessor {
    width: u32,
    height: u32,
}

impl NopPostProcessor {
    /// Create a new nop post-processor
    pub fn new() -> Self {
        NopPostProcessor::default()
    }
}

impl PostProcessor for NopPostProcessor {
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
        Ok(Vec::from(pixel_colors))
    }

    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
}
