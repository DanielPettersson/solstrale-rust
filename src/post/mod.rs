//! Post processors for applying effects to the raw rendered image

mod bloom;
mod nop;
mod saturation;

use std::error::Error;

use enum_dispatch::enum_dispatch;

use crate::geo::vec3::Vec3;
pub use crate::post::bloom::BloomPostProcessor;
pub use crate::post::nop::NopPostProcessor;
pub use crate::post::saturation::SaturationPostProcessor;

/// Responsible for taking the rendered image and transforming it
#[enum_dispatch]
pub trait PostProcessor {
    /// Does post-construct initialization for the post-processor when with and height are known
    fn initialize(&mut self, width: u32, height: u32);

    /// Execute final postprocessing of the rendered image
    fn post_process(
        &self,
        pixel_colors: &[Vec3],
        num_samples: u32,
    ) -> Result<image::RgbImage, Box<dyn Error>> {
        let pixel_colors = self.intermediate_post_process(pixel_colors, num_samples)?;
        Ok(pixel_colors_to_rgb_image(
            &pixel_colors,
            self.width(),
            self.height(),
            num_samples,
        ))
    }

    /// Execute intermediate postprocessing of the rendered image
    fn intermediate_post_process(
        &self,
        pixel_colors: &[Vec3],
        num_samples: u32,
    ) -> Result<Vec<Vec3>, Box<dyn Error>>;

    /// Returns the width of the image
    fn width(&self) -> u32;

    /// Returns the height of the image
    fn height(&self) -> u32;
}

#[enum_dispatch(PostProcessor)]
#[derive(Clone)]
/// An enum of available post-processors
pub enum PostProcessors {
    /// [`PostProcessor`] of type [`BloomPostProcessor`]
    BloomPostProcessor,
    /// [`PostProcessor`] of type [`SaturationPostProcessor`]
    SaturationPostProcessor,
    /// [`PostProcessor`] of type [`NopPostProcessor`]
    NopPostProcessor,
}

fn pixel_colors_to_rgb_image(
    pixel_colors: &[Vec3],
    width: u32,
    height: u32,
    num_samples: u32,
) -> image::RgbImage {
    let mut img: image::RgbImage = image::ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            img.put_pixel(
                x,
                y,
                crate::util::rgb_color::to_rgb_color(pixel_colors[i], num_samples),
            )
        }
    }

    img
}
