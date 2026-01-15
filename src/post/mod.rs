//! Post processors for applying effects to the raw rendered image

mod bloom_cpu;
mod bloom_gpu;
mod nop;
mod oidn;
mod saturation_gpu;

use std::error::Error;

use enum_dispatch::enum_dispatch;

use crate::geo::vec3::Vec3;
#[cfg(not(feature = "gpu"))]
pub use crate::post::bloom_cpu::BloomPostProcessor;
#[cfg(feature = "gpu")]
pub use crate::post::bloom_gpu::BloomPostProcessor;
pub use crate::post::nop::NopPostProcessor;
pub use crate::post::oidn::OidnPostProcessor;
pub use crate::post::saturation_gpu::SaturationPostProcessor;

/// Responsible for taking the rendered image and transforming it
#[enum_dispatch]
pub trait PostProcessor {

    /// Do post-construct initialization for the post-processor when with and height it known
    fn initialize(&mut self, width: u32, height: u32);

    /// Execute final postprocessing of the rendered image
    fn post_process(
        &self,
        pixel_colors: &[Vec3],
        albedo_colors: &[Vec3],
        normal_colors: &[Vec3],
        num_samples: u32,
    ) -> Result<image::RgbImage, Box<dyn Error>> {
        let pixel_colors = self.intermediate_post_process(
            pixel_colors,
            albedo_colors,
            normal_colors,
            num_samples,
        )?;
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
        albedo_colors: &[Vec3],
        normal_colors: &[Vec3],
        num_samples: u32,
    ) -> Result<Vec<Vec3>, Box<dyn Error>>;

    /// Does this post-processor need albedo or normal colors?
    fn needs_albedo_and_normal_colors(&self) -> bool {
        false
    }

    /// Returns the width of the image
    fn width(&self) -> u32;

    /// Returns the height of the image
    fn height(&self) -> u32;
}

#[enum_dispatch(PostProcessor)]
#[derive(Clone)]
/// An enum of available post-processors
pub enum PostProcessors {
    /// [`PostProcessor`] of type [`OidnPostProcessor`]
    OidnPostProcessor,
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
