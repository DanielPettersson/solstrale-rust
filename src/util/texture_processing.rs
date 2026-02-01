//! Utilities for processing textures.

use image::{DynamicImage, RgbImage};
use image::imageops::FilterType;

/// Resizes a texture to the standard 1024x1024 resolution required by the GPU renderer.
pub fn standardize_texture(image: &RgbImage) -> RgbImage {
    let dyn_img = DynamicImage::ImageRgb8(image.clone());
    let resized = dyn_img.resize_exact(1024, 1024, FilterType::Lanczos3);
    resized.into_rgb8()
}
