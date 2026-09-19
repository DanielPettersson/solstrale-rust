//! Utilities for processing textures.

use image::RgbImage;
use image::imageops::{self, FilterType};
use rayon::prelude::*;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

/// Represents the position and dimensions of a texture within the atlas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextureRect {
    /// X coordinate in the atlas.
    pub x: u32,
    /// Y coordinate in the atlas.
    pub y: u32,
    /// Width of the texture.
    pub width: u32,
    /// Height of the texture.
    pub height: u32,
    /// Index of the original texture in the input list.
    pub original_index: usize,
}

/// The result of the packing process, containing the atlas dimensions and texture placements.
pub struct AtlasLayout {
    /// Width of the atlas.
    pub width: u32,
    /// Height of the atlas.
    pub height: u32,
    /// List of texture placements.
    pub placements: Vec<TextureRect>,
}

/// Utility for packing multiple textures into a single atlas.
pub struct TexturePacker {
    max_width: u32,
    max_height: u32,
}

/// Error returned when textures cannot be packed into the atlas.
#[derive(Debug)]
pub struct PackingError;

impl fmt::Display for PackingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Textures could not be packed into the atlas")
    }
}

impl Error for PackingError {}

impl TexturePacker {
    /// Creates a new TexturePacker with the specified maximum dimensions.
    pub fn new(max_width: u32, max_height: u32) -> Self {
        Self {
            max_width,
            max_height,
        }
    }

    /// Packs the given images, halving every one of them until the set fits.
    ///
    /// Returns the layout together with the images at the size they were placed
    /// at, which is the original `Arc` whenever nothing had to shrink. A single
    /// 8192x8192 map already fills an 8192 atlas, so a scene carrying a handful
    /// of them can only be rendered with them scaled down.
    pub fn pack_images(
        &self,
        images: &[Arc<RgbImage>],
    ) -> Result<(AtlasLayout, Vec<Arc<RgbImage>>), PackingError> {
        let original: Vec<(u32, u32)> = images.iter().map(|i| (i.width(), i.height())).collect();
        let mut divisor = 1;

        loop {
            let dims: Vec<(u32, u32)> = original
                .iter()
                .map(|&(w, h)| ((w / divisor).max(1), (h / divisor).max(1)))
                .collect();

            if let Ok(layout) = self.pack(&dims) {
                let placed = if divisor == 1 {
                    images.to_vec()
                } else {
                    println!(
                        "Scaling {} textures down by {divisor} to fit the {}x{} atlas",
                        images.len(),
                        self.max_width,
                        self.max_height
                    );
                    images
                        .par_iter()
                        .zip(&dims)
                        .map(|(img, &(w, h))| {
                            Arc::new(imageops::resize(img.as_ref(), w, h, FilterType::Triangle))
                        })
                        .collect()
                };
                return Ok((layout, placed));
            }

            // Every image is a single pixel and they still do not fit.
            if dims.iter().all(|&(w, h)| w == 1 && h == 1) {
                return Err(PackingError);
            }
            divisor *= 2;
        }
    }

    /// Packs the given textures into an atlas.
    ///
    /// Returns the layout of the packed textures or an error if they don't fit.
    pub fn pack(&self, textures: &[(u32, u32)]) -> Result<AtlasLayout, PackingError> {
        let mut indexed_textures: Vec<(usize, u32, u32)> = textures
            .iter()
            .enumerate()
            .map(|(i, &(w, h))| (i, w, h))
            .collect();

        // Sort by height descending for better shelf packing efficiency
        indexed_textures.sort_by_key(|a| std::cmp::Reverse(a.2));

        let mut placements = Vec::new();
        let mut shelves: Vec<(u32, u32, u32)> = Vec::new(); // y, current_x, height

        // Initialize with first shelf
        let mut current_y = 0;

        for (original_index, width, height) in indexed_textures {
            if width > self.max_width || height > self.max_height {
                return Err(PackingError);
            }

            let mut placed = false;

            // Try to fit in existing shelves
            for shelf in &mut shelves {
                if shelf.1 + width <= self.max_width && height <= shelf.2 {
                    placements.push(TextureRect {
                        x: shelf.1,
                        y: shelf.0,
                        width,
                        height,
                        original_index,
                    });
                    shelf.1 += width;
                    placed = true;
                    break;
                }
            }

            if !placed {
                // Start a new shelf
                if current_y + height <= self.max_height {
                    placements.push(TextureRect {
                        x: 0,
                        y: current_y,
                        width,
                        height,
                        original_index,
                    });
                    shelves.push((current_y, width, height));
                    current_y += height;
                    placed = true;
                }
            }

            if !placed {
                return Err(PackingError);
            }
        }

        // Restore original order
        placements.sort_by_key(|p| p.original_index);

        // Calculate actual used bounds
        let mut used_width = 0;
        let mut used_height = 0;
        for p in &placements {
            used_width = used_width.max(p.x + p.width);
            used_height = used_height.max(p.y + p.height);
        }

        // Align width to 64 pixels (256 bytes for RGBA8) to satisfy WebGPU requirements
        let align = 64;
        let aligned_width = used_width.div_ceil(align) * align;

        Ok(AtlasLayout {
            width: aligned_width.max(1),
            height: used_height.max(1),
            placements,
        })
    }
}
