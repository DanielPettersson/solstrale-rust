//! Utilities for processing textures.

use std::error::Error;
use std::fmt;

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
        indexed_textures.sort_by(|a, b| b.2.cmp(&a.2));

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
