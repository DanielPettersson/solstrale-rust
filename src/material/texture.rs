//! Contains textures to be used by materials
use std::error::Error;
use std::sync::Arc;

use enum_dispatch::enum_dispatch;
use image::ImageReader;
use image::RgbImage;
use simple_error::SimpleError;

use crate::geo::Uv;
use crate::geo::vec3::Vec3;
use crate::material::texture::BumpMap::{Height, Normal};
use crate::util::height_map;
use crate::util::rgb_color::rgb_to_vec3;

/// Describes the color of a material.
/// The color can vary by the uv coordinates of the hittable
#[enum_dispatch]
pub trait Texture {
    /// Return the color of the texture at a given hit
    fn color(&self, uv: Uv) -> Vec3;
}

#[enum_dispatch(Texture)]
#[derive(Debug, Clone)]
/// An enum of available textures types
pub enum Textures {
    /// [`Texture`] of the type [`SolidColor`]
    SolidColor,
    /// [`Texture`] of the type [`ImageMap`]
    ImageMap,
}

/// The variants of bump maps supported.
pub enum BumpMap {
    /// Each pixel in the image describes the normal vector directly
    Normal(RgbImage),
    /// Each pixel in the image describes the relative height in the surface
    Height(RgbImage),
}

/// Load a bump map image texture and detect if it is a normal or height map
fn load_bump_map(path: &str) -> Result<BumpMap, Box<dyn Error>> {
    let mut reader = ImageReader::open(path).map_err(|err| {
        SimpleError::new(format!("Failed to open bump texture {}: {}", path, err))
    })?;
    reader.no_limits();
    reader = reader.with_guessed_format().map_err(|err| {
        SimpleError::new(format!("Failed to load bump texture {}: {}", path, err))
    })?;
    let image = reader
        .decode()
        .map_err(|err| {
            SimpleError::new(format!("Failed to decode bump texture {}: {}", path, err))
        })?
        .into_rgb8();

    let mut num_normal = 0;
    let mut num_height = 0;

    for pixel in image.pixels() {
        let p = rgb_to_vec3(pixel);
        if (p.length() - 1.).abs() < 0.05 {
            num_normal += 1;
        }
        if (p.x - p.y).abs() < 0.05 && (p.y - p.z).abs() < 0.05 {
            num_height += 1;
        }
    }

    if num_height > num_normal {
        Ok(Height(image))
    } else {
        Ok(Normal(image))
    }
}

/// Load a normal map texture. Source image can either be a normal or height map
pub fn load_normal_texture(path: &str) -> Result<ImageMap, Box<dyn Error>> {
    match load_bump_map(path)? {
        Normal(n) => Ok(ImageMap::new(Arc::new(n))),
        Height(h) => {
            let n = height_map::to_normal_map(h);
            Ok(ImageMap::new(Arc::new(n)))
        }
    }
}

/// A texture with just the same color everywhere
#[derive(Clone, Debug)]
pub struct SolidColor(Vec3);

impl SolidColor {
    /// Create a new solid color texture
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        SolidColor::new_from_vec3(Vec3::new(r, g, b))
    }
    /// Create a new solid color texture from an array
    /// where colors are in the order r, g, b
    pub fn new_from_f32_array(c: [f32; 3]) -> Self {
        SolidColor::new(c[0] as f64, c[1] as f64, c[2] as f64)
    }
    /// Create a new solid color texture from a [`Vec3`]
    pub fn new_from_vec3(color: Vec3) -> Self {
        SolidColor(color)
    }
}

impl Texture for SolidColor {
    fn color(&self, _: Uv) -> Vec3 {
        self.0
    }
}

/// Texture that uses image data for color by loading the image from the path
#[derive(Clone, Debug)]
pub struct ImageMap {
    image: Arc<RgbImage>,
    max_x: f32,
    max_y: f32,
}

impl ImageMap {
    /// Creates a new image texture from a file path
    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let mut reader = ImageReader::open(path).map_err(|err| {
            SimpleError::new(format!("Failed to open image texture {}: {}", path, err))
        })?;
        reader.no_limits();
        reader = reader.with_guessed_format().map_err(|err| {
            SimpleError::new(format!("Failed to load image texture {}: {}", path, err))
        })?;
        let image = reader
            .decode()
            .map_err(|err| {
                SimpleError::new(format!("Failed to decode image texture {}: {}", path, err))
            })?
            .into_rgb8();

        Ok(Self::new(Arc::new(image)))
    }

    /// Creates a texture that uses image data for color
    pub fn new(image: Arc<RgbImage>) -> Self {
        let w = image.width();
        let h = image.height();
        ImageMap {
            image,
            max_x: w as f32 - 1.,
            max_y: h as f32 - 1.,
        }
    }

    /// Returns the underlying image
    pub fn get_image(&self) -> Arc<RgbImage> {
        self.image.clone()
    }
}

impl Texture for ImageMap {
    /// Returns the color in the image data that corresponds to the UV coordinate of the hittable
    /// If UV coordinates from hit record is <0 or >1 texture wraps
    fn color(&self, uv: Uv) -> Vec3 {
        let u = uv.u.abs() % 1.;
        let v = 1. - uv.v.abs() % 1.;

        let x = u * self.max_x;
        let y = v * self.max_y;

        let pixel = self.image.get_pixel(x as u32, y as u32);
        rgb_to_vec3(pixel)
    }
}

#[cfg(test)]
mod tests {
    use crate::material::texture::{BumpMap, load_bump_map};

    #[test]
    fn test_load_normal_bump_map() {
        let res = load_bump_map("resources/textures/wall_n.png").unwrap();
        match res {
            BumpMap::Normal(n) => assert!(n.width() > 0 && n.height() > 0),
            BumpMap::Height(_) => panic!("Should not be a height map"),
        }
    }

    #[test]
    fn test_load_height_bump_map() {
        let res = load_bump_map("resources/textures/sponza-h.jpg").unwrap();
        match res {
            BumpMap::Normal(_) => panic!("Should not be a height map"),
            BumpMap::Height(n) => assert!(n.width() > 0 && n.height() > 0),
        }
    }
}
