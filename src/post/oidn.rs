use crate::geo::vec3::Vec3;
use crate::post::{PostProcessor, PostProcessors};
use std::error::Error;

#[derive(Clone)]
/// A post-processor that uses Intel Open Image DeNoise on the image
pub struct OidnPostProcessor();

impl OidnPostProcessor {
    #![allow(clippy::new_ret_no_self)]
    /// Create a new oidn post processor
    pub fn new() -> PostProcessors {
        PostProcessors::from(OidnPostProcessor())
    }
}

#[cfg(feature = "oidn-postprocessor")]
impl PostProcessor for OidnPostProcessor {
    fn post_process(
        &self,
        pixel_colors: &[Vec3],
        albedo_colors: &[Vec3],
        normal_colors: &[Vec3],
        width: u32,
        height: u32,
        num_samples: u32,
    ) -> Result<image::RgbImage, Box<dyn Error>> {
        let pixel_rgb = to_rgb_vec(pixel_colors, num_samples);
        let albedo_rgb = to_rgb_vec(albedo_colors, num_samples);
        let normal_rgb = to_rgb_vec(normal_colors, num_samples);
        let mut output = vec![0.0f32; pixel_rgb.len()];

        let device = oidn::Device::new();
        oidn::RayTracing::new(&device)
            .image_dimensions(width as usize, height as usize)
            .albedo_normal(&albedo_rgb, &normal_rgb)
            .srgb(true)
            .hdr(false)
            .clean_aux(true)
            .filter(&pixel_rgb, &mut output)
            .expect("Failed to apply Oidn post processing");

        if let Err(e) = device.get_error() {
            return Err(Box::new(simple_error::SimpleError::new(e.1)));
        }

        let mut img: image::RgbImage = image::ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let i = ((y * width + x) * 3) as usize;
                img.put_pixel(
                    x,
                    y,
                    image::Rgb([
                        (output[i] * 256.) as u8,
                        (output[i + 1] * 256.) as u8,
                        (output[i + 2] * 256.) as u8,
                    ]),
                );
            }
        }

        Ok(img)
    }

    fn intermediate_post_process(
        &self,
        _pixel_colors: &[Vec3],
        _albedo_colors: &[Vec3],
        _normal_colors: &[Vec3],
        _width: u32,
        _height: u32,
        _num_samples: u32,
    ) -> Result<Vec<Vec3>, Box<dyn Error>> {
        Err(Box::new(simple_error::SimpleError::new(
            "Intel Open Image DeNoise can not be used as an intermediate post processor",
        )))
    }

    fn needs_albedo_and_normal_colors(&self) -> bool {
        true
    }
}

#[cfg(not(feature = "oidn-postprocessor"))]
impl PostProcessor for OidnPostProcessor {
    fn post_process(
        &self,
        pixel_colors: &[Vec3],
        albedo_colors: &[Vec3],
        normal_colors: &[Vec3],
        width: u32,
        height: u32,
        num_samples: u32,
    ) -> Result<image::RgbImage, Box<dyn Error>> {
        crate::post::nop::NopPostProcessor::new().post_process(
            pixel_colors,
            albedo_colors,
            normal_colors,
            width,
            height,
            num_samples,
        )
    }

    fn intermediate_post_process(
        &self,
        pixel_colors: &[Vec3],
        albedo_colors: &[Vec3],
        normal_colors: &[Vec3],
        width: u32,
        height: u32,
        num_samples: u32,
    ) -> Result<Vec<Vec3>, Box<dyn Error>> {
        crate::post::nop::NopPostProcessor::new().intermediate_post_process(
            pixel_colors,
            albedo_colors,
            normal_colors,
            width,
            height,
            num_samples,
        )
    }

    fn needs_albedo_and_normal_colors(&self) -> bool {
        false
    }
}

#[cfg(feature = "oidn-postprocessor")]
fn to_rgb_vec(vec: &[Vec3], num_samples: u32) -> Vec<f32> {
    vec.iter()
        .flat_map(|v| {
            let c = crate::util::rgb_color::to_float(*v, num_samples);
            vec![c.x as f32, c.y as f32, c.z as f32]
        })
        .collect()
}
