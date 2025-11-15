use crate::geo::vec3::Vec3;
use crate::post::PostProcessor;
use std::error::Error;

#[derive(Clone, Default)]
/// A post-processor that uses Intel Open Image DeNoise on the image
pub struct OidnPostProcessor {
    width: u32,
    height: u32,
}

impl OidnPostProcessor {
    /// Create a new oidn post processor
    pub fn new() -> Self {
        OidnPostProcessor::default()
    }
}

#[cfg(feature = "oidn-postprocessor")]
impl PostProcessor for OidnPostProcessor {
    fn initialize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    fn post_process(
        &self,
        pixel_colors: &[Vec3],
        albedo_colors: &[Vec3],
        normal_colors: &[Vec3],
        num_samples: u32,
    ) -> Result<image::RgbImage, Box<dyn Error>> {
        let pixel_rgb = to_rgb_vec(pixel_colors, num_samples);
        let albedo_rgb = to_rgb_vec(albedo_colors, num_samples);
        let normal_rgb = to_rgb_vec(normal_colors, num_samples);
        let mut output = vec![0.0f32; pixel_rgb.len()];

        let device = oidn::Device::new();
        oidn::RayTracing::new(&device)
            .image_dimensions(self.width as usize, self.height as usize)
            .albedo_normal(&albedo_rgb, &normal_rgb)
            .srgb(true)
            .hdr(false)
            .clean_aux(true)
            .filter(&pixel_rgb, &mut output)
            .expect("Failed to apply Oidn post processing");

        if let Err(e) = device.get_error() {
            return Err(Box::new(simple_error::SimpleError::new(e.1)));
        }

        let mut img: image::RgbImage = image::ImageBuffer::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let i = ((y * self.width + x) * 3) as usize;
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
        _num_samples: u32,
    ) -> Result<Vec<Vec3>, Box<dyn Error>> {
        Err(Box::new(simple_error::SimpleError::new(
            "Intel Open Image DeNoise can not be used as an intermediate post processor",
        )))
    }

    fn needs_albedo_and_normal_colors(&self) -> bool {
        true
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }
}

#[cfg(not(feature = "oidn-postprocessor"))]
impl PostProcessor for OidnPostProcessor {
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

#[cfg(feature = "oidn-postprocessor")]
fn to_rgb_vec(vec: &[Vec3], num_samples: u32) -> Vec<f32> {
    vec.iter()
        .flat_map(|v| {
            let c = crate::util::rgb_color::to_float(*v, num_samples);
            vec![c.x as f32, c.y as f32, c.z as f32]
        })
        .collect()
}
