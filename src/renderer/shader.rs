use crate::geo::ray::Ray;
use crate::geo::vec3::{Vec3, ZERO_VECTOR};
use crate::hittable::hittable_pdf::HittablePdf;
use crate::material::HitRecord;
use crate::pdf::{mix_generate, mix_value};
use crate::renderer::Renderer;

/// shader calculates the color from a ray hitting a hittable object
pub trait Shader {
    fn shade(&self, renderer: &Renderer, rec: &HitRecord, ray: &Ray, depth: i32) -> Vec3;
}

/// A full raytracing shader
pub struct PathTracingShader {
    max_depth: i32,
}

impl Shader for PathTracingShader {
    /// Calculates the color using path tracing
    fn shade(&self, renderer: &Renderer, rec: &HitRecord, ray: &Ray, depth: i32) -> Vec3 {
        if depth >= self.max_depth {
            return ZERO_VECTOR;
        }

        let emitted_color = rec.material.emitted(rec);
        let scatter_res = rec.material.scatter(ray, rec);

        match scatter_res {
            None => emitted_color,
            Some(scatter_record) => match scatter_record.skip_pdf_ray {
                Some(skip_pdf_ray) => {
                    let (rc, _, _) = renderer.ray_color(skip_pdf_ray, depth + 1);
                    return scatter_record.attenuation * rc;
                }
                None => {
                    let light_pdf = HittablePdf::new(&renderer.lights, rec.hit_point);

                    let pdf_direction = mix_generate(&light_pdf, scatter_record.pdf.as_ref());
                    let scattered = Ray::new(rec.hit_point, pdf_direction, ray.time);
                    let pdf_val =
                        mix_value(&light_pdf, scatter_record.pdf.as_ref(), scattered.direction);
                    let scattering_pdf = rec.material.scattering_pdf(rec, &scattered);
                    let (rc, _, _) = renderer.ray_color(scattered, depth + 1);
                    let scatter_color = scatter_record.attenuation * scattering_pdf * rc / pdf_val;

                    filter_invalid_color_values(emitted_color + scatter_color)
                }
            },
        }
    }
}

fn filter_invalid_color_values(col: Vec3) -> Vec3 {
    Vec3::new(
        filter_color_value(col.x),
        filter_color_value(col.y),
        filter_color_value(col.z),
    )
}

fn filter_color_value(val: f64) -> f64 {
    if val.is_nan() {
        0.
    } else {
        // A subjectively chosen value that is a trade off between
        // color acne and suppressing intensity
        val.min(3.)
    }
}

/// Outputs flat color
pub struct AlbedoShader {}

impl AlbedoShader {
    pub fn new() -> AlbedoShader {
        AlbedoShader {}
    }
}

impl Shader for AlbedoShader {
    /// Calculates the color only attenuation color
    fn shade(&self, _: &Renderer, rec: &HitRecord, ray: &Ray, _: i32) -> Vec3 {
        match rec.material.scatter(ray, rec) {
            None => rec.material.emitted(rec),
            Some(scatter_record) => scatter_record.attenuation,
        }
    }
}

/// Outputs the normals of the ray hit point
pub struct NormalShader {}

impl NormalShader {
    pub fn new() -> NormalShader {
        NormalShader {}
    }
}

impl Shader for NormalShader {
    /// Calculates the color only using normal
    fn shade(&self, _: &Renderer, rec: &HitRecord, _: &Ray, _: i32) -> Vec3 {
        rec.normal.unit()
    }
}

/// A simple shader for quick rendering
pub struct SimpleShader {
    light_dir: Vec3,
}

impl SimpleShader {
    pub fn new() -> SimpleShader {
        SimpleShader {
            light_dir: Vec3::new(1., 1., -1.),
        }
    }
}

impl Shader for SimpleShader {
    /// Calculates the color only using normal and attenuation color
    fn shade(&self, _: &Renderer, rec: &HitRecord, ray: &Ray, _: i32) -> Vec3 {
        match rec.material.scatter(ray, rec) {
            None => rec.material.emitted(rec),
            Some(scatter_record) => {
                // Get a factor to multiply attenuation color, range between .25 -> 1.25
                // To get some decent flat shading
                let normal_factor = rec.normal.unit().dot(self.light_dir) * 0.5 + 0.75;

                scatter_record.attenuation * normal_factor
            }
        }
    }
}