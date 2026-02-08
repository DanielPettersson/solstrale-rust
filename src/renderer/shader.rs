//! Contains the path-tracing shader used by the renderer

use crate::geo::Ray;
use crate::geo::vec3::Vec3;
use crate::material::Material;
use crate::material::RayScatter::{ScatterBasic, ScatterEmission, ScatterPdf};
use crate::material::{AttenuatedColor, RayHit};
use crate::renderer::Renderer;

#[derive(Clone)]
/// A full raytracing shader
pub struct PathTracingShader {
    /// Maximum recursive depth of the path tracing
    pub max_depth: u32,
}

impl PathTracingShader {
    /// Create a new path tracing shader
    pub fn new(max_depth: u32) -> Self {
        PathTracingShader { max_depth }
    }

    /// Calculates the color using path tracing
    pub fn shade(
        &self,
        renderer: &Renderer,
        rec: &RayHit,
        ray: &Ray,
        depth: u32,
        accumulated_ray_length: f64,
    ) -> AttenuatedColor {
        if depth >= self.max_depth {
            return AttenuatedColor::default();
        }

        let total_ray_length = rec.ray_length + accumulated_ray_length;
        let ray_scatter = rec.material.scatter(ray, rec, &renderer.lights);

        match ray_scatter {
            ScatterEmission(s) => AttenuatedColor {
                color: s.color,
                attenuation_factor: s.attenuation_factor,
                accumulated_ray_length: total_ray_length,
            },
            ScatterBasic(s) => {
                let ray_color_res = renderer.ray_color(&s.ray, depth + 1, total_ray_length);

                AttenuatedColor {
                    color: s.color * ray_color_res.pixel_color.color,
                    attenuation_factor: ray_color_res.pixel_color.attenuation_factor,
                    accumulated_ray_length: ray_color_res.pixel_color.accumulated_ray_length,
                }
            }
            ScatterPdf(s) => {
                let ray_color_res = renderer.ray_color(&s.ray, depth + 1, total_ray_length);
                let scatter_color = s.color * s.probability * ray_color_res.pixel_color.color;

                AttenuatedColor {
                    color: filter_invalid_color_values(scatter_color),
                    attenuation_factor: ray_color_res.pixel_color.attenuation_factor,
                    accumulated_ray_length: ray_color_res.pixel_color.accumulated_ray_length,
                }
            }
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
        // A subjectively chosen value that is a trade-off between
        // color acne and suppressing intensity
        val.min(3.)
    }
}
