//! Provides a camera used by raytracer to shoot rays into the scene

use crate::geo::vec3::{Vec3, ZERO_VECTOR, random_in_unit_disc};
use crate::geo::{Ray, Uv};
use crate::util::degrees_to_radians;

/// Contains all needed parameters for constructing a camera
pub struct CameraConfig {
    /// Vertical field of view in degrees
    pub vertical_fov_degrees: f64,
    /// Radius of the lens of the camera, affects the depth of field
    pub aperture_size: f64,
    /// Point where the camera is located
    pub look_from: Vec3,
    /// Point where the camera is looking
    pub look_at: Vec3,
    /// Direction pointing "up" for the camera
    pub up: Vec3,
}

impl Default for CameraConfig {
    fn default() -> Self {
        CameraConfig {
            vertical_fov_degrees: 50.0,
            aperture_size: 0.0,
            look_from: ZERO_VECTOR,
            look_at: ZERO_VECTOR,
            up: Vec3::new(0., 1., 0.),
        }
    }
}

/// Contains all data needed to describe a cameras position, field of view and
/// where it is pointing
pub struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    lens_radius: f64,
}

impl Camera {
    /// Create a new camera instance
    pub fn new(image_width: usize, image_height: usize, c: &CameraConfig) -> Camera {
        let aspect_ratio = image_width as f64 / image_height as f64;
        let theta = degrees_to_radians(c.vertical_fov_degrees);
        let h = (theta / 2.).tan();
        let view_port_height = 2. * h;
        let view_port_width = aspect_ratio * view_port_height;

        let look_v = c.look_from - c.look_at;
        let focus_distance = look_v.length();
        let w = look_v.unit();
        let u = c.up.unit().cross(w).unit();
        let v = w.cross(u);

        let horizontal = (u * view_port_width) * focus_distance;
        let vertical = (v * view_port_height) * focus_distance;
        let lower_left_corner =
            c.look_from - (horizontal / 2.) - (vertical / 2.) - (w * focus_distance);

        Camera {
            origin: c.look_from,
            lower_left_corner,
            horizontal,
            vertical,
            u,
            v,
            lens_radius: c.aperture_size / 2.,
        }
    }

    /// A function for generating a ray for a certain u/v for the raytraced image
    pub fn get_ray(&self, uv: Uv) -> Ray {
        let offset = if self.lens_radius > 0. {
            let rd = random_in_unit_disc() * self.lens_radius;
            self.u * rd.x + self.v * rd.y
        } else {
            ZERO_VECTOR
        };

        let r_dir = self.lower_left_corner + (self.horizontal * uv.u) + (self.vertical * uv.v)
            - self.origin
            - offset;
        Ray::new(self.origin + offset, r_dir)
    }
}
