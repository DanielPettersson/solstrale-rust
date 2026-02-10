//! Materials to be applied to hittable objects

use enum_dispatch::enum_dispatch;

use crate::geo::vec3::Vec3;
use crate::material::texture::SolidColor;
use crate::material::texture::Textures;

pub mod texture;

/// The trait for types that describe how
/// a ray behaves when hitting an object.
#[enum_dispatch]
pub trait Material {
    /// Is the material emitting light
    fn is_light(&self) -> bool {
        false
    }
}

#[derive(Default)]
/// A color along with attenuation information
pub struct AttenuatedColor {
    /// Color value before attenuation
    pub color: Vec3,
    /// Factor for calculating the amount of attenuation
    pub attenuation_factor: Option<f64>,
    /// Distance the light has travelled
    pub accumulated_ray_length: f64,
}

impl AttenuatedColor {
    /// Calculate the actual color based on the original color
    /// and the attenuation information
    pub fn get_attenuated_color(&self) -> Vec3 {
        self.attenuation_factor.map_or(self.color, |af| {
            self.color * 1. / (1. + af * self.accumulated_ray_length)
        })
    }
}

#[enum_dispatch(Material)]
#[derive(Debug, Clone)]
/// An enum of available materials
pub enum Materials {
    /// [`Material`] of type [`Lambertian`]
    Lambertian,
    /// [`Material`] of type [`Metal`]
    Metal,
    /// [`Material`] of type [`Dielectric`]
    Dielectric,
    /// [`Material`] of type [`DiffuseLight`]
    DiffuseLight,
    /// [`Material`] of type [`Blend`]
    Blend,
}

/// A typical matte material
#[derive(Clone, Debug)]
pub struct Lambertian {
    pub(crate) albedo: Textures,
    pub(crate) normal: Option<Textures>,
}

impl Lambertian {
    /// Create a new lambertian material
    pub fn new(albedo: Textures, normal: Option<Textures>) -> Lambertian {
        Lambertian { albedo, normal }
    }
}

impl Material for Lambertian {}

/// Metal is a material that is reflective
#[derive(Clone, Debug)]
pub struct Metal {
    pub(crate) albedo: Textures,
    pub(crate) normal: Option<Textures>,
    pub(crate) fuzz: f64,
}

impl Metal {
    /// Creates a metal material
    pub fn new(albedo: Textures, normal: Option<Textures>, fuzz: f64) -> Metal {
        Metal {
            albedo,
            normal,
            fuzz,
        }
    }
}

impl Material for Metal {}

/// A glass type material with an index of refraction
#[derive(Clone, Debug)]
pub struct Dielectric {
    pub(crate) albedo: Textures,
    pub(crate) normal: Option<Textures>,
    pub(crate) index_of_refraction: f64,
}

impl Dielectric {
    /// Creates a new dielectric material
    pub fn new(albedo: Textures, normal: Option<Textures>, index_of_refraction: f64) -> Self {
        Dielectric {
            albedo,
            normal,
            index_of_refraction,
        }
    }
}

impl Material for Dielectric {}

/// A material used for emitting light
#[derive(Clone, Debug)]
pub struct DiffuseLight {
    pub(crate) tex: Textures,
    pub(crate) attenuation_factor: Option<f64>,
}

impl DiffuseLight {
    /// Creates a new diffuse light material
    ///
    /// # Arguments
    /// * `r` - The red component of the light
    /// * `g` - The green component of the light
    /// * `b` - The blue component of the light
    /// * `attenuation_half_length` - The distance at which the light is attenuated to half its strength
    pub fn new(r: f64, g: f64, b: f64, attenuation_half_length: Option<f64>) -> Self {
        DiffuseLight {
            tex: SolidColor::new(r, g, b).into(),
            attenuation_factor: attenuation_half_length.map(|a| 1. / a),
        }
    }

    /// Creates a new diffuse light material
    ///
    /// # Arguments
    /// * `v` - The [`Vec3`] representation of the light color
    pub fn new_from_vec3(v: Vec3) -> Self {
        DiffuseLight {
            tex: SolidColor::new_from_vec3(v).into(),
            attenuation_factor: None,
        }
    }
}

impl Material for DiffuseLight {
    fn is_light(&self) -> bool {
        true
    }
}

/// A blend of two underlying materials
#[derive(Clone, Debug)]
pub struct Blend {
    pub(crate) material_1: Box<Materials>,
    pub(crate) material_2: Box<Materials>,
    pub(crate) blend_factor: f64,
}

impl Blend {
    /// Create a new blend material from two underlying material and a blend factor [0..1]
    pub fn new(material_1: Materials, material_2: Materials, blend_factor: f64) -> Self {
        Blend {
            material_1: Box::new(material_1),
            material_2: Box::new(material_2),
            blend_factor,
        }
    }
}

impl Material for Blend {}
