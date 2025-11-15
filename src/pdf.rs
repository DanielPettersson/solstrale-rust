//! Probability density functions

use std::f64::consts::PI;

use enum_dispatch::enum_dispatch;

use crate::geo::Onb;
use crate::geo::vec3::{random_cosine_direction, random_unit_vector, Vec3};
use crate::hittable::{Hittable, Hittables};
use crate::random::{random_element_index, random_normal_float};

const SPHERE_PDF_VALUE: f64 = 1. / (4. * PI);

#[enum_dispatch]
/// Probability density function
pub trait Pdf {
    /// Returns the pdf value for a given vector
    fn value(&self, direction: Vec3) -> f64;
    /// Generate a random direction for the pdf shape
    fn generate(&self) -> Vec3;
}

#[enum_dispatch(Pdf)]
/// The available probability density functions
pub enum Pdfs<'a> {
    /// [`Pdf`] of type [`CosinePdf`]
    CosinePdfType(CosinePdf),
    /// [`Pdf`] of type [`ContainerPdf`]
    ContainerPdfType(ContainerPdf<'a>),
    /// [`Pdf`] of type [`SpherePdf`]
    SpherePdfType(SpherePdf),
}

/// Returns the pdf value for a given vector for the pdfs.
/// Which is the average of the two base pdfs
pub fn mix_value(p0: &Pdfs, p1: &Pdfs, direction: Vec3) -> f64 {
    0.5 * p0.value(direction) + 0.5 * p1.value(direction)
}

/// Random direction for the pdfs shape.
/// Which is randomly chosen between the two base pdfs.
pub fn mix_generate(p0: &Pdfs, p1: &Pdfs) -> Vec3 {
    if random_normal_float() < 0.5 {
        p0.generate()
    } else {
        p1.generate()
    }
}

/// A probability density function with a cosine distribution
pub struct CosinePdf {
    uvw: Onb,
}

impl CosinePdf {
    /// Creates a new instance of a CosinePdf
    pub fn new(w: Vec3) -> Self {
        CosinePdf { uvw: Onb::new(w) }
    }
}

impl Pdf for CosinePdf {
    fn value(&self, direction: Vec3) -> f64 {
        let cosine_theta = direction.unit().dot(self.uvw.normal);
        (cosine_theta / PI).max(0.)
    }

    fn generate(&self) -> Vec3 {
        self.uvw.local(random_cosine_direction())
    }
}

/// A wrapper for generating pdfs for a list of hittable objects
pub struct ContainerPdf<'a> {
    objects: &'a [Hittables],
    origin: Vec3,
}

impl<'a> ContainerPdf<'a> {
    /// Creates a new instance of ContainerPdf
    pub fn new(objects: &'a [Hittables], origin: Vec3) -> Self {
        ContainerPdf { objects, origin }
    }
}

impl Pdf for ContainerPdf<'_> {
    fn value(&self, direction: Vec3) -> f64 {
        let sum: f64 = self
            .objects
            .iter()
            .map(|i| i.pdf_value(self.origin, direction))
            .sum();
        sum / self.objects.len() as f64
    }

    fn generate(&self) -> Vec3 {
        let idx = random_element_index(self.objects);
        self.objects[idx].random_direction(self.origin)
    }
}

/// A probability density functions with a sphere distribution
#[derive(Default)]
pub struct SpherePdf();

impl SpherePdf {
    /// Creates a new instance of SpherePdf
    pub fn new() -> Self {
        SpherePdf::default()
    }
}

impl Pdf for SpherePdf {
    /// returns the pdf value for a given vector for the SpherePdf
    fn value(&self, _: Vec3) -> f64 {
        SPHERE_PDF_VALUE
    }

    /// Generate a random direction for the SpherePdf shape
    fn generate(&self) -> Vec3 {
        random_unit_vector()
    }
}
