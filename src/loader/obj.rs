//! Reads a Wavefront .obj file and creates a bvh containing
//! all triangles. It also read materials from the referred .mat file.
//! Support for colored and textured lambertian materials.
//! Applies supplied default material if none in model
use std::collections::HashMap;
use std::error::Error;

use simple_error::SimpleError;
use tobj::LoadOptions;

use crate::geo::Uv;
use crate::geo::transformation::Transformer;
use crate::geo::vec3::Vec3;
use crate::hittable::Bvh;
use crate::hittable::Hittables;
use crate::hittable::Triangle;
use crate::loader::Loader;
use crate::material::texture::{ImageMap, SolidColor, Textures};
use crate::material::{Lambertian, Materials, texture};

/// Contains file information about the obj to load
pub struct Obj {
    path: String,
    filename: String,
}

impl Obj {
    /// Creates a new [`Obj`] instance
    pub fn new(path: &str, filename: &str) -> Obj {
        Obj {
            path: path.to_string(),
            filename: filename.to_string(),
        }
    }
}

impl Loader for Obj {
    fn load(
        &self,
        transformation: &dyn Transformer,
        default_material: Option<Materials>,
    ) -> Result<Bvh, Box<dyn Error>> {
        let default_material = default_material
            .unwrap_or(Lambertian::new(SolidColor::new(1., 1., 1.).into(), None).into());
        let load_options = LoadOptions {
            triangulate: true,
            ..Default::default()
        };

        let filepath = format!("{}{}", self.path, self.filename);
        let (models, materials) = tobj::load_obj(&filepath, &load_options).map_err(|_| {
            SimpleError::new(format!("failed to load obj model from {}", &filepath))
        })?;
        let materials =
            materials.map_err(|_| format!("failed to load MTL file for {}", &filepath))?;

        let mut mat_map = HashMap::from([(-1, default_material.clone())]);
        for (i, m) in materials.iter().enumerate() {
            let albedo_texture: Textures = match &m.diffuse_texture {
                None => match m.diffuse {
                    None => SolidColor::new(1., 1., 1.).into(),
                    Some(c) => SolidColor::new_from_f32_array(c).into(),
                },
                Some(diffuse_texture_filename) => {
                    ImageMap::load(&format!("{}{}", self.path, diffuse_texture_filename))?.into()
                }
            };
            let normal_texture: Option<Textures> = match &m.normal_texture {
                None => None,
                Some(bump_texture_filename) => {
                    let bump_texture_path = format!("{}{}", self.path, bump_texture_filename);
                    Some(texture::load_normal_texture(&bump_texture_path)?.into())
                }
            };
            mat_map.insert(
                i as i8,
                Lambertian::new(albedo_texture, normal_texture).into(),
            );
        }

        let mut triangles: Vec<Hittables> = Vec::new();

        for m in models {
            let mesh = &m.mesh;
            for i in (0..mesh.indices.len()).step_by(3) {
                let mut pos_offset = (mesh.indices[i] * 3) as usize;

                let v0 = vec3_from_mesh_vec(&mesh.positions, pos_offset);
                pos_offset = (mesh.indices[i + 1] * 3) as usize;
                let v1 = vec3_from_mesh_vec(&mesh.positions, pos_offset);
                pos_offset = (mesh.indices[i + 2] * 3) as usize;
                let v2 = vec3_from_mesh_vec(&mesh.positions, pos_offset);

                let (uv0, uv1, uv2) = if mesh.texcoords.is_empty() {
                    (Uv::default(), Uv::default(), Uv::default())
                } else {
                    let tex_offset1 = (mesh.texcoord_indices[i] * 2) as usize;
                    let tex_offset2 = (mesh.texcoord_indices[i + 1] * 2) as usize;
                    let tex_offset3 = (mesh.texcoord_indices[i + 2] * 2) as usize;
                    (
                        Uv {
                            u: mesh.texcoords[tex_offset1],
                            v: mesh.texcoords[tex_offset1 + 1],
                        },
                        Uv {
                            u: mesh.texcoords[tex_offset2],
                            v: mesh.texcoords[tex_offset2 + 1],
                        },
                        Uv {
                            u: mesh.texcoords[tex_offset3],
                            v: mesh.texcoords[tex_offset3 + 1],
                        },
                    )
                };

                let material_id = match mesh.material_id {
                    None => -1,
                    Some(id) => id as i8,
                };
                let material = match mat_map.get(&material_id) {
                    None => default_material.to_owned(),
                    Some(m) => m.to_owned(),
                };

                triangles.push(
                    Triangle::new_with_tex_coords(
                        v0,
                        v1,
                        v2,
                        uv0,
                        uv1,
                        uv2,
                        material,
                        transformation,
                    )
                    .into(),
                );
            }
        }

        Ok(Bvh::new(triangles))
    }
}

fn vec3_from_mesh_vec(positions: &[f32], offset: usize) -> Vec3 {
    Vec3::new(
        positions[offset] as f64,
        positions[offset + 1] as f64,
        positions[offset + 2] as f64,
    )
}

#[cfg(test)]
mod tests {
    use crate::geo::transformation::NopTransformer;

    use super::*;

    #[test]
    fn missing_file() {
        let res = Obj::new("resources/obj/", "missing.obj").load(&NopTransformer(), None);
        assert_eq!(
            "failed to load obj model from resources/obj/missing.obj",
            format!("{}", res.err().unwrap())
        );
    }

    #[test]
    fn missing_material_file() {
        let res =
            Obj::new("resources/obj/", "missingMaterialLib.obj").load(&NopTransformer(), None);
        assert_eq!(
            "failed to load MTL file for resources/obj/missingMaterialLib.obj",
            format!("{}", res.err().unwrap())
        );
    }

    #[test]
    fn missing_image_file() {
        let res = Obj::new("resources/obj/", "missingImage.obj").load(&NopTransformer(), None);
        assert!(
            format!("{}", res.err().unwrap())
                .contains("Failed to open image texture resources/obj/missing.jpg")
        );
    }

    #[test]
    fn invalid_image_file() {
        let res = Obj::new("resources/obj/", "invalidImage.obj").load(&NopTransformer(), None);
        assert!(
            format!("{}", res.err().unwrap())
                .contains("Failed to decode image texture resources/obj/invalidImage.mtl")
        );
    }
}
