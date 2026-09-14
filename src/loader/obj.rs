//! Reads a Wavefront .obj file and creates a bvh containing
//! all triangles. It also read materials from the referred .mat file.
//! Support for colored and textured lambertian materials.
//! Applies supplied default material if none in model
use std::error::Error;

use rayon::prelude::*;
use simple_error::SimpleError;
use tobj::LoadOptions;

use crate::geo::Uv;
use crate::geo::transformation::{NopTransformer, Transformer};
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
        let (models, materials) = tobj::load_obj(&filepath, &load_options)
            .map_err(|_| SimpleError::new(format!("failed to load obj model from {}", filepath)))?;
        let materials =
            materials.map_err(|_| format!("failed to load MTL file for {}", filepath))?;

        // Indexed by tobj's material id. This used to be a `HashMap<i8, Materials>` with
        // the default material parked at key -1, so `id as i8` made material 128 alias
        // onto the default and every 256th material alias onto another.
        let mut mats: Vec<Materials> = Vec::with_capacity(materials.len());
        for m in materials.iter() {
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
            mats.push(Lambertian::new(albedo_texture, normal_texture).into());
        }

        // The total is known up front, so the vector never has to grow. At ~312 bytes per
        // `Hittables::Triangle` a 250k-triangle mesh is ~78 MB, and growing that by
        // doubling from zero copied about twice that much for nothing.
        let face_count: usize = models.iter().map(|m| m.mesh.indices.len() / 3).sum();
        let mut triangles: Vec<Hittables> = Vec::with_capacity(face_count);

        for m in &models {
            let mesh = &m.mesh;

            // `material_id` belongs to the mesh, not to the face. Resolving it inside the
            // face loop did a hash lookup and a `Materials` clone per triangle to arrive
            // at the same answer every time.
            let material = mesh
                .material_id
                .and_then(|id| mats.get(id))
                .unwrap_or(&default_material);

            // Transform once per vertex instead of once per triangle corner. A closed
            // mesh shares each vertex between ~6 faces, so the old code applied the
            // transformation ~6 times over. It also keeps the rayon closure below from
            // needing to capture `&dyn Transformer`, which `Transformer` does not require
            // to be `Sync` -- adding that bound would be a breaking change to the public
            // `Loader`, `Triangle` and `Quad` signatures.
            //
            // Results are bit-identical: every `Transformer` in this crate is a pure
            // function of its argument, so transforming a vertex once and reusing it
            // produces exactly the bits the per-corner version did. Only the *number* of
            // `transform` calls changes, which an implementor with interior mutability
            // would notice.
            let positions: Vec<Vec3> = (0..mesh.positions.len() / 3)
                .map(|v| transformation.transform(vec3_from_mesh_vec(&mesh.positions, v * 3), false))
                .collect();

            // `(0..n).into_par_iter()` is indexed, so `par_extend` reserves exactly and
            // writes each triangle into its own slot: the output order is identical to
            // the serial loop's. That matters -- `Bvh::new` splits on input order, so a
            // reordering here would change every leaf and every rendered image.
            triangles.par_extend(
                (0..mesh.indices.len() / 3)
                    .into_par_iter()
                    // spider.obj is 19 meshes averaging ~70 faces. Without a floor rayon
                    // would spend more on splitting them than on the triangles.
                    .with_min_len(2048)
                    .map(|f| {
                        let i = f * 3;
                        let v0 = positions[mesh.indices[i] as usize];
                        let v1 = positions[mesh.indices[i + 1] as usize];
                        let v2 = positions[mesh.indices[i + 2] as usize];

                        let (uv0, uv1, uv2) = if mesh.texcoords.is_empty() {
                            (Uv::default(), Uv::default(), Uv::default())
                        } else {
                            (
                                uv_from_mesh(mesh, i),
                                uv_from_mesh(mesh, i + 1),
                                uv_from_mesh(mesh, i + 2),
                            )
                        };

                        Triangle::new_with_tex_coords(
                            v0,
                            v1,
                            v2,
                            uv0,
                            uv1,
                            uv2,
                            material.clone(),
                            &NopTransformer(),
                        )
                        .into()
                    }),
            );
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

/// Texture coordinate for the `i`th entry of the mesh's index array.
///
/// `single_index` is off, so texture coordinates carry their own index array rather than
/// sharing the position one.
fn uv_from_mesh(mesh: &tobj::Mesh, i: usize) -> Uv {
    let offset = (mesh.texcoord_indices[i] * 2) as usize;
    Uv {
        u: mesh.texcoords[offset],
        v: mesh.texcoords[offset + 1],
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::geo::transformation::{NopTransformer, RotationY, Transformations, Translation};
    use crate::hittable::Hittables;
    use crate::material::texture::Textures;

    use super::*;

    /// FNV-1a over every geometric field of every loaded triangle, in `prims` order.
    ///
    /// The golden-image tests are stochastic GPU renders compared at 0.95 RMS, so they
    /// happily absorb a reordering or a drifted transform. This is the oracle that does
    /// not: it pins the exact bytes the loader produces, in the exact order the BVH build
    /// sees them. Any change to `Obj::load` that leaves these constants alone is
    /// geometrically a no-op.
    fn geometry_checksum(bvh: &Bvh) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |bytes: &[u8]| {
            for b in bytes {
                h ^= *b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };

        for prim in &bvh.prims {
            let t = match prim {
                Hittables::Triangle(t) => t,
                other => panic!("expected only triangles, got {:?}", other),
            };
            for v in [t.v0, t.v0v1, t.v0v2, t.normal, t.tangent, t.bi_tangent] {
                eat(&v.x.to_le_bytes());
                eat(&v.y.to_le_bytes());
                eat(&v.z.to_le_bytes());
            }
            for uv in [t.uv0, t.uv1, t.uv2] {
                eat(&uv.u.to_le_bytes());
                eat(&uv.v.to_le_bytes());
            }
            eat(&t.area.to_le_bytes());
        }
        h
    }

    /// How many distinct decoded images the loaded triangles point at.
    ///
    /// `spider.mtl` has 19 `usemtl` groups over 4 JPEGs, so this is what proves that
    /// resolving the material once per *mesh* still hands every triangle the same shared
    /// `Arc` the per-triangle lookup did -- the flattener dedups textures by
    /// `Arc::ptr_eq`, so collapsing or splitting those Arcs is observable downstream.
    fn distinct_albedo_images(bvh: &Bvh) -> usize {
        let mut ptrs: Vec<usize> = bvh
            .prims
            .iter()
            .filter_map(|p| match p {
                Hittables::Triangle(t) => Some(&t.mat),
                _ => None,
            })
            .filter_map(|m| match m {
                Materials::Lambertian(l) => match &l.albedo {
                    Textures::ImageMap(im) => Some(Arc::as_ptr(&im.get_image()) as usize),
                    Textures::SolidColor(_) => None,
                },
                _ => None,
            })
            .collect();
        ptrs.sort_unstable();
        ptrs.dedup();
        ptrs.len()
    }

    #[test]
    fn spider_geometry_is_stable() {
        let bvh = Obj::new("resources/spider/", "spider.obj")
            .load(&NopTransformer(), None)
            .unwrap();

        assert_eq!(1368, bvh.prims.len());
        assert_eq!(4, distinct_albedo_images(&bvh));
        assert_eq!(0x3dbd_c185_013f_ed05, geometry_checksum(&bvh));
    }

    #[test]
    fn spider_geometry_is_stable_under_transformation() {
        let transformation = Transformations::new(vec![
            Box::new(RotationY::new(37.)),
            Box::new(Translation::new(Vec3::new(1.5, -2.25, 0.75))),
        ]);
        let bvh = Obj::new("resources/spider/", "spider.obj")
            .load(&transformation, None)
            .unwrap();

        assert_eq!(1368, bvh.prims.len());
        assert_eq!(0xd439_8fec_0262_fe56, geometry_checksum(&bvh));
    }

    #[test]
    fn box_loads_twelve_triangles_with_the_default_material() {
        let bvh = Obj::new("resources/obj/", "box.obj")
            .load(&NopTransformer(), None)
            .unwrap();

        assert_eq!(12, bvh.prims.len());
        assert_eq!(0xe45a_ef74_7f0c_bc65, geometry_checksum(&bvh));
    }

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
