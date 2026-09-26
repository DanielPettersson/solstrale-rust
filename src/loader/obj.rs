//! Reads a Wavefront .obj file and creates a bvh containing
//! all triangles. It also read materials from the referred .mat file.
//! MTL materials are mapped onto this crate's material types by
//! [`material_from_mtl`].
//! Applies supplied default material if none in model
use std::error::Error;

use rayon::prelude::*;
use simple_error::SimpleError;
use tobj::LoadOptions;

use crate::geo::Uv;
use crate::geo::transformation::{NopTransformer, Transformer};
use crate::geo::vec3::{Vec3, ZERO_VECTOR};
use crate::hittable::Bvh;
use crate::hittable::Hittables;
use crate::hittable::Triangle;
use crate::loader::Loader;
use crate::material::texture::{ImageMap, SolidColor, Textures};
use crate::material::{Blend, Dielectric, DiffuseLight, Lambertian, Materials, Metal, texture};
use crate::util::luminance::luminance;

/// Crease angle used when the caller does not pick one.
///
/// Measured corner-normal against face-normal (see
/// [`Obj::with_generated_normals`]), so it has to sit above a coarse curved
/// surface and below a box. `sphere.obj` reads 13.5 to 23.8 degrees and
/// `box.obj` 48.2 to 70.5, so 40 leaves 16 degrees of margin on one and 8 on
/// the other.
pub const DEFAULT_CREASE_ANGLE_DEGREES: f64 = 40.;

/// Contains file information about the obj to load
pub struct Obj {
    path: String,
    filename: String,
    /// Cosine of the crease angle, or `None` for [`Obj::with_flat_shading`].
    generated_crease_cos: Option<f64>,
}

impl Obj {
    /// Creates a new [`Obj`] instance.
    ///
    /// A mesh that ships `vn` records is shaded with them; one that does not
    /// gets normals generated at [`DEFAULT_CREASE_ANGLE_DEGREES`]. See
    /// [`Obj::with_generated_normals`] and [`Obj::with_flat_shading`].
    pub fn new(path: &str, filename: &str) -> Obj {
        Obj {
            path: path.to_string(),
            filename: filename.to_string(),
            generated_crease_cos: Some(DEFAULT_CREASE_ANGLE_DEGREES.to_radians().cos()),
        }
    }

    /// Sets the crease angle for generated normals. No effect on a mesh that
    /// has its own.
    ///
    /// The angle is between a corner's area-weighted vertex normal and its own
    /// face normal, not between two faces across an edge -- roughly half of
    /// that for a two-face fold, and more where several faces disagree. 180
    /// smooths everything, 0 nothing.
    ///
    /// # Examples:
    /// ```no_run
    /// # use solstrale::geo::transformation::NopTransformer;
    /// # use solstrale::loader::Loader;
    /// # use solstrale::loader::obj::Obj;
    /// let bvh = Obj::new("resources/obj/", "box.obj")
    ///     .with_generated_normals(60.)
    ///     .load(&NopTransformer(), None)
    ///     .unwrap();
    /// ```
    pub fn with_generated_normals(mut self, crease_angle_degrees: f64) -> Obj {
        self.generated_crease_cos = Some(crease_angle_degrees.to_radians().cos());
        self
    }

    /// Generates nothing, leaving a mesh without `vn` records flat-shaded. For
    /// the model whose facets are the point. No effect on a mesh that has its
    /// own normals.
    ///
    /// # Examples:
    /// ```no_run
    /// # use solstrale::geo::transformation::NopTransformer;
    /// # use solstrale::loader::Loader;
    /// # use solstrale::loader::obj::Obj;
    /// let bvh = Obj::new("resources/obj/", "box.obj")
    ///     .with_flat_shading()
    ///     .load(&NopTransformer(), None)
    ///     .unwrap();
    /// ```
    pub fn with_flat_shading(mut self) -> Obj {
        self.generated_crease_cos = None;
        self
    }
}

/// Where a mesh's shading normals come from, resolved once per mesh.
enum ShadingNormals {
    /// None: [`Triangle`] gives all three corners the geometric normal.
    Flat,
    /// The mesh's own `vn` records, in world space. Indexed by
    /// `normal_indices`, its own array because `single_index` is off.
    FromFile(Vec<Vec3>),
    /// From [`generate_normals`]: one per *corner*, so indexed by position in
    /// the index array. A corner across a crease keeps its own face normal,
    /// which no per-vertex array could express.
    Generated(Vec<Vec3>),
}

impl ShadingNormals {
    /// The three shading normals for the face starting at index-array entry `i`,
    /// or `None` when the mesh is flat-shaded.
    fn face(&self, mesh: &tobj::Mesh, i: usize) -> Option<[Vec3; 3]> {
        match self {
            ShadingNormals::Flat => None,
            ShadingNormals::FromFile(normals) => Some([
                normals[mesh.normal_indices[i] as usize],
                normals[mesh.normal_indices[i + 1] as usize],
                normals[mesh.normal_indices[i + 2] as usize],
            ]),
            ShadingNormals::Generated(corners) => {
                Some([corners[i], corners[i + 1], corners[i + 2]])
            }
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

        // Indexed by tobj's material id. A map keyed on `i8` would make material
        // 128 alias onto the default and every 256th onto another.
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
            mats.push(material_from_mtl(m, albedo_texture, normal_texture));
        }

        // The total is known up front, so the vector never has to grow. At 376
        // bytes per `Hittables::Triangle` a 250k-triangle mesh is ~94 MB, and
        // doubling from zero copies about twice that for nothing. Host memory
        // only -- the GPU copy is octahedral and fits in `TriangleAttr`'s
        // existing padding.
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

            // Once per vertex rather than once per triangle corner: a closed
            // mesh shares each vertex between ~6 faces. It also keeps the rayon
            // closure below from capturing `&dyn Transformer`, which
            // `Transformer` is not required to be `Sync` -- adding that bound
            // would break the public `Loader`, `Triangle` and `Quad`
            // signatures.
            //
            // Only an implementor with interior mutability could notice: every
            // `Transformer` here is a pure function of its argument, so the
            // resulting vertices are bit-identical.
            let positions: Vec<Vec3> = (0..mesh.positions.len() / 3)
                .map(|v| {
                    transformation.transform(vec3_from_mesh_vec(&mesh.positions, v * 3), false)
                })
                .collect();

            // Resolved once per mesh, outside the parallel loop: generation is
            // a scatter into shared slots, so it stays serial and the
            // par_extend below is untouched.
            let shading_normals = if !mesh.normals.is_empty() {
                // Once per `vn` record rather than per corner, as the
                // positions above are. Left un-normalised: normalising here as
                // well would turn spider.obj's zero-length `vn` into a NaN
                // before `Triangle::new_with_normals` could guard it.
                ShadingNormals::FromFile(
                    (0..mesh.normals.len() / 3)
                        .map(|v| {
                            transformation.transform(vec3_from_mesh_vec(&mesh.normals, v * 3), true)
                        })
                        .collect(),
                )
            } else {
                match self.generated_crease_cos {
                    // Generated from the already-transformed positions, so they
                    // need no transformation of their own.
                    Some(crease_cos) => {
                        ShadingNormals::Generated(generate_normals(mesh, &positions, crease_cos))
                    }
                    None => ShadingNormals::Flat,
                }
            };

            // `(0..n).into_par_iter()` is indexed, so `par_extend` writes each
            // triangle into its own slot and the output order matches a serial
            // loop's. `Bvh::new` splits on input order, so a reordering here
            // would change every leaf and every rendered image.
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

                        match shading_normals.face(mesh, i) {
                            None => Triangle::new_with_tex_coords(
                                v0,
                                v1,
                                v2,
                                uv0,
                                uv1,
                                uv2,
                                material.clone(),
                                &NopTransformer(),
                            ),
                            Some(n) => Triangle::new_with_normals(
                                [v0, v1, v2],
                                n,
                                [uv0, uv1, uv2],
                                material.clone(),
                                &NopTransformer(),
                            ),
                        }
                        .into()
                    }),
            );
        }

        Ok(Bvh::new(triangles))
    }
}

/// Relative luminance of an MTL colour, used only to weigh `Kd` against `Ks`.
fn mtl_luminance(c: [f32; 3]) -> f64 {
    luminance(Vec3::new(c[0] as f64, c[1] as f64, c[2] as f64))
}

/// `Ns` to the perceptual roughness [`Metal::new`] takes.
///
/// `alpha = sqrt(2 / (Ns + 2))` is the usual Phong-to-GGX conversion and
/// `Metal` squares its parameter to get `alpha`, so this is the fourth root.
/// A file with no `Ns` has no specular sharpness to declare, and comes out
/// fully rough just as `Ns 0` does.
fn roughness_from_shininess(ns: f32) -> f64 {
    (2. / (ns.max(0.) as f64 + 2.)).powf(0.25)
}

/// Three floats from an `unknown_param` value, `Ke 1 0.5 0.2` style. A single
/// float is read as grey.
fn parse_f32_3(s: &str) -> Option<[f32; 3]> {
    let v = s
        .split_whitespace()
        .map(str::parse::<f32>)
        .collect::<Result<Vec<f32>, _>>()
        .ok()?;
    match v.len() {
        1 => Some([v[0]; 3]),
        3 => Some([v[0], v[1], v[2]]),
        _ => None,
    }
}

/// Maps one MTL material onto the material types this crate has.
///
/// Three things decide, in this order, each more specific than the next:
///
/// 1. `Ke`, when non-zero. A file that declares emission means the surface to
///    emit, and [`DiffuseLight`] has no diffuse lobe to share with.
/// 2. `d` / `Tr`, when they say the surface is not opaque. Opacity is data
///    about the surface, where `illum` is only a hint about which model to
///    shade it with.
/// 3. `illum`, when the file states one. Heuristics otherwise.
///
/// `map_Ks`, `map_Ns` and `map_d` are dropped: [`crate::renderer::gpu_data`]'s
/// `Material` carries one albedo slot and one normal slot, and a third texture
/// would take the struct from 96 to 112 bytes. `Tf` is dropped for want of
/// anything to map a transmission filter onto.
fn material_from_mtl(m: &tobj::Material, albedo: Textures, normal: Option<Textures>) -> Materials {
    if let Some(ke) = m.unknown_param.get("Ke").and_then(|s| parse_f32_3(s))
        && ke.iter().any(|&c| c > 0.)
    {
        return DiffuseLight::new(ke[0] as f64, ke[1] as f64, ke[2] as f64, None).into();
    }

    // Zero when absent, so a material that never mentions `Ks` cannot pick up a
    // specular lobe it did not ask for.
    let ks_luminance = m.specular.map(mtl_luminance).unwrap_or(0.);
    // `map_Kd` is the albedo when it is there, and the `Kd` beside it is
    // discarded -- so it has to be discarded here too, or a `Kd 0 0 0` next to
    // a texture and a `Ks` would weigh the blend all the way to metal and drop
    // the texture on the floor. White otherwise, which is what the albedo
    // defaults to when the file gives neither.
    let kd_luminance = match m.diffuse_texture {
        Some(_) => 1.,
        None => m.diffuse.map(mtl_luminance).unwrap_or(1.),
    };

    // `Tr` is `1 - d`, and is not among the keys tobj parses.
    let dissolve = m.dissolve.or_else(|| {
        m.unknown_param
            .get("Tr")
            .and_then(|s| s.trim().parse::<f32>().ok())
            .map(|tr| 1. - tr)
    });

    let diffuse = || -> Materials { Lambertian::new(albedo.clone(), normal.clone()).into() };

    let specular = || -> Materials {
        // White when absent: a file that declares a mirror without `Ks` means a
        // mirror, not a black surface. The zero above is only there to keep a
        // missing `Ks` out of the blend weight.
        let f0 = m.specular.unwrap_or([1.; 3]);
        Metal::new(
            SolidColor::new_from_f32_array(f0).into(),
            normal.clone(),
            roughness_from_shininess(m.shininess.unwrap_or(0.)),
        )
        .into()
    };

    let transparent = || -> Materials {
        // `albedo` is Beer-Lambert absorption per world unit here rather than a
        // surface colour, and `Kd` is the only tint the file offers. `Ni` has a
        // spec default of 1, which is glass that does not bend light at all, so
        // a surface that declares itself transparent and omits `Ni` gets 1.5.
        Dielectric::new(
            albedo.clone(),
            normal.clone(),
            m.optical_density.unwrap_or(1.5) as f64,
            0.,
        )
        .into()
    };

    // `Blend` is a stochastic choice between two materials, not a layer: right
    // in expectation, wrong in variance against a real layered BSDF, which is
    // the price of covering `Kd` + `Ks` without a new material type.
    // `blend_factor` is the chance of the second material.
    //
    // A zero weight has to collapse to a plain `Lambertian` rather than a
    // `Blend` that never picks its second arm, or every `Ks 0` model would pay
    // for `has_blends`.
    let plastic = || -> Materials {
        let total = ks_luminance + kd_luminance;
        let w = if total > 0. { ks_luminance / total } else { 0. };
        if w <= 0. {
            diffuse()
        } else if w >= 1. {
            specular()
        } else {
            Blend::new(diffuse(), specular(), w).into()
        }
    };

    if dissolve.is_some_and(|d| d < 1.) {
        return transparent();
    }

    match m.illumination_model {
        Some(0 | 1) => diffuse(),
        Some(3 | 5) => specular(),
        // 2 is the highlight model this blend exists for. 4, 6, 7 and 9 are
        // the transparent family, and reach here only on a surface `d` called
        // opaque -- the common case, not a corner: `fireplace_room` marks 21 of
        // its 22 materials `illum 4` or `illum 7`, floor and dirt and leaves
        // included, and declares `d` on none of them. 8 disables ray-traced
        // reflection and 10 is a shadow-matte flag, neither of which describes
        // the surface, so both fall through.
        _ => plastic(),
    }
}

fn vec3_from_mesh_vec(positions: &[f32], offset: usize) -> Vec3 {
    Vec3::new(
        positions[offset] as f64,
        positions[offset + 1] as f64,
        positions[offset + 2] as f64,
    )
}

/// Per-corner shading normals for a mesh that ships none.
///
/// Area-weighted for free: `(v1-v0) x (v2-v0)` has twice the triangle's area as
/// its magnitude, so accumulating un-normalised is the weighting. That makes it
/// invariant to how a flat region happens to be triangulated, where an
/// unweighted mean lets a sliver pull the vertex normal toward itself.
///
/// The crease test runs per corner after the accumulation, comparing the vertex
/// normal to that corner's own face, so a sharp edge comes out faceted on both
/// sides while a smooth region sharing the vertex stays smooth. One pass, no
/// adjacency structure.
fn generate_normals(mesh: &tobj::Mesh, positions: &[Vec3], crease_cos: f64) -> Vec<Vec3> {
    let face_count = mesh.indices.len() / 3;
    let mut face_normals: Vec<Vec3> = Vec::with_capacity(face_count);
    let mut accumulated = vec![ZERO_VECTOR; positions.len()];

    for f in 0..face_count {
        let i = f * 3;
        let a = mesh.indices[i] as usize;
        let b = mesh.indices[i + 1] as usize;
        let c = mesh.indices[i + 2] as usize;

        let n = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        face_normals.push(n);
        accumulated[a] += n;
        accumulated[b] += n;
        accumulated[c] += n;
    }

    let mut corners: Vec<Vec3> = Vec::with_capacity(mesh.indices.len());
    for f in 0..face_count {
        let face = face_normals[f].unit();
        for k in 0..3 {
            let sum = accumulated[mesh.indices[f * 3 + k] as usize];
            // Faces that cancel exactly -- two back-to-back triangles -- have
            // no meaningful average, and `unit()` of zero is NaN.
            if sum.near_zero() {
                corners.push(face);
                continue;
            }
            let vertex = sum.unit();
            corners.push(if vertex.dot(face) < crease_cos {
                face
            } else {
                vertex
            });
        }
    }
    corners
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
    use crate::material::texture::{Texture, Textures};

    use super::*;

    /// FNV-1a over every geometric field of every loaded triangle, in `prims`
    /// order.
    ///
    /// The golden-image tests compare at 0.95 RMS, so they absorb a reordering
    /// or a drifted transform. This pins the exact bytes the loader produces in
    /// the exact order the BVH build sees them, so any change to `Obj::load`
    /// that leaves these constants alone is geometrically a no-op.
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

    /// FNV-1a over the three shading normals of every loaded triangle, in
    /// `prims` order.
    ///
    /// A second oracle rather than three more fields in `geometry_checksum`:
    /// those constants predate per-vertex normals, and them staying green is
    /// the evidence that nothing but shading moved.
    fn shading_checksum(bvh: &Bvh) -> u64 {
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
            for n in [t.n0, t.n1, t.n2] {
                eat(&n.x.to_le_bytes());
                eat(&n.y.to_le_bytes());
                eat(&n.z.to_le_bytes());
            }
        }
        h
    }

    /// How many distinct decoded images the loaded triangles point at.
    ///
    /// `spider.mtl` has 19 `usemtl` groups over 4 JPEGs, so this proves that
    /// resolving the material once per mesh still hands every triangle the same
    /// shared `Arc`. The flattener dedups textures by `Arc::ptr_eq`, so
    /// collapsing or splitting those Arcs is observable downstream.
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
        assert_eq!(0x7c5a_647c_6fcc_9999, geometry_checksum(&bvh));
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
        assert_eq!(0x2f0c_b703_7f6c_30a2, geometry_checksum(&bvh));
        assert_eq!(0x0063_80cc_a792_03f9, shading_checksum(&bvh));
    }

    #[test]
    fn box_loads_twelve_triangles_with_the_default_material() {
        let bvh = Obj::new("resources/obj/", "box.obj")
            .load(&NopTransformer(), None)
            .unwrap();

        assert_eq!(12, bvh.prims.len());
        assert_eq!(0x0cf6_1486_0adc_e665, geometry_checksum(&bvh));
    }

    #[test]
    fn spider_uses_the_file_normals() {
        let bvh = Obj::new("resources/spider/", "spider.obj")
            .load(&NopTransformer(), None)
            .unwrap();

        // A smooth triangle is one whose three corners disagree with each
        // other and with the face -- either test alone would pass on a mesh
        // handed the geometric normal three times. spider.obj ships 747 `vn`
        // records.
        let smooth = bvh
            .prims
            .iter()
            .filter_map(|p| match p {
                Hittables::Triangle(t) => Some(t),
                _ => None,
            })
            .filter(|t| {
                t.n0.dot(t.n1) < 0.9999 && t.n1.dot(t.n2) < 0.9999 && t.n0.dot(t.normal) < 0.9999
            })
            .count();

        assert!(
            smooth > 1000,
            "expected almost every spider triangle to be smooth-shaded, got {} of {}",
            smooth,
            bvh.prims.len()
        );

        // Every normal is a unit vector, which the octahedral packing in the
        // flattener assumes -- except on the 56 zero-area faces spider.obj
        // ships, whose geometric normal is the NaN `unit()` gives for a zero
        // cross product. Those are rejected by Moeller-Trumbore's determinant
        // test and never reach `resolve_hit`, so the assertion excludes them.
        for prim in &bvh.prims {
            if let Hittables::Triangle(t) = prim {
                if t.normal.x.is_nan() {
                    continue;
                }
                for n in [t.n0, t.n1, t.n2] {
                    assert!((n.length() - 1.).abs() < 1e-9, "not unit: {:?}", n);
                }
            }
        }

        assert_eq!(0xdb5a_d8e7_e2be_37fa, shading_checksum(&bvh));
    }

    #[test]
    fn box_stays_a_box_under_the_default() {
        // Generation runs by default and decides correctly to smooth nothing:
        // box.obj's corners read 48.2 degrees and up against the default's 40.
        // This is the case that caps the default.
        let bvh = Obj::new("resources/obj/", "box.obj")
            .load(&NopTransformer(), None)
            .unwrap();

        for prim in &bvh.prims {
            let t = match prim {
                Hittables::Triangle(t) => t,
                other => panic!("expected only triangles, got {:?}", other),
            };
            for n in [t.n0, t.n1, t.n2] {
                assert!(
                    n.dot(t.normal) > 0.9999999,
                    "corner normal {:?} strayed from the face normal {:?}",
                    n,
                    t.normal
                );
            }
        }
    }

    #[test]
    fn with_flat_shading_is_exactly_the_geometric_normal() {
        // Not the same as the crease deciding not to smooth, which routes the
        // face normal through an accumulate and two `unit()` calls. Only this
        // path is bit-exact, which is what makes the GPU's unconditional
        // interpolation reproduce flat shading rather than approximate it.
        let bvh = Obj::new("resources/obj/", "box.obj")
            .with_flat_shading()
            .load(&NopTransformer(), None)
            .unwrap();

        for prim in &bvh.prims {
            let t = match prim {
                Hittables::Triangle(t) => t,
                other => panic!("expected only triangles, got {:?}", other),
            };
            assert_eq!(t.normal, t.n0);
            assert_eq!(t.normal, t.n1);
            assert_eq!(t.normal, t.n2);
        }
    }

    #[test]
    fn the_crease_angle_is_what_decides() {
        // box.obj's corners span 48.2 to 70.5 degrees, so it is wholly sharp
        // below the first and wholly round above the second. The angle is what
        // decides, not anything hardcoded about boxes.
        let smoothed = Obj::new("resources/obj/", "box.obj")
            .with_generated_normals(75.)
            .load(&NopTransformer(), None)
            .unwrap();
        let smooth = smoothed
            .prims
            .iter()
            .filter_map(|p| match p {
                Hittables::Triangle(t) => Some(t),
                _ => None,
            })
            .filter(|t| t.n0.dot(t.normal) < 0.9999)
            .count();
        assert_eq!(12, smooth);

        let creased = Obj::new("resources/obj/", "box.obj")
            .with_generated_normals(45.)
            .load(&NopTransformer(), None)
            .unwrap();
        for prim in &creased.prims {
            if let Hittables::Triangle(t) = prim {
                assert!(t.n0.dot(t.normal) > 0.9999999);
            }
        }

        // Shading is all that moves, at either setting.
        assert_eq!(0x0cf6_1486_0adc_e665, geometry_checksum(&creased));
        assert_eq!(0x0cf6_1486_0adc_e665, geometry_checksum(&smoothed));
    }

    #[test]
    fn a_curved_mesh_smooths_under_the_default() {
        // The case the default exists for, and the one box.obj cannot show.
        // sphere.obj's corners read 23.8 degrees against the default's 40.
        let bvh = Obj::new("resources/obj/", "sphere.obj")
            .load(&NopTransformer(), None)
            .unwrap();
        assert_eq!(100, bvh.prims.len());

        let smooth = bvh
            .prims
            .iter()
            .filter_map(|p| match p {
                Hittables::Triangle(t) => Some(t),
                _ => None,
            })
            .filter(|t| t.n0.dot(t.normal) < 0.9999 && t.n0.dot(t.n1) < 0.9999)
            .count();
        assert_eq!(100, smooth);

        // Area weighting should land near the sphere's analytic normal. A
        // sanity bound rather than a precision claim: the bands are not
        // equal-area, and the worst corner measures 6.18 degrees.
        for prim in &bvh.prims {
            if let Hittables::Triangle(t) = prim {
                for (n, v) in [(t.n0, t.v0), (t.n1, t.v0 + t.v0v1), (t.n2, t.v0 + t.v0v2)] {
                    let degrees = n.dot(v.unit()).clamp(-1., 1.).acos().to_degrees();
                    assert!(degrees < 8., "corner normal was {} degrees off", degrees);
                }
            }
        }
    }

    #[test]
    fn a_curved_mesh_stays_faceted_with_flat_shading() {
        let bvh = Obj::new("resources/obj/", "sphere.obj")
            .with_flat_shading()
            .load(&NopTransformer(), None)
            .unwrap();

        for prim in &bvh.prims {
            let t = match prim {
                Hittables::Triangle(t) => t,
                other => panic!("expected only triangles, got {:?}", other),
            };
            assert_eq!(t.normal, t.n0);
            assert_eq!(t.normal, t.n1);
            assert_eq!(t.normal, t.n2);
        }
    }

    /// The material of the fixture triangle sitting at `x = 2 * group`.
    ///
    /// Keyed on position because `Bvh::new` reorders, so `prims` order says
    /// nothing about which `usemtl` group a triangle came from.
    fn material_at(bvh: &Bvh, group: usize) -> Materials {
        bvh.prims
            .iter()
            .find_map(|p| match p {
                Hittables::Triangle(t) if t.v0.x == (group * 2) as f64 => Some(t.mat.clone()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no triangle for group {}", group))
    }

    fn materials_fixture() -> Bvh {
        Obj::new("resources/obj/", "materials.obj")
            .load(&NopTransformer(), None)
            .unwrap()
    }

    /// MTL colours are parsed as `f32` and widened, so nothing here is exact.
    fn assert_color(expected: Vec3, tex: &Textures) {
        let actual = tex.color(Uv::default());
        assert!(
            (actual - expected).length() < 1e-6,
            "expected {:?}, got {:?}",
            expected,
            actual
        );
    }

    fn assert_near(expected: f64, actual: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() < tolerance,
            "expected {}, got {}",
            expected,
            actual
        );
    }

    #[test]
    fn kd_alone_is_lambertian() {
        match material_at(&materials_fixture(), 0) {
            Materials::Lambertian(l) => assert_color(Vec3::new(0.8, 0.2, 0.2), &l.albedo),
            other => panic!("expected Lambertian, got {:?}", other),
        }
    }

    #[test]
    fn kd_and_ks_blend_by_luminance() {
        match material_at(&materials_fixture(), 1) {
            Materials::Blend(b) => {
                // Kd (0.2, 0.4, 0.8) is 0.38636 of luminance against Ks 0.5,
                // so the specular arm is picked 0.5 / 0.88636 of the time.
                assert!(
                    (b.blend_factor - 0.564104).abs() < 1e-5,
                    "blend factor was {}",
                    b.blend_factor
                );
                match *b.material_1 {
                    Materials::Lambertian(l) => assert_color(Vec3::new(0.2, 0.4, 0.8), &l.albedo),
                    other => panic!("expected the diffuse arm first, got {:?}", other),
                }
                match *b.material_2 {
                    Materials::Metal(m) => {
                        assert_color(Vec3::new(0.5, 0.5, 0.5), &m.albedo);
                        // Ns 200 through (2 / (Ns + 2))^(1/4).
                        assert_near((2f64 / 202.).powf(0.25), m.fuzz, 1e-9);
                    }
                    other => panic!("expected the specular arm second, got {:?}", other),
                }
            }
            other => panic!("expected Blend, got {:?}", other),
        }
    }

    #[test]
    fn ke_is_a_diffuse_light() {
        // Ke is not one of the fields tobj 4.0.3 parses, so this is also the
        // test that the `unknown_param` read works.
        match material_at(&materials_fixture(), 2) {
            Materials::DiffuseLight(d) => {
                assert_color(Vec3::new(3., 2., 1.), &d.tex);
                assert_eq!(None, d.attenuation_factor);
            }
            other => panic!("expected DiffuseLight, got {:?}", other),
        }
    }

    #[test]
    fn dissolve_is_a_dielectric() {
        match material_at(&materials_fixture(), 3) {
            Materials::Dielectric(d) => {
                assert_near(1.52, d.index_of_refraction, 1e-6);
                assert_color(Vec3::new(0.9, 0.95, 0.9), &d.albedo);
                // MTL has nothing to say about roughness, and Ns 0 would read
                // as fully rough, which is not what a plain `d` means.
                assert_eq!(0., d.roughness);
            }
            other => panic!("expected Dielectric, got {:?}", other),
        }
    }

    #[test]
    fn illum_3_is_metal() {
        match material_at(&materials_fixture(), 4) {
            Materials::Metal(m) => {
                assert_color(Vec3::new(0.9, 0.8, 0.7), &m.albedo);
                assert_near((2f64 / 802.).powf(0.25), m.fuzz, 1e-9);
            }
            other => panic!("expected Metal, got {:?}", other),
        }
    }

    #[test]
    fn tr_is_read_as_one_minus_dissolve() {
        match material_at(&materials_fixture(), 5) {
            Materials::Dielectric(d) => assert_near(1.33, d.index_of_refraction, 1e-6),
            other => panic!("expected Dielectric, got {:?}", other),
        }
    }

    #[test]
    fn illum_1_overrides_the_specular_heuristic() {
        // Ks 0.9 against Kd 0.3 would blend to a mostly specular surface on the
        // heuristics. `illum 1` says diffuse, and that is the authority.
        match material_at(&materials_fixture(), 6) {
            Materials::Lambertian(l) => assert_color(Vec3::new(0.3, 0.3, 0.3), &l.albedo),
            other => panic!("expected Lambertian, got {:?}", other),
        }
    }

    #[test]
    fn the_transparent_illum_family_does_not_override_an_opaque_d() {
        // The case that keeps fireplace_room a room: it marks 21 of its 22
        // materials `illum 4` or `illum 7` while declaring `d` on none of them.
        match material_at(&materials_fixture(), 7) {
            Materials::Blend(b) => {
                assert!(
                    (b.blend_factor - 0.166667).abs() < 1e-5,
                    "blend factor was {}",
                    b.blend_factor
                );
            }
            other => panic!("expected Blend, got {:?}", other),
        }
    }

    #[test]
    fn map_kd_outweighs_the_kd_beside_it() {
        // `Kd 0 0 0` next to a `map_Kd` is what an exporter writes when the
        // texture is the whole diffuse answer. Weighing the blend with that
        // `Kd` would read the surface as pure specular and lose the texture --
        // which is how `fireplace_room`'s wooden table is written.
        match material_at(&materials_fixture(), 8) {
            Materials::Blend(b) => {
                // Ks 0.04 against the texture taken as white.
                assert!(
                    (b.blend_factor - 0.038462).abs() < 1e-5,
                    "blend factor was {}",
                    b.blend_factor
                );
                match *b.material_1 {
                    Materials::Lambertian(l) => {
                        assert!(matches!(l.albedo, Textures::ImageMap(_)))
                    }
                    other => panic!("expected the texture on the diffuse arm, got {:?}", other),
                }
            }
            other => panic!("expected Blend, got {:?}", other),
        }
    }

    #[test]
    fn spider_materials_stay_lambertian() {
        // Every material the repo ships has `Ks 0 0 0` or no `Ks` at all, so a
        // blend weight of zero has to collapse to a `Lambertian` -- otherwise
        // every such scene pays for the blend walk.
        for (path, file) in [
            ("resources/spider/", "spider.obj"),
            ("resources/obj/", "boxWithMat.obj"),
            ("resources/obj/", "triWithNormalMap.obj"),
            ("resources/obj/", "triWithHeightMap.obj"),
        ] {
            let bvh = Obj::new(path, file).load(&NopTransformer(), None).unwrap();
            for prim in &bvh.prims {
                if let Hittables::Triangle(t) = prim {
                    assert!(
                        matches!(t.mat, Materials::Lambertian(_)),
                        "{} produced {:?}",
                        file,
                        t.mat
                    );
                }
            }
        }
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
