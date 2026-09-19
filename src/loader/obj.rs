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
use crate::geo::vec3::{Vec3, ZERO_VECTOR};
use crate::hittable::Bvh;
use crate::hittable::Hittables;
use crate::hittable::Triangle;
use crate::loader::Loader;
use crate::material::texture::{ImageMap, SolidColor, Textures};
use crate::material::{Lambertian, Materials, texture};

/// Crease angle used when the caller does not pick one.
///
/// Measured corner-normal against face-normal (see
/// [`Obj::with_generated_normals`]), so it has to sit above a coarse curved
/// surface and below a box. Measured on the two resources: `sphere.obj`, a
/// 10-by-6 UV sphere, reads 13.5 to 23.8 degrees, and `box.obj` reads 48.2 to
/// 70.5 -- the spread is the triangulation, which gives a cube corner an uneven
/// fan rather than the clean 54.7 of three whole faces. 40 leaves 16 degrees of
/// margin on one and 8 on the other.
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

        // The total is known up front, so the vector never has to grow. At 376 bytes per
        // `Hittables::Triangle` a 250k-triangle mesh is ~94 MB, and growing that by
        // doubling from zero copied about twice that much for nothing.
        //
        // 376 rather than 312: the three shading normals are 72 bytes, 8 of
        // which came out of padding. Host memory only -- the GPU copy is
        // octahedral and fits in `TriangleAttr`'s existing padding.
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

fn vec3_from_mesh_vec(positions: &[f32], offset: usize) -> Vec3 {
    Vec3::new(
        positions[offset] as f64,
        positions[offset + 1] as f64,
        positions[offset + 2] as f64,
    )
}

/// Per-corner shading normals for a mesh that ships none.
///
/// Area-weighted, for free: `(v1-v0) x (v2-v0)` has twice the triangle's area
/// as its magnitude, so accumulating un-normalised is the weighting. It is
/// invariant to how a flat region happens to be triangulated, where an
/// unweighted mean lets a sliver pull the vertex normal toward itself.
///
/// The crease test runs per corner after the accumulation, comparing the
/// vertex normal to that corner's own face. A sharp edge then comes out
/// faceted on both sides while a smooth region sharing the vertex stays
/// smooth. One pass, no adjacency structure -- hence per corner rather than
/// per vertex.
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
        assert_eq!(0x8fc3_bc2d_b210_63bd, shading_checksum(&bvh));
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
    fn spider_uses_the_file_normals() {
        let bvh = Obj::new("resources/spider/", "spider.obj")
            .load(&NopTransformer(), None)
            .unwrap();

        // spider.obj ships 747 `vn` records, and before this they were all
        // discarded. A smooth triangle is one whose three corners disagree with
        // each other and with the face -- either alone would pass on a mesh
        // that had merely been handed the geometric normal three times.
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

        // Every normal is a unit vector, which is what the octahedral packing in
        // the flattener assumes.
        //
        // Except on the 56 zero-area faces spider.obj ships, whose geometric
        // normal is the NaN that `unit()` of a zero cross product gives and has
        // been since long before shading normals existed. Those faces are
        // rejected by Moeller-Trumbore's determinant test and never reach
        // `resolve_hit`, so the assertion excludes them rather than pretending
        // this change could have fixed them.
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

        assert_eq!(0x50e7_a653_f15e_e1b2, shading_checksum(&bvh));
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
        // Not the same as the crease deciding not to smooth: that routes the
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
        assert_eq!(0xe45a_ef74_7f0c_bc65, geometry_checksum(&creased));
        assert_eq!(0xe45a_ef74_7f0c_bc65, geometry_checksum(&smoothed));
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
