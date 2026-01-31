//! Utilities for flattening the scene graph into linear buffers for the GPU

use crate::geo::Uv;
use crate::geo::vec3::{Vec3, ZERO_VECTOR};
use crate::hittable::{Bvh, BvhItem, Hittable, Hittables};
use crate::material::Materials;
use crate::material::texture::{Texture, Textures};
use crate::renderer::Scene;
use crate::renderer::gpu_data::{
    BvhNode as GpuBvhNode, Material as GpuMaterial, Quad as GpuQuad, Sphere as GpuSphere,
    Triangle as GpuTriangle,
};

/// Container for all scene data flattened for the GPU
pub struct SceneData {
    /// Flattened BVH nodes
    pub nodes: Vec<GpuBvhNode>,
    /// Spheres
    pub spheres: Vec<GpuSphere>,
    /// Triangles
    pub triangles: Vec<GpuTriangle>,
    /// Quads
    pub quads: Vec<GpuQuad>,
    /// Materials
    pub materials: Vec<GpuMaterial>,
}

/// Flattens the scene into linear buffers
pub fn flatten_scene(scene: &Scene) -> SceneData {
    let mut data = SceneData {
        nodes: Vec::new(),
        spheres: Vec::new(),
        triangles: Vec::new(),
        quads: Vec::new(),
        materials: Vec::new(),
    };

    // Process world
    match &scene.world {
        Hittables::Bvh(bvh) => {
            process_node(bvh, &mut data);
        }
        _ => {
            let (prim_index, prim_type) = add_primitive(&scene.world, &mut data);
            let bbox = scene.world.bounding_box();

            let flag = 0x80000000;
            data.nodes.push(GpuBvhNode {
                min_and_left: [
                    (bbox.x.min as f32).to_bits(),
                    (bbox.y.min as f32).to_bits(),
                    (bbox.z.min as f32).to_bits(),
                    prim_index,
                ],
                max_and_right: [
                    (bbox.x.max as f32).to_bits(),
                    (bbox.y.max as f32).to_bits(),
                    (bbox.z.max as f32).to_bits(),
                    prim_type | flag,
                ],
            });
        }
    }

    data
}

fn process_node(bvh: &Bvh, data: &mut SceneData) -> u32 {
    let index = data.nodes.len() as u32;
    // Reserve slot
    data.nodes.push(GpuBvhNode {
        min_and_left: [0; 4],
        max_and_right: [0; 4],
    });

    let left_idx = process_item(&bvh.left, data);
    let right_idx = process_item(&bvh.right, data);

    let bbox = bvh.bounding_box();

    data.nodes[index as usize] = GpuBvhNode {
        min_and_left: [
            (bbox.x.min as f32).to_bits(),
            (bbox.y.min as f32).to_bits(),
            (bbox.z.min as f32).to_bits(),
            left_idx,
        ],
        max_and_right: [
            (bbox.x.max as f32).to_bits(),
            (bbox.y.max as f32).to_bits(),
            (bbox.z.max as f32).to_bits(),
            right_idx,
        ],
    };

    index
}

fn process_item(item: &BvhItem, data: &mut SceneData) -> u32 {
    match item {
        BvhItem::Node(bvh) => process_node(bvh, data),
        BvhItem::Leaf(hittable) => {
            let (prim_index, prim_type) = add_primitive(hittable, data);

            let index = data.nodes.len() as u32;
            let bbox = hittable.bounding_box();

            let flag = 0x80000000;

            data.nodes.push(GpuBvhNode {
                min_and_left: [
                    (bbox.x.min as f32).to_bits(),
                    (bbox.y.min as f32).to_bits(),
                    (bbox.z.min as f32).to_bits(),
                    prim_index,
                ],
                max_and_right: [
                    (bbox.x.max as f32).to_bits(),
                    (bbox.y.max as f32).to_bits(),
                    (bbox.z.max as f32).to_bits(),
                    prim_type | flag,
                ],
            });

            index
        }
        BvhItem::None => 0x0FFFFFFF,
    }
}

fn add_primitive(hittable: &Hittables, data: &mut SceneData) -> (u32, u32) {
    match hittable {
        Hittables::Sphere(s) => {
            let index = data.spheres.len() as u32;
            let mat_idx = add_material(&s.mat, data);
            data.spheres.push(GpuSphere {
                center_and_radius: [
                    s.center.x as f32,
                    s.center.y as f32,
                    s.center.z as f32,
                    s.radius as f32,
                ],
                material_index: mat_idx,
                _padding: [0; 3],
            });
            (index, 0) // Type 0 = Sphere
        }
        Hittables::Triangle(t) => {
            let index = data.triangles.len() as u32;
            let mat_idx = add_material(&t.mat, data);
            let v1 = t.v0 + t.v0v1;
            let v2 = t.v0 + t.v0v2;
            data.triangles.push(GpuTriangle {
                v0: to_array(t.v0),
                _pad0: 0.0,
                v1: to_array(v1),
                _pad1: 0.0,
                v2: to_array(v2),
                _pad2: 0.0,
                normal: to_array(t.normal),
                material_index: mat_idx,
            });
            (index, 1) // Type 1 = Triangle
        }
        Hittables::Quad(q) => {
            let index = data.quads.len() as u32;
            let mat_idx = add_material(&q.mat, data);
            data.quads.push(GpuQuad {
                q: to_array(q.q),
                _pad0: 0.0,
                u: to_array(q.u),
                _pad1: 0.0,
                v: to_array(q.v),
                _pad2: 0.0,
                normal: to_array(q.normal),
                _pad3: 0.0,
                w: to_array(q.w),
                d: q.d as f32,
                material_index: mat_idx,
                _pad4: [0; 3],
            });
            (index, 2) // Type 2 = Quad
        }
        Hittables::Bvh(_) => (0xFFFFFFFF, 0),
        Hittables::ConstantMedium(_) => (0xFFFFFFFF, 0),
    }
}

fn add_material(material: &Materials, data: &mut SceneData) -> u32 {
    let index = data.materials.len() as u32;

    let (albedo, emission, fuzz, ref_idx, mat_type, attenuation_factor) = match material {
        Materials::Lambertian(m) => (sample_texture(&m.albedo), ZERO_VECTOR, 0.0, 0.0, 0, 0.0),
        Materials::Metal(m) => (
            sample_texture(&m.albedo),
            ZERO_VECTOR,
            m.fuzz as f32,
            0.0,
            1,
            0.0,
        ),
        Materials::Dielectric(m) => (
            sample_texture(&m.albedo),
            ZERO_VECTOR,
            0.0,
            m.index_of_refraction as f32,
            2,
            0.0,
        ),
        Materials::DiffuseLight(m) => (
            ZERO_VECTOR,
            sample_emission(m),
            0.0,
            0.0,
            3,
            m.attenuation_factor.unwrap_or(0.0) as f32,
        ),
        Materials::Isotropic(_) => (ZERO_VECTOR, ZERO_VECTOR, 0.0, 0.0, 0, 0.0),
        Materials::Blend(_) => (ZERO_VECTOR, ZERO_VECTOR, 0.0, 0.0, 0, 0.0),
    };

    data.materials.push(GpuMaterial {
        albedo: to_array(albedo),
        attenuation_factor,
        emission: to_array(emission),
        _padding2: 0.0,
        fuzz,
        refraction_index: ref_idx,
        mat_type,
        _padding3: 0,
    });

    index
}

fn sample_texture(tex: &Textures) -> Vec3 {
    tex.color(Uv::default())
}

fn sample_emission(mat: &crate::material::DiffuseLight) -> Vec3 {
    mat.tex.color(Uv::default())
}

fn to_array(v: Vec3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

#[cfg(test)]
mod tests {
    use crate::geo::vec3::Vec3;
    use crate::hittable::{Bvh, Hittables, Sphere};
    use crate::material::texture::SolidColor;
    use crate::material::{Lambertian, Materials};
    use crate::renderer::scene_flattener::flatten_scene;
    use crate::renderer::{RenderConfig, Scene};

    #[test]
    fn test_flatten_scene_simple() {
        let mat =
            Materials::Lambertian(Lambertian::new(SolidColor::new(1.0, 0.0, 0.0).into(), None));
        let sphere = Sphere::new(Vec3::new(0., 0., -2.), 1.0, mat);
        let scene = Scene {
            world: Hittables::Bvh(Bvh::new(vec![Hittables::Sphere(sphere)])),
            camera: Default::default(),
            background_color: Default::default(),
            render_config: RenderConfig::default(),
        };

        let data = flatten_scene(&scene);

        assert_eq!(data.spheres.len(), 1);
        assert_eq!(data.materials.len(), 1);
        assert_eq!(data.nodes.len(), 2);

        // Check sphere data
        let s = &data.spheres[0];
        assert_eq!(s.center_and_radius, [0.0, 0.0, -2.0, 1.0]);
        assert_eq!(s.material_index, 0);

        // Check root node (inner)
        let n0 = &data.nodes[0];
        assert_eq!(n0.min_and_left[3], 1);
        assert_eq!(n0.max_and_right[3], 0x0FFFFFFF);

        // Check leaf node
        let n1 = &data.nodes[1];
        // left_child_index should point to sphere index (0)
        assert_eq!(n1.min_and_left[3], 0);
        // right_child_index should have flag set and type 0 (Sphere)
        assert_eq!(n1.max_and_right[3], 0x80000000 | 0);
    }

    #[test]
    fn test_flatten_scene_nested_bvh() {
        let mat =
            Materials::Lambertian(Lambertian::new(SolidColor::new(1.0, 0.0, 0.0).into(), None));
        let sphere = Sphere::new(Vec3::new(0., 0., -2.), 1.0, mat.clone());
        
        let mut sub_world: Vec<Hittables> = Vec::new();
        sub_world.push(Hittables::Sphere(sphere.clone()));
        let bvh = Bvh::new(sub_world);

        let scene = Scene {
            world: Hittables::Bvh(Bvh::new(vec![Hittables::Bvh(bvh)])),
            camera: Default::default(),
            background_color: Default::default(),
            render_config: RenderConfig::default(),
        };

        let data = flatten_scene(&scene);

        // We expect:
        // 1. Root Bvh Node
        // 2. Nested Bvh Node
        // 3. Leaf Bvh Node (containing the sphere)
        // Total: 3 nodes
        // Spheres: 1
        
        assert_eq!(data.spheres.len(), 1, "Should have 1 sphere");
        assert_eq!(data.nodes.len(), 3, "Should have 3 nodes");
    }
}
