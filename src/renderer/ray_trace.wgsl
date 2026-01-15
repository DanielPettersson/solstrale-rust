struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct Sphere {
    center: vec3<f32>,
    radius: f32,
    material_index: u32,
}

struct Triangle {
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
    normal: vec3<f32>,
    material_index: u32,
}

struct Quad {
    Q: vec3<f32>,
    u: vec3<f32>,
    v: vec3<f32>,
    normal: vec3<f32>,
    w: vec3<f32>,
    d: f32,
    material_index: u32,
}

struct BvhNode {
    min: vec3<f32>,
    max: vec3<f32>,
    left_child_index: u32,
    right_child_index: u32,
}

struct Material {
    albedo: vec3<f32>,
    emission: vec3<f32>,
    fuzz: f32,
    refraction_index: f32,
    mat_type: u32,
}

@group(0) @binding(0)
var<storage, read_write> output_buffer: array<vec3<f32>>;

@compute @workgroup_size(64)
fn compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_buffer)) {
        return;
    }
    // Red color
    output_buffer[index] = vec3<f32>(1.0, 0.0, 0.0);
}