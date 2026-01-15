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



struct Camera {

    origin: vec3<f32>,

    lower_left_corner: vec3<f32>,

    horizontal: vec3<f32>,

    vertical: vec3<f32>,

    lens_radius: f32,

}



@group(0) @binding(0)

var<storage, read_write> output_buffer: array<vec3<f32>>;



@group(0) @binding(1)

var<storage, read> nodes: array<BvhNode>;



@group(0) @binding(2)

var<storage, read> spheres: array<Sphere>;



@group(0) @binding(3)

var<storage, read> triangles: array<Triangle>;



@group(0) @binding(4)

var<storage, read> quads: array<Quad>;



@group(0) @binding(5)

var<storage, read> materials: array<Material>;



@group(0) @binding(6)

var<uniform> camera: Camera;



struct RenderConfig {

    width: u32,

    height: u32,

}



@group(0) @binding(7)

var<uniform> config: RenderConfig;



@compute @workgroup_size(64)

fn compute(@builtin(global_invocation_id) global_id: vec3<u32>) {

    let index = global_id.x;

    if (index >= arrayLength(&output_buffer)) {

        return;

    }



    let x = f32(index % config.width);

    let y = f32(index / config.width);



    let u = x / f32(config.width - 1u);

    let v = y / f32(config.height - 1u);



    let ray_direction = normalize(camera.lower_left_corner + u * camera.horizontal + v * camera.vertical - camera.origin);



    // Visualize ray direction

    output_buffer[index] = ray_direction * 0.5 + 0.5;

}
