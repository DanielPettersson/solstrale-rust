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

struct RenderConfig {
    width: u32,
    height: u32,
    sample_count: u32,
}

struct HitRecord {
    t: f32,
    p: vec3<f32>,
    normal: vec3<f32>,
    material_index: u32,
    front_face: bool,
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

@group(0) @binding(7)
var<uniform> config: RenderConfig;

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_float(state: ptr<function, u32>) -> f32 {
    *state = pcg_hash(*state);
    return f32(*state) / 4294967296.0;
}

fn ray_at(r: Ray, t: f32) -> vec3<f32> {
    return r.origin + t * r.direction;
}

fn hit_sphere(r: Ray, s: Sphere, t_min: f32, t_max: f32, rec: ptr<function, HitRecord>) -> bool {
    let oc = r.origin - s.center;
    let a = dot(r.direction, r.direction);
    let half_b = dot(oc, r.direction);
    let c = dot(oc, oc) - s.radius * s.radius;

    let discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0) { return false; }
    let sqrtd = sqrt(discriminant);

    var root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root) {
            return false;
        }
    }

    (*rec).t = root;
    (*rec).p = ray_at(r, root);
    let outward_normal = ((*rec).p - s.center) / s.radius;
    (*rec).front_face = dot(r.direction, outward_normal) < 0.0;
    if ((*rec).front_face) {
        (*rec).normal = outward_normal;
    } else {
        (*rec).normal = -outward_normal;
    }
    (*rec).material_index = s.material_index;

    return true;
}

fn hit_triangle(r: Ray, t: Triangle, t_min: f32, t_max: f32, rec: ptr<function, HitRecord>) -> bool {
    let v0v1 = t.v1 - t.v0;
    let v0v2 = t.v2 - t.v0;
    let p_vec = cross(r.direction, v0v2);
    let det = dot(v0v1, p_vec);

    if (abs(det) < 1e-8) { return false; }

    let inv_det = 1.0 / det;
    let t_vec = r.origin - t.v0;
    let u = dot(t_vec, p_vec) * inv_det;
    if (u < 0.0 || u > 1.0) { return false; }

    let q_vec = cross(t_vec, v0v1);
    let v = dot(r.direction, q_vec) * inv_det;
    if (v < 0.0 || u + v > 1.0) { return false; }

    let tt = dot(v0v2, q_vec) * inv_det;
    if (tt < t_min || tt > t_max) { return false; }

    (*rec).t = tt;
    (*rec).p = ray_at(r, tt);
    (*rec).front_face = dot(r.direction, t.normal) < 0.0;
    if ((*rec).front_face) {
        (*rec).normal = t.normal;
    } else {
        (*rec).normal = -t.normal;
    }
    (*rec).material_index = t.material_index;

    return true;
}

fn hit_quad(r: Ray, q: Quad, t_min: f32, t_max: f32, rec: ptr<function, HitRecord>) -> bool {
    let denom = dot(q.normal, r.direction);
    if (abs(denom) < 1e-8) { return false; }

    let t = (q.d - dot(q.normal, r.origin)) / denom;
    if (t < t_min || t > t_max) { return false; }

    let p = ray_at(r, t);
    let planar_hit_point_vector = p - q.Q;
    let alpha = dot(q.w, cross(planar_hit_point_vector, q.v));
    let beta = dot(q.w, cross(q.u, planar_hit_point_vector));

    if (alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0) { return false; }

    (*rec).t = t;
    (*rec).p = p;
    (*rec).front_face = dot(r.direction, q.normal) < 0.0;
    if ((*rec).front_face) {
        (*rec).normal = q.normal;
    } else {
        (*rec).normal = -q.normal;
    }
    (*rec).material_index = q.material_index;

    return true;
}

fn hit_aabb(r: Ray, min: vec3<f32>, max: vec3<f32>, t_min_in: f32, t_max_in: f32) -> bool {
    var t_min = t_min_in;
    var t_max = t_max_in;
    let inv_dir = 1.0 / r.direction;

    for (var i = 0u; i < 3u; i++) {
        var t0 = (min[i] - r.origin[i]) * inv_dir[i];
        var t1 = (max[i] - r.origin[i]) * inv_dir[i];
        if (inv_dir[i] < 0.0) {
            let tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        t_min = max(t0, t_min);
        t_max = min(t1, t_max);
        if (t_max <= t_min) { return false; }
    }
    return true;
}

fn world_hit(r: Ray, t_min: f32, t_max: f32, rec: ptr<function, HitRecord>) -> bool {
    var hit_anything = false;
    var closest_so_far = t_max;
    
    let num_nodes = arrayLength(&nodes);
    if (num_nodes == 0u) { return false; }

    var stack: array<u32, 64>;
    var stack_ptr = 0u;
    stack[stack_ptr] = 0u;
    stack_ptr++;

    var safety_counter = 0u;
    while (stack_ptr > 0u && safety_counter < 1000u) {
        safety_counter++;
        stack_ptr--;
        let node_idx = stack[stack_ptr];
        let node = nodes[node_idx];

        if (hit_aabb(r, node.min, node.max, t_min, closest_so_far)) {
            let is_leaf = (node.right_child_index & 0x80000000u) != 0u;
            if (is_leaf) {
                let prim_type = node.right_child_index & 0x7FFFFFFFu;
                let prim_idx = node.left_child_index;
                var temp_rec: HitRecord;
                var hit = false;

                if (prim_type == 0u) {
                    hit = hit_sphere(r, spheres[prim_idx], t_min, closest_so_far, &temp_rec);
                } else if (prim_type == 1u) {
                    hit = hit_triangle(r, triangles[prim_idx], t_min, closest_so_far, &temp_rec);
                } else if (prim_type == 2u) {
                    hit = hit_quad(r, quads[prim_idx], t_min, closest_so_far, &temp_rec);
                }

                if (hit) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    *rec = temp_rec;
                }
                        } else {
                            if (stack_ptr < 62u) {
                                stack[stack_ptr] = node.right_child_index;
                                stack_ptr++;
                                stack[stack_ptr] = node.left_child_index;
                                stack_ptr++;
                            }
                        }
                    }
                }
            
                return hit_anything;
            }
            
            @compute @workgroup_size(64)
            fn compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                if (index >= arrayLength(&output_buffer)) {
                    return;
                }
            
                var rng_state = index + config.sample_count * 712371u;
                let r_val = rand_float(&rng_state);
                output_buffer[index] = vec3<f32>(r_val, r_val, r_val);
            }
            
    