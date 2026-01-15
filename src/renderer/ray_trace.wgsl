// ... (structs same as before) ...
struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct Sphere {
    center_and_radius: vec4<f32>,
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
    min_and_left: vec4<u32>,
    max_and_right: vec4<u32>,
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

fn random_in_unit_sphere(state: ptr<function, u32>) -> vec3<f32> {
    for (var i = 0u; i < 100u; i++) {
        let p = vec3<f32>(rand_float(state), rand_float(state), rand_float(state)) * 2.0 - 1.0;
        if (dot(p, p) < 1.0) { return p; }
    }
    return vec3<f32>(0.0);
}

fn random_unit_vector(state: ptr<function, u32>) -> vec3<f32> {
    return normalize(random_in_unit_sphere(state));
}

fn reflect(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return v - 2.0 * dot(v, n) * n;
}

fn refract(uv: vec3<f32>, n: vec3<f32>, etai_over_etat: f32) -> vec3<f32> {
    let cos_theta = min(dot(-uv, n), 1.0);
    let r_out_perp = etai_over_etat * (uv + cos_theta * n);
    let r_out_parallel = -sqrt(abs(1.0 - dot(r_out_perp, r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

struct ScatterRecord {
    attenuation: vec3<f32>,
    scattered: Ray,
    emitted: vec3<f32>,
    is_scattered: bool,
}

fn scatter(r_in: Ray, rec: HitRecord, state: ptr<function, u32>, s_rec: ptr<function, ScatterRecord>) -> bool {
    let mat = materials[rec.material_index];
    (*s_rec).emitted = vec3<f32>(0.0);
    (*s_rec).is_scattered = true;

    if (mat.mat_type == 0u) { // Lambertian
        var scatter_direction = rec.normal + random_unit_vector(state);
        // Catch degenerate scatter direction
        if (all(abs(scatter_direction) < vec3<f32>(1e-8))) {
            scatter_direction = rec.normal;
        }
        (*s_rec).scattered = Ray(rec.p, scatter_direction);
        (*s_rec).attenuation = mat.albedo;
        return true;
    } else if (mat.mat_type == 1u) { // Metal
        let reflected = reflect(normalize(r_in.direction), rec.normal);
        (*s_rec).scattered = Ray(rec.p, reflected + mat.fuzz * random_in_unit_sphere(state));
        (*s_rec).attenuation = mat.albedo;
        return dot((*s_rec).scattered.direction, rec.normal) > 0.0;
    } else if (mat.mat_type == 2u) { // Dielectric
        (*s_rec).attenuation = vec3<f32>(1.0, 1.0, 1.0);
        var refraction_ratio = mat.refraction_index;
        if (rec.front_face) {
            refraction_ratio = 1.0 / mat.refraction_index;
        }

        let unit_direction = normalize(r_in.direction);
        let cos_theta = min(dot(-unit_direction, rec.normal), 1.0);
        let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        var direction: vec3<f32>;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > rand_float(state)) {
            direction = reflect(unit_direction, rec.normal);
        } else {
            direction = refract(unit_direction, rec.normal, refraction_ratio);
        }

        (*s_rec).scattered = Ray(rec.p, direction);
        return true;
    } else if (mat.mat_type == 3u) { // DiffuseLight
        (*s_rec).emitted = mat.emission;
        (*s_rec).is_scattered = false;
        return true;
    }

    return false;
}

fn hit_sphere(r: Ray, s: Sphere, t_min: f32, t_max: f32, rec: ptr<function, HitRecord>) -> bool {
    let center = s.center_and_radius.xyz;
    let radius = s.center_and_radius.w;
    
    let oc = r.origin - center;
    let a = dot(r.direction, r.direction);
    let half_b = dot(oc, r.direction);
    let c = dot(oc, oc) - radius * radius;

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
    let outward_normal = ((*rec).p - center) / radius;
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

fn hit_aabb(r: Ray, min_val: vec3<f32>, max_val: vec3<f32>, t_min_in: f32, t_max_in: f32) -> bool {
    var t_min = t_min_in;
    var t_max = t_max_in;
    
    // Improved precision hit_aabb
    let inv_dir = 1.0 / (r.direction + vec3<f32>(1e-6));
    let t0 = (min_val - r.origin) * inv_dir;
    let t1 = (max_val - r.origin) * inv_dir;
    
    let t_min_v = min(t0, t1);
    let t_max_v = max(t0, t1);
    
    let t_min_max = max(t_min_v.x, max(t_min_v.y, t_min_v.z));
    let t_max_min = min(t_max_v.x, min(t_max_v.y, t_max_v.z));
    
    t_min = max(t_min, t_min_max);
    t_max = min(t_max, t_max_min);
    
    return t_min < t_max;
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
        
        if (node_idx == 0xFFFFFFFFu) { continue; }
        
        let node = nodes[node_idx];

        let node_min = vec3<f32>(
            bitcast<f32>(node.min_and_left.x),
            bitcast<f32>(node.min_and_left.y),
            bitcast<f32>(node.min_and_left.z)
        );
        let node_max = vec3<f32>(
            bitcast<f32>(node.max_and_right.x),
            bitcast<f32>(node.max_and_right.y),
            bitcast<f32>(node.max_and_right.z)
        );

        if (hit_aabb(r, node_min, node_max, t_min, closest_so_far)) {
            let left_idx = node.min_and_left.w;
            let right_idx = node.max_and_right.w;
            
            let is_leaf = (right_idx & 0x80000000u) != 0u;
            if (is_leaf) {
                let prim_type = right_idx & 0x7FFFFFFFu;
                let prim_idx = left_idx;
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
                    stack[stack_ptr] = right_idx;
                    stack_ptr++;
                    stack[stack_ptr] = left_idx;
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
    
    let x = f32(index % config.width);
    let y = f32(index / config.width);

    let u = x / f32(config.width - 1u);
    let v = y / f32(config.height - 1u);

    let ray_direction = normalize(camera.lower_left_corner + u * camera.horizontal + v * camera.vertical - camera.origin);
    let r = Ray(camera.origin, ray_direction);

    var rec: HitRecord;
    if (world_hit(r, 0.001, 10000.0, &rec)) {
        var s_rec: ScatterRecord;
        if (scatter(r, rec, &rng_state, &s_rec)) {
            var color: vec3<f32>;
            if (s_rec.is_scattered) {
                color = s_rec.attenuation;
            } else {
                color = s_rec.emitted;
            }
            output_buffer[index] = color + rand_float(&rng_state) * 0.001;
        } else {
            output_buffer[index] = vec3<f32>(0.0, 0.0, 0.0);
        }
        } else {
            // Black miss color (standard)
            output_buffer[index] = vec3<f32>(0.0, 0.0, 0.0);
        }
    }
    