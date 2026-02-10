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
    area: f32,
    v1: vec3<f32>,
    _pad1: f32,
    v2: vec3<f32>,
    _pad2: f32,
    normal: vec3<f32>,
    material_index: u32,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    uv2: vec2<f32>,
    tangent: vec3<f32>,
    _pad3: f32,
    bi_tangent: vec3<f32>,
    _pad4: f32,
    _pad5: vec4<f32>,
}

struct Quad {
    Q: vec3<f32>,
    area: f32,
    u: vec3<f32>,
    _pad1: f32,
    v: vec3<f32>,
    _pad2: f32,
    normal: vec3<f32>,
    _pad3: f32,
    w: vec3<f32>,
    d: f32,
    material_index: u32,
    tangent: vec3<f32>,
    _pad_align_bitangent: f32,
    bi_tangent: vec3<f32>,
    _pad_end: f32,
    _pad4: vec4<u32>,
}

struct BvhNode {
    min_and_left: vec4<u32>,
    max_and_right: vec4<u32>,
}

struct Material {
    albedo: vec3<f32>,
    attenuation_factor: f32,
    emission: vec3<f32>,
    blend_factor: f32,
    fuzz: f32,
    refraction_index: f32,
    mat_type: u32,
    _padding3: u32,
    texture_index: i32,
    normal_texture_index: i32,
    blend_indices: vec2<u32>,
    albedo_offset: vec2<f32>,
    albedo_scale: vec2<f32>,
    normal_offset: vec2<f32>,
    normal_scale: vec2<f32>,
}

const MAT_LAMBERTIAN = 0u;
const MAT_METAL = 1u;
const MAT_DIELECTRIC = 2u;
const MAT_DIFFUSE_LIGHT = 3u;
const MAT_BLEND = 4u;

struct Camera {
    origin: vec3<f32>,
    lens_radius: f32,
    lower_left_corner: vec3<f32>,
    horizontal: vec3<f32>,
    vertical: vec3<f32>,
    u: vec3<f32>,
    v: vec3<f32>,
}

struct RenderConfig {
    width: u32,
    height: u32,
    sample_count: u32,
    max_depth: u32,
    background_color: vec3<f32>,
    light_count: u32,
}

struct LightRef {
    prim_type: u32,
    prim_index: u32,
}

struct HitRecord {
    t: f32,
    p: vec3<f32>,
    normal: vec3<f32>,
    tangent: vec3<f32>,
    bi_tangent: vec3<f32>,
    material_index: u32,
    front_face: bool,
    uv: vec2<f32>,
}

@group(0) @binding(0)
var<storage, read_write> output_buffer: array<vec4<f32>>;

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

@group(0) @binding(8)
var texture_array: texture_2d<f32>;

@group(0) @binding(9)
var texture_sampler: sampler;

@group(0) @binding(10)
var<storage, read> lights: array<LightRef>;

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
        let p = vec3<f32>(rand_float(state) * 2.0 - 1.0, rand_float(state) * 2.0 - 1.0, rand_float(state) * 2.0 - 1.0);
        if (dot(p, p) < 1.0) { return p; }
    }
    return vec3<f32>(0.0);
}

fn random_in_unit_disk(state: ptr<function, u32>) -> vec3<f32> {
    for (var i = 0u; i < 100u; i++) {
        let p = vec3<f32>(rand_float(state) * 2.0 - 1.0, rand_float(state) * 2.0 - 1.0, 0.0);
        if (dot(p, p) < 1.0) { return p; }
    }
    return vec3<f32>(0.0);
}

fn random_unit_vector(state: ptr<function, u32>) -> vec3<f32> {
    return normalize(random_in_unit_sphere(state));
}

struct ONB {
    u: vec3<f32>,
    v: vec3<f32>,
    w: vec3<f32>,
}

fn onb_from_w(n: vec3<f32>) -> ONB {
    var onb: ONB;
    onb.w = normalize(n);
    var a: vec3<f32>;
    if (abs(onb.w.x) > 0.9) {
        a = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        a = vec3<f32>(1.0, 0.0, 0.0);
    }
    onb.v = normalize(cross(onb.w, a));
    onb.u = cross(onb.w, onb.v);
    return onb;
}

fn onb_local(onb: ONB, a: vec3<f32>) -> vec3<f32> {
    return a.x * onb.u + a.y * onb.v + a.z * onb.w;
}

fn random_cosine_direction(state: ptr<function, u32>) -> vec3<f32> {
    let r1 = rand_float(state);
    let r2 = rand_float(state);

    let phi = 2.0 * 3.14159265359 * r1;
    let x = cos(phi) * sqrt(r2);
    let y = sin(phi) * sqrt(r2);
    let z = sqrt(1.0 - r2);

    return vec3<f32>(x, y, z);
}

fn random_to_sphere(radius: f32, distance_squared: f32, state: ptr<function, u32>) -> vec3<f32> {
    let r1 = rand_float(state);
    let r2 = rand_float(state);
    let z = 1.0 + r2 * (sqrt(abs(1.0 - radius * radius / distance_squared)) - 1.0);

    let phi = 2.0 * 3.14159265359 * r1;
    let zz = sqrt(abs(1.0 - z * z));
    let x = cos(phi) * zz;
    let y = sin(phi) * zz;

    return vec3<f32>(x, y, z);
}

fn triangle_random_direction(t: Triangle, origin: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    var a = rand_float(state);
    var b = rand_float(state);
    if (a + b > 1.0) {
        a = 1.0 - a;
        b = 1.0 - b;
    }
    let p = t.v0 + a * (t.v1 - t.v0) + b * (t.v2 - t.v0);
    return p - origin;
}

fn quad_random_direction(q: Quad, origin: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    let p = q.Q + q.u * rand_float(state) + q.v * rand_float(state);
    return p - origin;
}

fn sphere_random_direction(s: Sphere, origin: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    let center = s.center_and_radius.xyz;
    let radius = s.center_and_radius.w;
    let direction = center - origin;
    let uvw = onb_from_w(direction);
    return onb_local(uvw, random_to_sphere(radius, dot(direction, direction), state));
}

fn light_pdf_value(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
    if (config.light_count == 0u) { return 0.0; }

    var sum = 0.0;
    for (var i = 0u; i < config.light_count; i++) {
        let light = lights[i];
        let r = Ray(origin, direction);
        var rec: HitRecord;

        if (light.prim_type == 0u) { // Sphere
            if (hit_sphere(r, spheres[light.prim_index], 0.001, 1e20, &rec)) {
                let s = spheres[light.prim_index];
                let center = s.center_and_radius.xyz;
                let radius = s.center_and_radius.w;
                let dist_sq = dot(center - origin, center - origin);
                let cos_theta_max = sqrt(abs(1.0 - radius * radius / dist_sq));
                let solid_angle = 2.0 * 3.14159265359 * (1.0 - cos_theta_max);
                sum += 1.0 / solid_angle;
            }
        } else if (light.prim_type == 1u) { // Triangle
            if (hit_triangle(r, triangles[light.prim_index], 0.001, 1e20, &rec)) {
                let tri = triangles[light.prim_index];
                let dist_sq = rec.t * rec.t * dot(direction, direction);
                let cosine = abs(dot(direction, rec.normal) / length(direction));
                sum += dist_sq / (cosine * tri.area);
            }
        } else if (light.prim_type == 2u) { // Quad
            if (hit_quad(r, quads[light.prim_index], 0.001, 1e20, &rec)) {
                let q = quads[light.prim_index];
                let dist_sq = rec.t * rec.t * dot(direction, direction);
                let cosine = abs(dot(direction, rec.normal) / length(direction));
                sum += dist_sq / (cosine * q.area);
            }
        }
    }
    return sum / f32(config.light_count);
}

fn light_random_direction(origin: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    if (config.light_count == 0u) { return vec3<f32>(1.0, 0.0, 0.0); }
    let idx = u32(rand_float(state) * f32(config.light_count));
    let light = lights[min(idx, config.light_count - 1u)];

    if (light.prim_type == 0u) {
        return sphere_random_direction(spheres[light.prim_index], origin, state);
    } else if (light.prim_type == 1u) {
        return triangle_random_direction(triangles[light.prim_index], origin, state);
    } else if (light.prim_type == 2u) {
        return quad_random_direction(quads[light.prim_index], origin, state);
    }
    return vec3<f32>(1.0, 0.0, 0.0);
}

fn cosine_pdf_value(normal: vec3<f32>, direction: vec3<f32>) -> f32 {
    let cos_theta = dot(normalize(direction), normal);
    return max(0.0, cos_theta / 3.14159265359);
}

fn mixture_pdf_value(origin: vec3<f32>, normal: vec3<f32>, direction: vec3<f32>) -> f32 {
    return 0.5 * cosine_pdf_value(normal, direction) + 0.5 * light_pdf_value(origin, direction);
}

fn mixture_pdf_generate(origin: vec3<f32>, normal: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    if (rand_float(state) < 0.5) {
        return light_random_direction(origin, state);
    } else {
        let uvw = onb_from_w(normal);
        return onb_local(uvw, random_cosine_direction(state));
    }
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
    attenuation_factor: f32,
    is_scattered: bool,
    pdf_value: f32,
}

fn scatter(r_in: Ray, rec: HitRecord, state: ptr<function, u32>, s_rec: ptr<function, ScatterRecord>) -> bool {
    var mat_idx = rec.material_index;
    for (var i = 0u; i < 10u; i++) {
        let material = materials[mat_idx];
        if (material.mat_type == MAT_BLEND) {
            if (rand_float(state) > material.blend_factor) {
                mat_idx = material.blend_indices.x;
            } else {
                mat_idx = material.blend_indices.y;
            }
        } else {
            break;
        }
    }

    let material = materials[mat_idx];
    (*s_rec).emitted = vec3<f32>(0.0);
    (*s_rec).is_scattered = true;
    (*s_rec).attenuation_factor = 0.0;
    (*s_rec).pdf_value = 1.0;

    var albedo = material.albedo;
    if (material.texture_index >= 0) {
        let uv = vec2<f32>(fract(abs(rec.uv.x)), 1.0 - fract(abs(rec.uv.y)));
        let uv_atlas = material.albedo_offset + uv * material.albedo_scale;
        albedo = textureSampleLevel(texture_array, texture_sampler, uv_atlas, 0.0).rgb;
    }

    var normal = rec.normal;
    if (material.normal_texture_index >= 0) {
         let uv = vec2<f32>(fract(abs(rec.uv.x)), 1.0 - fract(abs(rec.uv.y)));
         let uv_atlas = material.normal_offset + uv * material.normal_scale;
         let map_color = textureSampleLevel(texture_array, texture_sampler, uv_atlas, 0.0).rgb;
         let map_n = map_color * 2.0 - 1.0;
         normal = normalize(map_n.x * rec.tangent + map_n.y * rec.bi_tangent + map_n.z * rec.normal);
    }

    if (material.mat_type == MAT_LAMBERTIAN) { // Lambertian
        let direction = mixture_pdf_generate(rec.p, normal, state);
        (*s_rec).scattered = Ray(rec.p, direction);
        (*s_rec).attenuation = albedo;
        let scattering_pdf = cosine_pdf_value(normal, direction);
        let pdf_val = mixture_pdf_value(rec.p, normal, direction);
        (*s_rec).pdf_value = scattering_pdf / pdf_val;
        return true;
    } else if (material.mat_type == MAT_METAL) { // Metal
        let reflected = reflect(normalize(r_in.direction), normal);
        (*s_rec).scattered = Ray(rec.p, reflected + material.fuzz * random_in_unit_sphere(state));
        (*s_rec).attenuation = albedo;
        return dot((*s_rec).scattered.direction, normal) > 0.0;
    } else if (material.mat_type == MAT_DIELECTRIC) { // Dielectric
        (*s_rec).attenuation = vec3<f32>(1.0, 1.0, 1.0);
        var refraction_ratio = material.refraction_index;
        if (rec.front_face) {
            refraction_ratio = 1.0 / material.refraction_index;
        }

        let unit_direction = normalize(r_in.direction);
        let cos_theta = min(dot(-unit_direction, normal), 1.0);
        let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        var direction: vec3<f32>;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > rand_float(state)) {
            direction = reflect(unit_direction, normal);
        } else {
            direction = refract(unit_direction, normal, refraction_ratio);
        }

        (*s_rec).scattered = Ray(rec.p, direction);
        return true;
    } else if (material.mat_type == MAT_DIFFUSE_LIGHT) { // DiffuseLight
        if (rec.front_face) {
            (*s_rec).emitted = material.emission;
        } else {
            (*s_rec).emitted = vec3<f32>(0.0);
        }
        (*s_rec).is_scattered = false;
        (*s_rec).attenuation_factor = material.attenuation_factor;
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

    let theta = acos(-outward_normal.y);
    let phi = atan2(-outward_normal.z, outward_normal.x) + 3.14159265359;
    let u = phi / (2.0 * 3.14159265359);
    let v = theta / 3.14159265359;
    (*rec).uv = vec2<f32>(u, v);

    (*rec).tangent = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), outward_normal));
    (*rec).bi_tangent = cross(outward_normal, (*rec).tangent);

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

    let w = 1.0 - u - v;
    (*rec).uv = w * t.uv0 + u * t.uv1 + v * t.uv2;

    (*rec).tangent = t.tangent;
    (*rec).bi_tangent = t.bi_tangent;

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
    (*rec).uv = vec2<f32>(alpha, beta);

    (*rec).tangent = q.tangent;
    (*rec).bi_tangent = q.bi_tangent;

    return true;
}

fn hit_aabb(r: Ray, min_val: vec3<f32>, max_val: vec3<f32>, t_min_in: f32, t_max_in: f32) -> bool {
    var t_min = t_min_in;
    var t_max = t_max_in;
    
    let inv_dir = 1.0 / (r.direction + vec3<f32>(1e-9));
    let t0 = (min_val - r.origin) * inv_dir;
    let t1 = (max_val - r.origin) * inv_dir;
    
    let t_near = min(t0, t1);
    let t_far = max(t0, t1);
    
    t_min = max(t_min, max(t_near.x, max(t_near.y, t_near.z)));
    t_max = min(t_max, min(t_far.x, min(t_far.y, t_far.z)));
    
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
        
        if (node_idx == 0x0FFFFFFFu) { continue; }
        
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

    let u = (x + rand_float(&rng_state)) / f32(config.width - 1u);
    let v = 1.0 - (y + rand_float(&rng_state)) / f32(config.height - 1u);

    var offset = vec3<f32>(0.0);
    if (camera.lens_radius > 0.0) {
        let rd = random_in_unit_disk(&rng_state) * camera.lens_radius;
        offset = camera.u * rd.x + camera.v * rd.y;
    }

    let ray_direction = camera.lower_left_corner + u * camera.horizontal + v * camera.vertical - camera.origin - offset;
    var r = Ray(camera.origin + offset, ray_direction);

    var accumulated_color = vec3<f32>(0.0);
    var current_attenuation = vec3<f32>(1.0);
    var accumulated_ray_length = 0.0;

    for (var depth = 0u; depth < config.max_depth; depth++) {
        var rec: HitRecord;
        if (world_hit(r, 0.001, 10000.0, &rec)) {
            var s_rec: ScatterRecord;
            if (scatter(r, rec, &rng_state, &s_rec)) {
                accumulated_ray_length += rec.t;
                
                var emitted = s_rec.emitted;
                if (s_rec.attenuation_factor > 0.0) {
                    emitted *= 1.0 / (1.0 + s_rec.attenuation_factor * accumulated_ray_length);
                }
                
                accumulated_color += emitted * current_attenuation;
                
                if (s_rec.is_scattered) {
                    current_attenuation *= s_rec.attenuation * s_rec.pdf_value;
                    r = Ray(s_rec.scattered.origin, normalize(s_rec.scattered.direction));
                } else {
                    break;
                }
            } else {
                break;
            }
        } else {
            accumulated_color += config.background_color * current_attenuation;
            break;
        }

        if (max(current_attenuation.x, max(current_attenuation.y, current_attenuation.z)) < 0.0001) {
            break;
        }
    }

    if (config.sample_count <= 1u) {
        output_buffer[index] = vec4<f32>(accumulated_color, 1.0);
    } else {
        let weight = 1.0 / f32(config.sample_count);
        let prev_color = output_buffer[index].xyz;
        output_buffer[index] = vec4<f32>(prev_color * (1.0 - weight) + accumulated_color * weight, 1.0);
    }
}
