struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct Sphere {
    center_and_radius: vec4<f32>,
    material_index: u32,
}

// Geometry read by the traversal inner loop. Edges rather than absolute
// vertices, because that is what Moeller-Trumbore wants and what the CPU has.
struct TrianglePos {
    v0: vec3<f32>,
    _pad0: f32,
    e1: vec3<f32>,
    _pad1: f32,
    e2: vec3<f32>,
    _pad2: f32,
}

// Shading attributes, read once per ray after traversal has settled on a hit.
struct TriangleAttr {
    normal: vec3<f32>,
    material_index: u32,
    tangent: vec3<f32>,
    area: f32,
    bi_tangent: vec3<f32>,
    _pad0: f32,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    uv2: vec2<f32>,
    _pad1: vec2<f32>,
}

struct QuadPos {
    Q: vec3<f32>,
    d: f32,
    u: vec3<f32>,
    _pad0: f32,
    v: vec3<f32>,
    _pad1: f32,
    normal: vec3<f32>,
    _pad2: f32,
    w: vec3<f32>,
    _pad3: f32,
}

struct QuadAttr {
    tangent: vec3<f32>,
    area: f32,
    bi_tangent: vec3<f32>,
    material_index: u32,
}

struct BvhNode {
    left_min: vec3<f32>,
    left_meta: u32,
    left_max: vec3<f32>,
    right_meta: u32,
    right_min: vec3<f32>,
    _pad0: u32,
    right_max: vec3<f32>,
    _pad1: u32,
}

const LEAF_FLAG = 0x80000000u;
const LEAF_COUNT_SHIFT = 24u;
const LEAF_COUNT_MASK = 0x7Fu;
const LEAF_OFFSET_MASK = 0x00FFFFFFu;

const PRIM_TYPE_SHIFT = 30u;
const PRIM_INDEX_MASK = 0x3FFFFFFFu;

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

const CLAMPING_THRESHOLD = 3.5;

const PI = 3.14159265359;
const TWO_PI = 6.28318530718;

// Path depth at which Russian roulette starts. Below it every path survives,
// so the cheap early bounces that carry most of the energy are never cut.
const RR_MIN_DEPTH = 3u;
// Floor on the survival probability, so a dark path still terminates promptly
// without the weight correction exploding.
const RR_MIN_SURVIVAL = 0.05;

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
    // Samples already accumulated before this dispatch.
    sample_count: u32,
    max_depth: u32,
    background_color: vec3<f32>,
    light_count: u32,
    samples_per_batch: u32,
    // Scalar pads, not a vec3: a vec3 here would align to 16 and push the
    // struct to 64 bytes, which no longer matches the Rust mirror.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct LightRef {
    prim_type: u32,
    prim_index: u32,
}

// What traversal actually tracks: enough to identify the winning primitive and
// reconstruct its shading data afterwards, and nothing more. Keeping this
// small is what removes the ~25-float HitRecord copy from the inner loop.
struct HitRef {
    t: f32,
    prim_type: u32,
    prim_idx: u32,
    bary: vec2<f32>,
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
var<storage, read> triangle_pos: array<TrianglePos>;

@group(0) @binding(4)
var<storage, read> quad_pos: array<QuadPos>;

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

@group(0) @binding(11)
var<storage, read> prim_refs: array<u32>;

@group(0) @binding(12)
var<storage, read> triangle_attr: array<TriangleAttr>;

@group(0) @binding(13)
var<storage, read> quad_attr: array<QuadAttr>;

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

// Closed-form samplers.
//
// These were rejection loops of up to 100 iterations. On a GPU every lane in a
// wavefront waits for its unluckiest neighbour, so a loop with a ~48% per-
// iteration rejection rate costs far more than the arithmetic below.

fn random_unit_vector(state: ptr<function, u32>) -> vec3<f32> {
    let z = 1.0 - 2.0 * rand_float(state);
    let r = sqrt(max(0.0, 1.0 - z * z));
    let phi = TWO_PI * rand_float(state);
    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

fn random_in_unit_sphere(state: ptr<function, u32>) -> vec3<f32> {
    // cbrt of a uniform variate makes the radius uniform by volume.
    let dir = random_unit_vector(state);
    return dir * pow(rand_float(state), 1.0 / 3.0);
}

fn random_in_unit_disk(state: ptr<function, u32>) -> vec3<f32> {
    // sqrt of a uniform variate makes the radius uniform by area.
    let r = sqrt(rand_float(state));
    let phi = TWO_PI * rand_float(state);
    return vec3<f32>(r * cos(phi), r * sin(phi), 0.0);
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

fn triangle_random_direction(t: TrianglePos, origin: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    var a = rand_float(state);
    var b = rand_float(state);
    if (a + b > 1.0) {
        a = 1.0 - a;
        b = 1.0 - b;
    }
    let p = t.v0 + a * t.e1 + b * t.e2;
    return p - origin;
}

fn quad_random_direction(q: QuadPos, origin: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
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

// Evaluates the mixture PDF's light term.
//
// Uses the distance-only intersection variants: this runs for every light on
// every diffuse bounce, and the full UV/tangent work the shared hit routines
// used to do was discarded here every single time.
fn light_pdf_value(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
    if (config.light_count == 0u) { return 0.0; }

    let r = Ray(origin, direction);
    let dir_len = length(direction);
    let dir_len_sq = dot(direction, direction);
    var sum = 0.0;

    for (var i = 0u; i < config.light_count; i++) {
        let light = lights[i];
        var t_hit = 0.0;
        var bary = vec2<f32>(0.0);

        if (light.prim_type == 0u) { // Sphere
            if (hit_sphere_t(r, spheres[light.prim_index], 0.001, 1e20, &t_hit)) {
                let s = spheres[light.prim_index];
                let center = s.center_and_radius.xyz;
                let radius = s.center_and_radius.w;
                let dist_sq = dot(center - origin, center - origin);
                let cos_theta_max = sqrt(abs(1.0 - radius * radius / dist_sq));
                let solid_angle = 2.0 * 3.14159265359 * (1.0 - cos_theta_max);
                sum += 1.0 / solid_angle;
            }
        } else if (light.prim_type == 1u) { // Triangle
            if (hit_triangle_t(r, triangle_pos[light.prim_index], 0.001, 1e20, &t_hit, &bary)) {
                let attr = triangle_attr[light.prim_index];
                let dist_sq = t_hit * t_hit * dir_len_sq;
                let cosine = abs(dot(direction, attr.normal) / dir_len);
                sum += dist_sq / (cosine * attr.area);
            }
        } else if (light.prim_type == 2u) { // Quad
            if (hit_quad_t(r, quad_pos[light.prim_index], 0.001, 1e20, &t_hit, &bary)) {
                let attr = quad_attr[light.prim_index];
                let normal = quad_pos[light.prim_index].normal;
                let dist_sq = t_hit * t_hit * dir_len_sq;
                let cosine = abs(dot(direction, normal) / dir_len);
                sum += dist_sq / (cosine * attr.area);
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
        return triangle_random_direction(triangle_pos[light.prim_index], origin, state);
    } else if (light.prim_type == 2u) {
        return quad_random_direction(quad_pos[light.prim_index], origin, state);
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

// ---------------------------------------------------------------------------
// Intersection
//
// Traversal only needs the distance and (for triangles/quads) the surface
// parameters. UVs, tangent frames and the sphere's acos/atan2 mapping used to
// be computed for every candidate hit and then thrown away by the next closer
// one; they are now derived once, in resolve_hit, from the winning primitive.
// ---------------------------------------------------------------------------

fn hit_sphere_t(r: Ray, s: Sphere, t_min: f32, t_max: f32, t_out: ptr<function, f32>) -> bool {
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

    *t_out = root;
    return true;
}

fn hit_triangle_t(
    r: Ray,
    t: TrianglePos,
    t_min: f32,
    t_max: f32,
    t_out: ptr<function, f32>,
    bary_out: ptr<function, vec2<f32>>,
) -> bool {
    let p_vec = cross(r.direction, t.e2);
    let det = dot(t.e1, p_vec);

    if (abs(det) < 1e-8) { return false; }

    let inv_det = 1.0 / det;
    let t_vec = r.origin - t.v0;
    let u = dot(t_vec, p_vec) * inv_det;
    if (u < 0.0 || u > 1.0) { return false; }

    let q_vec = cross(t_vec, t.e1);
    let v = dot(r.direction, q_vec) * inv_det;
    if (v < 0.0 || u + v > 1.0) { return false; }

    let tt = dot(t.e2, q_vec) * inv_det;
    if (tt < t_min || tt > t_max) { return false; }

    *t_out = tt;
    *bary_out = vec2<f32>(u, v);
    return true;
}

fn hit_quad_t(
    r: Ray,
    q: QuadPos,
    t_min: f32,
    t_max: f32,
    t_out: ptr<function, f32>,
    ab_out: ptr<function, vec2<f32>>,
) -> bool {
    let denom = dot(q.normal, r.direction);
    if (abs(denom) < 1e-8) { return false; }

    let t = (q.d - dot(q.normal, r.origin)) / denom;
    if (t < t_min || t > t_max) { return false; }

    let p = ray_at(r, t);
    let planar_hit_point_vector = p - q.Q;
    let alpha = dot(q.w, cross(planar_hit_point_vector, q.v));
    let beta = dot(q.w, cross(q.u, planar_hit_point_vector));

    if (alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0) { return false; }

    *t_out = t;
    *ab_out = vec2<f32>(alpha, beta);
    return true;
}

// Reciprocal of the ray direction, computed once per ray.
//
// Exact zeros are replaced by a tiny magnitude so the slab test cannot produce
// 0 * inf -> NaN on an axis-parallel ray. The sign of the substitute does not
// matter: t_near/t_far are a min/max pair, and both signs yield the correct
// "inside the slab / never enters" answer for a direction that does not move.
fn ray_inv_dir(direction: vec3<f32>) -> vec3<f32> {
    let tiny = vec3<f32>(1e-20);
    return 1.0 / select(direction, tiny, abs(direction) < tiny);
}

// Slab test that also reports the entry distance, which is what orders the
// two children of a node.
fn hit_aabb(
    origin: vec3<f32>,
    inv_dir: vec3<f32>,
    min_val: vec3<f32>,
    max_val: vec3<f32>,
    t_min_in: f32,
    t_max_in: f32,
    t_entry: ptr<function, f32>,
) -> bool {
    let t0 = (min_val - origin) * inv_dir;
    let t1 = (max_val - origin) * inv_dir;

    let t_near = min(t0, t1);
    let t_far = max(t0, t1);

    let enter = max(t_min_in, max(t_near.x, max(t_near.y, t_near.z)));
    let exit = min(t_max_in, min(t_far.x, min(t_far.y, t_far.z)));

    *t_entry = enter;
    return enter <= exit;
}

// Intersects the primitives of one inline leaf, tightening closest_so_far.
fn hit_leaf(
    r: Ray,
    leaf: u32,
    t_min: f32,
    closest_so_far: ptr<function, f32>,
    hit_ref: ptr<function, HitRef>,
) -> bool {
    let count = (leaf >> LEAF_COUNT_SHIFT) & LEAF_COUNT_MASK;
    let offset = leaf & LEAF_OFFSET_MASK;

    var hit_any = false;
    for (var i = 0u; i < count; i++) {
        let prim_ref = prim_refs[offset + i];
        let prim_type = prim_ref >> PRIM_TYPE_SHIFT;
        let prim_idx = prim_ref & PRIM_INDEX_MASK;

        var t_hit = 0.0;
        var bary = vec2<f32>(0.0);
        var hit = false;
        if (prim_type == 0u) {
            hit = hit_sphere_t(r, spheres[prim_idx], t_min, *closest_so_far, &t_hit);
        } else if (prim_type == 1u) {
            hit = hit_triangle_t(r, triangle_pos[prim_idx], t_min, *closest_so_far, &t_hit, &bary);
        } else if (prim_type == 2u) {
            hit = hit_quad_t(r, quad_pos[prim_idx], t_min, *closest_so_far, &t_hit, &bary);
        }

        if (hit) {
            hit_any = true;
            *closest_so_far = t_hit;
            (*hit_ref).t = t_hit;
            (*hit_ref).prim_type = prim_type;
            (*hit_ref).prim_idx = prim_idx;
            (*hit_ref).bary = bary;
        }
    }
    return hit_any;
}

// Sentinel for "this child slot has nothing to descend into".
const NO_NODE = 0xFFFFFFFFu;

fn world_hit(r: Ray, t_min: f32, t_max: f32, hit_ref: ptr<function, HitRef>) -> bool {
    if (arrayLength(&nodes) == 0u) { return false; }

    let inv_dir = ray_inv_dir(r.direction);

    var hit_anything = false;
    var closest_so_far = t_max;

    // Only farther children are ever pushed, so the tree depth bounds this and
    // 32 entries covers any balanced tree far beyond the addressable
    // primitive count.
    var stack: array<u32, 32>;
    var stack_ptr = 0u;
    var node_idx = 0u;

    loop {
        let node = nodes[node_idx];

        var t_left = 0.0;
        var t_right = 0.0;
        let left_hit = hit_aabb(r.origin, inv_dir, node.left_min, node.left_max, t_min, closest_so_far, &t_left);
        let right_hit = hit_aabb(r.origin, inv_dir, node.right_min, node.right_max, t_min, closest_so_far, &t_right);

        var left_next = NO_NODE;
        var right_next = NO_NODE;

        // Leaves are resolved in place; they never occupy a stack slot.
        if (left_hit) {
            if ((node.left_meta & LEAF_FLAG) != 0u) {
                if (hit_leaf(r, node.left_meta, t_min, &closest_so_far, hit_ref)) { hit_anything = true; }
            } else {
                left_next = node.left_meta;
            }
        }
        if (right_hit) {
            if ((node.right_meta & LEAF_FLAG) != 0u) {
                if (hit_leaf(r, node.right_meta, t_min, &closest_so_far, hit_ref)) { hit_anything = true; }
            } else {
                right_next = node.right_meta;
            }
        }

        if (left_next != NO_NODE && right_next != NO_NODE) {
            // Descend into the nearer child, defer the farther one.
            if (t_left <= t_right) {
                stack[stack_ptr] = right_next;
                stack_ptr++;
                node_idx = left_next;
            } else {
                stack[stack_ptr] = left_next;
                stack_ptr++;
                node_idx = right_next;
            }
        } else if (left_next != NO_NODE) {
            node_idx = left_next;
        } else if (right_next != NO_NODE) {
            node_idx = right_next;
        } else {
            if (stack_ptr == 0u) { break; }
            stack_ptr--;
            node_idx = stack[stack_ptr];
        }
    }

    return hit_anything;
}

// Expands the winning primitive into full shading data. Called once per ray,
// not once per candidate intersection.
fn resolve_hit(r: Ray, hit_ref: HitRef) -> HitRecord {
    var rec: HitRecord;
    rec.t = hit_ref.t;
    rec.p = ray_at(r, hit_ref.t);

    if (hit_ref.prim_type == 0u) {
        let s = spheres[hit_ref.prim_idx];
        let center = s.center_and_radius.xyz;
        let radius = s.center_and_radius.w;
        let outward_normal = (rec.p - center) / radius;

        rec.front_face = dot(r.direction, outward_normal) < 0.0;
        rec.normal = select(-outward_normal, outward_normal, rec.front_face);
        rec.material_index = s.material_index;

        let theta = acos(-outward_normal.y);
        let phi = atan2(-outward_normal.z, outward_normal.x) + 3.14159265359;
        rec.uv = vec2<f32>(phi / (2.0 * 3.14159265359), theta / 3.14159265359);

        rec.tangent = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), outward_normal));
        rec.bi_tangent = cross(outward_normal, rec.tangent);
    } else if (hit_ref.prim_type == 1u) {
        let attr = triangle_attr[hit_ref.prim_idx];
        let u = hit_ref.bary.x;
        let v = hit_ref.bary.y;
        let w = 1.0 - u - v;

        rec.front_face = dot(r.direction, attr.normal) < 0.0;
        rec.normal = select(-attr.normal, attr.normal, rec.front_face);
        rec.material_index = attr.material_index;
        rec.uv = w * attr.uv0 + u * attr.uv1 + v * attr.uv2;
        rec.tangent = attr.tangent;
        rec.bi_tangent = attr.bi_tangent;
    } else {
        let attr = quad_attr[hit_ref.prim_idx];
        let normal = quad_pos[hit_ref.prim_idx].normal;

        rec.front_face = dot(r.direction, normal) < 0.0;
        rec.normal = select(-normal, normal, rec.front_face);
        rec.material_index = attr.material_index;
        rec.uv = hit_ref.bary;
        rec.tangent = attr.tangent;
        rec.bi_tangent = attr.bi_tangent;
    }

    return rec;
}

// 8x8 tiles rather than 64 pixels of one scanline: neighbouring rays in a
// workgroup then stay coherent through the first bounce or two, which is where
// BVH traversal divergence actually costs.
// Traces one path for the given pixel and sample index.
fn trace_sample(pixel: vec2<u32>, sample_index: u32) -> vec3<f32> {
    let index = pixel.y * config.width + pixel.x;
    var rng_state = pcg_hash(index ^ (sample_index * 0x9E3779B9u));

    let u = (f32(pixel.x) + rand_float(&rng_state)) / f32(config.width - 1u);
    let v = 1.0 - (f32(pixel.y) + rand_float(&rng_state)) / f32(config.height - 1u);

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
        var hit_ref: HitRef;
        if (world_hit(r, 0.001, 10000.0, &hit_ref)) {
            let rec = resolve_hit(r, hit_ref);
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

        let max_attenuation = max(current_attenuation.x, max(current_attenuation.y, current_attenuation.z));
        if (max_attenuation < 0.0001) {
            break;
        }

        // Russian roulette: terminate dim paths early and scale the survivors
        // up to compensate, which keeps the estimator unbiased while cutting
        // the average path length.
        if (depth >= RR_MIN_DEPTH) {
            let survival = clamp(max_attenuation, RR_MIN_SURVIVAL, 1.0);
            if (rand_float(&rng_state) > survival) {
                break;
            }
            current_attenuation /= survival;
        }
    }

    // Firefly clamp, per sample.
    return min(accumulated_color, vec3<f32>(CLAMPING_THRESHOLD));
}

// 8x8 tiles rather than 64 pixels of one scanline: neighbouring rays in a
// workgroup then stay coherent through the first bounce or two, which is where
// BVH traversal divergence actually costs.
@compute @workgroup_size(8, 8)
fn compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= config.width || global_id.y >= config.height) {
        return;
    }
    let index = global_id.y * config.width + global_id.x;
    let pixel = global_id.xy;

    // Several samples per dispatch, summed in registers, so the 16-byte-per-pixel
    // accumulation buffer is read and written once per batch rather than once
    // per sample.
    let batch = max(config.samples_per_batch, 1u);
    var batch_sum = vec3<f32>(0.0);
    for (var s = 0u; s < batch; s++) {
        batch_sum += trace_sample(pixel, config.sample_count + s);
    }

    let completed = f32(config.sample_count);
    let total = completed + f32(batch);

    if (config.sample_count == 0u) {
        output_buffer[index] = vec4<f32>(batch_sum / f32(batch), 1.0);
    } else {
        let prev_color = output_buffer[index].xyz;
        output_buffer[index] = vec4<f32>((prev_color * completed + batch_sum) / total, 1.0);
    }
}
