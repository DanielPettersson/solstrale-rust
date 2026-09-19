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
// Not a material. Marks a primary ray that left the scene, so the denoiser can
// tell "background" from "a surface that happens to be dark".
const MAT_MISS = 5u;

// Depth recorded for a primary ray that hit nothing. Far enough that the
// denoiser's relative depth weight reads background as identical to background
// and as wildly different from any geometry, which preserves the silhouette
// without needing a validity branch in the filter's inner loop.
const GUIDE_FAR = 1e7;

// Ceiling on the indirect radiance a single sample may carry, which is what
// stops one improbable bright bounce from leaving a permanent speck.
//
// `min(X, t)` has an expectation strictly below `E[X]`, so a clamp always
// biases the image darker, and the bias is only worth paying where the
// variance it suppresses is real. Two things narrow it down to what is left
// here, both measured on `create_test_scene` against an unclamped reference:
//
// - It applies to depth >= 1 only (see `trace_sample`). The first path vertex
//   has no firefly failure mode, and clamping it was eating 26% of the
//   scene's energy -- most of that on diffuse surfaces, whose NEE estimate
//   divides by a `pdf_light` that goes small for a light subtending a large
//   solid angle, so a single legitimate direct sample lands far above any
//   sane threshold.
// - The threshold is 10, not the 3.5 it was before next-event estimation.
//   With NEE carrying the direct lighting, no indirect sample in that scene
//   reaches 10: per-sample standard deviation at 50 spp is 0.862 here and
//   0.862 with the clamp removed altogether, while 3.5 cost a further 9% of
//   the energy to buy that same 0.862 -> 0.672. What survives is a backstop
//   for scenes that do produce outliers -- a small intense light, an emissive
//   mesh -- priced so it does not tax the scenes that do not.
const CLAMPING_THRESHOLD = 10.0;

const PI = 3.14159265359;

// Offset used to push ray origins off the surface they start from, and to stop
// shadow rays short of the light they are aimed at.
const RAY_EPS = 0.001;
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
    // Minimum samples a pixel must have before adaptive sampling may skip it.
    min_samples_per_pixel: u32,
    // Relative standard-error threshold below which a pixel is converged.
    variance_threshold: f32,
    // Distinguishes successive accumulation restarts. Dragging the camera
    // restarts the accumulation every frame, and without this the seed in
    // trace_sample is a pure function of pixel and sample index, so every
    // frame replays an identical sample sequence -- which reads as a static
    // grain pinned to the screen rather than as noise.
    //
    // Scalar, not a vec3: a vec3 here would align to 16 and push the struct
    // to 64 bytes, which no longer matches the Rust mirror.
    restart_index: u32,
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

// Actual samples accumulated per pixel so far. Diverges from the uniform
// `sample_count` once adaptive sampling starts skipping converged pixels.
@group(0) @binding(14)
var<storage, read_write> sample_count_buffer: array<u32>;

// Albedo, shading normal and camera distance of the first surface along the
// view ray that is not a mirror or a lens, packed into 16 bytes. Written once
// per accumulation run by trace_guide and read only by the denoiser, which
// fetches it 125 times per pixel -- which is why it is packed rather than
// stored as two plain vec4<f32>. See pack_guide below; the denoise shaders
// carry a matching oct_decode that must stay in step with oct_encode here.
@group(0) @binding(15)
var<storage, read_write> gbuffer: array<vec4<u32>>;

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

// Scalar proxy used for the per-pixel variance estimate that drives adaptive
// sampling. Perceptual weighting doesn't matter here, only that it's a single
// number cheap to accumulate.
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
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

// One sample of the direct-lighting strategy: pick a light uniformly, sample a
// point on it, and report what a shadow ray would need to reach it.
struct LightSample {
    // Unit vector from the shading point toward the sampled point.
    direction: vec3<f32>,
    // Distance to the sampled point, so the shadow ray can stop short of it.
    distance: f32,
    emission: vec3<f32>,
    attenuation_factor: f32,
    valid: bool,
}

fn sample_light(origin: vec3<f32>, state: ptr<function, u32>) -> LightSample {
    var ls: LightSample;
    ls.direction = vec3<f32>(0.0, 1.0, 0.0);
    ls.distance = 0.0;
    ls.emission = vec3<f32>(0.0);
    ls.attenuation_factor = 0.0;
    ls.valid = false;

    if (config.light_count == 0u) { return ls; }

    let idx = min(u32(rand_float(state) * f32(config.light_count)), config.light_count - 1u);
    let light = lights[idx];

    var to_light = vec3<f32>(0.0);
    if (light.prim_type == 0u) {
        to_light = sphere_random_direction(spheres[light.prim_index], origin, state);
    } else if (light.prim_type == 1u) {
        to_light = triangle_random_direction(triangle_pos[light.prim_index], origin, state);
    } else {
        to_light = quad_random_direction(quad_pos[light.prim_index], origin, state);
    }

    let len_sq = dot(to_light, to_light);
    if (len_sq < 1e-12) { return ls; }
    let dir = to_light * inverseSqrt(len_sq);

    // Intersect the chosen light itself: sphere sampling yields a direction
    // rather than a point, and we need the distance either way so the shadow
    // ray can stop just short of the light instead of hitting it.
    let probe = Ray(origin, dir);
    var t_hit = 0.0;
    var bary = vec2<f32>(0.0);
    var normal = vec3<f32>(0.0);
    var mat_idx = 0u;

    if (light.prim_type == 0u) {
        if (!hit_sphere_t(probe, spheres[light.prim_index], RAY_EPS, 1e20, &t_hit)) { return ls; }
        let sph = spheres[light.prim_index];
        normal = (ray_at(probe, t_hit) - sph.center_and_radius.xyz) / sph.center_and_radius.w;
        mat_idx = sph.material_index;
    } else if (light.prim_type == 1u) {
        if (!hit_triangle_t(probe, triangle_pos[light.prim_index], RAY_EPS, 1e20, &t_hit, &bary)) { return ls; }
        let attr = triangle_attr[light.prim_index];
        normal = attr.normal;
        mat_idx = attr.material_index;
    } else {
        if (!hit_quad_t(probe, quad_pos[light.prim_index], RAY_EPS, 1e20, &t_hit, &bary)) { return ls; }
        normal = quad_pos[light.prim_index].normal;
        mat_idx = quad_attr[light.prim_index].material_index;
    }

    // Lights emit from their front face only, matching what a BSDF path sees
    // when it lands on one. Sampling the back is a valid direction with a real
    // PDF, it just carries no radiance -- so the MIS weights stay consistent.
    if (dot(dir, normal) >= 0.0) { return ls; }

    let material = materials[mat_idx];
    let emission = select(vec3<f32>(0.0), material.emission, material.mat_type == MAT_DIFFUSE_LIGHT);

    ls.direction = dir;
    ls.distance = t_hit;
    ls.emission = emission;
    ls.attenuation_factor = material.attenuation_factor;
    ls.valid = true;
    return ls;
}

fn cosine_pdf_value(normal: vec3<f32>, direction: vec3<f32>) -> f32 {
    let cos_theta = dot(normalize(direction), normal);
    return max(0.0, cos_theta / PI);
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

// What the pixel is looking at, before any light transport: the first surface
// along the view ray that is not a mirror or a lens. Filled by trace_guide and
// used only to guide the denoiser's edge-stopping functions -- it is a guide,
// not a signal, which is what makes the lossy packing below acceptable.
struct GuideSample {
    // The surface's own albedo, tinted by whatever specular surfaces the guide
    // ray passed through to reach it, so it describes the colour this pixel
    // should end up rather than the colour of a surface it only sees in a
    // mirror.
    albedo: vec3<f32>,
    normal: vec3<f32>,
    // Path length from the camera in world units, summed over the whole guide
    // chain. The guide ray is normalised from the start, unlike the primary
    // ray in trace_sample (see issue #57), so every segment is in the same units.
    depth: f32,
    mat_type: u32,
    // How many specular bounces the guide ray took to get here. Lets the filter
    // tell a wall seen in a mirror from the same wall seen directly, which the
    // other three channels can agree on by coincidence.
    specular_depth: u32,
}

// Octahedral normal encoding. Two floats instead of three, with ~0.01 degrees of
// error at 16-bit -- far below anything an edge-stop with exponent 128 resolves.
fn oct_encode(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));
    if (n.z <= 0.0) {
        let s = vec2<f32>(select(-1.0, 1.0, p.x >= 0.0), select(-1.0, 1.0, p.y >= 0.0));
        return (1.0 - abs(vec2<f32>(p.y, p.x))) * s;
    }
    return p;
}

// 16 bytes per pixel. The depth goes through bitcast rather than into an f32
// slot alongside packed bits, because a packed bit pattern can land on a NaN
// encoding and some drivers canonicalise NaN payloads across a store/load.
// pack4x8unorm clamps, so an albedo above 1 degrades the guide but never the image.
//
// The material type needs three bits of the last slot, so the specular depth
// rides in the byte above it. denoise_atrous.wgsl unpacks both.
fn pack_guide(g: GuideSample) -> vec4<u32> {
    return vec4<u32>(
        pack4x8unorm(vec4<f32>(g.albedo, 0.0)),
        pack2x16float(oct_encode(g.normal)),
        bitcast<u32>(g.depth),
        (g.mat_type & 0xFFu) | (g.specular_depth << 8u),
    );
}

// A hit surface with its material fully resolved: blend chosen, albedo and
// shading normal sampled from textures.
struct Surface {
    albedo: vec3<f32>,
    normal: vec3<f32>,
    emission: vec3<f32>,
    attenuation_factor: f32,
    fuzz: f32,
    refraction_index: f32,
    mat_type: u32,
}

fn resolve_surface(rec: HitRecord, state: ptr<function, u32>) -> Surface {
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

    return surface_at(mat_idx, rec);
}

// The blend walk above, resolved deterministically: the branch the coin flip
// would take more often than not. Used only by trace_guide, where a stochastic
// choice would make neighbouring pixels disagree about what surface they are
// looking at, which is the one thing an edge stop cannot survive.
//
// Deliberately not shared with resolve_surface: that walk draws one rand_float
// per nesting level, and the sample stream has to stay bit-identical.
fn resolve_material_index_dominant(start: u32) -> u32 {
    var mat_idx = start;
    for (var i = 0u; i < 10u; i++) {
        let material = materials[mat_idx];
        if (material.mat_type == MAT_BLEND) {
            if (material.blend_factor < 0.5) {
                mat_idx = material.blend_indices.x;
            } else {
                mat_idx = material.blend_indices.y;
            }
        } else {
            break;
        }
    }
    return mat_idx;
}

// Everything after the blend is chosen: albedo and shading normal sampled from
// their textures. Shared by the stochastic and deterministic walks above.
fn surface_at(mat_idx: u32, rec: HitRecord) -> Surface {
    let material = materials[mat_idx];

    var surface: Surface;
    surface.mat_type = material.mat_type;
    surface.emission = material.emission;
    surface.attenuation_factor = material.attenuation_factor;
    surface.fuzz = material.fuzz;
    surface.refraction_index = material.refraction_index;

    surface.albedo = material.albedo;
    if (material.texture_index >= 0) {
        let uv = vec2<f32>(fract(abs(rec.uv.x)), 1.0 - fract(abs(rec.uv.y)));
        let uv_atlas = material.albedo_offset + uv * material.albedo_scale;
        surface.albedo = textureSampleLevel(texture_array, texture_sampler, uv_atlas, 0.0).rgb;
    }

    surface.normal = rec.normal;
    if (material.normal_texture_index >= 0) {
        let uv = vec2<f32>(fract(abs(rec.uv.x)), 1.0 - fract(abs(rec.uv.y)));
        let uv_atlas = material.normal_offset + uv * material.normal_scale;
        let map_color = textureSampleLevel(texture_array, texture_sampler, uv_atlas, 0.0).rgb;
        let map_n = map_color * 2.0 - 1.0;
        surface.normal = normalize(map_n.x * rec.tangent + map_n.y * rec.bi_tangent + map_n.z * rec.normal);
    }

    return surface;
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

// Does any primitive of this leaf block the segment?
fn leaf_occluded(r: Ray, leaf: u32, t_max: f32) -> bool {
    let count = (leaf >> LEAF_COUNT_SHIFT) & LEAF_COUNT_MASK;
    let offset = leaf & LEAF_OFFSET_MASK;

    for (var i = 0u; i < count; i++) {
        let prim_ref = prim_refs[offset + i];
        let prim_type = prim_ref >> PRIM_TYPE_SHIFT;
        let prim_idx = prim_ref & PRIM_INDEX_MASK;

        var t_hit = 0.0;
        var bary = vec2<f32>(0.0);
        if (prim_type == 0u) {
            if (hit_sphere_t(r, spheres[prim_idx], RAY_EPS, t_max, &t_hit)) { return true; }
        } else if (prim_type == 1u) {
            if (hit_triangle_t(r, triangle_pos[prim_idx], RAY_EPS, t_max, &t_hit, &bary)) { return true; }
        } else {
            if (hit_quad_t(r, quad_pos[prim_idx], RAY_EPS, t_max, &t_hit, &bary)) { return true; }
        }
    }
    return false;
}

// Any-hit traversal for shadow rays.
//
// Cheaper than world_hit in three ways: it returns on the first blocker instead
// of tracking the closest, it never resolves shading attributes, and it needs
// no front-to-back ordering because any hit is as good as any other. With NEE
// roughly half of all rays are shadow rays, so this pays for itself.
fn occluded(origin: vec3<f32>, direction: vec3<f32>, t_max: f32) -> bool {
    if (arrayLength(&nodes) == 0u || t_max <= RAY_EPS) { return false; }

    let inv_dir = ray_inv_dir(direction);
    let r = Ray(origin, direction);

    var stack: array<u32, 32>;
    var stack_ptr = 0u;
    var node_idx = 0u;

    loop {
        let node = nodes[node_idx];

        var t_left = 0.0;
        var t_right = 0.0;
        let left_hit = hit_aabb(origin, inv_dir, node.left_min, node.left_max, RAY_EPS, t_max, &t_left);
        let right_hit = hit_aabb(origin, inv_dir, node.right_min, node.right_max, RAY_EPS, t_max, &t_right);

        var left_next = NO_NODE;
        var right_next = NO_NODE;

        if (left_hit) {
            if ((node.left_meta & LEAF_FLAG) != 0u) {
                if (leaf_occluded(r, node.left_meta, t_max)) { return true; }
            } else {
                left_next = node.left_meta;
            }
        }
        if (right_hit) {
            if ((node.right_meta & LEAF_FLAG) != 0u) {
                if (leaf_occluded(r, node.right_meta, t_max)) { return true; }
            } else {
                right_next = node.right_meta;
            }
        }

        if (left_next != NO_NODE) {
            if (right_next != NO_NODE) {
                stack[stack_ptr] = right_next;
                stack_ptr++;
            }
            node_idx = left_next;
        } else if (right_next != NO_NODE) {
            node_idx = right_next;
        } else {
            if (stack_ptr == 0u) { break; }
            stack_ptr--;
            node_idx = stack[stack_ptr];
        }
    }

    return false;
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

// Routes one radiance contribution to the direct or the indirect accumulator.
//
// Depth 0 is what the camera can see without an intervening bounce: the
// visible surface's own emission, the shadow ray cast from it, and the
// background behind it. None of those is a firefly, so none of them is
// clamped. Everything deeper goes through CLAMPING_THRESHOLD.
fn add_contribution(
    direct: ptr<function, vec3<f32>>,
    indirect: ptr<function, vec3<f32>>,
    depth: u32,
    contribution: vec3<f32>,
) {
    if (depth == 0u) {
        *direct += contribution;
    } else {
        *indirect += contribution;
    }
}

// Ceiling on specular bounces the guide ray will follow. A guide chain longer
// than this is describing a hall of mirrors the eye cannot follow either, and
// every extra bounce is another ray per pixel.
const GUIDE_MAX_SPECULAR = 6u;

// Traces the denoiser's guide ray: what does this pixel actually look at?
//
// Not the primary hit. On a mirror or a glass surface the primary hit describes
// the surface rather than the image in or through it, so every tap across the
// mirror looks like the same surface to the edge stop and the reflection is
// smeared sideways along it. Following the specular chain to the first surface
// that scatters is what gives the filter something to hold on to.
//
// Deliberately separate from trace_sample rather than gathered from one of its
// samples. A sampled chain is stochastic -- the dielectric Fresnel coin flip,
// metal fuzz -- so two neighbouring pixels on a glass sphere would record
// unrelated guides, the edge stop would reject nearly every tap, and the filter
// would stop working there instead of over-blurring. Everything below is
// deterministic and shared with its neighbours: the pixel centre rather than a
// jittered position, no lens offset, no fuzz, and the dominant Fresnel branch
// rather than a coin flip.
//
// Costs one ray per pixel, and only on a restart dispatch, against
// samples_per_pixel rays for the render itself.
fn trace_guide(pixel: vec2<u32>) -> GuideSample {
    var out: GuideSample;
    out.specular_depth = 0u;

    let u = (f32(pixel.x) + 0.5) / f32(config.width);
    let v = 1.0 - (f32(pixel.y) + 0.5) / f32(config.height);
    // Normalised, unlike the primary ray in trace_sample, so rec.t is in world
    // units at every segment and the lengths below can simply be summed.
    var r = Ray(
        camera.origin,
        normalize(camera.lower_left_corner + u * camera.horizontal + v * camera.vertical - camera.origin),
    );

    // What the specular chain has done to the colour on the way, so the guide
    // albedo describes the pixel rather than a surface it only sees reflected.
    var tint = vec3<f32>(1.0);
    var distance = 0.0;

    for (var bounce = 0u; bounce <= GUIDE_MAX_SPECULAR; bounce++) {
        var hit_ref: HitRef;
        if (!world_hit(r, RAY_EPS, 10000.0, &hit_ref)) {
            out.albedo = config.background_color * tint;
            // Face the camera, so background filters against background at
            // full normal weight.
            out.normal = -r.direction;
            out.depth = GUIDE_FAR;
            out.mat_type = MAT_MISS;
            return out;
        }

        let rec = resolve_hit(r, hit_ref);
        let surface = surface_at(resolve_material_index_dominant(rec.material_index), rec);
        distance += rec.t;

        let specular = surface.mat_type == MAT_METAL || surface.mat_type == MAT_DIELECTRIC;
        if (!specular || bounce == GUIDE_MAX_SPECULAR) {
            // The first surface that scatters -- or, once the budget is spent,
            // whatever specular surface the chain stalled on, which is the old
            // primary-hit guide generalised.
            out.albedo = surface.albedo * tint;
            // The normal-mapped shading normal, so bump detail reaches the
            // edge stop rather than just the geometric silhouette.
            out.normal = surface.normal;
            out.depth = distance;
            out.mat_type = surface.mat_type;
            return out;
        }

        let unit_direction = normalize(r.direction);
        var direction: vec3<f32>;
        if (surface.mat_type == MAT_METAL) {
            // Fuzz ignored: it is what makes a sampled chain diverge between
            // neighbours, and the mirror direction is the mean it scatters
            // around anyway.
            direction = reflect(unit_direction, surface.normal);
            tint *= surface.albedo;
        } else {
            var refraction_ratio = surface.refraction_index;
            if (rec.front_face) {
                refraction_ratio = 1.0 / surface.refraction_index;
            }
            let cos_theta = min(dot(-unit_direction, surface.normal), 1.0);
            let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
            // The branch the coin flip in trace_sample would take more often
            // than not: refraction everywhere but total internal reflection and
            // the grazing rim.
            if (refraction_ratio * sin_theta > 1.0
                || reflectance(cos_theta, refraction_ratio) > 0.5) {
                direction = reflect(unit_direction, surface.normal);
            } else {
                direction = refract(unit_direction, surface.normal, refraction_ratio);
            }
        }

        out.specular_depth += 1u;
        r = Ray(rec.p, normalize(direction));
    }

    // Unreachable: the loop returns at bounce == GUIDE_MAX_SPECULAR at the
    // latest. WGSL needs the function to end in a return all the same.
    return out;
}

// Traces one path for the given pixel and sample index.
//
// Next-event estimation with multiple importance sampling: at every diffuse
// vertex the direct lighting is estimated with an explicit shadow ray, and the
// BSDF-sampled continuation is weighted so that a path which happens to land on
// a light is not counted twice. Both strategies use the balance heuristic, for
// which `w / pdf` collapses to `1 / (pdf_light + pdf_bsdf)`.
//
// Specular bounces (metal, dielectric) have no light-sampling counterpart, so
// they skip NEE and the emitter they reach is taken at full weight.
fn trace_sample(pixel: vec2<u32>, sample_index: u32) -> vec3<f32> {
    let index = pixel.y * config.width + pixel.x;
    var rng_state = pcg_hash(
        index ^ (sample_index * 0x9E3779B9u) ^ (config.restart_index * 0x85EBCA6Bu)
    );

    let u = (f32(pixel.x) + rand_float(&rng_state)) / f32(config.width);
    let v = 1.0 - (f32(pixel.y) + rand_float(&rng_state)) / f32(config.height);

    var offset = vec3<f32>(0.0);
    if (camera.lens_radius > 0.0) {
        let rd = random_in_unit_disk(&rng_state) * camera.lens_radius;
        offset = camera.u * rd.x + camera.v * rd.y;
    }

    let ray_direction = camera.lower_left_corner + u * camera.horizontal + v * camera.vertical - camera.origin - offset;
    var r = Ray(camera.origin + offset, ray_direction);

    var direct = vec3<f32>(0.0);
    var indirect = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    var path_length = 0.0;

    // State describing how the current ray was generated, needed to weight an
    // emitter it may land on. The camera ray counts as specular: a directly
    // visible light is seen at full brightness.
    var prev_specular = true;
    var prev_bsdf_pdf = 0.0;

    for (var depth = 0u; depth < config.max_depth; depth++) {
        var hit_ref: HitRef;
        if (!world_hit(r, RAY_EPS, 10000.0, &hit_ref)) {
            add_contribution(&direct, &indirect, depth, config.background_color * throughput);
            break;
        }

        let rec = resolve_hit(r, hit_ref);
        let surface = resolve_surface(rec, &rng_state);
        path_length += rec.t;

        if (surface.mat_type == MAT_DIFFUSE_LIGHT) {
            if (rec.front_face) {
                var emitted = surface.emission;
                if (surface.attenuation_factor > 0.0) {
                    emitted *= 1.0 / (1.0 + surface.attenuation_factor * path_length);
                }

                // Weight against the direct-lighting strategy that could also
                // have produced this direction, unless the previous bounce was
                // specular and no such strategy exists.
                var weight = 1.0;
                if (!prev_specular) {
                    let pdf_light = light_pdf_value(r.origin, r.direction);
                    weight = prev_bsdf_pdf / (prev_bsdf_pdf + pdf_light);
                }
                add_contribution(&direct, &indirect, depth, throughput * emitted * weight);
            }
            break;
        }

        if (surface.mat_type == MAT_LAMBERTIAN) {
            // --- Direct lighting (next-event estimation) ---
            let ls = sample_light(rec.p, &rng_state);
            let cos_light = dot(surface.normal, ls.direction);
            if (ls.valid && cos_light > 0.0) {
                let pdf_light = light_pdf_value(rec.p, ls.direction);
                if (pdf_light > 0.0) {
                    let pdf_bsdf = cos_light / PI;
                    // Shadow ray last: everything above is cheaper to reject on.
                    if (!occluded(rec.p, ls.direction, ls.distance - RAY_EPS)) {
                        var emitted = ls.emission;
                        if (ls.attenuation_factor > 0.0) {
                            emitted *= 1.0 / (1.0 + ls.attenuation_factor * (path_length + ls.distance));
                        }
                        let brdf = surface.albedo / PI;
                        add_contribution(
                            &direct,
                            &indirect,
                            depth,
                            throughput * brdf * cos_light * emitted / (pdf_light + pdf_bsdf),
                        );
                    }
                }
            }

            // --- BSDF continuation ---
            let uvw = onb_from_w(surface.normal);
            let direction = onb_local(uvw, random_cosine_direction(&rng_state));
            let cos_theta = dot(surface.normal, direction);
            if (cos_theta <= 0.0) { break; }

            // Cosine sampling cancels the BRDF and the cosine exactly, leaving
            // the albedo: (albedo/PI) * cos / (cos/PI).
            throughput *= surface.albedo;
            prev_bsdf_pdf = cos_theta / PI;
            prev_specular = false;
            r = Ray(rec.p, normalize(direction));
        } else if (surface.mat_type == MAT_METAL) {
            let reflected = reflect(normalize(r.direction), surface.normal);
            let direction = reflected + surface.fuzz * random_in_unit_sphere(&rng_state);
            if (dot(direction, surface.normal) <= 0.0) { break; }

            throughput *= surface.albedo;
            prev_specular = true;
            r = Ray(rec.p, normalize(direction));
        } else if (surface.mat_type == MAT_DIELECTRIC) {
            var refraction_ratio = surface.refraction_index;
            if (rec.front_face) {
                refraction_ratio = 1.0 / surface.refraction_index;
            }

            let unit_direction = normalize(r.direction);
            let cos_theta = min(dot(-unit_direction, surface.normal), 1.0);
            let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

            var direction: vec3<f32>;
            if (refraction_ratio * sin_theta > 1.0
                || reflectance(cos_theta, refraction_ratio) > rand_float(&rng_state)) {
                direction = reflect(unit_direction, surface.normal);
            } else {
                direction = refract(unit_direction, surface.normal, refraction_ratio);
            }

            prev_specular = true;
            r = Ray(rec.p, normalize(direction));
        } else {
            break;
        }

        let max_throughput = max(throughput.x, max(throughput.y, throughput.z));
        if (max_throughput < 0.0001) {
            break;
        }

        // Russian roulette: terminate dim paths early and scale the survivors
        // up to compensate, which keeps the estimator unbiased while cutting
        // the average path length.
        if (depth >= RR_MIN_DEPTH) {
            let survival = clamp(max_throughput, RR_MIN_SURVIVAL, 1.0);
            if (rand_float(&rng_state) > survival) {
                break;
            }
            throughput /= survival;
        }
    }

    // Firefly clamp, per sample, on the indirect term only.
    return direct + min(indirect, vec3<f32>(CLAMPING_THRESHOLD));
}

// Floor on the luminance used as the denominator of the relative variance
// check, so a near-black pixel's tiny absolute noise doesn't look enormous
// relative to it and keep the pixel sampling forever.
const ADAPTIVE_LUMINANCE_FLOOR = 1e-4;

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

    // A restart (accumulation reset on camera/depth change) discards whatever
    // the buffers held for a previous, unrelated accumulation.
    let restart = config.sample_count == 0u;

    // One guide ray per accumulation run, before the adaptive early-out below
    // can return, so every pixel gets exactly one and the guide can never go
    // stale: a camera change restarts the accumulation and rewrites the guide
    // in the same dispatch that resets the accumulator.
    if (restart) {
        gbuffer[index] = pack_guide(trace_guide(pixel));
    }

    let n0 = select(sample_count_buffer[index], 0u, restart);
    let prev = select(output_buffer[index], vec4<f32>(0.0), restart);
    let prev_mean = prev.xyz;
    // Sum of squared deviations from the running luminance mean, not a raw sum
    // of squares. See the merge at the bottom for why.
    let prev_m2 = prev.w;

    // Skip pixels that have already converged: no trace_sample, no BVH
    // traversal, no shadow rays for this dispatch. `min_samples_per_pixel` is
    // floored at 2 so the variance below is never evaluated against a sample
    // count that cannot support it.
    let min_samples = max(config.min_samples_per_pixel, 2u);
    if (n0 >= min_samples) {
        let mean_luminance = luminance(prev_mean);
        let variance = prev_m2 / f32(n0 - 1u);

        // The variance is itself estimated from n0 samples, and that estimate
        // has a relative standard deviation of sqrt(2/(n-1)) -- 36% at n = 16.
        // Testing the raw point estimate therefore lets a pixel that merely
        // drew an unlucky run of similar samples pass as converged, and the
        // skip is effectively permanent: a pixel that stops sampling can never
        // revise the numbers that silenced it, so the noise it happened to
        // hold is frozen into the image.
        //
        // Simulated on a pixel whose true relative standard error sat 11%
        // above the threshold, 42% of runs froze it early at n = 16 on the
        // point estimate; testing one standard deviation above the estimate
        // instead cut that to 13%. The band this matters in is narrow -- by
        // n = 64 the estimate is sharp enough that the bound changes almost
        // nothing -- and it is paid for in extra samples on pixels that had in
        // fact converged, which is the right way round for an artifact that
        // never averages out.
        //
        // This is why min_samples_per_pixel wants to be a few dozen, not a
        // handful: it is what sets the precision of the estimate being tested.
        let estimator_uncertainty = sqrt(2.0 / f32(n0 - 1u));
        let variance_bound = variance * (1.0 + estimator_uncertainty);

        let standard_error = sqrt(variance_bound / f32(n0));
        if (standard_error <= config.variance_threshold * max(mean_luminance, ADAPTIVE_LUMINANCE_FLOOR)) {
            return;
        }
    }

    // Several samples per dispatch, accumulated in registers, so the
    // accumulation buffers are read and written once per batch rather than
    // once per sample. Welford within the batch, for the same reason it is
    // used across batches below.
    let batch = max(config.samples_per_batch, 1u);
    var batch_sum = vec3<f32>(0.0);
    var batch_mean = 0.0;
    var batch_m2 = 0.0;
    for (var s = 0u; s < batch; s++) {
        let sample = trace_sample(pixel, config.sample_count + s);
        batch_sum += sample;

        let l = luminance(sample);
        let delta = l - batch_mean;
        batch_mean += delta / f32(s + 1u);
        batch_m2 += delta * (l - batch_mean);
    }

    let total = n0 + batch;
    let new_mean = (prev_mean * f32(n0) + batch_sum) / f32(total);

    // Chan's parallel Welford merge of the batch into the accumulated state.
    //
    // The previous form stored a raw sum of squares and recovered the variance
    // as E[L^2] - E[L]^2, a difference of two terms that each grow with the
    // sample count while their difference does not. In practice that was less
    // dire than it looks: measured against f64 on clamped lognormal samples,
    // the f32 error was 0.4% at n = 256 and 0.01% by n = 16384. What the old
    // form did get wrong systematically was the divisor -- it used the
    // population form M2/n where the estimate wants M2/(n-1), understating the
    // variance by exactly 1/n and so freezing pixels slightly early.
    //
    // The difference can still go negative for a genuinely low-variance pixel,
    // and the clamp to zero then hands the test above a standard error of
    // exactly zero, which always passes and freezes that pixel permanently.
    // M2 never subtracts two large numbers, so it has no such failure mode,
    // and the merge costs a handful of scalar ops per batch.
    //
    // luminance() is linear, so luminance(prev_mean) is exactly the running
    // mean of the per-sample luminances and needs no separate accumulator. At
    // n0 == 0 the correction term vanishes and this reduces to batch_m2.
    let delta = batch_mean - luminance(prev_mean);
    let new_m2 = prev_m2 + batch_m2
        + delta * delta * f32(n0) * f32(batch) / f32(total);

    output_buffer[index] = vec4<f32>(new_mean, new_m2);
    sample_count_buffer[index] = total;
}
