struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct Sphere {
    center_and_radius: vec4<f32>,
    material_index: u32,
}

// Geometry for the traversal inner loop. Edges rather than absolute vertices,
// which is what Moeller-Trumbore wants.
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
    n0_oct: u32,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    uv2: vec2<f32>,
    n1_oct: u32,
    n2_oct: u32,
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

const PRIM_TYPE_SPHERE = 0u;
const PRIM_TYPE_TRIANGLE = 1u;
const PRIM_TYPE_QUAD = 2u;

// Traversal defers only the farther child, so a tree of this depth pushes at
// most one entry fewer than there are slots here. `Bvh::new` asserts its depth
// against the matching MAX_TRAVERSAL_DEPTH: an overflow is not an error the GPU
// can report, the deferred subtree is simply lost and geometry vanishes.
//
// ACO keeps the stack in registers, so this is also the largest single lever on
// occupancy: 44 of 128 VGPRs at depth 32, giving 8 waves/SIMD against 12 at
// depth 8. Shrinking the stack is worth real occupancy; moving it to scratch or
// LDS would not be, since it is not there now.
const MAX_TRAVERSAL_DEPTH = 32u;

// One stack, shared by world_hit and occluded: the two are never live at once.
var<private> traversal_stack: array<u32, MAX_TRAVERSAL_DEPTH>;

struct Material {
    albedo: vec3<f32>,
    attenuation_factor: f32,
    emission: vec3<f32>,
    blend_factor: f32,
    fuzz: f32,
    refraction_index: f32,
    mat_type: u32,
    // Start of this material's pair of energy grids in `dielectric_energy`.
    energy_offset: u32,
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
// and as unlike any geometry, so the silhouette survives without a validity
// branch in the filter's inner loop.
const GUIDE_FAR = 1e7;

// Ceiling on the indirect radiance a single sample may carry, which is what
// stops one improbable bright bounce from leaving a permanent speck.
//
// `min(X, t)` biases the image darker, so the clamp is confined to where the
// variance it suppresses is real. Measured on `create_test_scene`: it applies
// to depth >= 1 only, since clamping the first vertex ate 26% of the scene's
// energy, and the threshold is 10 rather than 3.5, which cost a further 9% of
// the energy without lowering the per-sample standard deviation at all once NEE
// carried the direct lighting. What is left is a backstop for scenes that do
// produce outliers.
//
// An override so a test can render the same scene with the bias lifted; see
// `Renderer::bsdf_only_reference`.
override clamping_threshold: f32 = 10.0;

const PI = 3.14159265359;

// Offset used to push ray origins off the surface they start from, and to stop
// shadow rays short of the light they are aimed at.
const RAY_EPS = 0.001;
const TWO_PI = 6.28318530718;

// Dimension-pair budget for the sampler, which is indexed in pairs because its
// Sobol backing is 2D. Pixel jitter, then the lens, then three pairs per
// bounce: 2 + 3 * max_depth, which is 32 pairs at the default depth of 10.
const PAIR_PIXEL_JITTER = 0u;
const PAIR_LENS = 1u;
const PAIR_BOUNCE_BASE = 2u;
const PAIRS_PER_BOUNCE = 3u;
// Offsets within a bounce's three pairs: the point sampled on the light, the
// BSDF direction, and a shared pair whose .x is the bounce's one scalar draw
// -- light selection, or the Fresnel coin on smooth glass, which are mutually
// exclusive -- and whose .y is Russian roulette.
const PAIR_LIGHT_POINT = 0u;
const PAIR_BSDF = 1u;
const PAIR_SCALARS = 2u;

// The rough dielectric's reflect-or-refract coin, one pair per bounce, indexed
// as `PAIR_GLASS_COIN_BASE + bounce_pair`. Outside the budget above because a
// rough dielectric takes a shadow ray, so the shared scalar's `.x` is already
// the light pick. A pair index only seeds a hash, so a disjoint range overflows
// nothing.
const PAIR_GLASS_COIN_BASE = 4096u;

// Keeps sampler_extra's stream clear of the budgeted pairs'.
const EXTRA_TAG_SALT = 0x51633e2du;

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

// Only what a dispatch varies. The image size and the light count are fixed
// for the life of a pipeline, so they are overrides below rather than fields
// here.
struct RenderConfig {
    // Samples already accumulated before this dispatch.
    sample_count: u32,
    max_depth: u32,
    samples_per_batch: u32,
    // Minimum samples a pixel must have before adaptive sampling may skip it.
    min_samples_per_pixel: u32,
    background_color: vec3<f32>,
    // Relative standard-error threshold below which a pixel is converged.
    variance_threshold: f32,
    // Distinguishes successive accumulation restarts, and carries
    // RenderConfig::seed as its initial value. Without it a dragged camera
    // replays an identical sample sequence every frame, which reads as grain
    // pinned to the screen rather than as noise.
    restart_index: u32,
}

// An emitter, plus its share of the scene's emitted power. `sample_light` and
// `light_hit_pdf` both read the pick probability from `select_pdf`, so the two
// cannot disagree. See `LightRef` and `build_alias_table` in the flattener.
struct LightRef {
    // (prim_type << PRIM_TYPE_SHIFT) | prim_index, the packing prim_refs uses.
    prim: u32,
    // Probability that sample_light picks this light.
    select_pdf: f32,
    // Vose's alias table over select_pdf: the chance of keeping this slot,
    // and the slot to jump to otherwise.
    alias_prob: f32,
    alias_index: u32,
}

// What traversal tracks: enough to identify the winning primitive and rebuild
// its shading data afterwards. Keeping it small keeps the ~25-float HitRecord
// copy out of the inner loop.
struct HitRef {
    t: f32,
    prim_type: u32,
    prim_idx: u32,
    bary: vec2<f32>,
}

struct HitRecord {
    t: f32,
    p: vec3<f32>,
    // Interpolated across a smooth triangle; what the BSDF and the normal map
    // are evaluated against.
    normal: vec3<f32>,
    // The facet. Near a silhouette it can disagree with `normal` by most of a
    // facet, so facing and the scattering gates key off this one.
    geometric_normal: vec3<f32>,
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
// per accumulation run by trace_guide, and only when `writes_guide` is set.
// Packed because the denoiser fetches it 125 times per pixel. Three copies of
// the oct encode/decode pair must stay in step: this file's, denoise_atrous.wgsl's
// and `pack_oct` in scene_flattener.rs.
@group(0) @binding(15)
var<storage, read_write> gbuffer: array<vec4<u32>>;

// Single-scatter directional albedo of the rough dielectric, one 16x16 grid per
// side per index of refraction, concatenated. Built by dielectric_energy.rs;
// `Material::energy_offset` says where a material's pair of grids starts. Empty
// unless `has_rough_dielectrics`.
@group(0) @binding(16)
var<storage, read> dielectric_energy: array<f32>;

// ---------------------------------------------------------------------------
// Pipeline specialisation
//
// Facts that hold for the life of a pipeline, as override constants rather than
// uniforms: naga substitutes them before emitting SPIR-V, so a `false` deletes
// the branch it guards. Every flag below removes a branch the scene could never
// have taken, so none of them changes a sample -- see `Specialisation` in
// renderer/mod.rs.
//
// `max_depth` is deliberately not among them: a constant trip count only buys
// unrolling, and unrolling the whole material switch plus two BVH traversals
// ten times is an instruction cache disaster.
// ---------------------------------------------------------------------------

override width: u32 = 1u;
override height: u32 = 1u;

// Emitters in the scene, so the alias table's scale is a constant multiply and
// find_light's binary search has a constant bound.
override light_count: u32 = 0u;

// Which primitive types the scene contains. Each `false` strips one arm from
// hit_leaf, leaf_occluded, resolve_hit, sample_light and light_prim_pdf.
// Triangles are the fallback arm, so they need no flag.
override has_spheres: bool = true;
override has_quads: bool = true;

// Whether `prim_refs` is the identity map, which it is exactly when every
// primitive is a triangle. Worth the most of anything here: the innermost
// traversal loop then reads the triangle's address straight out of the leaf
// slot, removing one of the two dependent loads per primitive test.
override identity_prim_refs: bool = false;

// Which material kinds the scene contains, counting the ones inside blends.
// `has_blends` fires on the widest range of scenes: without it every bounce
// pays for resolve_surface's ten-iteration walk.
override has_blends: bool = true;
override has_metal: bool = true;
override has_dielectrics: bool = true;

// Whether any dielectric in the scene has a roughness above zero. Separate from
// `has_dielectrics` because the microfacet arm is much the larger half of the
// material code: on `test_scene` leaving it in costs 128 -> 168 VGPRs and drops
// occupancy from 8 subgroups per SIMD to 6.
override has_rough_dielectrics: bool = true;

// Whether any material samples the atlas, for albedo and for normals. Strips
// the textureSampleLevel branches from surface_at.
override has_textures: bool = true;
override has_normal_maps: bool = true;

// Next-event estimation. Here so the estimator can be measured against plain
// BSDF sampling without editing this file.
override nee_enabled: bool = true;

// Selects the sampler backing: 1 draws from an Owen-scrambled Sobol sequence,
// 0 from white noise. An override rather than a uniform, which would cost a
// branch on every draw.
override low_discrepancy: f32 = 1.0;

// Whether the G-buffer is filled at all. Only a denoising post-processor reads
// it, and filling it costs an extra ray per pixel per accumulation run: under
// 0.2% of a long render, but 9% of a frame during a camera drag, where every
// frame is a restart. Off strips trace_guide and its helpers from the module.
// See `interactive_restart_frame_cost`.
override writes_guide: bool = true;

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Combining seed terms with a bare XOR collides: `a ^ (b * A) ^ (c * B)` is not
// injective in the triple, and two pixels that share a seed trace identical
// relative paths, which reads as low-frequency blotching. Hashing one side
// before the XOR removes the structure.
fn hash_combine(a: u32, b: u32) -> u32 {
    return pcg_hash(a ^ pcg_hash(b));
}

// 24 bits rather than 32: f32(u32) rounds to nearest, so the top of the range
// converts to exactly 1.0 and a draw used to index an array would run off its
// end. Taking the high bits keeps the stratification, which lives there.
fn unit_float(x: u32) -> f32 {
    return f32(x >> 8u) * 5.9604645e-8;
}

// ---------------------------------------------------------------------------
// Sampler
//
// Owen-scrambled Sobol, hash-based and table-free, after Burley 2020.
//
// Owen rather than stratification: adaptive sampling retires each pixel at its
// own unpredictable n, and a partial set of strata is biased, not merely
// noisier. Owen scrambling makes every point marginally uniform, so a truncated
// prefix stays unbiased.
//
// Everything down to sobol_1 is mirrored in Rust in renderer/sampler_test.rs,
// which checks the bijection and the net property on the CPU. Its constants are
// transcribed by hand, so it also pins them against this file's text.

// Laine-Karras permutation. Every multiplier is even, which makes each
// `v ^= v * C` triangular with a unit diagonal and so a bijection on u32 -- the
// property the whole scramble rests on.
fn laine_karras_permutation(x: u32, seed: u32) -> u32 {
    var v = x + seed;
    v ^= v * 0x6c50b47cu;
    v ^= v * 0xb82f1e52u;
    v ^= v * 0xc7afe638u;
    v ^= v * 0x8d22f6e6u;
    return v;
}

// Hash-based Owen scramble. Bit k of the result depends only on bits 31..k of
// the input, which is what makes it a nested permutation of the unit interval
// rather than an arbitrary shuffle, and therefore what lets it preserve the net
// property of the sequence it is applied to.
fn nested_uniform_scramble(x: u32, seed: u32) -> u32 {
    return reverseBits(laine_karras_permutation(reverseBits(x), seed));
}

// Sobol dimension 0 is plain van der Corput.
fn sobol_0(index: u32) -> u32 {
    return reverseBits(index);
}

// Sobol dimension 1, without the Antonov-Saleev loop. Its direction numbers are
// the rows of Pascal's triangle mod 2, so by Lucas' theorem bit j of the result
// is the parity of the index bits that are supersets of j -- a superset zeta
// transform, five shift-and-XOR layers. Verified exhaustively against the
// recurrence in renderer/sampler_test.rs.
fn sobol_1(index: u32) -> u32 {
    var g = index;
    g ^= (g >> 1u) & 0x55555555u;
    g ^= (g >> 2u) & 0x33333333u;
    g ^= (g >> 4u) & 0x0f0f0f0fu;
    g ^= (g >> 8u) & 0x00ff00ffu;
    g ^= (g >> 16u) & 0x0000ffffu;
    // The transform indexes bits from the top, the sequence from the bottom.
    return reverseBits(g);
}

// Everything that identifies a sample, and nothing else.
//
// Deliberately no running counter: a draw is a pure function of its dimension,
// so a branch that skips a draw cannot shift the draws after it.
//
// The invariant, not visible from the code: the seed is a pure function of
// (pixel, sample index, restart index), and a pixel always traces the prefix
// 0..n regardless of how the CPU groups those into batches. That keeps the
// stream independent of `samples_per_batch`, and so of GPU timing.
struct Sampler {
    pixel_seed: u32,
    index: u32,
}

fn sampler_new(pixel_index: u32, sample_index: u32) -> Sampler {
    return Sampler(hash_combine(pixel_index, config.restart_index), sample_index);
}

// One 2D draw. `pair` indexes dimension *pairs*, not dimensions.
//
// Each pair gets its own scramble seed, so the budget is N independent 2D
// (0,2)-sequences rather than one 2N-dimensional Sobol sequence. A high-
// dimensional sequence's later dimensions are poorly stratified at any sample
// count a renderer reaches, and the integrand's smooth structure lives within
// each pair -- a lens disc, a light's surface, a cosine hemisphere.
fn sampler_2d(s: Sampler, pair: u32) -> vec2<f32> {
    let seed = hash_combine(s.pixel_seed, pair);
    if (low_discrepancy != 0.0) {
        // The sample index is shuffled per pair before the sequence is
        // generated, not merely scrambled after. Scrambling the outputs alone
        // leaves the pads walking in lockstep -- every dimension crosses into
        // its other half on the same sample -- and the error then stops falling
        // (RMSE flat at 0.17 from 64 spp up, rather than halving per 4x).
        //
        // Safe for the same reason Owen scrambling is: a nested permutation
        // maps any prefix of 2^m samples onto a 2^m-aligned contiguous block,
        // and every such block of a (0,2)-sequence is a (0,m,2)-net. No sample
        // count has to be known up front, which adaptive sampling requires.
        let i = nested_uniform_scramble(s.index, seed);
        let seed_x = pcg_hash(seed);
        let seed_y = pcg_hash(seed_x);
        return vec2<f32>(
            unit_float(nested_uniform_scramble(sobol_0(i), seed_x)),
            unit_float(nested_uniform_scramble(sobol_1(i), seed_y)),
        );
    }
    let x = hash_combine(seed, s.index);
    return vec2<f32>(unit_float(x), unit_float(pcg_hash(x)));
}

// The two dimensions of a pair, for the slots that want a single scalar.
fn dim_x(pair: u32) -> u32 { return pair * 2u; }
fn dim_y(pair: u32) -> u32 { return pair * 2u + 1u; }

// One 1D draw. `dim` indexes dimensions, so `dim_x(p)` and `dim_y(p)` are the
// two halves of pair p.
fn sampler_1d(s: Sampler, dim: u32) -> f32 {
    let p = sampler_2d(s, dim >> 1u);
    if ((dim & 1u) == 0u) {
        return p.x;
    }
    return p.y;
}

// A draw outside the budget, decorrelated from every budgeted dimension and
// every other tag. For draws whose count is data-dependent, which cannot get a
// fixed slot without reserving their worst case.
fn sampler_extra(s: Sampler, tag: u32) -> f32 {
    return unit_float(hash_combine(hash_combine(s.pixel_seed, s.index), tag ^ EXTRA_TAG_SALT));
}

fn ray_at(r: Ray, t: f32) -> vec3<f32> {
    return r.origin + t * r.direction;
}

// sRGB EOTF, the exact piecewise form rather than `pow(x, 2.2)`: it is what the
// encoders that wrote these images used. Mirrors `srgb_to_linear` in
// `util/rgb_color.rs`.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let low = c / 12.92;
    let high = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(high, low, c <= vec3<f32>(0.04045));
}

// Closed-form samplers, rather than rejection loops: on a GPU every lane in a
// wavefront waits for its unluckiest neighbour.

fn random_in_unit_disk(u: vec2<f32>) -> vec3<f32> {
    // sqrt of a uniform variate makes the radius uniform by area.
    let r = sqrt(u.x);
    let phi = TWO_PI * u.y;
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

// The inverse of onb_local: a world direction in the frame's coordinates, where
// the shading normal is +z and every cosine is just a component.
fn onb_from_world(onb: ONB, a: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(a, onb.u), dot(a, onb.v), dot(a, onb.w));
}

fn random_cosine_direction(u: vec2<f32>) -> vec3<f32> {
    let r1 = u.x;
    let r2 = u.y;

    let phi = 2.0 * 3.14159265359 * r1;
    let x = cos(phi) * sqrt(r2);
    let y = sin(phi) * sqrt(r2);
    let z = sqrt(1.0 - r2);

    return vec3<f32>(x, y, z);
}

fn random_to_sphere(radius: f32, distance_squared: f32, u: vec2<f32>) -> vec3<f32> {
    let r1 = u.x;
    let r2 = u.y;
    let z = 1.0 + r2 * (sqrt(abs(1.0 - radius * radius / distance_squared)) - 1.0);

    let phi = 2.0 * 3.14159265359 * r1;
    let zz = sqrt(abs(1.0 - z * z));
    let x = cos(phi) * zz;
    let y = sin(phi) * zz;

    return vec3<f32>(x, y, z);
}

fn triangle_random_direction(t: TrianglePos, origin: vec3<f32>, u: vec2<f32>) -> vec3<f32> {
    var a = u.x;
    var b = u.y;
    if (a + b > 1.0) {
        a = 1.0 - a;
        b = 1.0 - b;
    }
    let p = t.v0 + a * t.e1 + b * t.e2;
    return p - origin;
}

// Where a sampled quad point becomes a direction, on both arms: `u` is the
// point in the quad's own (u, v) parameters, as `quad_solid_angle_uv` returns.
// On its own this is area sampling, the fallback for a quad the spherical-
// rectangle parametrisation does not cover.
fn quad_random_direction(q: QuadPos, origin: vec3<f32>, u: vec2<f32>) -> vec3<f32> {
    let p = q.Q + q.u * u.x + q.v * u.y;
    return p - origin;
}

// How near to a right angle a quad's edges must be for the spherical-rectangle
// parametrisation to apply. `Quad::new` takes any two edge vectors, so a
// sheared parallelogram is representable; the parametrisation is not defined
// for one.
const QUAD_ORTHOGONAL_EPS = 1e-4;

// Smallest solid angle worth the parametrisation, and the numerical guard.
// `solid_angle` is four interior angles less 2*pi, a difference of quantities
// near pi, which leaves about 1e-6 of absolute f32 error however small the true
// value is. Below 1e-3 area sampling is already near-optimal anyway.
const MIN_SOLID_ANGLE = 1e-3;

// A quad seen from a shading point, in the frame the spherical-rectangle
// parametrisation works in (Urena, Fajardo & King 2013, "An Area-Preserving
// Parametrization for Spherical Rectangles").
//
// The frame's axes are not kept: everything downstream is a coordinate in it.
// Carrying three more vec3s through `sample_light` costs 40 VGPRs and a quarter
// of the occupancy.
//
// A zero `solid_angle` means this quad is not sampled that way from here, which
// is what `quad_use_solid_angle` reads -- one computation, so the sampler and
// the PDF cannot disagree about which arm a quad is on.
struct QuadSphericalRect {
    // Signed distance to the quad's plane, negative by construction.
    z0: f32,
    // The quad's extent in the frame, as seen from the shading point.
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
    // The two constants the inverse CDF in x needs, and its lower limit.
    b0: f32,
    b1: f32,
    k: f32,
    solid_angle: f32,
}

fn quad_spherical(q: QuadPos, origin: vec3<f32>) -> QuadSphericalRect {
    var sr: QuadSphericalRect;
    sr.solid_angle = 0.0;

    let ul = length(q.u);
    let vl = length(q.v);
    let ex = q.u / ul;
    let ey = q.v / vl;
    // Negated rather than written as `>=`, so a degenerate quad -- whose edges
    // normalise to NaN -- falls out here too.
    if (!(abs(dot(ex, ey)) < QUAD_ORTHOGONAL_EPS)) { return sr; }

    let d = q.Q - origin;
    // The paper points the third axis at the quad, which negates `z0` and the
    // axis together and leaves the other two alone. Only `z0` survives into the
    // arithmetic below, so the flip is just its sign.
    sr.z0 = -abs(dot(d, cross(ex, ey)));
    sr.x0 = dot(d, ex);
    sr.y0 = dot(d, ey);
    sr.x1 = sr.x0 + ul;
    sr.y1 = sr.y0 + vl;

    // Girard's theorem: the solid angle is the four interior angles less 2*pi.
    // In this frame each of the paper's corner normals has a zero component, so
    // its cross products and vec3 normalizes collapse into the four scalar
    // reciprocal square roots below, and `b0` and `b1` with them.
    let z0sq = sr.z0 * sr.z0;
    let ix0 = inverseSqrt(z0sq + sr.x0 * sr.x0);
    let ix1 = inverseSqrt(z0sq + sr.x1 * sr.x1);
    let iy0 = inverseSqrt(z0sq + sr.y0 * sr.y0);
    let iy1 = inverseSqrt(z0sq + sr.y1 * sr.y1);

    let g0 = acos(clamp(sr.x1 * sr.y0 * ix1 * iy0, -1.0, 1.0));
    let g1 = acos(clamp(-sr.x1 * sr.y1 * ix1 * iy1, -1.0, 1.0));
    let g2 = acos(clamp(sr.x0 * sr.y1 * ix0 * iy1, -1.0, 1.0));
    let g3 = acos(clamp(-sr.x0 * sr.y0 * ix0 * iy0, -1.0, 1.0));

    sr.b0 = -sr.y0 * iy0;
    sr.b1 = sr.y1 * iy1;
    sr.k = TWO_PI - g2 - g3;
    sr.solid_angle = g0 + g1 - sr.k;
    return sr;
}

fn quad_use_solid_angle(sr: QuadSphericalRect) -> bool {
    return sr.solid_angle > MIN_SOLID_ANGLE;
}

// The one definition of a quad light's density, on either arm, so the sampler
// and the PDF cannot drift apart.
//
// `direction`, `distance` and `normal` are the area arm's conversion and unread
// on the solid-angle arm, which is already a density over directions.
fn quad_prim_pdf(
    sr: QuadSphericalRect,
    prim_index: u32,
    direction: vec3<f32>,
    distance: f32,
    normal: vec3<f32>,
) -> f32 {
    if (quad_use_solid_angle(sr)) { return 1.0 / sr.solid_angle; }
    return (distance * distance) / (abs(dot(direction, normal)) * quad_attr[prim_index].area);
}

// A point on the quad whose direction is uniform in solid angle, so the density
// is the constant `1 / solid_angle`. In the quad's own (u, v) parameters, for
// `quad_random_direction` to turn into a direction.
fn quad_solid_angle_uv(sr: QuadSphericalRect, u: vec2<f32>) -> vec2<f32> {
    // u.x picks a solid-angle-proportional slice of the rectangle; `xu` is
    // where that slice cuts it.
    let au = u.x * sr.solid_angle + sr.k;
    let fu = (cos(au) * sr.b0 - sr.b1) / sin(au);
    // fu overflowing to infinity is the sin(au) -> 0 limit, and lands on the
    // right answer of cu = 0.
    var cu = inverseSqrt(fu * fu + sr.b0 * sr.b0) * select(-1.0, 1.0, fu > 0.0);
    cu = clamp(cu, -1.0, 1.0);

    var xu = -(cu * sr.z0) * inverseSqrt(max(1.0 - cu * cu, 1e-20));
    xu = clamp(xu, sr.x0, sr.x1);

    // Then u.y along that slice, which is a 1D projected-angle integral with a
    // closed form.
    let d2 = xu * xu + sr.z0 * sr.z0;
    let h0 = sr.y0 * inverseSqrt(d2 + sr.y0 * sr.y0);
    let h1 = sr.y1 * inverseSqrt(d2 + sr.y1 * sr.y1);
    let hv = h0 + u.y * (h1 - h0);
    let hv2 = hv * hv;
    // |hv| at 1 is the grazing limit, where yv runs off to an edge. The paper's
    // code always takes y1 there; the sign of hv says which edge it is.
    let edge = select(sr.y0, sr.y1, hv > 0.0);
    let yv = select(edge, hv * sqrt(d2) * inverseSqrt(1.0 - hv2), hv2 < 1.0 - 1e-6);

    return vec2<f32>((xu - sr.x0) / (sr.x1 - sr.x0), (yv - sr.y0) / (sr.y1 - sr.y0));
}

fn sphere_random_direction(s: Sphere, origin: vec3<f32>, u: vec2<f32>) -> vec3<f32> {
    let center = s.center_and_radius.xyz;
    let radius = s.center_and_radius.w;
    let direction = center - origin;
    let uvw = onb_from_w(direction);
    return onb_local(uvw, random_to_sphere(radius, dot(direction, direction), u));
}

// The one definition of a light's solid-angle density, called by sample_light
// and by the BSDF-side MIS weight, so the two strategies cannot drift apart.
// Returns the density of this primitive alone; the caller multiplies by the
// probability of having picked it.
//
// `direction` must be unit and `distance` in world units -- the primary ray is
// not normalised (issue #57) and never reaches either caller.
//
// `normal` is the geometric normal: the area-to-solid-angle Jacobian belongs to
// the facet the point was sampled on. Unused for spheres, and for a quad on the
// solid-angle arm, which are sampled in the direction domain already.
fn light_prim_pdf(
    prim_type: u32,
    prim_index: u32,
    origin: vec3<f32>,
    direction: vec3<f32>,
    distance: f32,
    normal: vec3<f32>,
) -> f32 {
    // Triangles are the fallback arm here and everywhere else a primitive type
    // is dispatched on, so a scene without spheres or quads leaves nothing of
    // this chain behind.
    if (has_spheres && prim_type == PRIM_TYPE_SPHERE) {
        let s = spheres[prim_index];
        let center = s.center_and_radius.xyz;
        let radius = s.center_and_radius.w;
        let dist_sq = dot(center - origin, center - origin);
        let cos_theta_max = sqrt(abs(1.0 - radius * radius / dist_sq));
        let solid_angle = 2.0 * 3.14159265359 * (1.0 - cos_theta_max);
        return 1.0 / solid_angle;
    }

    if (has_quads && prim_type == PRIM_TYPE_QUAD) {
        return quad_prim_pdf(
            quad_spherical(quad_pos[prim_index], origin),
            prim_index,
            direction,
            distance,
            normal,
        );
    }
    let area = triangle_attr[prim_index].area;
    return (distance * distance) / (abs(dot(direction, normal)) * area);
}

// Which entry of `lights` a primitive is, or `light_count` if it is not an
// emitter sample_light draws from. The index rather than a yes/no, because the
// caller needs the entry's `select_pdf`.
//
// `lights` is sorted by the packed `(prim_type, prim_index)` key `prim_refs`
// uses, so this is a binary search: three iterations at eight lights, ten at a
// thousand. Storing a light index on the primitive is the escape hatch if a
// profile ever shows this search, but QuadAttr is exactly 32 bytes with no
// slack for one.
fn find_light(prim_type: u32, prim_index: u32) -> u32 {
    let key = (prim_type << PRIM_TYPE_SHIFT) | prim_index;
    var lo = 0u;
    var hi = light_count;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (lights[mid].prim < key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo >= light_count || lights[lo].prim != key) { return light_count; }
    return lo;
}

// The light strategy's density for a direction a BSDF path followed to an
// emitter: the other half of the MIS weight at that vertex.
//
// Only the emitter the path reached can have produced a light sample along this
// direction, since any emitter further along is blocked by this one. So the
// density is exactly `P(pick this one) * p_this(w)`, with no sum over the scene
// and no extra intersection test.
//
// A `Blend` holding a DiffuseLight is not in `lights`, so it lands here as 0
// and the hit is taken at weight 1.
fn light_hit_pdf(
    prim_type: u32,
    prim_index: u32,
    origin: vec3<f32>,
    direction: vec3<f32>,
    distance: f32,
    normal: vec3<f32>,
) -> f32 {
    if (light_count == 0u) { return 0.0; }
    let idx = find_light(prim_type, prim_index);
    if (idx >= light_count) { return 0.0; }
    return light_prim_pdf(prim_type, prim_index, origin, direction, distance, normal)
        * lights[idx].select_pdf;
}

// One sample of the direct-lighting strategy: pick a light in proportion to
// its emitted power, sample a point on it, and report what a shadow ray would
// need to reach it.
struct LightSample {
    // Unit vector from the shading point toward the sampled point.
    direction: vec3<f32>,
    // Distance to the sampled point, so the shadow ray can stop short of it.
    distance: f32,
    // Solid-angle density of the whole strategy at `direction`: the chance of
    // having picked this light times the density of the point on it.
    pdf: f32,
    emission: vec3<f32>,
    attenuation_factor: f32,
    valid: bool,
}

// `pick` selects the light, `u` the point on it. Two slots rather than one
// stream, so a scene with one light draws the same pair as a scene with ten.
//
// Selection is power-proportional and costs that one scalar draw: `pick * L`
// splits into a slot and a coin, and the alias table turns the pair into a draw
// from the light distribution. Uniform selection is the special case where
// every `alias_prob` is 1 and every `select_pdf` is `1 / L`.
fn sample_light(origin: vec3<f32>, pick: f32, u: vec2<f32>) -> LightSample {
    var ls: LightSample;
    ls.direction = vec3<f32>(0.0, 1.0, 0.0);
    ls.distance = 0.0;
    ls.pdf = 0.0;
    ls.emission = vec3<f32>(0.0);
    ls.attenuation_factor = 0.0;
    ls.valid = false;

    if (light_count == 0u) { return ls; }

    // Integer part picks the slot, fractional part is the alias coin. The two
    // are independent for a uniform `pick`, which is what makes one draw enough.
    let scaled = pick * f32(light_count);
    let slot = min(u32(scaled), light_count - 1u);
    var light = lights[slot];
    if (scaled - f32(slot) >= light.alias_prob) {
        light = lights[light.alias_index];
    }

    let prim_type = light.prim >> PRIM_TYPE_SHIFT;
    let prim_index = light.prim & PRIM_INDEX_MASK;

    // Hoisted only so the density below can reuse it rather than rebuild it;
    // every field but `solid_angle` is dead by then.
    var sr: QuadSphericalRect;
    var to_light = vec3<f32>(0.0);
    if (has_spheres && prim_type == PRIM_TYPE_SPHERE) {
        to_light = sphere_random_direction(spheres[prim_index], origin, u);
    } else if (has_quads && prim_type == PRIM_TYPE_QUAD) {
        sr = quad_spherical(quad_pos[prim_index], origin);
        var quv = u;
        if (quad_use_solid_angle(sr)) { quv = quad_solid_angle_uv(sr, u); }
        to_light = quad_random_direction(quad_pos[prim_index], origin, quv);
    } else {
        to_light = triangle_random_direction(triangle_pos[prim_index], origin, u);
    }

    let len_sq = dot(to_light, to_light);
    if (len_sq < 1e-12) { return ls; }
    let dir = to_light * inverseSqrt(len_sq);

    // Intersect the chosen light itself: sphere sampling yields a direction
    // rather than a point, and the distance is needed either way so the shadow
    // ray can stop just short of the light. A solid-angle-sampled quad is
    // always hit, so there it doubles as a guard against a parametrisation that
    // has gone wrong, turning a bias into a lost sample.
    let probe = Ray(origin, dir);
    var t_hit = 0.0;
    var bary = vec2<f32>(0.0);
    var normal = vec3<f32>(0.0);
    var mat_idx = 0u;

    if (has_spheres && prim_type == PRIM_TYPE_SPHERE) {
        if (!hit_sphere_t(probe, spheres[prim_index], RAY_EPS, 1e20, &t_hit)) { return ls; }
        let sph = spheres[prim_index];
        normal = (ray_at(probe, t_hit) - sph.center_and_radius.xyz) / sph.center_and_radius.w;
        mat_idx = sph.material_index;
    } else if (has_quads && prim_type == PRIM_TYPE_QUAD) {
        if (!hit_quad_t(probe, quad_pos[prim_index], RAY_EPS, 1e20, &t_hit, &bary)) { return ls; }
        normal = quad_pos[prim_index].normal;
        mat_idx = quad_attr[prim_index].material_index;
    } else {
        if (!hit_triangle_t(probe, triangle_pos[prim_index], RAY_EPS, 1e20, &t_hit, &bary)) { return ls; }
        let attr = triangle_attr[prim_index];
        normal = attr.normal;
        mat_idx = attr.material_index;
    }

    // Lights emit from their front face only, matching what a BSDF path sees
    // when it lands on one. Sampling the back is a valid direction with a real
    // PDF, it just carries no radiance -- so the MIS weights stay consistent.
    if (dot(dir, normal) >= 0.0) { return ls; }

    let material = materials[mat_idx];
    let emission = select(vec3<f32>(0.0), material.emission, material.mat_type == MAT_DIFFUSE_LIGHT);

    var prim_pdf = 0.0;
    if (has_quads && prim_type == PRIM_TYPE_QUAD) {
        prim_pdf = quad_prim_pdf(sr, prim_index, dir, t_hit, normal);
    } else {
        prim_pdf = light_prim_pdf(prim_type, prim_index, origin, dir, t_hit, normal);
    }

    ls.direction = dir;
    ls.distance = t_hit;
    ls.pdf = prim_pdf * light.select_pdf;
    ls.emission = emission;
    ls.attenuation_factor = material.attenuation_factor;
    ls.valid = true;
    return ls;
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

// Fresnel reflectance at a dielectric interface, for unpolarised light: the
// exact mean of the two polarisations, and 1 past the critical angle.
//
// `eta_it` is `Bsdf.ior`, the near medium's index over the far one's, and the
// term-for-term counterpart of PBRT-v4's `FrDielectric` written with that
// reciprocal.
//
// Exact rather than Schlick, which is wrong where a dielectric spends most of
// its bounces: looking out from inside 1.5 glass, the exact term reaches 1 at
// the critical angle of 41.8 degrees where Schlick reports 0.041, so the two
// lobes are mixed wrongly across the whole band total internal reflection
// dominates. From outside the two agree to 0.006. Costs 23 instructions out of
// 6266 on `test_scene`, with occupancy unchanged.
fn fresnel_dielectric(cos_i: f32, eta_it: f32) -> f32 {
    let c = clamp(cos_i, 0.0, 1.0);
    let sin2_t = eta_it * eta_it * (1.0 - c * c);
    // Total internal reflection: no transmitted direction exists, so all of it
    // comes back.
    if (sin2_t >= 1.0) { return 1.0; }
    let cos_t = sqrt(1.0 - sin2_t);

    let r_parallel = (c - eta_it * cos_t) / (c + eta_it * cos_t);
    let r_perpendicular = (eta_it * c - cos_t) / (eta_it * c + cos_t);
    return 0.5 * (r_parallel * r_parallel + r_perpendicular * r_perpendicular);
}

// ---------------------------------------------------------------------------
// GGX microfacet conductor
//
// Isotropic GGX with the height-correlated Smith masking-shadowing term and
// Schlick's Fresnel. `alpha` is the roughness-squared parameter; the vectors
// below live in the shading frame, where the shading normal is +z, so every
// cosine is a `.z`.
// ---------------------------------------------------------------------------

// Below this alpha the lobe is narrower than the sampler can resolve and the
// conductor becomes a Dirac lobe instead, which keeps `Metal::new(.., 0.)` an
// exact mirror.
const GGX_ALPHA_MIN = 1e-3;
// The same cutoff in the user's units, since alpha = fuzz * fuzz.
const FUZZ_SPECULAR_THRESHOLD = 0.0316227766;

fn ggx_d(cos_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let t = 1.0 + cos_h * cos_h * (a2 - 1.0);
    return a2 / (PI * t * t);
}

// Smith's Lambda for GGX, from which G1 = 1 / (1 + Lambda) and the height-
// correlated G2 = 1 / (1 + Lambda(wo) + Lambda(wi)). Kept in Lambda form
// because the sampled weight wants G2 / G1(wo), which then cancels to
// (1 + Lambda_o) / (1 + Lambda_o + Lambda_i) with no divisions.
fn smith_lambda(cos_w: f32, alpha: f32) -> f32 {
    let c2 = cos_w * cos_w;
    let tan2 = max(0.0, 1.0 - c2) / max(c2, 1e-8);
    return 0.5 * (sqrt(1.0 + alpha * alpha * tan2) - 1.0);
}

// Schlick's approximation, with the conductor's normal-incidence reflectance as
// f0. What makes a metal go white at a grazing angle, and why `Metal::albedo`
// means f0 rather than a flat multiplier.
fn fresnel_schlick(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    let m = clamp(1.0 - cos_theta, 0.0, 1.0);
    let m2 = m * m;
    return f0 + (vec3<f32>(1.0) - f0) * (m2 * m2 * m);
}

// The directional albedo of the single-scatter lobe with F = 1: how much of
// what arrives from a uniform environment at `cos_o` leaves again. The rest is
// light a microfacet reflected onto another microfacet and the single-scatter
// model then dropped.
//
// A fit rather than Kulla-Conty's precomputed table, which would mean a large
// `const` array initialiser. Fitted in log space, since the factor built on it
// divides by E, against E from quadrature over the BRDF's own definition. In
// `fuzz` rather than `alpha`, where the same sixteen terms are worth 3.3%
// against 6.8% over the range. The residual is the last few degrees of the
// silhouette; end to end the furnace test reads better than 0.3%.
fn ggx_directional_albedo(cos_o: f32, alpha: f32) -> f32 {
    let r = sqrt(alpha);
    let m = clamp(cos_o, 0.0, 1.0);
    let p1 = -1.152223 + m * (8.928465 + m * (-16.619661 + m * 8.678675));
    let p2 = 3.873487 + m * (-43.813387 + m * (91.923225 + m * -50.492525));
    let p3 = -4.397634 + m * (58.765802 + m * (-136.618743 + m * 78.573419));
    let p4 = 1.598114 + m * (-26.063835 + m * (63.194110 + m * -37.572948));
    // Clamped both ways: the fit overshoots 1 by a per cent near the mirror
    // end, where the compensation should do nothing at all.
    return clamp(exp(r * (p1 + r * (p2 + r * (p3 + r * p4)))), 0.05, 1.0);
}

// Turquin 2019, "Practical multiple scattering compensation for microfacet
// models": one multiplicative factor returns the energy the single-scatter lobe
// drops.
//
//   f_ms = f_ss * (1 + f0 * (1 / E(cos_o, alpha) - 1))
//
// The f0 weighting makes a coloured metal saturate rather than merely brighten:
// light that bounces twice between microfacets is tinted twice. Multiplies `f`
// alone, so the sampling density and MIS are untouched.
fn ggx_multiscatter(f0: vec3<f32>, cos_o: f32, alpha: f32) -> vec3<f32> {
    return vec3<f32>(1.0) + f0 * (1.0 / ggx_directional_albedo(cos_o, alpha) - 1.0);
}

// Samples a visible normal of the GGX distribution: Dupuy & Benyoub 2023,
// "Sampling Visible GGX Normals with Spherical Caps". The same distribution as
// Heitz 2018, with no stretched frame to build and no special case for a `wo`
// that the shading normal disagrees with.
fn sample_ggx_vndf(wo: vec3<f32>, alpha: f32, u: vec2<f32>) -> vec3<f32> {
    // Warp to the hemisphere configuration, where the normals visible from wo
    // are exactly a spherical cap.
    let wo_std = normalize(vec3<f32>(wo.xy * alpha, wo.z));
    // Sample that cap: z uniform in (-wo_std.z, 1].
    let phi = TWO_PI * u.x;
    let z = fma(1.0 - u.y, 1.0 + wo_std.z, -wo_std.z);
    let sin_theta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
    let c = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), z);
    // The cap sample plus wo_std is the half vector in the hemisphere
    // configuration; unwarping it lands back on the ellipsoid.
    let h_std = c + wo_std;
    return normalize(vec3<f32>(h_std.xy * alpha, h_std.z));
}

// ---------------------------------------------------------------------------
// GGX microfacet dielectric
//
// Walter et al. 2007, "Microfacet Models for Refraction through Rough
// Surfaces", sampled with the same Dupuy & Benyoub visible-normal routine the
// conductor uses, and sharing everything above it.
//
// Two things separate it from the conductor: the lobe has a transmission half
// carrying its own Jacobian, and the half vector may be on either side of the
// shading normal -- so the arms below work in a frame flipped so `wo` is up,
// and flip the sampled direction back.
//
// `eta_ti` is the far medium's index over the near one's, PBRT-v4's `etap`, so
// the transmission algebra matches that reference term for term. `Bsdf.ior` is
// its reciprocal, which is what `refract` wants.
// ---------------------------------------------------------------------------

// The half vector a transmitted pair implies, unnormalised: Walter's
// generalised half vector, scaled by the near medium's index so only the ratio
// appears. Faced toward +z by the caller, as the reflection half vector already
// is.
fn dielectric_half_transmit(wo: vec3<f32>, wi: vec3<f32>, eta_ti: f32) -> vec3<f32> {
    return wi * eta_ti + wo;
}

// Grid shape, mirroring `ENERGY_COS_STEPS` and `ENERGY_ALPHA_STEPS`.
const ENERGY_COS_STEPS = 16u;
const ENERGY_ALPHA_STEPS = 16u;
const ENERGY_SIDE_STRIDE = ENERGY_COS_STEPS * ENERGY_ALPHA_STEPS;

// Turquin 2019's compensation for the dielectric: dividing `f` by the energy
// the single-scatter lobe does return puts back what it drops.
//
// Two differences from the conductor's. No `f0` weighting, since a dielectric
// interface tints nothing, so the factor is a plain `1 / E`. And `E` comes from
// a bilinear table rather than a polynomial, because it depends on the index of
// refraction through the critical angle, a kink no polynomial follows.
//
// Multiplies `f` alone, so the sampling density and MIS are untouched.
fn dielectric_multiscatter(offset: u32, cos_o: f32, alpha: f32) -> f32 {
    let x = clamp(cos_o, 0.0, 1.0) * f32(ENERGY_COS_STEPS - 1u);
    let y = clamp(sqrt(alpha), 0.0, 1.0) * f32(ENERGY_ALPHA_STEPS - 1u);
    let x0 = min(u32(x), ENERGY_COS_STEPS - 2u);
    let y0 = min(u32(y), ENERGY_ALPHA_STEPS - 2u);
    let fx = x - f32(x0);
    let fy = y - f32(y0);

    let row0 = offset + y0 * ENERGY_COS_STEPS + x0;
    let row1 = row0 + ENERGY_COS_STEPS;
    let lo = mix(dielectric_energy[row0], dielectric_energy[row0 + 1u], fx);
    let hi = mix(dielectric_energy[row1], dielectric_energy[row1 + 1u], fx);
    // Floored for the same reason the conductor's fit is clamped: nothing here
    // may turn into a division by zero at the rough, grazing corner.
    return 1.0 / max(mix(lo, hi, fy), 0.05);
}

// d(w_h) / d(w_i) for the transmission lobe. Easy to write upside down, which
// is why `test_bsdf_only_sampling_converges_to_the_same_image` carries a rough
// glass scene: a Jacobian wrong by a constant moves the BSDF-only mean and
// nothing else in the suite.
fn dielectric_dwh_dwi(cos_ih: f32, cos_oh: f32, eta_ti: f32) -> f32 {
    let d = cos_ih + cos_oh / eta_ti;
    return abs(cos_ih) / max(d * d, 1e-12);
}

// What the pixel is looking at, before any light transport: the first surface
// along the view ray that is not a mirror or a lens. Only the denoiser's
// edge-stopping functions read it -- a guide, not a signal, which is what makes
// the lossy packing below acceptable.
struct GuideSample {
    // Tinted by whatever specular surfaces the guide ray passed through, so it
    // describes the colour this pixel should end up rather than the colour of a
    // surface it only sees in a mirror.
    albedo: vec3<f32>,
    normal: vec3<f32>,
    // Path length from the camera in world units, summed over the guide chain.
    // The guide ray is normalised from the start, unlike the primary ray in
    // trace_sample (issue #57), so the segments are in the same units.
    depth: f32,
    mat_type: u32,
    // How many specular bounces the guide ray took to get here. Lets the filter
    // tell a wall seen in a mirror from the same wall seen directly, which the
    // other three channels can agree on by coincidence.
    specular_depth: u32,
}

// Octahedral normal encoding. Two floats instead of three, with ~0.01 degrees of
// error at 16-bit. Three copies of the encode/decode pair exist -- this one,
// post/denoise_atrous.wgsl's and `pack_oct` in renderer/scene_flattener.rs --
// and all must agree, including the `n.z <= 0.0` polarity of the fold.
fn oct_encode(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));
    if (n.z <= 0.0) {
        let s = vec2<f32>(select(-1.0, 1.0, p.x >= 0.0), select(-1.0, 1.0, p.y >= 0.0));
        return (1.0 - abs(vec2<f32>(p.y, p.x))) * s;
    }
    return p;
}

fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    var v = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) {
        let s = vec2<f32>(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
        v = vec3<f32>((1.0 - abs(vec2<f32>(v.y, v.x))) * s, v.z);
    }
    return normalize(v);
}

// 16 bytes per pixel. The depth goes through bitcast rather than sharing an f32
// slot with packed bits, because a packed bit pattern can land on a NaN encoding
// and some drivers canonicalise NaN payloads across a store/load. pack4x8unorm
// clamps, so an albedo above 1 degrades the guide but never the image.
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
    // After interpolation and the normal map.
    normal: vec3<f32>,
    // Carried from the hit unchanged: neither interpolation nor a normal map
    // may move what the scattering gates check against.
    geometric_normal: vec3<f32>,
    emission: vec3<f32>,
    attenuation_factor: f32,
    fuzz: f32,
    refraction_index: f32,
    mat_type: u32,
    // Carried straight from the material; `bsdf_from_surface` picks the half
    // that matches the side the ray is on.
    energy_offset: u32,
}

// The blend walk nests to a data-dependent depth, so its coin flips go to
// `sampler_extra` rather than the pair budget: stratifying a discrete material
// choice buys nothing, and reserving ten pairs per bounce would cost more in
// decorrelation than it returns.
fn resolve_surface(rec: HitRecord, smp: Sampler, depth: u32) -> Surface {
    var mat_idx = rec.material_index;
    if (has_blends) {
        for (var i = 0u; i < 10u; i++) {
            let material = materials[mat_idx];
            if (material.mat_type == MAT_BLEND) {
                if (sampler_extra(smp, depth * 16u + i) > material.blend_factor) {
                    mat_idx = material.blend_indices.x;
                } else {
                    mat_idx = material.blend_indices.y;
                }
            } else {
                break;
            }
        }
    }

    return surface_at(mat_idx, rec);
}

// The blend walk above, resolved deterministically: the branch the coin flip
// would take more often than not. Only trace_guide wants this, because a
// stochastic choice would make neighbouring pixels disagree about what surface
// they are looking at, which is the one thing an edge stop cannot survive.
fn resolve_material_index_dominant(start: u32) -> u32 {
    var mat_idx = start;
    if (has_blends) {
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
    }
    return mat_idx;
}

// Everything after the blend is chosen: albedo and shading normal sampled from
// their textures. Shared by the stochastic and deterministic walks above.
fn surface_at(mat_idx: u32, rec: HitRecord) -> Surface {
    let material = materials[mat_idx];

    var surface: Surface;
    surface.geometric_normal = rec.geometric_normal;
    surface.mat_type = material.mat_type;
    surface.emission = material.emission;
    surface.attenuation_factor = material.attenuation_factor;
    surface.fuzz = material.fuzz;
    surface.refraction_index = material.refraction_index;
    surface.energy_offset = material.energy_offset;

    surface.albedo = material.albedo;
    if (has_textures && material.texture_index >= 0) {
        let uv = vec2<f32>(fract(abs(rec.uv.x)), 1.0 - fract(abs(rec.uv.y)));
        let uv_atlas = material.albedo_offset + uv * material.albedo_scale;
        // Decoded here rather than by the hardware, because the atlas is
        // shared with normal maps, which must stay raw. Free while the sampler
        // is `Nearest`; if mipmaps land (#60), switch to a second
        // `Rgba8UnormSrgb` view over the same texture.
        surface.albedo = srgb_to_linear(textureSampleLevel(texture_array, texture_sampler, uv_atlas, 0.0).rgb);
    }

    surface.normal = rec.normal;
    if (has_normal_maps && material.normal_texture_index >= 0) {
        let uv = vec2<f32>(fract(abs(rec.uv.x)), 1.0 - fract(abs(rec.uv.y)));
        let uv_atlas = material.normal_offset + uv * material.normal_scale;
        let map_color = textureSampleLevel(texture_array, texture_sampler, uv_atlas, 0.0).rgb;
        let map_n = map_color * 2.0 - 1.0;
        surface.normal = normalize(map_n.x * rec.tangent + map_n.y * rec.bi_tangent + map_n.z * rec.normal);
    }

    return surface;
}

// Beer-Lambert absorption across `t` world units of glass.
//
// `transmission_per_unit` is a dielectric's albedo, read as the fraction that
// survives one world unit of travel; `(1,1,1)` is an exact no-op. The floor
// keeps `pow` off `log2(0)`, which would be NaN.
//
// Callers apply it on a back-face dielectric hit, where `rec.t` is the length
// of the segment inside the glass. That assumes entry and exit belong to the
// same material, which holds for any closed glass object.
fn dielectric_transmittance(transmission_per_unit: vec3<f32>, t: f32) -> vec3<f32> {
    return pow(max(transmission_per_unit, vec3<f32>(1e-4)), vec3<f32>(t));
}

// ---------------------------------------------------------------------------
// BSDF
//
// One layer between the transport loop and the material arms, so next-event
// estimation and MIS are written once rather than per material. The loop asks
// three things of a surface: is this lobe sampleable by a light, what does it
// evaluate to in a given direction, and where does the path go next.
//
// `wo` points *away* from the surface, `-normalize(r.direction)`, in all three
// functions.
// ---------------------------------------------------------------------------

const BSDF_DIFFUSE = 0u;
const BSDF_CONDUCTOR = 1u;
const BSDF_DIELECTRIC = 2u;
// A surface that scatters nothing: only a blend chain deeper than
// `resolve_surface` walks.
const BSDF_NONE = 3u;

// Everything the BSDF layer needs about one vertex, built once per bounce.
struct Bsdf {
    // Shading frame. `w` is the shading normal, and is what every cosine below
    // is taken against.
    frame: ONB,
    // The geometric normal, which the shading frame may disagree with wherever
    // an interpolated normal or a normal map has moved it.
    ng: vec3<f32>,
    // f0 for a conductor, the diffuse reflectance otherwise.
    base_color: vec3<f32>,
    // GGX roughness, the squared-roughness convention: alpha = fuzz * fuzz.
    alpha: f32,
    // Relative index of refraction, already resolved against which side of the
    // surface the ray is on.
    ior: f32,
    // Start of the energy grid for this side of this material, resolved the
    // same way `ior` is.
    energy_offset: u32,
    kind: u32,
}

struct BsdfSample {
    wi: vec3<f32>,
    // f * |cos| / pdf, already divided, so a Dirac lobe can return a weight
    // with pdf = 0 and no special case above.
    weight: vec3<f32>,
    // Solid-angle pdf; 0 for a Dirac lobe.
    pdf: f32,
    specular: bool,
    // False terminates the path: the sampled direction went below the surface,
    // or there is no lobe to sample.
    valid: bool,
}

struct BsdfEval {
    // f(wo, wi) * |dot(ns, wi)|; every caller wants the product.
    f_cos: vec3<f32>,
    pdf: f32,
}

fn bsdf_from_surface(surface: Surface, front_face: bool) -> Bsdf {
    var b: Bsdf;
    b.frame = onb_from_w(surface.normal);
    b.ng = surface.geometric_normal;
    b.base_color = surface.albedo;
    // Roughness squared, the usual convention, clamped because nothing stops a
    // caller passing a fuzz above 1 where GGX stops being meaningful.
    b.alpha = clamp(surface.fuzz * surface.fuzz, 0.0, 1.0);
    b.ior = select(surface.refraction_index, 1.0 / surface.refraction_index, front_face);
    // Entering is the first half of the table, leaving the second.
    b.energy_offset = surface.energy_offset + select(ENERGY_SIDE_STRIDE, 0u, front_face);

    if (surface.mat_type == MAT_LAMBERTIAN) {
        b.kind = BSDF_DIFFUSE;
    } else if (has_metal && surface.mat_type == MAT_METAL) {
        b.kind = BSDF_CONDUCTOR;
    } else if (has_dielectrics && surface.mat_type == MAT_DIELECTRIC) {
        b.kind = BSDF_DIELECTRIC;
    } else {
        b.kind = BSDF_NONE;
    }
    return b;
}

// Whether the lobe is a Dirac delta, and so has no density for a light sample
// to land on. What gates next-event estimation, rather than the material type.
fn bsdf_is_specular(b: Bsdf) -> bool {
    if (has_metal && b.kind == BSDF_CONDUCTOR) {
        // A rough conductor has a density for a light sample to land on, so it
        // gets next-event estimation like any other spread lobe. Only the
        // mirror end of the range is a delta.
        return b.alpha < GGX_ALPHA_MIN;
    }
    // `has_rough_dielectrics` belongs in the condition, not the body: naga
    // folds an override out of an `if` condition and deletes the arm, but
    // leaves `!has_rough_dielectrics || ...` in the body as real code. On
    // `test_scene` that is 8 subgroups per SIMD against 6.
    if (has_dielectrics && has_rough_dielectrics && b.kind == BSDF_DIELECTRIC) {
        // Smooth glass stays a Dirac delta: a shadow ray has zero probability
        // of landing on it.
        return b.alpha < GGX_ALPHA_MIN;
    }
    return b.kind != BSDF_DIFFUSE;
}

// f and pdf for a direction chosen by something other than the BSDF. Returns
// zero for a Dirac lobe, and for any direction on the wrong side of either
// normal.
fn bsdf_eval(b: Bsdf, wo: vec3<f32>, wi: vec3<f32>) -> BsdfEval {
    var e: BsdfEval;
    e.f_cos = vec3<f32>(0.0);
    e.pdf = 0.0;

    if (b.kind == BSDF_DIFFUSE) {
        let cos_i = dot(b.frame.w, wi);
        // The geometric test is not redundant with the shading one: an
        // interpolated normal or a steep normal map near a silhouette can face
        // a light the facet faces away from, carrying light through the
        // surface.
        if (cos_i <= 0.0 || dot(b.ng, wi) <= 0.0) { return e; }
        e.f_cos = (b.base_color / PI) * cos_i;
        e.pdf = cos_i / PI;
    } else if (has_metal && b.kind == BSDF_CONDUCTOR) {
        // A mirror has no density a light sample can land on.
        if (b.alpha < GGX_ALPHA_MIN) { return e; }

        let wo_l = onb_from_world(b.frame, wo);
        let wi_l = onb_from_world(b.frame, wi);
        // Same two-normal gate as the diffuse arm, and for the same reason.
        if (wo_l.z <= 0.0 || wi_l.z <= 0.0 || dot(b.ng, wi) <= 0.0) { return e; }

        let h = normalize(wo_l + wi_l);
        let d = ggx_d(h.z, b.alpha);
        let lambda_o = smith_lambda(wo_l.z, b.alpha);
        let lambda_i = smith_lambda(wi_l.z, b.alpha);
        let g2 = 1.0 / (1.0 + lambda_o + lambda_i);
        let f = fresnel_schlick(b.base_color, dot(wo_l, h));

        // f * cos_i, with the BRDF's own 1 / cos_i already cancelled against it.
        e.f_cos = f * (d * g2 / (4.0 * wo_l.z)) * ggx_multiscatter(b.base_color, wo_l.z, b.alpha);
        // The VNDF sampling density, G1(wo) * D(h) / (4 cos_o): the pdf this
        // lobe would have had for the light's direction, which is what MIS
        // needs.
        e.pdf = d / ((1.0 + lambda_o) * 4.0 * wo_l.z);
    } else if (has_dielectrics && has_rough_dielectrics && b.kind == BSDF_DIELECTRIC) {
        // Smooth glass is a delta, as above.
        if (b.alpha < GGX_ALPHA_MIN) { return e; }

        // Into the frame where wo is up, so the microfacet normal the pair
        // implies is the one `bsdf_sample` would have drawn.
        let cos_no = dot(wo, b.frame.w);
        if (abs(cos_no) < 1e-6) { return e; }
        let flip = select(-1.0, 1.0, cos_no > 0.0);
        let wo_f = onb_from_world(b.frame, wo * flip);
        let wi_f = onb_from_world(b.frame, wi * flip);
        if (abs(wi_f.z) < 1e-6) { return e; }
        let is_reflect = wi_f.z > 0.0;

        // Sided, unlike the conductor's: refraction is supposed to cross the
        // surface, so the geometric normal rules out a direction on the wrong
        // side for the lobe half it claims to be. resolve_hit faces the
        // geometric normal at the ray, so the halves split on the sign alone.
        let cos_ng = dot(b.ng, wi);
        if (is_reflect != (cos_ng > 0.0)) { return e; }

        let eta_ti = 1.0 / b.ior;
        var h = select(
            dielectric_half_transmit(wo_f, wi_f, eta_ti),
            wo_f + wi_f,
            is_reflect,
        );
        let h_len = length(h);
        if (h_len < 1e-6) { return e; }
        h = h / h_len;
        // Faced toward +z, matching the sampled half vector, which comes out of
        // the VNDF routine in the upper hemisphere.
        h = select(-h, h, h.z > 0.0);

        let cos_oh = dot(wo_f, h);
        let cos_ih = dot(wi_f, h);
        // A microfacet facing away from either direction is a solution of the
        // half-vector equation the surface cannot actually present.
        if (cos_oh <= 0.0 || cos_ih * wi_f.z <= 0.0) { return e; }

        let f = fresnel_dielectric(cos_oh, b.ior);
        let d = ggx_d(h.z, b.alpha);
        let lambda_o = smith_lambda(wo_f.z, b.alpha);
        let lambda_i = smith_lambda(wi_f.z, b.alpha);
        let g2 = 1.0 / (1.0 + lambda_o + lambda_i);
        // G1(wo) * D(h) * |wo.h| / |cos_o|, the visible-normal density, which
        // both halves scale by their own Jacobian and lobe-selection chance.
        let d_vis = d * cos_oh / ((1.0 + lambda_o) * wo_f.z);

        // Both halves get the same factor: the single-scatter model cannot say
        // which lobe the returned energy comes back through.
        let comp = dielectric_multiscatter(b.energy_offset, wo_f.z, b.alpha);

        if (is_reflect) {
            // f * |cos_i| with the BSDF's own 1 / |cos_i| already cancelled.
            e.f_cos = vec3<f32>(comp * f * d * g2 / (4.0 * wo_f.z));
            e.pdf = f * d_vis / (4.0 * cos_oh);
        } else {
            let t = 1.0 - f;
            if (t <= 0.0) { return e; }
            let jacobian = dielectric_dwh_dwi(cos_ih, cos_oh, eta_ti);
            // No eta^2 radiance factor, here or in bsdf_sample: it cancels
            // over a closed object, and MIS only needs the two estimators to
            // agree with each other.
            e.f_cos = vec3<f32>(comp * t * d * g2 * cos_oh * jacobian / wo_f.z);
            e.pdf = t * d_vis * jacobian;
        }
    }

    return e;
}

// Samples a continuation direction. The bounce's dimension pairs are passed in
// rather than a stream state, so a lobe that skips a draw leaves a gap rather
// than shifting everything after it.
fn bsdf_sample(b: Bsdf, wo: vec3<f32>, smp: Sampler, bounce_pair: u32) -> BsdfSample {
    var s: BsdfSample;
    s.wi = vec3<f32>(0.0);
    s.weight = vec3<f32>(0.0);
    s.pdf = 0.0;
    s.specular = true;
    s.valid = false;

    let scalars = bounce_pair + PAIR_SCALARS;

    if (b.kind == BSDF_DIFFUSE) {
        let direction = onb_local(b.frame, random_cosine_direction(sampler_2d(smp, bounce_pair + PAIR_BSDF)));
        let cos_theta = dot(b.frame.w, direction);
        // Cosine sampling about the shading normal can aim below the geometry.
        // Terminating rather than resampling keeps the stream deterministic;
        // the lost energy is the shadow terminator, see LIMITATIONS.md.
        if (cos_theta <= 0.0 || dot(b.ng, direction) <= 0.0) { return s; }

        s.wi = normalize(direction);
        // Cosine sampling cancels the BRDF and the cosine exactly, leaving the
        // albedo: (albedo/PI) * cos / (cos/PI).
        s.weight = b.base_color;
        s.pdf = cos_theta / PI;
        s.specular = false;
        s.valid = true;
    } else if (has_metal && b.kind == BSDF_CONDUCTOR) {
        let wo_l = onb_from_world(b.frame, wo);
        // A shading normal the view ray is already behind has no lobe above
        // the surface to sample.
        if (wo_l.z <= 0.0) { return s; }

        if (b.alpha < GGX_ALPHA_MIN) {
            // Mirror: a Dirac lobe, so pdf 0 and the weight is the Fresnel
            // term alone -- the division by the pdf has nothing left to do.
            let direction = reflect(-wo, b.frame.w);
            if (dot(b.ng, direction) <= 0.0) { return s; }

            s.wi = normalize(direction);
            s.weight = fresnel_schlick(b.base_color, wo_l.z);
            s.valid = true;
        } else {
            let h = sample_ggx_vndf(wo_l, b.alpha, sampler_2d(smp, bounce_pair + PAIR_BSDF));
            let wi_l = reflect(-wo_l, h);
            // A visible normal can still reflect below the horizon. Dropping
            // that sample is the single-scatter shadowing term; what it costs
            // is the multiple-scattering energy the furnace test measures.
            if (wi_l.z <= 0.0) { return s; }

            let direction = onb_local(b.frame, wi_l);
            if (dot(b.ng, direction) <= 0.0) { return s; }

            let lambda_o = smith_lambda(wo_l.z, b.alpha);
            let lambda_i = smith_lambda(wi_l.z, b.alpha);

            s.wi = normalize(direction);
            // The VNDF cancellation: f * cos / pdf collapses to F * G2 / G1(wo),
            // with D, the 4 cos_o cos_i and G1 all gone.
            s.weight = fresnel_schlick(b.base_color, dot(wo_l, h))
                * ((1.0 + lambda_o) / (1.0 + lambda_o + lambda_i))
                * ggx_multiscatter(b.base_color, wo_l.z, b.alpha);
            s.pdf = ggx_d(h.z, b.alpha) / ((1.0 + lambda_o) * 4.0 * wo_l.z);
            s.specular = false;
            s.valid = true;
        }
    } else if (has_dielectrics && b.kind == BSDF_DIELECTRIC) {
        // Rough first, so the flag sits in a condition naga folds; see the
        // note in `bsdf_is_specular`.
        if (has_rough_dielectrics && b.alpha >= GGX_ALPHA_MIN) {
            // Sample in the frame where wo is up, then flip the result back.
            // Unlike the conductor there is no hemisphere to reject against --
            // half the lobe is supposed to end up behind the surface -- so a
            // shading normal the view ray is behind is a frame to turn over
            // rather than a sample to drop.
            let cos_no = dot(wo, b.frame.w);
            if (abs(cos_no) < 1e-6) { return s; }
            let flip = select(-1.0, 1.0, cos_no > 0.0);
            let wo_f = onb_from_world(b.frame, wo * flip);

            let h = sample_ggx_vndf(wo_f, b.alpha, sampler_2d(smp, bounce_pair + PAIR_BSDF));
            let cos_oh = dot(wo_f, h);
            if (cos_oh <= 0.0) { return s; }

            let f = fresnel_dielectric(cos_oh, b.ior);

            // Its own slot rather than the bounce's shared scalar: a rough
            // dielectric takes a shadow ray, so `dim_x(scalars)` is already the
            // light pick at that vertex.
            let coin = sampler_1d(smp, dim_x(PAIR_GLASS_COIN_BASE + bounce_pair));

            var wi_f: vec3<f32>;
            var is_reflect = true;
            if (coin < f) {
                wi_f = reflect(-wo_f, h);
                if (wi_f.z <= 0.0) { return s; }
            } else {
                wi_f = refract(-wo_f, h, b.ior);
                is_reflect = false;
                // `fresnel_dielectric` is 1 past the critical angle, so this
                // branch is unreachable there. The test guards a half vector
                // that lands just inside it.
                if (wi_f.z >= 0.0) { return s; }
            }

            let direction = onb_local(b.frame, wi_f * flip);
            // Sided, as in bsdf_eval and for the same reason.
            if (is_reflect != (dot(b.ng, direction) > 0.0)) { return s; }

            let lambda_o = smith_lambda(wo_f.z, b.alpha);
            let lambda_i = smith_lambda(wi_f.z, b.alpha);

            s.wi = normalize(direction);
            // The VNDF cancellation, the conductor's minus the Fresnel term,
            // which cancels against the chance of having picked this half of
            // the lobe. Both halves collapse to the same G2 / G1.
            s.weight = vec3<f32>(
                dielectric_multiscatter(b.energy_offset, wo_f.z, b.alpha)
                    * (1.0 + lambda_o) / (1.0 + lambda_o + lambda_i)
            );
            let d_vis = ggx_d(h.z, b.alpha) * cos_oh / ((1.0 + lambda_o) * wo_f.z);
            if (is_reflect) {
                s.pdf = f * d_vis / (4.0 * cos_oh);
            } else {
                s.pdf = (1.0 - f) * d_vis
                    * dielectric_dwh_dwi(dot(wi_f, h), cos_oh, 1.0 / b.ior);
            }
            s.specular = false;
            s.valid = true;
        } else {
            // Smooth glass: the Dirac path.
            let cos_theta = min(dot(wo, b.frame.w), 1.0);
            let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

            var direction: vec3<f32>;
            // Total internal reflection never makes the draw; the slot is
            // fixed, so nothing after it moves.
            if (b.ior * sin_theta > 1.0
                || fresnel_dielectric(cos_theta, b.ior) > sampler_1d(smp, dim_x(scalars))) {
                direction = reflect(-wo, b.frame.w);
            } else {
                direction = refract(-wo, b.frame.w, b.ior);
            }

            s.wi = normalize(direction);
            // The interface tints nothing: a dielectric's albedo is interior
            // absorption, applied per unit travelled by the transport loop.
            s.weight = vec3<f32>(1.0);
            s.valid = true;
        }
    }

    return s;
}

// ---------------------------------------------------------------------------
// Intersection
//
// Traversal only needs the distance and (for triangles/quads) the surface
// parameters. UVs, tangent frames and the sphere's acos/atan2 mapping are
// derived once, in resolve_hit, from the winning primitive.
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

// Reciprocal of the ray direction, computed once per ray. Exact zeros become a
// tiny magnitude so the slab test cannot produce 0 * inf -> NaN on an
// axis-parallel ray; either sign of the substitute gives the right answer.
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

// One entry of a leaf's primitive list, decoded.
struct PrimRef {
    prim_type: u32,
    prim_idx: u32,
}

// Reads leaf slot `slot`.
//
// On an all-triangle scene `prim_refs` is the identity -- `flatten_scene` walks
// the leaf order handing out per-type indices, so slot k holds triangle k -- so
// the geometry's address is known from the slot alone. That takes one of the
// two dependent global loads out of the innermost loop of a latency-bound
// tracer, which is worth more than the arithmetic the other flags save.
fn prim_ref_at(slot: u32) -> PrimRef {
    if (identity_prim_refs) {
        return PrimRef(PRIM_TYPE_TRIANGLE, slot);
    }
    let prim_ref = prim_refs[slot];
    return PrimRef(prim_ref >> PRIM_TYPE_SHIFT, prim_ref & PRIM_INDEX_MASK);
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
        let p = prim_ref_at(offset + i);

        var t_hit = 0.0;
        var bary = vec2<f32>(0.0);
        var hit = false;
        if (has_spheres && p.prim_type == PRIM_TYPE_SPHERE) {
            hit = hit_sphere_t(r, spheres[p.prim_idx], t_min, *closest_so_far, &t_hit);
        } else if (has_quads && p.prim_type == PRIM_TYPE_QUAD) {
            hit = hit_quad_t(r, quad_pos[p.prim_idx], t_min, *closest_so_far, &t_hit, &bary);
        } else {
            hit = hit_triangle_t(r, triangle_pos[p.prim_idx], t_min, *closest_so_far, &t_hit, &bary);
        }

        if (hit) {
            hit_any = true;
            *closest_so_far = t_hit;
            (*hit_ref).t = t_hit;
            (*hit_ref).prim_type = p.prim_type;
            (*hit_ref).prim_idx = p.prim_idx;
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
                traversal_stack[stack_ptr] = right_next;
                stack_ptr++;
                node_idx = left_next;
            } else {
                traversal_stack[stack_ptr] = left_next;
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
            node_idx = traversal_stack[stack_ptr];
        }
    }

    return hit_anything;
}

// Does any primitive of this leaf block the segment?
fn leaf_occluded(r: Ray, leaf: u32, t_max: f32) -> bool {
    let count = (leaf >> LEAF_COUNT_SHIFT) & LEAF_COUNT_MASK;
    let offset = leaf & LEAF_OFFSET_MASK;

    for (var i = 0u; i < count; i++) {
        let p = prim_ref_at(offset + i);

        var t_hit = 0.0;
        var bary = vec2<f32>(0.0);
        if (has_spheres && p.prim_type == PRIM_TYPE_SPHERE) {
            if (hit_sphere_t(r, spheres[p.prim_idx], RAY_EPS, t_max, &t_hit)) { return true; }
        } else if (has_quads && p.prim_type == PRIM_TYPE_QUAD) {
            if (hit_quad_t(r, quad_pos[p.prim_idx], RAY_EPS, t_max, &t_hit, &bary)) { return true; }
        } else {
            if (hit_triangle_t(r, triangle_pos[p.prim_idx], RAY_EPS, t_max, &t_hit, &bary)) { return true; }
        }
    }
    return false;
}

// Any-hit traversal for shadow rays: returns on the first blocker, resolves no
// shading attributes, and needs no front-to-back ordering. With NEE roughly
// half of all rays are shadow rays.
fn occluded(origin: vec3<f32>, direction: vec3<f32>, t_max: f32) -> bool {
    if (arrayLength(&nodes) == 0u || t_max <= RAY_EPS) { return false; }

    let inv_dir = ray_inv_dir(direction);
    let r = Ray(origin, direction);

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
                traversal_stack[stack_ptr] = right_next;
                stack_ptr++;
            }
            node_idx = left_next;
        } else if (right_next != NO_NODE) {
            node_idx = right_next;
        } else {
            if (stack_ptr == 0u) { break; }
            stack_ptr--;
            node_idx = traversal_stack[stack_ptr];
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

    if (has_spheres && hit_ref.prim_type == PRIM_TYPE_SPHERE) {
        let s = spheres[hit_ref.prim_idx];
        let center = s.center_and_radius.xyz;
        let radius = s.center_and_radius.w;
        let outward_normal = (rec.p - center) / radius;

        rec.front_face = dot(r.direction, outward_normal) < 0.0;
        rec.normal = select(-outward_normal, outward_normal, rec.front_face);
        // A sphere has no facets, so shading and geometry agree everywhere.
        rec.geometric_normal = rec.normal;
        rec.material_index = s.material_index;

        let theta = acos(-outward_normal.y);
        let phi = atan2(-outward_normal.z, outward_normal.x) + 3.14159265359;
        rec.uv = vec2<f32>(phi / (2.0 * 3.14159265359), theta / 3.14159265359);

        // onb_from_w's guard, adapted to a default axis of (0,1,0). A fixed
        // axis would be normalize(0) at the poles, which is exactly where the
        // spherical UV mapping puts its singularity.
        var a = vec3<f32>(0.0, 1.0, 0.0);
        if (abs(outward_normal.y) > 0.9) {
            a = vec3<f32>(1.0, 0.0, 0.0);
        }
        rec.tangent = normalize(cross(a, outward_normal));
        rec.bi_tangent = cross(outward_normal, rec.tangent);
    } else if (has_quads && hit_ref.prim_type == PRIM_TYPE_QUAD) {
        let attr = quad_attr[hit_ref.prim_idx];
        let normal = quad_pos[hit_ref.prim_idx].normal;

        rec.front_face = dot(r.direction, normal) < 0.0;
        rec.normal = select(-normal, normal, rec.front_face);
        // A quad is planar, so shading and geometry agree everywhere.
        rec.geometric_normal = rec.normal;
        rec.material_index = attr.material_index;
        rec.uv = hit_ref.bary;
        rec.tangent = attr.tangent;
        rec.bi_tangent = attr.bi_tangent;
    } else {
        let attr = triangle_attr[hit_ref.prim_idx];
        let u = hit_ref.bary.x;
        let v = hit_ref.bary.y;
        let w = 1.0 - u - v;

        // A flat triangle stores the same normal at all three corners, so this
        // returns it unchanged.
        let shading_normal = normalize(
            w * oct_decode(unpack2x16snorm(attr.n0_oct))
            + u * oct_decode(unpack2x16snorm(attr.n1_oct))
            + v * oct_decode(unpack2x16snorm(attr.n2_oct))
        );

        // Facing keyed off the geometric normal: the interpolated one makes a
        // closed mesh report both faces along a silhouette edge.
        rec.front_face = dot(r.direction, attr.normal) < 0.0;
        rec.geometric_normal = select(-attr.normal, attr.normal, rec.front_face);
        rec.normal = select(-shading_normal, shading_normal, rec.front_face);
        rec.material_index = attr.material_index;
        rec.uv = w * attr.uv0 + u * attr.uv1 + v * attr.uv2;
        rec.tangent = attr.tangent;
        rec.bi_tangent = attr.bi_tangent;
    }

    return rec;
}

// Routes one radiance contribution to the direct or the indirect accumulator.
// Depth 0 is what the camera sees without an intervening bounce -- emission,
// the shadow ray cast from the visible surface, the background -- none of which
// is a firefly. Everything deeper goes through `clamping_threshold`.
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

// Ceiling on specular bounces the guide ray will follow: past this it is
// describing a hall of mirrors, and every bounce is another ray per pixel.
const GUIDE_MAX_SPECULAR = 6u;

// Traces the denoiser's guide ray: what does this pixel actually look at?
//
// Not the primary hit. On a mirror or a glass surface that describes the
// surface rather than the image in or through it, so every tap across the
// mirror looks alike to the edge stop and the reflection smears sideways.
// Following the specular chain to the first surface that scatters gives the
// filter something to hold on to.
//
// Separate from trace_sample rather than gathered from one of its samples,
// because a sampled chain is stochastic: neighbouring pixels on a glass sphere
// would record unrelated guides and the edge stop would reject nearly every
// tap. Everything below is deterministic and shared with its neighbours -- the
// pixel centre, no lens offset, no fuzz, the dominant Fresnel branch.
//
// Costs one ray per pixel on a restart dispatch only. See `writes_guide`.
fn trace_guide(pixel: vec2<u32>) -> GuideSample {
    var out: GuideSample;
    out.specular_depth = 0u;

    let u = (f32(pixel.x) + 0.5) / f32(width);
    let v = 1.0 - (f32(pixel.y) + 0.5) / f32(height);
    // Normalised, unlike the primary ray in trace_sample, so rec.t is in world
    // units at every segment and the lengths below can be summed.
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

        // The same absorption trace_sample applies, or the edge stop sees a
        // clear glass where the samples see a green one.
        if (has_dielectrics && surface.mat_type == MAT_DIELECTRIC && !rec.front_face) {
            tint *= dielectric_transmittance(surface.albedo, rec.t);
        }

        // A rough metal or a rough glass ends the chain: what it passes on is
        // not a sharp image, so following it would make neighbouring pixels
        // record unrelated guides. One test for both, since `fuzz` carries the
        // roughness of either.
        let refractive_or_reflective = (has_metal && surface.mat_type == MAT_METAL)
            || (has_dielectrics && surface.mat_type == MAT_DIELECTRIC);
        let specular = refractive_or_reflective && surface.fuzz < FUZZ_SPECULAR_THRESHOLD;
        if (!specular || bounce == GUIDE_MAX_SPECULAR) {
            // The first surface that scatters, or whatever the chain stalled
            // on once the budget is spent.
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
        if (has_metal && surface.mat_type == MAT_METAL) {
            // Only a near-mirror metal gets here, so the mirror direction is
            // the whole lobe.
            direction = reflect(unit_direction, surface.normal);
            tint *= surface.albedo;
        } else {
            var refraction_ratio = surface.refraction_index;
            if (rec.front_face) {
                refraction_ratio = 1.0 / surface.refraction_index;
            }
            let cos_theta = min(dot(-unit_direction, surface.normal), 1.0);
            let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
            // The branch trace_sample's coin flip would take more often than
            // not: refraction everywhere but TIR and the grazing rim.
            if (refraction_ratio * sin_theta > 1.0
                || fresnel_dielectric(cos_theta, refraction_ratio) > 0.5) {
                direction = reflect(unit_direction, surface.normal);
            } else {
                direction = refract(unit_direction, surface.normal, refraction_ratio);
            }
        }

        out.specular_depth += 1u;
        r = Ray(rec.p, normalize(direction));
    }

    // Unreachable -- the loop always returns -- but WGSL wants a return here.
    return out;
}

// Traces one path for the given pixel and sample index.
//
// Next-event estimation with multiple importance sampling: at every
// non-specular vertex the direct lighting is estimated with an explicit shadow
// ray, and the BSDF-sampled continuation is weighted so a path that lands on a
// light is not counted twice. Both strategies use the balance heuristic, whose
// `w / pdf` collapses to `1 / (pdf_light + pdf_bsdf)`.
//
// A Dirac lobe has no light-sampling counterpart, so it skips NEE and the
// emitter it reaches is taken at full weight. Which lobes those are is the BSDF
// layer's business: everything below dispatches on `Bsdf`, never on `mat_type`,
// except the emitter test and the segment's absorption.
fn trace_sample(pixel: vec2<u32>, sample_index: u32) -> vec3<f32> {
    let index = pixel.y * width + pixel.x;
    let smp = sampler_new(index, sample_index);

    let jitter = sampler_2d(smp, PAIR_PIXEL_JITTER);
    let u = (f32(pixel.x) + jitter.x) / f32(width);
    let v = 1.0 - (f32(pixel.y) + jitter.y) / f32(height);

    var offset = vec3<f32>(0.0);
    if (camera.lens_radius > 0.0) {
        let rd = random_in_unit_disk(sampler_2d(smp, PAIR_LENS)) * camera.lens_radius;
        offset = camera.u * rd.x + camera.v * rd.y;
    }

    let ray_direction = camera.lower_left_corner + u * camera.horizontal + v * camera.vertical - camera.origin - offset;
    var r = Ray(camera.origin + offset, ray_direction);

    var direct = vec3<f32>(0.0);
    var indirect = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    var path_length = 0.0;

    // How the current ray was generated, needed to weight an emitter it may
    // land on. The camera ray counts as specular, so a directly visible light
    // is seen at full brightness.
    var prev_specular = true;
    var prev_bsdf_pdf = 0.0;

    for (var depth = 0u; depth < config.max_depth; depth++) {
        // This bounce's three pairs, at fixed slots.
        let bounce_pair = PAIR_BOUNCE_BASE + PAIRS_PER_BOUNCE * depth;
        let scalars = bounce_pair + PAIR_SCALARS;

        var hit_ref: HitRef;
        if (!world_hit(r, RAY_EPS, 10000.0, &hit_ref)) {
            add_contribution(&direct, &indirect, depth, config.background_color * throughput);
            break;
        }

        let rec = resolve_hit(r, hit_ref);
        let surface = resolve_surface(rec, smp, depth);
        path_length += rec.t;

        // Absorption inside a dielectric: a property of the segment just
        // travelled rather than of any lobe.
        if (has_dielectrics && surface.mat_type == MAT_DIELECTRIC && !rec.front_face) {
            throughput *= dielectric_transmittance(surface.albedo, rec.t);
        }

        if (surface.mat_type == MAT_DIFFUSE_LIGHT) {
            if (rec.front_face) {
                var emitted = surface.emission;
                if (surface.attenuation_factor > 0.0) {
                    emitted *= 1.0 / (1.0 + surface.attenuation_factor * path_length);
                }

                // Weight against the direct-lighting strategy that could also
                // have produced this direction, if there is one.
                var weight = 1.0;
                if (nee_enabled && !prev_specular) {
                    // Geometric normal: the density being weighed against is
                    // an area measure on the facet.
                    let pdf_light = light_hit_pdf(
                        hit_ref.prim_type,
                        hit_ref.prim_idx,
                        r.origin,
                        r.direction,
                        rec.t,
                        rec.geometric_normal,
                    );
                    weight = prev_bsdf_pdf / (prev_bsdf_pdf + pdf_light);
                }
                add_contribution(&direct, &indirect, depth, throughput * emitted * weight);
            }
            break;
        }

        let b = bsdf_from_surface(surface, rec.front_face);
        if (b.kind == BSDF_NONE) { break; }
        let wo = -normalize(r.direction);

        // --- Direct lighting (next-event estimation) ---
        //
        // Gated on the lobe, not the material: anything with a density a light
        // sample can land on gets a shadow ray.
        if (nee_enabled && !bsdf_is_specular(b)) {
            let ls = sample_light(
                rec.p,
                sampler_1d(smp, dim_x(scalars)),
                sampler_2d(smp, bounce_pair + PAIR_LIGHT_POINT),
            );
            if (ls.valid) {
                let e = bsdf_eval(b, wo, ls.direction);
                if (e.pdf > 0.0) {
                    // Shadow ray last: everything above is cheaper to reject on.
                    if (ls.pdf > 0.0 && !occluded(rec.p, ls.direction, ls.distance - RAY_EPS)) {
                        var emitted = ls.emission;
                        if (ls.attenuation_factor > 0.0) {
                            emitted *= 1.0 / (1.0 + ls.attenuation_factor * (path_length + ls.distance));
                        }
                        // The balance heuristic's w / pdf, collapsed.
                        add_contribution(
                            &direct,
                            &indirect,
                            depth,
                            throughput * e.f_cos * emitted / (ls.pdf + e.pdf),
                        );
                    }
                }
            }
        }

        // --- BSDF continuation ---
        let s = bsdf_sample(b, wo, smp, bounce_pair);
        if (!s.valid) { break; }

        throughput *= s.weight;
        prev_bsdf_pdf = s.pdf;
        prev_specular = s.specular;
        r = Ray(rec.p, s.wi);

        let max_throughput = max(throughput.x, max(throughput.y, throughput.z));
        if (max_throughput < 0.0001) {
            break;
        }

        // Russian roulette: terminate dim paths early and scale the survivors
        // up, which keeps the estimator unbiased while cutting path length.
        if (depth >= RR_MIN_DEPTH) {
            let survival = clamp(max_throughput, RR_MIN_SURVIVAL, 1.0);
            if (sampler_1d(smp, dim_y(scalars)) > survival) {
                break;
            }
            throughput /= survival;
        }
    }

    // Firefly clamp, per sample, on the indirect term only.
    return direct + min(indirect, vec3<f32>(clamping_threshold));
}

// Floor on the denominator of the relative variance check, so a near-black
// pixel's tiny absolute noise does not keep it sampling forever.
const ADAPTIVE_LUMINANCE_FLOOR = 1e-4;

// 8x8 tiles rather than 64 pixels of a scanline, so neighbouring rays in a
// workgroup stay coherent through the first bounce or two.
@compute @workgroup_size(8, 8)
fn compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= width || global_id.y >= height) {
        return;
    }
    let index = global_id.y * width + global_id.x;
    let pixel = global_id.xy;

    // A restart (accumulation reset on camera/depth change) discards whatever
    // the buffers held for a previous, unrelated accumulation.
    let restart = config.sample_count == 0u;

    // One guide ray per accumulation run, before the adaptive early-out below
    // can return, so the guide can never go stale: a camera change restarts the
    // accumulation and rewrites the guide in the same dispatch.
    //
    // `writes_guide` leads the condition rather than sitting in the body, so
    // naga can delete the arm; see the note in bsdf_is_specular.
    if (writes_guide && restart) {
        gbuffer[index] = pack_guide(trace_guide(pixel));
    }

    let n0 = select(sample_count_buffer[index], 0u, restart);
    let prev = select(output_buffer[index], vec4<f32>(0.0), restart);
    let prev_mean = prev.xyz;
    // Sum of squared deviations from the running luminance mean; see the merge
    // at the bottom.
    let prev_m2 = prev.w;

    // Skip pixels that have already converged: no trace_sample, no BVH
    // traversal, no shadow rays for this dispatch. `min_samples_per_pixel` is
    // floored at 2 so the variance below is never evaluated against a sample
    // count that cannot support it.
    let min_samples = max(config.min_samples_per_pixel, 2u);
    if (n0 >= min_samples) {
        let mean_luminance = luminance(prev_mean);
        let variance = prev_m2 / f32(n0 - 1u);

        // The variance is itself estimated from n0 samples, with a relative
        // standard deviation of sqrt(2/(n-1)) -- 36% at n = 16. Testing the raw
        // point estimate lets a pixel that drew an unlucky run of similar
        // samples freeze permanently, since a pixel that stops sampling can
        // never revise the numbers that silenced it.
        //
        // Simulated on a pixel 11% above the threshold, 42% of runs froze early
        // at n = 16 on the point estimate against 13% on this bound. By n = 64
        // the bound changes almost nothing, and what it costs is extra samples
        // on pixels that had converged -- the right way round for an artifact
        // that never averages out.
        //
        // Hence min_samples_per_pixel wanting to be a few dozen: it sets the
        // precision of the estimate being tested.
        let estimator_uncertainty = sqrt(2.0 / f32(n0 - 1u));
        let variance_bound = variance * (1.0 + estimator_uncertainty);

        let standard_error = sqrt(variance_bound / f32(n0));
        if (standard_error <= config.variance_threshold * max(mean_luminance, ADAPTIVE_LUMINANCE_FLOOR)) {
            return;
        }
    }

    // Several samples per dispatch, accumulated in registers, so the buffers
    // are read and written once per batch rather than once per sample.
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
    // Welford rather than a raw sum of squares with variance recovered as
    // E[L^2] - E[L]^2: that difference can go negative for a genuinely
    // low-variance pixel, and clamping it to zero hands the test above a
    // standard error of exactly zero, which always passes and freezes the pixel
    // permanently. M2 never subtracts two large numbers.
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
