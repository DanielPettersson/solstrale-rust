// Edge-avoiding A-Trous wavelet filter (Dammertz et al. 2010), with the tap
// weights guided by the per-pixel variance the sample loop already tracks
// (Schied et al. 2017, "Spatiotemporal Variance-Guided Filtering").
//
// Spatial only. The temporal accumulation SVGF needs is what solstrale's sample
// loop already is -- and it is exact rather than reprojected, so there is
// nothing a temporal stage could add.
//
// Two entry points share this module so that the guide decoding below is
// written once: `prefilter_variance` is SVGF's 3x3 variance pre-pass, `compute`
// is one A-Trous iteration.

override width: u32 = 1u;
override height: u32 = 1u;
// Tap spacing for this iteration: 1, 2, 4, 8, 16. One pipeline per value, so the
// spacing is a compile-time constant and the address arithmetic folds.
override step_width: i32 = 1;

// Edge-stopping falloffs. Only sigma_colour is exposed on the public API (as
// `strength`); the rest are the paper's values, kept here so a future tuning
// pass has one place to look.
override sigma_colour: f32 = 4.0;
override sigma_normal: f32 = 128.0;
override sigma_depth: f32 = 1.0;
override sigma_albedo: f32 = 0.1;
// Set to 0 by DenoiseGuide::ColorOnly, which drops the three guide weights and
// leaves the filter running on colour and variance alone.
override use_guide: f32 = 1.0;

// xyz: colour, w: variance of the colour estimate.
@group(0) @binding(0)
var<storage, read> src: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read_write> dst: array<vec4<f32>>;

// Packed primary-hit guide. The oct_decode below is the inverse of oct_encode in
// renderer/ray_trace.wgsl and must stay in step with it; WGSL has no include
// mechanism, so the pair is deliberately duplicated rather than shared.
@group(0) @binding(2)
var<storage, read> gbuffer: array<vec4<u32>>;

struct Guide {
    albedo: vec3<f32>,
    normal: vec3<f32>,
    depth: f32,
}

fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    var v = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) {
        let s = vec2<f32>(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
        v = vec3<f32>((1.0 - abs(vec2<f32>(v.y, v.x))) * s, v.z);
    }
    return normalize(v);
}

fn load_guide(index: u32) -> Guide {
    let g = gbuffer[index];
    var out: Guide;
    out.albedo = unpack4x8unorm(g.x).rgb;
    out.normal = oct_decode(unpack2x16float(g.y));
    out.depth = bitcast<f32>(g.z);
    return out;
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// How much a neighbouring pixel looks like it belongs to the same surface,
// judged on geometry and material rather than on the noisy radiance.
fn guide_weight(centre: Guide, tap: Guide, spacing: f32) -> f32 {
    // Cosine lobe, sharpened. Rejects a neighbour facing a different way -- the
    // inside of a corner, where the colour weight alone would happily average
    // two walls together.
    let w_normal = pow(max(dot(centre.normal, tap.normal), 0.0), sigma_normal);

    // Relative rather than absolute, so one sigma works at any scene scale and
    // the background's far sentinel compares sanely against itself. Scaled by
    // the tap spacing, so the test stays comparable as the kernel widens.
    let w_depth = exp(-abs(centre.depth - tap.depth)
                      / (sigma_depth * spacing * max(abs(centre.depth), 1e-4)));

    // Not in SVGF, which demodulates albedo out of the signal and remodulates it
    // afterwards. We filter full radiance -- there is no separate direct and
    // indirect to demodulate against -- so albedo enters as a weight instead.
    // This is what keeps texture detail from being smeared into flat colour.
    let w_albedo = exp(-length(centre.albedo - tap.albedo) / sigma_albedo);

    return mix(1.0, w_normal * w_depth * w_albedo, use_guide);
}

// SVGF's 3x3 variance pre-pass. Without it the single-pixel spikes in the raw
// variance make the colour tolerance flicker from pixel to pixel, which reads as
// a shimmer the eye picks up immediately. Colour passes through untouched.
//
// Deliberately no colour weight here: using a variance-driven weight to filter
// the variance itself would be circular.
@compute @workgroup_size(8, 8)
fn prefilter_variance(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let index = gid.y * width + gid.x;

    // Function-scope var rather than a module-scope const: naga will not always
    // accept a runtime index into a const array, and the failure is opaque.
    var g = array<f32, 3>(0.25, 0.5, 0.25);

    let centre_guide = load_guide(index);

    var sum_variance = 0.0;
    var sum_weight = 0.0;

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let x = clamp(i32(gid.x) + dx, 0, i32(width) - 1);
            let y = clamp(i32(gid.y) + dy, 0, i32(height) - 1);
            let n_index = u32(y) * width + u32(x);

            let weight = g[dx + 1] * g[dy + 1]
                * guide_weight(centre_guide, load_guide(n_index), 1.0);

            sum_variance += src[n_index].w * weight;
            sum_weight += weight;
        }
    }

    dst[index] = vec4<f32>(src[index].xyz, sum_variance / sum_weight);
}

@compute @workgroup_size(8, 8)
fn compute(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let index = gid.y * width + gid.x;

    // 5x5 B-spline, the outer product of (1,4,6,4,1)/16.
    var h = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);

    let centre = src[index];
    let centre_lum = luminance(centre.xyz);
    let centre_guide = load_guide(index);

    // How far a neighbour's luminance may stray before it reads as a different
    // surface rather than as noise. Driven by the pre-filtered variance of the
    // estimate, so a pixel that has converged tolerates almost nothing and a
    // pixel at one sample tolerates almost anything. This is the whole of the
    // "variance-guided" part.
    let lum_tolerance = sigma_colour * sqrt(max(centre.w, 1e-8)) + 1e-8;

    var sum = vec3<f32>(0.0);
    var sum_variance = 0.0;
    var sum_weight = 0.0;

    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let x = clamp(i32(gid.x) + dx * step_width, 0, i32(width) - 1);
            let y = clamp(i32(gid.y) + dy * step_width, 0, i32(height) - 1);
            let n_index = u32(y) * width + u32(x);

            let tap = src[n_index];
            let w_colour = exp(-abs(centre_lum - luminance(tap.xyz)) / lum_tolerance);

            let weight = h[dx + 2] * h[dy + 2] * w_colour
                * guide_weight(centre_guide, load_guide(n_index), f32(step_width));

            sum += tap.xyz * weight;
            // Variance is a second moment, so it filters with the *squared*
            // weights. That is what shrinks lum_tolerance from one iteration to
            // the next on its own, with no explicit sigma schedule -- and why
            // sigma_colour is deliberately not divided by 2^i here.
            sum_variance += tap.w * weight * weight;
            sum_weight += weight;
        }
    }

    // sum_weight can never reach zero: the centre tap contributes h*h = 0.140625
    // with every edge-stopping term identically one against itself.
    dst[index] = vec4<f32>(sum / sum_weight, sum_variance / (sum_weight * sum_weight));
}
