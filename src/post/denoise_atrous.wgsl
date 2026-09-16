// Edge-avoiding A-Trous wavelet filter (Dammertz et al. 2010), with the tap
// weights guided by the per-pixel variance the sample loop already tracks
// (Schied et al. 2017, "Spatiotemporal Variance-Guided Filtering").
//
// Spatial only. The temporal accumulation SVGF needs is what solstrale's sample
// loop already is -- and it is exact rather than reprojected, so there is
// nothing a temporal stage could add.
//
// Two entry points share this module so that the guide decoding below is
// written once: `prefilter_variance` is SVGF's variance pre-pass, `compute` is
// one A-Trous iteration.

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
override sigma_specular: f32 = 1.0;
// Set to 0 by DenoiseGuide::ColorOnly, which drops the four guide weights and
// leaves the filter running on colour and variance alone.
override use_guide: f32 = 1.0;
// Radius of the variance pre-pass. See prefilter_variance for why it is 2 and
// not SVGF's 1.
override prefilter_radius: i32 = 2;

// xyz: colour, w: variance of the colour estimate.
@group(0) @binding(0)
var<storage, read> src: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read_write> dst: array<vec4<f32>>;

// Packed guide describing the first non-specular surface the pixel looks at.
// The oct_decode below is the inverse of oct_encode in
// renderer/ray_trace.wgsl and must stay in step with it; WGSL has no include
// mechanism, so the pair is deliberately duplicated rather than shared.
@group(0) @binding(2)
var<storage, read> gbuffer: array<vec4<u32>>;

// Written by prefilter_variance only, and read by denoise_resolve.wgsl, which
// needs the same pooled variance this pass hands the iterations below. The
// copy in `dst.w` cannot serve: the iterations filter it along with the colour,
// and by the last one it describes the kernel rather than the pixel.
@group(0) @binding(3)
var<storage, read_write> pooled_variance: array<f32>;

struct Guide {
    albedo: vec3<f32>,
    normal: vec3<f32>,
    depth: f32,
    specular_depth: f32,
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
    // Low byte is the material type, which nothing here reads yet; the rest is
    // the number of specular bounces the guide ray took. See pack_guide in
    // renderer/ray_trace.wgsl.
    out.specular_depth = f32(g.w >> 8u);
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

    // A wall seen directly and the same wall seen in a mirror can agree on all
    // three channels above by coincidence -- same material, same orientation,
    // and a path length that happens to match. Separating them on how many
    // specular bounces it took to reach them is the one test that cannot be
    // fooled that way. Soft rather than a hard gate, at one e-fold per bounce,
    // so a mirror's silhouette does not turn into a segmentation edge the
    // filter refuses to cross at all.
    let w_specular = exp(-abs(centre.specular_depth - tap.specular_depth) / sigma_specular);

    return mix(1.0, w_normal * w_depth * w_albedo * w_specular, use_guide);
}

// SVGF's variance pre-pass. Without it the single-pixel spikes in the raw
// variance make the colour tolerance flicker from pixel to pixel, which reads as
// a shimmer the eye picks up immediately. Colour passes through untouched.
//
// It is also what makes the variance usable at all at a handful of samples per
// pixel, which is the case the filter exists for. A pixel's own M2 carries
// n - 1 degrees of freedom, so at n == 2 the variance it implies is chi-squared
// with one degree of freedom: its median sits at 45% of the truth and a quarter
// of all pixels land below a tenth of it. A pixel that lands there declares
// itself converged while holding pure noise -- and in a path tracer that is not
// an unlucky accident but the ordinary case, because the two samples of a pixel
// that both missed the light agree closely.
//
// Pooling over neighbours that share a surface multiplies the degrees of
// freedom by the kernel's effective tap count -- the inverse of its summed
// squared weights, 12.1 at radius 2 against 7.1 for SVGF's binomial 3x3 -- and
// costs no samples. Measured on the test scene at 2 spp, widening the kernel
// this far cut linear RMSE against a converged reference by 10%; radius 3
// bought another 3% for twice the taps.
//
// Deliberately no colour weight here: using a variance-driven weight to filter
// the variance itself would be circular.
@compute @workgroup_size(8, 8)
fn prefilter_variance(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let index = gid.y * width + gid.x;

    let centre_guide = load_guide(index);

    var sum_variance = 0.0;
    var sum_weight = 0.0;

    // Half the radius, so the kernel reaches two standard deviations.
    let sigma = max(f32(prefilter_radius), 1.0) * 0.5;
    for (var dy = -prefilter_radius; dy <= prefilter_radius; dy++) {
        for (var dx = -prefilter_radius; dx <= prefilter_radius; dx++) {
            let x = clamp(i32(gid.x) + dx, 0, i32(width) - 1);
            let y = clamp(i32(gid.y) + dy, 0, i32(height) - 1);
            let n_index = u32(y) * width + u32(x);

            let d2 = f32(dx * dx + dy * dy);
            let weight = exp(-d2 / (2.0 * sigma * sigma))
                * guide_weight(centre_guide, load_guide(n_index), 1.0);

            sum_variance += src[n_index].w * weight;
            sum_weight += weight;
        }
    }

    let variance = sum_variance / sum_weight;
    pooled_variance[index] = variance;
    dst[index] = vec4<f32>(src[index].xyz, variance);
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
