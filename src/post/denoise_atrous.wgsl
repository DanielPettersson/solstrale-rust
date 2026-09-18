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

// Outlier rejection, applied in prefilter_variance. See the block comment there
// for what it does and why none of these is scaled by `strength`. Chosen on
// `cornell_firefly_sweep` against `test_denoise_improves_specular_image`, which
// is the scene that pushes back: a caustic is the same shape as a firefly and
// these four are what have to tell them apart.
override despeckle_k: f32 = 2.0;
override despeckle_floor: f32 = 1.0;
// Minimum summed guide weight over the 24 non-centre taps before the clamp may
// fire at all, out of a possible 5.169. Much the most important of the four and
// the least obvious. At 1.0 the pass would judge a pixel against whichever one
// or two neighbours happened to survive the guide weighting, which on a curved
// mirror or a textured floor is a couple of dark taps -- and it duly clamped
// genuinely bright pixels to a tenth of their value, 41% of the specular scene
// touched for a 0.5% energy loss concentrated in exactly the wrong places.
// Raising it to 2.5 took that scene from 4% worse than no despeckle at all to
// slightly better, and cost no fireflies.
override despeckle_min_weight: f32 = 2.5;
// Relative standard error at which the clamp reaches full strength. Matches
// `full_strength_error` in denoise_resolve.wgsl deliberately: the two are the
// same test, and the comment there is the one that explains the number.
override despeckle_full_strength_error: f32 = 0.4;
// Set to 0 when strength is 0, which is documented to be the identity.
override despeckle_enabled: f32 = 1.0;

// Floor on the luminance used as the denominator of a relative test, so a
// near-black neighbourhood's tiny absolute noise does not read as enormous
// relative to it. Matches LUMINANCE_FLOOR in denoise_resolve.wgsl and
// ADAPTIVE_LUMINANCE_FLOOR in renderer/ray_trace.wgsl.
const LUMINANCE_FLOOR = 1e-4;

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

// The post-processing chain's working image, which denoise_resolve.wgsl blends
// the filtered result back over. Written by prefilter_variance only, and only
// for the pixels whose colour it clamped, so that the image the resolve pass
// calls "the original" is the despeckled one rather than the raw outlier. The
// A-Trous iterations bind it and never touch it.
//
// Without this the whole despeckle is defeated: the filter would remove a
// firefly and the resolve pass would mix a fraction of it straight back in.
@group(0) @binding(4)
var<storage, read_write> working: array<vec4<f32>>;

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
//
// The pass does one more thing, over the same taps and the same weights:
// outlier rejection. The two belong together because they need exactly the same
// gather -- a guide-weighted neighbourhood -- and the tap colours are already
// in registers by the time the variance has been pooled. Nothing the despeckle
// computes feeds back into `sum_variance`, so the no-colour-weight invariant
// above survives intact.
//
// Why the filter needs it at all. An edge-avoiding filter cannot remove a
// firefly on its own, because a firefly is an edge by every measure the filter
// has. What saved it until now was the fade in denoise_resolve.wgsl -- and that
// is precisely where it leaked. For a pixel whose mean comes from one outlier
// sample out of n, the Welford variance of the mean works out to the pixel's
// own value squared, so the standard error and the mean cancel and the relative
// error the resolve pass tests is a constant: sqrt of this kernel's centre
// share, 1/6.169, which is 0.4026 against a `full_strength_error` of 0.4. Every
// firefly lands on the threshold, the test is scale-free in how bright the
// firefly is, and any ordinary perturbation -- two large samples rather than
// one, or a pixel that also carries real signal -- pushes it under. What is
// left on screen is (1 - blend) times the raw outlier: unbounded in its
// brightness, and independent of `strength`, since the blend reads no sigma at
// all. Measured on the Cornell scene at 2 spp and strength 5, forcing the blend
// to 1 took the firefly count from 220 to 1.
//
// So the fix is not to widen a tolerance. It is to make sure that by the time
// the resolve pass looks at "the original", the outlier is already gone from
// it -- which is what the write to `working` below is for.
@compute @workgroup_size(8, 8)
fn prefilter_variance(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let index = gid.y * width + gid.x;

    let centre = src[index];
    let centre_lum = luminance(centre.xyz);
    let centre_guide = load_guide(index);

    var sum_variance = 0.0;
    var sum_weight = 0.0;

    // The neighbourhood's own luminance, with the centre left out, so the
    // despeckle below compares a pixel against its surroundings rather than
    // against a statistic it is itself part of.
    var n_weight = 0.0;
    var n_weight_sq = 0.0;
    var n_lum = 0.0;
    var n_lum_sq = 0.0;

    // Half the radius, so the kernel reaches two standard deviations.
    let sigma = max(f32(prefilter_radius), 1.0) * 0.5;
    for (var dy = -prefilter_radius; dy <= prefilter_radius; dy++) {
        for (var dx = -prefilter_radius; dx <= prefilter_radius; dx++) {
            let x = clamp(i32(gid.x) + dx, 0, i32(width) - 1);
            let y = clamp(i32(gid.y) + dy, 0, i32(height) - 1);
            let n_index = u32(y) * width + u32(x);

            let d2 = f32(dx * dx + dy * dy);
            let tap = src[n_index];
            let weight = exp(-d2 / (2.0 * sigma * sigma))
                * guide_weight(centre_guide, load_guide(n_index), 1.0);

            sum_variance += tap.w * weight;
            sum_weight += weight;

            if (dx != 0 || dy != 0) {
                let l = luminance(tap.xyz);
                n_weight += weight;
                n_weight_sq += weight * weight;
                n_lum += weight * l;
                n_lum_sq += weight * l * l;
            }
        }
    }

    let variance = sum_variance / sum_weight;
    pooled_variance[index] = variance;

    var colour = centre.xyz;

    // The guide is what makes this safe, and it is not a refinement. It carries
    // no sample noise -- trace_guide fires one deterministic ray per pixel per
    // accumulation run -- so it separates "a bright pixel on the same flat
    // surface as its neighbours", which is a firefly, from "a bright pixel
    // looking at something else entirely", which is detail. A small emitter, a
    // highlight on differently-oriented geometry, or a light against the
    // background sentinel all collapse their neighbours' guide weights, and
    // `despeckle_min_weight` then declines to judge the pixel at all. The
    // maximum `n_weight` can reach is 5.169, the kernel's weight sum less its
    // centre.
    if (despeckle_enabled > 0.0 && n_weight >= despeckle_min_weight) {
        // Reliability-weighted variance, Bessel-corrected. The kernel's
        // effective tap count excluding the centre is about eleven, so the
        // correction is worth roughly a tenth and `despeckle_k` is calibrated
        // with it in. The sum-of-squares form is the one denoise_prepare.wgsl
        // already uses for its spatial fallback, with the same clamp at zero.
        let n_mean = n_lum / n_weight;
        let n_var = max(n_lum_sq / n_weight - n_mean * n_mean, 0.0)
            * (n_weight * n_weight)
            / max(n_weight * n_weight - n_weight_sq, 1e-8);

        // How far above its neighbours a pixel may sit before it reads as an
        // outlier rather than as detail. Both terms are needed:
        //
        // - `despeckle_k` standard deviations covers the neighbourhood's own
        //   spread, so a pixel on a gradient, a texture, or an edge the guide
        //   did not catch is left alone.
        // - the relative floor holds the bound at no less than (1 + floor)
        //   times the local level. That is what stops a uniformly dark and
        //   quiet neighbourhood -- where the spread really is near zero -- from
        //   clamping a pixel down to its neighbours, and it is what carries the
        //   converged case, where the spread has gone to nothing everywhere.
        let bound = n_mean * (1.0 + despeckle_floor) + despeckle_k * sqrt(n_var);

        if (centre_lum > bound && centre_lum > 1e-8) {
            // Being far above your neighbours is not enough to make a pixel a
            // firefly -- a caustic is too, and so is the lit side of anything
            // small. What separates them is whether the pixel has any evidence
            // behind it, and the accumulator already knows: fade the clamp in
            // on the same relative standard error denoise_resolve.wgsl and the
            // adaptive sampler in renderer/ray_trace.wgsl already test.
            //
            // Two departures from those two, both deliberate.
            //
            // The denominator is the *neighbourhood's* level rather than the
            // pixel's own. A firefly's own mean is the thing the outlier
            // inflated, so dividing by it is exactly what lets a firefly
            // declare itself converged in proportion to how bright it is -- the
            // defect described at the top of this function. The neighbourhood
            // has no such conflict of interest.
            //
            // The numerator takes whichever of the pixel's own variance and the
            // pooled one is larger. The pixel's own is the sharper signal where
            // it can be trusted, but at two samples it carries one degree of
            // freedom: a quarter of pixels read below a tenth of the truth, and
            // a firefly that draws such a reading would talk its way out of
            // being clamped. Pooling is what denoise_resolve.wgsl already
            // reaches for against exactly that, and taking the larger keeps the
            // sharper estimate wherever it is not the one that collapsed.
            // Worth a third of the survivors at 2 spp and nothing anywhere
            // else, which is the shape of a fix aimed at the right failure.
            let evidence = sqrt(max(max(centre.w, variance), 0.0))
                / max(n_mean, LUMINANCE_FLOOR);
            let fade = clamp(evidence / despeckle_full_strength_error, 0.0, 1.0);

            // A uniform scale rather than a move toward the neighbourhood, so
            // hue and saturation survive at any fade.
            colour = mix(colour, colour * (bound / centre_lum), fade);

            // Written only where the clamp fired. A converged image therefore
            // leaves this pass bit-identical rather than merely close to it,
            // which is the property test_denoise_is_near_identity_at_high_samples
            // pins. `.w` is carried through untouched, as the resolve pass does.
            working[index] = vec4<f32>(colour, working[index].w);
        }
    }

    dst[index] = vec4<f32>(colour, variance);
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
