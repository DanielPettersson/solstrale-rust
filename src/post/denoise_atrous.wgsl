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
// Corrects the variance recursion at the bottom of `compute` for this
// iteration's taps not being independent of each other. One per iteration, from
// VARIANCE_CORRELATION in denoise.rs. 1.0 for iteration 0, where the taps
// really are independent, and for the pre-pass, which does not filter.
override variance_correlation: f32 = 1.0;

// Effective tap count of the 5x5 B-spline with every edge stop identically one:
// the reciprocal of its summed squared weights. What `variance_correlation` is
// measured against, so that a kernel the edge stops have narrowed gets a
// proportionally smaller share of it.
const KERNEL_TAPS = 13.374;

// Outlier rejection, applied in prefilter_variance; see the block comment
// there. Chosen on `cornell_firefly_sweep` against
// `test_denoise_improves_specular_image`, the scene that pushes back: a caustic
// is the same shape as a firefly and these four have to tell them apart.
override despeckle_k: f32 = 2.0;
override despeckle_floor: f32 = 1.0;
// Minimum summed guide weight over the 24 non-centre taps before the clamp may
// fire, out of a possible 5.169. The most important of the four: at 1.0 the
// pass judges a pixel against whichever one or two neighbours survived the
// guide weighting, which on a curved mirror or a textured floor is a couple of
// dark taps, and clamps genuinely bright pixels to a tenth of their value. 2.5
// took the specular scene from 4% worse than no despeckle to slightly better,
// and cost no fireflies.
override despeckle_min_weight: f32 = 2.5;
// Relative standard error at which the clamp reaches full strength.
//
// Deliberately a *relative* standard error, because the calibration depends on
// being one: for a mean carried by a single outlier sample the standard error
// and the mean cancel exactly, leaving 0.4026 however bright the firefly is, so
// one threshold catches all of them. Unrelated to `full_strength_error` in
// denoise_resolve.wgsl, which asks a different question.
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
// WGSL has no include mechanism, so the oct_decode below is a duplicate of the
// one in renderer/ray_trace.wgsl and must stay in step with it, as must
// `pack_oct` in renderer/scene_flattener.rs.
@group(0) @binding(2)
var<storage, read> gbuffer: array<vec4<u32>>;

// Written by prefilter_variance, read by denoise_resolve.wgsl. The copy in
// `dst.w` cannot serve: the iterations filter it along with the colour, so by
// the last one it describes the kernel rather than the pixel.
@group(0) @binding(3)
var<storage, read_write> pooled_variance: array<f32>;

// The post-processing chain's working image, which denoise_resolve.wgsl blends
// the filtered result back over. Written by prefilter_variance, and only for
// the pixels whose colour it clamped, so the image the resolve pass calls "the
// original" is the despeckled one -- otherwise it would mix a fraction of every
// removed firefly straight back in. The A-Trous iterations never touch it.
@group(0) @binding(4)
var<storage, read_write> working: array<vec4<f32>>;

// The guide-weighted luminance of each pixel's neighbourhood, centre excluded:
// a measure of how bright a pixel's surroundings are that the pixel's own noise
// cannot move. Written by prefilter_variance, read by denoise_resolve.wgsl.
@group(0) @binding(5)
var<storage, read_write> local_level: array<f32>;

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
    // Low byte is the material type, unread here; the rest is the number of
    // specular bounces. See pack_guide in renderer/ray_trace.wgsl.
    out.specular_depth = f32(g.w >> 8u);
    return out;
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

    // Not in SVGF, which demodulates albedo out of the signal and remodulates
    // it afterwards. Here full radiance is filtered, so albedo enters as a
    // weight instead, which is what keeps texture detail out of flat colour.
    let w_albedo = exp(-length(centre.albedo - tap.albedo) / sigma_albedo);

    // A wall seen directly and the same wall seen in a mirror can agree on all
    // three channels above by coincidence. Specular depth is the one test that
    // cannot be fooled that way. Soft rather than a hard gate, so a mirror's
    // silhouette does not become an edge the filter refuses to cross.
    let w_specular = exp(-abs(centre.specular_depth - tap.specular_depth) / sigma_specular);

    return mix(1.0, w_normal * w_depth * w_albedo * w_specular, use_guide);
}

// SVGF's variance pre-pass, plus outlier rejection. Colour passes through
// untouched except where the despeckle fires.
//
// Without the pooling, single-pixel spikes in the raw variance make the colour
// tolerance flicker from pixel to pixel, which reads as a shimmer. It is also
// what makes the variance usable at a handful of samples per pixel: a pixel's
// own M2 carries n - 1 degrees of freedom, so at n == 2 its median sits at 45%
// of the truth and a quarter of all pixels land below a tenth of it -- and a
// pixel that lands there declares itself converged while holding pure noise.
// Pooling over neighbours that share a surface multiplies the degrees of
// freedom by the kernel's effective tap count (12.1 at radius 2 against 7.1 for
// SVGF's binomial 3x3) and costs no samples. At 2 spp that is 10% of the linear
// RMSE against a converged reference; radius 3 buys another 3% for twice the
// taps.
//
// Deliberately no colour weight: using a variance-driven weight to filter the
// variance itself would be circular. Nothing the despeckle computes feeds back
// into `sum_variance`, so that holds.
//
// The despeckle rides along because it needs exactly the same gather. An
// edge-avoiding filter cannot remove a firefly on its own -- a firefly is an
// edge by every measure the filter has -- and the fade in denoise_resolve.wgsl
// only catches part of it. The extreme tail lands here: for a pixel whose mean
// comes from one outlier sample, the standard error and the mean cancel, so the
// relative error is the constant 0.4026 whatever the firefly's brightness.
// Below two samples the fade does not run at all, so at 1 spp nothing else
// stands between an outlier and the image -- 1701 specks against 357 with this
// pass. Clamped pixels are written into `working` as well, so the resolve pass
// blends against the despeckled original.
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

    // Reliability-weighted variance, Bessel-corrected. The kernel's effective
    // tap count excluding the centre is about eleven, so the correction is
    // worth roughly a tenth and `despeckle_k` is calibrated with it in.
    let n_mean = n_lum / max(n_weight, 1e-8);
    let n_var = max(n_lum_sq / max(n_weight, 1e-8) - n_mean * n_mean, 0.0)
        * (n_weight * n_weight)
        / max(n_weight * n_weight - n_weight_sq, 1e-8);

    // Published for the resolve pass. Where the guide leaves too little
    // neighbourhood to average -- a silhouette, a curved mirror -- there is no
    // honest local level, so hand back the pixel's own luminance.
    local_level[index] = select(centre_lum, n_mean, n_weight >= despeckle_min_weight);

    var colour = centre.xyz;

    // The guide is what makes this safe. It carries no sample noise, so it
    // separates "a bright pixel on the same flat surface as its neighbours",
    // which is a firefly, from "a bright pixel looking at something else",
    // which is detail. A small emitter, a highlight on differently-oriented
    // geometry or a light against the background sentinel all collapse their
    // neighbours' guide weights, and `despeckle_min_weight` then declines to
    // judge the pixel at all. `n_weight` tops out at 5.169.
    if (despeckle_enabled > 0.0 && n_weight >= despeckle_min_weight) {
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
            // Being far above your neighbours does not make a pixel a firefly
            // -- a caustic is too, and so is the lit side of anything small.
            // What separates them is whether the pixel has evidence behind it,
            // so the clamp fades in on a relative standard error.
            //
            // Two departures from the adaptive sampler's version of that test.
            // The denominator is the neighbourhood's level, not the pixel's
            // own, which the outlier inflated -- dividing by that is what lets
            // a firefly declare itself converged in proportion to how bright it
            // is. And the numerator takes whichever of the pixel's own variance
            // and the pooled one is larger: at two samples a quarter of pixels
            // read below a tenth of the truth, and a firefly drawing such a
            // reading would talk its way out of being clamped. Worth a third of
            // the survivors at 2 spp and nothing elsewhere.
            let evidence = sqrt(max(max(centre.w, variance), 0.0))
                / max(n_mean, LUMINANCE_FLOOR);
            let fade = clamp(evidence / despeckle_full_strength_error, 0.0, 1.0);

            // A uniform scale rather than a move toward the neighbourhood, so
            // hue and saturation survive at any fade.
            colour = mix(colour, colour * (bound / centre_lum), fade);

            // Written only where the clamp fired, so a converged image leaves
            // this pass bit-identical -- see
            // test_denoise_is_near_identity_at_high_samples.
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

    // How far a neighbour may stray before it reads as a different surface
    // rather than as noise. Driven by the pre-filtered variance of the estimate,
    // so a converged pixel tolerates almost nothing and a one-sample pixel
    // tolerates almost anything. This is the whole of the "variance-guided"
    // part.
    //
    // Measured and rejected: comparing compressed luminance, log(1 + L). It
    // cost 4% of the specular scene's RMSE and 7% of the diffuse scene's, for
    // fireflies the budget was not under pressure on. See LIMITATIONS.md.
    let lum_tolerance = sigma_colour * sqrt(max(centre.w, 1e-8)) + 1e-8;

    var sum = vec3<f32>(0.0);
    var sum_variance = 0.0;
    var sum_weight = 0.0;
    var sum_weight_sq = 0.0;

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
            // Variance is a second moment, so it filters with the squared
            // weights, which is what shrinks lum_tolerance from one iteration
            // to the next with no explicit sigma schedule.
            //
            // That form assumes independent taps, which only holds on the first
            // iteration: after it the taps already averaged overlapping
            // neighbourhoods of each other. Without `variance_correlation` the
            // tracked variance is about 96 times too small by the fifth
            // iteration and the wide passes are nearly the identity. See
            // VARIANCE_CORRELATION in denoise.rs.
            sum_variance += tap.w * weight * weight;
            sum_weight += weight;
            sum_weight_sq += weight * weight;
        }
    }

    // How much of the kernel survived the edge stops, as a fraction of the
    // 13.374 effective taps it would have with every weight at one.
    //
    // `variance_correlation` is derived for the unweighted kernel, and applying
    // it whole where the edge stops have already cut the kernel down overstates
    // the correction badly -- 18% of the specular scene's RMSE, concentrated on
    // the mirror and the caustic. So it is faded in on how much averaging the
    // pass is actually doing. Both ends are exact: a kernel reduced to its
    // centre tap introduces no correlation, and a kernel with every weight at
    // one is the case the factor was derived for. In between is a first-order
    // interpolation, erring toward under-correction, which costs smoothing
    // rather than detail.
    let overlap = clamp(sum_weight * sum_weight
        / (max(sum_weight_sq, 1e-12) * KERNEL_TAPS), 0.0, 1.0);
    let correlation = 1.0 + (variance_correlation - 1.0) * overlap;

    // sum_weight can never reach zero: the centre tap contributes h*h = 0.140625
    // with every edge-stopping term identically one against itself.
    dst[index] = vec4<f32>(
        sum / sum_weight,
        correlation * sum_variance / (sum_weight * sum_weight),
    );
}
