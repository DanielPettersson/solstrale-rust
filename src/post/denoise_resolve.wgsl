// Blends the filtered image back over the original, in proportion to how much
// of the remaining noise a viewer would actually see.
//
// The variance guidance inside the a-trous passes shrinks the filter's tolerance
// as a render converges, but not far enough on its own: the tolerance tracks the
// residual noise, and once an image is nearly converged the real detail at pixel
// scale is the same size as that noise. So the filter is faded out, and the
// question is what to fade it against.
//
// It used to be the renderer's own definition of a converged pixel -- the
// relative standard error that adaptive sampling retires pixels on, reaching
// full strength at 0.4. That is the wrong yardstick, for a reason that is only
// visible once it is written in the units the image is finally looked at. The
// readback applies a tone curve and then gamma 2.0, so a relative error of 0.4
// is worth, in code values of the 0-255 scale:
//
//     linear L        0.05   0.18   0.445   0.64   0.89
//     0.4 L in cv     16.9   29.8    22.8   17.7   13.4
//
// The filter reached full strength only once grain would show at 13 to 30 code
// values. The eye picks grain out of a flat wall at about one. Everywhere below
// that the pixel got a proportional slice of the filtered result and kept the
// rest raw -- and since the slice is linear in sigma while sigma itself falls as
// 1/sqrt(n), the residual came out as sigma * (1 - sigma / (0.4 * L)): a
// downward parabola peaking at sigma = 0.2 L. Two noise levels either side of
// that peak leave *the same* grain. A Cornell wall at 10 and at 100 samples per
// pixel is almost exactly that pair, and the two denoised images were
// indistinguishable. The fade handed noise back at the rate the sampler removed
// it.
//
// So the test is now: how many code values would this pixel's remaining noise
// move it by, on screen? Because the fade is a linear ramp, the residual it
// leaves is sigma_d * (1 - sigma_d / tau), which peaks at sigma_d = tau / 2 and
// is worth tau / 4 there. That is the design equation -- tau / 4 bounds the
// visible grain this pass will ever leave, at any sample count -- and it is why
// the threshold is set in code values rather than as a ratio. See
// FULL_STRENGTH_GRAIN in denoise.rs for the number and its justification.

override width: u32 = 1u;
override height: u32 = 1u;
// Blend per code value of displayed noise: the reciprocal of the threshold at
// which the filter reaches full strength, premultiplied by `strength` so that
// the public knob reaches this pass. Zero means the pass is the identity, which
// is what a strength of 0 is documented to be.
override grain_blend_gain: f32 = 0.5;

// Effective tap count of prefilter_variance's kernel -- the inverse of its
// summed squared weights -- which is how many independent pixels its pooled
// variance is worth. Must be kept in step with `prefilter_radius` there.
const POOLED_TAPS = 12.1;

@group(0) @binding(0)
var<storage, read> sample_count: array<u32>;

@group(0) @binding(1)
var<storage, read> filtered: array<vec4<f32>>;

// Read and written, but only ever at this invocation's own index.
@group(0) @binding(2)
var<storage, read_write> working: array<vec4<f32>>;

// Variance of each pixel's estimate, pooled over its neighbourhood by
// denoise_atrous.wgsl's prefilter_variance. Read-only here.
@group(0) @binding(3)
var<storage, read> pooled_variance: array<f32>;

// The guide-weighted luminance of each pixel's neighbourhood, centre excluded,
// as denoise_atrous.wgsl's prefilter_variance measured it. Read-only here.
@group(0) @binding(4)
var<storage, read> local_level: array<f32>;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Linear radiance as the image will finally be written: through the tone curve,
// then gamma 2.0, on the 0-255 scale. The curve is spliced in ahead of this file
// by DenoisePostProcessor::initialize; the 0.999 and the 256.0 are
// buffer_to_image's, not decoration, and must stay in step with it.
//
// Evaluated on a grey of the pixel's level rather than per channel. The chain
// has exactly one variance and it is a luminance variance -- denoise_prepare
// takes the accumulator's .w as the Welford M2 of per-sample *luminance*, and
// local_level and the a-trous edge stop are luminance too -- so a per-channel
// criterion would dress a single number up as three it does not have. And the
// blend has to stay scalar or mix() shifts hue, which is the same argument the
// despeckle makes for scaling colour uniformly. Grey in gives grey out for every
// curve in ToneMapper, PbrNeutral included, so one channel is the whole answer.
//
// The cost is that a saturated surface is judged by its luminance, which for a
// pure primary understates its noise by up to 2x -- the direction of leaving a
// little more grain on a coloured wall than on a white one.
fn displayed(radiance: f32) -> f32 {
    let mapped = solstrale_tone_map(vec3<f32>(radiance)).x;
    return min(sqrt(mapped), 0.999) * 256.0;
}

@compute @workgroup_size(8, 8)
fn compute(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let index = gid.y * width + gid.x;

    let original = working[index];
    let n = sample_count[index];

    // Below two samples there is no Welford estimate to speak of and the image
    // is as noisy as it ever gets, so take the filtered result whole -- unless
    // the pass is switched off entirely, which is the one case that has to beat
    // the missing estimate. Parallel to despeckle_enabled in denoise_atrous.wgsl,
    // and for the same reason: "off" has to mean off at every sample count.
    var blend = select(0.0, 1.0, grain_blend_gain > 0.0);
    if (n >= 2u) {
        // The neighbourhood-pooled variance, not this pixel's own M2. A raw M2
        // at two or three samples reads near zero for a large share of pixels
        // -- the two samples of a pixel that both missed the light agree
        // closely -- and each of those pixels would be handed its unfiltered
        // value straight back here, which is the noise the filter had just
        // removed, restored one pixel at a time. That is what made the image
        // come out noisier at 2 samples per pixel than at 1, where this branch
        // does not run at all. See prefilter_variance for what pooling buys.
        //
        // Pooled over the pre-pass kernel's 12.1 effective taps, so about that
        // many times n - 1 degrees of freedom. As in the adaptive sampling test
        // in renderer/ray_trace.wgsl, what is tested is one standard deviation
        // above the point estimate rather than the estimate itself: the two
        // failures are not symmetric, since filtering a pixel that had in fact
        // converged costs a little sharpness while passing through one that has
        // not costs the visible noise the filter exists to remove. The term is
        // worth 19% at n == 2 and 7% by n == 8.
        let dof = POOLED_TAPS * f32(n - 1u);
        let variance_bound = pooled_variance[index] * (1.0 + sqrt(2.0 / dof));

        let standard_error = sqrt(variance_bound);

        // Where on the display curve to measure -- the pixel's brightness, but
        // *not* its own, where its own is the higher of the two readings
        // available.
        //
        // Reading `luminance(original)` alone is not a neutral choice, and it is
        // what put a field of bright speckles on an otherwise smooth image.
        // Noise moves a pixel up as often as down; the display curve is concave,
        // so a pixel noise moved up is measured where the curve is flatter, gets
        // a smaller displayed error, a smaller blend, and keeps more of the raw
        // value that was too bright. A pixel it moved down gets the opposite and
        // is filtered harder. The test is therefore biased to preserve upward
        // noise and remove downward noise, and preserved upward noise is exactly
        // what a speckle is.
        //
        // The neighbourhood's level is the reading the pixel's own noise cannot
        // move. Taking the lower of the two keeps the fix one-directional, and
        // it points the same way here as it did when this was a denominator: the
        // lower reading sits where the curve is steeper, so it yields the larger
        // displayed error and the stronger filtering. Worth saying, because a
        // `min` inside a denominator and a `min` inside a curve argument are not
        // obviously the same choice.
        let level = min(luminance(original.xyz), local_level[index]);

        // How far this pixel's noise would move it on screen, in code values.
        //
        // A secant rather than the derivative, and the delta method is what
        // justifies the constant rather than what computes it. Four reasons, in
        // order of weight:
        //
        // - ToneMapper::wgsl emits the curve and no derivative of it. Deriving
        //   one here would be a second hand-written copy of a curve, which is
        //   the drift `wgsl_matches_the_cpu_curve` exists to prevent. The secant
        //   works for all four curves, and for any future one, for free.
        // - It is exact where the linearisation is not. The delta method wants
        //   sigma << L, and in deep shadow or at a handful of samples sigma is
        //   the size of L or larger. The encode is concave there, so the tangent
        //   *understates* visible grain in exactly the places grain is worst; at
        //   sigma == L the secant is 41% larger, which is the safe direction.
        // - The shoulder and the 0.999 ceiling come out right with no special
        //   case. A blown highlight has displayed(L + sigma) == displayed(L -
        //   sigma), so its error is zero and it passes through untouched -- an
        //   emitter is never filtered, by construction rather than by threshold.
        // - The derivative form goes as 1/sqrt(L) and diverges at black, so it
        //   would need a luminance floor. This does not: displayed(0) is 0, and
        //   a genuinely black pixel has sigma 0 and gets blend 0.
        let displayed_error = 0.5 * (displayed(level + standard_error)
                                     - displayed(max(level - standard_error, 0.0)));

        blend = clamp(displayed_error * grain_blend_gain, 0.0, 1.0);
    }

    working[index] = vec4<f32>(
        mix(original.xyz, filtered[index].xyz, blend),
        original.w,
    );
}
