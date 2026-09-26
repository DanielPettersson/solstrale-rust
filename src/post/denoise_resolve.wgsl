// Blends the filtered image back over the original, in proportion to how much
// of the remaining noise a viewer would actually see.
//
// The variance guidance inside the a-trous passes shrinks the filter's tolerance
// as a render converges, but not far enough on its own: the tolerance tracks the
// residual noise, and once an image is nearly converged the real detail at pixel
// scale is the same size as that noise. So the filter is faded out, and the
// question is what to fade it against.
//
// The test is: how many code values would this pixel's remaining noise move it
// by, on screen? Not the relative standard error adaptive sampling retires
// pixels on -- through the tone curve and transfer function a relative error of
// 0.4 is worth 13 to 30 code values depending on brightness, where the eye picks
// grain out of a flat wall at about one, so that fade handed noise back at the
// rate the sampler removed it.
//
// Because the fade is a linear ramp, the residual it leaves is
// sigma_d * (1 - sigma_d / tau), which peaks at sigma_d = tau / 2 and is worth
// tau / 4 there. That is the design equation: tau / 4 bounds the visible grain
// this pass will ever leave, at any sample count, which is why the threshold is
// in code values rather than a ratio. See FULL_STRENGTH_GRAIN in denoise.rs.

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

// Linear radiance roughly as the image will finally be written: through the
// tone curve, then a transfer function, on the 0-255 scale. The curve is
// spliced in ahead of this file by DenoisePostProcessor::initialize.
//
// Note the transfer function here is the sRGB OETF while `tone_map_pack.wgsl`
// encodes gamma 2.0, so the two disagree by up to a handful of code values.
// What this pass needs is a visibility yardstick, not the exact encode, so the
// difference is well inside the fade's tolerance -- but it is a difference.
//
// Evaluated on a grey of the pixel's level rather than per channel: the chain
// has exactly one variance and it is a luminance variance, and the blend has to
// stay scalar or mix() shifts hue. Grey in gives grey out for every curve in
// ToneMapper. The cost is that a pure primary's noise is understated by up to
// 2x, which leaves a little more grain on a coloured wall than on a white one.
fn linear_to_srgb(c: f32) -> f32 {
    return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
}

fn displayed(radiance: f32) -> f32 {
    let mapped = solstrale_tone_map(vec3<f32>(radiance)).x;
    return min(linear_to_srgb(mapped), 0.999) * 256.0;
}

@compute @workgroup_size(8, 8)
fn compute(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let index = gid.y * width + gid.x;

    let original = working[index];
    let n = sample_count[index];

    // Below two samples there is no Welford estimate and the image is as noisy
    // as it ever gets, so take the filtered result whole -- unless the pass is
    // switched off entirely, since "off" has to mean off at every sample
    // count.
    var blend = select(0.0, 1.0, grain_blend_gain > 0.0);
    if (n >= 2u) {
        // The neighbourhood-pooled variance, not this pixel's own M2. A raw M2
        // at two or three samples reads near zero for a large share of pixels,
        // and each of those would be handed its unfiltered value straight back
        // -- which made the image noisier at 2 samples per pixel than at 1,
        // where this branch does not run. See prefilter_variance.
        //
        // Pooled over the pre-pass kernel's 12.1 effective taps, so about that
        // many times n - 1 degrees of freedom. As in the adaptive sampling test
        // in renderer/ray_trace.wgsl, one standard deviation above the point
        // estimate is tested rather than the estimate itself, since filtering a
        // converged pixel costs a little sharpness where passing through an
        // unconverged one costs visible noise. Worth 19% at n == 2, 7% by 8.
        let dof = POOLED_TAPS * f32(n - 1u);
        let variance_bound = pooled_variance[index] * (1.0 + sqrt(2.0 / dof));

        let standard_error = sqrt(variance_bound);

        // Where on the display curve to measure. Not the pixel's own luminance
        // alone: noise moves a pixel up as often as down, and the display curve
        // is concave, so a pixel noise moved up is measured where the curve is
        // flatter, gets a smaller blend and keeps more of the raw value that was
        // too bright. That bias preserves upward noise, which is a speckle.
        //
        // The neighbourhood's level is the reading the pixel's own noise cannot
        // move, and the lower of the two sits where the curve is steeper, so it
        // yields the stronger filtering.
        let level = min(luminance(original.xyz), local_level[index]);

        // How far this pixel's noise would move it on screen, in code values.
        //
        // A secant rather than the derivative:
        //
        // - ToneMapper::wgsl emits the curve and no derivative of it, so a
        //   tangent would mean a second hand-written copy of every curve.
        // - It is exact where the linearisation is not. In deep shadow or at a
        //   handful of samples sigma is the size of L or larger, and the concave
        //   encode makes the tangent understate grain exactly where grain is
        //   worst; at sigma == L the secant is 41% larger.
        // - The shoulder and the 0.999 ceiling come out right with no special
        //   case: a blown highlight has zero error and passes through untouched.
        // - The derivative form goes as 1/sqrt(L) and would need a luminance
        //   floor. This does not: displayed(0) is 0.
        let displayed_error = 0.5 * (displayed(level + standard_error)
                                     - displayed(max(level - standard_error, 0.0)));

        blend = clamp(displayed_error * grain_blend_gain, 0.0, 1.0);
    }

    working[index] = vec4<f32>(
        mix(original.xyz, filtered[index].xyz, blend),
        original.w,
    );
}
