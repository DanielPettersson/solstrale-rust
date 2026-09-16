// Blends the filtered image back over the original, in proportion to how noisy
// each pixel still is.
//
// The variance guidance inside the a-trous passes shrinks the filter's tolerance
// as a render converges, but not far enough on its own: the tolerance tracks the
// residual noise, and once an image is nearly converged the real detail at pixel
// scale is the same size as that noise. Measured on the test scene, the filter
// removed less error than it introduced from roughly 64 samples per pixel
// upwards.
//
// So the filter is faded out against the renderer's own definition of a
// converged pixel -- the same relative standard error that adaptive sampling
// retires pixels on. A pixel that is still visibly noisy is taken filtered, a
// pixel that has converged is left alone, and the crossover is a knob.

override width: u32 = 1u;
override height: u32 = 1u;
// Relative standard error at which the filter reaches full strength. Pixels
// quieter than this are blended back toward their unfiltered value.
override full_strength_error: f32 = 0.4;

// Floor on the luminance used as the denominator, so a near-black pixel's tiny
// absolute noise does not read as enormous relative to it. Matches
// ADAPTIVE_LUMINANCE_FLOOR in renderer/ray_trace.wgsl.
const LUMINANCE_FLOOR = 1e-4;

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

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
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
    // is as noisy as it ever gets, so take the filtered result whole.
    var blend = 1.0;
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
        let relative = standard_error / max(luminance(original.xyz), LUMINANCE_FLOOR);
        blend = clamp(relative / full_strength_error, 0.0, 1.0);
    }

    working[index] = vec4<f32>(
        mix(original.xyz, filtered[index].xyz, blend),
        original.w,
    );
}
