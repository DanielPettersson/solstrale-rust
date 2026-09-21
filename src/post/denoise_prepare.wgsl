// Seeds the denoise chain: colour from the working image, and a per-pixel
// variance derived from what the sample loop already measured.

override width: u32 = 1u;
override height: u32 = 1u;
// Radius of the spatial variance fallback used while a pixel has too few
// samples for Welford to say anything about itself.
override spatial_radius: i32 = 3;

// The renderer's pristine accumulator: xyz is the running mean colour, w is the
// Welford M2 of the per-sample luminance. Read-only -- the render loop keeps
// accumulating into it, and it is the only untouched source of variance once a
// grading post-processor has run ahead of us.
@group(0) @binding(0)
var<storage, read> accumulator: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> sample_count: array<u32>;

// The chain's working copy. Colour comes from here rather than from the
// accumulator, so a denoiser placed after another post-processor filters what
// it actually sees.
@group(0) @binding(2)
var<storage, read> working: array<vec4<f32>>;

// xyz: colour, w: variance of the colour estimate.
@group(0) @binding(3)
var<storage, read_write> dst: array<vec4<f32>>;

// Variance of the neighbourhood's luminance. At n == 1 each pixel is a single
// sample, so the spread across neighbours is a direct estimate of the spread
// this pixel's own sample would have had.
fn spatial_luminance_variance(p: vec2<i32>) -> f32 {
    var sum = 0.0;
    var sum_sq = 0.0;
    var count = 0.0;
    for (var dy = -spatial_radius; dy <= spatial_radius; dy++) {
        for (var dx = -spatial_radius; dx <= spatial_radius; dx++) {
            let x = clamp(p.x + dx, 0, i32(width) - 1);
            let y = clamp(p.y + dy, 0, i32(height) - 1);
            let l = luminance(working[u32(y) * width + u32(x)].xyz);
            sum += l;
            sum_sq += l * l;
            count += 1.0;
        }
    }
    let mean = sum / count;
    return max(sum_sq / count - mean * mean, 0.0);
}

@compute @workgroup_size(8, 8)
fn compute(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let index = gid.y * width + gid.x;

    let n = sample_count[index];
    var variance: f32;

    if (n >= 2u) {
        // M2/(n-1) is the variance of one sample. Dividing by n again gives the
        // variance of the *mean* -- the squared standard error -- which is what
        // this pixel actually holds, and the only scale a luminance tolerance
        // can sensibly be measured in.
        //
        // This is also what makes the filter fade itself out as the render
        // converges: the tolerance goes as 1/sqrt(n), so a finished image passes
        // through close to untouched and there is nothing to turn off.
        variance = accumulator[index].w / (f32(n - 1u) * f32(n));
    } else {
        // A single sample has M2 exactly zero. Taken at face value that declares
        // a 1-spp image perfectly converged and passes it through raw, which is
        // the worst possible answer. Fall back to the spatial estimate, as SVGF
        // does when its temporal history is short.
        variance = spatial_luminance_variance(vec2<i32>(gid.xy));
    }

    dst[index] = vec4<f32>(working[index].xyz, variance);
}
