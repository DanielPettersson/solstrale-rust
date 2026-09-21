// Tone maps the linear HDR buffer and packs it to RGBA8, so what crosses the
// bus on readback is 4 bytes a pixel instead of 16.
//
// `solstrale_tone_map` is not defined here: `ToneMapper::wgsl` emits it and
// `buffer_to_image` concatenates the two, the same splice the denoiser's
// resolve pass does.

override width: u32 = 1u;
override height: u32 = 1u;

@group(0) @binding(0) var<storage, read> pixels: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> packed_pixels: array<u32>;

// 2-D, like every other pass over the image: a 1-D dispatch crosses
// `max_compute_workgroups_per_dimension` below 4K.
@compute @workgroup_size(8, 8)
fn compute(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let index = gid.y * width + gid.x;

    let mapped = solstrale_tone_map(pixels[index].xyz);
    // Gamma 2.0, and the 0.999 ceiling so the `* 256` cannot reach 256 and wrap
    // the truncation to u32. The curve already guarantees a non-negative input
    // to `sqrt`, NaN included.
    let encoded = vec3<u32>(min(sqrt(mapped), vec3<f32>(0.999)) * 256.0);

    // Red in the low byte. The host unpacks by shifting the u32 rather than by
    // reinterpreting it as bytes, so this order is the only one that matters.
    packed_pixels[index] = encoded.r | (encoded.g << 8u) | (encoded.b << 16u);
}
