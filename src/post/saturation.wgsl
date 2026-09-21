override width: u32 = 1u;
override height: u32 = 1u;
override saturation_factor: f32 = 0;

@group(0) @binding(0)
var<storage, read_write> pixels: array<vec4<f32>>;

// 2-D, like every other pass over the image: a 1-D dispatch needs one workgroup
// per 64 pixels, which crosses `max_compute_workgroups_per_dimension` (65535 on
// a Radeon RX 5700 XT) at 4194240 pixels and fails validation outright above it.
@compute @workgroup_size(8, 8)
fn compute(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let curr_index = gid.y * width + gid.x;
    let pixel = pixels[curr_index].xyz;

    let gray = luminance(pixel);
    let g = -gray * saturation_factor;
    let gg = 1.0 + saturation_factor;
    pixels[curr_index] = vec4<f32>(
        g + pixel.x * gg,
        g + pixel.y * gg,
        g + pixel.z * gg,
        1.0
    );
}
