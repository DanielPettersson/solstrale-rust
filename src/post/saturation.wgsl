override saturation_factor: f32 = 0;

@group(0) @binding(0)
var<storage, read_write> pixels: array<vec4<f32>>;

@compute @workgroup_size(64)
fn compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let curr_index = global_id.x;
    let num_pixels = arrayLength(&pixels);

    if (curr_index >= num_pixels) {
        return;
    }
    let pixel = pixels[curr_index].xyz;

    let gray = 0.2989 * pixel.x + 0.587 * pixel.y + 0.114 * pixel.z;
    let g = -gray * saturation_factor;
    let gg = 1.0 + saturation_factor;
    pixels[curr_index] = vec4<f32>(
        g + pixel.x * gg,
        g + pixel.y * gg,
        g + pixel.z * gg,
        1.0
    );
}
