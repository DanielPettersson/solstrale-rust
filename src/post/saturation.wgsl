override saturation_factor: f32 = 0;

@group(0) @binding(0)
var<storage, read> input_pixels: array<vec3<f32>>;

@group(0) @binding(1)
var<storage, read_write> output_pixels: array<vec3<f32>>;

@compute @workgroup_size(64)
fn compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let curr_index = global_id.x;
    let num_pixels = arrayLength(&input_pixels);

    if (curr_index >= num_pixels) {
        return;
    }
    let pixel = input_pixels[curr_index];

    let gray = 0.2989 * pixel.x + 0.587 * pixel.y + 0.114 * pixel.z;
    let g = -gray * saturation_factor;
    let gg = 1 + saturation_factor;
    output_pixels[curr_index] = vec3(
        g + pixel.x * gg,
        g + pixel.y * gg,
        g + pixel.z * gg
    );
}
