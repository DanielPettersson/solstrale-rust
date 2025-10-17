@group(0) @binding(0)
var<storage, read> input_pixels: array<vec3<f32>>;

@group(0) @binding(1)
var<storage, read> bloom_pixels: array<vec3<f32>>;

@group(0) @binding(2)
var<storage, read_write> output_pixels: array<vec3<f32>>;

@compute @workgroup_size(64)
fn compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let curr_index = global_id.x;
    let num_pixels = arrayLength(&input_pixels);

    if (curr_index >= num_pixels) {
        return;
    }

    output_pixels[curr_index] = input_pixels[curr_index] + bloom_pixels[curr_index];
}
