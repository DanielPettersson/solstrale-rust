@group(0) @binding(0)
var<storage, read_write> pixels: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> bloom_pixels: array<vec4<f32>>;

@compute @workgroup_size(64)
fn compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let curr_index = global_id.x;
    let num_pixels = arrayLength(&pixels);

    if (curr_index >= num_pixels) {
        return;
    }

    pixels[curr_index] = vec4<f32>(pixels[curr_index].xyz + bloom_pixels[curr_index].xyz, 1.0);
}
