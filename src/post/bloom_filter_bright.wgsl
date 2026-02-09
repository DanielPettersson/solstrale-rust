override threshold: f32 = 1.0;
override max_intensity: f32 = 1000.0;

@group(0) @binding(0)
var<storage, read> input_pixels: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read_write> output_pixels: array<vec4<f32>>;

@compute @workgroup_size(64)
fn compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let curr_index = global_id.x;
    let num_pixels = arrayLength(&input_pixels);

    if (curr_index >= num_pixels) {
        return;
    }

    output_pixels[curr_index] = vec4<f32>(get_bloom_color(input_pixels[curr_index].xyz), 1.0);
}

fn get_bloom_color(col: vec3<f32>) -> vec3<f32> {
    let len = length(col);
    if (len >= threshold) {
        if (len > max_intensity) {
            return col * (max_intensity / len);
        }
        return col;
    }
    return vec3<f32>(0.0);
}
