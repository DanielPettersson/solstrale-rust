struct Config {
    threshold: f32,
    max_intensity: f32,
}

@group(0) @binding(0)
var<storage, read> config: Config;

@group(0) @binding(1)
var<storage, read> input_pixels: array<vec3<f32>>;

@group(0) @binding(2)
var<storage, read_write> output_pixels: array<vec3<f32>>;

@compute @workgroup_size(64)
fn compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let curr_index = global_id.x;
    let num_pixels = arrayLength(&input_pixels);

    if (curr_index >= num_pixels) {
        return;
    }

    output_pixels[curr_index] = get_bloom_color(input_pixels[curr_index], config.threshold, config.max_intensity);
}

fn get_bloom_color(col: vec3<f32>, threshold: f32, max_intensity: f32) -> vec3<f32> {
    let len = length(col);
    if (len >= threshold) {
        if (len > max_intensity) {
            return col * (max_intensity / len);
        }
        return col;
    }
    return vec3(0.0);
}
