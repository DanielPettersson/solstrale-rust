struct Config {
    width: u32,
    x_dir: i32,
    y_dir: i32
}

@group(0) @binding(0)
var<storage, read> config: Config;

@group(0) @binding(1)
var<storage, read> weights: array<f32>;

@group(0) @binding(2)
var<storage, read> input_pixels: array<vec3<f32>>;

@group(0) @binding(3)
var<storage, read_write> output_pixels: array<vec3<f32>>;

@compute @workgroup_size(64)
fn compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let curr_index = global_id.x;
    let num_pixels = arrayLength(&input_pixels);
    let num_weights = arrayLength(&weights);
    let half_num_weights = i32(num_weights / 2);
    
    if (curr_index >= num_pixels) {
        return;
    }

    var ret = vec3(0.0);
    for (var i: u32 = 0; i < num_weights; i++) {
        let index = get_index(
                        curr_index,
                        (i32(i) - half_num_weights) * config.x_dir,
                        (i32(i) - half_num_weights) * config.y_dir,
                        config.width,
                        num_pixels);
        ret += input_pixels[index] * weights[i];
    }

    output_pixels[curr_index] = ret;
}

fn get_index(curr_index: u32, dx: i32, dy: i32, width: u32, num_pixels: u32) -> u32 {
    let new_index = i32(curr_index) + dx + dy * i32(width);
    if (new_index < 0 || new_index >= i32(num_pixels)) {
        return curr_index;
    }
    return u32(new_index);
}