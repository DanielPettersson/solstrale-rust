override width: u32 = 1u;
override height: u32 = 1u;
override x_dir: i32 = 0;
override y_dir: i32 = 0;

@group(0) @binding(0)
var<storage, read> weights: array<f32>;

@group(0) @binding(1)
var<storage, read> input_pixels: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> output_pixels: array<vec4<f32>>;

@compute @workgroup_size(8, 8)
fn compute(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let curr_index = gid.y * width + gid.x;
    let num_weights = arrayLength(&weights);
    let half_num_weights = i32(num_weights / 2);

    var ret = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < num_weights; i++) {
        let index = get_index(
                        vec2<i32>(gid.xy),
                        (i32(i) - half_num_weights) * x_dir,
                        (i32(i) - half_num_weights) * y_dir);
        ret += input_pixels[index].xyz * weights[i];
    }

    output_pixels[curr_index] = vec4<f32>(ret, 1.0);
}

fn get_index(p: vec2<i32>, dx: i32, dy: i32) -> u32 {
    let nx = clamp(p.x + dx, 0, i32(width) - 1);
    let ny = clamp(p.y + dy, 0, i32(height) - 1);

    return u32(nx + ny * i32(width));
}
