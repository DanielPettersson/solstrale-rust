override width: u32 = 1u;
override height: u32 = 1u;

@group(0) @binding(0)
var<storage, read_write> pixels: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> bloom_pixels: array<vec4<f32>>;

@compute @workgroup_size(8, 8)
fn compute(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let curr_index = gid.y * width + gid.x;

    pixels[curr_index] = vec4<f32>(pixels[curr_index].xyz + bloom_pixels[curr_index].xyz, 1.0);
}
