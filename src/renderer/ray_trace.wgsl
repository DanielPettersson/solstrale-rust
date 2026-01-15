@group(0) @binding(0)
var<storage, read_write> output_buffer: array<vec3<f32>>;

@compute @workgroup_size(64)
fn compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_buffer)) {
        return;
    }
    // Red color
    output_buffer[index] = vec3<f32>(1.0, 0.0, 0.0);
}
