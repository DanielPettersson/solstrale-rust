// Rec. 709 relative luminance. Spliced ahead of every shader that calls it --
// WGSL has no include directive -- so the weights live in exactly one place.
// Mirrors `util::luminance` on the CPU.
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}
