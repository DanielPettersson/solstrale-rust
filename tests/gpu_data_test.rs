#[cfg(test)]
mod tests {
    use solstrale::renderer::gpu_data::{
        BvhNode, GpuRenderConfig, Material, QuadAttr, QuadPos, Ray, Sphere, TriangleAttr, TrianglePos,
    };
    use std::mem::size_of;

    #[test]
    fn test_struct_sizes() {
        assert_eq!(size_of::<Ray>(), 32);
        assert_eq!(size_of::<Sphere>(), 32);
        assert_eq!(size_of::<Material>(), 96);
        // Split hot/cold: traversal reads only the *Pos structs.
        assert_eq!(size_of::<TrianglePos>(), 48);
        assert_eq!(size_of::<TriangleAttr>(), 80);
        assert_eq!(size_of::<QuadPos>(), 80);
        assert_eq!(size_of::<QuadAttr>(), 32);
        // Two child boxes + two meta words, padded for WGSL vec3 alignment.
        assert_eq!(size_of::<BvhNode>(), 64);
        assert_eq!(size_of::<GpuRenderConfig>(), 48);
    }
}
