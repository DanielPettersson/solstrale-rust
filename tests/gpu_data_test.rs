#[cfg(test)]
mod tests {
    use solstrale::renderer::gpu_data::{
        BvhNode, GpuRenderConfig, Material, Quad, Ray, Sphere, Triangle,
    };
    use std::mem::size_of;

    #[test]
    fn test_struct_sizes() {
        assert_eq!(size_of::<Ray>(), 32);
        assert_eq!(size_of::<Sphere>(), 32);
        assert_eq!(size_of::<Material>(), 64);
        assert_eq!(size_of::<Triangle>(), 144);
        assert_eq!(size_of::<Quad>(), 144);
        assert_eq!(size_of::<BvhNode>(), 32);
        assert_eq!(size_of::<GpuRenderConfig>(), 32);
    }
}
