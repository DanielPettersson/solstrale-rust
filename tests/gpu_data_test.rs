#[cfg(test)]
mod tests {
    use solstrale::renderer::gpu_data::{
        BvhNode, GpuRenderConfig, LightRef, Material, QuadAttr, QuadPos, Ray, Sphere, TriangleAttr,
        TrianglePos,
    };
    use std::mem::{offset_of, size_of};

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
        // Packed primitive reference, selection probability, alias pair.
        assert_eq!(size_of::<LightRef>(), 16);
        // Two child boxes + two meta words, padded for WGSL vec3 alignment.
        assert_eq!(size_of::<BvhNode>(), 64);
        assert_eq!(size_of::<GpuRenderConfig>(), 48);
    }

    /// Size alone cannot tell "packed into the padding" from "happens to still
    /// be 80 bytes", so the offsets are pinned too. WGSL's 16-byte `vec3` and
    /// 8-byte `vec2` alignment leaves one free word at 44 and two at 72 --
    /// exactly what three `pack2x16snorm` normals need. Get one wrong and the
    /// shader reads a UV as a normal.
    #[test]
    fn triangle_attr_normals_sit_in_the_old_padding() {
        assert_eq!(offset_of!(TriangleAttr, normal), 0);
        assert_eq!(offset_of!(TriangleAttr, material_index), 12);
        assert_eq!(offset_of!(TriangleAttr, tangent), 16);
        assert_eq!(offset_of!(TriangleAttr, area), 28);
        assert_eq!(offset_of!(TriangleAttr, bi_tangent), 32);
        assert_eq!(offset_of!(TriangleAttr, n0_oct), 44);
        assert_eq!(offset_of!(TriangleAttr, uv0), 48);
        assert_eq!(offset_of!(TriangleAttr, uv1), 56);
        assert_eq!(offset_of!(TriangleAttr, uv2), 64);
        assert_eq!(offset_of!(TriangleAttr, n1_oct), 72);
        assert_eq!(offset_of!(TriangleAttr, n2_oct), 76);
    }
}
