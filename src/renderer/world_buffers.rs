//! The world's GPU buffers, kept in step with a [`WorldLayout`].
//!
//! Every array is allocated with the layout's extent and grows by half again
//! when an update outgrows it. A grown array keeps what the old one held,
//! copied on the GPU, since the segments already in it stay where they are;
//! the bind group has to be built again after a growth, and the renderer
//! builds it again after every world update anyway.

use std::sync::Arc;

use image::{DynamicImage, Rgb, RgbImage};
use wgpu::BufferUsages;

use crate::hittable::Hittables;
use crate::renderer::gpu_data::{
    BvhNode, LightRef, Material, QuadAttr, QuadPos, Sphere, TriangleAttr, TrianglePos,
};
use crate::renderer::scene_flattener::{AtlasChange, Features, WorldLayout, WorldWrites};
use crate::util::texture_processing::AtlasLayout;

/// One arena array on the GPU.
pub(crate) struct GpuArray {
    pub(crate) buffer: wgpu::Buffer,
    label: &'static str,
    element: u64,
}

/// Bytes a buffer holding `len` elements takes: at least one element, rounded
/// up to 16 bytes for WGSL array compatibility.
fn byte_size(len: u32, element: u64) -> u64 {
    (len.max(1) as u64 * element).div_ceil(16) * 16
}

impl GpuArray {
    fn new<T>(device: &wgpu::Device, label: &'static str, len: u32) -> GpuArray {
        let element = size_of::<T>() as u64;
        GpuArray {
            buffer: create(device, label, byte_size(len, element)),
            label,
            element,
        }
    }

    /// Makes room for `len` elements. `keep` copies what the array held into
    /// the new buffer, which is only worth doing when some of it is still in
    /// use. Returns whether the buffer was replaced.
    fn ensure(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, len: u32, keep: bool) -> bool {
        let needed = byte_size(len, self.element);
        let capacity = self.buffer.size();
        if needed <= capacity {
            return false;
        }
        let grown = create(
            device,
            self.label,
            needed.max((capacity + capacity / 2).div_ceil(16) * 16),
        );
        if keep {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &grown, 0, capacity);
            // Submitted now, so that the writes queued after it -- which land at
            // the start of the next submission -- are not overwritten by it.
            queue.submit([encoder.finish()]);
        }
        self.buffer = grown;
        true
    }

    fn write<T: bytemuck::Pod>(&self, queue: &wgpu::Queue, chunks: &[(u32, Vec<T>)]) {
        for (offset, data) in chunks {
            if !data.is_empty() {
                queue.write_buffer(
                    &self.buffer,
                    *offset as u64 * self.element,
                    bytemuck::cast_slice(data),
                );
            }
        }
    }
}

fn create(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Every buffer the tracer reads the world from, and the atlas.
pub(crate) struct GpuWorld {
    layout: WorldLayout,
    pub(crate) nodes: GpuArray,
    pub(crate) prim_refs: GpuArray,
    pub(crate) triangle_pos: GpuArray,
    pub(crate) triangle_attr: GpuArray,
    pub(crate) spheres: GpuArray,
    pub(crate) quad_pos: GpuArray,
    pub(crate) quad_attr: GpuArray,
    pub(crate) materials: GpuArray,
    pub(crate) lights: GpuArray,
    pub(crate) dielectric_energy: GpuArray,
    atlas: wgpu::Texture,
    pub(crate) atlas_view: wgpu::TextureView,
    pub(crate) features: Features,
    pub(crate) light_count: u32,
}

impl GpuWorld {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue, world: &Hittables) -> Self {
        let mut layout = WorldLayout::new(true);
        let writes = layout.update(world);
        let e = writes.extents;
        let (atlas, atlas_view) = upload_atlas(device, queue, None);
        let mut gpu = GpuWorld {
            layout,
            nodes: GpuArray::new::<BvhNode>(device, "Nodes Buffer", e.nodes),
            prim_refs: GpuArray::new::<u32>(device, "Prim Refs Buffer", e.prims),
            triangle_pos: GpuArray::new::<TrianglePos>(
                device,
                "Triangle Positions Buffer",
                e.triangles,
            ),
            triangle_attr: GpuArray::new::<TriangleAttr>(
                device,
                "Triangle Attributes Buffer",
                e.triangles,
            ),
            spheres: GpuArray::new::<Sphere>(device, "Spheres Buffer", e.spheres),
            quad_pos: GpuArray::new::<QuadPos>(device, "Quad Positions Buffer", e.quads),
            quad_attr: GpuArray::new::<QuadAttr>(device, "Quad Attributes Buffer", e.quads),
            materials: GpuArray::new::<Material>(
                device,
                "Materials Buffer",
                writes.materials.len() as u32,
            ),
            lights: GpuArray::new::<LightRef>(device, "Lights Buffer", writes.lights.len() as u32),
            // Empty unless the scene can present a rough dielectric, in which
            // case it is one 512-entry table per distinct index of refraction.
            dielectric_energy: GpuArray::new::<f32>(
                device,
                "Dielectric Energy Buffer",
                writes.dielectric_energy.len() as u32,
            ),
            atlas,
            atlas_view,
            features: Features::default(),
            light_count: 0,
        };
        gpu.apply(device, queue, writes);
        gpu
    }

    /// Brings the buffers in step with `world`. The bind group has to be
    /// built again afterwards, since any array may have grown.
    pub(crate) fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, world: &Hittables) {
        let writes = self.layout.update(world);
        self.apply(device, queue, writes);
    }

    fn apply(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, w: WorldWrites) {
        let keep = !w.full;
        let e = w.extents;
        self.nodes.ensure(device, queue, e.nodes, keep);
        self.prim_refs.ensure(device, queue, e.prims, keep);
        self.triangle_pos.ensure(device, queue, e.triangles, keep);
        self.triangle_attr.ensure(device, queue, e.triangles, keep);
        self.spheres.ensure(device, queue, e.spheres, keep);
        self.quad_pos.ensure(device, queue, e.quads, keep);
        self.quad_attr.ensure(device, queue, e.quads, keep);
        self.materials
            .ensure(device, queue, w.materials.len() as u32, false);
        self.lights
            .ensure(device, queue, w.lights.len() as u32, false);
        self.dielectric_energy
            .ensure(device, queue, w.dielectric_energy.len() as u32, false);

        self.nodes.write(queue, &w.nodes);
        self.prim_refs.write(queue, &w.prim_refs);
        self.triangle_pos.write(queue, &w.triangle_pos);
        self.triangle_attr.write(queue, &w.triangle_attr);
        self.spheres.write(queue, &w.spheres);
        self.quad_pos.write(queue, &w.quad_pos);
        self.quad_attr.write(queue, &w.quad_attr);
        self.materials.write(queue, &[(0, w.materials)]);
        self.lights.write(queue, &[(0, w.lights.clone())]);
        self.dielectric_energy
            .write(queue, &[(0, w.dielectric_energy)]);

        match w.atlas {
            AtlasChange::Keep => {}
            AtlasChange::Empty => {
                (self.atlas, self.atlas_view) = upload_atlas(device, queue, None);
            }
            AtlasChange::Packed(layout, textures) => {
                (self.atlas, self.atlas_view) =
                    upload_atlas(device, queue, Some((&layout, &textures)));
            }
        }

        self.features = w.features;
        self.light_count = w.lights.len() as u32;
    }
}

/// Blits the atlas using the layout the flattener computed, converts it to
/// RGBA and uploads it. `None` is a 1x1 white texture, just to have a valid
/// binding.
fn upload_atlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    packed: Option<(&AtlasLayout, &[Arc<RgbImage>])>,
) -> (wgpu::Texture, wgpu::TextureView) {
    let atlas_image = match packed {
        Some((layout, textures)) => {
            let mut atlas_image = RgbImage::new(layout.width, layout.height);
            for placement in layout.placements.iter() {
                image::imageops::replace(
                    &mut atlas_image,
                    textures[placement.original_index].as_ref(),
                    placement.x as i64,
                    placement.y as i64,
                );
            }
            atlas_image
        }
        None => RgbImage::from_pixel(1, 1, Rgb([255, 255, 255])),
    };

    let texture_extent = wgpu::Extent3d {
        width: atlas_image.width(),
        height: atlas_image.height(),
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Texture Atlas"),
        size: texture_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let atlas_rgba = DynamicImage::ImageRgb8(atlas_image).to_rgba8();

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &atlas_rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * atlas_rgba.width()),
            rows_per_image: Some(atlas_rgba.height()),
        },
        texture_extent,
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2),
        ..Default::default()
    });
    (texture, view)
}
