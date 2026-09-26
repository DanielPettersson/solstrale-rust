//! Post processors for applying effects to the raw rendered image

mod bloom;
mod denoise;
mod saturation;

use std::error::Error;

use enum_dispatch::enum_dispatch;

use crate::util::gpu_timing::GpuTimer;

pub use crate::post::bloom::BloomPostProcessor;
pub use crate::post::denoise::{DenoiseGuide, DenoisePostProcessor};
pub use crate::post::saturation::SaturationPostProcessor;

/// Size in bytes of one pixel in the working and accumulation buffers: a
/// `vec4<f32>`, since a `vec3` is 16-byte aligned on the GPU anyway.
pub(crate) const PIXEL_SIZE: u64 = 16;

/// Everything a [`PostProcessor`] may draw on for one invocation.
///
/// Constructed by the renderer. The field set is deliberately open, so a future
/// post-processor needing another of the renderer's buffers does not force a
/// second breaking change on the trait.
#[non_exhaustive]
pub struct PostProcessContext<'a> {
    /// Command encoder the post-processor records its passes into. The renderer
    /// submits it.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// The chain's working image, `array<vec4<f32>>` in linear HDR. Read and
    /// written in place: whatever the last post-processor leaves here is what
    /// the caller is handed.
    pub buffer: &'a wgpu::Buffer,
    /// The renderer's untouched accumulator: `xyz` is the running mean colour
    /// and `w` the Welford M2 of the per-sample luminance.
    ///
    /// Read-only, and the reason the chain runs on a copy at all. The render
    /// loop keeps accumulating into this buffer, so a write here would feed a
    /// filtered mean back into the next batch's Welford merge: a little more
    /// blur every batch, a destroyed variance estimate, and adaptive sampling
    /// retiring pixels on a fabricated number. Silent, and worse the longer you
    /// render.
    pub accumulator: &'a wgpu::Buffer,
    /// Samples actually accumulated per pixel, `array<u32>`. Diverges from
    /// `samples_completed` wherever adaptive sampling has retired a pixel.
    pub sample_count_buffer: &'a wgpu::Buffer,
    /// Primary-hit albedo, shading normal and camera distance, packed into
    /// `array<vec4<u32>>` at 16 bytes per pixel. See `pack_guide` in
    /// `renderer/ray_trace.wgsl` for the layout.
    pub gbuffer: &'a wgpu::Buffer,
    /// Samples accumulated into the accumulator before this invocation.
    pub samples_completed: u32,
    /// The device the passes run on.
    pub device: &'a wgpu::Device,
    /// Per-pass GPU timing. `None` unless `SOLSTRALE_GPU_TIMING` is set and
    /// the device supports timestamp queries, which is the usual case -- see
    /// [`GpuTimer`].
    ///
    /// A post-processor passes this down to each of its compute passes, which
    /// is what names them separately in the report. Per-processor rather than
    /// one block around the chain, because the denoiser's cost is not one
    /// number: a prepare, a prefilter, five iterations and a resolve.
    pub timer: Option<&'a mut GpuTimer>,
}

/// Responsible for taking the rendered image and transforming it
#[enum_dispatch]
pub trait PostProcessor {
    /// Does post-construct initialization for the post-processor when width and height are known
    fn initialize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32);

    /// Execute final postprocessing of the rendered image
    fn post_process(&self, ctx: &mut PostProcessContext) -> Result<(), Box<dyn Error>>;

    /// Whether this processor also runs on the unfinished image a live viewer
    /// is watching, rather than only on the last batch.
    ///
    /// Only consulted when [`RenderConfig::preview`] is on. Costs a full run of
    /// the processor per batch, so it is for the ones whose absence makes the
    /// preview misleading rather than merely unstyled. The denoiser is the case
    /// it exists for: a camera drag restarts the accumulation every frame, so
    /// the preview is always a one-sample image.
    ///
    /// [`RenderConfig::preview`]: crate::renderer::RenderConfig::preview
    fn preview(&self) -> bool {
        false
    }

    /// Whether this processor reads [`PostProcessContext::gbuffer`].
    ///
    /// The tracer is compiled without `trace_guide` when nothing in the chain
    /// says yes here, so a processor that reads the guide and forgets to say so
    /// reads zeroes.
    fn needs_guide(&self) -> bool {
        false
    }
}

#[enum_dispatch(PostProcessor)]
#[derive(Clone)]
#[non_exhaustive]
/// An enum of available post-processors
pub enum PostProcessors {
    /// [`PostProcessor`] of type [`BloomPostProcessor`]
    BloomPostProcessor,
    /// [`PostProcessor`] of type [`SaturationPostProcessor`]
    SaturationPostProcessor,
    /// [`PostProcessor`] of type [`DenoisePostProcessor`]
    DenoisePostProcessor,
}
