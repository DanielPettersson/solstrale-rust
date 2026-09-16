//! Post-processor for removing Monte Carlo noise

use crate::post::{PIXEL_SIZE, PostProcessContext, PostProcessor};
use crate::util::wgpu_util::{
    add_compute_pass_2d, bind_group, bind_group_layout, compute_pipeline,
    compute_pipeline_with_entry, storage_binding,
};
use std::error::Error;
use wgpu::BufferUsages;

/// Which primary-hit channels the edge-stopping functions are allowed to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DenoiseGuide {
    /// Use the renderer's primary-hit albedo, normal and depth. What you want
    /// unless you are measuring what the guide buys.
    #[default]
    Full,
    /// Use only the accumulated colour and its per-pixel variance. Noticeably
    /// softer: silhouettes between similarly-lit surfaces merge, and texture
    /// detail is the first thing to go.
    ColorOnly,
}

/// Removes Monte Carlo noise with an edge-avoiding à-trous wavelet filter
/// (Dammertz et al. 2010), with the tap weights guided by the per-pixel variance
/// the sample loop already tracks (Schied et al. 2017).
///
/// Spatial only. The temporal accumulation SVGF needs is what the sample loop
/// already is, and it is exact rather than reprojected.
///
/// The filter fades out as the render converges, so it is safe to leave enabled:
/// its tolerance is set by the variance of the pixel *mean*, which falls as
/// `1/n`, and the result is then blended back over the original in proportion to
/// each pixel's remaining relative standard error. Measured against a converged
/// reference on the test scene, it cuts linear RMSE by about a third at 8
/// samples per pixel, by about a sixth at 64, and changes a 2000-sample image by
/// 0.2%.
///
/// Costs seven compute dispatches, run once on the finished image: 4.2 ms at
/// 800x600 on a Radeon RX 5700 XT, against 52 ms for the render itself at 16
/// samples per pixel. The cost is per pixel and independent of sample count, so
/// it matters less the longer the render.
///
/// Place this first in [`crate::renderer::RenderConfig::post_processors`].
/// Denoising a bloomed image blurs the bloom; bloom applied to a denoised image
/// is what you want.
///
/// Known limitation: the guide describes the primary hit, so on a mirror or a
/// glass surface it describes that surface rather than what is seen through or
/// in it, and the reflected or refracted image is blurred along the surface.
#[derive(Clone)]
pub struct DenoisePostProcessor {
    width: u32,
    height: u32,

    iterations: u32,
    strength: f64,
    guide: DenoiseGuide,

    prepare_module: wgpu::ShaderModule,
    atrous_module: wgpu::ShaderModule,
    resolve_module: wgpu::ShaderModule,

    prepare_bind_group_layout: wgpu::BindGroupLayout,
    atrous_bind_group_layout: wgpu::BindGroupLayout,

    prepare_pipeline: Option<wgpu::ComputePipeline>,
    prefilter_pipeline: Option<wgpu::ComputePipeline>,
    resolve_pipeline: Option<wgpu::ComputePipeline>,
    /// One per à-trous iteration, differing only in their `step_width`
    /// override, so the tap spacing is a compile-time constant.
    atrous_pipelines: Vec<wgpu::ComputePipeline>,

    buffer_a: Option<wgpu::Buffer>,
    buffer_b: Option<wgpu::Buffer>,
}

/// Bounds on the iteration count. Five reaches 32 pixels, which is the usual
/// choice; eight reaches 255, which only makes sense above 1080p.
const MIN_ITERATIONS: u32 = 1;
const MAX_ITERATIONS: u32 = 8;

impl DenoisePostProcessor {
    /// Creates a new denoiser.
    ///
    /// # Arguments
    /// * `strength` Multiplies the luminance tolerance. 1 is neutral, below 1
    ///   keeps more detail and more noise, above 1 blurs harder. 0 to 10.
    /// * `iterations` Number of à-trous iterations, each doubling the tap
    ///   spacing. If not specified, defaults to 5. 1 to 8.
    /// * `guide` Which primary-hit channels may guide the filter. If not
    ///   specified, defaults to [`DenoiseGuide::Full`].
    pub fn new(
        strength: f64,
        iterations: Option<u32>,
        guide: Option<DenoiseGuide>,
        device: &wgpu::Device,
    ) -> Result<Self, simple_error::SimpleError> {
        if !(0. ..=10.).contains(&strength) {
            return Err(simple_error::SimpleError::new(
                "strength must be between 0 and 10",
            ));
        }

        let iterations = iterations.unwrap_or(5);
        if !(MIN_ITERATIONS..=MAX_ITERATIONS).contains(&iterations) {
            return Err(simple_error::SimpleError::new(
                "iterations must be between 1 and 8",
            ));
        }

        let prepare_module =
            device.create_shader_module(wgpu::include_wgsl!("denoise_prepare.wgsl"));
        let atrous_module = device.create_shader_module(wgpu::include_wgsl!("denoise_atrous.wgsl"));
        let resolve_module =
            device.create_shader_module(wgpu::include_wgsl!("denoise_resolve.wgsl"));

        let prepare_bind_group_layout = bind_group_layout(
            device,
            &[
                storage_binding(true, 16),  // accumulator
                storage_binding(true, 4),   // per-pixel sample count
                storage_binding(true, 16),  // working image
                storage_binding(false, 16), // destination
            ],
        );

        let atrous_bind_group_layout = bind_group_layout(
            device,
            &[
                storage_binding(true, 16),  // source
                storage_binding(false, 16), // destination
                storage_binding(true, 16),  // primary-hit guide
            ],
        );

        Ok(DenoisePostProcessor {
            width: 0,
            height: 0,
            iterations,
            strength,
            guide: guide.unwrap_or_default(),
            prepare_module,
            atrous_module,
            resolve_module,
            prepare_bind_group_layout,
            atrous_bind_group_layout,
            prepare_pipeline: None,
            prefilter_pipeline: None,
            resolve_pipeline: None,
            atrous_pipelines: Vec::new(),
            buffer_a: None,
            buffer_b: None,
        })
    }
}

impl PostProcessor for DenoisePostProcessor {
    fn initialize(&mut self, device: &wgpu::Device, _queue: &wgpu::Queue, width: u32, height: u32) {
        if self.width == width && self.height == height && self.buffer_a.is_some() {
            return;
        }

        self.width = width;
        self.height = height;

        let dimensions = [("width", width as f64), ("height", height as f64)];

        self.prepare_pipeline = Some(compute_pipeline(
            device,
            &self.prepare_bind_group_layout,
            &self.prepare_module,
            &dimensions,
        ));

        // The paper's defaults, with only the luminance falloff exposed. Held in
        // one place so every pipeline built below agrees on them.
        //
        // SVGF's sigma_l is 4. Half that measured best here across 8, 64 and
        // 2000 samples per pixel on the test scene -- unsurprising, since SVGF
        // filters a reprojected temporal estimate whose variance is far shakier
        // than the exact Welford one the sample loop hands us. Folded into the
        // base so that a `strength` of 1 is the tuned default rather than a
        // number users have to know to halve.
        let sigmas = [
            ("sigma_colour", 2.0 * self.strength),
            ("sigma_normal", 128.),
            ("sigma_depth", 1.),
            ("sigma_albedo", 0.1),
            (
                "use_guide",
                match self.guide {
                    DenoiseGuide::Full => 1.,
                    DenoiseGuide::ColorOnly => 0.,
                },
            ),
        ];

        let constants = |step_width: u32| {
            let mut c = dimensions.to_vec();
            c.extend_from_slice(&sigmas);
            c.push(("step_width", step_width as f64));
            c
        };

        self.prefilter_pipeline = Some(compute_pipeline_with_entry(
            device,
            &self.atrous_bind_group_layout,
            &self.atrous_module,
            "prefilter_variance",
            &constants(1),
        ));

        // Shares the prepare layout: both are [ro accumulator, ro sample count,
        // ro image, rw image].
        self.resolve_pipeline = Some(compute_pipeline(
            device,
            &self.prepare_bind_group_layout,
            &self.resolve_module,
            &dimensions,
        ));

        self.atrous_pipelines = (0..self.iterations)
            .map(|i| {
                compute_pipeline(
                    device,
                    &self.atrous_bind_group_layout,
                    &self.atrous_module,
                    &constants(1 << i),
                )
            })
            .collect();

        let size = (width * height) as u64 * PIXEL_SIZE;
        let scratch = || {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Denoise Buffer"),
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }))
        };
        self.buffer_a = scratch();
        self.buffer_b = scratch();
    }

    fn post_process(&self, ctx: &mut PostProcessContext) -> Result<(), Box<dyn Error>> {
        let buffer_a = self.buffer_a.as_ref().ok_or("Not initialized")?;
        let buffer_b = self.buffer_b.as_ref().ok_or("Not initialized")?;
        let prepare_pipeline = self.prepare_pipeline.as_ref().ok_or("Not initialized")?;
        let prefilter_pipeline = self.prefilter_pipeline.as_ref().ok_or("Not initialized")?;
        let resolve_pipeline = self.resolve_pipeline.as_ref().ok_or("Not initialized")?;

        let prepare_bind_group = bind_group(
            ctx.device,
            &self.prepare_bind_group_layout,
            &[
                wgpu::BindingResource::Buffer(ctx.accumulator.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(ctx.sample_count_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(ctx.buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(buffer_a.as_entire_buffer_binding()),
            ],
        );

        let ping_pong = |src: &wgpu::Buffer, dst: &wgpu::Buffer| {
            bind_group(
                ctx.device,
                &self.atrous_bind_group_layout,
                &[
                    wgpu::BindingResource::Buffer(src.as_entire_buffer_binding()),
                    wgpu::BindingResource::Buffer(dst.as_entire_buffer_binding()),
                    wgpu::BindingResource::Buffer(ctx.gbuffer.as_entire_buffer_binding()),
                ],
            )
        };
        let a_to_b = ping_pong(buffer_a, buffer_b);
        let b_to_a = ping_pong(buffer_b, buffer_a);

        let groups_x = self.width.div_ceil(8);
        let groups_y = self.height.div_ceil(8);

        // Colour and variance into A, variance pre-filtered from A into B.
        add_compute_pass_2d(
            ctx.encoder,
            prepare_pipeline,
            &prepare_bind_group,
            groups_x,
            groups_y,
        );
        add_compute_pass_2d(ctx.encoder, prefilter_pipeline, &a_to_b, groups_x, groups_y);

        // Iteration 0 reads B, and they alternate from there, so the result ends
        // up in A for an odd iteration count and in B for an even one.
        for (i, pipeline) in self.atrous_pipelines.iter().enumerate() {
            let group = if i % 2 == 0 { &b_to_a } else { &a_to_b };
            add_compute_pass_2d(ctx.encoder, pipeline, group, groups_x, groups_y);
        }
        let result = if self.iterations % 2 == 1 {
            buffer_a
        } else {
            buffer_b
        };

        // Blend the filtered image back over the original in proportion to how
        // noisy each pixel still is, so a converged render passes through close
        // to untouched. See denoise_resolve.wgsl for why the variance guidance
        // inside the filter is not enough on its own.
        let resolve_bind_group = bind_group(
            ctx.device,
            &self.prepare_bind_group_layout,
            &[
                wgpu::BindingResource::Buffer(ctx.accumulator.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(ctx.sample_count_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(result.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(ctx.buffer.as_entire_buffer_binding()),
            ],
        );
        add_compute_pass_2d(
            ctx.encoder,
            resolve_pipeline,
            &resolve_bind_group,
            groups_x,
            groups_y,
        );

        Ok(())
    }
}
