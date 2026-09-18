//! Post-processor for removing Monte Carlo noise

use crate::post::{PIXEL_SIZE, PostProcessContext, PostProcessor};
use crate::util::wgpu_util::{
    add_compute_pass_2d, bind_group, bind_group_layout, compute_pipeline,
    compute_pipeline_with_entry, storage_binding,
};
use std::error::Error;
use wgpu::BufferUsages;

/// Which guide channels the edge-stopping functions are allowed to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DenoiseGuide {
    /// Use the renderer's guide albedo, normal, distance and specular depth.
    /// What you want unless you are measuring what the guide buys.
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
/// reference on the test scene, it cuts linear RMSE by about 45% at 8 samples
/// per pixel, by about a fifth at 64, and changes a 2000-sample image by 0.2%.
///
/// An edge-avoiding filter cannot remove a firefly on its own -- a firefly is an
/// edge by every measure such a filter has -- so what decided a firefly's fate
/// was the fade above, and it leaked at two scales. The fade's denominator was
/// the pixel's own luminance, which biases it to preserve noise that moved a
/// pixel up and remove noise that moved it down; it is now the lower of that and
/// the pixel's neighbourhood level. And the variance pre-pass now also does
/// outlier rejection, which covers the extreme tail and the below-two-samples
/// case the fade never runs on at all. See the block comments on `level` in
/// `denoise_resolve.wgsl` and on `prefilter_variance` in `denoise_atrous.wgsl`.
///
/// Measured on a Cornell box through the display transform, counting pixels more
/// than 20 of 255 brighter than every neighbour, the denoiser used to leave
/// 1000-3000 of them at every sample count from 2 upwards. It now leaves 7-46 at
/// strength 5 and under 4% of the raw render's at strength 1.
///
/// Costs eight compute dispatches, run once on the finished image: 6.6 ms at
/// 800x600 on a Radeon RX 5700 XT, against 52 ms for the render itself at 16
/// samples per pixel. The cost is per pixel and independent of sample count, so
/// it matters less the longer the render. Outlier rejection adds no dispatch of
/// its own: it needs the same guide-weighted neighbourhood the variance pre-pass
/// was already gathering.
///
/// Place this first in [`crate::renderer::RenderConfig::post_processors`].
/// Denoising a bloomed image blurs the bloom; bloom applied to a denoised image
/// is what you want.
///
/// The guide follows the specular chain, so on a mirror or a glass surface it
/// describes what is seen in or through it rather than the surface itself, and
/// the reflected and refracted image survives the filter. Two limitations are
/// left: a chain longer than `GUIDE_MAX_SPECULAR` (6) in
/// `renderer/ray_trace.wgsl` falls back to describing the specular surface it
/// stalled on, and with a wide aperture the guide ray is the pixel-centre ray,
/// so it describes the point in focus rather than the defocused average the
/// samples actually see.
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
    resolve_bind_group_layout: wgpu::BindGroupLayout,

    prepare_pipeline: Option<wgpu::ComputePipeline>,
    prefilter_pipeline: Option<wgpu::ComputePipeline>,
    resolve_pipeline: Option<wgpu::ComputePipeline>,
    /// One per à-trous iteration, differing only in their `step_width`
    /// override, so the tap spacing is a compile-time constant.
    atrous_pipelines: Vec<wgpu::ComputePipeline>,

    buffer_a: Option<wgpu::Buffer>,
    buffer_b: Option<wgpu::Buffer>,
    /// The variance the pre-filter pooled, kept aside from the image the à-trous
    /// iterations then filter, so the resolve pass can still read it.
    variance_buffer: Option<wgpu::Buffer>,
    /// The luminance of each pixel's guide-weighted neighbourhood, measured by
    /// the pre-filter and kept aside for the same reason.
    level_buffer: Option<wgpu::Buffer>,
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
    /// * `guide` Which guide channels may guide the filter. If not specified,
    ///   defaults to [`DenoiseGuide::Full`].
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
                storage_binding(true, 16),  // guide
                storage_binding(false, 4),  // pooled variance, written by the pre-filter
                storage_binding(false, 16), // working image, despeckled by the pre-filter
                storage_binding(false, 4),  // neighbourhood level, written by the pre-filter
            ],
        );

        let resolve_bind_group_layout = bind_group_layout(
            device,
            &[
                storage_binding(true, 4),   // per-pixel sample count
                storage_binding(true, 16),  // filtered image
                storage_binding(false, 16), // working image
                storage_binding(true, 4),   // pooled variance
                storage_binding(true, 4),   // neighbourhood level
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
            resolve_bind_group_layout,
            prepare_pipeline: None,
            prefilter_pipeline: None,
            resolve_pipeline: None,
            atrous_pipelines: Vec::new(),
            buffer_a: None,
            buffer_b: None,
            variance_buffer: None,
            level_buffer: None,
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
            ("sigma_specular", 1.),
            (
                "use_guide",
                match self.guide {
                    DenoiseGuide::Full => 1.,
                    DenoiseGuide::ColorOnly => 0.,
                },
            ),
            // Outlier rejection, deliberately *not* scaled by `strength`. The
            // whole reason fireflies survived was that the stage which decided
            // their fate read no sigma at all, so turning the knob did nothing;
            // re-coupling the two would reintroduce exactly that. See
            // prefilter_variance in denoise_atrous.wgsl.
            //
            // At a strength of 0 the luminance tolerance collapses to 1e-8 and
            // every non-centre tap goes to zero weight, which is as close to
            // the identity as this filter gets. The despeckle reads no sigma,
            // so it would go on clamping right through that unless it is
            // switched off too -- and "off" would quietly stop meaning off.
            // test_denoise_strength_zero_is_the_identity is the guard.
            ("despeckle_enabled", f64::from(self.strength != 0.)),
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

        self.resolve_pipeline = Some(compute_pipeline(
            device,
            &self.resolve_bind_group_layout,
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

        let scalar = |label| {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (width * height) as u64 * 4,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            }))
        };
        self.variance_buffer = scalar("Denoise Variance Buffer");
        self.level_buffer = scalar("Denoise Level Buffer");
    }

    fn post_process(&self, ctx: &mut PostProcessContext) -> Result<(), Box<dyn Error>> {
        let buffer_a = self.buffer_a.as_ref().ok_or("Not initialized")?;
        let buffer_b = self.buffer_b.as_ref().ok_or("Not initialized")?;
        let variance_buffer = self.variance_buffer.as_ref().ok_or("Not initialized")?;
        let level_buffer = self.level_buffer.as_ref().ok_or("Not initialized")?;
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
                    wgpu::BindingResource::Buffer(variance_buffer.as_entire_buffer_binding()),
                    wgpu::BindingResource::Buffer(ctx.buffer.as_entire_buffer_binding()),
                    wgpu::BindingResource::Buffer(level_buffer.as_entire_buffer_binding()),
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
            &self.resolve_bind_group_layout,
            &[
                wgpu::BindingResource::Buffer(ctx.sample_count_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(result.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(ctx.buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(variance_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(level_buffer.as_entire_buffer_binding()),
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
