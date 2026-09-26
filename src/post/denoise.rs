//! Post-processor for removing Monte Carlo noise

use crate::post::{PIXEL_SIZE, PostProcessContext, PostProcessor};
use crate::util::tone_map::ToneMapper;
use crate::util::wgpu_util::{
    add_compute_pass_2d, bind_group, bind_group_layout, compute_pipeline,
    compute_pipeline_with_entry, shader_module_with_luminance, storage_binding,
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
/// how much of the remaining noise would be **visible on screen** -- through the
/// same tone curve and gamma the readback applies, reaching full strength at
/// `FULL_STRENGTH_GRAIN` code values. Measured against a converged reference
/// on the test scene, it cuts linear RMSE by half at 8 samples per pixel and by
/// about a quarter at 64. Displayed grain on a Cornell box, in code values:
///
/// ```text
///                2 spp   8 spp  32 spp  128 spp
/// raw           15.366   9.428   5.645    3.121
/// denoised       2.066   1.739   1.426    1.060
/// ```
///
/// The cost of judging visibility rather than convergence is that "converged"
/// no longer means "untouched": at 2000 samples per pixel the filter still moves
/// the test scene by 2.4% of linear RMSE, though half the frame moves by under
/// half a code value. See `denoise_resolve.wgsl`.
///
/// Fireflies are handled apart from the filter, which cannot remove one on its
/// own -- a firefly is an edge by every measure such a filter has. The fade's
/// denominator is the lower of the pixel's own luminance and its neighbourhood
/// level, and the variance pre-pass also does outlier rejection for the extreme
/// tail and the below-two-samples case the fade never runs on. Counting pixels
/// more than 20 of 255 brighter than every neighbour on a Cornell box, that
/// leaves at most one at strength 5, and 129 of a raw render's 11780 at
/// strength 1. See the block comments on `level` in `denoise_resolve.wgsl` and
/// on `prefilter_variance` in `denoise_atrous.wgsl`.
///
/// Costs eight compute dispatches, run once on the finished image: 6.6 ms at
/// 800x600 on a Radeon RX 5700 XT, against 52 ms for the render itself at 16
/// samples per pixel. The cost is per pixel and independent of sample count, so
/// it matters less the longer the render.
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
    /// The display transform the resolve pass judges residual noise through.
    /// See [`DenoisePostProcessor::with_tone_mapper`].
    tone_mapper: ToneMapper,

    prepare_module: wgpu::ShaderModule,
    atrous_module: wgpu::ShaderModule,
    /// Built in `initialize` rather than at construction, because the tone
    /// curve is spliced into its source and `with_tone_mapper` may still change
    /// it after `new` has returned.
    resolve_module: Option<wgpu::ShaderModule>,

    prepare_bind_group_layout: wgpu::BindGroupLayout,
    atrous_bind_group_layout: wgpu::BindGroupLayout,
    resolve_bind_group_layout: wgpu::BindGroupLayout,

    prepare_pipeline: Option<wgpu::ComputePipeline>,
    prefilter_pipeline: Option<wgpu::ComputePipeline>,
    resolve_pipeline: Option<wgpu::ComputePipeline>,
    /// One per à-trous iteration, differing in their `step_width` override, so
    /// the tap spacing is a compile-time constant, and in the
    /// `variance_correlation` that goes with it.
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

/// How much grain, in code values of the final 0-255 image, the filter must be
/// facing before it commits to its result whole -- at a `strength` of 1.
///
/// The fade in `denoise_resolve.wgsl` is a linear ramp, so the grain it leaves
/// behind is `g * (1 - g / tau)` for a displayed noise level `g`. That peaks at
/// `g = tau / 2` and is worth `tau / 4` there, whatever the sample count. So
/// this constant bounds what the filter is allowed to leave rather than setting
/// how hard it runs: at 2 code values the worst case anywhere in the image is
/// half a code value, below the quantisation step of the image it is written
/// into.
///
/// 2 is also about where grain stops being visible: on a flat mid-grey wall it
/// is roughly 0.8% contrast, against the ~1% the eye resolves in a gradient.
/// Adaptive sampling retires a pixel at 0.42 to 0.75 code values, so full
/// strength sits a factor of about three above where the sampler stops caring,
/// and the ramp's tail covers the gap.
///
/// Chosen on `denoise_display_sweep`, which is what to re-run to change it.
/// `strength` scales this and `sigma_colour` together, so moving the threshold
/// alone means editing this constant.
const FULL_STRENGTH_GRAIN: f64 = 2.0;

/// Per-iteration repair of the variance recursion in `denoise_atrous.wgsl`,
/// which assumes the taps it is averaging are independent of one another.
///
/// They are, on the first iteration: each pixel holds its own Monte Carlo mean.
/// They are not on any iteration after that, because every pixel the second
/// iteration reads was itself an average over a neighbourhood overlapping its
/// neighbours'. Tracking `sum(w^2 * var)` through that under-reports the
/// variance, compounding once per iteration.
///
/// How badly is exactly computable. Each iteration convolves with the 5x5
/// B-spline `h = (1,4,6,4,1)/16` at a tap spacing of `2^i`, and `h` is
/// `((1 + z)/2)^4`, so after `N` iterations the composite kernel is
///
/// ```text
/// prod_{i<N} ((1 + z^(2^i))/2)^4 = [(1 + z + ... + z^(M-1)) / M]^4,  M = 2^N
/// ```
///
/// -- the four-fold self-convolution of a *box* of width `M`, whose `sum k^2`
/// falls as `0.4886 / M` rather than as `sum h^2` to the power `N`:
///
/// ```text
/// after N iterations      1       2       3       4       5
/// true sum k^2 (2-D)   7.5e-2  1.5e-2  3.6e-3  9.0e-4  2.2e-4
/// tracked              7.5e-2  5.6e-3  4.2e-4  3.1e-5  2.3e-6
/// under-reported by      1.0     2.7     8.7    28.8    96.1
/// ```
///
/// A tolerance is a square root of that, so uncorrected the fifth iteration is
/// ten times too tight and the widest passes do essentially nothing.
///
/// Each entry is `sum k^2(i+1) / (sum k^2(i) * sum h^2)`, in 2-D. With the
/// correction in, the luminance tolerance shrinks by 0.273 across the first
/// iteration and then by 0.452, 0.489, 0.497, 0.499 -- halving per iteration
/// from the second onward, which is Dammertz's `sigma / 2^i` schedule arrived at
/// rather than assumed.
///
/// The derivation is for the unweighted kernel, so applying it whole where the
/// edge stops have narrowed the kernel overstates it -- 18% of the specular
/// scene's RMSE, concentrated on the mirror and the caustic.
/// `denoise_atrous.wgsl` therefore fades each factor in on how much of the
/// kernel survived its weights. The specular scene stays the control here.
const VARIANCE_CORRELATION: [f64; MAX_ITERATIONS as usize] =
    [1.0, 2.7272, 3.1961, 3.3072, 3.3346, 3.3414, 3.3431, 3.3435];

impl DenoisePostProcessor {
    /// Creates a new denoiser.
    ///
    /// # Arguments
    /// * `strength` How hard to filter, from 0 to 10. 1 is the tuned default,
    ///   below 1 keeps more detail and more noise, above 1 blurs harder. It
    ///   scales the luminance tolerance the edge stop allows (as its square
    ///   root, so the top of the range stays usable) and the threshold at which
    ///   the fade commits to the filtered result. 0 is exactly the identity;
    ///   10 puts the fade's threshold below the final image's quantisation
    ///   step, so the filtered result is taken essentially whole.
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

        let prepare_module = shader_module_with_luminance(
            device,
            "denoise_prepare.wgsl",
            include_str!("denoise_prepare.wgsl"),
        );
        let atrous_module = shader_module_with_luminance(
            device,
            "denoise_atrous.wgsl",
            include_str!("denoise_atrous.wgsl"),
        );

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
            tone_mapper: ToneMapper::default(),
            prepare_module,
            atrous_module,
            resolve_module: None,
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

    /// Sets the display transform the resolve pass judges residual noise
    /// through. Defaults to [`ToneMapper::default`].
    ///
    /// The denoiser fades itself out on how visible the remaining noise would
    /// be *on screen*, so it has to know the curve the image will be shown
    /// through. That curve is chosen by whoever calls
    /// [`buffer_to_image`](crate::util::wgpu_util::buffer_to_image), so this is
    /// how the two are kept in step. Getting it wrong is bounded -- at linear
    /// 0.64 the ACES slope is 69 code values per unit radiance against a plain
    /// gamma's 160, a factor of 2.3 -- but there is no reason to.
    pub fn with_tone_mapper(mut self, tone_mapper: ToneMapper) -> Self {
        self.tone_mapper = tone_mapper;
        self
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

        // The paper's defaults, with only the luminance falloff exposed, in one
        // place so every pipeline built below agrees on them.
        //
        // SVGF's sigma_l is 4; half that measured best here across 8, 64 and
        // 2000 samples per pixel, since SVGF filters a reprojected temporal
        // estimate whose variance is far shakier than the exact Welford one the
        // sample loop hands us.
        //
        // Sub-linear in `strength`. Two independent noisy pixels differ by
        // something with a standard deviation of sqrt(2) * SE, so a sigma of 2
        // accepts taps within 1.4 standard deviations of that difference and
        // SVGF's 4 within 2.8. Linear scaling put a strength of 10 at 20, or
        // fourteen standard deviations, which accepts every gradient, soft
        // shadow and colour bleed in a Cornell box. A square root is monotone
        // with fixed points at exactly 0 and 1, so the identity and the tuned
        // default are untouched and 10 gives 6.3.
        let sigmas = [
            ("sigma_colour", 2.0 * self.strength.sqrt()),
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
            // Outlier rejection, deliberately not scaled by `strength`; see
            // prefilter_variance in denoise_atrous.wgsl.
            //
            // At a strength of 0 the luminance tolerance collapses and every
            // non-centre tap goes to zero weight. The despeckle reads no sigma,
            // so it has to be switched off explicitly or "off" would stop
            // meaning off. test_denoise_strength_zero_is_the_identity guards
            // this.
            ("despeckle_enabled", f64::from(self.strength != 0.)),
        ];

        let constants = |step_width: u32, variance_correlation: f64| {
            let mut c = dimensions.to_vec();
            c.extend_from_slice(&sigmas);
            c.push(("step_width", step_width as f64));
            c.push(("variance_correlation", variance_correlation));
            c
        };

        self.prefilter_pipeline = Some(compute_pipeline_with_entry(
            device,
            &self.atrous_bind_group_layout,
            &self.atrous_module,
            "prefilter_variance",
            // No correlation to correct: the pre-pass pools per-pixel variance
            // estimates with unsquared weights and never filters the image.
            &constants(1, 1.),
        ));

        // The tone curve goes ahead of the shader's own source, so nothing
        // relies on WGSL resolving a call to a function declared later.
        // `ToneMapper::wgsl` emits one free function with no bindings or entry
        // point, which is what makes it safe to concatenate.
        let resolve_module = self.resolve_module.insert(shader_module_with_luminance(
            device,
            "denoise_resolve.wgsl",
            &format!(
                "{}\n{}",
                self.tone_mapper.wgsl(),
                include_str!("denoise_resolve.wgsl")
            ),
        ));

        // Passed as the reciprocal so that a strength of 0 is exactly 0 rather
        // than an infinity narrowed through an f64 -> f32 override, which makes
        // the resolve pass the identity by arithmetic rather than by luck.
        let mut resolve_constants = dimensions.to_vec();
        resolve_constants.push(("grain_blend_gain", self.strength / FULL_STRENGTH_GRAIN));

        self.resolve_pipeline = Some(compute_pipeline(
            device,
            &self.resolve_bind_group_layout,
            resolve_module,
            &resolve_constants,
        ));

        self.atrous_pipelines = (0..self.iterations)
            .map(|i| {
                compute_pipeline(
                    device,
                    &self.atrous_bind_group_layout,
                    &self.atrous_module,
                    &constants(1 << i, VARIANCE_CORRELATION[i as usize]),
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
            ctx.timer.as_deref_mut(),
            "denoise_prepare",
        );
        add_compute_pass_2d(
            ctx.encoder,
            prefilter_pipeline,
            &a_to_b,
            groups_x,
            groups_y,
            ctx.timer.as_deref_mut(),
            "denoise_prefilter",
        );

        // Iteration 0 reads B, and they alternate from there, so the result ends
        // up in A for an odd iteration count and in B for an even one.
        for (i, pipeline) in self.atrous_pipelines.iter().enumerate() {
            let group = if i % 2 == 0 { &b_to_a } else { &a_to_b };
            // One label for every iteration: they are the same shader at a
            // different step width, and the report folds them into a single
            // line with a count.
            add_compute_pass_2d(
                ctx.encoder,
                pipeline,
                group,
                groups_x,
                groups_y,
                ctx.timer.as_deref_mut(),
                "denoise_atrous",
            );
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
            ctx.timer.as_deref_mut(),
            "denoise_resolve",
        );

        Ok(())
    }

    /// Yes. A camera drag restarts the accumulation on every frame, so the
    /// image a viewer is dragging is a one-sample image -- the noisiest the
    /// renderer ever produces and the only one the filter cannot be left out
    /// of. See [`RenderConfig::preview`], which is what pays for it.
    ///
    /// [`RenderConfig::preview`]: crate::renderer::RenderConfig::preview
    fn preview(&self) -> bool {
        true
    }

    /// Only with [`DenoiseGuide::Full`]. `ColorOnly` reads nothing but the
    /// image and its variance, so it also spares the tracer the guide ray.
    fn needs_guide(&self) -> bool {
        self.guide == DenoiseGuide::Full
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::wgpu_util::get_wgpu_device_and_queue;

    /// The resolve shader is assembled from two sources at run time, so a
    /// splice that does not parse is not a compile error -- it surfaces as a
    /// panic out of wgpu's uncaptured-error handler mid-render, for whichever
    /// tone mapper the user picked. So build it for every curve in the enum.
    ///
    /// The sibling of `wgsl_matches_the_cpu_curve` in `util/tone_map.rs`: that
    /// one pins what the emitted source computes, this one that it compiles.
    #[test]
    fn the_resolve_shader_compiles_against_every_tone_mapper() {
        let (device, queue) = get_wgpu_device_and_queue();

        for mapper in [
            ToneMapper::Aces,
            ToneMapper::PbrNeutral,
            ToneMapper::Reinhard { white_point: 4. },
            ToneMapper::Clamp,
        ] {
            let mut denoiser = DenoisePostProcessor::new(1., None, None, device)
                .unwrap()
                .with_tone_mapper(mapper);

            denoiser.initialize(device, queue, 64, 32);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

            assert!(
                denoiser.resolve_pipeline.is_some(),
                "resolve pipeline missing for {:?}",
                mapper
            );
        }
    }
}
