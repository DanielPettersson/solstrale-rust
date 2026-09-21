//! The renderer takes a [`Scene`] as input, renders it and reports [`RenderProgress`]

use crate::hittable::Hittable;
use crate::post::PostProcessor;
use std::collections::VecDeque;
use std::error::Error;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use crate::camera::{Camera, CameraConfig};
use crate::geo::vec3::Vec3;
use crate::hittable::Hittables;
use crate::post::PostProcessors;
use crate::renderer::gpu_data::{GpuCamera, GpuRenderConfig, MAT_BLEND, MAT_DIELECTRIC, MAT_METAL};
use crate::renderer::scene_flattener::{SceneData, flatten_scene};
use crate::util::gpu_timing::{GpuTimer, report_dispatch};
use crate::util::wgpu_util::{
    add_compute_pass_2d, bind_group, bind_group_layout, compute_pipeline, sampler_binding,
    shader_module_with_luminance, storage_binding, texture_binding, uniform_binding,
};
use image::{DynamicImage, Rgb, RgbImage};
use simple_error::SimpleError;
use wgpu::BufferUsages;

pub mod dielectric_energy;
pub mod gpu_data;
#[cfg(test)]
mod sampler_test;
pub mod scene_flattener;
#[cfg(test)]
mod specialisation_test;

/// How the tracer is compiled when it is not compiled as a reference.
///
/// Neither is a [`RenderConfig`] field. Next-event estimation does not change
/// what the image converges to, only how fast; the firefly clamp is a bias the
/// renderer takes deliberately and documents in LIMITATIONS.md. Neither is a
/// knob a caller should be reaching for, and the one thing that does reach for
/// them is [`Renderer::bsdf_only_reference`]. See issue #47 for the depth
/// cutoff that will eventually replace the first of them.
const SHIPPED_ESTIMATOR: Estimator = Estimator {
    nee_enabled: true,
    clamping_threshold: 10.,
};

/// The two things that decide what a sample carries, as opposed to how fast it
/// is computed. Everything in [`Specialisation`] leaves the samples alone; both
/// of these change them, which is why they are apart from it.
#[derive(Clone, Copy)]
struct Estimator {
    /// Whether direct lighting is estimated with an explicit shadow ray and
    /// weighed against the BSDF sample, or left for the BSDF sample to find.
    nee_enabled: bool,
    /// Ceiling on the indirect radiance one sample may carry.
    /// `ray_trace.wgsl` documents what the shipped value buys.
    clamping_threshold: f32,
}

/// What the tracer is compiled against for one scene.
///
/// Every field is a fact that holds for the life of a [`Renderer`], so it
/// reaches the shader as an override constant rather than as a uniform: naga
/// substitutes overrides before it emits SPIR-V, so a `false` here deletes the
/// branch it guards instead of merely making it predictable. `Renderer::new`
/// already builds the module and its one pipeline per render, so the only cost
/// is a shader-cache miss the first time a given combination is seen.
///
/// Each flag removes a branch the scene could never have taken, which is why
/// none of them changes a single sample -- the invariant `specialisation_test`
/// holds the whole thing to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Specialisation {
    has_spheres: bool,
    has_quads: bool,
    /// See [`SceneData::prim_refs_are_identity`], which this is taken from.
    identity_prim_refs: bool,
    has_blends: bool,
    has_metal: bool,
    has_dielectrics: bool,
    /// Whether any dielectric is rough, which is what the microfacet arm costs
    /// its occupancy for. See the override's note in `ray_trace.wgsl`.
    has_rough_dielectrics: bool,
    has_textures: bool,
    has_normal_maps: bool,
    light_count: u32,
}

impl Specialisation {
    fn from_scene_data(data: &SceneData) -> Self {
        // Blend children are flattened into `materials` by `add_material`, so
        // one pass over it sees the materials inside a blend as well as the
        // ones a primitive names directly.
        let any = |f: fn(&gpu_data::Material) -> bool| data.materials.iter().any(f);

        Specialisation {
            has_spheres: !data.spheres.is_empty(),
            has_quads: !data.quad_pos.is_empty(),
            identity_prim_refs: data.prim_refs_are_identity,
            has_blends: any(|m| m.mat_type == MAT_BLEND),
            has_metal: any(|m| m.mat_type == MAT_METAL),
            has_dielectrics: any(|m| m.mat_type == MAT_DIELECTRIC),
            // Any roughness at all, not a threshold: the shader still routes a
            // roughness below `GGX_ALPHA_MIN` down the Dirac path, so this only
            // has to be conservative, and a duplicated cutoff on this side is a
            // thing to keep in step for no gain.
            has_rough_dielectrics: any(|m| m.mat_type == MAT_DIELECTRIC && m.fuzz > 0.),
            has_textures: any(|m| m.texture_index >= 0),
            has_normal_maps: any(|m| m.normal_texture_index >= 0),
            light_count: data.lights.len() as u32,
        }
    }

    /// The tracer with nothing stripped: every branch left in, which is what
    /// the defaults in `ray_trace.wgsl` already are.
    #[cfg(test)]
    fn unspecialised(light_count: u32) -> Self {
        Specialisation {
            has_spheres: true,
            has_quads: true,
            identity_prim_refs: false,
            has_blends: true,
            has_metal: true,
            has_dielectrics: true,
            has_rough_dielectrics: true,
            has_textures: true,
            has_normal_maps: true,
            light_count,
        }
    }

    /// The override constants `ray_trace.wgsl` declares, in the form
    /// `PipelineCompilationOptions` wants them. A `bool` override takes any
    /// non-zero value as true.
    fn constants(
        &self,
        width: u32,
        height: u32,
        low_discrepancy: bool,
        estimator: Estimator,
        writes_guide: bool,
    ) -> Vec<(&'static str, f64)> {
        let flag = |b: bool| if b { 1. } else { 0. };
        vec![
            ("width", width as f64),
            ("height", height as f64),
            ("light_count", self.light_count as f64),
            ("has_spheres", flag(self.has_spheres)),
            ("has_quads", flag(self.has_quads)),
            ("identity_prim_refs", flag(self.identity_prim_refs)),
            ("has_blends", flag(self.has_blends)),
            ("has_metal", flag(self.has_metal)),
            ("has_dielectrics", flag(self.has_dielectrics)),
            ("has_rough_dielectrics", flag(self.has_rough_dielectrics)),
            ("has_textures", flag(self.has_textures)),
            ("has_normal_maps", flag(self.has_normal_maps)),
            ("nee_enabled", flag(estimator.nee_enabled)),
            ("clamping_threshold", estimator.clamping_threshold as f64),
            ("low_discrepancy", flag(low_discrepancy)),
            ("writes_guide", flag(writes_guide)),
        ]
    }
}

///Input to the ray tracer for how the image should be rendered
#[derive(Clone)]
pub struct RenderConfig {
    /// Width in pixels of the rendered image
    pub width: usize,
    /// Height in pixels of the rendered image
    pub height: usize,
    /// Number of times each pixel should be sampled
    pub samples_per_pixel: u32,
    /// Maximum number of ray bounces before a path is cut off.
    pub max_depth: u32,
    /// Samples traced per GPU dispatch, used as the starting point and the
    /// upper bound for the batch size.
    ///
    /// Larger batches amortise dispatch overhead and collapse the per-sample
    /// read-modify-write of the accumulation buffer, at the cost of coarser
    /// progress reporting and slower response to camera changes. The renderer
    /// tunes the actual size down from here to keep a single dispatch within
    /// `TARGET_DISPATCH`.
    pub samples_per_batch: u32,
    /// Minimum samples a pixel must accumulate before adaptive sampling can
    /// consider it converged and stop sampling it further.
    ///
    /// Skipping is effectively permanent -- a pixel that stops sampling can no
    /// longer revise the variance estimate that silenced it -- so this wants to
    /// be high enough that the estimate is trustworthy. Path-traced luminance
    /// is heavy-tailed, and a couple of dozen samples is not much to judge it
    /// on. Set above `samples_per_pixel` to disable adaptive sampling.
    pub min_samples_per_pixel: u32,
    /// Relative standard-error threshold below which a pixel is considered
    /// converged and skipped by adaptive sampling. Lower is stricter: less
    /// noise tolerated, less speedup gained.
    ///
    /// This is the render's noise floor, not merely a speed knob. Skipping is
    /// permanent, so a pixel retires holding whatever error it had when it
    /// passed this test, and no amount of `samples_per_pixel` can push it
    /// lower -- more samples only retire more pixels sooner. Measured on the
    /// 200x100 test scene against a 32k-spp reference, the relative RMSE at
    /// the old default of 0.05 sat at 0.059 / 0.057 / 0.058 for 1024 / 4096 /
    /// 16384 spp: flat, while the same renders with adaptive sampling off ran
    /// 0.037 / 0.018 / 0.007, halving per 4x samples as Monte Carlo should.
    ///
    /// 0.05 is a 5% relative luminance error, well above the ~1% contrast the
    /// eye resolves in a smooth gradient, and it reads as grain that never
    /// sands out. The denoiser used to agree with it and leave it there, since
    /// `denoise_resolve` faded the filter on this same relative error and
    /// reached full strength only at 0.4, so a pixel retired at 0.05 took an
    /// eighth of the filtered result. That is no longer true, and the division
    /// of labour is better for it: that pass now fades on how visible the
    /// residue would be rather than on how converged the estimate is, and even
    /// at the 0.01 default a retired pixel still carries 0.4 to 0.8 code values
    /// of grain. Adaptive sampling stops spending samples; the denoiser cleans
    /// up what it left.
    ///
    /// At 0.01 the floor lands below what 4096 spp reaches anyway (0.021
    /// against 0.020 with adaptive sampling off), for roughly 2x the time of
    /// 0.05. Going stricter than that buys nothing. Adaptive sampling earns
    /// its keep in the interactive viewport, where the alternative is a
    /// blurrier frame, rather than in a long offline render.
    pub variance_threshold: f32,
    /// Whether samples are drawn from an Owen-scrambled Sobol sequence rather
    /// than from white noise.
    ///
    /// Resolved into the tracing pipeline as an override constant, so switching
    /// it off costs nothing at run time. It changes nothing about what the
    /// converged image is -- only how many samples it takes to get there -- and
    /// is on by default. Off is for measuring what it buys;
    /// `sampler_convergence_sweep` is what does the measuring.
    ///
    /// One second-order cost, recorded in LIMITATIONS.md: low-discrepancy
    /// samples are negatively correlated, so the Welford estimator behind
    /// adaptive sampling overstates the variance of the mean and retires each
    /// pixel a little later than it needs to.
    pub low_discrepancy: bool,
    /// Distinguishes two otherwise identical renders of the same scene.
    ///
    /// Feeds the initial `restart_index`, which is mixed into every pixel's
    /// sampler seed. Equal seeds give bit-identical images; different seeds
    /// give the same image with independent noise.
    ///
    /// This matters for reference renders. A converged reference that shares
    /// its seed with the render being measured against it shares the sample
    /// stream too -- with `low_discrepancy` on, an 8-sample render is literally
    /// a sub-net of a 4000-sample reference -- so the error between them is
    /// correlated and biased low. Give references a different seed.
    pub seed: u32,
    /// Post processor to apply to the rendered image
    pub post_processors: Vec<PostProcessors>,
    /// Whether the post-processors that declare themselves preview processors
    /// run on every batch, or only on the last one.
    ///
    /// Off, the default: the chain runs once, on the final batch, as it always
    /// did. The buffer published on [`RenderProgress`] still follows the
    /// accumulation, it is just raw until the end.
    ///
    /// On: every batch gets a processed image. **Turn this on for an
    /// interactive viewport**, which is the case it exists for -- a camera drag
    /// restarts the accumulation on every frame and so never reaches a last
    /// batch at all, which made the one regime a denoiser is built for the one
    /// regime it never ran in. A drag frame on `create_test_scene` at 800x600
    /// goes from 2.7 ms to 4.8 ms and stops being a one-sample image; see
    /// `interactive_restart_frame_cost` in `tests/interactive_test.rs`.
    ///
    /// Default off because the cost is charged per batch whether or not anyone
    /// is looking. The only expensive preview processor is the denoiser,
    /// measured on a Radeon RX 5700 XT at 800x600 as 1.8 ms of GPU time per
    /// batch against the 12 ms a batch of samples is sized to fill, so a
    /// denoised batch render would pay something like a sixth of its time for
    /// previews nobody reads. A viewport renders one batch per frame and reads
    /// every one of them; a batch render reads the last.
    ///
    /// Which processors those are is [`crate::post::PostProcessor::preview`]:
    /// the denoiser and the saturation grade, not bloom. Nothing about the
    /// final image depends on this. The chain always runs on a fresh copy of
    /// the accumulator, so running it four times over a render and running it
    /// once produce the same last batch.
    pub preview: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        RenderConfig {
            width: 300,
            height: 200,
            samples_per_pixel: 50,
            max_depth: 10,
            samples_per_batch: 4,
            min_samples_per_pixel: 32,
            variance_threshold: 0.01,
            low_discrepancy: true,
            seed: 0,
            post_processors: vec![],
            preview: false,
        }
    }
}

/// Contains all information needed to render an image
pub struct Scene {
    /// World is the hittable objects in the scene
    pub world: Hittables,
    /// A camera for defining the view of the world
    pub camera: CameraConfig,
    /// Background color of the scene
    pub background_color: Vec3,
    /// Render configuration
    pub render_config: RenderConfig,
}

/// Progress reported back to the caller of the raytrace function
pub struct RenderProgress {
    /// progress is reported between 0 -> 1 and represents a percentage of completion
    pub progress: f64,
    /// Current speed of rendering in number of frames per second
    pub fps: Option<f64>,
    /// Estimated time left until rendering is complete
    pub estimated_time_left: Duration,
    /// Output buffer containing the image data
    pub output_buffer: wgpu::Buffer,
}

/// Wall clock budget for the *work* a single dispatch adds.
///
/// A caller that renders interactively hands us the same device and queue its
/// user interface draws on, so a dispatch that overruns a display frame is a
/// dropped frame for whoever is waiting behind us. Roughly one vsync interval.
///
/// This budgets the sample-proportional part of a dispatch only -- see
/// [`DispatchCost`] for why the fixed part is deliberately left out of it.
const TARGET_DISPATCH: Duration = Duration::from_millis(12);

/// Ceiling on the adaptive batch size, so a very cheap scene does not end up
/// reporting progress once a second.
const MAX_BATCH: u32 = 64;

/// What a dispatch costs in wall clock time: a fixed per-dispatch term plus a
/// per-sample one.
///
/// Both terms are needed, because the fixed one is not ours. A caller that
/// renders interactively shares its queue with a vsync-throttled presenter, so
/// our submission is regularly serialised behind a swapchain acquire and a
/// dispatch takes a display frame longer than the work in it -- measured at
/// 1384x784 on a 60 Hz display, one batch of 8 samples costs 22.4 ms against a
/// marginal cost of 2.5 ms per sample.
///
/// Dividing the whole dispatch by its batch size, which is what a single
/// milliseconds-per-sample figure amounts to, charges that latency to the
/// samples. Every batch size is then its own fixed point: at a batch of one the
/// measurement above reads 8.3 ms per sample, [`TARGET_DISPATCH`] divided by
/// that asks for a batch of one, and a batch of one is what it stays at --
/// while shrinking the batch is the one thing that cannot make the dispatch
/// shorter, so nothing ever contradicts the estimate. The render then runs at
/// a third of the throughput the same machine reaches when the estimate
/// happens to settle higher instead, which is why the reported speed used to
/// vary several-fold between runs of the same scene.
///
/// Fitting the two terms apart keeps the budget on the work a dispatch adds
/// rather than on the latency it merely waits through.
#[derive(Debug)]
struct DispatchCost {
    /// Recent `(batch, milliseconds)` observations, oldest first.
    history: VecDeque<(f64, f64)>,
    /// Fixed part of a dispatch, in milliseconds.
    overhead: f64,
    /// Marginal cost of one sample, in milliseconds. What the batch size is
    /// tuned against.
    slope: f64,
    /// Moving average of a whole dispatch divided by its batch size: what a
    /// sample costs the caller once the fixed part is shared out over the
    /// batch. Reported and used for the time estimate, never for tuning.
    per_sample: Option<f64>,
}

/// How many observations the fit sees. Long enough to average out a queue that
/// hands us a display frame on one dispatch and not the next, short enough to
/// follow a scene whose cost falls as adaptive sampling retires pixels.
const COST_HISTORY: usize = 16;

/// Spread in batch size, as a variance, below which the two terms cannot be
/// told apart and the fit is not attempted.
const MIN_BATCH_VARIANCE: f64 = 0.2;

/// Floor under a per-sample estimate. Only guards the divisions; a slope this
/// small already asks for [`MAX_BATCH`].
const MIN_SLOPE: f64 = 1e-3;

impl DispatchCost {
    fn new() -> Self {
        DispatchCost {
            history: VecDeque::with_capacity(COST_HISTORY),
            overhead: 0.,
            slope: MIN_SLOPE,
            per_sample: None,
        }
    }

    /// Folds in one completed dispatch.
    fn record(&mut self, batch: u32, elapsed: Duration) {
        let batch = batch as f64;
        let ms = elapsed.as_secs_f64() * 1000.;

        let per_sample = ms / batch;
        self.per_sample = Some(
            self.per_sample
                .map_or(per_sample, |prev| prev * 0.8 + per_sample * 0.2),
        );

        if self.history.len() == COST_HISTORY {
            self.history.pop_front();
        }
        self.history.push_back((batch, ms));

        if self.history.len() == 1 {
            // Nothing to fit a line through yet. Start by charging the whole
            // dispatch to the samples and let the first batch change separate
            // the terms.
            self.slope = per_sample.max(MIN_SLOPE);
            return;
        }

        self.fit();
    }

    /// Spread in the batch sizes the window was measured at.
    fn batch_variance(&self) -> f64 {
        let n = self.history.len() as f64;
        let mean = self.history.iter().map(|(b, _)| b).sum::<f64>() / n;
        self.history
            .iter()
            .map(|(b, _)| (b - mean).powi(2))
            .sum::<f64>()
            / n
    }

    /// Least squares over [`Self::history`], falling back to correcting the
    /// slope alone while the batch size sits still.
    fn fit(&mut self) {
        let n = self.history.len() as f64;
        let mean_batch = self.history.iter().map(|(b, _)| b).sum::<f64>() / n;
        let mean_ms = self.history.iter().map(|(_, ms)| ms).sum::<f64>() / n;
        let variance = self.batch_variance();

        if variance < MIN_BATCH_VARIANCE {
            // One batch size tells us what a dispatch costs there and nothing
            // about how that splits. Charge the drift to the samples and leave
            // the fixed part where the last fit put it, so the estimate at
            // least follows a scene whose cost is moving.
            let residual = mean_ms - (self.overhead + self.slope * mean_batch);
            self.slope = (self.slope + residual / mean_batch).max(MIN_SLOPE);
            return;
        }

        let covariance = self
            .history
            .iter()
            .map(|(b, ms)| (b - mean_batch) * (ms - mean_ms))
            .sum::<f64>()
            / n;

        // A non-positive slope means the dispatch time did not follow the batch
        // size at all over the window: the whole cost is latency, and the batch
        // wants to be as large as the caller tolerates.
        self.slope = (covariance / variance).max(MIN_SLOPE);
        self.overhead = (mean_ms - self.slope * mean_batch).max(0.);
    }

    /// Whether the batch size has to be moved before the fit can say anything.
    ///
    /// A window measured at a single batch size cannot separate the two terms,
    /// and the fallback in [`Self::fit`] then charges everything above the
    /// standing fixed part to the samples -- which reproduces whatever estimate
    /// pinned the batch size there, right or wrong. Nothing else in the loop
    /// moves the batch size, so the model has to ask for the measurement it is
    /// missing.
    fn needs_probe(&self) -> bool {
        self.history.len() == COST_HISTORY && self.batch_variance() < MIN_BATCH_VARIANCE
    }

    /// Batch size whose samples fill [`TARGET_DISPATCH`], given what the
    /// dispatch before it was.
    fn target_batch(&self, current: u32) -> u32 {
        if self.needs_probe() {
            // Step away from a batch size that has told us all it can, so the
            // next fit has two of them to compare. Downwards at the ceiling,
            // where there is no room to step up.
            return if current >= MAX_BATCH {
                current / 2
            } else {
                current.saturating_mul(2)
            };
        }
        (TARGET_DISPATCH.as_secs_f64() * 1000. / self.slope) as u32
    }

    /// Milliseconds a sample costs the caller, fixed part included. `None`
    /// until a dispatch has completed.
    fn ms_per_sample(&self) -> Option<f64> {
        self.per_sample
    }
}

/// Length of a single wait slice.
///
/// Waits are sliced rather than indefinite so that `abort` is still observed
/// while a dispatch is in flight, and so a wedged submission cannot hang the
/// render thread forever. Doubles as the idle poll interval.
const POLL_SLICE: Duration = Duration::from_millis(50);

/// Blocks until `index` has finished executing on the GPU.
///
/// Returns `Ok(false)` if an abort arrived while waiting.
///
/// Note this is a native-only guarantee: on WebGPU `PollType::Wait` is a no-op
/// and callbacks are driven by the window event loop instead.
fn wait_for_submission(
    device: &wgpu::Device,
    index: &wgpu::SubmissionIndex,
    abort: &Receiver<bool>,
) -> Result<bool, Box<dyn Error>> {
    loop {
        match device.poll(wgpu::PollType::Wait {
            submission_index: Some(index.clone()),
            timeout: Some(POLL_SLICE),
        }) {
            Ok(_) => return Ok(true),
            Err(wgpu::PollError::Timeout) => {
                if abort.try_recv().is_ok() {
                    return Ok(false);
                }
                // Still running. Recording the next dispatch here would defeat
                // the point of waiting, so go around again.
            }
            Err(e) => {
                return Err(SimpleError::new(format!("Failed to poll device: {}", e)).into());
            }
        }
    }
}

/// Renderer is a central part of the raytracer responsible for controlling the
/// process reporting back progress to the caller
pub struct Renderer<'a> {
    /// Only the sample count is retained from the scene. Holding the whole
    /// `Scene` kept the entire CPU scene graph and every decoded texture
    /// resident for the life of the render -- on an integrated GPU that is
    /// the GPU's memory too.
    samples_per_pixel: u32,
    samples_per_batch: u32,
    width: u32,
    height: u32,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    output_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    sample_count_buffer: wgpu::Buffer,
    gbuffer: wgpu::Buffer,
    /// Scratch copy the post-processing chain runs on, so the accumulator is
    /// never written by a post-processor. `None` when there is no chain.
    post_buffer: Option<wgpu::Buffer>,
    /// See [`RenderConfig::preview`].
    preview: bool,
    bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    nodes_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    spheres_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    triangle_pos_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    triangle_attr_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    quad_pos_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    quad_attr_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    materials_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    camera_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    config_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    lights_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    prim_refs_buffer: wgpu::Buffer,
    post_processors: Vec<PostProcessors>,
    render_config: GpuRenderConfig,
    /// Per-pass GPU timing. `None` unless `SOLSTRALE_GPU_TIMING` is set and the
    /// device supports timestamp queries, so the ordinary render allocates
    /// nothing and encodes nothing for it -- see [`GpuTimer`].
    ///
    /// Deliberately not surfaced on [`RenderProgress`]: that type
    /// is public, the field would be an `Option` forever, and filling it would
    /// oblige a map and a poll every batch for a number no library consumer
    /// asked for.
    timer: Option<GpuTimer>,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
}

impl<'a> Renderer<'a> {
    /// Creates a new GPU renderer given a scene
    pub fn new(
        scene: Scene,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
    ) -> Result<Self, Box<dyn Error>> {
        Self::build(scene, device, queue, None, SHIPPED_ESTIMATOR)
    }

    /// As [`Renderer::new`], but compiled as a reference estimator: next-event
    /// estimation off, so no light is ever sampled or weighed and every emitter
    /// a BSDF path lands on is taken at full weight, and the firefly clamp
    /// lifted, so nothing is thrown away on the way.
    ///
    /// The same integral as [`Renderer::new`] by a route that shares none of
    /// its light PDFs, none of its MIS weights and none of its bias. Converged,
    /// the two have to agree -- and that is the only check the suite has on
    /// those PDFs that does not go through the very PDFs it is checking, which
    /// is why this is reachable from outside the crate at all.
    ///
    /// Far noisier per sample on a scene lit by discrete lights, which is the
    /// whole reason `SHIPPED_ESTIMATOR` is what it is. For converged
    /// comparisons, not for rendering.
    pub fn bsdf_only_reference(
        scene: Scene,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
    ) -> Result<Self, Box<dyn Error>> {
        let estimator = Estimator {
            nee_enabled: false,
            // Finite rather than infinite: an override reaches the shader as a
            // literal, and `min(x, inf)` is a NaN waiting for a backend that
            // folds it differently. Nothing a path can carry comes near this.
            clamping_threshold: f32::MAX,
        };
        Self::build(scene, device, queue, None, estimator)
    }

    /// As [`Renderer::new`], but able to compile the tracer against something
    /// other than what the scene says. `None` -- what every caller but
    /// `specialisation_test` passes -- derives it from the scene; see
    /// [`Specialisation`].
    #[cfg(test)]
    fn with_specialisation(
        scene: Scene,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        specialisation: Option<Specialisation>,
    ) -> Result<Self, Box<dyn Error>> {
        Self::build(scene, device, queue, specialisation, SHIPPED_ESTIMATOR)
    }

    /// The one constructor. Everything above it differs only in what the
    /// tracer is compiled as.
    fn build(
        scene: Scene,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        specialisation: Option<Specialisation>,
        estimator: Estimator,
    ) -> Result<Self, Box<dyn Error>> {
        // A scene lit only by its background is a scene -- a uniform
        // environment is what a furnace test is made of, and next-event
        // estimation simply has nothing to sample there. What is rejected is a
        // scene with no light of either kind, which can only render black.
        if !scene.world.has_lights() && scene.background_color.near_zero() {
            return Err(Box::new(SimpleError::new(
                "Scene should have at least one light or a non-black background",
            )));
        }

        let width = scene.render_config.width as u32;
        let height = scene.render_config.height as u32;

        let module =
            shader_module_with_luminance(device, "ray_trace.wgsl", include_str!("ray_trace.wgsl"));

        // Flatten scene
        let scene_data = flatten_scene(&scene);

        // Create buffers
        let nodes_buffer = create_and_upload_buffer(
            device,
            queue,
            "Nodes Buffer",
            &scene_data.nodes,
            BufferUsages::STORAGE,
        );
        let prim_refs_buffer = create_and_upload_buffer(
            device,
            queue,
            "Prim Refs Buffer",
            &scene_data.prim_refs,
            BufferUsages::STORAGE,
        );
        let spheres_buffer = create_and_upload_buffer(
            device,
            queue,
            "Spheres Buffer",
            &scene_data.spheres,
            BufferUsages::STORAGE,
        );
        let triangle_pos_buffer = create_and_upload_buffer(
            device,
            queue,
            "Triangle Positions Buffer",
            &scene_data.triangle_pos,
            BufferUsages::STORAGE,
        );
        let triangle_attr_buffer = create_and_upload_buffer(
            device,
            queue,
            "Triangle Attributes Buffer",
            &scene_data.triangle_attr,
            BufferUsages::STORAGE,
        );
        let quad_pos_buffer = create_and_upload_buffer(
            device,
            queue,
            "Quad Positions Buffer",
            &scene_data.quad_pos,
            BufferUsages::STORAGE,
        );
        // Empty unless the scene can present a rough dielectric, in which case
        // it is one 512-entry table per distinct index of refraction.
        // `create_and_upload_buffer` pads an empty slice up to a valid binding.
        let dielectric_energy_buffer = create_and_upload_buffer(
            device,
            queue,
            "Dielectric Energy Buffer",
            &scene_data.dielectric_energy,
            BufferUsages::STORAGE,
        );

        let quad_attr_buffer = create_and_upload_buffer(
            device,
            queue,
            "Quad Attributes Buffer",
            &scene_data.quad_attr,
            BufferUsages::STORAGE,
        );
        let materials_buffer = create_and_upload_buffer(
            device,
            queue,
            "Materials Buffer",
            &scene_data.materials,
            BufferUsages::STORAGE,
        );
        let lights_buffer = create_and_upload_buffer(
            device,
            queue,
            "Lights Buffer",
            &scene_data.lights,
            BufferUsages::STORAGE,
        );

        // Blit the atlas using the layout `flatten_scene` already computed.
        // This used to re-run the identical packing here and rely on it being
        // deterministic.
        let mut atlas_image;

        if let Some(layout) = scene_data.atlas_layout.as_ref() {
            atlas_image = RgbImage::new(layout.width, layout.height);

            for placement in layout.placements.iter() {
                let texture = &scene_data.textures[placement.original_index];
                image::imageops::replace(
                    &mut atlas_image,
                    texture.as_ref(),
                    placement.x as i64,
                    placement.y as i64,
                );
            }
        } else {
            // Create a 1x1 white pixel if no textures, just to have valid binding
            atlas_image = RgbImage::from_pixel(1, 1, Rgb([255, 255, 255]));
        }

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

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let camera_inst = Camera::new(width as usize, height as usize, &scene.camera);

        let gpu_camera = camera_to_gpu(&camera_inst);
        let camera_buffer = create_and_upload_buffer(
            device,
            queue,
            "Camera Buffer",
            &[gpu_camera],
            BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
        );

        let render_config = GpuRenderConfig {
            sample_count: 0,
            max_depth: scene.render_config.max_depth.max(1),
            samples_per_batch: scene.render_config.samples_per_batch.max(1),
            min_samples_per_pixel: scene.render_config.min_samples_per_pixel,
            background_color: [
                scene.background_color.x as f32,
                scene.background_color.y as f32,
                scene.background_color.z as f32,
            ],
            variance_threshold: scene.render_config.variance_threshold,
            restart_index: scene.render_config.seed,
            _padding: [0; 3],
        };
        let config_buffer = create_and_upload_buffer(
            device,
            queue,
            "Config Buffer",
            &[render_config],
            BufferUsages::UNIFORM,
        );

        let bind_group_layout = bind_group_layout(
            device,
            &[
                storage_binding(false, 0), // 0: output buffer
                storage_binding(true, 0),  // 1: nodes
                storage_binding(true, 0),  // 2: spheres
                storage_binding(true, 0),  // 3: triangles
                storage_binding(true, 0),  // 4: quads
                storage_binding(true, 0),  // 5: materials
                uniform_binding(std::mem::size_of::<GpuCamera>() as u64), // 6: camera
                uniform_binding(std::mem::size_of::<GpuRenderConfig>() as u64), // 7: config
                texture_binding(wgpu::TextureViewDimension::D2), // 8: texture array
                sampler_binding(),         // 9: sampler
                storage_binding(true, 0),  // 10: lights
                storage_binding(true, 0),  // 11: primitive references
                storage_binding(true, 0),  // 12: triangle attributes
                storage_binding(true, 0),  // 13: quad attributes
                storage_binding(false, 0), // 14: per-pixel sample count
                storage_binding(false, 0), // 15: guide (denoiser G-buffer)
                storage_binding(true, 0),  // 16: dielectric energy tables
            ],
        );

        // Everything the tracer is specialised on. See [`Specialisation`] for
        // why these are overrides and not uniforms, and `ray_trace.wgsl` for
        // what each one strips.
        let specialisation =
            specialisation.unwrap_or_else(|| Specialisation::from_scene_data(&scene_data));
        // Nothing but a denoising post-processor opens the G-buffer, so a chain
        // without one is a whole extra ray per pixel per accumulation run spent
        // on a buffer that is never read. Asked of the chain rather than
        // inferred from it, so a post-processor added later cannot silently get
        // zeroes.
        let writes_guide = scene
            .render_config
            .post_processors
            .iter()
            .any(|p| p.needs_guide());
        let pipeline = compute_pipeline(
            device,
            &bind_group_layout,
            &module,
            &specialisation.constants(
                width,
                height,
                scene.render_config.low_discrepancy,
                estimator,
                writes_guide,
            ),
        );

        let size = (width * height * 16) as u64; // vec3 is 16 bytes aligned (as vec4 effectively)

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Actual accumulated sample count per pixel, used by adaptive sampling
        // to skip converged pixels. Its initial contents are never read as-is:
        // the shader always resets a pixel's count when `sample_count == 0`.
        let sample_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sample Count Buffer"),
            size: (width * height * 4) as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Primary-hit albedo, normal and camera distance, packed 16 bytes per
        // pixel. Written by the tracer on the first dispatch of an accumulation
        // run and read only by a denoising post-processor. COPY_SRC so tests can
        // read it back.
        let gbuffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Guide Buffer"),
            size: (width * height * 16) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = bind_group(
            device,
            &bind_group_layout,
            &[
                wgpu::BindingResource::Buffer(output_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(nodes_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(spheres_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(triangle_pos_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(quad_pos_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(materials_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(camera_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(config_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::TextureView(&texture_view),
                wgpu::BindingResource::Sampler(&sampler),
                wgpu::BindingResource::Buffer(lights_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(prim_refs_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(triangle_attr_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(quad_attr_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(sample_count_buffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(gbuffer.as_entire_buffer_binding()),
                wgpu::BindingResource::Buffer(dielectric_energy_buffer.as_entire_buffer_binding()),
            ],
        );

        let mut post_processors = scene.render_config.post_processors.clone();
        for p in &mut post_processors {
            p.initialize(device, queue, width, height);
        }

        // The post-processing chain runs on a copy, never on the accumulator.
        // See the comment on PostProcessContext::accumulator for why: a
        // post-processor writing into output_buffer would feed its own output
        // back into the next batch's Welford merge. Only allocated when there is
        // a chain to run, so the default path costs nothing.
        let post_buffer = if post_processors.is_empty() {
            None
        } else {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Post Process Buffer"),
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        };

        Ok(Renderer {
            width,
            height,
            bind_group_layout,
            samples_per_pixel: scene.render_config.samples_per_pixel,
            samples_per_batch: scene.render_config.samples_per_batch.max(1),
            pipeline,
            output_buffer,
            sample_count_buffer,
            gbuffer,
            post_buffer,
            preview: scene.render_config.preview,
            bind_group,
            nodes_buffer,
            spheres_buffer,
            triangle_pos_buffer,
            triangle_attr_buffer,
            quad_pos_buffer,
            quad_attr_buffer,
            materials_buffer,
            camera_buffer,
            config_buffer,
            lights_buffer,
            prim_refs_buffer,
            post_processors,
            render_config,
            timer: GpuTimer::new(device, queue),
            device,
            queue,
        })
    }

    /// Updates the camera buffer with a new camera configuration
    pub fn update_camera(&mut self, camera_config: &CameraConfig) {
        let camera_inst = Camera::new(self.width as usize, self.height as usize, camera_config);
        let gpu_camera = camera_to_gpu(&camera_inst);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[gpu_camera]));
    }

    /// Executes the rendering of the image on the GPU
    pub fn render(
        &mut self,
        output: &Sender<RenderProgress>,
        camera_config: &Receiver<CameraConfig>,
        abort: &Receiver<bool>,
        idle_after_rendering: bool,
    ) -> Result<(), Box<dyn Error>> {
        let samples_per_pixel = self.samples_per_pixel;
        let workgroup_count_x = self.width.div_ceil(8);
        let workgroup_count_y = self.height.div_ceil(8);

        // Number of samples already accumulated into the output buffer.
        let mut completed = 0;
        // Seeded from the configured batch size, then continuously re-tuned to
        // keep the work in a single dispatch inside TARGET_DISPATCH.
        let mut batch_size = self.samples_per_batch.max(1);
        // What a dispatch costs. Dominated by the scene rather than by the
        // view, so it deliberately survives camera changes and only has to
        // re-converge when the view changes character.
        let mut cost = DispatchCost::new();
        // The first dispatch of a run pays for shader and allocation warm-up
        // that no later one does, and is left out of the cost model rather
        // than left to age out of it.
        let mut first_dispatch = true;
        // A camera config picked up while idling, handled at the top of the
        // next iteration together with any that arrived after it.
        let mut idle_camera_config = None;

        loop {
            if abort.try_recv().is_ok() {
                return Ok(());
            }

            let mut latest_camera_config = idle_camera_config.take();
            while let Ok(config) = camera_config.try_recv() {
                latest_camera_config = Some(config);
            }

            if let Some(config) = latest_camera_config {
                self.update_camera(&config);
                // Restart the accumulation. The output buffer deliberately is
                // not cleared: the shader overwrites every pixel it covers when
                // sample_count is zero, so clearing only costs a dispatch and
                // leaves a window in which a caller blitting the buffer sees
                // black.
                completed = 0;
                // Sample indices restart at zero too, so without a fresh
                // restart_index the RNG would hand every frame of a camera
                // drag the identical sample sequence.
                self.render_config.restart_index = self.render_config.restart_index.wrapping_add(1);
                // Get an image of the new view out as fast as possible, then
                // grow back into the budget. Carrying a large batch across the
                // restart would spend a whole dispatch before showing anything
                // of where the camera now points.
                batch_size = 1;
            }

            if completed >= samples_per_pixel {
                if !idle_after_rendering {
                    break;
                }
                // Block on the channel rather than polling it, so a converged
                // image costs nothing and reacts the moment a camera update
                // arrives.
                match camera_config.recv_timeout(POLL_SLICE) {
                    Ok(config) => idle_camera_config = Some(config),
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return Ok(()),
                }
                continue;
            }

            // Never overshoot the requested sample count.
            let batch = batch_size.min(samples_per_pixel - completed);
            self.render_config.sample_count = completed;
            self.render_config.samples_per_batch = batch;
            self.queue.write_buffer(
                &self.config_buffer,
                0,
                bytemuck::cast_slice(&[self.render_config]),
            );

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            add_compute_pass_2d(
                &mut encoder,
                &self.pipeline,
                &self.bind_group,
                workgroup_count_x,
                workgroup_count_y,
                self.timer.as_mut(),
                "trace",
            );

            let last_batch = completed + batch >= samples_per_pixel;

            // Refreshed every batch, not just the one the whole chain runs on:
            // the post buffer is what gets published on RenderProgress, so a
            // caller watching an unfinished render has to see the accumulated
            // image there rather than whatever the allocation came with. It is
            // cheap enough not to be worth a way out -- 0.05 ms of a 10 ms
            // dispatch at 800x600, which is what `post_copy` is timed for.
            if let Some(post_buffer) = &self.post_buffer {
                let size = (self.width * self.height) as u64 * crate::post::PIXEL_SIZE;
                let src = &self.output_buffer;
                let copy = |encoder: &mut wgpu::CommandEncoder| {
                    encoder.copy_buffer_to_buffer(src, 0, post_buffer, 0, size);
                };
                match self.timer.as_mut() {
                    Some(timer) => timer.encoder_scope(&mut encoder, "post_copy", copy),
                    None => copy(&mut encoder),
                }

                let mut ctx = crate::post::PostProcessContext {
                    encoder: &mut encoder,
                    buffer: post_buffer,
                    accumulator: &self.output_buffer,
                    sample_count_buffer: &self.sample_count_buffer,
                    gbuffer: &self.gbuffer,
                    samples_completed: completed,
                    device: self.device,
                    timer: self.timer.as_mut(),
                };
                // On any batch but the last, only the processors that say
                // they belong in a preview. A drag never reaches a last batch:
                // every frame of one restarts the accumulation, so this is the
                // only path on which a moving camera is ever filtered at all.
                for p in &self.post_processors {
                    if last_batch || (self.preview && p.preview()) {
                        p.post_process(&mut ctx)?;
                    }
                }
            }

            // After the last timed pass and before `finish`, so the resolve
            // sees every timestamp this command buffer wrote.
            if let Some(timer) = &self.timer {
                timer.resolve(&mut encoder);
            }

            let command_buffer = encoder.finish();
            let dispatch_start = Instant::now();
            let submission = self.queue.submit([command_buffer]);

            // Back-pressure. Submissions on a queue execute in order, so
            // anything left queued here is added directly to the frame latency
            // of a caller sharing the device -- and without a wait this loop
            // runs thousands of iterations ahead of the GPU. Waiting also means
            // the progress reported below describes work that has actually
            // completed.
            if !wait_for_submission(self.device, &submission, abort)? {
                return Ok(());
            }
            let dispatch_time = dispatch_start.elapsed();

            if let Some(timer) = &mut self.timer {
                report_dispatch(batch, dispatch_time, &timer.take(self.device));
            }

            completed += batch;

            if first_dispatch {
                first_dispatch = false;
            } else {
                cost.record(batch, dispatch_time);
                // Grow at most by doubling, so a single anomalously cheap
                // dispatch cannot blow the batch size up and stall the next
                // frame.
                batch_size = cost
                    .target_batch(batch)
                    .clamp(1, (batch_size * 2).min(MAX_BATCH));
            }

            let ms_per_sample = cost
                .ms_per_sample()
                .unwrap_or_else(|| dispatch_time.as_secs_f64() * 1000. / batch as f64);

            output.send(RenderProgress {
                progress: completed as f64 / samples_per_pixel as f64,
                fps: Some(1000. / ms_per_sample.max(MIN_SLOPE)),
                estimated_time_left: calculate_estimated_time_left(
                    ms_per_sample,
                    samples_per_pixel - completed,
                ),
                // Stable for the life of the render -- always the post buffer
                // when there is a chain, always the accumulator when there is
                // not -- so a caller caching a bind group per buffer handle
                // never has to rebuild it between batches.
                output_buffer: self
                    .post_buffer
                    .as_ref()
                    .unwrap_or(&self.output_buffer)
                    .clone(),
            })?;
        }

        Ok(())
    }
}

/// Time left, from the measured cost of a sample rather than from the elapsed
/// time of the run. Restarting the accumulation on a camera change therefore
/// does not throw the estimate off.
fn calculate_estimated_time_left(ms_per_sample: f64, samples_left: u32) -> Duration {
    Duration::from_secs_f64(ms_per_sample / 1000. * samples_left as f64)
}

fn camera_to_gpu(camera_inst: &Camera) -> GpuCamera {
    GpuCamera {
        origin: [
            camera_inst.origin.x as f32,
            camera_inst.origin.y as f32,
            camera_inst.origin.z as f32,
        ],
        lens_radius: camera_inst.lens_radius as f32,
        lower_left_corner: [
            camera_inst.lower_left_corner.x as f32,
            camera_inst.lower_left_corner.y as f32,
            camera_inst.lower_left_corner.z as f32,
        ],
        _pad1: 0.0,
        horizontal: [
            camera_inst.horizontal.x as f32,
            camera_inst.horizontal.y as f32,
            camera_inst.horizontal.z as f32,
        ],
        _pad2: 0.0,
        vertical: [
            camera_inst.vertical.x as f32,
            camera_inst.vertical.y as f32,
            camera_inst.vertical.z as f32,
        ],
        _pad3: 0.0,
        u: [
            camera_inst.u.x as f32,
            camera_inst.u.y as f32,
            camera_inst.u.z as f32,
        ],
        _pad4: 0.0,
        v: [
            camera_inst.v.x as f32,
            camera_inst.v.y as f32,
            camera_inst.v.z as f32,
        ],
        _pad5: 0.0,
    }
}

fn create_and_upload_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &[T],
    usage: BufferUsages,
) -> wgpu::Buffer {
    let size_bytes = size_of_val(data) as u64;

    // Ensure the minimum size for a valid buffer and pad to 16 bytes for WGSL array compatibility
    let mut effective_size = if size_bytes == 0 {
        size_of::<T>() as u64
    } else {
        size_bytes
    };

    if effective_size % 16 != 0 {
        effective_size = ((effective_size / 16) + 1) * 16;
    }

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: effective_size,
        usage: usage | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    if size_bytes > 0 {
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));
    }

    buffer
}

#[cfg(test)]
mod test {
    use crate::renderer::{
        DispatchCost, MAX_BATCH, TARGET_DISPATCH, calculate_estimated_time_left,
    };
    use std::time::Duration;

    /// Runs the render loop's batch-size feedback against a cost function,
    /// returning the batch size used by each dispatch.
    ///
    /// `cost_ms` is given the dispatch index as well as the batch size, so a
    /// test can make the first dispatches cost what warm-up makes them cost.
    fn settle(cost_ms: impl Fn(u32, usize) -> f64, start_batch: u32) -> Vec<u32> {
        let mut cost = DispatchCost::new();
        let mut batch = start_batch;
        let mut batches = Vec::new();

        for dispatch in 0..400 {
            batches.push(batch);
            cost.record(
                batch,
                Duration::from_secs_f64(cost_ms(batch, dispatch) / 1000.),
            );
            batch = cost
                .target_batch(batch)
                .clamp(1, (batch * 2).min(MAX_BATCH));
        }

        batches
    }

    /// Mean throughput in samples per millisecond once the feedback has
    /// settled, which is what the batch size is ultimately chosen for.
    fn settled_throughput(batches: &[u32], cost_ms: impl Fn(u32, usize) -> f64) -> f64 {
        let from = batches.len() / 2;
        let samples: u32 = batches[from..].iter().sum();
        let ms: f64 = batches[from..]
            .iter()
            .enumerate()
            .map(|(i, &b)| cost_ms(b, from + i))
            .sum();
        samples as f64 / ms
    }

    #[test]
    fn test_dispatch_cost_separates_the_fixed_term() {
        // Shaped like the measurement in DispatchCost's documentation: a
        // dispatch costs a display frame of queue latency plus real work.
        let batches = settle(|b, _| 16.7 + 2.5 * b as f64, 4);

        let target = TARGET_DISPATCH.as_secs_f64() * 1000. / 2.5;
        let settled = batches[batches.len() - 1];
        assert!(
            (settled as f64 - target).abs() <= 1.,
            "batch size {} should fill the budget with samples at 2.5 ms each, expected about {}",
            settled,
            target
        );
    }

    #[test]
    fn test_dispatch_cost_does_not_collapse_under_queue_latency() {
        // The measured curve at 1384x784 on a 60 Hz display, where charging the
        // latency to the samples used to pin the batch size at one and run the
        // render at a third of the throughput the same machine reaches.
        let cost_ms = |b: u32, _: usize| 6.0 + 2.5 * b as f64;
        let peak = 1. / 2.5;

        for start_batch in [1, 2, 4, 16, MAX_BATCH] {
            let batches = settle(cost_ms, start_batch);
            let throughput = settled_throughput(&batches, cost_ms);

            assert!(
                batches[batches.len() - 1] > 1,
                "batch size collapsed to one from a start of {}",
                start_batch
            );
            assert!(
                throughput > peak * 0.6,
                "throughput {:.3} from a start of {} is far below the {:.3} the cost function allows",
                throughput,
                start_batch,
                peak
            );
        }
    }

    #[test]
    fn test_dispatch_cost_recovers_from_a_warm_up_dispatch() {
        // The first dispatches of a run pay for shader and allocation warm-up,
        // and an estimate taken from those alone asks for the smallest batch
        // there is. Reaching that batch size must not be the end of it: the
        // dispatch time does not follow the batch size down, and the model has
        // to keep asking until something says so.
        let cost_ms = |b: u32, dispatch: usize| {
            if dispatch < 2 {
                200.
            } else {
                6.0 + 2.5 * b as f64
            }
        };
        let batches = settle(cost_ms, 4);
        let throughput = settled_throughput(&batches, cost_ms);

        assert!(
            throughput > (1. / 2.5) * 0.6,
            "throughput {:.3} never recovered from the warm-up dispatches",
            throughput
        );
    }

    #[test]
    fn test_dispatch_cost_keeps_expensive_samples_in_small_batches() {
        // No fixed term worth speaking of and a sample that costs more than the
        // whole budget: one sample per dispatch is the right answer, and the
        // occasional probe must not turn into a standing larger batch.
        let batches = settle(|b, _| 0.2 + 30. * b as f64, 4);
        let settled = &batches[batches.len() / 2..];

        let ones = settled.iter().filter(|&&b| b == 1).count();
        assert!(
            ones * 4 > settled.len() * 3,
            "only {} of {} dispatches used a batch of one",
            ones,
            settled.len()
        );
        assert!(
            settled.iter().all(|&b| b <= 4),
            "probing ran the batch size up to {}",
            settled.iter().max().unwrap()
        );
    }

    #[test]
    fn test_dispatch_cost_reports_what_a_sample_costs_the_caller() {
        let mut cost = DispatchCost::new();
        assert_eq!(cost.ms_per_sample(), None);

        // The fixed part is the caller's to wait through, so it belongs in the
        // reported rate even though the batch size is not tuned against it.
        for _ in 0..100 {
            cost.record(8, Duration::from_secs_f64((6.0 + 2.5 * 8.) / 1000.));
        }
        let per_sample = cost.ms_per_sample().unwrap();
        assert!(
            (per_sample - 26. / 8.).abs() < 0.01,
            "reported {} ms per sample, expected {}",
            per_sample,
            26. / 8.
        );
    }

    #[test]
    fn test_calculate_estimated_time_left() {
        let mut time_left = calculate_estimated_time_left(1000., 99);
        assert_eq!(time_left, Duration::from_secs(99));

        time_left = calculate_estimated_time_left(20., 50);
        assert_eq!(time_left, Duration::from_secs(1));

        time_left = calculate_estimated_time_left(20., 0);
        assert_eq!(time_left, Duration::from_secs(0));
    }

    #[test]
    fn test_update_camera() {
        use crate::camera::CameraConfig;
        use crate::geo::transformation::NopTransformer;
        use crate::geo::vec3::Vec3;
        use crate::hittable::{Bvh, Sphere};
        use crate::material::DiffuseLight;
        use crate::renderer::gpu_data::GpuCamera;
        use crate::renderer::{RenderConfig, Renderer, Scene};
        use crate::util::wgpu_util::{get_result_from_buffer, get_wgpu_device_and_queue};

        let (device, queue) = get_wgpu_device_and_queue();
        let render_config = RenderConfig {
            width: 10,
            height: 10,
            ..Default::default()
        };
        let mut world = Vec::new();
        world.push(
            Sphere::new(
                Vec3::new(0., 10., 0.),
                1.,
                DiffuseLight::new(1., 1., 1., None).into(),
                &NopTransformer(),
            )
            .into(),
        );

        let scene = Scene {
            world: Bvh::new(world).into(),
            camera: CameraConfig {
                look_from: Vec3::new(0., 0., 1.),
                ..Default::default()
            },
            background_color: Vec3::new(0., 0., 0.),
            render_config,
        };

        let mut renderer = Renderer::new(scene, device, queue).unwrap();

        let new_config = CameraConfig {
            look_from: Vec3::new(0., 0., 10.),
            ..Default::default()
        };

        renderer.update_camera(&new_config);

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<GpuCamera>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            &renderer.camera_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of::<GpuCamera>() as u64,
        );
        queue.submit(Some(encoder.finish()));

        let camera_data: Vec<GpuCamera> = get_result_from_buffer(device, &staging_buffer);
        assert_eq!(camera_data[0].origin, [0., 0., 10.]);
    }

    /// Decodes an IEEE half. Only the cases octahedral coordinates actually
    /// produce are handled: zero and the normal range. Subnormals are below
    /// 6e-5 and read as zero, which is within this test's tolerances anyway.
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1f) as u32;
        let frac = (bits & 0x3ff) as u32;
        if exp == 0 {
            return f32::from_bits(sign << 31);
        }
        f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
    }

    /// A chain that reads the guide, which is what makes the tracer fill it.
    /// The filter's own output is beside the point here -- both G-buffer tests
    /// read `Renderer::gbuffer` directly -- but `writes_guide` is compiled from
    /// what the chain declares, so without one there is nothing to read.
    fn guide_reader(device: &wgpu::Device) -> crate::post::PostProcessors {
        crate::post::DenoisePostProcessor::new(1., None, None, device)
            .unwrap()
            .into()
    }

    /// Inverse of `oct_encode` in ray_trace.wgsl. Must stay in step with it.
    fn oct_decode(e: [f32; 2]) -> [f32; 3] {
        let mut v = [e[0], e[1], 1.0 - e[0].abs() - e[1].abs()];
        if v[2] < 0.0 {
            let x = (1.0 - v[1].abs()) * if v[0] >= 0.0 { 1.0 } else { -1.0 };
            let y = (1.0 - v[0].abs()) * if v[1] >= 0.0 { 1.0 } else { -1.0 };
            v[0] = x;
            v[1] = y;
        }
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / len, v[1] / len, v[2] / len]
    }

    /// The guide the denoiser reads has to describe the geometry the camera
    /// actually sees. This pins all four packed slots against a scene whose
    /// answer is known analytically, which is what catches the two traps in the
    /// encoding: the octahedral round trip, and using `rec.t` (in units of the
    /// unnormalised primary ray) where a world-space distance is meant.
    #[test]
    fn test_gbuffer_direct_hit() {
        use crate::camera::CameraConfig;
        use crate::geo::transformation::NopTransformer;
        use crate::geo::vec3::Vec3;
        use crate::hittable::{Bvh, Hittables, Sphere};
        use crate::material::texture::SolidColor;
        use crate::material::{DiffuseLight, Lambertian};
        use crate::renderer::{RenderConfig, Renderer, Scene};
        use crate::util::wgpu_util::{get_result_from_buffer, get_wgpu_device_and_queue};
        use std::sync::mpsc::channel;

        let (device, queue) = get_wgpu_device_and_queue();

        // Odd dimensions so there is a true centre pixel, and no aperture so the
        // primary ray carries no lens jitter.
        const SIZE: u32 = 41;
        let render_config = RenderConfig {
            width: SIZE as usize,
            height: SIZE as usize,
            samples_per_pixel: 1,
            // The tracer only fills the G-buffer when something in the chain
            // reads it, so a test that reads it has to ask for it the way a
            // caller would rather than through a back door.
            post_processors: vec![guide_reader(device)],
            ..Default::default()
        };

        // A yellow sphere of radius 0.5 at the origin, seen from 4 units away:
        // the centre ray hits its near pole at z = 0.5, so the camera distance
        // is 3.5 and the surface normal there points straight back at the camera.
        let world: Vec<Hittables> = vec![
            Sphere::new(
                Vec3::new(0., 0., 0.),
                0.5,
                Lambertian::new(SolidColor::new(1., 1., 0.).into(), None).into(),
                &NopTransformer(),
            )
            .into(),
            // The renderer requires a light. Far outside the 20 degree frustum,
            // so it cannot be the primary hit for any pixel under test.
            Sphere::new(
                Vec3::new(0., 100., 0.),
                20.,
                DiffuseLight::new(10., 10., 10., None).into(),
                &NopTransformer(),
            )
            .into(),
        ];

        let scene = Scene {
            world: Bvh::new(world).into(),
            camera: CameraConfig {
                vertical_fov_degrees: 20.,
                aperture_size: 0.,
                look_from: Vec3::new(0., 0., 4.),
                look_at: Vec3::new(0., 0., 0.),
                up: Vec3::new(0., 1., 0.),
            },
            background_color: Vec3::new(0.2, 0.3, 0.5),
            render_config,
        };

        let mut renderer = Renderer::new(scene, device, queue).unwrap();
        let (output_sender, output_receiver) = channel();
        let (_camera_sender, camera_receiver) = channel();
        let (_abort_sender, abort_receiver) = channel();
        renderer
            .render(&output_sender, &camera_receiver, &abort_receiver, false)
            .unwrap();
        drop(output_sender);
        for _ in output_receiver {}

        let size = (SIZE * SIZE * 16) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&renderer.gbuffer, 0, &staging_buffer, 0, size);
        queue.submit(Some(encoder.finish()));
        let guide: Vec<[u32; 4]> = get_result_from_buffer(device, &staging_buffer);

        let centre = guide[((SIZE / 2) * SIZE + SIZE / 2) as usize];

        let albedo = [
            (centre[0] & 0xff) as f32 / 255.,
            ((centre[0] >> 8) & 0xff) as f32 / 255.,
            ((centre[0] >> 16) & 0xff) as f32 / 255.,
        ];
        assert!(
            albedo[0] > 0.99 && albedo[1] > 0.99 && albedo[2] < 0.01,
            "centre albedo should be the sphere's yellow, was {:?}",
            albedo
        );

        let normal = oct_decode([
            f16_to_f32((centre[1] & 0xffff) as u16),
            f16_to_f32((centre[1] >> 16) as u16),
        ]);
        assert!(
            normal[0].abs() < 0.001 && normal[1].abs() < 0.001 && normal[2] > 0.999,
            "centre normal should point back at the camera, was {:?}",
            normal
        );

        let depth = f32::from_bits(centre[2]);
        assert!(
            (depth - 3.5).abs() < 0.001,
            "centre depth should be the 3.5 unit camera distance, was {}",
            depth
        );
        assert_eq!(centre[3], 0, "centre material should be MAT_LAMBERTIAN");

        // A corner ray clears the sphere entirely and must read as background,
        // so the filter keeps the silhouette rather than blending across it.
        let corner = guide[0];
        assert_eq!(corner[3], 5, "corner material should be MAT_MISS");
        assert_eq!(
            f32::from_bits(corner[2]),
            1e7,
            "corner depth should be the far sentinel"
        );
    }

    /// The other half of the same contract: with nothing in the chain reading
    /// the guide, the tracer must not trace it.
    ///
    /// A guide ray is a whole extra ray per pixel, and a specular one is up to
    /// seven. On a long render that is one restart against thousands of sample
    /// paths and no benchmark will ever see it; during a camera drag every
    /// frame is a restart at one sample per pixel, and it measures 9% of a
    /// frame on `create_test_scene` at 800x600.
    /// `interactive_restart_frame_cost` is where that number comes from; this
    /// pins the mechanism, which is that the override takes `trace_guide` out
    /// of the module rather than merely skipping a store.
    ///
    /// `ColorOnly` is in here too because it is the trap: it is a denoiser, it
    /// is bound to the G-buffer, and it reads nothing from it.
    #[test]
    fn test_guide_is_not_written_without_a_reader() {
        use crate::camera::CameraConfig;
        use crate::geo::transformation::NopTransformer;
        use crate::geo::vec3::Vec3;
        use crate::hittable::{Bvh, Hittables, Sphere};
        use crate::material::texture::SolidColor;
        use crate::material::{DiffuseLight, Lambertian};
        use crate::post::{DenoiseGuide, DenoisePostProcessor, PostProcessors};
        use crate::renderer::{RenderConfig, Renderer, Scene};
        use crate::util::wgpu_util::{get_result_from_buffer, get_wgpu_device_and_queue};
        use std::sync::mpsc::channel;

        let (device, queue) = get_wgpu_device_and_queue();

        const SIZE: u32 = 41;

        let guide_for = |post_processors: Vec<PostProcessors>| {
            let world: Vec<Hittables> = vec![
                Sphere::new(
                    Vec3::new(0., 0., 0.),
                    0.5,
                    Lambertian::new(SolidColor::new(1., 1., 0.).into(), None).into(),
                    &NopTransformer(),
                )
                .into(),
                Sphere::new(
                    Vec3::new(0., 100., 0.),
                    20.,
                    DiffuseLight::new(10., 10., 10., None).into(),
                    &NopTransformer(),
                )
                .into(),
            ];
            let scene = Scene {
                world: Bvh::new(world).into(),
                camera: CameraConfig {
                    vertical_fov_degrees: 20.,
                    aperture_size: 0.,
                    look_from: Vec3::new(0., 0., 4.),
                    look_at: Vec3::new(0., 0., 0.),
                    up: Vec3::new(0., 1., 0.),
                },
                background_color: Vec3::new(0.2, 0.3, 0.5),
                render_config: RenderConfig {
                    width: SIZE as usize,
                    height: SIZE as usize,
                    samples_per_pixel: 1,
                    post_processors,
                    ..Default::default()
                },
            };

            let mut renderer = Renderer::new(scene, device, queue).unwrap();
            let (output_sender, output_receiver) = channel();
            let (_camera_sender, camera_receiver) = channel();
            let (_abort_sender, abort_receiver) = channel();
            renderer
                .render(&output_sender, &camera_receiver, &abort_receiver, false)
                .unwrap();
            drop(output_sender);
            for _ in output_receiver {}

            let size = (SIZE * SIZE * 16) as u64;
            let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(&renderer.gbuffer, 0, &staging_buffer, 0, size);
            queue.submit(Some(encoder.finish()));
            get_result_from_buffer::<[u32; 4]>(device, &staging_buffer)
        };

        let denoiser = |guide| {
            let p: PostProcessors = DenoisePostProcessor::new(1., None, Some(guide), device)
                .unwrap()
                .into();
            vec![p]
        };

        let written = guide_for(denoiser(DenoiseGuide::Full));
        assert!(
            written.iter().any(|g| g != &[0; 4]),
            "a chain that reads the guide must get one"
        );

        for (name, chain) in [
            ("no chain", vec![]),
            ("colour-only denoiser", denoiser(DenoiseGuide::ColorOnly)),
        ] {
            let guide = guide_for(chain);
            assert!(
                guide.iter().all(|g| g == &[0; 4]),
                "{} must leave the guide buffer untouched",
                name
            );
        }
    }

    /// The point of `trace_guide`: on a mirror the guide has to describe what is
    /// reflected, not the mirror. Same analytic approach as above, folded once
    /// through a mirror so every slot has a different right answer than the
    /// primary hit would have given.
    #[test]
    fn test_gbuffer_follows_specular_chain() {
        use crate::camera::CameraConfig;
        use crate::geo::transformation::NopTransformer;
        use crate::geo::vec3::Vec3;
        use crate::hittable::{Bvh, Hittables, Quad, Sphere};
        use crate::material::texture::SolidColor;
        use crate::material::{DiffuseLight, Lambertian, Metal};
        use crate::renderer::{RenderConfig, Renderer, Scene};
        use crate::util::wgpu_util::{get_result_from_buffer, get_wgpu_device_and_queue};
        use std::sync::mpsc::channel;

        let (device, queue) = get_wgpu_device_and_queue();

        const SIZE: u32 = 41;
        let render_config = RenderConfig {
            width: SIZE as usize,
            height: SIZE as usize,
            samples_per_pixel: 1,
            post_processors: vec![guide_reader(device)],
            ..Default::default()
        };

        // Camera at z = 4 looking down -z at a mirror filling the z = 0 plane,
        // with a yellow sphere behind the camera at z = 8. The centre ray runs
        // 4 units to the mirror, reflects straight back along +z and runs 7.5
        // more to the sphere's near pole -- so the guide should report 11.5
        // units, the sphere's normal pointing back down -z at the mirror, and
        // the sphere's yellow dimmed by the mirror's half-grey.
        let world: Vec<Hittables> = vec![
            Quad::new(
                Vec3::new(-2., -2., 0.),
                Vec3::new(4., 0., 0.),
                Vec3::new(0., 4., 0.),
                Metal::new(SolidColor::new(0.5, 0.5, 0.5).into(), None, 0.).into(),
                &NopTransformer(),
            )
            .into(),
            Sphere::new(
                Vec3::new(0., 0., 8.),
                0.5,
                Lambertian::new(SolidColor::new(1., 1., 0.).into(), None).into(),
                &NopTransformer(),
            )
            .into(),
            // Off to the side of both the view ray and its reflection.
            Sphere::new(
                Vec3::new(0., 100., 0.),
                20.,
                DiffuseLight::new(10., 10., 10., None).into(),
                &NopTransformer(),
            )
            .into(),
        ];

        let scene = Scene {
            world: Bvh::new(world).into(),
            camera: CameraConfig {
                vertical_fov_degrees: 20.,
                aperture_size: 0.,
                look_from: Vec3::new(0., 0., 4.),
                look_at: Vec3::new(0., 0., 0.),
                up: Vec3::new(0., 1., 0.),
            },
            background_color: Vec3::new(0.2, 0.3, 0.5),
            render_config,
        };

        let mut renderer = Renderer::new(scene, device, queue).unwrap();
        let (output_sender, output_receiver) = channel();
        let (_camera_sender, camera_receiver) = channel();
        let (_abort_sender, abort_receiver) = channel();
        renderer
            .render(&output_sender, &camera_receiver, &abort_receiver, false)
            .unwrap();
        drop(output_sender);
        for _ in output_receiver {}

        let size = (SIZE * SIZE * 16) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&renderer.gbuffer, 0, &staging_buffer, 0, size);
        queue.submit(Some(encoder.finish()));
        let guide: Vec<[u32; 4]> = get_result_from_buffer(device, &staging_buffer);

        let centre = guide[((SIZE / 2) * SIZE + SIZE / 2) as usize];

        let albedo = [
            (centre[0] & 0xff) as f32 / 255.,
            ((centre[0] >> 8) & 0xff) as f32 / 255.,
            ((centre[0] >> 16) & 0xff) as f32 / 255.,
        ];
        assert!(
            (albedo[0] - 0.5).abs() < 0.01 && (albedo[1] - 0.5).abs() < 0.01 && albedo[2] < 0.01,
            "centre albedo should be the sphere's yellow tinted by the mirror, was {:?}",
            albedo
        );

        let normal = oct_decode([
            f16_to_f32((centre[1] & 0xffff) as u16),
            f16_to_f32((centre[1] >> 16) as u16),
        ]);
        // The centre pixel of an odd-sized frame maps to exactly 0.5 in both
        // axes, so the guide ray is on-axis and hits the pole dead on. What is
        // left is one ULP of the f16 octahedral encoding (1/2048), not a
        // geometric error.
        assert!(
            normal[0].abs() < 0.001 && normal[1].abs() < 0.001 && normal[2] < -0.999,
            "centre normal should be the sphere's, facing back at the mirror, was {:?}",
            normal
        );

        let depth = f32::from_bits(centre[2]);
        assert!(
            (depth - 11.5).abs() < 0.001,
            "centre depth should be the whole 11.5 unit folded path, was {}",
            depth
        );
        assert_eq!(
            centre[3],
            (1 << 8),
            "centre should read as MAT_LAMBERTIAN behind one specular bounce"
        );

        // A corner ray hits the mirror too, but its reflection diverges past the
        // sphere -- still one specular bounce deep, so the filter can tell it
        // from background seen directly.
        let corner = guide[0];
        assert_eq!(
            corner[3],
            5 | (1 << 8),
            "corner should read as MAT_MISS behind one specular bounce"
        );
    }
}
