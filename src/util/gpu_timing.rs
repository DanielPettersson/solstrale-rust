//! GPU-side timing of individual compute passes.
//!
//! Everything else in this crate that measures a dispatch measures CPU wall
//! clock around submit-plus-poll, which on a queue shared with a presenter
//! includes latency that is not ours. [`DispatchCost`] exists to model that
//! latency away; this exists to say what the work underneath it actually cost.
//!
//! Off unless [`TIMING_ENV`] is set, and absent entirely on a device without
//! [`wgpu::Features::TIMESTAMP_QUERY`], so the default path allocates nothing
//! and encodes nothing.
//!
//! [`DispatchCost`]: crate::renderer

use std::sync::Mutex;
use std::time::Duration;

/// Environment variable that switches GPU timing on. Any value but `0` or the
/// empty string enables it.
pub const TIMING_ENV: &str = "SOLSTRALE_GPU_TIMING";

/// Passes a single [`GpuTimer`] can hold timestamps for in one command buffer.
///
/// Two queries each. A render with every post-processor in the crate attached
/// and the denoiser at its default iteration count comes to fourteen, so this
/// is several times what a chain needs; passes beyond it are silently left
/// untimed rather than failing a render that only asked for a diagnostic.
const MAX_PASSES: usize = 64;

/// Bytes a resolved timestamp occupies.
const QUERY_SIZE: u64 = wgpu::QUERY_SIZE as u64;

/// What one labelled pass cost on the GPU.
///
/// Repeated labels within a report are folded together -- the denoiser's à-trous
/// iterations are one pipeline dispatched several times, and five lines saying
/// `denoise_atrous` are less use than one saying `denoise_atrous x5`.
#[derive(Clone, Debug)]
pub struct PassTiming {
    /// Label the pass was recorded under.
    pub label: &'static str,
    /// Passes folded into this entry.
    pub count: u32,
    /// GPU busy time summed over those passes, in milliseconds.
    pub ms: f64,
}

/// Per-pass totals since the last [`drain_totals`], summed over every dispatch.
static TOTALS: Mutex<Vec<PassTiming>> = Mutex::new(Vec::new());

/// Takes the accumulated per-pass totals and clears them.
///
/// A [`GpuTimer`] lives and dies with the [`Renderer`](crate::renderer::Renderer)
/// that owns it, so a caller wanting a whole-render figure rather than a
/// per-dispatch one has nowhere else to read it from. Only ever written while
/// timing is on.
pub fn drain_totals() -> Vec<PassTiming> {
    std::mem::take(&mut *TOTALS.lock().unwrap())
}

/// Folds one report into `report`, merging entries that share a label.
fn fold(report: &mut Vec<PassTiming>, label: &'static str, count: u32, ms: f64) {
    match report.iter_mut().find(|p| p.label == label) {
        Some(p) => {
            p.count += count;
            p.ms += ms;
        }
        None => report.push(PassTiming { label, count, ms }),
    }
}

/// Query set, staging buffers and timestamp period for timing the compute
/// passes of one command buffer at a time.
///
/// The cycle is: `pass_writes` once per pass while encoding, `resolve` before
/// `finish`, then `take` once the submission has completed. All of that is
/// crate-internal -- the renderer builds the timer and drives it. The type is
/// public only because it reaches a post-processor through
/// [`PostProcessContext::timer`](crate::post::PostProcessContext::timer);
/// what a caller reads is [`drain_totals`] and the stderr lines.
pub struct GpuTimer {
    query_set: wgpu::QuerySet,
    /// `QUERY_RESOLVE` destination. A separate buffer because a query-resolve
    /// target cannot also be `MAP_READ`.
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    /// Nanoseconds per timestamp tick, as the queue reports it.
    period_ns: f32,
    /// Whether the device can write a timestamp outside a pass, which is what
    /// [`GpuTimer::encoder_scope`] needs and a compute pass does not.
    inside_encoders: bool,
    /// Labels of the passes encoded so far, in the order they were encoded.
    /// Doubles as the query-index counter.
    labels: Vec<&'static str>,
}

impl GpuTimer {
    /// Builds a timer, or returns `None` if GPU timing is not switched on or
    /// the device cannot do it.
    ///
    /// Both checks are here rather than at the call site so that the renderer
    /// only ever has to keep an `Option` around.
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        match std::env::var(TIMING_ENV) {
            Ok(v) if v != "0" && !v.is_empty() => {}
            _ => return None,
        }

        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            eprintln!(
                "{} is set, but this device does not support TIMESTAMP_QUERY -- \
                 no GPU timing will be reported",
                TIMING_ENV
            );
            return None;
        }

        let size = MAX_PASSES as u64 * 2 * QUERY_SIZE;

        Some(GpuTimer {
            query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Pass Timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: MAX_PASSES as u32 * 2,
            }),
            resolve_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Timestamp Resolve Buffer"),
                size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Timestamp Readback Buffer"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            period_ns: queue.get_timestamp_period(),
            inside_encoders: device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
            labels: Vec::with_capacity(MAX_PASSES),
        })
    }

    /// Claims a pair of queries for the next pass, or `None` once the set is
    /// full.
    pub(crate) fn pass_writes(
        &mut self,
        label: &'static str,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let index = self.labels.len();
        if index >= MAX_PASSES {
            return None;
        }
        self.labels.push(label);

        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(index as u32 * 2),
            end_of_pass_write_index: Some(index as u32 * 2 + 1),
        })
    }

    /// Brackets encoder-level work in a pair of timestamps, the way
    /// [`Self::pass_writes`] brackets a compute pass.
    ///
    /// A compute pass carries its timestamps in its descriptor. A buffer copy
    /// has no descriptor, so this is the only way to give one a scope of its
    /// own -- and until it had one, the per-batch copy into the post buffer was
    /// the single piece of GPU work in the render loop that no measurement
    /// could see. `record` still runs on a device without
    /// `TIMESTAMP_QUERY_INSIDE_ENCODERS`, or once the query set is full; it is
    /// only the timing that is lost.
    pub(crate) fn encoder_scope(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        label: &'static str,
        record: impl FnOnce(&mut wgpu::CommandEncoder),
    ) {
        let index = self.labels.len();
        if !self.inside_encoders || index >= MAX_PASSES {
            record(encoder);
            return;
        }
        self.labels.push(label);

        encoder.write_timestamp(&self.query_set, index as u32 * 2);
        record(encoder);
        encoder.write_timestamp(&self.query_set, index as u32 * 2 + 1);
    }

    /// Encodes the resolve and the copy that bring this command buffer's
    /// timestamps back to the host. Call once, after the last timed pass.
    pub(crate) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let queries = self.labels.len() as u32 * 2;
        if queries == 0 {
            return;
        }
        encoder.resolve_query_set(&self.query_set, 0..queries, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback_buffer,
            0,
            queries as u64 * QUERY_SIZE,
        );
    }

    /// Reads back what the completed submission measured and arms the timer for
    /// the next one. The submission must have finished.
    pub(crate) fn take(&mut self, device: &wgpu::Device) -> Vec<PassTiming> {
        if self.labels.is_empty() {
            return Vec::new();
        }

        let bytes = self.labels.len() as u64 * 2 * QUERY_SIZE;
        let slice = self.readback_buffer.slice(..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        let mut report = Vec::new();
        {
            let data = slice.get_mapped_range().unwrap();
            let ticks: &[u64] = bytemuck::cast_slice(&data);
            for (i, label) in self.labels.iter().enumerate() {
                // Saturating because a driver is free to hand back timestamps
                // that do not increase, and a negative duration is worse than a
                // zero one.
                let elapsed = ticks[i * 2 + 1].saturating_sub(ticks[i * 2]);
                fold(
                    &mut report,
                    label,
                    1,
                    elapsed as f64 * self.period_ns as f64 / 1.0e6,
                );
            }
        }
        self.readback_buffer.unmap();
        self.labels.clear();

        let mut totals = TOTALS.lock().unwrap();
        for p in &report {
            fold(&mut totals, p.label, p.count, p.ms);
        }

        report
    }
}

/// Prints one dispatch's timings to stderr, the first of the three tiers
/// [`TIMING_ENV`] switches on.
///
/// The `wall` column is the validation that comes free with having the GPU
/// number at all: GPU busy time cannot exceed the wall clock the render loop
/// already measures around submit-plus-poll, so the two agreeing says the
/// dispatch is doing nothing but waiting for the GPU, and a gap is submission
/// latency that dispatch pipelining could recover.
pub(crate) fn report_dispatch(batch: u32, wall: Duration, passes: &[PassTiming]) {
    if passes.is_empty() {
        return;
    }

    let gpu: f64 = passes.iter().map(|p| p.ms).sum();
    let wall_ms = wall.as_secs_f64() * 1000.;

    let mut line = format!(
        "gpu timing: batch {:<3} wall {:8.3} ms  gpu {:8.3} ms ({:5.1}%)",
        batch,
        wall_ms,
        gpu,
        100. * gpu / wall_ms.max(f64::MIN_POSITIVE)
    );
    for p in passes {
        line += &format!("  {} {:.3}", p.label, p.ms);
        if p.count > 1 {
            line += &format!(" x{}", p.count);
        }
    }
    eprintln!("{}", line);
}
