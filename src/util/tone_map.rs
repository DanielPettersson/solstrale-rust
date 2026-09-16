//! Tone mapping: the curve that takes unbounded linear radiance down to the
//! `[0, 1]` a display can show.
//!
//! This is a *display* transform, not a post-process. It runs in
//! [`buffer_to_image`](crate::util::wgpu_util::buffer_to_image), at the point
//! the renderer's linear HDR buffer becomes an 8-bit image, and it is the last
//! thing to touch the values before gamma. Nothing upstream of it -- bloom,
//! the denoiser, the accumulator itself -- ever sees a tone-mapped value, which
//! is what keeps those filters operating on real radiance.
//!
//! Before this existed the readback simply did `sqrt(L).min(0.999)`, so every
//! value above linear 1.0 was the same white no matter how much brighter it
//! really was. That clip, rather than the firefly clamp, was what decided what
//! a highlight looked like.

/// A curve mapping unbounded linear radiance to the `[0, 1]` display range.
///
/// All of these operate per channel on linear values, and all are applied
/// *before* gamma encoding.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
#[non_exhaustive]
pub enum ToneMapper {
    /// The ACES filmic curve, via Krzysztof Narkowicz's cheap rational fit.
    ///
    /// The de-facto standard in games and offline renderers: contrasty, with a
    /// long smooth shoulder that never hard-clips. Midtones lift slightly
    /// (linear 0.5 comes out at 0.616) and bright saturated colours skew --
    /// reds and oranges drift toward yellow -- which is the well-known cost of
    /// the per-channel fit. Pick [`ToneMapper::PbrNeutral`] instead if that
    /// hue shift matters more than the filmic look.
    #[default]
    Aces,

    /// The Khronos PBR Neutral curve, from the glTF group.
    ///
    /// Passes midtones through untouched and only begins compressing at 0.8,
    /// desaturating toward white at the very top. More faithful to the
    /// radiance actually computed than [`ToneMapper::Aces`], and free of its
    /// hue skew, at the cost of a flatter, less graded look.
    PbrNeutral,

    /// Extended Reinhard: `x(1 + x/W²) / (1 + x)`, reaching exactly 1.0 at the
    /// white point `W`.
    ///
    /// The textbook operator. Gentle and hue-preserving, but it darkens
    /// midtones noticeably and flattens contrast. `W` is the linear radiance
    /// that should map to display white; anything brighter is clipped.
    Reinhard {
        /// Linear radiance mapping to display white. Must be > 0.
        white_point: f32,
    },

    /// No tone mapping: clip at 1.0, which is what the renderer did before any
    /// of the above existed. Kept so the old look stays reachable, and as the
    /// baseline to compare the curves against.
    Clamp,
}

/// Ceiling applied to radiance before any curve sees it.
///
/// Every curve here squares its input, and `f32` squares overflow to infinity
/// above ~1.8e19 -- in ACES that makes the rational `inf / inf`, which is NaN,
/// and a NaN pixel casts to a black one rather than the white the value
/// obviously wanted. Clamping first is simpler than making each curve
/// overflow-safe, and 1e18 is many orders of magnitude above any radiance a
/// render produces, so the clamp is unreachable in practice.
const MAX_RADIANCE: f32 = 1e18;

/// Floor on Reinhard's white point.
///
/// The curve needs `white_point²`, and anything below ~1e-19 squares to
/// subnormal or zero, turning `x / w²` into infinity (or `0/0` -> NaN at
/// x = 0). 1e-4 is far below any useful white point and keeps the square
/// comfortably normal.
const MIN_WHITE_POINT: f32 = 1e-4;

/// The shoulder start, compression amount and desaturation knee of the Khronos
/// PBR Neutral curve, named as in the reference implementation.
const PBR_NEUTRAL_START: f32 = 0.8;
const PBR_NEUTRAL_D: f32 = 0.15;
const PBR_NEUTRAL_KNEE: f32 = 0.4;

/// Narkowicz's coefficients for the ACES rational fit.
const ACES_A: f32 = 2.51;
const ACES_B: f32 = 0.03;
const ACES_C: f32 = 2.43;
const ACES_D: f32 = 0.59;
const ACES_E: f32 = 0.14;

/// Name of the function [`ToneMapper::wgsl`] emits.
///
/// Exposed so a caller splicing the source into a larger shader can call it
/// without hard-coding the name and silently breaking when it changes.
pub const WGSL_FN: &str = "solstrale_tone_map";

impl ToneMapper {
    /// Maps one linear RGB triple into `[0, 1]`.
    ///
    /// The input is first clamped to `[0, MAX_RADIANCE]`. Negatives because a
    /// filter with a negative lobe can undershoot and the curves are only
    /// meaningful on non-negative radiance; NaN falls out of the same clamp,
    /// since `f32::max` returns the non-NaN operand.
    pub fn map(&self, linear: [f32; 3]) -> [f32; 3] {
        // Not `clamp`: it propagates NaN, where `max` returns the non-NaN
        // operand and so folds NaN to 0. That is the whole reason the bound is
        // written this way -- see `non_finite_input_does_not_reach_the_cast`.
        #[allow(clippy::manual_clamp)]
        let c = linear.map(|v| v.max(0.).min(MAX_RADIANCE));

        match *self {
            ToneMapper::Aces => c.map(aces),
            ToneMapper::PbrNeutral => pbr_neutral(c),
            ToneMapper::Reinhard { white_point } => c.map(|x| reinhard(x, white_point)),
            ToneMapper::Clamp => c.map(|x| x.min(1.)),
        }
    }

    /// WGSL source for this curve, defining a single function
    /// ```wgsl
    /// fn solstrale_tone_map(color: vec3<f32>) -> vec3<f32>
    /// ```
    /// (the name is [`WGSL_FN`]) with no bindings, uniforms or entry point, so
    /// it can be concatenated into any shader that needs to display the
    /// renderer's linear buffer.
    ///
    /// This exists because the display transform has more than one
    /// implementation: [`buffer_to_image`](crate::util::wgpu_util::buffer_to_image)
    /// runs on the CPU when an image is saved, while an interactive viewer
    /// blits the buffer to a surface on the GPU and never goes near it. Two
    /// hand-written copies of a curve drift, and a preview that disagrees with
    /// the saved file is a bug that is easy to look straight past.
    ///
    /// Every coefficient here is formatted in from the same constant
    /// [`ToneMapper::map`] uses, and `wgsl_matches_the_cpu_curve` runs this
    /// source on the GPU and checks it against `map` over the range, so the two
    /// cannot drift apart unnoticed.
    pub fn wgsl(&self) -> String {
        // NaN is folded to zero explicitly rather than left to `max`: WGSL does
        // not define which operand `max` returns for NaN, where Rust's `f32::max`
        // does, and this function has to agree with `map` on that.
        let prelude = format!(
            "fn {fn_name}(color: vec3<f32>) -> vec3<f32> {{\n    \
             let finite = select(color, vec3<f32>(0.0), color != color);\n    \
             let c = clamp(finite, vec3<f32>(0.0), vec3<f32>({max_radiance}));\n",
            fn_name = WGSL_FN,
            max_radiance = wgsl_f32(MAX_RADIANCE),
        );

        let body = match *self {
            ToneMapper::Aces => format!(
                "    let n = c * ({a} * c + {b});\n    \
                 let d = c * ({cc} * c + {dd}) + {e};\n    \
                 return clamp(n / d, vec3<f32>(0.0), vec3<f32>(1.0));\n",
                a = wgsl_f32(ACES_A),
                b = wgsl_f32(ACES_B),
                cc = wgsl_f32(ACES_C),
                dd = wgsl_f32(ACES_D),
                e = wgsl_f32(ACES_E),
            ),

            ToneMapper::PbrNeutral => format!(
                "    let start_compression = {start} - {d};\n    \
                 let x = min(c.r, min(c.g, c.b));\n    \
                 var offset = {d};\n    \
                 if (x < 2.0 * {d}) {{\n        \
                 offset = x - x * x / (4.0 * {d});\n    \
                 }}\n    \
                 let shifted = c - vec3<f32>(offset);\n    \
                 let peak = max(shifted.r, max(shifted.g, shifted.b));\n    \
                 if (peak < start_compression) {{\n        \
                 return shifted;\n    \
                 }}\n    \
                 let new_peak = 1.0 - (1.0 - start_compression) * (1.0 - start_compression)\n        \
                 / (peak + (1.0 - 2.0 * start_compression));\n    \
                 let scaled = shifted * new_peak / peak;\n    \
                 let g = 1.0 - 1.0 / ({knee} * (peak - new_peak) + 1.0);\n    \
                 return clamp(scaled * (1.0 - g) + vec3<f32>(new_peak) * g, vec3<f32>(0.0), vec3<f32>(1.0));\n",
                start = wgsl_f32(PBR_NEUTRAL_START),
                d = wgsl_f32(PBR_NEUTRAL_D),
                knee = wgsl_f32(PBR_NEUTRAL_KNEE),
            ),

            ToneMapper::Reinhard { white_point } => {
                let w = white_point.max(MIN_WHITE_POINT);
                format!(
                    "    let w2 = {w} * {w};\n    \
                     let mapped = (c * (vec3<f32>(1.0) + c / w2)) / (vec3<f32>(1.0) + c);\n    \
                     return clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0));\n",
                    w = wgsl_f32(w),
                )
            }

            ToneMapper::Clamp => "    return min(c, vec3<f32>(1.0));\n".to_string(),
        };

        format!("{prelude}{body}}}\n")
    }
}

/// Formats an `f32` as a WGSL float literal.
///
/// `{:?}` gives the shortest representation that round-trips, but for a whole
/// number that is `4`, which WGSL parses as an integer -- and `4 * c` against a
/// `vec3<f32>` is a type error rather than a wrong answer, so it would surface
/// as a shader compile failure. Appending `.0` where there is no `.` or
/// exponent keeps every emitted literal floating point.
fn wgsl_f32(v: f32) -> String {
    let s = format!("{:?}", v);
    if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("NaN") {
        s
    } else {
        format!("{s}.0")
    }
}

/// Narkowicz's fit to the ACES RRT+ODT, per channel.
fn aces(x: f32) -> f32 {
    ((x * (ACES_A * x + ACES_B)) / (x * (ACES_C * x + ACES_D) + ACES_E)).clamp(0., 1.)
}

/// `x(1 + x/W²) / (1 + x)`, which is 0 at 0 and exactly 1 at `W`.
///
/// A non-positive or tiny white point would make the numerator blow up or go
/// negative, so it is floored rather than trusted -- this is a public config
/// value. See [`MIN_WHITE_POINT`].
fn reinhard(x: f32, white_point: f32) -> f32 {
    let w = white_point.max(MIN_WHITE_POINT);
    ((x * (1. + x / (w * w))) / (1. + x)).clamp(0., 1.)
}

/// The Khronos PBR Neutral tone mapper.
///
/// Unlike the other two this is not per channel: it compresses the *darkest*
/// channel to decide how far the colour has been pushed, then mixes the whole
/// triple toward white by that amount. That is what keeps hue intact while
/// still desaturating the top end.
fn pbr_neutral(c: [f32; 3]) -> [f32; 3] {
    let start_compression = PBR_NEUTRAL_START - PBR_NEUTRAL_D;
    let desaturation = PBR_NEUTRAL_KNEE;

    let x = c[0].min(c[1]).min(c[2]);
    let offset = if x < 2. * PBR_NEUTRAL_D {
        x - x * x / (4. * PBR_NEUTRAL_D)
    } else {
        PBR_NEUTRAL_D
    };
    let c = [c[0] - offset, c[1] - offset, c[2] - offset];

    let peak = c[0].max(c[1]).max(c[2]);
    if peak < start_compression {
        return c;
    }

    let new_peak = 1. - (1. - start_compression) * (1. - start_compression)
        / (peak + (1. - 2. * start_compression));
    let c = c.map(|v| v * new_peak / peak);

    let g = 1. - 1. / (desaturation * (peak - new_peak) + 1.);
    c.map(|v| (v * (1. - g) + new_peak * g).clamp(0., 1.))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The values the curve is chosen for: the shoulder. A clip would put all
    /// three of these at 1.0 and lose every distinction between them.
    #[test]
    fn aces_rolls_highlights_off_instead_of_clipping() {
        let at = |v: f32| ToneMapper::Aces.map([v, v, v])[0];

        assert!((at(1.) - 0.8038).abs() < 0.001, "was {}", at(1.));
        assert!((at(4.) - 0.9734).abs() < 0.001, "was {}", at(4.));

        // Strictly increasing all the way up, which is the whole point: the
        // image still distinguishes a 4x from a 16x highlight.
        assert!(at(16.) > at(4.) && at(4.) > at(1.));
        assert!(at(16.) <= 1.);
    }

    /// Midtones lift rather than pass through -- worth pinning, because it is
    /// the one thing about ACES that surprises people coming from a clip.
    #[test]
    fn aces_lifts_midtones() {
        let half = ToneMapper::Aces.map([0.5, 0.5, 0.5])[0];
        assert!((half - 0.6163).abs() < 0.001, "was {}", half);
    }

    /// Black has to stay black under every curve, or the background of every
    /// render lifts to grey.
    #[test]
    fn black_maps_to_black() {
        for m in [
            ToneMapper::Aces,
            ToneMapper::PbrNeutral,
            ToneMapper::Reinhard { white_point: 4. },
            ToneMapper::Clamp,
        ] {
            assert_eq!(m.map([0., 0., 0.]), [0., 0., 0.], "{:?}", m);
        }
    }

    /// A filter with a negative lobe can undershoot; that must not come back as
    /// a negative or NaN pixel.
    #[test]
    fn negatives_are_floored_not_propagated() {
        for m in [
            ToneMapper::Aces,
            ToneMapper::PbrNeutral,
            ToneMapper::Reinhard { white_point: 4. },
            ToneMapper::Clamp,
        ] {
            let out = m.map([-1., -0.001, 0.5]);
            assert!(
                out.iter().all(|v| v.is_finite() && *v >= 0. && *v <= 1.),
                "{:?} gave {:?}",
                m,
                out
            );
        }
    }

    /// Every curve has to stay inside the display range for any finite input,
    /// since the caller multiplies the result by 256 and casts to u8.
    #[test]
    fn output_is_always_in_display_range() {
        for m in [
            ToneMapper::Aces,
            ToneMapper::PbrNeutral,
            ToneMapper::Reinhard { white_point: 4. },
            ToneMapper::Clamp,
        ] {
            for v in [0., 0.001, 0.5, 1., 4., 100., 1e6, f32::MAX] {
                let out = m.map([v, v * 0.5, v * 0.25]);
                assert!(
                    out.iter().all(|o| o.is_finite() && (0. ..=1.).contains(o)),
                    "{:?} at {} gave {:?}",
                    m,
                    v,
                    out
                );
            }
        }
    }

    /// Reinhard's white point is defined as the linear value reaching display
    /// white, so that is what it has to do.
    #[test]
    fn reinhard_hits_white_at_its_white_point() {
        let m = ToneMapper::Reinhard { white_point: 4. };
        assert!((m.map([4., 4., 4.])[0] - 1.).abs() < 1e-5);

        // A degenerate white point must not produce NaN -- including at zero
        // radiance, where an unfloored `w * w` makes `x / w²` a `0 / 0`.
        for w in [0., -1., f32::MIN_POSITIVE, 1e-30] {
            let degenerate = ToneMapper::Reinhard { white_point: w };
            for v in [0., 0.5, 1e6] {
                let out = degenerate.map([v, v, v]);
                assert!(
                    out.iter().all(|o| o.is_finite() && (0. ..=1.).contains(o)),
                    "white_point {} at {} gave {:?}",
                    w,
                    v,
                    out
                );
            }
        }
    }

    /// NaN and infinity have to come out as a real colour. A NaN survives to
    /// the `as u8` cast as *black*, which is the opposite of what an overflowed
    /// highlight should look like, and the curves square their input, so a
    /// merely large finite value reaches infinity on its own.
    #[test]
    fn non_finite_input_does_not_reach_the_cast() {
        for m in [
            ToneMapper::Aces,
            ToneMapper::PbrNeutral,
            ToneMapper::Reinhard { white_point: 4. },
            ToneMapper::Clamp,
        ] {
            for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
                let out = m.map([v, v, v]);
                assert!(
                    out.iter().all(|o| o.is_finite() && (0. ..=1.).contains(o)),
                    "{:?} at {} gave {:?}",
                    m,
                    v,
                    out
                );
            }

            // And a blown-out highlight reads as white, not as black.
            assert_eq!(m.map([f32::INFINITY; 3]), [1., 1., 1.], "{:?}", m);
        }
    }

    /// The property PbrNeutral exists for: a saturated colour keeps its hue
    /// where ACES skews it. Pure red should stay pure red, with no green or
    /// blue mixed in until the desaturation knee.
    #[test]
    fn pbr_neutral_keeps_midtone_hue() {
        let out = ToneMapper::PbrNeutral.map([0.5, 0., 0.]);
        assert!(out[1] < 1e-6 && out[2] < 1e-6, "was {:?}", out);
        assert!((out[0] - 0.5).abs() < 1e-6, "was {:?}", out);
    }

    /// Clamp has to reproduce exactly what the readback did before tone mapping
    /// existed, or the old look is no longer reachable.
    #[test]
    fn clamp_is_the_old_behaviour() {
        assert_eq!(ToneMapper::Clamp.map([0.25, 1.5, 1.]), [0.25, 1., 1.]);
    }

    /// Emitted literals have to be floating point. A whole-number coefficient
    /// formatted as `4` makes `4 * c` a type error against a `vec3<f32>`, so
    /// this would show up as a shader compile failure rather than a wrong
    /// colour -- but only for whichever curve happened to have one.
    #[test]
    fn emitted_literals_are_floats() {
        assert_eq!(wgsl_f32(4.), "4.0");
        assert_eq!(wgsl_f32(0.5), "0.5");
        assert_eq!(wgsl_f32(1e18), "1e18");
        assert_eq!(wgsl_f32(0.), "0.0");
    }

    /// The point of [`ToneMapper::wgsl`]: the GPU curve and the CPU curve are
    /// separate implementations, and a preview that disagrees with the saved
    /// file is exactly the kind of bug nobody notices. Runs the emitted source
    /// over the range and checks it against `map`.
    #[test]
    fn wgsl_matches_the_cpu_curve() {
        use crate::util::wgpu_util::{
            add_compute_pass, bind_group, bind_group_layout, compute_pipeline,
            get_result_from_buffer, get_wgpu_device_and_queue, storage_binding,
        };
        use wgpu::util::DeviceExt;

        let (device, queue) = get_wgpu_device_and_queue();

        // Across the shoulder, either side of PbrNeutral's two branch points
        // (0.3 and 0.65), and far enough past 1.0 to exercise the clamp.
        let mut inputs: Vec<[f32; 4]> = Vec::new();
        for v in [
            0., 0.001, 0.05, 0.1, 0.2, 0.29, 0.3, 0.31, 0.5, 0.64, 0.65, 0.66, 0.8, 0.999, 1.0,
            1.5, 2., 4., 10., 100., 1e5, 1e18,
        ] {
            // Grey, and three saturated triples, since PbrNeutral is not a
            // per-channel curve and only the latter exercise its desaturation.
            inputs.push([v, v, v, 0.]);
            inputs.push([v, v * 0.5, v * 0.25, 0.]);
            inputs.push([v, 0., 0., 0.]);
            inputs.push([0., v, v * 0.1, 0.]);
        }

        for mapper in [
            ToneMapper::Aces,
            ToneMapper::PbrNeutral,
            ToneMapper::Reinhard { white_point: 4. },
            ToneMapper::Clamp,
        ] {
            let source = format!(
                "@group(0) @binding(0) var<storage, read_write> data: array<vec4<f32>>;\n\n\
                 {}\n\
                 @compute @workgroup_size(1)\n\
                 fn compute(@builtin(global_invocation_id) id: vec3<u32>) {{\n    \
                 data[id.x] = vec4<f32>({}(data[id.x].xyz), 1.0);\n\
                 }}\n",
                mapper.wgsl(),
                WGSL_FN,
            );

            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
            });
            let layout = bind_group_layout(device, &[storage_binding(false, 16)]);
            let pipeline = compute_pipeline(device, &layout, &module, &[]);

            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&inputs),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            let group = bind_group(
                device,
                &layout,
                &[wgpu::BindingResource::Buffer(
                    buffer.as_entire_buffer_binding(),
                )],
            );

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            add_compute_pass(&mut encoder, &pipeline, &group, inputs.len() as u32);

            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (inputs.len() * 16) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&buffer, 0, &staging, 0, (inputs.len() * 16) as u64);
            queue.submit(Some(encoder.finish()));

            let gpu: Vec<[f32; 4]> = get_result_from_buffer(device, &staging);

            for (input, got) in inputs.iter().zip(gpu.iter()) {
                let want = mapper.map([input[0], input[1], input[2]]);
                for ch in 0..3 {
                    assert!(
                        (got[ch] - want[ch]).abs() < 1e-4,
                        "{:?} at {:?}: gpu {:?} vs cpu {:?}",
                        mapper,
                        &input[..3],
                        &got[..3],
                        want
                    );
                }
            }
        }
    }
}
