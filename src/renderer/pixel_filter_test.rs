//! What the pixel reconstruction filter is worth, on a scene built to show it.
//!
//! The filter is invisible in the golden suite -- those images are scored on a
//! 100x50 downsample, which resamples the very pixels the filter shapes -- so
//! the measurement needs its own scene and its own metric. Both are below.
//!
//! The scene is a high-contrast edge at a shallow angle, which is where the box
//! filter's stair-stepping is most visible, and it is deliberately free of
//! Monte Carlo noise: an emissive quad against a black background, seen
//! directly, so every camera ray returns either the emission or nothing and the
//! only thing varying between samples is where in the pixel the ray went. What
//! the image contains is the filter and nothing else.

use std::sync::mpsc::channel;

use image::{Rgb, RgbImage};

use crate::camera::CameraConfig;
use crate::geo::transformation::NopTransformer;
use crate::geo::vec3::Vec3;
use crate::hittable::{Bvh, Hittables, Quad};
use crate::material::DiffuseLight;
use crate::renderer::{RenderConfig, Renderer, Scene};
use crate::util::wgpu_util::{get_result_from_buffer, get_wgpu_device_and_queue};

const WGSL: &str = include_str!("ray_trace.wgsl");

const WIDTH: usize = 240;
const HEIGHT: usize = 120;

/// Rise per pixel of column. Shallow enough that a box filter's transition row
/// spans many columns, which is what makes the steps long and visible, and not
/// a simple fraction, so the edge's sub-pixel phase walks through the whole
/// range across the frame instead of repeating every few columns.
const EDGE_SLOPE: f64 = 0.1301;

/// Radiance of the emissive half-plane. Above 1 so nothing about this depends
/// on where a display clips.
const EMISSION: f32 = 4.;

// ---------------------------------------------------------------------------
// The tent warp, mirrored on the CPU
// ---------------------------------------------------------------------------

/// Transcribed from `pixel_filter_warp` in `ray_trace.wgsl`.
fn tent_warp(u: f64, radius: f64) -> f64 {
    if u < 0.5 {
        radius * ((2. * u).sqrt() - 1.)
    } else {
        radius * (1. - (2. - 2. * u).sqrt())
    }
}

/// The shader is the definition; this file only mirrors it. If the two drift,
/// every number below is measuring something that is no longer shipped.
#[test]
fn the_warp_matches_the_shader() {
    assert!(
        WGSL.contains("return pixel_filter_radius * (sqrt(2.0 * u) - 1.0);"),
        "the shader's lower branch of tent_warp has changed"
    );
    assert!(
        WGSL.contains("return pixel_filter_radius * (1.0 - sqrt(2.0 - 2.0 * u));"),
        "the shader's upper branch of tent_warp has changed"
    );
    assert!(
        WGSL.contains("override pixel_filter_radius: f32 = 1.0;"),
        "the shader's default filter radius has changed"
    );
}

/// An inverse CDF has to be a monotone bijection onto the filter's support, or
/// it is warping the sampler's stratification into something else.
#[test]
fn the_warp_is_monotone_onto_the_support() {
    let radius = 1.;
    let n = 100_000;

    let mut previous = f64::NEG_INFINITY;
    for i in 0..=n {
        let u = i as f64 / n as f64;
        let x = tent_warp(u.min(1. - 1e-12), radius);
        assert!(x >= previous, "not monotone at u = {u}");
        assert!(x.abs() <= radius + 1e-9, "left the support at u = {u}: {x}");
        previous = x;
    }

    assert!((tent_warp(0., radius) + radius).abs() < 1e-12);
    assert!(tent_warp(0.5, radius).abs() < 1e-12);
    assert!((tent_warp(1. - 1e-15, radius) - radius).abs() < 1e-6);
}

/// And it has to produce the tent, not merely something centred and bounded.
/// Checked against the tent's own CDF, which is the thing being inverted.
#[test]
fn the_warp_produces_a_tent() {
    let radius = 1.;
    // CDF of a tent on [-r, r]: the integral of (1 - |x|/r) / r.
    let cdf = |x: f64| {
        let t = x / radius;
        if t <= -1. {
            0.
        } else if t <= 0. {
            0.5 * (1. + t) * (1. + t)
        } else if t <= 1. {
            1. - 0.5 * (1. - t) * (1. - t)
        } else {
            1.
        }
    };

    for i in 0..=1000 {
        let u = i as f64 / 1000.;
        let x = tent_warp(u.min(1. - 1e-12), radius);
        assert!(
            (cdf(x) - u).abs() < 1e-9,
            "warp is not the tent's inverse CDF at u = {u}: cdf({x}) = {}",
            cdf(x)
        );
    }
}

// ---------------------------------------------------------------------------
// The scene
// ---------------------------------------------------------------------------

/// An emissive half-plane with a shallow straight edge, filling the frame, with
/// nothing else in the world.
///
/// `min_samples_per_pixel` is above `samples_per_pixel` so adaptive sampling
/// never retires a pixel: the pixels away from the edge have exactly zero
/// sample variance and would retire at once, leaving the edge pixels with a
/// different sample count from their neighbours and the comparison measuring
/// that instead.
fn edge_scene(pixel_filter_radius: f32) -> Scene {
    // Wide and tall enough to run past the frame on every side, so the only
    // edge in the image is the one being measured.
    let half = 40.;
    let quad = Quad::new(
        // The edge passes through y = 0 at x = 0 and rises with EDGE_SLOPE.
        // Downward side first, so the quad's normal faces the camera: a
        // DiffuseLight emits from its front face only.
        Vec3::new(-half, -half * EDGE_SLOPE, 0.),
        Vec3::new(0., -2. * half, 0.),
        Vec3::new(2. * half, 2. * half * EDGE_SLOPE, 0.),
        DiffuseLight::new(EMISSION as f64, EMISSION as f64, EMISSION as f64, None).into(),
        &NopTransformer(),
    );

    let world: Vec<Hittables> = vec![quad.into()];

    Scene {
        world: Bvh::new(world).into(),
        camera: CameraConfig {
            vertical_fov_degrees: 30.,
            aperture_size: 0.,
            look_from: Vec3::new(0., 0., 10.),
            look_at: Vec3::new(0., 0., 0.),
            up: Vec3::new(0., 1., 0.),
        },
        background_color: Vec3::new(0., 0., 0.),
        render_config: RenderConfig {
            width: WIDTH,
            height: HEIGHT,
            samples_per_pixel: 256,
            samples_per_batch: 256,
            max_depth: 2,
            min_samples_per_pixel: 1024,
            pixel_filter_radius,
            ..Default::default()
        },
    }
}

fn render(radius: f32) -> Vec<f32> {
    let (device, queue) = get_wgpu_device_and_queue();

    let (progress_sender, progress_receiver) = channel();
    let (_camera_sender, camera_receiver) = channel();
    let (_abort_sender, abort_receiver) = channel();

    let mut renderer = Renderer::new(edge_scene(radius), device, queue).unwrap();
    renderer
        .render(&progress_sender, &camera_receiver, &abort_receiver, false)
        .unwrap();
    drop(progress_sender);

    let output_buffer = progress_receiver
        .into_iter()
        .last()
        .expect("render reported no progress")
        .output_buffer;

    let size = (WIDTH * HEIGHT * 16) as u64;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Pixel Filter Test Staging Buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, size);
    queue.submit(Some(encoder.finish()));

    get_result_from_buffer::<f32>(device, &staging_buffer)
}

/// Coverage -- the fraction of the pixel's filter that fell on the emissive
/// side -- which is what the rendered radiance is a multiple of.
fn coverage(pixels: &[f32], x: usize, y: usize) -> f64 {
    (pixels[(y * WIDTH + x) * 4] / EMISSION) as f64
}

// ---------------------------------------------------------------------------
// The metric
// ---------------------------------------------------------------------------

/// The sharpest pixel-to-pixel step across the edge in one column.
///
/// For a straight edge every filter's output is an exact sample of its own
/// smooth profile, so no metric on the pixel values can catch a filter "in the
/// act" of aliasing. What differs is the shape of that profile, and what the
/// eye reads as a staircase is the profile being steep enough that the step
/// lands almost entirely between two rows.
fn step_contrast(pixels: &[f32], x: usize) -> f64 {
    (0..HEIGHT - 1)
        .map(|y| (coverage(pixels, x, y + 1) - coverage(pixels, x, y)).abs())
        .fold(0., f64::max)
}

/// The staircase, in coverage units: how much the sharpest step *varies* along
/// the edge.
///
/// A well reconstructed shallow edge looks the same all the way along. The box
/// does not: as the edge drifts through a pixel it alternates between a soft
/// blend across two rows and a near-hard jump between them, once every
/// `1 / EDGE_SLOPE` columns, and that alternation is what reads as steps. The
/// variation rather than the mean, deliberately -- the mean is the blur, which
/// is the other column of the table and moves the opposite way.
fn staircase(pixels: &[f32]) -> f64 {
    let steps: Vec<f64> = (0..WIDTH).map(|x| step_contrast(pixels, x)).collect();
    let mean = steps.iter().sum::<f64>() / steps.len() as f64;
    (steps.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / steps.len() as f64).sqrt()
}

/// The worst hard jump anywhere along the edge, which is the single most
/// visible step in the image.
fn worst_step(pixels: &[f32]) -> f64 {
    (0..WIDTH)
        .map(|x| step_contrast(pixels, x))
        .fold(0., f64::max)
}

/// How far the transition is smeared, in pixels: the mean 10%-to-90% coverage
/// distance down a column. The price paid for the numbers above.
fn edge_width(pixels: &[f32]) -> f64 {
    let level = |x: usize, target: f64| -> Option<f64> {
        for y in 0..HEIGHT - 1 {
            let (a, b) = (coverage(pixels, x, y), coverage(pixels, x, y + 1));
            if (a - target) * (b - target) <= 0. && a != b {
                return Some(y as f64 + (target - a) / (b - a));
            }
        }
        None
    };

    let mut widths = Vec::new();
    for x in 0..WIDTH {
        if let (Some(lo), Some(hi)) = (level(x, 0.1), level(x, 0.9)) {
            widths.push((hi - lo).abs());
        }
    }
    widths.iter().sum::<f64>() / widths.len() as f64
}

/// The measurement the issue asks for, as a gate.
///
/// On a Radeon RX 5700 XT at 256 spp, from `radius_sweep`:
///
/// | radius     | staircase | worst step | edge width |
/// |------------|-----------|------------|------------|
/// | 0 (box)    | 0.142     | 0.977      | 1.332 px   |
/// | 0.5        | 0.150     | 1.000      | 1.156 px   |
/// | 0.75       | 0.119     | 0.895      | 1.332 px   |
/// | 1 (tent)   | 0.075     | 0.758      | 1.520 px   |
/// | 1.5        | 0.034     | 0.563      | 1.943 px   |
///
/// The 0.5 row is the interesting one: a filter that stays inside the pixel is
/// *worse* than the box on both counts, which is why the default is wider than
/// one. Past 1 the staircase keeps falling and the blur keeps rising, with no
/// knee to pick out -- 1 is the conventional two-pixel-wide tent, and the point
/// where the noise this costs is still in single-digit percentages on the
/// scenes in the suite. See `RenderConfig::pixel_filter_radius`.
#[test]
fn the_tent_takes_the_steps_out_of_a_shallow_edge() {
    let box_px = render(0.);
    let tent_px = render(1.);

    let (box_staircase, tent_staircase) = (staircase(&box_px), staircase(&tent_px));
    let (box_worst, tent_worst) = (worst_step(&box_px), worst_step(&tent_px));
    let (box_width, tent_width) = (edge_width(&box_px), edge_width(&tent_px));

    println!(
        "box:  staircase {box_staircase:.4}  worst step {box_worst:.4}  edge width {box_width:.3} px\n\
         tent: staircase {tent_staircase:.4}  worst step {tent_worst:.4}  edge width {tent_width:.3} px"
    );

    assert!(
        tent_staircase < box_staircase * 0.6,
        "the tent should take at least 40% off the staircase: \
         box {box_staircase:.4}, tent {tent_staircase:.4}"
    );
    assert!(
        tent_worst < box_worst * 0.85,
        "the tent should have no near-hard step left: \
         box {box_worst:.4}, tent {tent_worst:.4}"
    );
    // The other side of the trade, pinned so it cannot quietly grow. A tent of
    // radius 1 has the same peak density as a box of width 1, so it costs far
    // less sharpness than doubling the support suggests -- the extra width is
    // all in the tails.
    assert!(
        tent_width < box_width * 1.25,
        "the tent is blurring more than its tails can explain: \
         box {box_width:.3} px, tent {tent_width:.3} px"
    );
}

/// A tent narrower than the pixel is the version issue #68 asked for, on the
/// grounds that a wider one would need splatting. It is a regression: more
/// concentrated than the box it replaces, so the edge it leaves is harder.
/// Pinned because it is the reason the default is not 0.5.
#[test]
fn a_filter_inside_the_pixel_is_worse_than_the_box() {
    let box_px = render(0.);
    let narrow_px = render(0.5);

    assert!(
        worst_step(&narrow_px) >= worst_step(&box_px),
        "a within-pixel tent was expected to sharpen the step, not soften it: \
         box {:.4}, radius 0.5 {:.4}",
        worst_step(&box_px),
        worst_step(&narrow_px)
    );
    assert!(
        staircase(&narrow_px) > staircase(&box_px) * 0.9,
        "a within-pixel tent was not expected to help the staircase: \
         box {:.4}, radius 0.5 {:.4}",
        staircase(&box_px),
        staircase(&narrow_px)
    );
}

/// The visual pair issue #68 asks for: the same shallow edge under each filter,
/// cropped to where it crosses a pixel boundary and magnified with nearest
/// neighbour, so the pixels are the pixels.
/// `cargo test pixel_filter_visual_pair -- --ignored`
#[test]
#[ignore]
fn pixel_filter_visual_pair() {
    const ZOOM: u32 = 12;
    const CROP_W: u32 = 40;
    const CROP_H: u32 = 14;

    for (name, radius) in [("box", 0.), ("tent", 1.)] {
        let px = render(radius);
        // Centre the crop on the edge, at a column where it is crossing a row
        // boundary -- the phase where the box has nothing left to say.
        let x0 = 0;
        let y0 = (0..HEIGHT)
            .find(|&y| coverage(&px, CROP_W as usize / 2, y) > 0.5)
            .unwrap_or(HEIGHT / 2)
            .saturating_sub(CROP_H as usize / 2) as u32;

        let mut img = RgbImage::new(CROP_W * ZOOM, CROP_H * ZOOM);
        for y in 0..CROP_H * ZOOM {
            for x in 0..CROP_W * ZOOM {
                let c = coverage(&px, (x0 + x / ZOOM) as usize, (y0 + y / ZOOM) as usize);
                // Gamma 2.0, matching the readback encode; see LIMITATIONS.md.
                let v = (c.max(0.).sqrt().min(0.999) * 256.) as u8;
                img.put_pixel(x, y, Rgb([v, v, v]));
            }
        }
        img.save(format!("tests/output/out_actual_pixel_filter_{name}.png"))
            .unwrap();
    }
}

/// What the radius buys and what it costs, which is what picked the default.
#[test]
#[ignore]
fn radius_sweep() {
    for radius in [0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 2.] {
        let px = render(radius);
        println!(
            "radius {radius:4}: staircase {:.4}  worst step {:.4}  edge width {:.3} px",
            staircase(&px),
            worst_step(&px),
            edge_width(&px)
        );
    }
}

/// The raw vertical profiles either side of the edge, which is where the shape
/// of the thing being measured is actually visible: the box puts one pixel in
/// the transition and saturates on both sides of it, the tent puts two.
/// `cargo test dump_profiles -- --ignored --nocapture`
#[test]
#[ignore]
fn dump_profiles() {
    for (name, r) in [("box", 0.), ("tent", 1.)] {
        let px = render(r);
        println!("--- {name} (radius {r}) ---");
        for x in [0usize, 4, 8, 12, 16, 20] {
            let first = (0..HEIGHT)
                .find(|&y| coverage(&px, x, y) > 0.001)
                .unwrap_or(0);
            let lo = first.saturating_sub(3);
            let col: Vec<String> = (lo..(lo + 9).min(HEIGHT))
                .map(|y| format!("{:.3}", coverage(&px, x, y)))
                .collect();
            println!("x={x:3} rows {lo}..: {}", col.join(" "));
        }
    }
}
