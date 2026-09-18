# Outstanding work

Suggestions raised during the performance and integrator work that were **not**
implemented, plus limitations recorded deliberately so they don't get
re-litigated later.

Deliberately declined: a *relative* firefly clamp in the renderer, to replace
the absolute `CLAMPING_THRESHOLD = 10.0`. It would have to be relative to the
running mean, which depends on how many samples happened to land before the
current batch -- and `samples_per_batch` is re-tuned from measured dispatch
time, so the image would depend on GPU timing. That breaks the property
`test_denoise_improves_low_sample_image` rests on, that a noisy and a denoised
render trace bit-identical sample streams. It would also charge the bias to
every pixel permanently, and most to exactly the pixels whose true radiance is
driven by rare bright events. The denoiser has strictly better information --
the neighbourhood, the noise-free guide, the exact Welford variance -- runs once
after the fact, and a mistake there costs one frame rather than being baked into
the accumulator.

Already done and not repeated here: the 24-item performance sweep (BVH2 node
layout, binned SAH build, hot/cold primitive split, deferred shading, 2-D tiled
dispatch, sample batching, Russian roulette, closed-form sampling), next-event
estimation with MIS, per-pixel adaptive sampling, and narrowing the firefly
clamp to indirect light only. Also the half-pixel pixel-to-frame mapping, whose
`/ (width - 1)` divisor is now `/ width` -- the golden images turned out to be
insensitive to it (they downscale to 100x50 first), and both G-buffer tests now
assert the centre ray analytically, to 0.001 rather than 0.05.

---

## Recommended next

### A depth cutoff for NEE

NEE costs 1.89x per sample and pays for itself 2–4x on scenes lit by discrete
lights — but it is a **net ~13% loss** on scenes that are effectively ambient-lit
(`create_test_scene` has three huge lights, one a radius-10 sphere at distance
15, plus a bright background).

Direct lighting matters most at the first bounce. Skipping NEE past depth 1–2
would cut shadow-ray count sharply for a small variance increase, and would make
the ambient-lit case a win rather than a loss. Worth a knob and a measurement.

---

## Renderer

### Atlas mipmaps

The atlas is `mip_level_count: 1` with `FilterMode::Nearest` everywhere
(`renderer/mod.rs`). Incoherent bounce rays thrash the texture cache. Two real
blockers make this a feature rather than an optimisation:

- **LOD selection** needs ray differentials or ray cones, which the renderer does
  not track. Any LOD without them is a guess trading sharpness for cache hits.
- **Atlas bleeding**: mipmapping a packed atlas bleeds neighbouring textures into
  each other at higher levels unless `TexturePacker::pack`
  (`util/texture_processing.rs:61`) grows gutters between placements.

### Light BVH or power-weighted light selection

`light_pdf_value` (`ray_trace.wgsl:387`) loops over every light, and
`sample_light` chooses one uniformly. Fine at 1–3 lights; an emissive mesh
imported from an OBJ would make both terrible. Wants an alias table for
power-weighted selection, and a light BVH for the PDF sum.

### Denoiser — what is left

Done: `post/denoise.rs` is an edge-avoiding à-trous filter guided by a G-buffer
and by the per-pixel Welford variance, and the guide now follows the specular
chain — `trace_guide` (`renderer/ray_trace.wgsl`) traces one deterministic ray
per pixel per accumulation run to the first surface that is not a mirror or a
lens, and records its albedo (tinted by the chain), normal, total path length
and specular bounce count.

Measured on `create_specular_scene`, global linear RMSE is a wash against the
old primary-hit guide (8 spp: 0.0727 → 0.0740, 64 spp: 0.0281 → 0.0280, 200 spp:
0.0158 → 0.0157). That is the metric being the wrong one rather than the change
doing nothing: at a low sample count RMSE rewards blurring, and smearing the
reflection along the mirror is blurring. The difference is visible rather than
numeric — the refracted floor inside the glass sphere and the caustic under it
come out cleaner, and the reflected horizon in the mirror sphere stays a line.
`test_gbuffer_follows_specular_chain` (`renderer/mod.rs`) is what actually pins
the behaviour, analytically.

Also done: fireflies no longer survive the denoiser. An edge-avoiding filter
cannot remove one -- a firefly is an edge by every measure it has -- so what
decided their fate was the fade in `denoise_resolve.wgsl`, and it leaked at two
scales.

The broad leak was the fade's denominator. `relative = standard_error /
luminance(original)` divides by the pixel's own brightness, and noise moves a
pixel up as often as down: a pixel it moved up gets a larger denominator, a
smaller relative error, a smaller blend, and keeps more of the raw value that was
too bright, while one it moved down is filtered harder. The test was therefore
biased to preserve upward noise and remove downward noise, and preserved upward
noise is what a speckle is. The denominator is now the lower of the pixel's own
luminance and its guide-weighted neighbourhood level, which the variance
pre-pass publishes -- one-directional, so a pixel that is darker than its
surroundings, or one in a genuinely bright region where the two agree, behaves
exactly as before.

The narrow leak was the extreme tail, plus the case the fade never runs on. For
a pixel whose mean is carried by one outlier sample out of n, the Welford
variance of the mean comes out to the pixel's own value squared -- the standard
error and the mean cancel -- so the relative error is a constant: the square root
of the pre-filter kernel's centre share, 1/6.169, which is 0.4026 against a
`full_strength_error` of 0.4. Every such firefly landed on the threshold,
scale-free in how bright it was. And below two samples the fade does not run at
all, so at 1 spp nothing stood between an outlier and the image. So
`prefilter_variance` now also clamps a pixel sitting far above the neighbourhood
its guide says it belongs to, over the taps and weights it was already
gathering, and writes the result into the chain's working image. The two are
complementary rather than redundant: the fade's fix alone leaves 1701 specks at
1 spp where both together leave 357, and the clamp alone leaves over a thousand
at every sample count above 2.

Measured on `create_cornell_scene` through the display transform, counting
pixels more than 20 of 255 brighter than every neighbour, at 500x500:

    spp            1      2      5     10     16     64
    raw        18262  15237  10778   7515   5321    995
    before         0   2075   1872   1469   1042    167
    after          0     46      7     12     13     12

That is at strength 5; at strength 1 it is 2728/3068/2699/2105/1509/248 before
against 616/365/82/59/50/41 after. Both RMSE gates improved rather than
regressing -- 0.6720 to 0.6538 on the specular scene and 0.5513 to 0.5188 on the
diffuse one -- and a converged image still passes through close to untouched,
0.0045 against a gate of 0.02.

Worth knowing for whoever measures this next: the first version of the metric
counted outliers in linear radiance and it lied. ACES plus gamma 2.0 compresses
highlights hard, so a pixel pulled from twenty times its neighbourhood down to
three has lost 85% of its excess radiance and almost none of its visibility. The
linear metric scored a change at "all outliers removed" that was visually almost
indistinguishable from no change at all. `fireflies()` measures through the
display transform for that reason.

`test_denoise_removes_fireflies_at_every_sample_count` is the gate,
`test_denoise_preserves_bright_detail_when_converged` the control against
clamping a caustic, `test_denoise_strength_zero_is_the_identity` the guard on the
switch that turns the despeckle off with the filter, and `cornell_firefly_sweep`
the diagnostic the constants were chosen on.

Two smaller ones from the same work:

- **A compressed-luminance edge stop.** `w_colour` compares raw linear radiance
  while the image is shown through ACES and gamma 2.0, so noise inside a
  legitimately bright region survives the filter almost untouched -- its linear
  differences dwarf `lum_tolerance` even though they are invisible on screen.
  Replacing `luminance()` in the weight with `log(1 + L)` and carrying the
  variance through the same transform by the delta method,
  `Var(log(1+L)) ~= Var(L)/(1+L)^2`, would fix that. Deliberately not done with
  the despeckle above: it changes the filter's character everywhere and re-opens
  the tuned `sigma_colour` default, which wants its own `denoise_strength_sweep`
  and probably its own goldens. Note the textbook form makes firefly isolation
  slightly *worse* -- the tolerance is evaluated at the bright centre and shrinks
  faster than the numerator does -- so it needs the asymmetric variant, with
  `f'` at the tap rather than the centre.

- **Depth weight without a gradient.** `guide_weight` uses a relative depth test,
  which is scale-free and handles the background sentinel, but is more permissive
  than SVGF's screen-space gradient form at grazing incidence — a floor receding
  to the horizon will over-blur slightly near the horizon. The G-buffer's `.w`
  holds the material type in its low byte and the specular depth above it, so
  bits 12 upward are still free for a forward-difference gradient if it shows.
- **Interactive previews.** Post-processing still runs only on the final batch,
  so a camera drag is never denoised — which is the regime where it would help
  most. The chain already runs on a scratch copy, so this is now safe to add: it
  needs a per-processor "run on every batch" flag, and an accepted pop when bloom
  appears only at the end.

### Tone mapping — what is left

Done: `util/tone_map.rs` has a `ToneMapper` enum (ACES by default, plus Khronos
PBR Neutral, extended Reinhard and the old `Clamp`), applied in
`buffer_to_image` before gamma. It is a *display* transform, not a
post-processor, deliberately: the renderer publishes a linear HDR buffer, and
bloom and the denoiser have to keep seeing real radiance. All golden images
were regenerated against ACES.

Two things left:

- **Gamma is still `sqrt`, not sRGB.** `buffer_to_image` encodes with gamma 2.0
  where the Narkowicz ACES fit assumes an sRGB transfer. The difference is
  small (sRGB is ~2.2 with a linear toe) but it is a second display-transform
  decision left unmade, and it would shift the goldens again.
- **The desktop app's transfer function still differs.** Done: `ToneMapper::wgsl`
  emits the curve as WGSL, and `solstrale-desktop-rust` splices it into its blit
  shader, so all three of its display paths share one definition of the tone
  curve. What is still two things is the *transfer* function -- the viewport
  applies none and relies on an sRGB surface, `buffer_to_image` encodes with
  gamma 2.0 -- which is the same gap as the item above, seen from the other
  side. Fixing that one fixes both.

### Instancing (BLAS/TLAS)

Transforms are baked into vertices at construction, so repeated geometry costs
full duplicate storage. Note that `Bvh::new` now **flattens nested BVHs into one
global tree** — that was the right call absent instancing, but it is the decision
to revisit if instancing is added, since a nested `Bvh` would then carry a
transform and must stay a separate acceleration structure.

---

## Correctness and precision

### Camera ray is not normalised

`ray_trace.wgsl:1184` builds the primary ray direction without normalising, so
`rec.t` for the **first** path segment is in units of `|ray_direction|`
(≈ the focus distance) rather than world units. Every later segment is
normalised, so `path_length` mixes two scales.

Only observable through the light-attenuation falloff, which is the one feature
that depends on absolute path length. Pre-existing; left alone deliberately
because fixing it shifts the `light_attenuation_*` images again.

### Spheres cannot be transformed

`Sphere::new` (`hittable/sphere.rs:17`) takes no `Transformer`, unlike `Triangle`
and `Quad`. Scaling or rotating a sphere is therefore impossible through the
normal scene-building path.

---

## CPU and loader

### `tobj`'s parser is now the floor under OBJ loading

With the loader parallelised and the allocations out of `Bvh::new` and the
flattener, `tobj::load_obj` is about half of load time and is the only part left
that is still serial:

| | parse | total | was |
|---|---|---|---|
| `happy.obj` (98.6k tris) | 21.4 ms | 43.4 ms | 73.4 ms |
| `xyzrgb_dragon.obj` (249.9k tris) | 54.5 ms | 116.9 ms | 202.4 ms |

`load_obj_buf` (`tobj-4.0.3/src/lib.rs:1991`) is a `for line in reader.lines()`
loop — one heap-allocated `String` per line, 375k of them for the dragon — and
because `single_index` is off it takes `export_faces_multi_index` (`:1586`),
three `HashMap` lookups per face vertex. 4.0.5 has the identical loop, so
upgrading buys nothing.

Getting under it means a bespoke parser: read the file into one `Vec<u8>`, split
on `\n` with `memchr`, parse floats from `&str` slices, and split the work with
a two-pass rayon scheme (count records per chunk to assign offsets, then parse
chunks into preallocated arrays). Worth roughly another 2x on load, but it is a
multi-day job with a real correctness surface — negative and relative indices,
`f a/b/c` vs `a//c` vs `a`, polygon fans, `usemtl`/`o`/`g` grouping — and it
replaces a dependency that currently just works.

---

## Tooling and API

### No interactive viewer

No binary, no `src/bin/`, no `examples/`. `cargo run` does nothing — the only
entry points are the test suite and the benchmark. For a project whose stated
goal is learning path tracing and WGPU, being able to fly a camera around a scene
is worth a lot, and it surfaces behaviour that batch benchmarks do not.

The plumbing is already there and unused: the camera-update receiver, the abort
channel and `idle_after_rendering` all exist to support an interactive viewer
that nothing drives.

### `profile.sh` is broken

It runs `perf record ... target/release/profiling`, a binary that does not exist
in this repo. Either delete it or repoint it at the bench harness.

### Benchmark naming

`bvh_traversal/<n> false` still wraps the world in a top-level `Bvh`, so the
`use_bvh = false` case is not actually "no BVH" — it only skips the nested
sub-BVH. Misleading as a comparison.

---

## Deliberately declined

Recorded so they aren't reconsidered without new information.

- **Single-light fast path for `light_pdf_value`.** `sample_light` already
  intersects the chosen light, so its PDF could be computed without the extra
  traversal when `light_count == 1`. Worth ~2%, but it duplicates the PDF formula
  in two places where drift would silently bias the estimator. Not worth it.
- **Wavefront path tracing.** Splitting the megakernel into stages would cut
  material-branch divergence, but it is a full rewrite and premature — and the
  benefit is smaller on a small-wavefront iGPU.
- **Global early-exit for adaptive sampling.** Per-pixel adaptive sampling
  (`ray_trace.wgsl`'s `compute`) skips converged pixels inside each dispatch,
  but the CPU render loop still runs until `samples_per_pixel` batches are
  issued, even once every pixel has converged. Stopping the whole render early
  would need a new atomic-reduction pattern (a global "pixels still active"
  count) this crate doesn't have anywhere else, for benefit that vanishes on
  any scene with a persistently noisy region (a small light, a caustic). Not
  worth it until a scene shows up where it would matter.

---

## Known limitations (by design)

Correct but imperfect; documented so they read as choices rather than bugs.

- **ACES lifts midtones and skews saturated highlights.** The default curve is
  the per-channel Narkowicz fit, so linear 0.5 comes out at 0.616 rather than
  passing through, and bright reds and oranges drift toward yellow. Both are
  inherent to the cheap fit and were accepted for the filmic look;
  `ToneMapper::PbrNeutral` is in the enum for when neither is wanted.
- **Light seen through glass is still clamped.** A dielectric bounce puts the
  emitter at depth >= 1, so the indirect clamp covers it. Routing by "every
  vertex so far was specular" instead of `depth == 0` would exempt it, but it
  would equally exempt fuzzy-metal paths onto small lights, which are genuine
  fireflies. At a threshold of 10 the clamp barely fires on either, so this
  buys close to nothing today; revisit only alongside a scene built to show
  caustics.
- **Adaptive sampling got slower at high spp, and should have.** Unclamping
  direct light raises the per-sample variance the Welford estimator sees, so
  fewer pixels are declared converged: `adaptive_sampling/adaptive` (2000 spp)
  went 1.106 s -> 1.293 s. Raw tracing cost is unchanged —
  `adaptive_sampling/forced_off` and `render` (800x600, 64 spp) both moved
  within noise — so this is the skip heuristic no longer being fed an
  artificially quiet signal, not the shader doing more work per sample.
- **Dielectrics block NEE shadow rays.** Light through glass is found only by
  BSDF-sampled paths, at full MIS weight. Unbiased, but caustics stay noisy —
  the standard trade-off of naive NEE.
- **`Blend` is never `is_light()`.** A blend containing a `DiffuseLight` is not in
  the lights array, so it gets no NEE and is found by BSDF paths with weight 1.
  Consistent and unbiased, because the MIS PDF covers exactly the same set.
- **Rare MIS edge case.** A blend-emitter hit with a real light collinear behind
  it can be slightly under-weighted, since the PDF sum counts lights at any
  distance along the ray. Pre-existing structure, vanishingly rare.
- **The BVH leaf permutation trades build time for peak memory above ~400k
  primitives.** `permute_in_place` (`hittable/bvh.rs`) allocates nothing, where
  the staging-vector gather it replaced allocated a second full-size primitive
  array — 736.7 MB → 433.4 MB peak RSS on a 1M-primitive build. But all its work
  is random-access swaps, so it only runs *faster* while the array fits in
  last-level cache: −37% at 100k, +11% at 1M on a 96 MB-L3 part, crossing over
  sooner on a machine with less cache. Taken deliberately — build time is paid
  once at load, and the peak allocation is what decides whether a large scene
  fits at all. Revisit only if load time on huge scenes starts mattering more
  than footprint.
- **The guide ray is a centre ray, and six bounces deep.** `trace_guide`
  samples the pixel centre with no lens offset, so with a wide aperture it
  describes the point in focus rather than the defocused average the samples
  see; and a specular chain longer than `GUIDE_MAX_SPECULAR` (6) falls back to
  describing whatever specular surface it stalled on, which is the old
  primary-hit guide. Both are deliberate: a jittered guide ray would make
  neighbouring pixels disagree about what they are looking at, which is the one
  thing an edge stop cannot survive.
- **16.7M primitive cap.** The BVH leaf encoding uses a 24-bit offset
  (`hittable/bvh.rs`, `MAX_PRIMITIVES`), asserted at build time.
- **Golden images are lenient.** They downscale to 100x50 and compare RMS
  similarity at 0.9–0.95, which tolerates large quality changes. Any future
  integrator change needs a convergence check (render at 50/200/2000 spp and
  confirm the mean is flat), not just a green test run —
  `test_adaptive_sampling_convergence` is that check, and it now prints the
  absolute means, so `cargo test test_adaptive_sampling_convergence --
  --nocapture` before and after is what tells you whether energy moved. Its
  assert only checks flatness *within* one build, and is blind to a change
  that shifts all three means together. Note that output passes
  through `sqrt` gamma, which is concave — so a *noisier* image has a lower mean
  at identical linear radiance, and mean brightness must be compared at matched
  convergence.
