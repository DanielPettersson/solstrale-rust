# Known limitations and settled decisions

What is true of this renderer on purpose, what was tried and measured and
rejected, and the numbers those decisions rest on.

Nothing here is a to-do -- open work lives in the GitHub issue tracker. This is
what to read before changing anything in the integrator or the denoiser, because
several of the obvious metrics for this kind of work are recorded below as having
been measured and found to lie.

---

## Known limitations (by design)

Correct but imperfect; documented so they read as choices rather than bugs.

- **"Converged" no longer means "untouched".** The denoiser used to change a
  2000 spp render of the test scene by 0.2% of linear RMSE, and now changes it by
  2.4%. That is the fade's criterion working as intended rather than a
  regression: that image still carries about two code values of visible grain,
  and the filter is now built to remove visible grain rather than to retire on a
  relative error. What it costs is real, though, and it is charged in linear
  radiance -- `denoise_strength_sweep` shows the denoiser going from neutral to
  net-harmful in linear RMSE above roughly 200 spp on the diffuse scene and 800
  on the specular one, while `denoise_display_sweep` still shows it helping at
  512. The two metrics genuinely disagree, and which one is right depends on
  whether the output is being looked at or measured. `test_denoise_is_near_identity_at_high_samples`
  now pins the visible quantity: half the frame moves by under one code value
  (0.478 measured), and four times the samples still more than halves it (0.33),
  which is what rules out a permanent blur floor.

- **The fade's criterion is a luminance, not three.** `displayed()` in
  `denoise_resolve.wgsl` evaluates the display curve on a grey of the pixel's
  level, because the chain carries exactly one variance and it is a luminance
  variance. On a saturated surface that understates the noise by up to 2x -- a
  pure primary's luminance weight is as low as 0.0722 -- so a coloured wall keeps
  a little more grain than a white one at the same sample count. A
  channel-proportional gain, `max_c g(c_i) * c_i / Y`, reduces to `g(Y)` on grey
  and would fix it, at the cost of dragging the pixel's own chroma -- pure noise
  at 2 spp -- into the criterion.

- **The denoiser has an opinion about the display curve.** It has to: the fade
  measures visibility, which is meaningless without one. But the curve used for
  the readback is chosen by whoever calls `buffer_to_image`, independently of the
  post-processing chain, so `with_tone_mapper` exists to keep them in step and
  nothing enforces it. Getting it wrong is bounded rather than catastrophic -- at
  linear 0.64 the ACES slope is 69 code values per unit radiance against a plain
  gamma's 160 -- but it is the same "the display transform lives in more than one
  place" complaint recorded in the tone-mapping section, seen from a third side.

- **The variance-correlation correction assumes unit edge-stop weights.** Its
  derivation is for the unweighted à-trous kernel. `denoise_atrous.wgsl` fades
  each factor in on the fraction of the kernel that actually survived its
  weights, which is exact at both ends -- a kernel reduced to its centre tap
  introduces no correlation, a kernel with every weight at one is the derived
  case -- but a first-order interpolation in between. It errs toward leaving
  noise rather than blurring detail, which is the safe direction, and the
  specular scene is the control: applying the factors whole cost it 18% of its
  RMSE against a converged reference.

- **ACES lifts midtones and skews saturated highlights.** The default curve is
  the per-channel Narkowicz fit, so linear 0.5 comes out at 0.616 rather than
  passing through, and bright reds and oranges drift toward yellow. Both are
  inherent to the cheap fit and were accepted for the filmic look;
  `ToneMapper::PbrNeutral` is in the enum for when neither is wanted.
- **The shadow terminator.** Interpolating a shading normal across a triangle
  makes the mesh look smooth, but the mesh is still faceted, and the two
  disagree most where they are most visible: near the light boundary. A
  direction that passes the shading-normal cosine test can be blocked by the
  mesh's own geometry, and the BSDF layer gates both `bsdf_eval` (so next-event
  estimation) and `bsdf_sample` on `dot(geometric_normal, wi) > 0` as well -- without
  that gate an interpolated normal facing a light the facet faces away from
  admits light straight through a closed surface, which is by far the worse
  artefact. What the gate costs is a band of missing light along the
  terminator of a coarsely tessellated mesh: `create_smooth_vs_flat_scene`'s
  right-hand sphere, at 10 by 6, shows it plainly, and it fades as the
  tessellation gets finer.

  The gate is on the primitive's real normal rather than on "is this a smooth
  triangle", so it covers normal mapping as well, where a map steep enough to
  tilt the shading normal past the true surface was admitting light through it.
  That is a change to an existing image and it was measured:
  `normal_mapping_sphere_1` moves by 2.9 code values RMS, concentrated on the
  steepest bump-mapped terrain and on the pole band the tangent fix below
  touches, and still scores 0.995 against its golden.

  Shipped rather than fixed, because the fix is self-contained and is far
  easier to justify now that a scene demonstrates the artefact. Chiang et al.
  2019 (*Taming the Shadow Terminator*) softens the NEE term with a bounding
  ratio derived from the corner normals; the Estevez shading-point shift Cycles
  uses moves the shading point onto the interpolated surface instead. Either
  is a follow-up, and neither changes anything recorded above.

- **A flat triangle's shading normal takes a 16-bit round trip.** Flat shading
  is expressed by giving all three corners the geometric normal rather than by
  a flag, so a flat triangle's normal goes out through `pack_oct` and comes
  back through `oct_decode` like any other -- about 0.0036 degrees of angular
  error at worst, measured over 100k directions. Keeping the exact normal would
  need the shader to compare the three packed words and branch, which is the
  branch the no-flag design exists to avoid. What it costs on
  `test_adaptive_sampling_convergence`, whose scene is spheres, quads and flat
  triangles: the mean linear radiance moves from 0.7577084 to 0.7577159 at 200
  spp and from 0.7580927 to 0.7580732 at 2000, a relative 1e-5 -- two orders of
  magnitude below what that test can resolve.

- **Generated normals are a guess, and it is on by default.** A mesh with no
  `vn` records gets normals generated at `DEFAULT_CREASE_ANGLE_DEGREES` (40),
  because most such meshes want them and the alternative was every caller
  remembering to ask. The threshold is what keeps that honest, and it is
  measured rather than chosen: `box.obj`'s corners read 48.2 to 70.5 degrees
  against it and stay sharp, `sphere.obj`'s read 13.5 to 23.8 and smooth. The
  gap between those two is wide, but a mesh that lands inside it will be
  guessed at wrongly, and `Obj::with_flat_shading` is the way out.

  The angle is measured corner-normal against face-normal rather than between
  two faces across an edge, which is what the one-pass, no-adjacency generator
  can see. For a two-face fold that is half the angle a modelling tool would
  report; where several faces disagree it is more, so a corner creases sooner
  than the pairwise reading suggests. That is the conservative direction.

  Generation is serial, being a scatter into shared vertex slots, while the
  triangle construction around it is a `par_extend`. It has not mattered --
  every large mesh to hand ships `vn` and skips it entirely -- but a
  multi-million-face model without normals would pay for it at load.

- **Per-vertex normals assume a rigid or uniformly scaled transform.** Normals
  go through `Transformer::transform` with translation skipped and are then
  re-normalised, which is only the correct rule when the transform has no
  shear and no non-uniform scale. Every `Transformer` in the crate is rigid or
  uniform, and `Transformer` maps a `Vec3` to a `Vec3` with no way to express
  the matrix an inverse transpose would be taken from, so nothing reachable
  through the public API can violate it. Recorded because the rule is wrong in
  general and would have to change alongside any `Transformer` that could
  shear.

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
- **Welford under-reads the low-discrepancy sampler, and two things believe
  it.** `M2` estimates the *marginal* per-sample variance, and both consumers
  divide it by `n` to get the variance of the mean. That division assumes the
  samples are independent. Owen-scrambled Sobol samples are negatively
  correlated by construction -- that is the entire point of them -- so the true
  variance of the mean sits below `var/n` while the estimator still reports
  `var/n`. Every pixel therefore looks less converged than it is, so adaptive
  sampling retires each one later, and the denoiser is handed an overstated
  variance and filters a little harder than the residue warrants -- which is why
  `test_denoise_improves_low_sample_image`'s ratio loosened by more than the raw
  RMSE improved.

  Smaller than it sounds on the bench, and worth recording as such rather than
  as the regression it was expected to be: the sampler costs
  `adaptive_sampling/adaptive` +2.3% (1.855 s -> 1.898 s) against
  `adaptive_sampling/forced_off`'s +2.0%, so only 0.3 percentage points of that
  is the estimator being misled. Adaptive sampling still wins on that scene,
  just by 0.42% instead of 0.66% -- it was never winning much there, which is
  also why the effect has so little room to show. Expect a scene with large
  genuinely converged regions to pay more.

  Both consumers are accepted as they are: the sampler's win is far larger than
  what this costs back. The honest fix -- splitting the batch into half-streams
  and estimating the variance of the mean directly -- touches Welford, adaptive
  sampling and the denoiser's variance input at once, and is a change of its
  own.
- **Dielectrics block NEE shadow rays.** Light through glass is found only by
  BSDF-sampled paths, at full MIS weight. Unbiased, but caustics stay noisy —
  the standard trade-off of naive NEE.
- **`Blend` is never `is_light()`.** A blend containing a `DiffuseLight` is not in
  the lights array, so it gets no NEE and is found by BSDF paths with weight 1.
  Consistent and unbiased, because the MIS PDF covers exactly the same set.
- **Rare MIS edge case.** A blend-emitter hit with a real light collinear behind
  it can be slightly under-weighted, since the PDF sum counts lights at any
  distance along the ray. Pre-existing structure, vanishingly rare -- and the
  direct consequence of summing over every light rather than over the one the
  ray actually reaches, so it goes away on its own if the closest-emitter PDF
  lands.
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
- **Metal is single-scatter GGX, so it loses energy with roughness.** A
  microfacet that reflects light onto another microfacet instead of out of the
  surface is light the model drops, and the deficit grows with roughness. It is
  measured rather than guessed:
  `test_ggx_metal_is_energy_conserving_in_a_furnace` puts a white metal in a
  uniform unit environment, where an energy-conserving BRDF must read exactly
  1, and reads 1.0000 / 0.9942 / 0.8976 / 0.6318 / 0.3503 at fuzz 0 / 0.25 /
  0.5 / 0.75 / 1. A CPU quadrature of the BRDF's definition agrees with every
  one of those to 0.0015, so what the test pins is the model rather than this
  implementation of it. The same test is what a compensation term would have to
  move.

- **`Metal`'s parameters changed meaning with GGX.** `fuzz` was a sphere radius
  around the mirror direction, calibrated against nothing; it is now a
  perceptual roughness with `alpha = fuzz * fuzz`, the squared-roughness
  convention every other renderer's slider means. The same number reads
  noticeably sharper than it used to, and a scene sitting at `fuzz ~ 0.3`
  changes appearance. `alpha = fuzz` would have been closer to the old spread;
  the square won because the old parameter matched no other renderer and this
  one matches all of them. `albedo` likewise became f0, the reflectance at
  normal incidence, rather than a flat multiplier -- so every metal now has a
  Fresnel rim going white at a grazing angle.

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

---

## Deliberately declined

Recorded so they aren't reconsidered without new information.

- **A second step-1 à-trous pass.** The obvious lever once the cascade's outer
  iterations were found to be idle, and unnecessary once they were not. Every
  pass attenuates white noise, including the step-16 one -- taps 16 apart are
  still independent -- and in the all-weights-one limit the five-iteration
  cascade is a width-32 four-fold box, worth 65x on white noise. The schedule was
  never the problem; the variance feeding the tolerance was. A second step-1 pass
  would also cost a dispatch and muddy the documented "each doubling the tap
  spacing" contract.

- **Single-light fast path for `light_pdf_value`.** `sample_light` already
  intersects the chosen light, so its PDF could be computed without the extra
  traversal when `light_count == 1`. Worth ~2%, but it duplicates the PDF formula
  in two places where drift would silently bias the estimator. Not worth it.

  Superseded rather than reversed: the closest-emitter PDF formulation removes
  the loop entirely and leaves exactly one copy of the formula, so the ~2% comes
  for free without the duplication this objection was about. The objection was
  right; it was an objection to the duplication, not to the saving.
- **Wavefront path tracing.** The original reason recorded here -- "the benefit
  is smaller on a small-wavefront iGPU" -- was simply wrong about the hardware:
  the development machine is a 40-CU RDNA1 dGPU. The conclusion survives anyway,
  for a better reason.

  Path state that would have to cross global memory between stages is origin 12
  + direction 12 + throughput 12 + sampler 8 + flags 4 + `prev_bsdf_pdf` 4 +
  `path_length` 4 + the two accumulators 24, about **84 bytes**. At 800x600 each
  stage reads and writes it: 2 x 480000 x 80 B = 76.8 MB per stage per bounce.
  At roughly 4 bounces and 2 stages per bounce that is **~614 MB per sample**,
  which at the RX 5700 XT's 448 GB/s is **1.37 ms per sample** -- against a
  measured total cost of **2.82 ms per sample**. A wavefront rewrite adds
  traffic equal to half the entire current per-sample cost before it saves
  anything.

  For it to break even the megakernel would have to be losing more than half its
  time to register pressure, and sharing the two traversal stacks is a two-line
  change that tests the same hypothesis. Wave32 also halves what divergence costs
  relative to wave64, and `create_test_scene` is overwhelmingly Lambertian, so
  the material-coherence argument is weaker here than on the hardware it is
  usually made about. And the real divergence in this shader is lanes taking
  wildly different numbers of node tests, which wavefront does not fix -- only
  ray reordering does.

  Re-open condition: the ISA dump showing the tracer below ~6 waves/SIMD *after*
  the stack sharing and the pipeline specialisation land, on a scene with
  genuinely mixed materials.

- **Material sorting within a workgroup using subgroup operations.**
  `workgroup_size(8, 8)` is 64 threads, which is **two wave32s**. There is
  nothing to sort: cross-wave reordering needs LDS and a barrier, and with two
  waves the best possible outcome is moving a handful of lanes. The thing people
  usually want from subgroups here -- skipping a branch no lane needs -- the
  hardware already does with `s_cbranch_execz`.
- **Global early-exit for adaptive sampling.** Per-pixel adaptive sampling
  (`ray_trace.wgsl`'s `compute`) skips converged pixels inside each dispatch,
  but the CPU render loop still runs until `samples_per_pixel` batches are
  issued, even once every pixel has converged. Stopping the whole render early
  would need a new atomic-reduction pattern (a global "pixels still active"
  count) this crate doesn't have anywhere else, for benefit that vanishes on
  any scene with a persistently noisy region (a small light, a caustic). Not
  worth it until a scene shows up where it would matter.
- **A *relative* firefly clamp in the renderer**, to replace the absolute
  `CLAMPING_THRESHOLD = 10.0`. It would have to be relative to the running mean,
  which depends on how many samples happened to land before the current batch --
  and `samples_per_batch` is re-tuned from measured dispatch time, so the image
  would depend on GPU timing. That breaks the property
  `test_denoise_improves_low_sample_image` rests on, that a noisy and a denoised
  render trace bit-identical sample streams. It would also charge the bias to
  every pixel permanently, and most to exactly the pixels whose true radiance is
  driven by rare bright events. The denoiser has strictly better information --
  the neighbourhood, the noise-free guide, the exact Welford variance -- runs
  once after the fact, and a mistake there costs one frame rather than being
  baked into the accumulator.

- **Measured and not done: a compressed-luminance edge stop.** `w_colour`
  compares raw linear radiance while the image is shown through ACES and gamma
  2.0, so noise inside a legitimately bright region survives the filter almost
  untouched -- its linear differences dwarf `lum_tolerance` even though they are
  invisible on screen. Replacing `luminance()` in the weight with `log(1 + L)`
  and carrying the variance through by the delta method,
  `Var(log(1+L)) ~= Var(L)/(1+L)^2`, is the textbook fix, in the asymmetric
  variant with `f'` at the tap rather than the centre -- the centre-evaluated
  form is backwards on both halves of the firefly case, refusing to average the
  firefly down *and* letting it leak into its neighbours.

  Implemented and measured at strength 1 rather than left as a suggestion, and
  the answer was no. To first order it is *identically* the current filter: the
  delta method scales numerator and denominator by the same derivative and it
  cancels, so all it changes is the tail. Against the same build it took
  fireflies from 129 to 84 at 1 spp and 77 to 63 at 2, moved denoised grain by
  under 1%, and cost 4% of the specular scene's RMSE against a converged
  reference and 7% of the diffuse scene's -- a weight that is not symmetric in
  (centre, tap) does not conserve energy locally, and the RMSE is seeing that
  bias. The firefly budget was at 129 against a gate of 946, so it was buying
  margin that was already there with accuracy that was not. Revisit only if a
  scene turns up where fireflies survive at a sample count anyone renders at.

- **Screen-space blue-noise error distribution (Heitz-Belcour).** The obvious
  companion to the low-discrepancy sampler, and it should wait. Three reasons,
  in order of weight. It does not reduce per-pixel variance at all; it
  reorganises where the error sits spatially. This repo's metrics would punish
  it: `grain` and `displayed_grain` are 3x3 high-passes and blue noise by
  definition moves error *into* the high frequencies, so a genuine improvement
  would read as a 20-50% regression against the live gates in
  `test_denoise_grain_does_not_grow_with_samples` and
  `test_denoised_grain_keeps_falling_with_samples`. And it needs precomputed
  scrambling and ranking tiles, which `Cargo.toml`'s `exclude = ["resources/*"]`
  would keep from reaching crate users. Revisit alongside a gate on the RMSE of
  the *denoised* image, which is the metric that would see the win -- at which
  point the cheap version is a per-pixel blue-noise offset into the Owen
  scramble seed, about five lines.

---

## Measurement records

Numbers that were expensive to obtain and that future work will be judged
against. Kept because re-deriving them costs hours and because several of them
contradict the metric a newcomer would reach for first.

### Already settled, and not worth redoing

The 24-item performance sweep (BVH2 node layout, binned SAH build, hot/cold
primitive split, deferred shading, 2-D tiled dispatch, sample batching, Russian
roulette, closed-form sampling), next-event estimation with MIS, per-pixel
adaptive sampling, and narrowing the firefly clamp to indirect light only. Also
the half-pixel pixel-to-frame mapping, whose `/ (width - 1)` divisor is now
`/ width` -- the golden images turned out to be insensitive to it (they downscale
to 100x50 first), and both G-buffer tests now assert the centre ray
analytically, to 0.001 rather than 0.05.

### The denoiser

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
pixels more than 20 of 255 brighter than every neighbour, at 500x500. Note these
numbers predate the fade and `sigma_colour` changes recorded below and have not
been re-measured at 500x500 since; `test_denoise_removes_fireflies_at_every_sample_count`
is the current reading, and it is better at every point:

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

Also done: the denoised image now gets smoother as the sample count rises, which
it did not. Displayed grain on a Cornell box, in code values of the final image:

    spp             2       8      32     128
    raw        15.366   9.428   5.645   3.121
    before      3.009   3.566   3.512   2.632
    after       2.066   1.739   1.426   1.060

Sixty-four times the samples used to buy 13%, and 2 spp came out smoother than
8. Two independent causes, both now fixed.

The fade in `denoise_resolve.wgsl` blended on the pixel's relative standard
error over a `full_strength_error` of 0.4. That is linear in sigma while sigma
falls as `1/sqrt(n)`, so the residual is `sigma * (1 - sigma / (0.4 L))` -- a
downward parabola peaking at `sigma = 0.2 L`, and any two noise levels symmetric
about that peak leave *identical* grain. A wall at 10 and at 100 spp is almost
exactly that pair. And 0.4 relative linear error is not a small quantity: through
ACES and gamma 2.0 it is 13 to 30 code values depending on brightness, where the
eye picks grain out of a flat wall at about one, so the filter only ever
committed fully to noise nobody could miss. It now fades on how far the residue
would move the pixel *on screen*, as a secant through the same transform the
readback applies -- which needs no derivative of the tone curve, stays honest
where sigma is the size of L, and gives a blown highlight a blend of zero for
free. Because the ramp is linear, `FULL_STRENGTH_GRAIN / 4` bounds what it can
leave anywhere at any sample count: half a code value at the default.

`strength` reaches that threshold, which it did not reach before -- the override
was never passed from `denoise.rs`, so the knob moved `sigma_colour` and nothing
else. That is also why `sigma_colour` is now `2 * sqrt(strength)`: linear scaling
put strength 10 at fourteen standard deviations of the inter-pixel difference,
harmless only because the fade then discarded four fifths of the result.

The second cause was the variance the à-trous cascade tracks. `sum(w^2 * var)` is
the variance of a weighted mean of *independent* taps, true on the first
iteration and false after it, because each tap is a pixel that already averaged a
neighbourhood overlapping its neighbours'. The composite kernel after N
iterations is the four-fold self-convolution of a box of width `2^N`, whose
`sum k^2` falls as `0.4886/M` rather than as `sum h^2` to the N, so the tracked
variance was 96x too low by the fifth iteration and the tolerance ten times too
tight -- the four widest passes were very nearly the identity, which is exactly
the shape of "clears blotches, leaves grain". With the per-iteration factors in,
the tolerance halves from the second iteration onward, which is Dammertz's
`sigma/2^i` schedule arrived at rather than assumed. The factors are faded in on
how much of the kernel survived its edge stops: applying them whole cost the
specular scene 18% of its RMSE, concentrated on the mirror and the caustic.

`test_denoised_grain_keeps_falling_with_samples` is the gate,
`denoise_display_sweep` the diagnostic the threshold was chosen on, and
`displayed_grain` the metric -- a *median* absolute departure from the 3x3 mean,
through the display transform, because object silhouettes put a floor under any
RMS high-pass and that floor is precisely what hid the defect. The same scene at
4000 spp measures 0.628 code values of that floor, which is how much headroom is
left.

`test_denoise_is_near_identity_at_high_samples` was restated rather than retuned;
see the note under *Known limitations* above.

### The sampler

Done: every budgeted draw comes from an Owen-scrambled Sobol sequence
(`ray_trace.wgsl`, `sampler_2d`), hash-based and table-free after Burley 2020, so
there are no direction-number tables and `sobol_0` is `reverseBits`. It is behind
the `low_discrepancy` pipeline override, which is what `sampler_convergence_sweep`
flips to produce the table below.

Linear RMSE against a 4000 spp reference, adaptive sampling off in every arm,
white noise → Owen-scrambled Sobol. The `trimmed` column drops the worst 0.1% of
pixels, and is there because the plain figure on the Cornell box is four fifths
fireflies:

| scene | spp | rmse | trimmed |
|---|---|---|---|
| test scene | 8 | −25% | −24% |
| test scene | 64 | −39% | −39% |
| test scene | 256 | −39% | −40% |
| specular | 8 | −28% | −23% |
| specular | 256 | −46% | −40% |
| Cornell | 8 | −60% | −16% |
| Cornell | 256 | −75% | −42% |

Cost: `render` (800x600, 64 spp) 191.97 ms -> 199.24 ms, **+4.2%**. That figure
is entirely down to `sobol_1` not being a loop. The Antonov-Saleev recurrence
over the set bits of a *scrambled* index runs ~16 iterations every draw and put
the same benchmark at **+19%**; the five-layer closed form it was replaced with
(Pascal mod 2 plus Lucas, see the function) brought it back inside budget. The
fallback that was budgeted for and turned out not to be needed was an
`LD_PAIR_LIMIT` -- Sobol for the first few pairs, hashing for the deeper
bounces, which are the ones low-discrepancy sampling helps least.

A 39% RMSE reduction is 2.7x fewer samples for the same error. None of this beats
O(N^-0.5) asymptotically and it is not meant to — padded Sobol reverts to that
rate once discontinuities dominate, which for Cornell's shadow boundaries is
early.

**The expensive finding: scrambling the outputs without shuffling the index does
nothing, and it fails silently.** Owen-scrambling each pair's two dimensions with
its own seed is the obvious reading of "padding", and it is not enough. Work out
the top bit of a scrambled dimension and it comes to the *low bit of the sample
index* XOR a per-dimension constant: every dimension of every pair crosses into
its other half on the same sample, however independent the seeds are. The pads
walk in lockstep, so the joint distribution across pairs never equidistributes
and the estimator stops converging. Measured, linear RMSE on the test scene
flattened at 0.17 from 64 spp upward instead of halving per 4x samples, while at
8 spp it still looked like a 17% win — the regime a quick check would have looked
at. The fix is one line, `nested_uniform_scramble` applied to the sample index
before the sequence is generated, and it is safe for the same reason the output
scramble is: a nested permutation maps the prefix `0..2^m` onto a 2^m-*aligned
contiguous block* of the sequence, and every aligned block of a (0,2)-sequence is
a (0,m,2)-net exactly as a prefix is. No sample count has to be known up front,
which is the requirement adaptive sampling imposes.

`renderer/sampler_test.rs` is what would have caught it in milliseconds, and now
does: it mirrors the three shader functions on the CPU and checks the scramble is
a bijection, that it is nested, that the first 2^k points form a (0,k,2)-net for
k = 1..10 before and after scrambling, and that two pads agree on a top bit about
half the time rather than always.

**Why Owen and not stratification.** Plain stratified or jittered sampling cannot
be used here at all. Adaptive sampling retires each pixel at a different,
unpredictable `n`, and stratification needs N up front: a jittered point is
uniform only *within* its stratum, so evaluating a partial set of strata is
biased, not merely noisier. Owen scrambling makes every individual point
marginally uniform, so a truncated prefix stays unbiased. That is not a nicety
here, it is the thing that makes low-discrepancy sampling compatible with this
renderer.

**Seeds.** `RenderConfig::seed` exists because every reference render in the
sweeps used to share its exact sample-stream prefix with the render measured
against it, making the RMSE correlated and biased low. With white noise that was
about 0.2% at 8 spp against 4000 and ignorable. With this sampler it is
structural — an 8-sample render is literally a sub-net of the 4000-sample
reference — and it flattered every arm by roughly 10% before the references were
moved to `seed: 1`.

### GPU timing and occupancy

Done: `util/gpu_timing.rs` brackets every compute pass in a pair of timestamp
queries, behind `SOLSTRALE_GPU_TIMING` and behind a feature check, so the
default path allocates nothing and encodes nothing. Three ways to read it: a
per-dispatch line on stderr, the `gpu_pass_timings` diagnostic for a
whole-render table, and `./shader-stats.sh` for what the compiler made of the
shader. Deliberately not on `RenderProgress` -- it is public, the field would be
an `Option` forever, and filling it obliges a map and a poll every batch for a
number no library consumer asked for.

**Read a sub-millisecond pass twice.** The long passes are steady to under a
percent across runs, but the first run after a build reported the denoise chain
at 6.68 ms and every later one at 2.09 ms, and a bloom arm did the same thing
once. Everything above 1 ms was unaffected in both cases. All the figures below
are the median of three runs; a single reading of a 0.06 ms pass is not a
measurement.

**Wall clock was not lying.** The first thing the instrument was pointed at is
the assumption behind `DispatchCost`: that the wall clock measured around
submit-plus-poll carries latency that is not ours. On an unshared queue it
carries very little. Over a 64 spp render at 800x600, GPU busy time is 97-98% of
the measured dispatch in steady state, falling to 88-94% only on the first
dispatch of a run and on the one that carries the post-processing chain. So
there is nothing here for dispatch pipelining to recover, and the two-term fit
is not papering over a mismeasurement. On a queue shared with a vsync-throttled
presenter that gap is the whole reason the model exists, and this says nothing
about that case.

**Where a render's time goes**, 800x600, from `gpu_pass_timings`:

| chain | trace | chain | trace share |
|---|---|---|---|
| none, 64 spp | 180.47 ms | -- | 100% |
| none, 16 spp | 48.2 ms | -- | 100% |
| bloom 0.1 | 48.2 ms | 1.55 ms | 96.9% |
| bloom 0.002 | 48.2 ms | 0.29 ms | 99.4% |
| saturation | 48.3 ms | 0.06 ms | 99.9% |
| denoise, 5 iterations | 48.2 ms | 2.09 ms | 95.8% |

The headline render is 5.87 ns per sample path, against 5.88 ns predicted by
dividing the two criterion points apart -- so the arithmetic behind the backlog
was right, and it is the roofline read off it that needed checking rather than
the number. The whole post-processing backlog is competing for the last 0-4%;
the tracer is the only thing on this list worth optimising. Within bloom, the
161-tap separable blur is 1.38 ms of the 1.55 and the three-tap case is 0.13 ms,
so the blur is linear in the kernel as expected and bloom's fixed cost is
0.16 ms. Within the denoiser, prepare is 0.11 ms, the variance prefilter
0.39 ms, the five à-trous iterations 1.46 ms and the resolve 0.13 ms.

**The post chain costs three to fifteen times its compute.** Against
`post/none` at 53.31 ms, criterion puts `post/saturation` at 54.16 ms -- 0.86 ms
for 0.06 ms of shader. The three configurations differ in exactly three things,
and two of them are not compute passes: allocating the second full-size working
buffer, and the accumulator-to-post-buffer copy the render loop does *every*
batch rather than only on the one that runs the chain. That copy is deliberate
(`RenderProgress` publishes the post buffer, so a caller watching an unfinished
render has to find the accumulated image in it), but it is now a measured cost
rather than an assumed-free one, and it is larger than every post-processing
shader in the crate put together.

**Two fixed costs, measured directly instead of by subtraction.**
`render/renderer_setup` -- scene flatten, buffer upload, shader compile -- is
5.36 ms, where fitting `render` and `denoise/none` apart had put it at 7.98 ms:
the two-point fit was half as much again as the truth, which is the reason the
arm exists. `readback/800x600` is 2.02 ms and `readback/4K` is 26.59 ms
(4.6 GiB/s), confirming the 25 ms that `buffer_to_image`'s own doc comment
claims for the parallel in-place read and had no bench behind.

**The tracer is occupancy-bound, and the traversal stack is why.** From
`./shader-stats.sh` on a Radeon RX 5700 XT (RADV, Mesa 26.2.2): 128 VGPRs, 108
SGPRs, 5 spilled SGPRs, no spilled VGPRs, **scratch size 0**, 8 waves per SIMD
against the wave32 cap of 20. The 32-entry traversal stack is held entirely in
registers -- there is no `scratch_` instruction anywhere in the 34 KB of ISA --
and accounts for 44 of those 128: shrinking it to 8 entries takes the shader to
84 VGPRs and 12 waves, recovering four of the twelve missing ones. The table is
in the comment on `MAX_TRAVERSAL_DEPTH` in `ray_trace.wgsl`, next to the
constant it is about.

Two consequences. Sharing the stacks is already done and spent (`ray_trace.wgsl`
declares one stack, used by `world_hit` and `occluded` in turn), so the
remaining lever is a *shorter* stack: short-stack or stackless traversal, which
trades re-traversal work for occupancy. And moving the stack to LDS or scratch
buys nothing that is not already true -- it is not in memory now.
