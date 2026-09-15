# Outstanding work

Suggestions raised during the performance and integrator work that were **not**
implemented, plus limitations recorded deliberately so they don't get
re-litigated later.

Already done and not repeated here: the 24-item performance sweep (BVH2 node
layout, binned SAH build, hot/cold primitive split, deferred shading, 2-D tiled
dispatch, sample batching, Russian roulette, closed-form sampling), next-event
estimation with MIS, per-pixel adaptive sampling, and narrowing the firefly
clamp to indirect light only.

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

`light_pdf_value` (`ray_trace.wgsl:328`) loops over every light, and
`sample_light` chooses one uniformly. Fine at 1–3 lights; an emissive mesh
imported from an OBJ would make both terrible. Wants an alias table for
power-weighted selection, and a light BVH for the PDF sum.

### Denoiser

OIDN was removed in the `wgpu-render` merge and nothing replaced it. There is no
tone mapping either — just `sqrt` gamma applied on the CPU during readback
(`util/wgpu_util.rs`), which is now the binding constraint on highlights: see
the display-clip note under *Known limitations*.

### Instancing (BLAS/TLAS)

Transforms are baked into vertices at construction, so repeated geometry costs
full duplicate storage. Note that `Bvh::new` now **flattens nested BVHs into one
global tree** — that was the right call absent instancing, but it is the decision
to revisit if instancing is added, since a nested `Bvh` would then carry a
transform and must stay a separate acceleration structure.

---

## Correctness and precision

### Camera ray is not normalised

`ray_trace.wgsl:918` builds the primary ray direction without normalising, so
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

- **Readback clips linear radiance at 1.0.** `buffer_to_image`
  (`util/wgpu_util.rs`) does `sqrt(L).min(0.999)`, so everything above 1.0 is
  white regardless of how much brighter it really is. With the firefly clamp
  now at 10 and indirect-only, this — not the clamp — is what decides what a
  highlight looks like, and it is the reason a tone mapper would be the next
  thing to change the image rather than another clamp tweak.
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
