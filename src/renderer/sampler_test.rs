//! CPU mirror of the sampler in `ray_trace.wgsl`, and the tests that pin it.
//!
//! The whole "the sampler is subtly wrong" class -- a scramble that is not a
//! bijection, a Sobol recurrence off by a bit, an index shuffle that breaks the
//! stratification it is supposed to preserve -- is invisible in a rendered
//! image until it shows up as an error floor that no sample count clears. It is
//! also checkable exhaustively in milliseconds with no GPU, which is what this
//! module does.
//!
//! The functions below are transcribed from the shader by hand, so they can
//! drift from it. `test_constants_match_the_shader` is what stops that: it
//! reads the WGSL source and checks every constant here still appears there.

const WGSL: &str = include_str!("ray_trace.wgsl");

/// Laine-Karras permutation. Every multiplier is even, which is what makes each
/// `v ^= v * C` triangular with a unit diagonal and therefore a bijection.
const LK_MULTIPLIERS: [u32; 4] = [0x6c50b47c, 0xb82f1e52, 0xc7afe638, 0x8d22f6e6];

fn laine_karras_permutation(x: u32, seed: u32) -> u32 {
    let mut v = x.wrapping_add(seed);
    for m in LK_MULTIPLIERS {
        v ^= v.wrapping_mul(m);
    }
    v
}

fn nested_uniform_scramble(x: u32, seed: u32) -> u32 {
    laine_karras_permutation(x.reverse_bits(), seed).reverse_bits()
}

fn sobol_0(index: u32) -> u32 {
    index.reverse_bits()
}

fn sobol_1(index: u32) -> u32 {
    let mut g = index;
    g ^= (g >> 1) & 0x5555_5555;
    g ^= (g >> 2) & 0x3333_3333;
    g ^= (g >> 4) & 0x0f0f_0f0f;
    g ^= (g >> 8) & 0x00ff_00ff;
    g ^= (g >> 16) & 0x0000_ffff;
    g.reverse_bits()
}

/// The Antonov-Saleev recurrence `sobol_1` replaces, kept so the closed form
/// can be checked against the definition rather than against itself.
fn sobol_1_by_recurrence(index: u32) -> u32 {
    let mut v = 0x8000_0000u32;
    let mut result = 0;
    let mut n = index;
    while n != 0 {
        if n & 1 != 0 {
            result ^= v;
        }
        n >>= 1;
        v ^= v >> 1;
    }
    result
}

/// The shader's `sampler_2d` on its low-discrepancy path, as a point rather
/// than a pair of floats: shuffle the index, generate, scramble each output.
fn sample(index: u32, seed: u32) -> (u32, u32) {
    let i = nested_uniform_scramble(index, seed);
    let seed_x = pcg_hash(seed);
    let seed_y = pcg_hash(seed_x);
    (
        nested_uniform_scramble(sobol_0(i), seed_x),
        nested_uniform_scramble(sobol_1(i), seed_y),
    )
}

fn pcg_hash(input: u32) -> u32 {
    let state = input.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

fn hash_combine(a: u32, b: u32) -> u32 {
    pcg_hash(a ^ pcg_hash(b))
}

/// Seeds used wherever a test needs a few unrelated ones. Arbitrary, but fixed,
/// so a failure is reproducible.
const SEEDS: [u32; 5] = [0, 1, 0x9e37_79b9, 0xdead_beef, 0xffff_ffff];

/// Asserts that `points` is a (0,k,2)-net: every elementary dyadic box of area
/// 2^-k holds exactly one of the 2^k points.
///
/// That is the property the whole design rests on. It is what makes a Sobol
/// prefix better stratified than a jittered set without needing the sample
/// count up front, and it is what has to survive both the scramble and the
/// index shuffle.
fn assert_is_net(points: &[(u32, u32)], k: u32, what: &str) {
    assert_eq!(points.len(), 1 << k, "{}: wrong point count", what);

    // Every split of k bits between the two axes: 2^-k x 1, 2^-(k-1) x 2^-1,
    // ... 1 x 2^-k.
    for a in 0..=k {
        let b = k - a;
        let cell = |v: u32, bits: u32| if bits == 0 { 0 } else { v >> (32 - bits) };
        let mut occupied = vec![false; 1 << k];
        for &(x, y) in points {
            let i = ((cell(x, a) << b) | cell(y, b)) as usize;
            assert!(
                !occupied[i],
                "{}: two points share the 2^-{} x 2^-{} box at k={}",
                what, a, b, k
            );
            occupied[i] = true;
        }
    }
}

#[test]
fn test_scramble_is_a_bijection() {
    // Exhaustive on the top 16 bits, which is as far as a test can go without
    // a 16 GB table. The scramble is nested (see below), so the top 16 bits of
    // the result depend only on the top 16 bits of the input and a permutation
    // there is the honest restriction of the full one.
    for seed in SEEDS {
        let mut seen = vec![false; 1 << 16];
        for i in 0..1u32 << 16 {
            let out = (nested_uniform_scramble(i << 16, seed) >> 16) as usize;
            assert!(!seen[out], "seed {:#x}: {:#x} is hit twice", seed, out);
            seen[out] = true;
        }
    }
}

#[test]
fn test_scramble_is_nested() {
    // Bit k of the result may depend only on bits 31..k of the input. That is
    // what makes this an Owen scramble of the unit interval -- a permutation
    // that refines the dyadic intervals rather than shuffling values around --
    // and it is why the net property survives it, and why the index shuffle in
    // `sample` maps a 2^m prefix onto a 2^m-aligned block.
    for seed in SEEDS {
        for x in [0u32, 1, 0x1234_5678, 0xa5a5_a5a5, u32::MAX] {
            for k in 0..32 {
                let high = u32::MAX.checked_shl(k).unwrap_or(0);
                let flipped = (x & high) | (!x & !high);
                assert_eq!(
                    nested_uniform_scramble(x, seed) & high,
                    nested_uniform_scramble(flipped, seed) & high,
                    "seed {:#x}: flipping the bits below {} moved a bit above it",
                    seed,
                    k
                );
            }
        }
    }
}

#[test]
fn test_sobol_prefix_is_a_net_before_and_after_scrambling() {
    for k in 1..=10 {
        let n = 1u32 << k;

        assert_is_net(
            &(0..n).map(|i| (sobol_0(i), sobol_1(i))).collect::<Vec<_>>(),
            k,
            "unscrambled",
        );

        for seed in SEEDS {
            assert_is_net(
                &(0..n).map(|i| sample(i, seed)).collect::<Vec<_>>(),
                k,
                &format!("scrambled, seed {:#x}", seed),
            );
        }
    }
}

#[test]
fn test_index_shuffle_maps_a_prefix_onto_an_aligned_block() {
    // Why the shuffle is allowed at all. Adaptive sampling retires each pixel
    // at an n nobody knows in advance, so the sampler cannot shuffle within a
    // known N the way a fixed-budget renderer would. It does not have to: a
    // nested permutation maps the prefix 0..2^m onto a *contiguous block* of
    // 2^m indices aligned to 2^m, and every aligned block of a (0,2)-sequence
    // is a (0,m,2)-net exactly as a prefix is. This is the property
    // `test_sobol_prefix_is_a_net_before_and_after_scrambling` rests on.
    for seed in SEEDS {
        for m in 0..=12 {
            let n = 1u32 << m;
            let mut shuffled: Vec<u32> = (0..n).map(|i| nested_uniform_scramble(i, seed)).collect();
            shuffled.sort_unstable();
            let base = shuffled[0];
            assert_eq!(
                base % n,
                0,
                "seed {:#x}, m={}: block is not aligned",
                seed,
                m
            );
            assert!(
                shuffled
                    .iter()
                    .enumerate()
                    .all(|(i, &v)| v == base + i as u32),
                "seed {:#x}, m={}: block is not contiguous",
                seed,
                m
            );
        }
    }
}

#[test]
fn test_the_pads_do_not_walk_in_lockstep() {
    // The failure the index shuffle exists to prevent: without it the top bit of
    // every scrambled dimension is the low bit of the sample index XOR a
    // per-dimension constant, so every pad crosses into its other half on the
    // same sample. The pads are then rigidly correlated however independent
    // their scramble seeds are, and the error stops falling -- RMSE on the test
    // scene flat at 0.17 from 64 spp up rather than halving per 4x.
    //
    // Tested as a correlation rather than an identity, because what is wrong
    // with lockstep is that it is perfect, not that it exists.
    let seeds: Vec<u32> = (0..8).map(|p| hash_combine(0x1234_5678, p)).collect();
    for (a, &sa) in seeds.iter().enumerate() {
        for &sb in &seeds[a + 1..] {
            let agree = (0..1024u32)
                .filter(|&i| (sample(i, sa).0 >> 31) == (sample(i, sb).0 >> 31))
                .count();
            // Two independent bit streams of 1024 samples agree 512 +- 16 times
            // (one standard deviation); the unshuffled sampler agreed 0 or 1024.
            assert!(
                (384..=640).contains(&agree),
                "pads {:#x} and {:#x} agreed on {} of 1024 top bits",
                sa,
                sb,
                agree
            );
        }
    }
}

#[test]
fn test_hash_combine_does_not_collide_where_xor_did() {
    // A bare XOR of the three terms, `index ^ (sample * A) ^ (restart * B)`, is
    // not injective in the triple: at 1920x1080 and 4096 samples it collides on
    // 2185 pairs, and colliding pixels trace identical relative paths. That
    // reads as low-frequency blotching, which `grain` (a 3x3 high-pass) cannot
    // see, so nothing else in the suite would catch it.
    let xor_seed = |index: u32, sample: u32| index ^ sample.wrapping_mul(0x9E37_79B9);
    let (pixel_a, sample_a) = (0u32, 0u32);
    let (pixel_b, sample_b) = (1_201_941u32, 1597u32);
    assert_eq!(
        xor_seed(pixel_a, sample_a),
        xor_seed(pixel_b, sample_b),
        "the collision this test is about no longer exists; re-derive it"
    );

    // What replaced it. Each term is hashed before it is combined, so the pair
    // above separates -- and so does every other pair over a realistic frame.
    let seed = |pixel: u32, sample: u32| hash_combine(hash_combine(pixel, 0), sample);
    assert_ne!(seed(pixel_a, sample_a), seed(pixel_b, sample_b));
}

#[test]
fn test_sobol_1_closed_form_matches_the_recurrence() {
    // The shader computes dimension 1 as five shift-and-XOR layers instead of a
    // loop over the index's set bits, which is a real derivation (Pascal mod 2
    // plus Lucas) and not an obvious rewrite. Checked against the definition
    // over the low 18 bits exhaustively, plus the corners and a deterministic
    // spread of full-width indices, because a scrambled index is full width.
    for i in 0..1u32 << 18 {
        assert_eq!(sobol_1(i), sobol_1_by_recurrence(i), "index {}", i);
    }
    let mut x = 0x1234_5678u32;
    for _ in 0..200_000 {
        x = pcg_hash(x);
        assert_eq!(sobol_1(x), sobol_1_by_recurrence(x), "index {:#x}", x);
    }
    for i in [0, 1, u32::MAX, 1 << 31, (1 << 31) + 1] {
        assert_eq!(sobol_1(i), sobol_1_by_recurrence(i), "index {:#x}", i);
    }
}

#[test]
fn test_constants_match_the_shader() {
    // The one kind of drift the tests above cannot see: this file agreeing with
    // itself while the shader has moved.
    for m in LK_MULTIPLIERS {
        let literal = format!("{:#010x}u", m);
        assert!(
            WGSL.contains(&literal),
            "{} is no longer in ray_trace.wgsl; the mirror in this file has drifted",
            literal
        );
    }
    for needle in [
        "g ^= (g >> 1u) & 0x55555555u;",
        "g ^= (g >> 16u) & 0x0000ffffu;",
        "return reverseBits(laine_karras_permutation(reverseBits(x), seed));",
        "let i = nested_uniform_scramble(s.index, seed);",
    ] {
        assert!(
            WGSL.contains(needle),
            "ray_trace.wgsl no longer contains `{}`; the mirror in this file has drifted",
            needle
        );
    }
}
