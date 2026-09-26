//! The single-scatter directional albedo of the rough dielectric, tabulated.
//!
//! What the microfacet dielectric drops is the light one microfacet sends onto
//! another: the model reflects or refracts once and stops, so a sample that
//! leaves below the horizon, and the whole of the Smith shadowing term, are
//! energy the surface was given and never returns. Measured in a furnace, a
//! white 1.5 ball kept 37% of it at roughness 1.
//!
//! `E` is how much a surface *does* return -- the integral of the single-
//! scatter BSDF against the cosine over the whole sphere, both lobes -- so
//! dividing `f` by it puts the rest back. That is Turquin 2019's compensation,
//! the same one the conductor uses, minus the `f0` weighting: light bouncing
//! twice between microfacets is tinted twice on a metal, and a dielectric
//! interface tints nothing.
//!
//! **A table rather than a fit, unlike the conductor's.** `E` also depends on
//! the relative index, and effectively discontinuously: below the critical
//! angle a microfacet transmits, above it the surface is a mirror, and the
//! critical angle *is* the index. Building the table for the index the material
//! has removes that from the problem, leaving a surface in two variables smooth
//! enough for 16x16 bilinear to carry to 0.005, against the 0.014 to 0.033 the
//! conductor's fit gives.
//!
//! The cost of that choice is a storage buffer and a per-material offset, and
//! one table build per distinct index in the scene -- `E` does not depend on
//! the material's own roughness, because roughness is one of the axes, so a
//! scene of five glass spheres at 1.5 builds one table.

use std::f64::consts::TAU;

use crate::geo::vec3::Vec3;

/// Samples along the `cos_o` axis, uniform in `cos_o` over `[0, 1]`.
pub const ENERGY_COS_STEPS: usize = 16;
/// Samples along the roughness axis, uniform in `sqrt(alpha)` over `[0, 1]`.
///
/// In `sqrt(alpha)` for the reason the conductor's fit is: in `alpha` nearly
/// all of the variation is crowded into the first quarter of the range.
pub const ENERGY_ALPHA_STEPS: usize = 16;
/// One side's grid, entering or leaving.
pub const ENERGY_SIDE_STRIDE: usize = ENERGY_COS_STEPS * ENERGY_ALPHA_STEPS;
/// Both sides, which is what one material's offset points at.
pub const ENERGY_TABLE_LEN: usize = 2 * ENERGY_SIDE_STRIDE;

/// Quadrature resolution per table entry, as an NxN stratified grid over the
/// visible-normal sampler's own unit square.
///
/// 32 is where it stops paying: against a 400k-sample Monte Carlo reference the
/// residual is 0.0007, an order below the bilinear interpolation error the
/// table already carries, while 16 leaves 0.006. Deterministic on purpose --
/// a table that varied between runs would put noise into every image built on
/// it.
const QUADRATURE_STEPS: usize = 32;

/// Both halves of one material's table: entering first, then leaving.
///
/// `index_of_refraction` is the material's own, so the two halves are built at
/// `1 / n` and `n` -- the same `Bsdf::ior` the shader resolves per hit.
pub fn build_table(index_of_refraction: f64) -> Vec<f32> {
    let n = if index_of_refraction.abs() < 1e-4 {
        1.
    } else {
        index_of_refraction
    };
    [1. / n, n]
        .iter()
        .flat_map(|&eta_it| {
            (0..ENERGY_ALPHA_STEPS).flat_map(move |ai| {
                let alpha = (ai as f64 / (ENERGY_ALPHA_STEPS - 1) as f64).powi(2);
                (0..ENERGY_COS_STEPS).map(move |ci| {
                    // The grazing end is a limit rather than a value: at
                    // cos_o = 0 the visible-normal distribution degenerates.
                    let cos_o = (ci as f64 / (ENERGY_COS_STEPS - 1) as f64).max(1e-3);
                    directional_albedo(cos_o, alpha, eta_it) as f32
                })
            })
        })
        .collect()
}

/// How much of what arrives from `cos_o` leaves again, under the single-scatter
/// model the shader implements.
///
/// Deliberately written as the average of the *sampled weight* rather than as
/// an integral of the BSDF: `bsdf_sample`'s `G2 / G1` already is `f |cos| /
/// pdf`, so averaging it over the visible-normal distribution is the albedo by
/// construction, and it can only disagree with the shader if the shader changes.
/// Every rejection the sampler makes -- a reflection below the horizon, the
/// half vector behind `wo` -- contributes its zero here, which is the point:
/// those rejections *are* the missing energy.
fn directional_albedo(cos_o: f64, alpha: f64, eta_it: f64) -> f64 {
    let sin_o = (1. - cos_o * cos_o).max(0.).sqrt();
    let wo = Vec3::new(sin_o, 0., cos_o);
    let lambda_o = smith_lambda(cos_o, alpha);

    let mut sum = 0.;
    for i in 0..QUADRATURE_STEPS {
        for j in 0..QUADRATURE_STEPS {
            let u = (
                (i as f64 + 0.5) / QUADRATURE_STEPS as f64,
                (j as f64 + 0.5) / QUADRATURE_STEPS as f64,
            );
            let h = sample_ggx_vndf(wo, alpha, u);
            let cos_oh = wo.dot(h);
            if cos_oh <= 0. {
                continue;
            }

            let f = fresnel_dielectric(cos_oh, eta_it);
            let weight =
                |cos_i: f64| (1. + lambda_o) / (1. + lambda_o + smith_lambda(cos_i.abs(), alpha));

            // Reflection, dropped when the visible normal sends it under the
            // surface.
            let wi_r = h * (2. * cos_oh) - wo;
            if wi_r.z > 0. {
                sum += f * weight(wi_r.z);
            }
            // Transmission. Past the critical angle `f` is 1 and this term is
            // already zero, so no separate test is needed.
            if f < 1. {
                let perp = (h * cos_oh - wo) * eta_it;
                let wi_t = perp - h * (1. - perp.dot(perp)).abs().sqrt();
                if wi_t.z < 0. {
                    sum += (1. - f) * weight(wi_t.z);
                }
            }
        }
    }
    sum / (QUADRATURE_STEPS * QUADRATURE_STEPS) as f64
}

/// `sample_ggx_vndf` in `ray_trace.wgsl`, term for term.
fn sample_ggx_vndf(wo: Vec3, alpha: f64, u: (f64, f64)) -> Vec3 {
    let wo_std = Vec3::new(wo.x * alpha, wo.y * alpha, wo.z).unit();
    let phi = TAU * u.0;
    let z = (1. - u.1) * (1. + wo_std.z) - wo_std.z;
    let sin_theta = (1. - z * z).clamp(0., 1.).sqrt();
    let c = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), z);
    let h = c + wo_std;
    Vec3::new(h.x * alpha, h.y * alpha, h.z).unit()
}

/// `smith_lambda` in `ray_trace.wgsl`, term for term.
fn smith_lambda(cos_w: f64, alpha: f64) -> f64 {
    let c2 = cos_w * cos_w;
    let tan2 = (1. - c2).max(0.) / c2.max(1e-8);
    0.5 * ((1. + alpha * alpha * tan2).sqrt() - 1.)
}

/// `fresnel_dielectric` in `ray_trace.wgsl`, term for term.
fn fresnel_dielectric(cos_i: f64, eta_it: f64) -> f64 {
    let c = cos_i.clamp(0., 1.);
    let sin2_t = eta_it * eta_it * (1. - c * c);
    if sin2_t >= 1. {
        return 1.;
    }
    let cos_t = (1. - sin2_t).sqrt();
    let r_parallel = (c - eta_it * cos_t) / (c + eta_it * cos_t);
    let r_perpendicular = (eta_it * c - cos_t) / (eta_it * c + cos_t);
    0.5 * (r_parallel * r_parallel + r_perpendicular * r_perpendicular)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth interface loses nothing, whichever way it is crossed. This is
    /// the anchor the whole table hangs off: if the `alpha = 0` row were not 1,
    /// the compensation would brighten smooth glass, which is exactly what it
    /// must never do.
    #[test]
    fn a_smooth_interface_returns_everything() {
        let table = build_table(1.5);
        for side in 0..2 {
            for ci in 0..ENERGY_COS_STEPS {
                let e = table[side * ENERGY_SIDE_STRIDE + ci];
                assert!(
                    (e - 1.).abs() < 1e-4,
                    "side {side}, cos index {ci}: a smooth interface returned {e}"
                );
            }
        }
    }

    /// An index of 1 is not an interface at all -- no reflection, no bending --
    /// so nothing can be lost to a second microfacet on the way through, at any
    /// roughness.
    ///
    /// It is the one case where the answer is known in closed form rather than
    /// measured, which is why it is worth asserting: the transmitted direction
    /// is exactly `-wo`, so `G2 / G1` is the only factor left, and the identity
    /// says the quadrature and the sampler agree about what that is.
    #[test]
    fn an_index_of_one_loses_a_predictable_amount() {
        let table = build_table(1.);
        // Both halves are built at eta = 1, so they have to come out identical.
        for k in 0..ENERGY_SIDE_STRIDE {
            assert_eq!(table[k], table[ENERGY_SIDE_STRIDE + k]);
        }
        // Monotone in roughness: a wider lobe shadows itself more.
        for ci in 0..ENERGY_COS_STEPS {
            for ai in 1..ENERGY_ALPHA_STEPS {
                let prev = table[(ai - 1) * ENERGY_COS_STEPS + ci];
                let cur = table[ai * ENERGY_COS_STEPS + ci];
                assert!(
                    cur <= prev + 1e-5,
                    "cos index {ci}: E rose from {prev} to {cur} as roughness went up"
                );
            }
        }
    }

    /// Nothing in the table may exceed 1: the compensation is `1 / E`, and an
    /// `E` above 1 would darken a surface the model has already darkened.
    #[test]
    fn nothing_returns_more_than_it_was_given() {
        for ior in [1., 1.33, 1.5, 2.4] {
            for (k, e) in build_table(ior).iter().enumerate() {
                assert!(
                    *e > 0.03 && *e <= 1.0001,
                    "ior {ior}, entry {k}: E is {e}, outside (0.03, 1]"
                );
            }
        }
    }
}
