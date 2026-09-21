//! The one definition of relative luminance, on both sides of the device.

use crate::geo::vec3::Vec3;

/// Rec. 709 relative luminance: the weights for the linear primaries the
/// renderer works in throughout.
///
/// Serves two unrelated purposes, which is why it is one function. Where the
/// number is shown to a viewer -- the saturation pivot, the denoiser's edge
/// stop -- the weights have to be the ones for these primaries. Where it is
/// only a cheap scalar proxy for a colour -- the adaptive sampler's variance,
/// a light's selection probability -- any consistent weighting would do, and
/// consistency is the whole point.
pub(crate) fn luminance(c: Vec3) -> f64 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

/// The shader-side mirror of [`luminance`]. Callers concatenate it ahead of
/// their own source, which is safe because it declares one free function and
/// no bindings or entry point.
pub(crate) const LUMINANCE_WGSL: &str = include_str!("luminance.wgsl");

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                sources(&path, out);
            } else if matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("rs") | Some("wgsl")
            ) {
                out.push(path);
            }
        }
    }

    /// The whole point of this module. The saturation pass carried NTSC weights
    /// for as long as it had a copy of its own, and nothing in the renderer
    /// disagreed loudly enough to notice -- so a second copy anywhere in the
    /// library is the failure, whatever weights it happens to hold.
    ///
    /// Scans the library only. The test metrics in `tests/` weigh luminance
    /// themselves on purpose: a metric that imported the code it measures would
    /// move with it.
    #[test]
    fn nothing_else_in_the_library_weighs_luminance() {
        let mut paths = Vec::new();
        sources(
            Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src")),
            &mut paths,
        );

        let mut found = Vec::new();
        for path in paths {
            let text = fs::read_to_string(&path).unwrap();
            let name = path.file_name().unwrap().to_str().unwrap().to_owned();
            if name.starts_with("luminance.") {
                assert!(
                    text.contains("0.2126"),
                    "{name} no longer holds the weights"
                );
                continue;
            }
            // Rec. 709's two distinctive weights, and NTSC's red -- enough
            // to name a copy of either triple, and none of them a plausible
            // constant to mean anything else by.
            for weight in ["0.2126", "0.7152", "0.2989"] {
                if text.contains(weight) {
                    found.push(format!("{}: {weight}", path.display()));
                }
            }
        }

        assert!(
            found.is_empty(),
            "luminance weights outside util/luminance: {found:?}"
        );
    }
}
