//! Utility to convert a [`RgbImage`] height map to a normal map
use image::RgbImage;

const STRENGTH: f32 = 6.0;

struct AdjacentPixels {
    nw: f32,
    n: f32,
    ne: f32,
    w: f32,
    e: f32,
    sw: f32,
    s: f32,
    se: f32,
}

impl AdjacentPixels {
    /// edge pixels are duplicated when necessary
    #[allow(clippy::many_single_char_names, clippy::absurd_extreme_comparisons)]
    fn new(x: u32, y: u32, img: &RgbImage) -> Self {
        let n = if y <= 0 { 0 } else { y - 1 };
        let s = if y >= (img.height() - 1) {
            img.height() - 1
        } else {
            y + 1
        };
        let w = if x <= 0 { 0 } else { x - 1 };
        let e = if x >= (img.width() - 1) {
            img.width() - 1
        } else {
            x + 1
        };

        AdjacentPixels {
            nw: fetch_pixel(n, w, img),
            n: fetch_pixel(n, x, img),
            ne: fetch_pixel(n, e, img),
            w: fetch_pixel(y, w, img),

            e: fetch_pixel(y, e, img),
            sw: fetch_pixel(s, w, img),
            s: fetch_pixel(s, x, img),
            se: fetch_pixel(s, e, img),
        }
    }

    /// Calculates the normals along the x-axis. Usually used for the red
    /// channel after normalization.
    fn x_normals(&self) -> f32 {
        -(self.se - self.sw + 2.0 * (self.e - self.w) + self.ne - self.nw)
    }

    /// Calculates the normals along the y-axis. Usually used for the green
    /// channel after normalization.
    fn y_normals(&self) -> f32 {
        -(self.nw - self.sw + 2.0 * (self.n - self.s) + self.ne - self.se)
    }
}

/// Fetches the pixel at (x,y) and returns its value as an f32 scaled to between
/// 0.0 and 1.0. Coordinate parameters are reversed from usual to better match
///   compass directions.
fn fetch_pixel(y: u32, x: u32, img: &RgbImage) -> f32 {
    (img.get_pixel(x, y)[0] as f32) / 255.0
}

/// Creates the normal mapping from the given image
pub fn to_normal_map(img: RgbImage) -> RgbImage {
    let mut normal_map = RgbImage::new(img.width(), img.height());

    for (x, y, p) in normal_map.enumerate_pixels_mut() {
        let mut new_p = [0.0, 0.0, 0.0];
        let s = AdjacentPixels::new(x, y, &img);

        new_p[0] = s.x_normals();
        new_p[1] = s.y_normals();
        new_p[2] = 1.0 / STRENGTH;

        let new_p = scale_normalized_to_0_to_1(&normalize(new_p));

        p[0] = (new_p[0] * 255.0) as u8;
        p[1] = (new_p[1] * 255.0) as u8;
        p[2] = (new_p[2] * 255.0) as u8;
    }
    normal_map
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let v_mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / v_mag, v[1] / v_mag, v[2] / v_mag]
}

fn scale_normalized_to_0_to_1(v: &[f32; 3]) -> [f32; 3] {
    [v[0] * 0.5 + 0.5, v[1] * 0.5 + 0.5, v[2] * 0.5 + 0.5]
}
