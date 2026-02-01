#[cfg(test)]
mod tests {
    use image::{RgbImage, ImageBuffer};
    use solstrale::util::texture_processing::standardize_texture;

    #[test]
    fn test_standardize_texture_resizes_correctly() {
        let img: RgbImage = ImageBuffer::new(100, 200);
        let processed = standardize_texture(&img);
        assert_eq!(processed.width(), 1024);
        assert_eq!(processed.height(), 1024);
    }
}
