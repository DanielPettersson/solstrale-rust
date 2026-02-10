#[cfg(test)]
mod tests {
    use image::{ImageBuffer, RgbImage};
    use solstrale::util::texture_processing::{TexturePacker, TextureRect};

    #[test]
    fn test_packer_simple_fit() {
        let textures = vec![(100, 100), (100, 100)];

        let packer = TexturePacker::new(200, 200);
        let layout = packer.pack(&textures).expect("Should fit");

        assert_eq!(layout.placements.len(), 2);
        // Check for no overlap
        let p1 = &layout.placements[0];
        let p2 = &layout.placements[1];

        assert!(!rects_overlap(p1, p2));
        assert!(p1.x + p1.width <= 200);
        assert!(p1.y + p1.height <= 200);
        assert!(p2.x + p2.width <= 200);
        assert!(p2.y + p2.height <= 200);
    }

    #[test]
    fn test_packer_alignment() {
        let textures = vec![(10, 10)]; // 10px width
        let packer = TexturePacker::new(100, 100);
        let layout = packer.pack(&textures).unwrap();

        // 10 aligned to 64 is 64.
        assert_eq!(layout.width, 64);
        assert_eq!(layout.height, 10);
    }

    #[test]
    fn test_packer_wont_fit() {
        let textures = vec![(100, 100), (200, 200)];

        // 200x200 atlas, but we have a 100x100 AND a 200x200.
        // 200x200 alone fills it. 100x100 won't fit.
        let packer = TexturePacker::new(200, 200);
        let result = packer.pack(&textures);

        assert!(result.is_err());
    }

    fn rects_overlap(r1: &TextureRect, r2: &TextureRect) -> bool {
        if r1.x >= r2.x + r2.width || r2.x >= r1.x + r1.width {
            return false;
        }
        if r1.y >= r2.y + r2.height || r2.y >= r1.y + r1.height {
            return false;
        }
        true
    }
}
