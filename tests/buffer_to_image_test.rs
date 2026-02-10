use image::RgbImage;
use solstrale::util::wgpu_util;

#[test]
fn test_buffer_to_image_compiles() {
    // This test is just to ensure the function exists and compiles.
    // The actual functionality requires a wgpu device which is hard to mock here without a full setup.
    // We will rely on integration tests for full verification.
    
    // We can't easily create a dummy wgpu::Buffer without a device. 
    // So we will just checking signature for now by attempting to assign it to a function pointer of expected type
    let _func: fn(&wgpu::Device, &wgpu::Buffer, u32, u32) -> RgbImage = wgpu_util::buffer_to_image;
}
