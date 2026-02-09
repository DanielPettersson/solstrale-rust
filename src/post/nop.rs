use crate::post::PostProcessor;
use std::error::Error;

#[derive(Clone, Default)]
/// A post-processor that does nothing
pub struct NopPostProcessor {
    width: u32,
    height: u32,
}

impl NopPostProcessor {
    /// Create a new nop post-processor
    pub fn new() -> Self {
        NopPostProcessor::default()
    }
}

impl PostProcessor for NopPostProcessor {
    fn initialize(&mut self, _device: &wgpu::Device, _queue: &wgpu::Queue, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    fn post_process(
        &self,
        _encoder: &mut wgpu::CommandEncoder,
        _output_buffer: &wgpu::Buffer,
        _num_samples: u32,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}
