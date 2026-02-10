//! Post processors for applying effects to the raw rendered image

mod bloom;
mod saturation;

use std::error::Error;

use enum_dispatch::enum_dispatch;

pub use crate::post::bloom::BloomPostProcessor;
pub use crate::post::saturation::SaturationPostProcessor;

/// Responsible for taking the rendered image and transforming it
#[enum_dispatch]
pub trait PostProcessor {
    /// Does post-construct initialization for the post-processor when width and height are known
    fn initialize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32);

    /// Execute final postprocessing of the rendered image
    fn post_process(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_buffer: &wgpu::Buffer,
        num_samples: u32,
    ) -> Result<(), Box<dyn Error>>;
}

#[enum_dispatch(PostProcessor)]
#[derive(Clone)]
/// An enum of available post-processors
pub enum PostProcessors {
    /// [`PostProcessor`] of type [`BloomPostProcessor`]
    BloomPostProcessor,
    /// [`PostProcessor`] of type [`SaturationPostProcessor`]
    SaturationPostProcessor,
}
