//! The renderer takes a [`Scene`] as input, renders it and reports [`RenderProgress`]

use std::error::Error;
use std::ops::Deref;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use image::RgbImage;
use simple_error::SimpleError;

use crate::camera::{Camera, CameraConfig};
use crate::geo::vec3::{Vec3, ZERO_VECTOR};
use crate::geo::{Ray, Uv};
use crate::hittable::{Hittable, Hittables};
use crate::material::AttenuatedColor;
use crate::post::{NopPostProcessor, PostProcessor, PostProcessors};
use crate::random::random_normal_float;
use crate::renderer::shader::{AlbedoShader, NormalShader, PathTracingShader, Shader, Shaders};
use crate::util::interval::RAY_INTERVAL;

pub mod shader;
pub mod gpu_renderer;

///Input to the ray tracer for how the image should be rendered
#[derive(Clone)]
pub struct RenderConfig {
    /// Width in pixels of the rendered image
    pub width: usize,
    /// Height in pixels of the rendered image
    pub height: usize,
    /// Number of times each pixel should be sampled
    pub samples_per_pixel: u32,
    /// Shader to use when rendering the image
    pub shader: Shaders,
    /// Post processor to apply to the rendered image
    pub post_processors: Vec<PostProcessors>,
    /// Describes at which points in time the render progress should contain an image
    pub render_image_strategy: RenderImageStrategy,
}

impl Default for RenderConfig {
    fn default() -> Self {
        RenderConfig {
            width: 300,
            height: 200,
            samples_per_pixel: 50,
            shader: PathTracingShader::new(50).into(),
            post_processors: vec![],
            render_image_strategy: RenderImageStrategy::OnlyFinal,
        }
    }
}

impl RenderConfig {
    fn needs_albedo_and_normal_colors(&self) -> bool {
        self.post_processors
            .iter()
            .any(|p| p.needs_albedo_and_normal_colors())
    }
}

/// Contains all information needed to render an image
pub struct Scene {
    /// World is the hittable objects in the scene
    pub world: Hittables,
    /// A camera for defining the view of the world
    pub camera: CameraConfig,
    /// Background color of the scene
    pub background_color: Vec3,
    /// Render configuration
    pub render_config: RenderConfig,
}

/// Progress reported back to the caller of the raytrace function
pub struct RenderProgress {
    /// progress is reported between 0 -> 1 and represents a percentage of completion
    pub progress: f64,
    /// Current speed of rendering in number of frames per second
    pub fps: Option<f64>,
    /// Estimated time left until rendering is complete
    pub estimated_time_left: Duration,
    /// Output image so far, will be final when progress is 1
    pub render_image: Option<RgbImage>,
}

#[derive(Copy, Clone)]
/// When should [`RenderProgress`] contain an image of the rendering
pub enum RenderImageStrategy {
    /// Every sample should contain an image
    EverySample,
    /// Only include an image if at least "duration" has elapsed since last time
    /// Plus always include final image
    Interval(Duration),
    /// Only include image in last rendered sample
    OnlyFinal,
}

impl RenderImageStrategy {
    /// Is it time to generate a new render image for the output channel?
    pub fn should_generate_image(
        &self,
        sample: u32,
        total_samples: u32,
        now: SystemTime,
        last_image_generated_time: SystemTime,
    ) -> bool {
        match self {
            RenderImageStrategy::EverySample => true,
            RenderImageStrategy::Interval(d) => {
                sample == total_samples
                    || now
                        .duration_since(last_image_generated_time)
                        .unwrap_or(Duration::from_millis(0))
                        > *d
            }
            RenderImageStrategy::OnlyFinal => sample == total_samples,
        }
    }
}

/// Renderer is a central part of the raytracer responsible for controlling the
/// process reporting back progress to the caller
pub struct Renderer {
    scene: Scene,
    /// All the light hittables in the world
    pub lights: Vec<Hittables>,
    albedo_shader: AlbedoShader,
    normal_shader: NormalShader,
}

/// Result of calculating color for a ray
pub(crate) struct RayColorResult {
    pixel_color: AttenuatedColor,
    albedo_color: Vec3,
    normal_color: Vec3,
}

impl Renderer {
    /// Creates a new renderer given a scene and channels for communicating with the caller
    pub fn new(mut scene: Scene) -> Result<Renderer, Box<dyn Error>> {
        let light_list = scene.world.get_lights();

        if light_list.is_empty() {
            return Err(Box::new(SimpleError::new(
                "Scene should have at least one light",
            )));
        }

        if scene.render_config.post_processors.is_empty() {
            scene
                .render_config
                .post_processors
                .push(NopPostProcessor::new().into());
        }
        scene
            .render_config
            .post_processors
            .iter_mut()
            .for_each(|p| {
                p.initialize(
                    scene.render_config.width as u32,
                    scene.render_config.height as u32,
                )
            });

        Ok(Renderer {
            scene,
            lights: light_list,
            albedo_shader: AlbedoShader {},
            normal_shader: NormalShader {},
        })
    }

    fn ray_color(&self, ray: &Ray, depth: u32, accumulated_ray_length: f64) -> RayColorResult {
        match self.scene.world.hit(ray, &RAY_INTERVAL) {
            Some(rec) => {
                let attenuated_color = self.scene.render_config.shader.shade(
                    self,
                    &rec,
                    ray,
                    depth,
                    accumulated_ray_length,
                );

                if depth == 0 && self.scene.render_config.needs_albedo_and_normal_colors() {
                    let albedo_color = self
                        .albedo_shader
                        .shade(self, &rec, ray, depth, accumulated_ray_length)
                        .color;
                    let normal_color = self
                        .normal_shader
                        .shade(self, &rec, ray, depth, accumulated_ray_length)
                        .color;
                    return RayColorResult {
                        pixel_color: attenuated_color,
                        albedo_color,
                        normal_color,
                    };
                }

                RayColorResult {
                    pixel_color: attenuated_color,
                    albedo_color: ZERO_VECTOR,
                    normal_color: ZERO_VECTOR,
                }
            }
            None => RayColorResult {
                pixel_color: AttenuatedColor {
                    color: self.scene.background_color,
                    ..AttenuatedColor::default()
                },
                albedo_color: self.scene.background_color,
                normal_color: ZERO_VECTOR,
            },
        }
    }

    /// Executes the rendering of the image
    pub fn render(
        &self,
        output: &Sender<RenderProgress>,
        abort: &Receiver<bool>,
    ) -> Result<(), Box<dyn Error>> {
        let mut last_image_generated_time = SystemTime::UNIX_EPOCH;
        let render_start_time = SystemTime::now();
        let image_width = self.scene.render_config.width;
        let image_height = self.scene.render_config.height;
        let pixel_count = image_width * image_height;
        let samples_per_pixel = self.scene.render_config.samples_per_pixel;
        let needs_albedo_and_normal_colors =
            !self.scene.render_config.needs_albedo_and_normal_colors();

        let pixel_colors: Arc<Mutex<Vec<Vec3>>> =
            Arc::new(Mutex::new(vec![ZERO_VECTOR; pixel_count]));
        let albedo_colors: Arc<Mutex<Vec<Vec3>>> =
            Arc::new(Mutex::new(vec![ZERO_VECTOR; pixel_count]));
        let normal_colors: Arc<Mutex<Vec<Vec3>>> =
            Arc::new(Mutex::new(vec![ZERO_VECTOR; pixel_count]));

        let camera = Arc::new(Camera::new(image_width, image_height, &self.scene.camera));

        let pool = rayon::ThreadPoolBuilder::new()
            .build()
            .expect("Failed to create thread pool");

        for sample in 1..=samples_per_pixel {
            if abort.try_recv().is_ok() {
                return Ok(());
            }

            pool.scope(|s| {
                for y in 0..image_height {
                    let camera = camera.clone();
                    let pixel_colors = pixel_colors.clone();
                    let albedo_colors = albedo_colors.clone();
                    let normal_colors = normal_colors.clone();

                    s.spawn(move |_| {
                        let mut row_pixel_colors: Vec<Vec3> = vec![ZERO_VECTOR; image_width];
                        let mut row_albedo_colors: Vec<Vec3> = if needs_albedo_and_normal_colors {
                            vec![ZERO_VECTOR; image_width]
                        } else {
                            Vec::new()
                        };
                        let mut row_normal_colors: Vec<Vec3> = if needs_albedo_and_normal_colors {
                            vec![ZERO_VECTOR; image_width]
                        } else {
                            Vec::new()
                        };

                        let yi = ((image_height - 1) - y) * image_width;
                        for x in 0..image_width {
                            let u = (x as f64 + random_normal_float()) / (image_width - 1) as f64;
                            let v = (y as f64 + random_normal_float()) / (image_height - 1) as f64;
                            let ray = camera.get_ray(Uv::new(u as f32, v as f32));
                            let ray_color_res = self.ray_color(&ray, 0, 0.);

                            row_pixel_colors[x] = ray_color_res.pixel_color.get_attenuated_color();

                            if needs_albedo_and_normal_colors {
                                row_albedo_colors[x] = ray_color_res.albedo_color;
                                row_normal_colors[x] = ray_color_res.normal_color;
                            }
                        }

                        add_row_data(yi, &mut pixel_colors.lock().unwrap(), &row_pixel_colors);
                        if needs_albedo_and_normal_colors {
                            add_row_data(
                                yi,
                                &mut albedo_colors.lock().unwrap(),
                                &row_albedo_colors,
                            );
                            add_row_data(
                                yi,
                                &mut normal_colors.lock().unwrap(),
                                &row_normal_colors,
                            );
                        }
                    });
                }
            });

            {
                let now = SystemTime::now();
                let render_image = if self
                    .scene
                    .render_config
                    .render_image_strategy
                    .should_generate_image(
                        sample,
                        samples_per_pixel,
                        now,
                        last_image_generated_time,
                    ) {
                    last_image_generated_time = now;

                    if let Some((last_post_processor, intermediate_post_processors)) =
                        self.scene.render_config.post_processors.split_last()
                    {
                        if abort.try_recv().is_ok() {
                            return Ok(());
                        }

                        let mut intermediate_pixel_colors = pixel_colors.lock().unwrap().clone();

                        for ipp in intermediate_post_processors {
                            let processed_pixel_colors = ipp.intermediate_post_process(
                                &intermediate_pixel_colors,
                                albedo_colors.lock().unwrap().deref(),
                                normal_colors.lock().unwrap().deref(),
                                sample,
                            )?;

                            intermediate_pixel_colors = processed_pixel_colors;
                        }

                        Some(last_post_processor.post_process(
                            &intermediate_pixel_colors,
                            albedo_colors.lock().unwrap().deref(),
                            normal_colors.lock().unwrap().deref(),
                            sample,
                        )?)
                    } else {
                        None
                    }
                } else {
                    None
                };

                output.send(RenderProgress {
                    progress: sample as f64 / samples_per_pixel as f64,
                    fps: Some(calculate_fps(render_start_time, now, sample)),
                    estimated_time_left: calculate_estimated_time_left(
                        render_start_time,
                        now,
                        sample,
                        samples_per_pixel,
                    ),
                    render_image,
                })?
            }
        }
        Ok(())
    }
}

fn add_row_data(yi: usize, colors: &mut [Vec3], row_colors: &[Vec3]) {
    for (x, c) in row_colors.iter().enumerate() {
        colors[yi + x] += *c;
    }
}

fn calculate_fps(render_start_time: SystemTime, now: SystemTime, samples_done: u32) -> f64 {
    let time_since_start = now
        .duration_since(render_start_time)
        .unwrap_or(Duration::from_millis(1));

    samples_done as f64 / time_since_start.as_secs_f64()
}

fn calculate_estimated_time_left(
    render_start_time: SystemTime,
    now: SystemTime,
    samples_done: u32,
    total_samples: u32,
) -> Duration {
    let time_since_start = now
        .duration_since(render_start_time)
        .unwrap_or(Duration::from_millis(1));
    let samples_left = total_samples - samples_done;

    time_since_start
        .div_f32(samples_done as f32)
        .mul_f32(samples_left as f32)
}

#[cfg(test)]
mod test {
    use std::time::{Duration, SystemTime};

    use crate::renderer::{calculate_estimated_time_left, calculate_fps};

    #[test]
    fn test_calculate_fps() {
        let render_start = SystemTime::UNIX_EPOCH + Duration::from_millis(900);
        let now = SystemTime::UNIX_EPOCH + Duration::from_millis(1000);

        let fps = calculate_fps(render_start, now, 5);
        assert_eq!(fps, 50.);
    }

    #[test]
    fn test_calculate_estimated_time_left() {
        let render_start = SystemTime::UNIX_EPOCH;
        let now = SystemTime::UNIX_EPOCH + Duration::from_millis(1000);

        let mut time_left = calculate_estimated_time_left(render_start, now, 1, 100);
        assert_eq!(time_left, Duration::from_secs(99));

        time_left = calculate_estimated_time_left(render_start, now, 50, 100);
        assert_eq!(time_left, Duration::from_secs(1));

        time_left = calculate_estimated_time_left(render_start, now, 100, 100);
        assert_eq!(time_left, Duration::from_secs(0));
    }
}
