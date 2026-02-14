use std::sync::mpsc::channel;
use std::thread;
use std::time::Duration;
use solstrale::camera::CameraConfig;
use solstrale::geo::vec3::Vec3;
use solstrale::hittable::{Bvh, Sphere};
use solstrale::material::DiffuseLight;
use solstrale::ray_trace;
use solstrale::renderer::{RenderConfig, Scene};
use solstrale::util::wgpu_util::get_wgpu_device_and_queue;

#[test]
fn test_interactive_camera_restart() {
    let (device, queue) = get_wgpu_device_and_queue();
    let render_config = RenderConfig {
        width: 10,
        height: 10,
        samples_per_pixel: 100,
        ..Default::default()
    };
    let mut world = Vec::new();
    world.push(Sphere::new(Vec3::new(0., 10., 0.), 1., DiffuseLight::new(1., 1., 1., None).into()).into());

    let scene = Scene {
        world: Bvh::new(world).into(),
        camera: CameraConfig {
            look_from: Vec3::new(0., 0., 1.),
            ..Default::default()
        },
        background_color: Vec3::new(0., 0., 0.),
        render_config,
    };

    let (output_sender, output_receiver) = channel();
    let (camera_config_sender, camera_config_receiver) = channel();
    let (abort_sender, abort_receiver) = channel();

    thread::spawn(move || {
        ray_trace(scene, &output_sender, &camera_config_receiver, &abort_receiver, device, queue, true).unwrap();
    });

    // Wait for some progress
    let mut first_progress = 0.0;
    for progress in &output_receiver {
        if progress.progress > 0.1 {
            first_progress = progress.progress;
            break;
        }
    }
    assert!(first_progress > 0.1);

    // Update camera
    camera_config_sender.send(CameraConfig {
        look_from: Vec3::new(0., 0., 2.),
        ..Default::default()
    }).unwrap();

    // Verify progress restarts
    let mut restarted = false;
    for progress in &output_receiver {
        if progress.progress < first_progress {
            restarted = true;
            break;
        }
    }
    assert!(restarted);

    // Abort
    abort_sender.send(true).unwrap();
}
