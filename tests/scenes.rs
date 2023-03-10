use solstrale::camera::CameraConfig;
use solstrale::geo::vec3::Vec3;
use solstrale::geo::Uv;
use solstrale::hittable::bvh::Bvh;
use solstrale::hittable::constant_medium::ConstantMedium;
use solstrale::hittable::hittable_list::HittableList;
use solstrale::hittable::motion_blur::MotionBlur;
use solstrale::hittable::obj_model::{load_obj_model, load_obj_model_with_default_material};
use solstrale::hittable::quad::Quad;
use solstrale::hittable::rotation_y::RotationY;
use solstrale::hittable::sphere::Sphere;
use solstrale::hittable::translation::Translation;
use solstrale::hittable::triangle::Triangle;
use solstrale::hittable::Hittable;
use solstrale::hittable::Hittables::TriangleType;
use solstrale::material::texture::{ImageTexture, SolidColor};
use solstrale::material::{Dielectric, DiffuseLight, Lambertian};
use solstrale::renderer::{RenderConfig, Scene};

pub fn create_test_scene(render_config: RenderConfig) -> Scene {
    let camera = CameraConfig {
        vertical_fov_degrees: 20.,
        aperture_size: 0.1,
        focus_distance: 10.,
        look_from: Vec3::new(-5., 3., 6.),
        look_at: Vec3::new(0.25, 1., 0.),
    };

    let mut world = HittableList::create();

    let image_tex = ImageTexture::load("resources/textures/tex.jpg").unwrap();

    let ground_material = Lambertian::create(image_tex);
    let glass_mat = Dielectric::create(SolidColor::create(1., 1., 1.), 1.5);
    let light_mat = DiffuseLight::create(10., 10., 10.);
    let red_mat = Lambertian::create(SolidColor::create(1., 0., 0.));

    world.add(Quad::create(
        Vec3::new(-5., 0., -15.),
        Vec3::new(20., 0., 0.),
        Vec3::new(0., 0., 20.),
        ground_material,
    ));
    world.add(Sphere::create(Vec3::new(-1., 1., 0.), 1., glass_mat));
    world.add(RotationY::create(
        Quad::create_box(
            Vec3::new(0., 0., -0.5),
            Vec3::new(1., 2., 0.5),
            red_mat.clone(),
        ),
        15.,
    ));
    world.add(ConstantMedium::create(
        Translation::create(
            Quad::create_box(
                Vec3::new(0., 0., -0.5),
                Vec3::new(1., 2., 0.5),
                red_mat.clone(),
            ),
            Vec3::new(0., 0., 1.),
        ),
        0.1,
        Vec3::new(1., 1., 1.),
    ));
    world.add(MotionBlur::create(
        Quad::create_box(
            Vec3::new(-1., 2., 0.),
            Vec3::new(-0.5, 2.5, 0.5),
            red_mat.clone(),
        ),
        Vec3::new(0., 1., 0.),
    ));

    let mut balls = Vec::new();
    for ii in (0..10).step_by(2) {
        let i = ii as f64 * 0.1;
        for jj in (0..10).step_by(2) {
            let j = jj as f64 * 0.1;
            for kk in (0..10).step_by(2) {
                let k = kk as f64 * 0.1;
                if let TriangleType(t) = Triangle::create(
                    Vec3::new(i, j + 0.05, k + 0.8),
                    Vec3::new(i, j, k + 0.8),
                    Vec3::new(i, j + 0.05, k),
                    red_mat.clone(),
                ) {
                    balls.push(t)
                }
            }
        }
    }
    world.add(Bvh::create(balls));

    world.add(Triangle::create(
        Vec3::new(1., 0.1, 2.),
        Vec3::new(3., 0.1, 2.),
        Vec3::new(2., 0.1, 1.),
        red_mat,
    ));

    // Lights

    world.add(Sphere::create(
        Vec3::new(10., 5., 10.),
        10.,
        light_mat.clone(),
    ));
    world.add(Translation::create(
        RotationY::create(
            Quad::create(
                Vec3::new(0., 0., 0.),
                Vec3::new(2., 0., 0.),
                Vec3::new(0., 0., 2.),
                light_mat.clone(),
            ),
            45.,
        ),
        Vec3::new(-1., 10., -1.),
    ));
    world.add(Triangle::create(
        Vec3::new(-2., 1., -3.),
        Vec3::new(0., 1., -3.),
        Vec3::new(-1., 2., -3.),
        light_mat,
    ));

    Scene {
        world,
        camera,
        background_color: Vec3::new(0.2, 0.3, 0.5),
        render_config,
    }
}

#[allow(dead_code)]
pub fn create_bvh_test_scene(
    render_config: RenderConfig,
    use_bvh: bool,
    num_triangles: u32,
) -> Scene {
    let camera = CameraConfig {
        vertical_fov_degrees: 20.,
        aperture_size: 0.1,
        focus_distance: 10.,
        look_from: Vec3::new(-0.5, 0., 4.),
        look_at: Vec3::new(-0.5, 0., 0.),
    };

    let mut world = HittableList::create();
    let yellow = Lambertian::create(SolidColor::create(1., 1., 0.));
    let light = DiffuseLight::create(10., 10., 10.);
    world.add(Sphere::create(Vec3::new(0., 4., 10.), 4., light));

    let mut triangles = Vec::new();
    for x in 0..num_triangles {
        let cx = x as f64 - num_triangles as f64 / 2.;
        let t = Triangle::create(
            Vec3::new(cx, -0.5, 0.),
            Vec3::new(cx + 1., -0.5, 0.),
            Vec3::new(cx + 0.5, 0.5, 0.),
            yellow.clone(),
        );
        if use_bvh {
            if let TriangleType(tri) = t {
                triangles.push(tri);
            }
        } else {
            world.add(t);
        }
    }

    if use_bvh {
        world.add(Bvh::create(triangles))
    }

    Scene {
        world,
        camera,
        background_color: Vec3::new(0.2, 0.3, 0.5),
        render_config,
    }
}

#[allow(dead_code)]
pub fn create_simple_test_scene(render_config: RenderConfig, add_light: bool) -> Scene {
    let camera = CameraConfig {
        vertical_fov_degrees: 20.,
        aperture_size: 0.1,
        focus_distance: 10.,
        look_from: Vec3::new(0., 0., 4.),
        look_at: Vec3::new(0., 0., 0.),
    };

    let mut world = HittableList::create();
    let yellow = Lambertian::create(SolidColor::create(1., 1., 0.));
    let light = DiffuseLight::create(10., 10., 10.);
    if add_light {
        world.add(Sphere::create(Vec3::new(0., 100., 0.), 20., light))
    }
    world.add(Sphere::create(Vec3::new(0., 0., 0.), 0.5, yellow));

    Scene {
        world,
        camera,
        background_color: Vec3::new(0.2, 0.3, 0.5),
        render_config,
    }
}

#[allow(dead_code)]
pub fn create_uv_scene(render_config: RenderConfig) -> Scene {
    let camera = CameraConfig {
        vertical_fov_degrees: 20.,
        aperture_size: 0.,
        focus_distance: 1.,
        look_from: Vec3::new(0., 1., 5.),
        look_at: Vec3::new(0., 1., 0.),
    };

    let mut world = HittableList::create();
    let light = DiffuseLight::create(10., 10., 10.);

    world.add(Sphere::create(Vec3::new(50., 50., 50.), 20., light));

    let tex = ImageTexture::load("resources/textures/checker.jpg").unwrap();
    let checker_mat = Lambertian::create(tex);

    world.add(Triangle::create_with_tex_coords(
        Vec3::new(-1., 0., 0.),
        Vec3::new(1., 0., 0.),
        Vec3::new(0., 2., 0.),
        Uv::new(-1., -1.),
        Uv::new(2., -1.),
        Uv::new(0., 2.),
        checker_mat,
    ));

    Scene {
        world,
        camera,
        background_color: Vec3::new(0.2, 0.3, 0.5),
        render_config,
    }
}

#[allow(dead_code)]
pub fn create_obj_scene(render_config: RenderConfig) -> Scene {
    let camera = CameraConfig {
        vertical_fov_degrees: 30.,
        aperture_size: 20.,
        focus_distance: 260.,
        look_from: Vec3::new(-250., 30., 150.),
        look_at: Vec3::new(-50., 0., 0.),
    };

    let mut world = HittableList::create();
    let light = DiffuseLight::create(15., 15., 15.);

    world.add(Sphere::create(Vec3::new(-100., 100., 40.), 35., light));
    let model = load_obj_model("resources/spider/", "spider.obj", 1.).unwrap();
    world.add(model);

    let image_tex = ImageTexture::load("resources/textures/tex.jpg").unwrap();
    let ground_material = Lambertian::create(image_tex);
    world.add(Quad::create(
        Vec3::new(-200., -30., -200.),
        Vec3::new(400., 0., 0.),
        Vec3::new(0., 0., 400.),
        ground_material,
    ));

    Scene {
        world,
        camera,
        background_color: Vec3::new(0.2, 0.3, 0.5),
        render_config,
    }
}

#[allow(dead_code)]
pub fn create_obj_with_box(render_config: RenderConfig, path: &str, filename: &str) -> Scene {
    let camera = CameraConfig {
        vertical_fov_degrees: 30.,
        aperture_size: 0.,
        focus_distance: 1.,
        look_from: Vec3::new(2., 1., 3.),
        look_at: Vec3::new(0., 0., 0.),
    };

    let mut world = HittableList::create();
    let light = DiffuseLight::create(15., 15., 15.);
    let red = Lambertian::create(SolidColor::create(1., 0., 0.));

    world.add(Sphere::create(Vec3::new(-100., 100., 40.), 35., light));
    world.add(load_obj_model_with_default_material(path, filename, 1., red).unwrap());

    Scene {
        world,
        camera,
        background_color: Vec3::new(0.2, 0.3, 0.5),
        render_config,
    }
}
