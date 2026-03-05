use solstrale::camera::CameraConfig;
use solstrale::geo::Uv;
use solstrale::geo::transformation::{
    NopTransformer, RotationY, Transformations, Transformer, Translation,
};
use solstrale::geo::vec3::Vec3;
use solstrale::hittable::Hittables;
use solstrale::hittable::Sphere;
use solstrale::hittable::Triangle;
use solstrale::hittable::{Bvh, Quad};
use solstrale::loader::Loader;
use solstrale::loader::obj::Obj;
use solstrale::material::texture::{ImageMap, SolidColor, Textures, load_normal_texture};
use solstrale::material::{Blend, Dielectric, DiffuseLight, Lambertian};
use solstrale::renderer::{RenderConfig, Scene};

pub fn create_test_scene(render_config: RenderConfig) -> Scene {
    let camera = CameraConfig {
        vertical_fov_degrees: 20.,
        aperture_size: 0.1,
        look_from: Vec3::new(-5., 3., 6.),
        look_at: Vec3::new(0.25, 1., 0.),
        up: Vec3::new(0., 1., 0.),
    };

    let mut world: Vec<Hittables> = Vec::new();

    let image_tex = ImageMap::load("resources/textures/tex.jpg").unwrap();

    let ground_material = Lambertian::new(image_tex.into(), None);
    let glass_mat = Dielectric::new(SolidColor::new(1., 1., 1.).into(), None, 1.5);
    let light_mat = DiffuseLight::new(10., 10., 10., None);
    let red_mat = Lambertian::new(SolidColor::new(1., 0., 0.).into(), None);

    world.push(
        Quad::new(
            Vec3::new(-5., 0., -15.),
            Vec3::new(20., 0., 0.),
            Vec3::new(0., 0., 20.),
            ground_material.into(),
            &NopTransformer(),
        )
        .into(),
    );
    world.push(Sphere::new(Vec3::new(-1., 1., 0.), 1., glass_mat.into()).into());
    world.append(&mut Quad::new_box(
        Vec3::new(0., 0., -0.5),
        Vec3::new(1., 2., 0.5),
        red_mat.clone().into(),
        &RotationY::new(15.),
    ));
    world.append(&mut Quad::new_box(
        Vec3::new(-1., 2., 0.),
        Vec3::new(-0.5, 2.5, 0.5),
        red_mat.clone().into(),
        &NopTransformer(),
    ));

    let nop_transformer = NopTransformer();

    let mut balls: Vec<Hittables> = Vec::new();
    for ii in (0..10).step_by(2) {
        let i = ii as f64 * 0.1;
        for jj in (0..10).step_by(2) {
            let j = jj as f64 * 0.1;
            for kk in (0..10).step_by(2) {
                let k = kk as f64 * 0.1;
                balls.push(
                    Triangle::new(
                        Vec3::new(i, j + 0.05, k + 0.8),
                        Vec3::new(i, j, k + 0.8),
                        Vec3::new(i, j + 0.05, k),
                        red_mat.clone().into(),
                        &nop_transformer,
                    )
                    .into(),
                );
            }
        }
    }
    world.push(Bvh::new(balls).into());

    world.push(
        Triangle::new(
            Vec3::new(1., 0.1, 2.),
            Vec3::new(3., 0.1, 2.),
            Vec3::new(2., 0.1, 1.),
            red_mat.into(),
            &nop_transformer,
        )
        .into(),
    );

    // Lights

    world.push(Sphere::new(Vec3::new(10., 5., 10.), 10., light_mat.clone().into()).into());
    world.push(
        Quad::new(
            Vec3::new(0., 0., 0.),
            Vec3::new(2., 0., 0.),
            Vec3::new(0., 0., 2.),
            light_mat.clone().into(),
            &Transformations::new(vec![
                Box::new(RotationY::new(45.)),
                Box::new(Translation::new(Vec3::new(-1., 10., -1.))),
            ]),
        )
        .into(),
    );
    world.push(
        Triangle::new(
            Vec3::new(-2., 1., -3.),
            Vec3::new(0., 1., -3.),
            Vec3::new(-1., 2., -3.),
            light_mat.into(),
            &nop_transformer,
        )
        .into(),
    );

    Scene {
        world: Bvh::new(world).into(),
        camera,
        background_color: Vec3::new(0.2, 0.3, 0.5),
        render_config,
    }
}

#[allow(dead_code)]
pub fn new_bvh_test_scene(render_config: RenderConfig, use_bvh: bool, num_triangles: u32) -> Scene {
    let camera = CameraConfig {
        vertical_fov_degrees: 20.,
        aperture_size: 0.1,
        look_from: Vec3::new(-0.5, 0., 4.),
        look_at: Vec3::new(-0.5, 0., 0.),
        up: Vec3::new(0., 1., 0.),
    };

    let mut world: Vec<Hittables> = Vec::new();
    let yellow = Lambertian::new(SolidColor::new(1., 1., 0.).into(), None);
    let light = DiffuseLight::new(10., 10., 10., None);
    world.push(Sphere::new(Vec3::new(0., 4., 10.), 4., light.into()).into());

    let nop_transformer = NopTransformer();
    let mut triangles: Vec<Hittables> = Vec::new();
    for x in 0..num_triangles {
        let cx = x as f64 - num_triangles as f64 / 2.;
        let t = Triangle::new(
            Vec3::new(cx, -0.5, 0.),
            Vec3::new(cx + 1., -0.5, 0.),
            Vec3::new(cx + 0.5, 0.5, 0.),
            yellow.clone().into(),
            &nop_transformer,
        );
        if use_bvh {
            triangles.push(t.into());
        } else {
            world.push(t.into());
        }
    }

    if use_bvh {
        world.push(Bvh::new(triangles).into())
    }

    Scene {
        world: Bvh::new(world).into(),
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
        look_from: Vec3::new(0., 0., 4.),
        look_at: Vec3::new(0., 0., 0.),
        up: Vec3::new(0., 1., 0.),
    };

    let mut world: Vec<Hittables> = Vec::new();
    let yellow = Lambertian::new(SolidColor::new(1., 1., 0.).into(), None);
    let light = DiffuseLight::new(10., 10., 10., None);
    if add_light {
        world.push(Sphere::new(Vec3::new(0., 100., 0.), 20., light.into()).into())
    }
    world.push(Sphere::new(Vec3::new(0., 0., 0.), 0.5, yellow.into()).into());

    Scene {
        world: Bvh::new(world).into(),
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
        look_from: Vec3::new(0., 1., 5.),
        look_at: Vec3::new(0., 1., 0.),
        up: Vec3::new(0., 1., 0.),
    };

    let mut world: Vec<Hittables> = Vec::new();
    let light = DiffuseLight::new(10., 10., 10., None);

    world.push(Sphere::new(Vec3::new(50., 50., 50.), 20., light.into()).into());

    let tex = ImageMap::load("resources/textures/checker.jpg").unwrap();
    let checker_mat = Lambertian::new(tex.into(), None);

    world.push(
        Triangle::new_with_tex_coords(
            Vec3::new(-1., 0., 0.),
            Vec3::new(1., 0., 0.),
            Vec3::new(0., 2., 0.),
            Uv::new(-1., -1.),
            Uv::new(2., -1.),
            Uv::new(0., 2.),
            checker_mat.into(),
            &NopTransformer(),
        )
        .into(),
    );

    Scene {
        world: Bvh::new(world).into(),
        camera,
        background_color: Vec3::new(0.2, 0.3, 0.5),
        render_config,
    }
}

#[allow(dead_code)]
pub fn create_normal_mapping_scene(
    render_config: RenderConfig,
    light_pos: Vec3,
    normal_mapping_enabled: bool,
) -> Scene {
    let camera = CameraConfig {
        vertical_fov_degrees: 40.,
        aperture_size: 0.,
        look_from: Vec3::new(0.2, 0.2, 2.),
        look_at: Vec3::new(0., 0., 0.),
        up: Vec3::new(0., 1., 0.),
    };

    let mut world: Vec<Hittables> = Vec::new();
    let light = DiffuseLight::new(45., 45., 45., None);

    world.push(Sphere::new(light_pos, 5., light.into()).into());

    let normal_tex: Option<Textures> = if normal_mapping_enabled {
        Some(
            load_normal_texture("resources/textures/normal.png")
                .unwrap()
                .into(),
        )
    } else {
        None
    };
    let mat = Lambertian::new(SolidColor::new(0.8, 0.8, 0.8).into(), normal_tex);
    let red = Lambertian::new(SolidColor::new(1., 0., 0.).into(), None);

    world.append(&mut Quad::new_box(
        Vec3::new(-0.1, -0.1, 0.),
        Vec3::new(0.1, 0.1, 1.),
        red.into(),
        &NopTransformer(),
    ));

    world.push(
        Quad::new(
            Vec3::new(-1., -1., 0.),
            Vec3::new(2., 0., 0.),
            Vec3::new(0., 2., 0.),
            mat.into(),
            &NopTransformer(),
        )
        .into(),
    );

    Scene {
        world: Bvh::new(world).into(),
        camera,
        background_color: Vec3::new(0., 0., 0.),
        render_config,
    }
}

#[allow(dead_code)]
pub fn create_normal_mapping_sphere_scene(render_config: RenderConfig, light_pos: Vec3) -> Scene {
    let camera = CameraConfig {
        vertical_fov_degrees: 40.,
        aperture_size: 0.,
        look_from: Vec3::new(0.2, 0.2, 2.),
        look_at: Vec3::new(0., 0., 0.),
        up: Vec3::new(0., 1., 0.),
    };

    let mut world: Vec<Hittables> = Vec::new();
    let light = DiffuseLight::new(45., 45., 45., None);

    world.push(Sphere::new(light_pos, 5., light.into()).into());

    let normal_tex = Some(
        load_normal_texture("resources/textures/earth_height.jpg")
            .unwrap()
            .into(),
    );
    let mat = Lambertian::new(SolidColor::new(0.8, 0.8, 0.8).into(), normal_tex);

    world.push(Sphere::new(Vec3::new(0., 0., 0.), 0.6, mat.into()).into());

    Scene {
        world: Bvh::new(world).into(),
        camera,
        background_color: Vec3::new(0., 0., 0.),
        render_config,
    }
}

#[allow(dead_code)]
pub fn create_obj_scene(render_config: RenderConfig) -> Scene {
    let camera = CameraConfig {
        vertical_fov_degrees: 30.,
        aperture_size: 20.,
        look_from: Vec3::new(-250., 30., 150.),
        look_at: Vec3::new(-50., 0., 0.),
        up: Vec3::new(0., 1., 0.),
    };

    let mut world: Vec<Hittables> = Vec::new();
    let light = DiffuseLight::new(15., 15., 15., None);

    world.push(Sphere::new(Vec3::new(-100., 100., 40.), 35., light.into()).into());
    let model = Obj::new("resources/spider/", "spider.obj")
        .load(&NopTransformer(), None)
        .unwrap();
    world.push(model.into());

    let image_tex = ImageMap::load("resources/textures/tex.jpg").unwrap().into();
    let ground_material = Lambertian::new(image_tex, None);
    world.push(
        Quad::new(
            Vec3::new(-200., -30., -200.),
            Vec3::new(400., 0., 0.),
            Vec3::new(0., 0., 400.),
            ground_material.into(),
            &NopTransformer(),
        )
        .into(),
    );

    Scene {
        world: Bvh::new(world).into(),
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
        look_from: Vec3::new(2., 1., 3.),
        look_at: Vec3::new(0., 0., 0.),
        up: Vec3::new(0., 1., 0.),
    };

    let mut world: Vec<Hittables> = Vec::new();
    let light = DiffuseLight::new(15., 15., 15., None);
    let red = Lambertian::new(SolidColor::new(1., 0., 0.).into(), None);

    world.push(Sphere::new(Vec3::new(-100., 100., 40.), 35., light.into()).into());
    world.push(
        Obj::new(path, filename)
            .load(&NopTransformer(), Some(red.into()))
            .unwrap()
            .into(),
    );

    Scene {
        world: Bvh::new(world).into(),
        camera,
        background_color: Vec3::new(0.2, 0.3, 0.5),
        render_config,
    }
}

#[allow(dead_code)]
pub fn create_obj_with_triangle(render_config: RenderConfig, path: &str, filename: &str) -> Scene {
    let camera = CameraConfig {
        vertical_fov_degrees: 30.,
        aperture_size: 0.,
        look_from: Vec3::new(0., 0., 2.),
        look_at: Vec3::new(0., 0., 0.),
        up: Vec3::new(0., 1., 0.),
    };

    let mut world: Vec<Hittables> = Vec::new();
    let light = DiffuseLight::new(15., 15., 15., None);

    world.push(Sphere::new(Vec3::new(100., 0., 100.), 35., light.into()).into());
    world.push(
        Obj::new(path, filename)
            .load(&NopTransformer(), None)
            .unwrap()
            .into(),
    );

    Scene {
        world: Bvh::new(world).into(),
        camera,
        background_color: Vec3::new(0., 0., 0.),
        render_config,
    }
}

#[allow(dead_code)]
pub fn create_light_attenuation_scene(
    render_config: RenderConfig,
    attenuation_half_length: Option<f64>,
) -> Scene {
    let camera = CameraConfig {
        vertical_fov_degrees: 20.,
        aperture_size: 0.,
        look_from: Vec3::new(0., 1., 2.),
        look_at: Vec3::new(0., 0.2, 0.),
        up: Vec3::new(0., 1., 0.),
    };

    let mut world: Vec<Hittables> = Vec::new();
    let light = DiffuseLight::new(25., 25., 25., attenuation_half_length);
    let red = Lambertian::new(SolidColor::new(1., 0., 0.).into(), None);
    let green = Lambertian::new(SolidColor::new(0., 1., 0.).into(), None);
    let blue = Lambertian::new(SolidColor::new(0., 0., 1.).into(), None);
    let glass = Dielectric::new(SolidColor::new(0.8, 0.8, 0.8).into(), None, 1.5);

    world.push(Sphere::new(Vec3::new(0., 0.2, 0.), 0.03, light.into()).into());
    world.push(Sphere::new(Vec3::new(0.25, 0.1, 0.25), 0.1, green.into()).into());
    world.push(Sphere::new(Vec3::new(0.25, 0.1, -0.5), 0.1, blue.into()).into());
    world.push(Sphere::new(Vec3::new(-0.1, 0.1, -0.1), 0.1, glass.into()).into());
    world.push(
        Quad::new(
            Vec3::new(-1., 0., -1.),
            Vec3::new(2., 0., 0.),
            Vec3::new(0., 0., 2.),
            red.into(),
            &NopTransformer(),
        )
        .into(),
    );

    Scene {
        world: Bvh::new(world).into(),
        camera,
        background_color: Vec3::new(0., 0., 0.),
        render_config,
    }
}

#[allow(dead_code)]
pub fn create_quad_rotation_scene(
    render_config: RenderConfig,
    rotation: &dyn Transformer,
) -> Scene {
    Scene {
        world: Bvh::new(vec![
            Quad::new(
                Vec3::new(-100., 0., -100.),
                Vec3::new(200., 0., 0.),
                Vec3::new(0., 0., 200.),
                Lambertian::new(SolidColor::new(0., 1., 0.).into(), None).into(),
                rotation,
            )
            .into(),
            Sphere::new(
                Vec3::new(100., 300., -500.),
                50.,
                DiffuseLight::new(15., 15., 15., None).into(),
            )
            .into(),
        ])
        .into(),
        camera: CameraConfig {
            vertical_fov_degrees: 35.0,
            look_from: Vec3::new(0., 200., -500.),
            ..CameraConfig::default()
        },
        background_color: Default::default(),
        render_config,
    }
}

#[allow(dead_code)]
pub fn create_blend_material_scene(render_config: RenderConfig, blend_factor: f64) -> Scene {
    Scene {
        world: Bvh::new(vec![
            Quad::new(
                Vec3::new(-100., 0., -100.),
                Vec3::new(200., 0., 0.),
                Vec3::new(0., 0., 200.),
                Blend::new(
                    Lambertian::new(
                        ImageMap::load("resources/textures/checker.jpg")
                            .unwrap()
                            .into(),
                        None,
                    )
                    .into(),
                    Lambertian::new(SolidColor::new(0., 1., 0.).into(), None).into(),
                    blend_factor,
                )
                .into(),
                &NopTransformer(),
            )
            .into(),
            Sphere::new(
                Vec3::new(0., 500., -200.),
                50.,
                DiffuseLight::new(15., 15., 15., None).into(),
            )
            .into(),
        ])
        .into(),
        camera: CameraConfig {
            vertical_fov_degrees: 35.0,
            look_from: Vec3::new(0., 400., -100.),
            ..CameraConfig::default()
        },
        background_color: Default::default(),
        render_config,
    }
}

#[allow(dead_code)]
pub fn create_texture_mapping_scene(render_config: RenderConfig) -> Scene {
    Scene {
        world: Bvh::new(vec![
            Quad::new(
                Vec3::new(-100., 0., -100.),
                Vec3::new(200., 0., 0.),
                Vec3::new(0., 0., 200.),
                Lambertian::new(
                    ImageMap::load("resources/textures/checker.jpg")
                        .unwrap()
                        .into(),
                    None,
                )
                .into(),
                &NopTransformer(),
            )
            .into(),
            Sphere::new(
                Vec3::new(0., 500., -200.),
                50.,
                DiffuseLight::new(15., 15., 15., None).into(),
            )
            .into(),
        ])
        .into(),
        camera: CameraConfig {
            vertical_fov_degrees: 35.0,
            look_from: Vec3::new(0., 400., -100.),
            ..CameraConfig::default()
        },
        background_color: Default::default(),
        render_config,
    }
}
