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
use solstrale::material::{Blend, Dielectric, DiffuseLight, Lambertian, Metal};
use solstrale::renderer::{RenderConfig, Scene};

/// A scene built to exercise the denoiser's specular guide: a mirror sphere and
/// a glass sphere fill much of the frame, and what they show is a textured
/// floor, so a filter that blurs the reflected and refracted image along the
/// surface has somewhere visible to do it.
///
/// A sphere rather than a flat mirror on purpose. A mirror wall reflects
/// whatever is behind the camera, which in a small scene is mostly empty
/// background; a sphere gathers the whole surroundings into a few hundred
/// pixels, so the reflected detail is dense and its depth varies sharply across
/// neighbouring pixels -- which is exactly what the guide has to track.
///
/// No aperture, so the guide's pixel-centre ray is the ray the samples average
/// around and depth of field is not part of what is being measured.
#[allow(dead_code)]
pub fn create_specular_scene(render_config: RenderConfig) -> Scene {
    let camera = CameraConfig {
        vertical_fov_degrees: 35.,
        aperture_size: 0.,
        look_from: Vec3::new(0., 2.2, 6.5),
        look_at: Vec3::new(0., 1.1, 0.),
        up: Vec3::new(0., 1., 0.),
    };

    let image_tex = ImageMap::load("resources/textures/tex.jpg").unwrap();
    let floor_material = Lambertian::new(image_tex.into(), None);
    let mirror_mat = Metal::new(SolidColor::new(0.9, 0.9, 0.9).into(), None, 0.);
    let glass_mat = Dielectric::new(SolidColor::new(1., 1., 1.).into(), None, 1.5);
    let red_mat = Lambertian::new(SolidColor::new(0.8, 0.1, 0.1).into(), None);
    let light_mat = DiffuseLight::new(10., 10., 10., None);

    let nop = NopTransformer();
    let mut world: Vec<Hittables> = Vec::new();

    // Textured floor. The texture is the detail the reflected and refracted
    // image is made of -- a flat colour here would measure nothing.
    world.push(
        Quad::new(
            Vec3::new(-10., 0., -8.),
            Vec3::new(20., 0., 0.),
            Vec3::new(0., 0., 16.),
            floor_material.into(),
            &nop,
        )
        .into(),
    );

    world.push(Sphere::new(Vec3::new(1.5, 1.2, 0.), 1.2, mirror_mat.into()).into());
    world.push(Sphere::new(Vec3::new(-1.5, 1., 0.3), 1., glass_mat.into()).into());

    // Something with a recognisable shape and a colour of its own, to be seen
    // reflected in one sphere and refracted through the other.
    world.append(&mut Quad::new_box(
        Vec3::new(-0.6, 0., -2.),
        Vec3::new(0.6, 1.4, -0.8),
        red_mat.into(),
        &RotationY::new(20.),
    ));

    // One light, above and in front, facing down.
    world.push(
        Quad::new(
            Vec3::new(-2., 6., -1.),
            Vec3::new(4., 0., 0.),
            Vec3::new(0., 0., 4.),
            light_mat.into(),
            &nop,
        )
        .into(),
    );

    Scene {
        world: Bvh::new(world).into(),
        camera,
        background_color: Vec3::new(0.1, 0.15, 0.25),
        render_config,
    }
}

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

/// A strip of `num_triangles` colinear triangles lit by a sphere light.
///
/// `nested` decides whether the triangles get their own [`Bvh`] *inside* the
/// world, not whether there is a BVH at all: the world itself is always wrapped
/// in one, because the flattener and the GPU traversal have no other shape to
/// take. So the `false` arm is a flat one-level tree over every triangle, not
/// a linear scan, and a measurement against it is a measurement of nesting.
#[allow(dead_code)]
pub fn new_bvh_test_scene(render_config: RenderConfig, nested: bool, num_triangles: u32) -> Scene {
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
        if nested {
            triangles.push(t.into());
        } else {
            world.push(t.into());
        }
    }

    if nested {
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

/// A coarse UV sphere built twice, side by side: the left one flat-shaded with
/// [`Triangle::new`], the right one smooth-shaded with
/// [`Triangle::new_with_normals`] from the analytic normals the sphere has by
/// construction.
///
/// Both in one frame rather than two goldens: a single smooth sphere at 100x50
/// is just a sphere, while faceted next to smooth is a difference no downscale
/// hides. If `oct_decode` ever stops agreeing with `pack_oct`, the two halves
/// stop differing and the test notices.
///
/// Under-tessellated on purpose -- 10 by 6, 100 triangles. Both silhouettes
/// stay polygonal, which is correct, and it is coarse enough to show the shadow
/// terminator recorded in LIMITATIONS.md.
#[allow(dead_code)]
pub fn create_smooth_vs_flat_scene(render_config: RenderConfig) -> Scene {
    const SEGMENTS: usize = 10;
    const RINGS: usize = 6;
    const RADIUS: f64 = 1.35;

    let camera = CameraConfig {
        vertical_fov_degrees: 30.,
        aperture_size: 0.,
        look_from: Vec3::new(0., 0., 7.),
        look_at: Vec3::new(0., 0., 0.),
        up: Vec3::new(0., 1., 0.),
    };

    // Polar angle from +Y, azimuth around it. The outward normal of a unit
    // sphere at the origin is the point itself, so the per-vertex normals are
    // analytic -- no generation, nothing that could be wrong the same way the
    // shader is.
    let unit_at = |theta: f64, phi: f64| {
        Vec3::new(
            theta.sin() * phi.cos(),
            theta.cos(),
            theta.sin() * phi.sin(),
        )
    };

    let mat = Lambertian::new(SolidColor::new(0.8, 0.75, 0.7).into(), None);

    let mut faceted: Vec<Hittables> = Vec::new();
    let mut smooth: Vec<Hittables> = Vec::new();
    for ring in 0..RINGS {
        let theta0 = std::f64::consts::PI * ring as f64 / RINGS as f64;
        let theta1 = std::f64::consts::PI * (ring + 1) as f64 / RINGS as f64;
        for segment in 0..SEGMENTS {
            let phi0 = 2. * std::f64::consts::PI * segment as f64 / SEGMENTS as f64;
            let phi1 = 2. * std::f64::consts::PI * (segment + 1) as f64 / SEGMENTS as f64;

            let a = unit_at(theta0, phi0);
            let b = unit_at(theta1, phi0);
            let c = unit_at(theta1, phi1);
            let d = unit_at(theta0, phi1);

            // The pole rings are fans: at ring 0 `a` and `d` are both the
            // north pole, at the last `b` and `c` are both the south, so one
            // triangle has zero area and is dropped.
            let mut corners: Vec<[Vec3; 3]> = Vec::with_capacity(2);
            if ring != RINGS - 1 {
                corners.push([a, c, b]);
            }
            if ring != 0 {
                corners.push([a, d, c]);
            }

            for n in corners {
                let v = n.map(|n| n * RADIUS);
                let left = Translation::new(Vec3::new(-1.5, 0., 0.));
                let right = Translation::new(Vec3::new(1.5, 0., 0.));

                faceted.push(Triangle::new(v[0], v[1], v[2], mat.clone().into(), &left).into());
                smooth.push(
                    Triangle::new_with_normals(
                        v,
                        n,
                        [Uv::default(); 3],
                        mat.clone().into(),
                        &right,
                    )
                    .into(),
                );
            }
        }
    }

    let mut world: Vec<Hittables> = vec![
        // Off to one side and above, so both spheres get a terminator running
        // across the visible face rather than a flat front-lit disc.
        Sphere::new(
            Vec3::new(-5., 3., 3.),
            1.,
            DiffuseLight::new(28., 28., 28., None).into(),
        )
        .into(),
    ];
    world.push(Bvh::new(faceted).into());
    world.push(Bvh::new(smooth).into());

    Scene {
        world: Bvh::new(world).into(),
        camera,
        background_color: Vec3::new(0.05, 0.06, 0.08),
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

/// A Cornell box: the canonical firefly generator, and the one thing missing
/// from the scenes above.
///
/// Every other scene here is lit by something huge -- `create_test_scene` has
/// three lights, one a radius-10 sphere, plus a bright background -- which is
/// precisely why the denoiser's firefly behaviour never showed up in CI. What
/// produces a firefly is the opposite: a *small* bright emitter, no ambient
/// background at all, and a surface dark enough that one improbable bright
/// bounce lands tens of times above its converged radiance. All three are
/// deliberate here.
///
/// The black box is the subject. At an albedo of 0.05 its converged radiance is
/// a few hundredths, so a single indirect sample that finds the light arrives
/// 40-75x above the mean -- far under `CLAMPING_THRESHOLD`, and exactly the
/// regime the renderer's absolute clamp is documented as not covering.
#[allow(dead_code)]
pub fn create_cornell_scene(render_config: RenderConfig) -> Scene {
    let camera = CameraConfig {
        vertical_fov_degrees: 40.,
        aperture_size: 0.,
        look_from: Vec3::new(278., 278., -800.),
        look_at: Vec3::new(278., 278., 0.),
        up: Vec3::new(0., 1., 0.),
    };

    let red = Lambertian::new(SolidColor::new(0.65, 0.05, 0.05).into(), None);
    let green = Lambertian::new(SolidColor::new(0.12, 0.45, 0.15).into(), None);
    let white = Lambertian::new(SolidColor::new(0.73, 0.73, 0.73).into(), None);
    // Dark enough that a single lucky bounce dwarfs the converged value.
    let black = Lambertian::new(SolidColor::new(0.05, 0.05, 0.05).into(), None);
    let light = DiffuseLight::new(25., 25., 25., None);

    let nop = NopTransformer();
    let mut world: Vec<Hittables> = Vec::new();

    // Walls, floor and ceiling of a 555-unit cube, open toward the camera.
    world.push(
        Quad::new(
            Vec3::new(555., 0., 0.),
            Vec3::new(0., 555., 0.),
            Vec3::new(0., 0., 555.),
            green.into(),
            &nop,
        )
        .into(),
    );
    world.push(
        Quad::new(
            Vec3::new(0., 0., 0.),
            Vec3::new(0., 555., 0.),
            Vec3::new(0., 0., 555.),
            red.into(),
            &nop,
        )
        .into(),
    );
    world.push(
        Quad::new(
            Vec3::new(0., 0., 0.),
            Vec3::new(555., 0., 0.),
            Vec3::new(0., 0., 555.),
            white.clone().into(),
            &nop,
        )
        .into(),
    );
    world.push(
        Quad::new(
            Vec3::new(0., 555., 0.),
            Vec3::new(555., 0., 0.),
            Vec3::new(0., 0., 555.),
            white.clone().into(),
            &nop,
        )
        .into(),
    );
    world.push(
        Quad::new(
            Vec3::new(0., 0., 555.),
            Vec3::new(555., 0., 0.),
            Vec3::new(0., 555., 0.),
            white.clone().into(),
            &nop,
        )
        .into(),
    );

    // Small emitter in the ceiling. Small is the point: a light subtending a
    // narrow solid angle is what makes an indirect hit on it improbable and
    // therefore bright when it happens.
    world.push(
        Quad::new(
            Vec3::new(213., 554., 227.),
            Vec3::new(130., 0., 0.),
            Vec3::new(0., 0., 105.),
            light.into(),
            &nop,
        )
        .into(),
    );

    // A tall white box, and a short black one that is where the fireflies land.
    world.append(&mut Quad::new_box(
        Vec3::new(0., 0., 0.),
        Vec3::new(165., 330., 165.),
        white.into(),
        &Transformations::new(vec![
            Box::new(RotationY::new(15.)),
            Box::new(Translation::new(Vec3::new(265., 0., 295.))),
        ]),
    ));
    world.append(&mut Quad::new_box(
        Vec3::new(0., 0., 0.),
        Vec3::new(165., 165., 165.),
        black.into(),
        &Transformations::new(vec![
            Box::new(RotationY::new(-18.)),
            Box::new(Translation::new(Vec3::new(130., 0., 65.))),
        ]),
    ));

    Scene {
        world: Bvh::new(world).into(),
        camera,
        // No ambient light whatsoever. An ambient background is what stops the
        // other scenes here producing fireflies at all.
        background_color: Vec3::new(0., 0., 0.),
        render_config,
    }
}

/// A five-walled box, every wall carrying the material under test, lit by a
/// sphere hanging inside it. Built for the sRGB decode tests.
///
/// The walls face each other, so a path bounces off the same albedo two or
/// three times before it dies and the albedo enters the product as `albedo^n`.
/// A decode that is missing, doubled, or applied at the wrong point in the
/// chain shows up here as a compounding error rather than a flat shift that a
/// loose tolerance could swallow.
#[allow(dead_code)]
pub fn create_srgb_decode_scene(
    render_config: RenderConfig,
    albedo: Textures,
    normal: Option<Textures>,
) -> Scene {
    let mat = Lambertian::new(albedo, normal);
    let nop = NopTransformer();

    let wall = |origin: Vec3, u: Vec3, v: Vec3| -> Hittables {
        Quad::new(origin, u, v, mat.clone().into(), &nop).into()
    };

    let world: Vec<Hittables> = vec![
        // Floor, ceiling, back, left, right. The front is left open for the
        // camera; rays that escape through it hit the black background.
        wall(
            Vec3::new(-1., 0., -1.),
            Vec3::new(2., 0., 0.),
            Vec3::new(0., 0., 2.),
        ),
        wall(
            Vec3::new(-1., 2., -1.),
            Vec3::new(2., 0., 0.),
            Vec3::new(0., 0., 2.),
        ),
        wall(
            Vec3::new(-1., 0., -1.),
            Vec3::new(2., 0., 0.),
            Vec3::new(0., 2., 0.),
        ),
        wall(
            Vec3::new(-1., 0., -1.),
            Vec3::new(0., 0., 2.),
            Vec3::new(0., 2., 0.),
        ),
        wall(
            Vec3::new(1., 0., -1.),
            Vec3::new(0., 0., 2.),
            Vec3::new(0., 2., 0.),
        ),
        Sphere::new(
            Vec3::new(0., 1.5, -0.2),
            0.25,
            DiffuseLight::new(12., 12., 12., None).into(),
        )
        .into(),
    ];

    Scene {
        world: Bvh::new(world).into(),
        camera: CameraConfig {
            vertical_fov_degrees: 55.,
            aperture_size: 0.,
            look_from: Vec3::new(0., 1., 2.6),
            look_at: Vec3::new(0., 0.9, 0.),
            up: Vec3::new(0., 1., 0.),
        },
        background_color: Vec3::new(0., 0., 0.),
        render_config,
    }
}
