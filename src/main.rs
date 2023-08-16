mod app;
mod multipass_system;
mod networking;
mod projectile_sim_manager;
mod rasterizer;
mod raytracer;
mod rollback_manager;
mod voxel_sim_manager;
mod world_gen;

use crate::app::RenderPipeline;
use cgmath::{Matrix4, Point3, SquareMatrix, Vector2, Vector3};
use multipass_system::Pass;
use networking::NetworkConnection;
use rand::Rng;
use rollback_manager::{Player, PlayerAction};
use std::time::Instant;
use vulkano::{
    image::view::ImageView,
    swapchain::{
        acquire_next_image, AcquireError, SwapchainCreateInfo, SwapchainCreationError,
        SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
};
use winit::{
    event::{ElementState, Event, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
    window::Window,
};

pub const WINDOW_WIDTH: f32 = 1024.0;
pub const WINDOW_HEIGHT: f32 = 1024.0;
pub const RENDER_SIZE: [u32; 3] = [8, 8, 8];
pub const CHUNK_SIZE: u32 = 16;
pub const TOTAL_VOXEL_COUNT: usize =
    (RENDER_SIZE[0] * RENDER_SIZE[1] * RENDER_SIZE[2] * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE)
        as usize;
pub const MIN_PLAYERS: usize = 1;

struct SimSettings {
    pub max_dist: u32,
    pub do_compute: bool,
}

pub struct SimData {
    max_dist: u32,
    render_size: [u32; 3],
    start_pos: [i32; 3],
}

pub struct Controls {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    sprint: bool,
    mouse_left: bool,
    mouse_right: bool,
    do_chunk_load: bool,
    mouse_move: [f32; 2],
}

pub const FIRST_START_POS: [i32; 3] = [100, 100, 100];
pub const SPAWN_LOCATION: Point3<f32> = Point3::new(
    ((FIRST_START_POS[0] + (RENDER_SIZE[0] as i32) / 2) * CHUNK_SIZE as i32) as f32,
    ((FIRST_START_POS[1] + (RENDER_SIZE[1] as i32) / 2) * CHUNK_SIZE as i32) as f32,
    ((FIRST_START_POS[2] + (RENDER_SIZE[2] as i32) / 2) * CHUNK_SIZE as i32) as f32,
);

fn main() {
    // Create event loop.
    let mut event_loop = EventLoop::new();

    // Create app with vulkano context.
    let mut app = RenderPipeline::new(&event_loop);

    // Time & inputs...
    let mut cursor_pos = Vector2::new(0.0, 0.0);
    let mut sim_settings = SimSettings {
        max_dist: 15,
        do_compute: true,
    };

    let mut sim_data = SimData {
        max_dist: sim_settings.max_dist,
        render_size: RENDER_SIZE,
        start_pos: FIRST_START_POS,
    };

    app.rollback_data.player_join(Player {
        pos: SPAWN_LOCATION,
        ..Default::default()
    });

    let mut controls = Controls {
        forward: false,
        backward: false,
        left: false,
        right: false,
        up: false,
        down: false,
        sprint: false,
        mouse_left: false,
        mouse_right: false,
        do_chunk_load: true,
        mouse_move: [0.0, 0.0],
    };

    app.voxel_compute.load_chunks(&mut sim_data);

    let mut network_connection = NetworkConnection::new();
    let mut time = Instant::now();
    let mut chunk_time = Instant::now();

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(app.vulkano_interface.device.clone()).boxed());

    loop {
        let should_continue = handle_events(
            &mut event_loop,
            &mut app,
            &mut cursor_pos,
            &mut sim_settings,
            &mut controls,
        );
        // Event handling.
        if !should_continue {
            break;
        }

        // Compute voxels & render 60fps.
        if (Instant::now() - time).as_secs_f64() > 1.0 / 30.0 {
            previous_frame_end.as_mut().unwrap().cleanup_finished();
            let action = PlayerAction {
                aim: controls.mouse_move,
                forward: controls.forward as u8,
                backward: controls.backward as u8,
                left: controls.left as u8,
                right: controls.right as u8,
                jump: controls.up as u8,
                crouch: controls.down as u8,
                shoot: controls.mouse_right as u8,
                sprint: controls.sprint as u8,
            };
            network_connection.network_update(&action, &mut app.rollback_data);
            controls.mouse_move = [0.0, 0.0];
            time += std::time::Duration::from_secs_f64(1.0 / 30.0);
            if app.rollback_data.rollback_state.players.len() >= MIN_PLAYERS {
                compute_then_render(
                    &mut app,
                    &sim_settings,
                    &mut sim_data,
                    &mut recreate_swapchain,
                    &mut previous_frame_end,
                    action,
                );
            }
        }
        if (Instant::now() - chunk_time).as_secs_f64() > 2.0
            && controls.do_chunk_load
            && app.rollback_data.rollback_state.players.len() >= MIN_PLAYERS
        {
            let cam_player = app.rollback_data.cached_current_state.players[0].clone();
            let diff_from_mid: Point3<i32> = (cam_player.pos.map(|c| c as i32)
                - (Vector3::from(sim_data.start_pos)
                    + Vector3::from(RENDER_SIZE).map(|c| c as i32) / 2)
                    * (CHUNK_SIZE as i32))
                / (CHUNK_SIZE as i32);
            if diff_from_mid != Point3::new(0, 0, 0) {
                app.voxel_compute
                    .move_start_pos(&mut sim_data, diff_from_mid.into());
            }
            chunk_time = Instant::now();
        }
    }
}

/// Handles events and returns a `bool` indicating if we should quit.
fn handle_events(
    event_loop: &mut EventLoop<()>,
    app: &mut RenderPipeline,
    cursor_pos: &mut Vector2<f32>,
    sim_settings: &mut SimSettings,
    controls: &mut Controls,
) -> bool {
    let mut is_running = true;

    event_loop.run_return(|event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match &event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    is_running = false;
                }
                // Resize window and its images.
                WindowEvent::Resized(..) | WindowEvent::ScaleFactorChanged { .. } => {
                    //TODO
                    // let vulkano_window = app.windows.get_renderer_mut(*window_id).unwrap();
                    // vulkano_window.resize();
                }
                // Handle mouse position events.
                WindowEvent::CursorMoved { position, .. } => {
                    // turn camera
                    if controls.mouse_left {
                        let delta =
                            Vector2::new(position.x as f32, position.y as f32) - *cursor_pos;
                        controls.mouse_move[0] += delta.x;
                        controls.mouse_move[1] += delta.y;
                    }
                    *cursor_pos = Vector2::new(position.x as f32, position.y as f32)
                }
                // Handle mouse button events.
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == &MouseButton::Left {
                        controls.mouse_left = state == &ElementState::Pressed;
                    }
                    if button == &MouseButton::Right {
                        controls.mouse_right = state == &ElementState::Pressed;
                    }
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    input.virtual_keycode.map(|key| match key {
                        winit::event::VirtualKeyCode::Escape => {
                            is_running = false;
                        }
                        winit::event::VirtualKeyCode::Space => {
                            controls.up = input.state == ElementState::Pressed;
                        }
                        winit::event::VirtualKeyCode::LControl => {
                            controls.down = input.state == ElementState::Pressed;
                        }
                        winit::event::VirtualKeyCode::D => {
                            controls.right = input.state == ElementState::Pressed;
                        }
                        winit::event::VirtualKeyCode::A => {
                            controls.left = input.state == ElementState::Pressed;
                        }
                        winit::event::VirtualKeyCode::W => {
                            controls.forward = input.state == ElementState::Pressed;
                        }
                        winit::event::VirtualKeyCode::S => {
                            controls.backward = input.state == ElementState::Pressed;
                        }
                        winit::event::VirtualKeyCode::LShift => {
                            controls.sprint = input.state == ElementState::Pressed;
                        }
                        winit::event::VirtualKeyCode::Up => {
                            if input.state == ElementState::Released {
                                sim_settings.max_dist += 1;
                                println!("max_dist: {}", sim_settings.max_dist);
                            }
                        }
                        winit::event::VirtualKeyCode::Down => {
                            if input.state == ElementState::Released {
                                sim_settings.max_dist -= 1;
                                println!("max_dist: {}", sim_settings.max_dist);
                            }
                        }
                        winit::event::VirtualKeyCode::P => {
                            if input.state == ElementState::Released {
                                sim_settings.do_compute = !sim_settings.do_compute;
                                println!("do_compute: {}", sim_settings.do_compute);
                            }
                        }
                        winit::event::VirtualKeyCode::R => {
                            if input.state == ElementState::Released {
                                let cam_player =
                                    app.rollback_data.cached_current_state.players[0].clone();
                                println!("cam_pos: {:?}", cam_player.pos);
                                println!("cam_dir: {:?}", cam_player.dir);
                            }
                        }
                        winit::event::VirtualKeyCode::C => {
                            if input.state == ElementState::Released {
                                controls.do_chunk_load = !controls.do_chunk_load;
                                println!("chunk load: {}", controls.do_chunk_load);
                            }
                        }
                        _ => (),
                    });
                }
                _ => (),
            },
            Event::MainEventsCleared => *control_flow = ControlFlow::Exit,
            _ => (),
        }
    });

    is_running
}

fn compute_then_render(
    pipeline: &mut RenderPipeline,
    sim_settings: &SimSettings,
    sim_data: &mut SimData,
    recreate_swapchain: &mut bool,
    previous_frame_end: &mut Option<Box<dyn GpuFuture>>,
    action: PlayerAction,
) {
    let window = pipeline
        .vulkano_interface
        .surface
        .object()
        .unwrap()
        .downcast_ref::<Window>()
        .unwrap();
    let dimensions = window.inner_size();
    if dimensions.width == 0 || dimensions.height == 0 {
        return;
    }

    if *recreate_swapchain {
        let (new_swapchain, new_images) =
            match pipeline
                .vulkano_interface
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: dimensions.into(),
                    ..pipeline.vulkano_interface.swapchain.create_info()
                }) {
                Ok(r) => r,
                Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                Err(e) => panic!("failed to recreate swapchain: {e}"),
            };
        let new_images = new_images
            .into_iter()
            .map(|image| ImageView::new_default(image).unwrap())
            .collect::<Vec<_>>();

        pipeline.vulkano_interface.swapchain = new_swapchain;
        pipeline.vulkano_interface.images = new_images;
        *recreate_swapchain = false;
    }

    let (image_index, suboptimal, acquire_future) =
        match acquire_next_image(pipeline.vulkano_interface.swapchain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                *recreate_swapchain = true;
                return;
            }
            Err(e) => panic!("failed to acquire next image: {e}"),
        };

    if suboptimal {
        *recreate_swapchain = true;
    }

    let future = previous_frame_end.take().unwrap().join(acquire_future);
    // Start the frame.
    let mut frame = if sim_settings.do_compute {
        sim_data.max_dist = sim_settings.max_dist;
        // create a set of random chunk coordinates to compute
        let mut chunk_updates: Vec<[i32; 3]> = Vec::new();
        const UPDATES_PER_CHUNK: u32 = CHUNK_SIZE / 8;
        for _ in 0..50 {
            let x = rand::thread_rng().gen_range(0..sim_data.render_size[0] * UPDATES_PER_CHUNK)
                as i32
                + sim_data.start_pos[0] * (UPDATES_PER_CHUNK as i32);
            let y = rand::thread_rng().gen_range(0..sim_data.render_size[1] * UPDATES_PER_CHUNK)
                as i32
                + sim_data.start_pos[1] * (UPDATES_PER_CHUNK as i32);
            let z = rand::thread_rng().gen_range(0..sim_data.render_size[2] * UPDATES_PER_CHUNK)
                as i32
                + sim_data.start_pos[2] * (UPDATES_PER_CHUNK as i32);
            chunk_updates.push([x, y, z]);
        }
        pipeline.voxel_compute.queue_updates(&chunk_updates);
        // Compute.
        pipeline.rollback_data.download_projectiles(&pipeline.projectile_compute);
        pipeline.rollback_data
        .send_action(action, 0, pipeline.rollback_data.current_time);
        pipeline.rollback_data.step();
        pipeline.projectile_compute.upload(&pipeline.rollback_data.rollback_state.projectiles);
        let after_proj_compute = pipeline.projectile_compute.compute(future, &pipeline.voxel_compute, sim_data);
        let after_compute = pipeline.voxel_compute.compute(after_proj_compute, sim_data);
        pipeline.vulkano_interface.frame_system.frame(
            after_compute,
            pipeline.vulkano_interface.images[image_index as usize].clone(),
            Matrix4::identity(),
        )
    } else {
        pipeline.vulkano_interface.frame_system.frame(
            future,
            pipeline.vulkano_interface.images[image_index as usize].clone(),
            Matrix4::identity(),
        )
    };
    // Render.

    let mut after_future = None;
    while let Some(pass) = frame.next_pass() {
        match pass {
            Pass::Deferred(mut draw_pass) => {
                let cam_player = pipeline.rollback_data.cached_current_state.players[0].clone();
                // let view_matrix = Matrix4::from_translation(-cam_player.pos.to_homogeneous().truncate());
                let view_matrix = Matrix4::from(cam_player.rot)
                    * Matrix4::from_translation(cam_player.pos.to_homogeneous().truncate());
                let cb = pipeline.vulkano_interface.rasterizer_system.draw(
                    draw_pass.viewport_dimensions(),
                    view_matrix,
                    &pipeline.rollback_data.cached_current_state,
                );
                draw_pass.execute(cb);
            }
            Pass::Lighting(mut lighting) => {
                let voxels = pipeline.voxel_compute.voxels();
                lighting.raytrace(
                    Vector3::new(0.5, -0.5, -0.1),
                    [1.0, 0.0, 0.0],
                    voxels,
                    &pipeline.rollback_data,
                    sim_data,
                );
            }
            Pass::Finished(af) => {
                after_future = Some(af);
            }
        }
    }

    let future = after_future
        .unwrap()
        .then_swapchain_present(
            pipeline.vulkano_interface.queue.clone(),
            SwapchainPresentInfo::swapchain_image_index(
                pipeline.vulkano_interface.swapchain.clone(),
                image_index,
            ),
        )
        .then_signal_fence_and_flush();

    match future {
        Ok(future) => {
            match future.wait(None) {
                Ok(x) => x,
                Err(e) => println!("{e}"),
            }

            *previous_frame_end = Some(future.boxed());
        }
        Err(FlushError::OutOfDate) => {
            *recreate_swapchain = true;
            *previous_frame_end =
                Some(sync::now(pipeline.vulkano_interface.device.clone()).boxed());
        }
        Err(e) => {
            println!("failed to flush future: {e}");
            *previous_frame_end =
                Some(sync::now(pipeline.vulkano_interface.device.clone()).boxed());
        }
    }
}
