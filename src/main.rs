mod app;
mod pixels_draw;
mod render_pass;
mod sim_manager;
mod world_gen;

use crate::app::{App, RenderPipeline};
use cgmath::{InnerSpace, Vector2, Vector3};
use rand::Rng;
use std::time::Instant;
use vulkano_util::{renderer::VulkanoWindowRenderer};
use winit::{
    event::{ElementState, Event, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
};

pub const WINDOW_WIDTH: f32 = 1024.0;
pub const WINDOW_HEIGHT: f32 = 1024.0;
pub const RENDER_SIZE: [u32; 3] = [8, 8, 8];
pub const CHUNK_SIZE: u32 = 16;
pub const TOTAL_VOXEL_COUNT: usize =
    (RENDER_SIZE[0] * RENDER_SIZE[1] * RENDER_SIZE[2] * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE)
        as usize;

struct SimSettings {
    pub max_dist: u32,
    pub do_compute: bool,
}

pub struct SimData {
    max_dist: u32,
    render_size: [u32; 3],
    start_pos: [i32; 3],
}

pub struct CamData {
    pos: Vector3<f32>,
    dir: Vector3<f32>,
    up: Vector3<f32>,
    right: Vector3<f32>,
}

pub struct Controls {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    mouse_left: bool,
    mouse_right: bool,
    do_chunk_load: bool,
}

fn main() {
    // Create event loop.
    let mut event_loop = EventLoop::new();

    // Create app with vulkano context.
    let mut app = App::default();
    app.open(&event_loop);

    // Time & inputs...
    let mut time = Instant::now();
    let mut chunk_time = Instant::now();
    let mut cursor_pos = Vector2::new(0.0, 0.0);
    let mut sim_settings = SimSettings {
        max_dist: 15,
        do_compute: true,
    };

    let mut sim_data = SimData {
        max_dist: sim_settings.max_dist,
        render_size: RENDER_SIZE,
        start_pos: [100, 100, 100],
    };
    let mut cam_data = CamData {
        pos: Vector3::new(
            ((sim_data.start_pos[0] + (RENDER_SIZE[0] as i32) / 2) * CHUNK_SIZE as i32) as f32,
            ((sim_data.start_pos[1] + (RENDER_SIZE[1] as i32) / 2) * CHUNK_SIZE as i32) as f32,
            ((sim_data.start_pos[2] + (RENDER_SIZE[2] as i32) / 2) * CHUNK_SIZE as i32) as f32,
        ),
        dir: Vector3::new(0.0, 0.0, 1.0),
        up: Vector3::new(0.0, 1.0, 0.0),
        right: Vector3::new(1.0, 0.0, 0.0),
    };

    let mut controls = Controls {
        forward: false,
        backward: false,
        left: false,
        right: false,
        up: false,
        down: false,
        mouse_left: false,
        mouse_right: false,
        do_chunk_load: false,
    };

    loop {
        // Event handling.
        if !handle_events(
            &mut event_loop,
            &mut app,
            &mut cursor_pos,
            &mut cam_data,
            &mut sim_settings,
            &mut controls,
        ) {
            break;
        }

        // Compute voxels & render 60fps.
        if (Instant::now() - chunk_time).as_secs_f64() > 2.0 && controls.do_chunk_load {
            let diff_from_mid:Vector3<i32> = (cam_data.pos.map(|c| c as i32) - (Vector3::from(sim_data.start_pos) + Vector3::from(RENDER_SIZE).map(|c| c as i32)/2) * (CHUNK_SIZE as i32)) / (CHUNK_SIZE as i32);
            if diff_from_mid != Vector3::new(0, 0, 0) {
                for (window_id, _) in app.windows.iter_mut() {
                    let pipeline = app.pipelines.get_mut(window_id).unwrap();
                    pipeline.compute.move_start_pos(&mut sim_data, diff_from_mid.into());
                }
            }
            chunk_time = Instant::now();
        }
        if (Instant::now() - time).as_secs_f64() > 1.0 / 60.0 {
            if controls.up {
                cam_data.pos += 0.1 * cam_data.up;
            }
            if controls.down {
                cam_data.pos -= 0.1 * cam_data.up;
            }
            if controls.right {
                cam_data.pos += 0.1 * cam_data.right;
            }
            if controls.left {
                cam_data.pos -= 0.1 * cam_data.right;
            }
            if controls.forward {
                cam_data.pos += 0.1 * cam_data.dir;
            }
            if controls.backward {
                cam_data.pos -= 0.1 * cam_data.dir;
            }
            time = Instant::now();
            compute_then_render_per_window(&mut app, &cam_data, &sim_settings, &mut sim_data);
            // println!("{}", 1.0 / (Instant::now() - time).as_secs_f64());
        }
    }
}

/// Handles events and returns a `bool` indicating if we should quit.
fn handle_events(
    event_loop: &mut EventLoop<()>,
    app: &mut App,
    cursor_pos: &mut Vector2<f32>,
    cam_data: &mut CamData,
    sim_settings: &mut SimSettings,
    controls: &mut Controls,
) -> bool {
    let mut is_running = true;

    event_loop.run_return(|event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match &event {
            Event::WindowEvent {
                event, window_id, ..
            } => match event {
                WindowEvent::CloseRequested => {
                    if *window_id == app.windows.primary_window_id().unwrap() {
                        is_running = false;
                    } else {
                        // Destroy window by removing its renderer.
                        app.windows.remove_renderer(*window_id);
                        app.pipelines.remove(window_id);
                    }
                }
                // Resize window and its images.
                WindowEvent::Resized(..) | WindowEvent::ScaleFactorChanged { .. } => {
                    let vulkano_window = app.windows.get_renderer_mut(*window_id).unwrap();
                    vulkano_window.resize();
                }
                // Handle mouse position events.
                WindowEvent::CursorMoved { position, .. } => {
                    // turn camera
                    if controls.mouse_left {
                        let delta =
                            Vector2::new(position.x as f32, position.y as f32) - *cursor_pos;
                        cam_data.dir =
                            (cam_data.dir + cam_data.right * delta.x * 0.001).normalize();
                        cam_data.dir = (cam_data.dir + cam_data.up * -delta.y * 0.001).normalize();
                        cam_data.right = cam_data.dir.cross(cam_data.up).normalize();
                        cam_data.up = cam_data.right.cross(cam_data.dir).normalize();
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
                                println!("cam_dir: {:?}", cam_data.dir);
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

/// Compute and render per window.
fn compute_then_render_per_window(
    app: &mut App,
    cam_data: &CamData,
    sim_settings: &SimSettings,
    sim_data: &mut SimData,
) {
    for (window_id, window_renderer) in app.windows.iter_mut() {
        let pipeline = app.pipelines.get_mut(window_id).unwrap();
        compute_then_render(window_renderer, pipeline, cam_data, sim_settings, sim_data);
    }
}

fn compute_then_render(
    window_renderer: &mut VulkanoWindowRenderer,
    pipeline: &mut RenderPipeline,
    cam_data: &CamData,
    sim_settings: &SimSettings,
    sim_data: &mut SimData,
) {
    // Skip this window when minimized.
    match window_renderer.window_size() {
        [w, h] => {
            if w == 0.0 || h == 0.0 {
                return;
            }
        }
    }

    // Start the frame.
    let before_pipeline_future = match window_renderer.acquire() {
        Err(e) => {
            println!("{e}");
            return;
        }
        Ok(future) => future,
    };

    let after_render = if sim_settings.do_compute {
        sim_data.max_dist = sim_settings.max_dist;
        // create a set of random chunk coordinates to compute
        let mut chunk_updates: Vec<[i32; 3]> = Vec::new();
        const UPDATES_PER_CHUNK:u32 = CHUNK_SIZE / 8;
        for _ in 0..50 {
            let x = rand::thread_rng().gen_range(0..sim_data.render_size[0] * UPDATES_PER_CHUNK) as i32 + sim_data.start_pos[0] * (UPDATES_PER_CHUNK as i32);
            let y = rand::thread_rng().gen_range(0..sim_data.render_size[1] * UPDATES_PER_CHUNK) as i32 + sim_data.start_pos[1] * (UPDATES_PER_CHUNK as i32);
            let z = rand::thread_rng().gen_range(0..sim_data.render_size[2] * UPDATES_PER_CHUNK) as i32 + sim_data.start_pos[2] * (UPDATES_PER_CHUNK as i32);
            chunk_updates.push([x, y, z]);
        }
        pipeline.compute.queue_updates(&chunk_updates);
        // Compute.
        let after_compute = pipeline.compute.compute(before_pipeline_future, sim_data);

        // Render.
        let voxels = pipeline.compute.voxels();
        let target_image = window_renderer.swapchain_image_view();

        pipeline
            .place_over_frame
            .render(after_compute, voxels, target_image, cam_data, sim_data)
    } else {
        // Render.
        let voxels = pipeline.compute.voxels();
        let target_image = window_renderer.swapchain_image_view();

        pipeline.place_over_frame.render(
            before_pipeline_future,
            voxels,
            target_image,
            cam_data,
            sim_data,
        )
    };

    // Finish the frame. Wait for the future so resources are not in use when we render.
    window_renderer.present(after_render, true);
}
