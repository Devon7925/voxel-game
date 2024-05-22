mod app;
mod card_system;
mod game_manager;
mod gui;
mod lobby_browser;
mod multipass_system;
mod networking;
mod projectile_sim_manager;
mod rasterizer;
mod raytracer;
mod rollback_manager;
mod settings_manager;
mod utils;
mod voxel_sim_manager;

use crate::{
    app::RenderPipeline,
    card_system::Cooldown,
    gui::{GuiElement, GuiState, PaletteState},
    lobby_browser::LobbyBrowser,
    settings_manager::Settings, utils::Direction,
};
use cgmath::{EuclideanSpace, Matrix4, Point3, Rad, SquareMatrix, Vector3};
use multipass_system::Pass;
use std::io::Write;
use std::{fs, panic, time::Instant};
use vulkano::{
    image::view::ImageView,
    swapchain::{acquire_next_image, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::{self, GpuFuture},
    Validated, VulkanError,
};
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
    window::Window,
};

pub const WINDOW_WIDTH: f32 = 1024.0;
pub const WINDOW_HEIGHT: f32 = 1024.0;
pub const CHUNK_SIZE: u32 = 16;
const SUB_CHUNK_COUNT: u32 = CHUNK_SIZE / 16;
const WORLDGEN_CHUNK_COUNT: u32 = CHUNK_SIZE / 8;

const MAX_CHUNK_UPDATE_RATE: usize = 1024;
const MAX_WORLDGEN_RATE: usize = 1024;

const DEFAULT_DELTA_TIME: f32 = 1.0 / 60.0;

pub struct WindowProperties {
    pub width: u32,
    pub height: u32,
    pub fullscreen: bool,
}

pub const PLAYER_HITBOX_OFFSET: Vector3<f32> = Vector3::new(0.0, -2.0, 0.0);
pub const PLAYER_HITBOX_SIZE: Vector3<f32> = Vector3::new(1.8, 4.8, 1.8);

fn main() {
    // Create event loop.
    let mut event_loop = EventLoop::new();

    let settings = Settings::from_string(fs::read_to_string("settings.yaml").unwrap().as_str());

    panic::set_hook(Box::new(|panic_info| {
        let settings = Settings::from_string(fs::read_to_string("settings.yaml").unwrap().as_str());
        let mut crash_log = std::fs::File::create(settings.crash_log).unwrap();
        if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            eprintln!(
                "panic occurred: {s:?} at {}",
                panic_info.location().unwrap()
            );
            write!(
                crash_log,
                "panic occurred: {s:?} at {}",
                panic_info.location().unwrap()
            )
            .unwrap();
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            eprintln!(
                "panic occurred: {s:?} at {}",
                panic_info.location().unwrap()
            );
            write!(
                crash_log,
                "panic occurred: {s:?} at {}",
                panic_info.location().unwrap()
            )
            .unwrap();
        } else {
            write!(crash_log, "panic occurred").unwrap();
        }
        std::process::exit(1);
    }));

    if settings.do_profiling {
        start_puffin_server();
    }

    let player_deck =
        Cooldown::vec_from_string(fs::read_to_string(&settings.card_file).unwrap().as_str());
    assert!(player_deck.iter().all(|cooldown| cooldown.is_reasonable()));

    // Create app with vulkano context.
    let mut app = RenderPipeline::new(&event_loop, settings);

    // Time & inputs...
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(app.vulkano_interface.device.clone()).boxed());

    let mut window_props = WindowProperties {
        width: WINDOW_WIDTH as u32,
        height: WINDOW_HEIGHT as u32,
        fullscreen: false,
    };

    let mut gui_state = GuiState {
        menu_stack: vec![GuiElement::MainMenu],
        gui_cards: player_deck.clone(),
        palette_state: PaletteState::ProjectileModifiers,
        should_exit: false,
        game_just_started: false,
        lobby_browser: LobbyBrowser::new(),
    };

    let mut next_frame_time = Instant::now();
    loop {
        let should_continue = handle_events(
            &mut event_loop,
            &mut app,
            &mut window_props,
            &mut gui_state,
            &mut recreate_swapchain,
        );
        // Event handling.
        if !should_continue {
            break;
        }

        // Compute voxels & render 60fps.
        if (Instant::now() - next_frame_time).as_secs_f32() > 0.0 {
            puffin::GlobalProfiler::lock().new_frame();
            previous_frame_end.as_mut().unwrap().cleanup_finished();
            next_frame_time += std::time::Duration::from_secs_f32(
                app.game
                    .as_ref()
                    .map(|game| game.rollback_data.get_delta_time())
                    .unwrap_or(DEFAULT_DELTA_TIME),
            );
            let skip_render = if app.game.is_some() && !gui_state.game_just_started {
                (Instant::now() - next_frame_time).as_secs_f32() > 0.0
            } else {
                next_frame_time = Instant::now();
                false
            };
            gui_state.game_just_started = false;
            if let Some(game) = app.game.as_mut() {
                game.rollback_data
                    .network_update(&game.game_settings, &mut game.card_manager);
                if !game.has_started
                    && game.rollback_data.player_count() >= game.game_settings.player_count as usize
                {
                    game.has_started = true;
                    if gui_state.menu_stack.last() == Some(&GuiElement::LobbyQueue) {
                        gui_state.menu_stack.clear();
                    }
                }
                if game.rollback_data.is_render_behind_other_players() {
                    next_frame_time -=
                        std::time::Duration::from_secs_f32(game.rollback_data.get_delta_time());
                }
            }
            if skip_render {
                println!(
                    "skipping render: behind by {}s",
                    (Instant::now() - next_frame_time).as_secs_f32()
                );
            }
            compute_then_render(
                &mut app,
                &mut recreate_swapchain,
                &mut previous_frame_end,
                &mut gui_state,
                skip_render,
            );
            let window = app
                .vulkano_interface
                .surface
                .object()
                .unwrap()
                .downcast_ref::<Window>()
                .unwrap();
            window.set_cursor_visible(gui_state.menu_stack.len() > 0 && window.has_focus());
            if let Some(game) = app.game.as_mut() {
                game.rollback_data.end_frame();
            }
        }
    }
}

/// Handles events and returns a `bool` indicating if we should quit.
fn handle_events(
    event_loop: &mut EventLoop<()>,
    app: &mut RenderPipeline,
    window_props: &mut WindowProperties,
    gui_state: &mut GuiState,
    recreate_swapchain: &mut bool,
) -> bool {
    let mut is_running = true;

    event_loop.run_return(|event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match &event {
            Event::WindowEvent { event, .. } => {
                let gui_event = app.vulkano_interface.frame_system.gui.update(&event);
                if gui_state.should_exit {
                    is_running = false;
                }
                if gui_event {
                    return;
                }
                if let Some(game) = app.game.as_mut() {
                    game.rollback_data.process_event(
                        event,
                        &app.settings,
                        gui_state,
                        &window_props,
                    );
                }
                match event {
                    WindowEvent::CloseRequested => {
                        is_running = false;
                    }
                    // Resize window and its images.
                    WindowEvent::Resized(new_size) => {
                        window_props.width = new_size.width;
                        window_props.height = new_size.height;
                        *recreate_swapchain = true;
                    }
                    // Handle mouse position events.
                    WindowEvent::CursorMoved { .. } => {
                        if gui_state.menu_stack.len() == 0 {
                            let window = app
                                .vulkano_interface
                                .surface
                                .object()
                                .unwrap()
                                .downcast_ref::<Window>()
                                .unwrap();
                            if window.has_focus() {
                                window
                                    .set_cursor_position(PhysicalPosition::new(
                                        (window_props.width / 2) as f64,
                                        (window_props.height / 2) as f64,
                                    ))
                                    .unwrap_or_else(|_| println!("Failed to set cursor position"));
                            }
                        }
                    }
                    // Handle mouse button events.
                    WindowEvent::MouseInput { .. } => {}
                    WindowEvent::KeyboardInput { input, .. } => {
                        input.virtual_keycode.map(|key| {
                            let window = app
                                .vulkano_interface
                                .surface
                                .object()
                                .unwrap()
                                .downcast_ref::<Window>()
                                .unwrap();
                            if key == app.settings.fullscreen_toggle
                                && input.state == ElementState::Pressed
                            {
                                window_props.fullscreen = !window_props.fullscreen;
                                if window_props.fullscreen {
                                    window.set_fullscreen(Some(
                                        winit::window::Fullscreen::Borderless(None),
                                    ));
                                } else {
                                    window.set_fullscreen(None);
                                }
                            }
                            match key {
                                winit::event::VirtualKeyCode::Escape => {
                                    if input.state == ElementState::Released {
                                        if gui_state.menu_stack.len() > 0
                                            && !gui_state
                                                .menu_stack
                                                .last()
                                                .is_some_and(|gui| *gui == GuiElement::MainMenu)
                                        {
                                            let _exited_ui = gui_state.menu_stack.pop().unwrap();
                                        } else {
                                            gui_state.menu_stack.push(GuiElement::EscMenu);
                                        }
                                    }
                                }
                                winit::event::VirtualKeyCode::F1 => {
                                    if input.state == ElementState::Released {
                                        if let Some(game) = app.game.as_ref() {
                                            println!(
                                                "Chunk update count: {}",
                                                game.voxel_compute.update_count()
                                            );
                                            println!(
                                                "Chunk worldgen count: {}",
                                                game.voxel_compute.worldgen_count()
                                            );
                                            println!(
                                                "Chunk capacity: {}",
                                                game.voxel_compute.worldgen_capacity()
                                            );
                                        }
                                    }
                                }
                                winit::event::VirtualKeyCode::F2 => {
                                    if input.state == ElementState::Released {
                                        if let Some(game) = app.game.as_ref() {
                                            println!(
                                                "Player decks: {:?}",
                                                game.rollback_data
                                                    .get_players()
                                                    .iter()
                                                    .map(|p| p.abilities.clone())
                                                    .collect::<Vec<_>>()
                                            );
                                        }
                                    }
                                }
                                _ => (),
                            }
                        });
                    }
                    _ => (),
                }
            }
            Event::MainEventsCleared => *control_flow = ControlFlow::Exit,
            _ => (),
        }
    });

    is_running
}

fn compute_then_render(
    app: &mut RenderPipeline,
    recreate_swapchain: &mut bool,
    previous_frame_end: &mut Option<Box<dyn GpuFuture>>,
    gui_state: &mut GuiState,
    skip_render: bool,
) {
    puffin::profile_function!();
    let window = app
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
        let (new_swapchain, new_images) = app
            .vulkano_interface
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: dimensions.into(),
                ..app.vulkano_interface.swapchain.create_info()
            })
            .expect("failed to recreate swapchain");
        let new_images = new_images
            .into_iter()
            .map(|image| ImageView::new_default(image).unwrap())
            .collect::<Vec<_>>();

        app.vulkano_interface.swapchain = new_swapchain;
        app.vulkano_interface.images = new_images;
        *recreate_swapchain = false;
    }

    let (image_index, suboptimal, acquire_future) =
        match acquire_next_image(app.vulkano_interface.swapchain.clone(), None)
            .map_err(Validated::unwrap)
        {
            Ok(r) => r,
            Err(VulkanError::OutOfDate) => {
                *recreate_swapchain = true;
                return;
            }
            Err(e) => panic!("failed to acquire next image: {e}"),
        };

    if suboptimal {
        *recreate_swapchain = true;
    }

    let future = previous_frame_end.take().unwrap().join(acquire_future);

    let view_matrix = if let Some(game) = app.game.as_ref() {
        let camera = game.rollback_data.get_camera();
        (Matrix4::from_translation(camera.pos.to_vec()) * Matrix4::from(camera.rot))
            .invert()
            .unwrap()
    } else {
        Matrix4::identity()
    };
    let proj = cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), 1.0, 0.1, 100.0);
    // Start the frame.
    let future = if let Some(game) = app.game.as_mut() {
        if game.has_started {
            if game.rollback_data.can_step_rollback() {
                puffin::profile_scope!("do compute");
                game.voxel_compute
                    .push_updates_from_changed(&game.game_settings);

                // Compute.
                game.rollback_data.download_projectiles(
                    &game.card_manager,
                    &game.projectile_compute,
                    &mut game.voxel_compute,
                    &game.game_state,
                    &game.game_settings,
                );
                game.rollback_data.step_rollback(
                    &mut game.card_manager,
                    &mut game.voxel_compute,
                    &game.game_state,
                    &game.game_settings,
                );
                game.rollback_data.step_visuals(
                    &mut game.card_manager,
                    &mut game.voxel_compute,
                    &game.game_state,
                    &game.game_settings,
                );

                game.game_state.players_center = game
                    .rollback_data
                    .get_players()
                    .iter()
                    .map(|player| player.pos)
                    .fold(Point3::new(0.0, 0.0, 0.0), |acc, pos| acc + pos.to_vec())
                    / game.rollback_data.get_players().len() as f32;
                if !game.game_settings.fixed_center {
                    // consider moving start pos
                    let current_center =
                        game.game_state.start_pos + game.game_settings.render_size / 2;
                    let player_average_center = game
                        .game_state
                        .players_center
                        .map(|e| e as u32 / CHUNK_SIZE);
                    let distance = player_average_center
                        .zip(current_center, |a, b| a as i32 - b as i32)
                        .to_vec();

                    if distance != Vector3::new(0, 0, 0) {
                        // compute largest distance component
                        let mut largest_dist = 0;
                        let mut largest_component = 0;
                        for i in 0..3 {
                            if distance[i].abs() > largest_dist {
                                largest_dist = distance[i].abs();
                                largest_component = i;
                            }
                        }

                        let direction = Direction::from_component_direction(largest_component, distance[largest_component] > 0);
                            
                        game.voxel_compute.move_start_pos(
                            &mut game.game_state,
                            direction,
                            &game.game_settings,
                        );
                    }
                }

                game.projectile_compute
                    .upload(game.rollback_data.get_rollback_projectiles());
                let after_proj_compute = game.projectile_compute.compute(
                    future,
                    &game.game_state,
                    &game.game_settings,
                    &game.rollback_data,
                    &game.voxel_compute,
                );
                game.voxel_compute.compute(
                    after_proj_compute,
                    &mut game.game_state,
                    &game.game_settings,
                )
            } else {
                game.rollback_data.step_visuals(
                    &mut game.card_manager,
                    &mut game.voxel_compute,
                    &game.game_state,
                    &game.game_settings,
                );
                future.boxed()
            }
        } else {
            future.boxed()
        }
    } else {
        future.boxed()
    };

    let future = if skip_render {
        future
    } else {
        let mut frame = app.vulkano_interface.frame_system.frame(
            future,
            app.vulkano_interface.images[image_index as usize].clone(),
            proj * view_matrix,
        );

        let mut after_future = None;
        while let Some(pass) = frame.next_pass() {
            match pass {
                Pass::Deferred(mut draw_pass) => {
                    puffin::profile_scope!("rasterize");
                    if let Some(game) = app.game.as_mut() {
                        let camera = game.rollback_data.get_camera();
                        let view_matrix = (Matrix4::from_translation(-camera.pos.to_vec())
                            * Matrix4::from(camera.rot))
                        .invert()
                        .unwrap();
                        let cb = app.vulkano_interface.rasterizer_system.draw(
                            draw_pass.viewport_dimensions(),
                            view_matrix,
                            &game.rollback_data.get_current_state(),
                        );
                        draw_pass.execute(cb);
                    }
                }
                Pass::Lighting(mut lighting) => {
                    if let Some(game) = app.game.as_mut() {
                        lighting.raytrace(&game, &app.settings);
                    }
                    lighting.gui(
                        &mut app.game,
                        gui_state,
                        &app.settings,
                        &app.vulkano_interface.creation_interface,
                    );
                }
                Pass::Finished(af) => {
                    after_future = Some(af);
                }
            }
        }
        after_future.unwrap()
    };

    let future = {
        puffin::profile_scope!("fence and flush");
        future
            .then_swapchain_present(
                app.vulkano_interface.creation_interface.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    app.vulkano_interface.swapchain.clone(),
                    image_index,
                ),
            )
            .then_signal_fence_and_flush()
    };

    match future.map_err(Validated::unwrap) {
        Ok(future) => {
            puffin::profile_scope!("wait for gpu resources");
            match future.wait(None) {
                Ok(x) => x,
                Err(e) => println!("{e}"),
            }

            *previous_frame_end = Some(future.boxed());
        }
        Err(VulkanError::OutOfDate) => {
            *recreate_swapchain = true;
            *previous_frame_end = Some(sync::now(app.vulkano_interface.device.clone()).boxed());
        }
        Err(e) => {
            println!("failed to flush future: {e}");
            *previous_frame_end = Some(sync::now(app.vulkano_interface.device.clone()).boxed());
        }
    }
}

fn start_puffin_server() {
    puffin::set_scopes_on(true); // tell puffin to collect data

    match puffin_http::Server::new("0.0.0.0:8585") {
        Ok(puffin_server) => {
            std::process::Command::new("puffin_viewer")
                .arg("--url")
                .arg("127.0.0.1:8585")
                .spawn()
                .ok();

            // We can store the server if we want, but in this case we just want
            // it to keep running. Dropping it closes the server, so let's not drop it!
            #[allow(clippy::mem_forget)]
            std::mem::forget(puffin_server);
        }
        Err(err) => {
            eprintln!("Failed to start puffin server: {err}");
        }
    };
}
