mod app;
mod card_system;
mod game_manager;
mod game_modes;
mod gui;
mod lobby_browser;
mod multipass_system;
mod networking;
mod rasterizer;
mod raytracer;
mod rollback_manager;
mod settings_manager;
mod utils;
mod voxel_sim_manager;
mod cpu_simulation;

use crate::{
    app::RenderPipeline,
    gui::{GuiElement, GuiState, PaletteState},
    lobby_browser::LobbyBrowser,
    settings_manager::Settings,
    utils::Direction,
};
use card_system::Deck;
use cgmath::{EuclideanSpace, Matrix4, Point3, Rad, SquareMatrix, Vector3};
use itertools::Itertools;
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
    event::{ElementState, Event, KeyboardInput, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

pub const WINDOW_WIDTH: f32 = 1024.0;
pub const WINDOW_HEIGHT: f32 = 1024.0;
pub const RASTER_FAR_PLANE: f32 = 200.0;
pub const CHUNK_SIZE: usize = 16;
const SUB_CHUNK_COUNT: usize = CHUNK_SIZE / 16;
const WORLDGEN_CHUNK_COUNT: usize = CHUNK_SIZE / 8;

const MAX_CHUNK_UPDATE_RATE: usize = 1024;
const MAX_VOXEL_UPDATE_RATE: usize = 1024;
const MAX_WORLDGEN_RATE: usize = 1024;

const DEFAULT_DELTA_TIME: f32 = 1.0 / 60.0;
const RESPAWN_TIME: f32 = 5.0;

pub struct WindowProperties {
    pub width: u32,
    pub height: u32,
    pub fullscreen: bool,
}

pub const PLAYER_HITBOX_OFFSET: Vector3<f32> = Vector3::new(0.0, -2.0, 0.0);
pub const PLAYER_HITBOX_SIZE: Vector3<f32> = Vector3::new(1.8, 4.8, 1.8);
pub const PLAYER_DENSITY: f32 = 3.8;
pub const PLAYER_BASE_MAX_HEALTH: f32 = 100.0;

const SETTINGS_FILE: &str = "settings.yaml";

fn main() {
    // Create event loop.
    let event_loop = EventLoop::new();

    let settings = Settings::from_string(fs::read_to_string(SETTINGS_FILE).unwrap().as_str());

    panic::set_hook(Box::new(|panic_info| {
        let settings = Settings::from_string(fs::read_to_string(SETTINGS_FILE).unwrap().as_str());
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

    let player_deck: Deck =
        ron::from_str(fs::read_to_string(&settings.card_file).unwrap().as_str()).unwrap();
    assert!(player_deck.get_unreasonable_reason().is_none());

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
        errors: Vec::new(),
        gui_deck: player_deck.clone(),
        render_deck: player_deck.clone(),
        render_deck_idx: 0,
        dock_cards: vec![],
        cooldown_cache_refresh_delay: 0.0,
        palette_state: PaletteState::BaseCards,
        should_exit: false,
        game_just_started: false,
        lobby_browser: LobbyBrowser::new(),
    };

    let mut next_frame_time = Instant::now();
    let mut last_frame_time = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        let should_continue = handle_events(
            &event,
            &mut app,
            &mut window_props,
            &mut gui_state,
            &mut recreate_swapchain,
        );

        // Event handling.
        if !should_continue {
            if let Some(game) = app.game.as_mut() {
                game.rollback_data.leave_game();
            }
            *control_flow = ControlFlow::Exit;
            return;
        }
        match event {
            Event::RedrawEventsCleared => {
                // Compute voxels & render
                if (Instant::now() - next_frame_time).as_secs_f32() > 0.0 {
                    puffin::GlobalProfiler::lock().new_frame();
                    previous_frame_end.as_mut().unwrap().cleanup_finished();
                    let delta_time = app
                        .game
                        .as_ref()
                        .map(|game| game.rollback_data.get_delta_time())
                        .unwrap_or(DEFAULT_DELTA_TIME);
                    next_frame_time += std::time::Duration::from_secs_f32(delta_time);
                    gui_state.cooldown_cache_refresh_delay -= delta_time;
                    let skip_render = if app.game.is_some() && !gui_state.game_just_started {
                        if (Instant::now() - last_frame_time).as_secs_f32() > 2.0 {
                            // enforce minimum frame rate of 0.5fps
                            false
                        } else {
                            (Instant::now() - next_frame_time).as_secs_f32() > 0.0
                        }
                    } else {
                        next_frame_time = Instant::now();
                        false
                    };
                    if !skip_render {
                        last_frame_time = Instant::now();
                    }
                    gui_state.game_just_started = false;
                    if let Some(game) = app.game.as_mut() {
                        game.rollback_data.network_update(
                            &game.game_settings,
                            &mut game.card_manager,
                            &game.game_mode,
                        );
                        if !game.has_started
                            && game.rollback_data.player_count()
                                >= game.game_settings.player_count as usize
                        {
                            game.has_started = true;
                            if gui_state.menu_stack.last() == Some(&GuiElement::LobbyQueue) {
                                gui_state.menu_stack.clear();
                            }
                        }
                        if game.rollback_data.is_render_behind_other_players() {
                            next_frame_time -= std::time::Duration::from_secs_f32(
                                game.rollback_data.get_delta_time(),
                            );
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
                    window
                        .set_cursor_visible(gui_state.menu_stack.len() > 0 || !window.has_focus());
                    if let Some(game) = app.game.as_mut() {
                        game.rollback_data.end_frame();
                        if let Some(exit_reason) = game.rollback_data.get_exit_reason() {
                            gui_state.errors.push(exit_reason);
                            gui_state.menu_stack.clear();
                            gui_state.menu_stack.push(GuiElement::MainMenu);
                            app.game = None;
                        }
                    }
                }
            }
            _ => {}
        }
    });
}

/// Handles events and returns a `bool` indicating if we should quit.
fn handle_events(
    event: &Event<()>,
    app: &mut RenderPipeline,
    window_props: &mut WindowProperties,
    gui_state: &mut GuiState,
    recreate_swapchain: &mut bool,
) -> bool {
    let mut is_running = true;

    match &event {
        Event::WindowEvent { event, .. } => {
            let gui_event = app.vulkano_interface.frame_system.gui.update(&event);
            if gui_state.should_exit {
                is_running = false;
            }
            if gui_event {
                return is_running;
            }
            if let Some(game) = app.game.as_mut() {
                game.rollback_data
                    .process_event(event, &app.settings, gui_state, &window_props);
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
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state,
                            virtual_keycode: Some(key),
                            ..
                        },
                    ..
                } => {
                    let window = app
                        .vulkano_interface
                        .surface
                        .object()
                        .unwrap()
                        .downcast_ref::<Window>()
                        .unwrap();
                    if *key == app.settings.fullscreen_toggle && *state == ElementState::Pressed {
                        window_props.fullscreen = !window_props.fullscreen;
                        if window_props.fullscreen {
                            window
                                .set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                        } else {
                            window.set_fullscreen(None);
                        }
                    }
                    match key {
                        winit::event::VirtualKeyCode::Escape => {
                            if *state == ElementState::Released {
                                if gui_state.menu_stack.len() > 0
                                    && !gui_state
                                        .menu_stack
                                        .last()
                                        .is_some_and(|gui| *gui == GuiElement::MainMenu)
                                {
                                    let exited_ui = gui_state.menu_stack.pop().unwrap();
                                    match exited_ui {
                                        GuiElement::CardEditor => {
                                            let config = ron::ser::PrettyConfig::default();
                                            let export = ron::ser::to_string_pretty(
                                                &gui_state.gui_deck,
                                                config,
                                            )
                                            .unwrap();
                                            fs::write(&app.settings.card_file, export)
                                                .expect("failed to write card file");
                                        }
                                        _ => (),
                                    }
                                } else {
                                    gui_state.menu_stack.push(GuiElement::EscMenu);
                                }
                            }
                        }
                        winit::event::VirtualKeyCode::F1 => {
                            if *state == ElementState::Released {
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
                            if *state == ElementState::Released {
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
                }
                _ => (),
            }
        }
        _ => (),
    }

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
    let proj = cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), 1.0, 0.1, RASTER_FAR_PLANE);
    // Start the frame.
    let future = if let Some(game) = app.game.as_mut() {
        if game.has_started {
            if game.rollback_data.can_step_rollback() {
                puffin::profile_scope!("do compute");
                game.voxel_compute.push_updates_from_changed(
                    &game.game_settings,
                    &game
                        .rollback_data
                        .get_players()
                        .iter()
                        .map(|p| p.pos)
                        .collect_vec(),
                );
                // ensure chunks near players are loaded
                const NEARBY_CHUNK_RANGE: i32 = 2;
                for player in game.rollback_data.get_players() {
                    let player_chunk = player.pos.map(|e| e as u32 / CHUNK_SIZE as u32);
                    for x in -NEARBY_CHUNK_RANGE..=NEARBY_CHUNK_RANGE {
                        for y in -NEARBY_CHUNK_RANGE..=NEARBY_CHUNK_RANGE {
                            for z in -NEARBY_CHUNK_RANGE..=NEARBY_CHUNK_RANGE {
                                let chunk = player_chunk
                                    .zip(Point3::new(x, y, z), |a, b| a.wrapping_add_signed(b));
                                game.voxel_compute
                                    .ensure_chunk_loaded(chunk, &game.game_settings);
                            }
                        }
                    }
                }

                // Compute.
                game.rollback_data.download_projectiles(
                    &game.card_manager,
                    &mut game.voxel_compute,
                    &game.game_settings,
                );
                game.rollback_data.step_rollback(
                    &mut game.card_manager,
                    &mut game.voxel_compute,
                    &game.game_state,
                    &game.game_settings,
                    &mut game.game_mode,
                );
                game.rollback_data.step_visuals(
                    &mut game.card_manager,
                    &mut game.voxel_compute,
                    &game.game_state,
                    &game.game_settings,
                    &game.game_mode,
                    gui_state.menu_stack.is_empty(),
                );

                game.game_state.players_center = game
                    .rollback_data
                    .get_players()
                    .iter()
                    .map(|player| player.pos)
                    .fold(Point3::new(0.0, 0.0, 0.0), |acc, pos| acc + pos.to_vec())
                    / game.rollback_data.get_players().len() as f32;
                if !game.game_mode.fixed_center() {
                    // consider moving start pos
                    let current_center =
                        game.game_state.start_pos + game.game_settings.render_size / 2;
                    let player_average_center = game
                        .game_state
                        .players_center
                        .map(|e| e as u32 / CHUNK_SIZE as u32);
                    let distance = player_average_center
                        .zip(current_center, |a, b| a as i32 - b as i32)
                        .to_vec();

                    // compute largest distance component
                    let mut largest_dist = 0;
                    let mut largest_component = 0;
                    for i in 0..3 {
                        if distance[i].abs() > largest_dist {
                            largest_dist = distance[i].abs();
                            largest_component = i;
                        }
                    }

                    if largest_dist > 1 {
                        let direction = Direction::from_component_direction(
                            largest_component,
                            distance[largest_component] > 0,
                        );

                        game.voxel_compute.move_start_pos(
                            &mut game.game_state,
                            direction,
                            &game.game_settings,
                        );
                    }
                }

                game.voxel_compute.compute(
                    future,
                    &mut game.game_state,
                    &game.game_settings,
                    game.rollback_data.get_rollback_projectiles(),
                )
            } else {
                game.rollback_data.step_visuals(
                    &mut game.card_manager,
                    &mut game.voxel_compute,
                    &game.game_state,
                    &game.game_settings,
                    &game.game_mode,
                    !gui_state.menu_stack.is_empty(),
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
                        if gui_state.menu_stack.last() != Some(&GuiElement::CardEditor) {
                            lighting.raytrace(&game, &app.settings);
                        }
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
