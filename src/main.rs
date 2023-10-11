mod app;
mod card_system;
mod gui;
mod multipass_system;
mod networking;
mod projectile_sim_manager;
mod rasterizer;
mod raytracer;
mod rollback_manager;
mod settings_manager;
mod utils;
mod voxel_sim_manager;
mod world_gen;

use crate::{
    app::RenderPipeline,
    card_system::BaseCard,
    gui::{GuiElement, GuiState},
    rollback_manager::PlayerAbility,
    settings_manager::Settings, networking::NetworkPacket,
};
use cgmath::{EuclideanSpace, Matrix4, Point3, Rad, SquareMatrix, Vector2, Vector3};
use multipass_system::Pass;
use networking::NetworkConnection;
use rollback_manager::{Player, PlayerAction};
use settings_manager::Control;
use std::io::Write;
use std::{fs, panic, time::Instant};
use vulkano::{
    image::view::ImageView,
    swapchain::{
        acquire_next_image, AcquireError, SwapchainCreateInfo, SwapchainCreationError,
        SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
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
pub const RENDER_SIZE: [u32; 3] = [16, 16, 16];
pub const CHUNK_SIZE: u32 = 16;
const SUB_CHUNK_COUNT: u32 = CHUNK_SIZE / 8;

struct SimSettings {
    pub max_dist: u32,
    pub do_compute: bool,
}

pub struct SimData {
    max_dist: u32,
    render_size: [u32; 3],
    start_pos: [i32; 3],
}

struct WindowProperties {
    pub width: u32,
    pub height: u32,
    pub fullscreen: bool,
}

pub const FIRST_START_POS: [i32; 3] = [100, 105, 100];
pub const SPAWN_LOCATION: Point3<f32> = Point3::new(
    ((FIRST_START_POS[0] + (RENDER_SIZE[0] as i32) / 2) * CHUNK_SIZE as i32) as f32,
    ((FIRST_START_POS[1] + (RENDER_SIZE[1] as i32) / 2) * CHUNK_SIZE as i32) as f32,
    ((FIRST_START_POS[2] + (RENDER_SIZE[2] as i32) / 2) * CHUNK_SIZE as i32) as f32,
);

pub const PLAYER_HITBOX_OFFSET: Vector3<f32> = Vector3::new(0.0, -2.0, 0.0);
pub const PLAYER_HITBOX_SIZE: Vector3<f32> = Vector3::new(1.8, 4.8, 1.8);

pub const CHUNK_LOAD: bool = false;

fn main() {
    // Create event loop.
    let mut event_loop = EventLoop::new();

    let settings = Settings::from_string(fs::read_to_string("settings.yaml").unwrap().as_str());

    panic::set_hook(Box::new(|panic_info| {
        let settings = Settings::from_string(fs::read_to_string("settings.yaml").unwrap().as_str());
        let mut crash_log = std::fs::File::create(settings.crash_log).unwrap();
        if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            eprintln!("panic occurred: {s:?} at {}", panic_info.location().unwrap());
            write!(crash_log, "panic occurred: {s:?} at {}", panic_info.location().unwrap()).unwrap();
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            eprintln!("panic occurred: {s:?} at {}", panic_info.location().unwrap());
            write!(crash_log, "panic occurred: {s:?} at {}", panic_info.location().unwrap()).unwrap();
        } else {
            write!(crash_log, "panic occurred").unwrap();
        }
        std::process::exit(1);
    }));

    if settings.do_profiling {
        start_puffin_server();
    }

    let mut player_deck =
        BaseCard::vec_from_string(fs::read_to_string(&settings.card_file).unwrap().as_str());

    // Create app with vulkano context.
    let mut app = RenderPipeline::new(&event_loop, settings, &player_deck);

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

    assert!(player_deck.iter().all(|card| card.is_reasonable()));

    app.rollback_data.player_join(Player {
        pos: SPAWN_LOCATION,
        abilities: player_deck
            .iter()
            .map(|card| PlayerAbility {
                value: card.evaluate_value(true),
                ability: app.card_manager.register_base_card(card.clone()),
                cooldown: 0.0,
            })
            .collect(),
        ..Default::default()
    });

    let mut player_action = PlayerAction {
        forward: false,
        backward: false,
        left: false,
        right: false,
        jump: false,
        crouch: false,
        activate_ability: vec![false; app.settings.ability_controls.len()],
        aim: [0.0, 0.0],
    };

    let mut network_connection = NetworkConnection::new(&app.settings);
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
        in_game: false,
        should_exit: false,
    };

    let mut time = Instant::now();
    let mut chunk_time = Instant::now();
    loop {
        let should_continue = handle_events(
            &mut event_loop,
            &mut app,
            &mut cursor_pos,
            &mut sim_settings,
            &mut player_action,
            &mut window_props,
            &mut gui_state,
            &mut player_deck,
            &mut network_connection,
        );
        // Event handling.
        if !should_continue {
            break;
        }

        // Compute voxels & render 60fps.
        if (Instant::now() - time).as_secs_f32() > 0.0 {
            puffin::GlobalProfiler::lock().new_frame();
            previous_frame_end.as_mut().unwrap().cleanup_finished();
            if app.settings.player_count > 1 && gui_state.in_game {
                network_connection.network_update(
                    &player_action,
                    &player_deck,
                    &mut app.card_manager,
                    &mut app.rollback_data,
                );
            }
            time += std::time::Duration::from_secs_f32(app.rollback_data.delta_time);
            let skip_render = (Instant::now() - time).as_secs_f32() > 0.0;
            if skip_render {
                println!(
                    "skipping render: behind by {}s",
                    (Instant::now() - time).as_secs_f32()
                );
            }
            if app.rollback_data.rollback_state.players.len() >= app.settings.player_count as usize
            {
                compute_then_render(
                    &mut app,
                    &sim_settings,
                    &mut sim_data,
                    &mut recreate_swapchain,
                    &mut previous_frame_end,
                    &mut gui_state,
                    player_action.clone(),
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
            }
            player_action.aim = [0.0, 0.0];
        }
        if (Instant::now() - chunk_time).as_secs_f64() > 2.0
            && CHUNK_LOAD
            && app.rollback_data.rollback_state.players.len() >= app.settings.player_count as usize
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
    controls: &mut PlayerAction,
    window_props: &mut WindowProperties,
    gui_state: &mut GuiState,
    player_deck: &mut Vec<BaseCard>,
    network_connection: &mut NetworkConnection,
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
                match event {
                    WindowEvent::CloseRequested => {
                        is_running = false;
                    }
                    // Resize window and its images.
                    WindowEvent::Resized(new_size) => {
                        window_props.width = new_size.width;
                        window_props.height = new_size.height;
                    }
                    // Handle mouse position events.
                    WindowEvent::CursorMoved { position, .. } => {
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
                                // turn camera
                                let delta = app.settings.movement_controls.sensitivity
                                    * (Vector2::new(position.x as f32, position.y as f32)
                                        - Vector2::new(
                                            (window_props.width / 2) as f32,
                                            (window_props.height / 2) as f32,
                                        ));
                                controls.aim[0] += delta.x;
                                controls.aim[1] += delta.y;
                                *cursor_pos = Vector2::new(position.x as f32, position.y as f32)
                            }
                        }
                    }
                    // Handle mouse button events.
                    WindowEvent::MouseInput { state, button, .. } => {
                        macro_rules! mouse_match {
                            ($property:ident) => {
                                if let Control::Mouse(mouse_code) =
                                    app.settings.movement_controls.$property
                                {
                                    if button == &mouse_code {
                                        controls.$property = state == &ElementState::Pressed;
                                    }
                                }
                            };
                        }
                        mouse_match!(jump);
                        mouse_match!(crouch);
                        mouse_match!(right);
                        mouse_match!(left);
                        mouse_match!(forward);
                        mouse_match!(backward);
                        for (ability_idx, ability_key) in
                            app.settings.ability_controls.iter().enumerate()
                        {
                            if let Control::Mouse(mouse_code) = ability_key {
                                if button == mouse_code {
                                    controls.activate_ability[ability_idx] =
                                        state == &ElementState::Pressed;
                                }
                            }
                        }
                    }
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

                            macro_rules! key_match {
                                ($property:ident) => {
                                    if let Control::Key(key_code) =
                                        app.settings.movement_controls.$property
                                    {
                                        if key == key_code {
                                            controls.$property =
                                                input.state == ElementState::Pressed;
                                        }
                                    }
                                };
                            }
                            key_match!(jump);
                            key_match!(crouch);
                            key_match!(right);
                            key_match!(left);
                            key_match!(forward);
                            key_match!(backward);
                            for (ability_idx, ability_key) in
                                app.settings.ability_controls.iter().enumerate()
                            {
                                if let Control::Key(key_code) = ability_key {
                                    if key == *key_code {
                                        controls.activate_ability[ability_idx] =
                                            input.state == ElementState::Pressed;
                                    }
                                }
                            }
                            match key {
                                winit::event::VirtualKeyCode::Escape => {
                                    if input.state == ElementState::Released {
                                        if gui_state.menu_stack.len() > 0 && !gui_state.menu_stack.last().is_some_and(|gui| *gui == GuiElement::MainMenu) {
                                            let exited_ui = gui_state.menu_stack.pop().unwrap();
                                            match exited_ui {
                                                GuiElement::CardEditor => {
                                                    *player_deck = gui_state.gui_cards.clone();
                                                    app.rollback_data.send_deck_update(player_deck.clone(), 0, app.rollback_data.current_time);
                                                    network_connection.queue_packet(NetworkPacket::DeckUpdate(app.rollback_data.current_time, player_deck.clone()));
                                                }
                                                _ => (),
                                            }
                                        } else {
                                            gui_state.menu_stack.push(GuiElement::EscMenu);
                                        }
                                    }
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
                                            app.rollback_data.cached_current_state.players[0]
                                                .clone();
                                        println!("cam_pos: {:?}", cam_player.pos);
                                        println!("cam_dir: {:?}", cam_player.dir);
                                        println!("cam_vel: {:?}", cam_player.vel);
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
    pipeline: &mut RenderPipeline,
    sim_settings: &SimSettings,
    sim_data: &mut SimData,
    recreate_swapchain: &mut bool,
    previous_frame_end: &mut Option<Box<dyn GpuFuture>>,
    gui_state: &mut GuiState,
    action: PlayerAction,
    skip_render: bool,
) {
    puffin::profile_function!();
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
    let time_step = pipeline.rollback_data.delta_time;

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

    let view_matrix = if pipeline.rollback_data.cached_current_state.players.len() > 0 {
        let cam_player = pipeline.rollback_data.cached_current_state.players[0].clone();
        (Matrix4::from_translation(cam_player.pos.to_vec()) * Matrix4::from(cam_player.rot))
            .invert()
            .unwrap()
    } else {
        Matrix4::identity()
    };
    let aspect_ratio = dimensions.width as f32 / dimensions.height as f32;
    let proj = cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.1, 100.0);
    // Start the frame.
    let future = if sim_settings.do_compute && gui_state.in_game {
        puffin::profile_scope!("do compute");
        sim_data.max_dist = sim_settings.max_dist;
        pipeline.voxel_compute.push_updates_from_changed();

        // Compute.
        pipeline.rollback_data.download_projectiles(
            &pipeline.card_manager,
            &pipeline.projectile_compute,
            &mut pipeline.voxel_compute,
        );
        pipeline
            .rollback_data
            .send_action(action, 0, pipeline.rollback_data.current_time);
        pipeline.rollback_data.step(
            &mut pipeline.card_manager,
            time_step,
            &mut pipeline.voxel_compute,
        );
        pipeline
            .projectile_compute
            .upload(&pipeline.rollback_data.rollback_state.projectiles);
        let after_proj_compute = pipeline.projectile_compute.compute(
            future,
            &pipeline.voxel_compute,
            sim_data,
            time_step,
        );
        pipeline.voxel_compute.compute(after_proj_compute, sim_data)
    } else {
        future.boxed()
    };
    
    let future = if skip_render {
        future
    } else {
        let mut frame = pipeline.vulkano_interface.frame_system.frame(
            future,
            pipeline.vulkano_interface.images[image_index as usize].clone(),
            proj * view_matrix,
        );

        let mut after_future = None;
        while let Some(pass) = frame.next_pass() {
            match pass {
                Pass::Deferred(mut draw_pass) => {
                    puffin::profile_scope!("rasterize");
                    if gui_state.in_game {
                        let cam_player = pipeline.rollback_data.cached_current_state.players[0].clone();
                        let view_matrix = (Matrix4::from_translation(-cam_player.pos.to_vec())
                            * Matrix4::from(cam_player.rot))
                        .invert()
                        .unwrap();
                        let cb = pipeline.vulkano_interface.rasterizer_system.draw(
                            draw_pass.viewport_dimensions(),
                            view_matrix,
                            &pipeline.rollback_data.cached_current_state,
                        );
                        draw_pass.execute(cb);
                    }
                }
                Pass::Lighting(mut lighting) => {
                    let voxels = pipeline.voxel_compute.voxels();
                    if gui_state.in_game {
                        lighting.raytrace(
                            voxels,
                            &pipeline.rollback_data,
                            sim_data,
                            &pipeline.settings,
                        );
                    }
                    lighting.gui(
                        &mut pipeline.voxel_compute,
                        &pipeline.rollback_data,
                        gui_state,
                        sim_data,
                        &pipeline.settings,
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
                pipeline.vulkano_interface.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    pipeline.vulkano_interface.swapchain.clone(),
                    image_index,
                ),
            )
            .then_signal_fence_and_flush()
    };

    match future {
        Ok(future) => {
            puffin::profile_scope!("wait for gpu resources");
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
