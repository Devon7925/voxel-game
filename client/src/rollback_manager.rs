use core::panic;
use std::{
    collections::{HashMap, VecDeque},
    f32::consts::PI,
    fs::{self, File},
    io::{BufReader, Lines, Write},
    sync::Arc,
};

use bytemuck::{Pod, Zeroable};
use cgmath::{
    EuclideanSpace, InnerSpace, One, Point3, Quaternion, Rad, Rotation, Rotation3, Vector2, Vector3,
};
use matchbox_socket::{PeerId, PeerState};
use serde::{Deserialize, Serialize};
use vulkano::{
    buffer::{subbuffer::BufferReadGuard, Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};
use winit::event::{ElementState, WindowEvent};

use crate::{
    card_system::{
        BaseCard, CardManager, Cooldown, ReferencedCooldown, ReferencedEffect,
        ReferencedStatusEffect, ReferencedTrigger, StateKeybind, VoxelMaterial,
    },
    game_manager::GameState,
    gui::{GuiElement, GuiState},
    networking::{NetworkConnection, NetworkPacket},
    projectile_sim_manager::{Projectile, ProjectileComputePipeline},
    settings_manager::{Control, Settings},
    voxel_sim_manager::VoxelComputePipeline,
    WindowProperties, CHUNK_SIZE, PLAYER_HITBOX_OFFSET, PLAYER_HITBOX_SIZE,
};
use voxel_shared::{GameSettings, WorldGenSettings, RoomId};

const LOAD_LOCKOUT_TIME: u64 = 50;

#[derive(Clone, Debug)]
pub struct WorldState {
    pub players: Vec<Entity>,
    pub projectiles: Vec<Projectile>,
}

pub struct Camera {
    pub pos: Point3<f32>,
    pub rot: Quaternion<f32>,
}

pub trait PlayerSim {
    fn update_rollback_state(
        &mut self,
        card_manager: &mut CardManager,
        time_step: f32,
        vox_compute: &mut VoxelComputePipeline,
        game_state: &GameState,
        game_settings: &GameSettings,
    );

    fn get_current_state(&self) -> &WorldState;

    fn step(
        &mut self,
        card_manager: &mut CardManager,
        time_step: f32,
        vox_compute: &mut VoxelComputePipeline,
        game_state: &GameState,
        game_settings: &GameSettings,
    );

    fn download_projectiles(
        &mut self,
        card_manager: &CardManager,
        projectile_compute: &ProjectileComputePipeline,
        vox_compute: &mut VoxelComputePipeline,
        game_state: &GameState,
        game_settings: &GameSettings,
    );

    fn get_camera(&self) -> Camera;
    fn get_spectate_player(&self) -> Option<Entity>;

    fn get_delta_time(&self) -> f32;
    fn get_projectiles(&self) -> &Vec<Projectile>;
    fn get_render_projectiles(&self) -> &Vec<Projectile>;
    fn get_players(&self) -> &Vec<Entity>;
    fn player_count(&self) -> usize;

    fn visable_projectile_buffer(&self) -> Subbuffer<[Projectile; 1024]>;
    fn visable_player_buffer(&self) -> Subbuffer<[UploadPlayer; 128]>;

    fn network_update(&mut self, settings: &GameSettings, card_manager: &mut CardManager);

    fn process_event(
        &mut self,
        event: &winit::event::WindowEvent,
        settings: &Settings,
        gui_state: &mut GuiState,
        window_props: &WindowProperties,
    );
    fn end_frame(&mut self);

    fn is_sim_behind(&self) -> bool;
    fn get_rollback_time(&self) -> u64;
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Action {
    pub primary_action: Option<PlayerAction>,
    pub meta_action: Option<MetaAction>,
}

impl Action {
    fn new() -> Self {
        Action {
            primary_action: None,
            meta_action: None,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub enum EntityMetaData {
    Player(VecDeque<Action>),
    TrainingBot,
}

impl EntityMetaData {
    fn step(&mut self) {
        match self {
            EntityMetaData::Player(actions) => {
                let latest = actions.pop_front();
                assert!(latest.is_some());
                actions.push_back(Action::new());
            }
            EntityMetaData::TrainingBot => {}
        }
    }

    fn get_action(&self, rollback_offset: u64) -> Action {
        match self {
            EntityMetaData::Player(actions) => actions
                .get(rollback_offset as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "cannot access index {} in action deque of length {}",
                        rollback_offset,
                        actions.len()
                    )
                })
                .clone(),
            EntityMetaData::TrainingBot => Action::new(),
        }
    }
}

#[derive(Debug)]
pub struct RollbackData {
    pub current_time: u64,
    pub rollback_time: u64,
    pub delta_time: f32,
    pub rollback_state: WorldState,
    pub cached_current_state: WorldState,
    pub entity_metadata: Vec<EntityMetaData>,
    pub rendered_projectile_buffer: Subbuffer<[Projectile; 1024]>,
    pub rendered_player_buffer: Subbuffer<[UploadPlayer; 128]>,
    replay_file: Option<File>,
    network_connection: Option<NetworkConnection>,
    controls: Vec<Vec<StateKeybind>>,
    player_action: PlayerAction,
    player_deck: Vec<Cooldown>,
    player_idx_map: HashMap<PeerId, usize>,
    most_future_time_recorded: u64,
    connected_player_count: usize,
}

pub struct ReplayData {
    pub current_time: u64,
    pub delta_time: f32,
    pub state: WorldState,
    pub actions: VecDeque<Vec<Action>>,
    pub projectile_buffer: Subbuffer<[Projectile; 1024]>,
    pub player_buffer: Subbuffer<[UploadPlayer; 128]>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PlayerAction {
    pub aim: [f32; 2],
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub jump: bool,
    pub crouch: bool,
    pub activate_ability: Vec<Vec<bool>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MetaAction {
    pub deck_update: Option<Vec<Cooldown>>,
}

#[derive(Clone, Debug)]
pub struct Entity {
    pub pos: Point3<f32>,
    pub facing: [f32; 2],
    pub rot: Quaternion<f32>,
    pub size: f32,
    pub vel: Vector3<f32>,
    pub dir: Vector3<f32>,
    pub up: Vector3<f32>,
    pub right: Vector3<f32>,
    pub health: Vec<HealthSection>,
    pub abilities: Vec<PlayerAbility>,
    pub respawn_timer: f32,
    pub collision_vec: Vector3<i32>,
    pub status_effects: Vec<AppliedStatusEffect>,
    pub player_piercing_invincibility: f32,
}

#[derive(Clone, Debug)]
pub enum HealthSection {
    Health(f32, f32),
    Overhealth(f32, f32),
}

#[derive(Clone, Debug)]
pub struct AppliedStatusEffect {
    pub effect: ReferencedStatusEffect,
    pub time_left: f32,
}

#[derive(Clone, Debug)]
pub struct PlayerAbility {
    pub ability: ReferencedCooldown,
    pub value: (f32, Vec<f32>),
    pub cooldown: f32,
    pub recovery: f32,
}

#[derive(Clone, Copy, Zeroable, Debug, Pod)]
#[repr(C)]
pub struct UploadPlayer {
    pub pos: [f32; 4],
    pub rot: [f32; 4],
    pub size: [f32; 4],
    pub vel: [f32; 4],
    pub dir: [f32; 4],
    pub up: [f32; 4],
    pub right: [f32; 4],
}

struct PlayerEffectStats {
    speed: f32,
    damage_taken: f32,
    gravity: f32,
    invincible: bool,
    lockout: bool,
}

impl Default for PlayerAction {
    fn default() -> Self {
        PlayerAction {
            aim: [0.0, 0.0],
            forward: false,
            backward: false,
            left: false,
            right: false,
            jump: false,
            crouch: false,
            activate_ability: vec![],
        }
    }
}

impl Default for MetaAction {
    fn default() -> Self {
        MetaAction { deck_update: None }
    }
}

impl Default for Entity {
    fn default() -> Self {
        Entity {
            pos: Point3::new(0.0, 0.0, 0.0),
            facing: [0.0, 0.0],
            dir: Vector3::new(0.0, 0.0, 1.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            right: Vector3::new(1.0, 0.0, 0.0),
            rot: Quaternion::one(),
            size: 1.0,
            vel: Vector3::new(0.0, 0.0, 0.0),
            health: vec![HealthSection::Health(100.0, 100.0)],
            abilities: Vec::new(),
            respawn_timer: 0.0,
            collision_vec: Vector3::new(0, 0, 0),
            status_effects: Vec::new(),
            player_piercing_invincibility: 0.0,
        }
    }
}

pub fn abilities_from_cooldowns(
    cooldowns: &Vec<Cooldown>,
    card_manager: &mut CardManager,
) -> Vec<PlayerAbility> {
    let total_impact = cooldowns
        .iter()
        .map(|card| card.get_impact_multiplier())
        .sum();
    cooldowns
        .iter()
        .map(|cooldown| PlayerAbility {
            value: cooldown.get_cooldown_recovery(total_impact),
            ability: card_manager.register_cooldown(cooldown.clone()),
            cooldown: 0.0,
            recovery: 0.0,
        })
        .collect()
}

impl PlayerSim for RollbackData {
    fn update_rollback_state(
        &mut self,
        card_manager: &mut CardManager,
        time_step: f32,
        vox_compute: &mut VoxelComputePipeline,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) {
        if let Some(replay_file) = self.replay_file.as_mut() {
            write!(replay_file, "\nTIME {}", self.current_time).unwrap();
        }
        self.rollback_time += 1;
        let rollback_actions: Vec<Action> = self
            .entity_metadata
            .iter()
            .map(|x| x.get_action(0))
            .collect();
        {
            if let Some(replay_file) = self.replay_file.as_mut() {
                replay_file.write_all(b"\n").unwrap();
                ron::ser::to_writer(replay_file, &rollback_actions).unwrap();
            }
            for (player_idx, meta_action) in
                rollback_actions.iter().map(|a| &a.meta_action).enumerate()
            {
                if let Some(MetaAction { deck_update }) = meta_action {
                    if let Some(new_deck) = deck_update {
                        self.rollback_state.players[player_idx].abilities =
                            abilities_from_cooldowns(new_deck, card_manager)
                    }
                }
            }
        }
        if self.rollback_time < LOAD_LOCKOUT_TIME {
            return;
        }
        {
            self.rollback_state.step_sim(
                rollback_actions,
                true,
                card_manager,
                time_step,
                vox_compute,
                game_state,
                game_settings,
            );
        }
    }

    fn get_current_state(&self) -> &WorldState {
        &self.cached_current_state
    }

    fn step(
        &mut self,
        card_manager: &mut CardManager,
        time_step: f32,
        vox_compute: &mut VoxelComputePipeline,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) {
        puffin::profile_function!();
        let on_ground = self
            .get_spectate_player()
            .map(|player| player.collision_vec.y > 0)
            .unwrap_or(false);
        self.controls.iter_mut().for_each(|cd| {
            cd.iter_mut()
                .for_each(|ability| ability.update_on_ground(on_ground))
        });
        self.player_action.activate_ability = self
            .controls
            .iter()
            .map(|cd| cd.iter().map(|ability| ability.get_state()).collect())
            .collect();
        self.send_action(self.player_action.clone(), 0, self.current_time);
        self.controls
            .iter_mut()
            .for_each(|cd| cd.iter_mut().for_each(|ability| ability.clear()));
        self.update_rollback_state(
            card_manager,
            time_step,
            vox_compute,
            game_state,
            game_settings,
        );
        self.current_time += 1;
        self.entity_metadata.iter_mut().for_each(|x| x.step());
        self.cached_current_state = self.gen_current_state(
            card_manager,
            time_step,
            vox_compute,
            game_state,
            game_settings,
        );
        //send projectiles
        let projectile_count = 128.min(self.cached_current_state.projectiles.len());
        {
            let mut rendered_projectiles_buffer = self.rendered_projectile_buffer.write().unwrap();
            for i in 0..projectile_count {
                let projectile = self.cached_current_state.projectiles.get(i).unwrap();
                rendered_projectiles_buffer[i] = projectile.clone();
            }
        }
        //send players
        let player_count = 128.min(self.cached_current_state.players.len());
        {
            let mut player_buffer = self.rendered_player_buffer.write().unwrap();
            for i in 0..player_count {
                let player = self.cached_current_state.players.get(i).unwrap();
                player_buffer[i] = UploadPlayer {
                    pos: [player.pos.x, player.pos.y, player.pos.z, 0.0],
                    rot: [
                        player.rot.v[0],
                        player.rot.v[1],
                        player.rot.v[2],
                        player.rot.s,
                    ],
                    size: [player.size, player.size, player.size, 0.0],
                    vel: [player.vel.x, player.vel.y, player.vel.z, 0.0],
                    dir: [player.dir.x, player.dir.y, player.dir.z, 0.0],
                    up: [player.up.x, player.up.y, player.up.z, 0.0],
                    right: [player.right.x, player.right.y, player.right.z, 0.0],
                };
            }
        }
    }

    fn download_projectiles(
        &mut self,
        card_manager: &CardManager,
        projectile_compute: &ProjectileComputePipeline,
        vox_compute: &mut VoxelComputePipeline,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) {
        self.rollback_state.projectiles = projectile_compute.download_projectiles(
            card_manager,
            vox_compute,
            game_state,
            game_settings,
        );
    }

    fn get_camera(&self) -> Camera {
        let player = self
            .get_spectate_player()
            .unwrap_or_else(|| Entity::default());
        Camera {
            pos: player.pos,
            rot: player.rot,
        }
    }

    fn get_delta_time(&self) -> f32 {
        self.delta_time
    }

    fn get_projectiles(&self) -> &Vec<Projectile> {
        &self.rollback_state.projectiles
    }

    fn get_render_projectiles(&self) -> &Vec<Projectile> {
        &self.cached_current_state.projectiles
    }

    fn get_players(&self) -> &Vec<Entity> {
        &self.rollback_state.players
    }

    fn player_count(&self) -> usize {
        self.rollback_state.players.len()
    }

    fn network_update(&mut self, settings: &GameSettings, card_manager: &mut CardManager) {
        let Some(network_connection) = self.network_connection.as_mut() else {
            return;
        };
        let packet_data = NetworkPacket::Action(self.current_time, self.player_action.clone());
        network_connection.queue_packet(packet_data);
        let (connection_changes, recieved_packets) =
            network_connection.network_update(self.connected_player_count);
        for (peer, state) in connection_changes {
            match state {
                PeerState::Connected => {
                    println!("Peer joined: {:?}", peer);
                    network_connection
                        .send_packet(peer, NetworkPacket::Join(self.player_deck.clone()));
                    self.connected_player_count += 1;
                }
                PeerState::Disconnected => {
                    println!("Peer left: {:?}", peer);
                }
            }
        }

        let peers = self.player_idx_map.keys().collect();
        network_connection.send_packet_queue(peers);

        for (peer, packet) in recieved_packets {
            let player_idx = self.player_idx_map.get(&peer).map(|id| id.clone());
            match packet {
                NetworkPacket::Action(time, action) => {
                    self.send_action(action, player_idx.unwrap().clone(), time);
                }
                NetworkPacket::DeckUpdate(time, cards) => {
                    self.send_deck_update(cards, player_idx.unwrap().clone(), time);
                }
                NetworkPacket::Join(cards) => {
                    self.player_idx_map
                        .insert(peer, self.rollback_state.players.len());

                    let new_player = Entity {
                        pos: settings.spawn_location.into(),
                        abilities: abilities_from_cooldowns(&cards, card_manager),
                        ..Default::default()
                    };

                    self.rollback_state.players.push(new_player);
                    self.entity_metadata
                        .push(EntityMetaData::Player(VecDeque::from(vec![
                            Action::new();
                            (self.current_time - self.rollback_time + 15)
                                as usize
                        ])));
                }
            }
        }
    }

    fn visable_player_buffer(&self) -> Subbuffer<[UploadPlayer; 128]> {
        self.rendered_player_buffer.clone()
    }

    fn visable_projectile_buffer(&self) -> Subbuffer<[Projectile; 1024]> {
        self.rendered_projectile_buffer.clone()
    }

    fn get_spectate_player(&self) -> Option<Entity> {
        self.cached_current_state.players.get(0).cloned()
    }

    fn process_event(
        &mut self,
        event: &winit::event::WindowEvent,
        settings: &Settings,
        gui_state: &mut GuiState,
        window_props: &WindowProperties,
    ) {
        match event {
            // Handle mouse position events.
            WindowEvent::CursorMoved { position, .. } => {
                if gui_state.menu_stack.len() == 0 {
                    // turn camera
                    let delta = settings.movement_controls.sensitivity
                        * (Vector2::new(position.x as f32, position.y as f32)
                            - Vector2::new(
                                (window_props.width / 2) as f32,
                                (window_props.height / 2) as f32,
                            ));
                    self.player_action.aim[0] += delta.x;
                    self.player_action.aim[1] += delta.y;
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                macro_rules! mouse_match {
                    ($property:ident) => {
                        if let Control::Mouse(mouse_code) = settings.movement_controls.$property {
                            if button == &mouse_code {
                                self.player_action.$property = state == &ElementState::Pressed;
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
                for cooldown in self.controls.iter_mut() {
                    for ability in cooldown.iter_mut() {
                        ability.update(&Control::Mouse(*button), state == &ElementState::Pressed);
                    }
                }
            }
            WindowEvent::KeyboardInput { input, .. } => {
                input.virtual_keycode.map(|key| {
                    macro_rules! key_match {
                        ($property:ident) => {
                            if let Control::Key(key_code) = settings.movement_controls.$property {
                                if key == key_code {
                                    self.player_action.$property =
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
                    for cooldown in self.controls.iter_mut() {
                        for ability in cooldown.iter_mut() {
                            ability
                                .update(&Control::Key(key), input.state == ElementState::Pressed);
                        }
                    }
                    match key {
                        winit::event::VirtualKeyCode::R => {
                            if input.state == ElementState::Released {
                                // print entity positions and health
                                for (i, player) in self.rollback_state.players.iter().enumerate() {
                                    println!(
                                        "player {} pos: {:?} health: {:?}",
                                        i,
                                        player.pos,
                                        player.get_health_stats()
                                    );
                                }
                            }
                        }
                        winit::event::VirtualKeyCode::Escape => {
                            if input.state == ElementState::Released {
                                if gui_state.menu_stack.len() > 0
                                    && !gui_state
                                        .menu_stack
                                        .last()
                                        .is_some_and(|gui| *gui == GuiElement::MainMenu)
                                {
                                    let exited_ui = gui_state.menu_stack.last().unwrap();
                                    match exited_ui {
                                        GuiElement::CardEditor => {
                                            self.player_deck.clear();
                                            self.player_deck.extend(gui_state.gui_cards.clone());
                                            self.send_deck_update(
                                                self.player_deck.clone(),
                                                0,
                                                self.current_time,
                                            );
                                            if let Some(network_connection) =
                                                self.network_connection.as_mut()
                                            {
                                                network_connection.queue_packet(
                                                    NetworkPacket::DeckUpdate(
                                                        self.current_time,
                                                        self.player_deck.clone(),
                                                    ),
                                                );
                                            }
                                        }
                                        _ => (),
                                    }
                                }
                            }
                        }
                        _ => (),
                    }
                });
            }
            _ => {}
        }
    }

    fn end_frame(&mut self) {
        self.player_action.aim = [0.0, 0.0];
    }

    fn is_sim_behind(&self) -> bool {
        self.most_future_time_recorded > self.current_time
    }

    fn get_rollback_time(&self) -> u64 {
        self.rollback_time
    }
}

impl RollbackData {
    pub fn new(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        settings: &Settings,
        game_settings: &GameSettings,
        deck: &Vec<Cooldown>,
        card_manager: &mut CardManager,
        lobby_id: Option<RoomId>,
    ) -> Self {
        let rendered_projectile_buffer = Buffer::new_sized(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        )
        .unwrap();

        let player_buffer = Buffer::new_sized(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        )
        .unwrap();

        let mut replay_file_path = std::env::current_dir().unwrap();
        replay_file_path.push(settings.replay_settings.replay_folder.clone());
        fs::create_dir_all(replay_file_path.clone()).unwrap();
        replay_file_path.push(format!(
            "replay_{}.replay",
            chrono::Local::now().format("%Y-%m-%d_%H-%M-%S"),
        ));
        let mut replay_file = settings
            .replay_settings
            .record_replay
            .then(|| std::fs::File::create(replay_file_path).unwrap());

        if let Some(replay_file) = replay_file.as_mut() {
            write!(
                replay_file,
                "GAME SETTINGS {}\n",
                ron::ser::to_string(game_settings).unwrap()
            )
            .unwrap();
            write!(replay_file, "PLAYER DECK ").unwrap();
            ron::ser::to_writer(replay_file, &deck).unwrap();
        }

        let current_time: u64 = 5;
        let rollback_time: u64 = 0;

        let mut rollback_state = WorldState::new();
        let mut entity_metadata = Vec::new();

        let first_player = Entity {
            pos: game_settings.spawn_location.into(),
            abilities: abilities_from_cooldowns(deck, card_manager),
            ..Default::default()
        };
        rollback_state.players.push(first_player.clone());
        entity_metadata.push(EntityMetaData::Player(VecDeque::from(vec![
            Action::new();
            (current_time - rollback_time + 15)
                as usize
        ])));

        if matches!(game_settings.world_gen, WorldGenSettings::PracticeRange) {
            let bot = Entity {
                pos: game_settings.spawn_location.into(),
                abilities: vec![],
                ..Default::default()
            };
            rollback_state.players.push(bot.clone());
            entity_metadata.push(EntityMetaData::TrainingBot);
        }

        let network_connection =
            lobby_id.map(|lobby_id| NetworkConnection::new(settings, &game_settings, lobby_id));

        let controls: Vec<Vec<StateKeybind>> = first_player
            .abilities
            .into_iter()
            .map(|a| {
                a.ability
                    .abilities
                    .into_iter()
                    .map(|a| StateKeybind::from(a.1))
                    .collect()
            })
            .collect();

        let activate_ability = controls
            .iter()
            .map(|x| x.iter().map(|_| false).collect())
            .collect::<Vec<Vec<bool>>>();

        let player_action = PlayerAction {
            activate_ability,
            ..Default::default()
        };

        RollbackData {
            current_time,
            rollback_time,
            delta_time: game_settings.delta_time,
            rollback_state,
            cached_current_state: WorldState::new(),
            entity_metadata,
            rendered_player_buffer: player_buffer,
            rendered_projectile_buffer,
            replay_file,
            network_connection,
            controls,
            player_action,
            player_deck: deck.clone(),
            player_idx_map: HashMap::new(),
            most_future_time_recorded: 0,
            connected_player_count: 1,
        }
    }

    fn gen_current_state(
        &self,
        card_manager: &CardManager,
        time_step: f32,
        vox_compute: &mut VoxelComputePipeline,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) -> WorldState {
        let mut state = self.rollback_state.clone();
        for i in self.rollback_time..self.current_time {
            let actions = self
                .entity_metadata
                .iter()
                .map(|e| e.get_action(i - self.rollback_time))
                .collect();
            state.step_sim(
                actions,
                false,
                card_manager,
                time_step,
                vox_compute,
                game_state,
                game_settings,
            );
        }
        state
    }

    pub fn send_deck_update(
        &mut self,
        new_deck: Vec<Cooldown>,
        player_idx: usize,
        time_stamp: u64,
    ) {
        if time_stamp < self.rollback_time {
            println!(
                "cannot send dt update with timestamp {} when rollback time is {}",
                time_stamp, self.rollback_time
            );
            return;
        }
        let time_idx = time_stamp - self.rollback_time;
        let EntityMetaData::Player(actions) =
            self.entity_metadata.get_mut(player_idx).unwrap_or_else(|| {
                panic!(
                    "cannot access index {} on sending with timestamp {}",
                    player_idx, time_stamp
                )
            })
        else {
            panic!("cannot send dt update from non player");
        };
        let action = actions.get_mut(time_idx as usize).unwrap_or_else(|| {
            panic!(
                "cannot access index {} on sending with timestamp {}",
                time_idx, time_stamp
            )
        });
        if action.meta_action.is_none() {
            action.meta_action = Some(MetaAction {
                deck_update: Some(new_deck),
                ..Default::default()
            });
        } else {
            action.meta_action.as_mut().unwrap().deck_update = Some(new_deck);
        }
    }

    pub fn send_action(&mut self, entity_action: PlayerAction, player_idx: usize, time_stamp: u64) {
        if time_stamp > self.most_future_time_recorded {
            self.most_future_time_recorded = time_stamp;
        }
        if time_stamp < self.rollback_time {
            println!(
                "cannot send action with timestamp {} when rollback time is {}",
                time_stamp, self.rollback_time
            );
            return;
        }
        let time_idx = time_stamp - self.rollback_time;
        let EntityMetaData::Player(actions) =
            self.entity_metadata.get_mut(player_idx).unwrap_or_else(|| {
                panic!(
                    "cannot access index {} on sending with timestamp {}",
                    player_idx, time_stamp
                )
            })
        else {
            panic!("cannot send dt update from non player");
        };
        let action = actions.get_mut(time_idx as usize).unwrap_or_else(|| {
            panic!(
                "cannot access index {} on sending with timestamp {}",
                time_idx, time_stamp
            )
        });
        action.primary_action = Some(entity_action);
    }
}

impl PlayerSim for ReplayData {
    fn update_rollback_state(
        &mut self,
        card_manager: &mut CardManager,
        time_step: f32,
        vox_compute: &mut VoxelComputePipeline,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) {
        if self.actions.is_empty() {
            return;
        }
        self.current_time += 1;
        let rollback_actions: Vec<Action> = self.actions.pop_front().unwrap();
        {
            for (player_idx, meta_action) in
                rollback_actions.iter().map(|a| &a.meta_action).enumerate()
            {
                if let Some(MetaAction { deck_update }) = meta_action {
                    if let Some(new_deck) = deck_update {
                        self.state.players[player_idx].abilities =
                            abilities_from_cooldowns(new_deck, card_manager)
                    }
                }
            }
        }
        if self.current_time < LOAD_LOCKOUT_TIME {
            return;
        }
        {
            self.state.step_sim(
                rollback_actions,
                true,
                card_manager,
                time_step,
                vox_compute,
                game_state,
                game_settings,
            );
        }
    }

    fn get_current_state(&self) -> &WorldState {
        &self.state
    }

    fn step(
        &mut self,
        card_manager: &mut CardManager,
        time_step: f32,
        vox_compute: &mut VoxelComputePipeline,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) {
        puffin::profile_function!();
        self.update_rollback_state(
            card_manager,
            time_step,
            vox_compute,
            game_state,
            game_settings,
        );
        //send projectiles
        let projectile_count = 128.min(self.state.projectiles.len());
        {
            let mut projectiles_buffer = self.projectile_buffer.write().unwrap();
            for i in 0..projectile_count {
                let projectile = self.state.projectiles.get(i).unwrap();
                projectiles_buffer[i] = projectile.clone();
            }
        }
        //send players
        let player_count = 128.min(self.state.players.len());
        {
            let mut player_buffer = self.player_buffer.write().unwrap();
            for i in 0..player_count {
                let player = self.state.players.get(i).unwrap();
                player_buffer[i] = UploadPlayer {
                    pos: [player.pos.x, player.pos.y, player.pos.z, 0.0],
                    rot: [
                        player.rot.v[0],
                        player.rot.v[1],
                        player.rot.v[2],
                        player.rot.s,
                    ],
                    size: [player.size, player.size, player.size, 0.0],
                    vel: [player.vel.x, player.vel.y, player.vel.z, 0.0],
                    dir: [player.dir.x, player.dir.y, player.dir.z, 0.0],
                    up: [player.up.x, player.up.y, player.up.z, 0.0],
                    right: [player.right.x, player.right.y, player.right.z, 0.0],
                };
            }
        }
    }

    fn download_projectiles(
        &mut self,
        card_manager: &CardManager,
        projectile_compute: &ProjectileComputePipeline,
        vox_compute: &mut VoxelComputePipeline,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) {
        self.state.projectiles = projectile_compute.download_projectiles(
            card_manager,
            vox_compute,
            game_state,
            game_settings,
        );
    }

    fn get_camera(&self) -> Camera {
        let player = self
            .get_spectate_player()
            .unwrap_or_else(|| Entity::default());
        Camera {
            pos: player.pos,
            rot: player.rot,
        }
    }

    fn get_delta_time(&self) -> f32 {
        self.delta_time
    }

    fn get_projectiles(&self) -> &Vec<Projectile> {
        &self.state.projectiles
    }

    fn get_render_projectiles(&self) -> &Vec<Projectile> {
        &self.state.projectiles
    }

    fn get_players(&self) -> &Vec<Entity> {
        &self.state.players
    }

    fn player_count(&self) -> usize {
        self.state.players.len()
    }

    fn network_update(&mut self, _settings: &GameSettings, card_manager: &mut CardManager) {}

    fn visable_player_buffer(&self) -> Subbuffer<[UploadPlayer; 128]> {
        self.player_buffer.clone()
    }

    fn visable_projectile_buffer(&self) -> Subbuffer<[Projectile; 1024]> {
        self.projectile_buffer.clone()
    }

    fn get_spectate_player(&self) -> Option<Entity> {
        self.state.players.get(0).cloned()
    }

    fn process_event(
        &mut self,
        event: &winit::event::WindowEvent,
        _settings: &Settings,
        _gui_state: &mut GuiState,
        _window_props: &WindowProperties,
    ) {
        match event {
            _ => {}
        }
    }

    fn end_frame(&mut self) {}

    fn is_sim_behind(&self) -> bool {
        false
    }

    fn get_rollback_time(&self) -> u64 {
        self.current_time
    }
}

impl ReplayData {
    pub fn new(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        game_settings: &GameSettings,
        replay_lines: &mut Lines<BufReader<File>>,
        card_manager: &mut CardManager,
    ) -> Self {
        let projectile_buffer = Buffer::new_sized(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        )
        .unwrap();

        let player_buffer = Buffer::new_sized(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        )
        .unwrap();

        let current_time: u64 = 0;

        let mut state = WorldState::new();
        let mut actions = VecDeque::new();
        while let Some(line) = replay_lines.next() {
            let Ok(line) = line else {
                continue;
            };
            if let Some(_game_settings_string) = line.strip_prefix("GAME SETTINGS ") {
                panic!("Game settings should have already been handled")
            } else if let Some(deck_string) = line.strip_prefix("PLAYER DECK ") {
                let deck: Vec<Cooldown> = ron::de::from_str(deck_string).unwrap();

                state.players.push(Entity {
                    pos: game_settings.spawn_location.into(),
                    abilities: abilities_from_cooldowns(&deck, card_manager),
                    ..Default::default()
                });

                if matches!(game_settings.world_gen, WorldGenSettings::PracticeRange) {
                    let bot = Entity {
                        pos: game_settings.spawn_location.into(),
                        abilities: vec![],
                        ..Default::default()
                    };
                    state.players.push(bot.clone());
                }
            } else if let Some(_time_stamp_string) = line.strip_prefix("TIME ") {
                let actions_string = replay_lines.next().unwrap().unwrap();
                let line_actions: Vec<Action> = ron::de::from_str(&actions_string).unwrap();
                actions.push_back(line_actions);
            }
        }

        println!("Loaded replay with {} actions", actions.len());

        ReplayData {
            current_time,
            delta_time: game_settings.delta_time,
            state,
            actions,
            player_buffer,
            projectile_buffer,
        }
    }
}

impl WorldState {
    pub fn new() -> Self {
        WorldState {
            players: Vec::new(),
            projectiles: Vec::new(),
        }
    }

    pub fn step_sim(
        &mut self,
        player_actions: Vec<Action>,
        is_real_update: bool,
        card_manager: &CardManager,
        time_step: f32,
        vox_compute: &mut VoxelComputePipeline,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) {
        let voxels = vox_compute.voxels();
        let chunks = vox_compute.chunks();
        let mut new_projectiles = Vec::new();
        let mut voxels_to_write: Vec<(Point3<u32>, u32)> = Vec::new();
        let mut new_effects: Vec<(usize, Point3<f32>, Vector3<f32>, ReferencedEffect)> = Vec::new();
        let mut step_triggers: Vec<(ReferencedTrigger, u32)> = Vec::new();

        let player_stats: Vec<PlayerEffectStats> = self.get_player_effect_stats();

        for (player, player_stats) in self.players.iter_mut().zip(player_stats.iter()) {
            let mut health_adjustment = 0.0;
            for status_effect in player.status_effects.iter_mut() {
                match status_effect.effect {
                    ReferencedStatusEffect::DamageOverTime => {
                        health_adjustment += -10.0 * player_stats.damage_taken * time_step;
                    }
                    ReferencedStatusEffect::HealOverTime => {
                        health_adjustment += 10.0 * player_stats.damage_taken * time_step;
                    }
                    _ => {}
                }
                status_effect.time_left -= time_step;
            }
            if health_adjustment != 0.0 {
                player.adjust_health(health_adjustment);
            }
            player.status_effects.retain(|x| x.time_left > 0.0);
        }

        {
            let voxel_reader = voxels.read().unwrap();
            let chunk_reader = chunks.read().unwrap();
            for (player_idx, (player, entity_action)) in self
                .players
                .iter_mut()
                .zip(player_actions.into_iter())
                .enumerate()
            {
                player.simple_step(
                    time_step,
                    entity_action,
                    &player_stats,
                    player_idx,
                    card_manager,
                    &voxel_reader,
                    &chunk_reader,
                    &mut new_projectiles,
                    &mut voxels_to_write,
                    &mut new_effects,
                    &mut step_triggers,
                    game_state,
                    game_settings,
                );
            }
        }

        self.projectiles.iter_mut().for_each(|proj| {
            proj.simple_step(
                &self.players,
                card_manager,
                time_step,
                &mut new_projectiles,
                &mut voxels_to_write,
                &mut new_effects,
                &mut step_triggers,
            )
        });

        let collision_pairs = self.get_collision_pairs(card_manager);

        for (i, j) in collision_pairs {
            let damage_1 = self.projectiles.get(i).unwrap().damage;
            let damage_2 = self.projectiles.get(j).unwrap().damage;
            {
                let proj1_mut = self.projectiles.get_mut(j).unwrap();
                proj1_mut.health -= damage_2;
                proj1_mut.health -= damage_1;
            }
            {
                let proj2_mut = self.projectiles.get_mut(i).unwrap();
                proj2_mut.health -= damage_1;
                proj2_mut.health -= damage_2;
            }
        }

        for (player_idx, player) in self.players.iter_mut().enumerate() {
            if player.respawn_timer > 0.0 || player_stats[player_idx].invincible {
                continue;
            }

            // check piercing invincibility at start to prevent order from mattering
            let player_piercing_invincibility = player.player_piercing_invincibility > 0.0;
            // check for collision with projectiles
            for proj in self.projectiles.iter_mut() {
                if player_idx as u32 == proj.owner && proj.lifetime < 1.0 && proj.is_from_head == 1
                {
                    continue;
                }
                let proj_card = card_manager.get_referenced_proj(proj.proj_card_idx as usize);

                if proj_card.no_friendly_fire && proj.owner == player_idx as u32 {
                    continue;
                }
                if proj_card.no_enemy_fire && proj.owner != player_idx as u32 {
                    continue;
                }
                if player_piercing_invincibility && proj_card.pierce_players {
                    continue;
                }

                let projectile_rot =
                    Quaternion::new(proj.dir[3], proj.dir[0], proj.dir[1], proj.dir[2]);
                let projectile_dir = projectile_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0));
                let projectile_right = projectile_rot.rotate_vector(Vector3::new(1.0, 0.0, 0.0));
                let projectile_up = projectile_rot.rotate_vector(Vector3::new(0.0, 1.0, 0.0));
                let projectile_pos = Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]);
                let projectile_size = Vector3::new(proj.size[0], proj.size[1], proj.size[2]);

                let grid_iteration_count =
                    (2.0 * projectile_size * 2.0_f32.sqrt()).map(|c| c.ceil());
                let grid_dist =
                    2.0 * projectile_size.zip(grid_iteration_count, |size, count| size / count);

                let start_pos = projectile_pos
                    - projectile_size.x * projectile_right
                    - projectile_size.y * projectile_up
                    - projectile_size.z * projectile_dir;
                'outer: for grid_iter_x in 0..=(grid_iteration_count.x as i32) {
                    for grid_iter_y in 0..=(grid_iteration_count.y as i32) {
                        for grid_iter_z in 0..=(grid_iteration_count.z as i32) {
                            let pos = start_pos
                                + grid_dist.x * grid_iter_x as f32 * projectile_right
                                + grid_dist.y * grid_iter_y as f32 * projectile_up
                                + grid_dist.z * grid_iter_z as f32 * projectile_dir;
                            let likely_hit = Entity::HITSPHERES
                                .iter()
                                .min_by(|sphere_a, sphere_b| {
                                    (player.pos + player.size * sphere_a.offset - pos)
                                        .magnitude()
                                        .total_cmp(
                                            &(player.pos + player.size * sphere_b.offset - pos)
                                                .magnitude(),
                                        )
                                })
                                .unwrap();

                            if (player.pos + player.size * likely_hit.offset - pos).magnitude()
                                > likely_hit.radius * player.size
                            {
                                continue;
                            }

                            if !proj_card.pierce_players {
                                proj.health = 0.0;
                            } else {
                                player.player_piercing_invincibility = 0.3;
                            }
                            let mut hit_cards = card_manager
                                .get_referenced_proj(proj.proj_card_idx as usize)
                                .on_hit
                                .clone();
                            if likely_hit.headshot {
                                hit_cards.extend(
                                    card_manager
                                        .get_referenced_proj(proj.proj_card_idx as usize)
                                        .on_headshot
                                        .clone(),
                                );
                            }
                            for card_ref in hit_cards {
                                let proj_rot = proj.dir;
                                let proj_rot = Quaternion::new(
                                    proj_rot[3],
                                    proj_rot[0],
                                    proj_rot[1],
                                    proj_rot[2],
                                );
                                let (on_hit_projectiles, on_hit_voxels, effects, triggers) =
                                    card_manager.get_effects_from_base_card(
                                        card_ref,
                                        &Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]),
                                        &proj_rot,
                                        proj.owner,
                                        false,
                                    );
                                new_projectiles.extend(on_hit_projectiles);
                                for (pos, material) in on_hit_voxels {
                                    voxels_to_write.push((pos, material.to_memory()));
                                }
                                for effect in effects {
                                    new_effects.push((player_idx, pos, player.pos - pos, effect));
                                }
                                step_triggers.extend(triggers);
                            }
                            break 'outer;
                        }
                    }
                }
            }
        }

        let mut player_player_collision_pairs: Vec<(usize, usize)> = vec![];
        for i in 0..self.players.len() {
            let player1 = self.players.get(i).unwrap();
            if player1.respawn_timer > 0.0 || player_stats[i].invincible {
                continue;
            }
            for j in 0..self.players.len() {
                if i == j {
                    continue;
                }
                let player2 = self.players.get(j).unwrap();
                if player2.respawn_timer > 0.0 || player_stats[j].invincible {
                    continue;
                }
                if 5.0 * (player1.size + player2.size) > (player1.pos - player2.pos).magnitude() {
                    for si in 0..Entity::HITSPHERES.len() {
                        for sj in 0..Entity::HITSPHERES.len() {
                            let pos1 = player1.pos + player1.size * Entity::HITSPHERES[si].offset;
                            let pos2 = player2.pos + player2.size * Entity::HITSPHERES[sj].offset;
                            if (pos1 - pos2).magnitude()
                                < (Entity::HITSPHERES[si].radius + Entity::HITSPHERES[sj].radius)
                                    * (player1.size + player2.size)
                            {
                                player_player_collision_pairs.push((i, j));
                            }
                        }
                    }
                }
            }
        }
        for (i, j) in player_player_collision_pairs {
            let player1_pos = self.players.get(i).unwrap().pos;
            let player2_pos = self.players.get(j).unwrap().pos;
            let hit_effects = {
                let player1 = self.players.get_mut(i).unwrap();
                let hit_effects = player1
                    .status_effects
                    .iter()
                    .filter_map(|effect| match effect {
                        AppliedStatusEffect {
                            effect: ReferencedStatusEffect::OnHit(hit_card),
                            time_left: _,
                        } => Some(hit_card),
                        _ => None,
                    })
                    .map(|hit_effect| {
                        card_manager.get_effects_from_base_card(
                            *hit_effect,
                            &player1.pos,
                            &player1.rot,
                            i as u32,
                            false,
                        )
                    })
                    .collect::<Vec<_>>();

                player1.status_effects.retain(|effect| match effect {
                    AppliedStatusEffect {
                        effect: ReferencedStatusEffect::OnHit(_),
                        time_left: _,
                    } => false,
                    _ => true,
                });
                hit_effects
            };
            for (on_hit_projectiles, on_hit_voxels, effects, triggers) in hit_effects {
                new_projectiles.extend(on_hit_projectiles);
                for (pos, material) in on_hit_voxels {
                    voxels_to_write.push((pos, material.to_memory()));
                }
                for effect in effects {
                    new_effects.push((j, player1_pos, player2_pos - player1_pos, effect));
                }
                step_triggers.extend(triggers);
            }
        }

        for player in self.players.iter_mut() {
            if player.get_health_stats().0 <= 0.0 && player.respawn_timer <= 0.0 {
                player.respawn_timer = 5.0;
            }
        }

        for (ReferencedTrigger(trigger_id), trigger_player) in step_triggers {
            for proj in self.projectiles.iter_mut() {
                if proj.owner != trigger_player {
                    continue;
                }
                let proj_card = card_manager
                    .get_referenced_proj(proj.proj_card_idx as usize)
                    .clone();
                let projectile_rot =
                    Quaternion::new(proj.dir[3], proj.dir[0], proj.dir[1], proj.dir[2]);
                let projectile_dir = projectile_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0));
                for (proj_trigger_id, on_trigger) in proj_card.on_trigger {
                    if proj_trigger_id == trigger_id {
                        proj.health = 0.0;
                        let (proj_effects, vox_effects, effects, _) = card_manager
                            .get_effects_from_base_card(
                                on_trigger,
                                &Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]),
                                &projectile_rot,
                                proj.owner,
                                false,
                            );
                        new_projectiles.extend(proj_effects);
                        for (pos, material) in vox_effects {
                            voxels_to_write.push((pos, material.to_memory()));
                        }
                        for effect in effects {
                            new_effects.push((
                                proj.owner as usize,
                                Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]),
                                projectile_dir,
                                effect,
                            ));
                        }
                    }
                }
            }
        }

        // remove dead projectiles and add new ones
        self.projectiles.retain(|proj| proj.health > 0.0);
        self.projectiles.extend(new_projectiles);

        // update voxels
        if is_real_update && voxels_to_write.len() > 0 {
            let mut writer = voxels.write().unwrap();
            let chunk_reader = chunks.read().unwrap();
            for (pos, material) in voxels_to_write {
                vox_compute.queue_update_from_voxel_pos(&[pos.x, pos.y, pos.z], game_settings);
                let Some(index) = get_index(pos, &chunk_reader, game_state, game_settings) else {
                    panic!("voxel out of bounds");
                };
                writer[index as usize] = material;
            }
        }

        for (player_idx, effect_pos, effect_direction, effect) in new_effects {
            let player = self.players.get_mut(player_idx).unwrap();
            match effect {
                ReferencedEffect::Damage(damage) => {
                    player.adjust_health(-player_stats[player_idx].damage_taken * damage as f32);
                }
                ReferencedEffect::Knockback(knockback) => {
                    let knockback = 10.0 * knockback as f32;
                    let knockback_dir = effect_direction;
                    if knockback_dir.magnitude() > 0.0 {
                        player.vel += knockback * (knockback_dir).normalize();
                    } else {
                        player.vel.y += knockback;
                    }
                }
                ReferencedEffect::StatusEffect(effect, duration) => {
                    match effect {
                        ReferencedStatusEffect::Overheal => {
                            player.health.push(HealthSection::Overhealth(
                                10.0,
                                BaseCard::EFFECT_LENGTH_SCALE * duration as f32,
                            ));
                        }
                        _ => {}
                    }
                    player.status_effects.push(AppliedStatusEffect {
                        effect,
                        time_left: BaseCard::EFFECT_LENGTH_SCALE * duration as f32,
                    })
                }
                ReferencedEffect::Cleanse => {
                    player.status_effects.clear();
                }
                ReferencedEffect::Teleport => {
                    player.pos = effect_pos;
                    player.pos.y +=
                        player.size * (PLAYER_HITBOX_SIZE[1] / 2.0 - PLAYER_HITBOX_OFFSET[1]);
                    player.vel = Vector3::new(0.0, 0.0, 0.0);
                }
            }
        }
    }

    fn get_player_effect_stats(&mut self) -> Vec<PlayerEffectStats> {
        self.players
            .iter()
            .map(|player| {
                let mut speed = 1.0;
                let mut damage_taken = 1.0;
                let mut gravity = 1.0;
                let mut invincible = false;
                let mut lockout = false;

                for status_effect in player.status_effects.iter() {
                    match status_effect.effect {
                        ReferencedStatusEffect::DamageOverTime => {
                            // wait for damage taken to be calculated
                        }
                        ReferencedStatusEffect::HealOverTime => {
                            // wait for damage taken to be calculated
                        }
                        ReferencedStatusEffect::Speed => {
                            speed *= 1.25;
                        }
                        ReferencedStatusEffect::Slow => {
                            speed *= 0.75;
                        }
                        ReferencedStatusEffect::IncreaceDamageTaken => {
                            damage_taken *= 1.25;
                        }
                        ReferencedStatusEffect::DecreaceDamageTaken => {
                            damage_taken *= 0.75;
                        }
                        ReferencedStatusEffect::IncreaceGravity => {
                            gravity += 0.5;
                        }
                        ReferencedStatusEffect::DecreaceGravity => {
                            gravity -= 0.5;
                        }
                        ReferencedStatusEffect::Overheal => {
                            // managed seperately
                        }
                        ReferencedStatusEffect::Invincibility => {
                            invincible = true;
                        }
                        ReferencedStatusEffect::Trapped => {
                            speed *= 0.0;
                        }
                        ReferencedStatusEffect::Lockout => {
                            lockout = true;
                        }
                        ReferencedStatusEffect::OnHit(_) => {
                            // managed seperately
                        }
                    }
                }
                PlayerEffectStats {
                    speed,
                    damage_taken,
                    gravity,
                    invincible,
                    lockout,
                }
            })
            .collect()
    }

    fn get_collision_pairs(&self, card_manager: &CardManager) -> Vec<(usize, usize)> {
        let mut collision_pairs = Vec::new();
        for i in 0..self.projectiles.len() {
            let proj1 = self.projectiles.get(i).unwrap();
            let proj1_card = card_manager.get_referenced_proj(proj1.proj_card_idx as usize);
            let projectile_1_rot =
                Quaternion::new(proj1.dir[3], proj1.dir[0], proj1.dir[1], proj1.dir[2]);
            let projectile_1_vectors = [
                projectile_1_rot.rotate_vector(Vector3::new(1.0, 0.0, 0.0)),
                projectile_1_rot.rotate_vector(Vector3::new(0.0, 1.0, 0.0)),
                projectile_1_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0)),
            ];
            let projectile_1_pos = Point3::new(proj1.pos[0], proj1.pos[1], proj1.pos[2]);
            let projectile_1_size = Vector3::new(proj1.size[0], proj1.size[1], proj1.size[2]);
            let projectile_1_coords = vec![
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
            ]
            .iter()
            .map(|c| {
                let mut pos = projectile_1_pos;
                for i in 0..3 {
                    pos += projectile_1_size[i] * projectile_1_vectors[i] * c[i];
                }
                pos
            })
            .collect::<Vec<_>>();
            'second_proj_loop: for j in i + 1..self.projectiles.len() {
                let proj2 = self.projectiles.get(j).unwrap();
                let proj2_card = card_manager.get_referenced_proj(proj2.proj_card_idx as usize);

                if (proj1_card.no_friendly_fire || proj2_card.no_friendly_fire)
                    && proj1.owner == proj2.owner
                {
                    continue;
                }
                if (proj1_card.no_enemy_fire || proj2_card.no_enemy_fire)
                    && proj1.owner != proj2.owner
                {
                    continue;
                }

                let projectile_2_rot =
                    Quaternion::new(proj2.dir[3], proj2.dir[0], proj2.dir[1], proj2.dir[2]);
                let projectile_2_vectors = [
                    projectile_2_rot.rotate_vector(Vector3::new(1.0, 0.0, 0.0)),
                    projectile_2_rot.rotate_vector(Vector3::new(0.0, 1.0, 0.0)),
                    projectile_2_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0)),
                ];
                let projectile_2_pos = Point3::new(proj2.pos[0], proj2.pos[1], proj2.pos[2]);
                let projectile_2_size = Vector3::new(proj2.size[0], proj2.size[1], proj2.size[2]);

                let projectile_2_coords = vec![
                    [-1.0, -1.0, -1.0],
                    [-1.0, -1.0, 1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, 1.0, 1.0],
                    [1.0, -1.0, -1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, -1.0],
                    [1.0, 1.0, 1.0],
                ]
                .iter()
                .map(|c| {
                    let mut pos = projectile_2_pos;
                    for i in 0..3 {
                        pos += projectile_2_size[i] * projectile_2_vectors[i] * c[i];
                    }
                    pos
                })
                .collect::<Vec<_>>();

                // sat collision detection
                for i in 0..3 {
                    let (min_proj_2, max_proj_2) = projectile_2_coords
                        .iter()
                        .map(|c| c.to_vec().dot(projectile_1_vectors[i]))
                        .fold((f32::INFINITY, f32::NEG_INFINITY), |acc, x| {
                            (acc.0.min(x), acc.1.max(x))
                        });
                    if min_proj_2
                        > projectile_1_pos.to_vec().dot(projectile_1_vectors[i])
                            + projectile_1_size[i]
                        || max_proj_2
                            < projectile_1_pos.to_vec().dot(projectile_1_vectors[i])
                                - projectile_1_size[i]
                    {
                        continue 'second_proj_loop;
                    }
                }
                for i in 0..3 {
                    let (min_proj_1, max_proj_1) = projectile_1_coords
                        .iter()
                        .map(|c| c.to_vec().dot(projectile_2_vectors[i]))
                        .fold((f32::INFINITY, f32::NEG_INFINITY), |acc, x| {
                            (acc.0.min(x), acc.1.max(x))
                        });
                    if min_proj_1
                        > projectile_2_pos.to_vec().dot(projectile_2_vectors[i])
                            + projectile_2_size[i]
                        || max_proj_1
                            < projectile_2_pos.to_vec().dot(projectile_2_vectors[i])
                                - projectile_2_size[i]
                    {
                        continue 'second_proj_loop;
                    }
                }
                // collision detected
                collision_pairs.push((i, j));
            }
        }
        collision_pairs
    }
}

impl Projectile {
    fn simple_step(
        &mut self,
        players: &Vec<Entity>,
        card_manager: &CardManager,
        time_step: f32,
        new_projectiles: &mut Vec<Projectile>,
        voxels_to_write: &mut Vec<(Point3<u32>, u32)>,
        new_effects: &mut Vec<(usize, Point3<f32>, Vector3<f32>, ReferencedEffect)>,
        step_triggers: &mut Vec<(ReferencedTrigger, u32)>,
    ) {
        let proj_card = card_manager.get_referenced_proj(self.proj_card_idx as usize);
        let projectile_rot = Quaternion::new(self.dir[3], self.dir[0], self.dir[1], self.dir[2]);
        let projectile_dir = projectile_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0));
        let mut proj_vel = projectile_dir * self.vel;

        let new_projectile_rot: Quaternion<f32> = if proj_card.lock_owner {
            let player_dir = players[self.owner as usize].dir;
            let player_up = players[self.owner as usize].up;
            let proj_pos = players[self.owner as usize].pos + 0.1 * proj_card.speed * player_dir
                - 0.25 * proj_card.gravity * player_up;
            for i in 0..3 {
                self.pos[i] = proj_pos[i];
            }
            players[self.owner as usize].rot
        } else {
            proj_vel.y -= proj_card.gravity * time_step;
            for i in 0..3 {
                self.pos[i] += proj_vel[i] * time_step;
            }
            // recompute vel and rot
            if proj_vel.magnitude() < 0.0001 {
                projectile_rot
            } else {
                Quaternion::from_arc(projectile_dir, proj_vel.normalize(), None) * projectile_rot
            }
        };
        self.dir = [
            new_projectile_rot.v[0],
            new_projectile_rot.v[1],
            new_projectile_rot.v[2],
            new_projectile_rot.s,
        ];
        self.vel = proj_vel.magnitude();

        self.lifetime += time_step;
        if self.lifetime >= proj_card.lifetime {
            self.health = 0.0;
            for card_ref in card_manager
                .get_referenced_proj(self.proj_card_idx as usize)
                .on_expiry
                .clone()
            {
                let (proj_effects, vox_effects, effects, triggers) = card_manager
                    .get_effects_from_base_card(
                        card_ref,
                        &Point3::new(self.pos[0], self.pos[1], self.pos[2]),
                        &new_projectile_rot,
                        self.owner,
                        false,
                    );
                new_projectiles.extend(proj_effects);
                for (pos, material) in vox_effects {
                    voxels_to_write.push((pos, material.to_memory()));
                }
                for effect in effects {
                    new_effects.push((
                        self.owner as usize,
                        Point3::new(self.pos[0], self.pos[1], self.pos[2]),
                        projectile_dir,
                        effect,
                    ));
                }
                step_triggers.extend(triggers);
            }
        }

        for (trail_time, trail_card) in proj_card.trail.iter() {
            if self.lifetime % trail_time >= trail_time - time_step {
                let (proj_effects, vox_effects, effects, triggers) = card_manager
                    .get_effects_from_base_card(
                        trail_card.clone(),
                        &Point3::new(self.pos[0], self.pos[1], self.pos[2]),
                        &new_projectile_rot,
                        self.owner,
                        false,
                    );
                new_projectiles.extend(proj_effects);
                for (pos, material) in vox_effects {
                    voxels_to_write.push((pos, material.to_memory()));
                }
                for effect in effects {
                    new_effects.push((
                        self.owner as usize,
                        Point3::new(self.pos[0], self.pos[1], self.pos[2]),
                        projectile_dir,
                        effect,
                    ));
                }
                step_triggers.extend(triggers);
            }
        }
    }
}

pub fn get_index(
    global_pos: Point3<u32>,
    chunk_reader: &BufferReadGuard<'_, [u32]>,
    game_state: &GameState,
    game_settings: &GameSettings,
) -> Option<u32> {
    if (global_pos / CHUNK_SIZE).zip(game_state.start_pos, |a, b| a >= b)
        != Point3::new(true, true, true)
    {
        return None;
    }
    if (global_pos / CHUNK_SIZE).zip(game_state.start_pos + game_settings.render_size, |a, b| {
        a < b
    }) != Point3::new(true, true, true)
    {
        return None;
    }
    let chunk_pos =
        (global_pos / CHUNK_SIZE).zip(Point3::from_vec(game_settings.render_size), |a, b| a % b);
    let pos_in_chunk = global_pos % CHUNK_SIZE;
    let chunk_ref = chunk_pos.x * game_settings.render_size[1] * game_settings.render_size[2]
        + chunk_pos.y * game_settings.render_size[2]
        + chunk_pos.z;
    let chunk_idx = chunk_reader[chunk_ref as usize];
    let idx_in_chunk =
        pos_in_chunk.x * CHUNK_SIZE * CHUNK_SIZE + pos_in_chunk.y * CHUNK_SIZE + pos_in_chunk.z;
    Some(chunk_idx * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE + idx_in_chunk)
}

struct Hitsphere {
    offset: Vector3<f32>,
    radius: f32,
    headshot: bool,
}

impl Entity {
    const HITSPHERES: [Hitsphere; 6] = [
        Hitsphere {
            offset: Vector3::new(0.0, 0.0, 0.0),
            radius: 0.6,
            headshot: true,
        },
        Hitsphere {
            offset: Vector3::new(0.0, -1.3, 0.0),
            radius: 0.6,
            headshot: false,
        },
        Hitsphere {
            offset: Vector3::new(0.0, -1.9, 0.0),
            radius: 0.9,
            headshot: false,
        },
        Hitsphere {
            offset: Vector3::new(0.0, -2.6, 0.0),
            radius: 0.8,
            headshot: false,
        },
        Hitsphere {
            offset: Vector3::new(0.0, -3.3, 0.0),
            radius: 0.6,
            headshot: false,
        },
        Hitsphere {
            offset: Vector3::new(0.0, -3.8, 0.0),
            radius: 0.6,
            headshot: false,
        },
    ];
    fn simple_step(
        &mut self,
        time_step: f32,
        action: Action,
        player_stats: &Vec<PlayerEffectStats>,
        player_idx: usize,
        card_manager: &CardManager,
        voxel_reader: &BufferReadGuard<'_, [u32]>,
        chunk_reader: &BufferReadGuard<'_, [u32]>,
        new_projectiles: &mut Vec<Projectile>,
        voxels_to_write: &mut Vec<(Point3<u32>, u32)>,
        new_effects: &mut Vec<(usize, Point3<f32>, Vector3<f32>, ReferencedEffect)>,
        step_triggers: &mut Vec<(ReferencedTrigger, u32)>,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) {
        if self.respawn_timer > 0.0 {
            self.respawn_timer -= time_step;
            if self.respawn_timer <= 0.0 {
                self.pos = Point3::from(game_settings.spawn_location);
                self.vel = Vector3::new(0.0, 0.0, 0.0);
                self.health = vec![HealthSection::Health(100.0, 100.0)];
                self.status_effects.clear();
            }
            return;
        }
        if self.player_piercing_invincibility > 0.0 {
            self.player_piercing_invincibility -= time_step;
        }
        if let Some(action) = action.primary_action {
            self.facing[0] = (self.facing[0] - action.aim[0] + 2.0 * PI) % (2.0 * PI);
            self.facing[1] = (self.facing[1] - action.aim[1])
                .min(PI / 2.0)
                .max(-PI / 2.0);
            self.rot =
                Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), Rad(self.facing[0]))
                    * Quaternion::from_axis_angle(
                        Vector3::new(1.0, 0.0, 0.0),
                        Rad(-self.facing[1]),
                    );
            let horizontal_rot =
                Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), Rad(self.facing[0]));
            self.dir = self.rot * Vector3::new(0.0, 0.0, 1.0);
            self.right = self.rot * Vector3::new(-1.0, 0.0, 0.0);
            self.up = self.right.cross(self.dir).normalize();
            let mut move_vec = Vector3::new(0.0, 0.0, 0.0);
            let player_forward = horizontal_rot * Vector3::new(0.0, 0.0, 1.0);
            let player_right = horizontal_rot * Vector3::new(-1.0, 0.0, 0.0);
            if action.forward {
                move_vec += player_forward;
            }
            if action.backward {
                move_vec -= player_forward;
            }
            if action.left {
                move_vec -= player_right;
            }
            if action.right {
                move_vec += player_right;
            }
            if action.jump {
                move_vec += Vector3::new(0.0, 0.5, 0.0);
            }
            if action.crouch {
                move_vec -= Vector3::new(0.0, 0.5, 0.0);
            }
            if move_vec.magnitude() > 0.0 {
                move_vec = move_vec.normalize();
            }
            let accel_speed = player_stats[player_idx].speed
                * if self.collision_vec != Vector3::new(0, 0, 0) {
                    80.0
                } else {
                    18.0
                };
            self.vel += accel_speed * move_vec * time_step;

            if action.jump {
                self.vel += self
                    .collision_vec
                    .zip(Vector3::new(0.3, 13.0, 0.3), |c, m| c as f32 * m);
            }

            for (cooldown_idx, cooldown) in self.abilities.iter_mut().enumerate() {
                if cooldown.cooldown <= cooldown.value.0 * cooldown.ability.add_charge as f32
                    && cooldown.recovery <= 0.0
                {
                    for (ability_idx, ability) in cooldown.ability.abilities.iter().enumerate() {
                        if action.activate_ability[cooldown_idx][ability_idx]
                            && !player_stats[player_idx].lockout
                        {
                            cooldown.cooldown += cooldown.value.0;
                            cooldown.recovery = cooldown.value.1[ability_idx];
                            let (proj_effects, vox_effects, effects, triggers) = card_manager
                                .get_effects_from_base_card(
                                    ability.0,
                                    &self.pos,
                                    &self.rot,
                                    player_idx as u32,
                                    true,
                                );
                            new_projectiles.extend(proj_effects);
                            for (pos, material) in vox_effects {
                                voxels_to_write.push((pos, material.to_memory()));
                            }
                            for effect in effects {
                                new_effects.push((player_idx, self.pos, self.dir, effect));
                            }
                            step_triggers.extend(triggers);
                            break;
                        }
                    }
                }
            }
        }
        for ability in self.abilities.iter_mut() {
            if ability.cooldown > 0.0 {
                ability.cooldown -= time_step;
            }
            if ability.recovery > 0.0 {
                ability.recovery -= time_step;
            }
        }
        self.vel.y -= player_stats[player_idx].gravity * 32.0 * time_step;
        if self.vel.magnitude() > 0.0 {
            self.vel -= 0.1 * self.vel * self.vel.magnitude() * time_step
                + 0.2 * self.vel.normalize() * time_step;
        }
        let prev_collision_vec = self.collision_vec.clone();
        self.collision_vec = Vector3::new(0, 0, 0);
        self.collide_player(
            time_step,
            voxel_reader,
            chunk_reader,
            prev_collision_vec,
            game_state,
            game_settings,
        );

        for health_section in self.health.iter_mut() {
            match health_section {
                HealthSection::Overhealth(_health, duration) => {
                    *duration -= time_step;
                }
                _ => {}
            }
        }
        self.health.retain(|health_section| match health_section {
            HealthSection::Health(_health, _max_health) => true,
            HealthSection::Overhealth(health, duration) => *health > 0.0 && *duration > 0.0,
        });
    }

    fn collide_player(
        &mut self,
        time_step: f32,
        voxel_reader: &BufferReadGuard<'_, [u32]>,
        chunk_reader: &BufferReadGuard<'_, [u32]>,
        prev_collision_vec: Vector3<i32>,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) {
        let mut player_move_pos = self.pos
            + PLAYER_HITBOX_OFFSET
            + self
                .vel
                .map(|c| c.signum())
                .zip(PLAYER_HITBOX_SIZE, |a, b| a * b)
                * 0.5
                * self.size;
        let mut distance_to_move = self.vel * time_step;
        let mut iteration_counter = 0;

        while distance_to_move.magnitude() > 0.0 {
            iteration_counter += 1;

            let vel_dir = distance_to_move.normalize();
            let delta = ray_box_dist(player_move_pos, vel_dir);
            let mut dist_diff = delta.x.min(delta.y).min(delta.z);
            if dist_diff == 0.0 {
                dist_diff = distance_to_move.magnitude();
                if delta.x != 0.0 {
                    dist_diff = dist_diff.min(delta.x);
                }
                if delta.y != 0.0 {
                    dist_diff = dist_diff.min(delta.y);
                }
                if delta.z != 0.0 {
                    dist_diff = dist_diff.min(delta.z);
                }
            }

            if iteration_counter > 100 {
                println!(
                    "iteration counter exceeded with dtm {:?} and delta {:?}",
                    distance_to_move, delta
                );
                break;
            }

            if dist_diff > distance_to_move.magnitude() {
                self.pos += distance_to_move;
                player_move_pos += distance_to_move;
                break;
            }

            distance_to_move -= dist_diff * vel_dir;
            self.pos += dist_diff * vel_dir;
            player_move_pos += dist_diff * vel_dir;
            for component in 0..3 {
                if delta[component] <= delta[(component + 1) % 3]
                    && delta[component] <= delta[(component + 2) % 3]
                {
                    // neccessary because otherwise side plane could hit on ground to prevent walking
                    // however this allows clipping when corners would collide
                    const HITBOX_SHRINK_FACTOR: f32 = 0.999;
                    let x_iter_count = (HITBOX_SHRINK_FACTOR
                        * self.size
                        * PLAYER_HITBOX_SIZE[(component + 1) % 3])
                        .ceil()
                        + 1.0;
                    let z_iter_count = (HITBOX_SHRINK_FACTOR
                        * self.size
                        * PLAYER_HITBOX_SIZE[(component + 2) % 3])
                        .ceil()
                        + 1.0;
                    let x_dist = (HITBOX_SHRINK_FACTOR
                        * self.size
                        * PLAYER_HITBOX_SIZE[(component + 1) % 3])
                        / x_iter_count;
                    let z_dist = (HITBOX_SHRINK_FACTOR
                        * self.size
                        * PLAYER_HITBOX_SIZE[(component + 2) % 3])
                        / z_iter_count;
                    let mut start_pos = self.pos + PLAYER_HITBOX_OFFSET
                        - HITBOX_SHRINK_FACTOR * 0.5 * self.size * PLAYER_HITBOX_SIZE;
                    start_pos[component] = player_move_pos[component];

                    let mut x_vec = Vector3::new(0.0, 0.0, 0.0);
                    let mut z_vec = Vector3::new(0.0, 0.0, 0.0);
                    x_vec[(component + 1) % 3] = 1.0;
                    z_vec[(component + 2) % 3] = 1.0;
                    'outer: for x_iter in 0..=(x_iter_count as u32) {
                        for z_iter in 0..=(z_iter_count as u32) {
                            let pos = start_pos
                                + x_dist * x_iter as f32 * x_vec
                                + z_dist * z_iter as f32 * z_vec;
                            let voxel_pos = pos.map(|c| c.floor() as u32);
                            let voxel = if let Some(index) =
                                get_index(voxel_pos, chunk_reader, game_state, game_settings)
                            {
                                voxel_reader[index as usize]
                            } else {
                                VoxelMaterial::Unloaded.to_memory()
                            };
                            if voxel >> 24 != 0 {
                                if component != 1
                                    && prev_collision_vec[1] == 1
                                    && (pos - start_pos).y < 1.0
                                    && self.can_step_up(
                                        voxel_reader,
                                        chunk_reader,
                                        component,
                                        player_move_pos,
                                        game_state,
                                        game_settings,
                                    )
                                {
                                    self.pos[1] += 1.0;
                                    player_move_pos[1] += 1.0;
                                    break 'outer;
                                }

                                self.pos[component] -= dist_diff * vel_dir[component];
                                self.vel[component] = 0.0;
                                // apply friction
                                let perp_vel = Vector2::new(
                                    self.vel[(component + 1) % 3],
                                    self.vel[(component + 2) % 3],
                                );
                                if perp_vel.magnitude() > 0.0 {
                                    let friction_factor = VoxelMaterial::FRICTION_COEFFICIENTS
                                        [(voxel >> 24) as usize];
                                    self.vel[(component + 1) % 3] -=
                                        (friction_factor * 0.5 * perp_vel.normalize().x
                                            + friction_factor * perp_vel.x)
                                            * time_step;
                                    self.vel[(component + 2) % 3] -=
                                        (friction_factor * 0.5 * perp_vel.normalize().y
                                            + friction_factor * perp_vel.y)
                                            * time_step;
                                }

                                self.collision_vec[component] = -vel_dir[component].signum() as i32;
                                distance_to_move[component] = 0.0;
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
    }

    fn can_step_up(
        &self,
        voxel_reader: &BufferReadGuard<'_, [u32]>,
        chunk_reader: &BufferReadGuard<'_, [u32]>,
        component: usize,
        player_move_pos: Point3<f32>,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) -> bool {
        let x_iter_count =
            (0.99 * self.size * PLAYER_HITBOX_SIZE[(component + 1) % 3]).ceil() + 1.0;
        let z_iter_count =
            (0.99 * self.size * PLAYER_HITBOX_SIZE[(component + 2) % 3]).ceil() + 1.0;
        let x_dist = (0.99 * self.size * PLAYER_HITBOX_SIZE[(component + 1) % 3]) / x_iter_count;
        let z_dist = (0.99 * self.size * PLAYER_HITBOX_SIZE[(component + 2) % 3]) / z_iter_count;
        let mut start_pos = self.pos + PLAYER_HITBOX_OFFSET
            - 0.99 * 0.5 * self.size * PLAYER_HITBOX_SIZE
            + Vector3::new(0.0, 1.0, 0.0);
        start_pos[component] = player_move_pos[component];

        let mut x_vec = Vector3::new(0.0, 0.0, 0.0);
        let mut z_vec = Vector3::new(0.0, 0.0, 0.0);
        x_vec[(component + 1) % 3] = 1.0;
        z_vec[(component + 2) % 3] = 1.0;
        for x_iter in 0..=(x_iter_count as u32) {
            for z_iter in 0..=(z_iter_count as u32) {
                let pos =
                    start_pos + x_dist * x_iter as f32 * x_vec + z_dist * z_iter as f32 * z_vec;
                let voxel_pos = pos.map(|c| c.floor() as u32);
                let voxel = if let Some(index) =
                    get_index(voxel_pos, chunk_reader, game_state, game_settings)
                {
                    voxel_reader[index as usize]
                } else {
                    VoxelMaterial::Unloaded.to_memory()
                };
                if voxel >> 24 != 0 {
                    return false;
                }
            }
        }
        true
    }

    fn adjust_health(&mut self, adjustment: f32) {
        if adjustment > 0.0 {
            let mut healing_left = adjustment;
            let mut health_idx = 0;
            while healing_left > 0.0 {
                let health_section = &mut self.health[health_idx];
                match health_section {
                    HealthSection::Health(current, max) => {
                        let health_to_add = (*max - *current).min(healing_left);
                        *current += health_to_add;
                        healing_left -= health_to_add;
                    }
                    HealthSection::Overhealth(_current, _duration) => {
                        // overhealth is not affected by healing
                    }
                }
                health_idx += 1;
                if health_idx >= self.health.len() {
                    break;
                }
            }
        } else {
            let mut damage_left = -adjustment;
            let mut health_idx = self.health.len() - 1;
            while damage_left > 0.0 {
                let health_section = &mut self.health[health_idx];
                match health_section {
                    HealthSection::Health(current, _) => {
                        let health_to_remove = (*current).min(damage_left);
                        *current -= health_to_remove;
                        damage_left -= health_to_remove;
                    }
                    HealthSection::Overhealth(current, _duration) => {
                        let health_to_remove = (*current).min(damage_left);
                        *current -= health_to_remove;
                        damage_left -= health_to_remove;
                    }
                }
                if health_idx == 0 {
                    break;
                }
                health_idx -= 1;
            }
        }
    }

    pub fn get_health_stats(&self) -> (f32, f32) {
        let mut current_health = 0.0;
        let mut max_health = 0.0;
        for health_section in self.health.iter() {
            match health_section {
                HealthSection::Health(current, max) => {
                    current_health += *current;
                    max_health += *max;
                }
                HealthSection::Overhealth(current, _duration) => {
                    current_health += *current;
                    max_health += *current;
                }
            }
        }
        (current_health, max_health)
    }
}

fn ray_box_dist(pos: Point3<f32>, ray: Vector3<f32>) -> Vector3<f32> {
    let vmin = pos.map(|c| c.floor());
    let vmax = pos.map(|c| c.ceil());
    let norm_min_diff: Vector3<f32> =
        (vmin - pos).zip(ray, |n, d| if d == 0.0 { 2.0 } else { n / d });
    let norm_max_diff: Vector3<f32> =
        (vmax - pos).zip(ray, |n, d| if d == 0.0 { 2.0 } else { n / d });
    return norm_min_diff.zip(norm_max_diff, |min_diff, max_diff| min_diff.max(max_diff));
}
