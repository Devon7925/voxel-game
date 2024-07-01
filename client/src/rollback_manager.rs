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
    vec3, ElementWise, EuclideanSpace, InnerSpace, One, Point3, Quaternion, Rad, Rotation,
    Rotation3, Vector2, Vector3,
};
use itertools::Itertools;
use matchbox_socket::{PeerId, PeerState};
use serde::{Deserialize, Serialize};
use vulkano::{
    buffer::{subbuffer::BufferReadGuard, Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};
use winit::event::{ElementState, WindowEvent};

use crate::{
    card_system::{
        BaseCard, CardManager, Deck, DirectionCard, ReferencedCooldown, ReferencedEffect,
        ReferencedStatusEffect, ReferencedStatusEffects, ReferencedTrigger, SimpleStatusEffectType,
        StateKeybind, StatusEffect, VoxelMaterial,
    },
    game_manager::GameState,
    game_modes::GameMode,
    gui::{GuiElement, GuiState},
    networking::{NetworkConnection, NetworkPacket},
    settings_manager::{Control, Settings},
    voxel_sim_manager::{Projectile, VoxelComputePipeline},
    WindowProperties, CHUNK_SIZE, PLAYER_BASE_MAX_HEALTH, PLAYER_DENSITY, PLAYER_HITBOX_OFFSET,
    PLAYER_HITBOX_SIZE, RESPAWN_TIME,
};
use voxel_shared::{GameModeSettings, GameSettings, RoomId};

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
    fn get_current_state(&self) -> &WorldState;

    fn can_step_rollback(&self) -> bool;
    fn step_rollback(
        &mut self,
        card_manager: &mut CardManager,
        voxel_compute: &mut VoxelComputePipeline,
        game_state: &GameState,
        game_settings: &GameSettings,
        game_mode: &mut Box<dyn GameMode>,
    );
    fn step_visuals(
        &mut self,
        card_manager: &mut CardManager,
        voxel_compute: &mut VoxelComputePipeline,
        game_state: &GameState,
        game_settings: &GameSettings,
        game_mode: &Box<dyn GameMode>,
        allow_player_action: bool,
    );

    fn download_projectiles(
        &mut self,
        card_manager: &CardManager,
        voxel_compute: &mut VoxelComputePipeline,
        game_settings: &GameSettings,
    );

    fn get_camera(&self) -> Camera;
    fn get_spectate_player(&self) -> Option<Entity>;

    fn get_delta_time(&self) -> f32;
    fn get_rollback_projectiles(&self) -> &Vec<Projectile>;
    fn get_render_projectiles(&self) -> &Vec<Projectile>;
    fn get_players(&self) -> &Vec<Entity>;
    fn player_count(&self) -> usize;
    fn get_entity_metadata(&self) -> &Vec<EntityMetaData>;

    fn visable_projectile_buffer(&self) -> Subbuffer<[Projectile; 1024]>;
    fn visable_player_buffer(&self) -> Subbuffer<[UploadPlayer; 128]>;

    fn network_update(
        &mut self,
        settings: &GameSettings,
        card_manager: &mut CardManager,
        game_mode: &Box<dyn GameMode>,
    );
    fn send_gamemode_packet(&mut self, packet: String);

    fn process_event(
        &mut self,
        event: &winit::event::WindowEvent,
        settings: &Settings,
        gui_state: &mut GuiState,
        window_props: &WindowProperties,
    );
    fn leave_game(&mut self);
    fn end_frame(&mut self);
    fn get_exit_reason(&self) -> Option<String>;

    fn is_render_behind_other_players(&self) -> bool;
    fn get_rollback_time(&self) -> u64;
    fn get_current_time(&self) -> u64;
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Action {
    pub primary_action: Option<PlayerAction>,
    pub meta_action: Option<MetaAction>,
}

impl Action {
    fn empty() -> Self {
        Action {
            primary_action: None,
            meta_action: None,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub enum EntityMetaData {
    Player(Deck, VecDeque<Action>),
    TrainingBot,
}

impl EntityMetaData {
    fn step(&mut self) {
        match self {
            EntityMetaData::Player(_, actions) => {
                let latest = actions.pop_front();
                assert!(latest.is_some());
                actions.push_back(Action::empty());
            }
            EntityMetaData::TrainingBot => {}
        }
    }

    fn get_action(&self, rollback_offset: u64) -> Action {
        match self {
            EntityMetaData::Player(_, actions) => actions
                .get(rollback_offset as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "cannot access index {} in action deque of length {}",
                        rollback_offset,
                        actions.len()
                    )
                })
                .clone(),
            EntityMetaData::TrainingBot => Action {
                primary_action: Some(PlayerAction::default()),
                meta_action: None,
            },
        }
    }
}

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
    player_deck: Deck,
    player_idx_map: HashMap<PeerId, usize>,
    most_future_time_recorded: u64,
    connected_player_count: usize,
    pub exit_reason: Option<String>,
}

pub struct ReplayData {
    pub current_time: u64,
    pub delta_time: f32,
    pub state: WorldState,
    pub entity_metadata: Vec<EntityMetaData>,
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
    pub deck_update: Option<Deck>,
    pub leave: Option<bool>,
    pub gamemode_action: Option<String>,
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
    pub passive_abilities: Vec<ReferencedStatusEffect>,
    pub respawn_timer: f32,
    pub collision_vec: Vector3<i32>,
    pub movement_direction: Vector3<f32>,
    pub status_effects: Vec<AppliedStatusEffect>,
    pub player_piercing_invincibility: f32,
    pub hitmarker: (f32, f32),
    pub hurtmarkers: Vec<(Vector3<f32>, f32, f32)>,
    pub gamemode_data: Vec<u32>,
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
    pub remaining_charges: u32,
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
    gravity: Vector3<f32>,
    size: f32,
    max_health: f32,
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
        MetaAction {
            deck_update: None,
            leave: None,
            gamemode_action: None,
        }
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
            health: vec![HealthSection::Health(
                PLAYER_BASE_MAX_HEALTH,
                PLAYER_BASE_MAX_HEALTH,
            )],
            abilities: Vec::new(),
            passive_abilities: Vec::new(),
            respawn_timer: 0.0,
            collision_vec: Vector3::new(0, 0, 0),
            movement_direction: Vector3::new(0.0, 0.0, 0.0),
            status_effects: Vec::new(),
            player_piercing_invincibility: 0.0,
            hitmarker: (0.0, 0.0),
            hurtmarkers: Vec::new(),
            gamemode_data: Vec::new(),
        }
    }
}

pub fn abilities_from_cooldowns(
    deck: &Deck,
    card_manager: &mut CardManager,
    give_cooldown: bool,
) -> (Vec<PlayerAbility>, Vec<ReferencedStatusEffect>) {
    let total_impact = deck.get_total_impact();
    (
        deck.cooldowns
            .iter()
            .map(|cooldown| {
                let cooldown_time = cooldown.get_cooldown_recovery(total_impact);
                let ability = card_manager.register_cooldown(cooldown.clone());
                PlayerAbility {
                    value: cooldown_time.clone(),
                    cooldown: cooldown_time.0,
                    recovery: 0.0,
                    remaining_charges: if give_cooldown {
                        ability.max_charges
                    } else {
                        0
                    },
                    ability,
                }
            })
            .collect(),
        deck.passive
            .passive_effects
            .iter()
            .flat_map(|x| card_manager.register_status_effect(x.clone()))
            .collect(),
    )
}

impl PlayerSim for RollbackData {
    fn can_step_rollback(&self) -> bool {
        let rollback_actions: Vec<Action> = self
            .entity_metadata
            .iter()
            .map(|x| x.get_action(0))
            .collect();
        rollback_actions.iter().all(|a| {
            a.primary_action.is_some()
                || a.meta_action
                    .clone()
                    .map(|m| m.leave.unwrap_or(false))
                    .unwrap_or(false)
        })
    }
    fn step_rollback(
        &mut self,
        card_manager: &mut CardManager,
        vox_compute: &mut VoxelComputePipeline,
        game_state: &GameState,
        game_settings: &GameSettings,
        game_mode: &mut Box<dyn GameMode>,
    ) {
        let rollback_actions: Vec<Action> = self
            .entity_metadata
            .iter()
            .map(|x| x.get_action(0))
            .collect();

        if let Some(replay_file) = self.replay_file.as_mut() {
            write!(replay_file, "\nTIME {}", self.rollback_time).unwrap();
        }
        self.rollback_time += 1;
        let mut leaving_players = Vec::new();
        {
            if let Some(replay_file) = self.replay_file.as_mut() {
                replay_file.write_all(b"\n").unwrap();
                ron::ser::to_writer(replay_file, &rollback_actions).unwrap();
            }
            for (player_idx, meta_action) in
                rollback_actions.iter().map(|a| &a.meta_action).enumerate()
            {
                if let Some(MetaAction {
                    deck_update,
                    leave,
                    gamemode_action,
                }) = meta_action
                {
                    if let Some(new_deck) = deck_update {
                        (
                            self.rollback_state.players[player_idx].abilities,
                            self.rollback_state.players[player_idx].passive_abilities,
                        ) = abilities_from_cooldowns(
                            new_deck,
                            card_manager,
                            game_mode.cooldowns_reset_on_deck_swap(),
                        )
                    }
                    if let Some(true) = leave {
                        println!("player {} left", player_idx);
                        leaving_players.push(player_idx);
                    }
                    if let Some(game_mode_action) = gamemode_action {
                        game_mode.send_action(
                            player_idx,
                            game_mode_action.clone(),
                            &mut self.rollback_state.players,
                        );
                    }
                }
            }
        }
        self.entity_metadata.iter_mut().for_each(|x| x.step());
        self.rollback_state.step_sim(
            rollback_actions,
            true,
            card_manager,
            self.get_delta_time(),
            vox_compute,
            game_state,
            game_settings,
            game_mode,
        );
        game_mode.update(
            &mut self.rollback_state.players,
            &mut self.rollback_state.projectiles,
            self.delta_time,
        );
        if leaving_players.len() > 0 {
            for player_idx in leaving_players.iter() {
                self.rollback_state.players.remove(*player_idx);
                self.entity_metadata.remove(*player_idx);
            }
            let mut new_player_idx_map = HashMap::new();
            for (peer, player_idx) in self.player_idx_map.iter() {
                let new_player_idx =
                    player_idx - leaving_players.iter().filter(|x| *x < player_idx).count();
                if new_player_idx < self.rollback_state.players.len() {
                    new_player_idx_map.insert(*peer, new_player_idx);
                }
            }
            self.player_idx_map = new_player_idx_map;
            self.connected_player_count -= leaving_players.len();
        }
    }

    fn get_current_state(&self) -> &WorldState {
        &self.cached_current_state
    }

    fn step_visuals(
        &mut self,
        card_manager: &mut CardManager,
        vox_compute: &mut VoxelComputePipeline,
        game_state: &GameState,
        game_settings: &GameSettings,
        game_mode: &Box<dyn GameMode>,
        allow_player_action: bool,
    ) {
        let on_ground = self
            .get_spectate_player()
            .map(|player| player.collision_vec.y > 0)
            .unwrap_or(false);
        self.controls.iter_mut().for_each(|cd| {
            cd.iter_mut()
                .for_each(|ability| ability.update_on_ground(on_ground))
        });
        if allow_player_action {
            self.player_action.activate_ability = self
                .controls
                .iter()
                .map(|cd| cd.iter().map(|ability| ability.get_state()).collect())
                .collect();
        } else {
            self.player_action.activate_ability = self
                .controls
                .iter()
                .map(|cd| cd.iter().map(|_| false).collect())
                .collect();
            self.player_action.backward = false;
            self.player_action.forward = false;
            self.player_action.left = false;
            self.player_action.right = false;
            self.player_action.jump = false;
            self.player_action.crouch = false;
        }
        self.send_action(
            Action {
                primary_action: Some(self.player_action.clone()),
                meta_action: None,
            },
            0,
            self.current_time,
        );
        if let Some(network_connection) = self.network_connection.as_mut() {
            let packet_data = NetworkPacket::Action(self.current_time, self.player_action.clone());
            network_connection.queue_packet(packet_data);
        };
        self.current_time += 1;
        self.controls
            .iter_mut()
            .for_each(|cd| cd.iter_mut().for_each(|ability| ability.clear()));
        self.cached_current_state = self.gen_current_state(
            card_manager,
            self.get_delta_time(),
            vox_compute,
            game_state,
            game_settings,
            game_mode,
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
        vox_compute: &mut VoxelComputePipeline,
        game_settings: &GameSettings,
    ) {
        self.rollback_state.projectiles =
            vox_compute.download_projectiles(card_manager, game_settings);
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

    fn get_rollback_projectiles(&self) -> &Vec<Projectile> {
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

    fn get_entity_metadata(&self) -> &Vec<EntityMetaData> {
        &self.entity_metadata
    }

    fn network_update(
        &mut self,
        settings: &GameSettings,
        card_manager: &mut CardManager,
        game_mode: &Box<dyn GameMode>,
    ) {
        let Some(network_connection) = self.network_connection.as_mut() else {
            return;
        };
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
                    self.send_action(
                        Action {
                            primary_action: Some(action),
                            meta_action: None,
                        },
                        player_idx.unwrap().clone(),
                        time,
                    );
                }
                NetworkPacket::DeckUpdate(time, cards) => {
                    self.send_action(
                        Action {
                            primary_action: None,
                            meta_action: Some(MetaAction {
                                deck_update: Some(cards.clone()),
                                ..Default::default()
                            }),
                        },
                        player_idx.unwrap().clone(),
                        time,
                    );
                    let EntityMetaData::Player(deck, _) = self
                        .entity_metadata
                        .get_mut(player_idx.unwrap().clone())
                        .unwrap()
                    else {
                        panic!("cannot update deck of non player");
                    };
                    *deck = cards;
                }
                NetworkPacket::Join(cards) => {
                    self.player_idx_map
                        .insert(peer, self.rollback_state.players.len());

                    let mut new_player = Entity::default();
                    new_player.pos = game_mode.spawn_location(&new_player);
                    (new_player.abilities, new_player.passive_abilities) = abilities_from_cooldowns(
                        &cards,
                        card_manager,
                        game_mode.cooldowns_reset_on_deck_swap(),
                    );

                    let new_player_effect_stats = new_player.get_effect_stats();
                    new_player.health = vec![HealthSection::Health(
                        new_player_effect_stats.max_health,
                        new_player_effect_stats.max_health,
                    )];

                    self.rollback_state.players.push(new_player);
                    self.entity_metadata.push(EntityMetaData::Player(
                        cards.clone(),
                        VecDeque::from(vec![
                            Action::empty();
                            (self.current_time - self.rollback_time + 15) as usize
                        ]),
                    ));

                    if let Some(replay_file) = self.replay_file.as_mut() {
                        write!(replay_file, "PLAYER DECK ").unwrap();
                        ron::ser::to_writer(replay_file, &cards).unwrap();
                    }
                }
                NetworkPacket::Leave(time) => {
                    self.send_action(
                        Action {
                            primary_action: None,
                            meta_action: Some(MetaAction {
                                leave: Some(true),
                                ..Default::default()
                            }),
                        },
                        player_idx.unwrap().clone(),
                        time,
                    );
                }
                NetworkPacket::GamemodePacket(time, packet) => {
                    self.send_action(
                        Action {
                            primary_action: None,
                            meta_action: Some(MetaAction {
                                gamemode_action: Some(packet),
                                ..Default::default()
                            }),
                        },
                        player_idx.unwrap().clone(),
                        time,
                    );
                }
            }
        }
    }

    fn send_gamemode_packet(&mut self, packet: String) {
        self.send_action(
            Action {
                primary_action: None,
                meta_action: Some(MetaAction {
                    gamemode_action: Some(packet.clone()),
                    ..Default::default()
                }),
            },
            0,
            self.current_time,
        );
        if let Some(network_connection) = self.network_connection.as_mut() {
            network_connection
                .queue_packet(NetworkPacket::GamemodePacket(self.current_time, packet));
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
                                            if let Some(unreasonable_reason) =
                                                gui_state.gui_deck.get_unreasonable_reason()
                                            {
                                                gui_state.errors.push(format!(
                                                    "Unreasonable deck not saved: {}",
                                                    unreasonable_reason
                                                ));
                                            } else {
                                                self.player_deck = gui_state.gui_deck.clone();
                                                self.controls = self
                                                    .player_deck
                                                    .cooldowns
                                                    .iter()
                                                    .map(|a| {
                                                        a.abilities
                                                            .iter()
                                                            .map(|a| {
                                                                StateKeybind::from(
                                                                    a.keybind.clone(),
                                                                )
                                                            })
                                                            .collect()
                                                    })
                                                    .collect();
                                                self.send_action(
                                                    Action {
                                                        primary_action: None,
                                                        meta_action: Some(MetaAction {
                                                            deck_update: Some(
                                                                self.player_deck.clone(),
                                                            ),
                                                            ..Default::default()
                                                        }),
                                                    },
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

    fn leave_game(&mut self) {
        if let Some(network_connection) = self.network_connection.as_mut() {
            network_connection.queue_packet(NetworkPacket::Leave(self.current_time));
            let peers = self.player_idx_map.keys().collect();
            network_connection.send_packet_queue(peers);
            println!("leaving game");
        }
    }

    fn get_exit_reason(&self) -> Option<String> {
        self.exit_reason.clone()
    }

    fn is_render_behind_other_players(&self) -> bool {
        self.most_future_time_recorded > self.current_time
    }

    fn get_rollback_time(&self) -> u64 {
        self.rollback_time
    }

    fn get_current_time(&self) -> u64 {
        self.current_time
    }
}

impl RollbackData {
    pub fn new(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        settings: &Settings,
        game_settings: &GameSettings,
        deck: &Deck,
        card_manager: &mut CardManager,
        lobby_id: Option<RoomId>,
        game_mode: &Box<dyn GameMode>,
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

        let current_time: u64 = 0;
        let rollback_time: u64 = 0;

        let mut rollback_state = WorldState::new();
        let mut entity_metadata = Vec::new();

        let mut first_player = Entity::default();
        first_player.pos = game_mode.spawn_location(&first_player);
        (first_player.abilities, first_player.passive_abilities) =
            abilities_from_cooldowns(deck, card_manager, game_mode.cooldowns_reset_on_deck_swap());
            

        let first_player_effect_stats = first_player.get_effect_stats();
        first_player.health = vec![HealthSection::Health(
            first_player_effect_stats.max_health,
            first_player_effect_stats.max_health,
        )];
        rollback_state.players.push(first_player.clone());
        entity_metadata.push(EntityMetaData::Player(
            deck.clone(),
            VecDeque::from(vec![
                Action::empty();
                game_settings.rollback_buffer_size as usize
            ]),
        ));

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
            exit_reason: None,
        }
    }

    fn gen_current_state(
        &self,
        card_manager: &CardManager,
        time_step: f32,
        vox_compute: &mut VoxelComputePipeline,
        game_state: &GameState,
        game_settings: &GameSettings,
        game_mode: &Box<dyn GameMode>,
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
                game_mode,
            );
        }
        state
    }

    pub fn send_action(&mut self, action_update: Action, player_idx: usize, time_stamp: u64) {
        if time_stamp > self.most_future_time_recorded {
            self.most_future_time_recorded = time_stamp;
        }
        if time_stamp < self.rollback_time {
            println!(
                "cannot send action {:?} for player {} with timestamp {} when rollback time is {}",
                action_update, player_idx, time_stamp, self.rollback_time
            );
            return;
        }
        let time_idx = time_stamp - self.rollback_time;
        let EntityMetaData::Player(_, actions) =
            self.entity_metadata.get_mut(player_idx).unwrap_or_else(|| {
                panic!(
                    "cannot access entity index {} on sending with timestamp {}",
                    player_idx, time_stamp
                )
            })
        else {
            panic!("cannot send action from non player");
        };
        let Some(action) = actions.get_mut(time_idx as usize) else {
            self.leave_game();
            self.exit_reason = Some(format!(
                "Exiting due to excessive lag: cannot access index {} on sending with timestamp {}",
                time_idx, time_stamp
            ));
            return;
        };
        if let Some(new_primary_action) = action_update.primary_action {
            action.primary_action = Some(new_primary_action);
        }
        if let Some(new_meta_action) = action_update.meta_action {
            if action.meta_action.is_none() {
                action.meta_action = Some(MetaAction::default());
            }
            if let Some(deck_update) = new_meta_action.deck_update {
                action.meta_action.as_mut().unwrap().deck_update = Some(deck_update);
            }
            if let Some(leave) = new_meta_action.leave {
                action.meta_action.as_mut().unwrap().leave = Some(leave);
            }
            if let Some(gamemode_action) = new_meta_action.gamemode_action {
                action.meta_action.as_mut().unwrap().gamemode_action = Some(gamemode_action);
            }
        }
    }
}

impl PlayerSim for ReplayData {
    fn can_step_rollback(&self) -> bool {
        !self.actions.is_empty()
    }

    fn step_rollback(
        &mut self,
        card_manager: &mut CardManager,
        voxel_compute: &mut VoxelComputePipeline,
        game_state: &GameState,
        game_settings: &GameSettings,
        game_mode: &mut Box<dyn GameMode>,
    ) {
        if self.actions.is_empty() {
            return;
        }
        self.current_time += 1;
        let rollback_actions: Vec<Action> = self.actions.pop_front().unwrap();
        let mut leaving_players = Vec::new();
        {
            for (player_idx, meta_action) in
                rollback_actions.iter().map(|a| &a.meta_action).enumerate()
            {
                if let Some(MetaAction {
                    deck_update, leave, ..
                }) = meta_action
                {
                    if let Some(new_deck) = deck_update {
                        (
                            self.state.players[player_idx].abilities,
                            self.state.players[player_idx].passive_abilities,
                        ) = abilities_from_cooldowns(
                            new_deck,
                            card_manager,
                            game_mode.cooldowns_reset_on_deck_swap(),
                        )
                    }
                    if let Some(true) = leave {
                        leaving_players.push(player_idx);
                    }
                }
            }
        }
        self.state.step_sim(
            rollback_actions,
            true,
            card_manager,
            self.get_delta_time(),
            voxel_compute,
            game_state,
            game_settings,
            game_mode,
        );
        game_mode.update(
            &mut self.state.players,
            &mut self.state.projectiles,
            self.delta_time,
        );
        if leaving_players.len() > 0 {
            for player_idx in leaving_players.iter() {
                self.state.players.remove(*player_idx);
                self.actions.remove(*player_idx);
            }
        }
    }

    fn get_current_state(&self) -> &WorldState {
        &self.state
    }

    fn step_visuals(
        &mut self,
        _card_manager: &mut CardManager,
        _voxel_compute: &mut VoxelComputePipeline,
        _game_state: &GameState,
        _game_settings: &GameSettings,
        _game_mode: &Box<dyn GameMode>,
        _allow_player_action: bool,
    ) {
        puffin::profile_function!();
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
        vox_compute: &mut VoxelComputePipeline,
        game_settings: &GameSettings,
    ) {
        self.state.projectiles = vox_compute.download_projectiles(card_manager, game_settings);
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

    fn get_rollback_projectiles(&self) -> &Vec<Projectile> {
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

    fn get_entity_metadata(&self) -> &Vec<EntityMetaData> {
        &self.entity_metadata
    }

    fn network_update(
        &mut self,
        _settings: &GameSettings,
        _card_manager: &mut CardManager,
        _game_mode: &Box<dyn GameMode>,
    ) {
    }
    fn send_gamemode_packet(&mut self, _packet: String) {}

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
    fn leave_game(&mut self) {}
    fn get_exit_reason(&self) -> Option<String> {
        None
    }

    fn is_render_behind_other_players(&self) -> bool {
        false
    }

    fn get_rollback_time(&self) -> u64 {
        self.current_time
    }

    fn get_current_time(&self) -> u64 {
        self.current_time
    }
}

impl ReplayData {
    pub fn new(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        game_settings: &GameSettings,
        replay_lines: &mut Lines<BufReader<File>>,
        card_manager: &mut CardManager,
        game_mode: &Box<dyn GameMode>,
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
        let mut entity_metadata = Vec::new();
        let mut actions = VecDeque::new();
        while let Some(line) = replay_lines.next() {
            let Ok(line) = line else {
                continue;
            };
            if let Some(_game_settings_string) = line.strip_prefix("GAME SETTINGS ") {
                panic!("Game settings should have already been handled")
            } else if let Some(deck_string) = line.strip_prefix("PLAYER DECK ") {
                let deck: Deck = ron::de::from_str(deck_string).unwrap();

                let mut new_player = Entity::default();
                new_player.pos = game_mode.spawn_location(&new_player);
                (new_player.abilities, new_player.passive_abilities) = abilities_from_cooldowns(
                    &deck,
                    card_manager,
                    game_mode.cooldowns_reset_on_deck_swap(),
                );
                
                let new_player_effect_stats = new_player.get_effect_stats();
                new_player.health = vec![HealthSection::Health(
                    new_player_effect_stats.max_health,
                    new_player_effect_stats.max_health,
                )];
                state.players.push(new_player);
                entity_metadata.push(EntityMetaData::Player(deck.clone(), VecDeque::new()));
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
            entity_metadata,
            actions,
            player_buffer,
            projectile_buffer,
        }
    }
}

struct NewEffects {
    new_projectiles: Vec<Projectile>,
    voxels_to_write: Vec<(Point3<u32>, u32)>,
    new_effects: Vec<(
        usize,
        usize,
        bool,
        Point3<f32>,
        Vector3<f32>,
        ReferencedEffect,
    )>,
    new_status_effects: Vec<(usize, ReferencedStatusEffects)>,
    step_triggers: Vec<(ReferencedTrigger, u32)>,
}

impl Default for NewEffects {
    fn default() -> Self {
        NewEffects {
            new_projectiles: Vec::new(),
            voxels_to_write: Vec::new(),
            new_effects: Vec::new(),
            new_status_effects: Vec::new(),
            step_triggers: Vec::new(),
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
        game_mode: &Box<dyn GameMode>,
    ) {
        let voxels = vox_compute.voxels();
        let mut new_effects = NewEffects::default();

        let player_stats: Vec<PlayerEffectStats> = self.get_player_effect_stats();

        for (player, player_stats) in self.players.iter_mut().zip(player_stats.iter()) {
            let mut health_adjustment = 0.0;
            for status_effect in player
                .status_effects
                .iter()
                .map(|e| &e.effect)
                .chain(player.passive_abilities.iter())
            {
                match status_effect {
                    ReferencedStatusEffect::DamageOverTime(stacks) => {
                        health_adjustment +=
                            -10.0 * player_stats.damage_taken * *stacks as f32 * time_step;
                    }
                    ReferencedStatusEffect::Overheal(stacks) => {
                        player.health.push(HealthSection::Overhealth(10.0 * *stacks as f32 * time_step / BaseCard::EFFECT_LENGTH_SCALE, BaseCard::EFFECT_LENGTH_SCALE));
                    }
                    _ => {}
                }
            }
            for status_effect in player.status_effects.iter_mut() {
                status_effect.time_left -= time_step;
            }
            if health_adjustment != 0.0 {
                player.adjust_health(health_adjustment);
            }
            player.status_effects.retain(|x| x.time_left > 0.0);
        }

        {
            let voxel_reader = voxels.read().unwrap();
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
                    &vox_compute.cpu_chunks(),
                    &mut new_effects,
                    game_state,
                    game_settings,
                    game_mode,
                );
            }
        }

        self.projectiles.iter_mut().for_each(|proj| {
            proj.simple_step(&self.players, card_manager, time_step, &mut new_effects)
        });

        let collision_pairs = self.get_collision_pairs(card_manager, time_step, game_mode);

        for (i, j) in collision_pairs.iter() {
            let damage_1 = self.projectiles.get(*i).unwrap().damage;
            let damage_2 = self.projectiles.get(*j).unwrap().damage;
            {
                let proj1_mut = self.projectiles.get_mut(*j).unwrap();
                proj1_mut.health -= damage_2;
                proj1_mut.health -= damage_1;
            }
            {
                let proj2_mut = self.projectiles.get_mut(*i).unwrap();
                proj2_mut.health -= damage_1;
                proj2_mut.health -= damage_2;
            }
            // trigger on hit effects
            {
                let proj = self.projectiles.get(*i).unwrap();
                let hit_cards = card_manager
                    .get_referenced_proj(proj.proj_card_idx as usize)
                    .on_hit
                    .iter()
                    .map(|x| (x.clone(), false))
                    .collect::<Vec<_>>();
                for (card_ref, _was_headshot) in hit_cards {
                    let proj_rot = proj.dir;
                    let proj_rot =
                        Quaternion::new(proj_rot[3], proj_rot[0], proj_rot[1], proj_rot[2]);
                    let (on_hit_projectiles, on_hit_voxels, _effects, _status_effects, triggers) =
                        card_manager.get_effects_from_base_card(
                            card_ref,
                            &Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]),
                            &proj_rot,
                            proj.owner,
                            false,
                        );
                    new_effects.new_projectiles.extend(on_hit_projectiles);
                    for (pos, material) in on_hit_voxels {
                        new_effects
                            .voxels_to_write
                            .push((pos, material.to_memory()));
                    }
                    new_effects.step_triggers.extend(triggers);
                }
            }
            {
                let proj = self.projectiles.get(*j).unwrap();
                let hit_cards = card_manager
                    .get_referenced_proj(proj.proj_card_idx as usize)
                    .on_hit
                    .iter()
                    .map(|x| (x.clone(), false))
                    .collect::<Vec<_>>();
                for (card_ref, _was_headshot) in hit_cards {
                    let proj_rot = proj.dir;
                    let proj_rot =
                        Quaternion::new(proj_rot[3], proj_rot[0], proj_rot[1], proj_rot[2]);
                    let (on_hit_projectiles, on_hit_voxels, _effects, _status_effects, triggers) =
                        card_manager.get_effects_from_base_card(
                            card_ref,
                            &Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]),
                            &proj_rot,
                            proj.owner,
                            false,
                        );
                    new_effects.new_projectiles.extend(on_hit_projectiles);
                    for (pos, material) in on_hit_voxels {
                        new_effects
                            .voxels_to_write
                            .push((pos, material.to_memory()));
                    }
                    new_effects.step_triggers.extend(triggers);
                }
            }
        }

        let proj_collisions =
            self.get_player_proj_collisions(&player_stats, card_manager, game_mode, time_step);

        proj_collisions
            .iter()
            .filter(|(_, proj_idx, damage_source_location, collision)| {
                let vec_start = damage_source_location.to_vec();
                let vec_end = collision.offset;
                self.projectiles
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| {
                        collision_pairs.contains(&(*proj_idx.min(idx), *proj_idx.max(idx)))
                    })
                    .all(|(_, proj2)| {
                        let mut adj_vec_start = vec_start;
                        let mut adj_vec_end = vec_end;
                        adj_vec_start -= Vector3::new(proj2.pos[0], proj2.pos[1], proj2.pos[2]);
                        adj_vec_end -= Vector3::new(proj2.pos[0], proj2.pos[1], proj2.pos[2]);
                        let proj2_rot =
                            Quaternion::new(proj2.dir[3], proj2.dir[0], proj2.dir[1], proj2.dir[2]);
                        let proj2_rot_inv = proj2_rot.invert();
                        adj_vec_start = proj2_rot_inv.rotate_vector(adj_vec_start);
                        adj_vec_end = proj2_rot_inv.rotate_vector(adj_vec_end);
                        adj_vec_start.div_assign_element_wise(Vector3::new(
                            proj2.size[0],
                            proj2.size[1],
                            proj2.size[2],
                        ));
                        adj_vec_end.div_assign_element_wise(Vector3::new(
                            proj2.size[0],
                            proj2.size[1],
                            proj2.size[2],
                        ));

                        let vec_dir = adj_vec_end - adj_vec_start;
                        let (t_min, t_max) = (0..3)
                            .map(|i| {
                                (
                                    (-1.0 - adj_vec_start[i]) / vec_dir[i],
                                    (1.0 - adj_vec_start[i]) / vec_dir[i],
                                )
                            })
                            .map(|t| (t.0.min(t.1), t.0.max(t.1)))
                            .reduce(|(t_min1, t_max1), (t_min2, t_max2)| {
                                (t_min1.max(t_min2), t_max1.min(t_max2))
                            })
                            .unwrap();
                        !(t_min < 1.0 && t_max > 0.0 && t_min < t_max)
                    })
            })
            .collect_vec()
            .iter()
            .for_each(|(player_idx, proj_idx, _, collision)| {
                let player = self.players.get_mut(*player_idx).unwrap();
                let proj = self.projectiles.get_mut(*proj_idx).unwrap();
                let proj_card = card_manager.get_referenced_proj(proj.proj_card_idx as usize);

                let projectile_rot =
                    Quaternion::new(proj.dir[3], proj.dir[0], proj.dir[1], proj.dir[2]);
                let projectile_vectors = [
                    projectile_rot.rotate_vector(Vector3::new(1.0, 0.0, 0.0)),
                    projectile_rot.rotate_vector(Vector3::new(0.0, 1.0, 0.0)),
                    projectile_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0)),
                ];
                let projectile_pos = Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]);

                if !proj_card.pierce_players {
                    proj.health = 0.0;
                } else {
                    player.player_piercing_invincibility = 0.3;
                }
                let mut hit_cards = card_manager
                    .get_referenced_proj(proj.proj_card_idx as usize)
                    .on_hit
                    .iter()
                    .map(|x| (x.clone(), false))
                    .collect::<Vec<_>>();
                if collision.headshot {
                    hit_cards.extend(
                        card_manager
                            .get_referenced_proj(proj.proj_card_idx as usize)
                            .on_headshot
                            .iter()
                            .map(|x| (x.clone(), true)),
                    );
                }
                for (card_ref, was_headshot) in hit_cards {
                    let proj_rot = proj.dir;
                    let proj_rot =
                        Quaternion::new(proj_rot[3], proj_rot[0], proj_rot[1], proj_rot[2]);
                    let (on_hit_projectiles, on_hit_voxels, effects, status_effects, triggers) =
                        card_manager.get_effects_from_base_card(
                            card_ref,
                            &Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]),
                            &proj_rot,
                            proj.owner,
                            false,
                        );
                    new_effects.new_projectiles.extend(on_hit_projectiles);
                    for (pos, material) in on_hit_voxels {
                        new_effects
                            .voxels_to_write
                            .push((pos, material.to_memory()));
                    }
                    for effect in effects {
                        new_effects.new_effects.push((
                            *player_idx,
                            proj.owner as usize,
                            was_headshot,
                            Point3::from_vec(collision.offset),
                            ((collision.offset - projectile_pos.to_vec()).normalize()
                                + projectile_vectors[2] * proj.vel)
                                .normalize(),
                            effect,
                        ));
                    }
                    for status_effects in status_effects {
                        new_effects
                            .new_status_effects
                            .push((*player_idx, status_effects));
                    }
                    new_effects.step_triggers.extend(triggers);
                }
            });

        let mut player_player_collision_pairs: Vec<(usize, usize)> = vec![];
        for i in 0..self.players.len() {
            let player1 = self.players.get(i).unwrap();
            if player1.respawn_timer > 0.0 || player_stats[i].invincible {
                continue;
            }
            for j in 0..self.players.len() {
                if game_mode.are_friends(i as u32, j as u32, &self.players) {
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
                    .chain(player1.passive_abilities.iter().filter_map(|effect| match effect {
                        ReferencedStatusEffect::OnHit(hit_card) => Some(hit_card),
                        _ => None,
                    }))
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
            for (on_hit_projectiles, on_hit_voxels, effects, status_effects, triggers) in
                hit_effects
            {
                new_effects.new_projectiles.extend(on_hit_projectiles);
                for (pos, material) in on_hit_voxels {
                    new_effects
                        .voxels_to_write
                        .push((pos, material.to_memory()));
                }
                for effect in effects {
                    new_effects.new_effects.push((
                        j,
                        i,
                        false,
                        player1_pos,
                        player2_pos - player1_pos,
                        effect,
                    ));
                }
                for status_effects in status_effects {
                    new_effects.new_status_effects.push((j, status_effects));
                }
                new_effects.step_triggers.extend(triggers);
            }
        }

        for player in self.players.iter_mut() {
            if player.get_health_stats().0 <= 0.0 && player.respawn_timer <= 0.0 {
                player.respawn_timer = RESPAWN_TIME;
            }
        }

        for (ReferencedTrigger(trigger_id), trigger_player) in new_effects.step_triggers {
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
                        let (proj_effects, vox_effects, effects, status_effects, _) = card_manager
                            .get_effects_from_base_card(
                                on_trigger,
                                &Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]),
                                &projectile_rot,
                                proj.owner,
                                false,
                            );
                        new_effects.new_projectiles.extend(proj_effects);
                        for (pos, material) in vox_effects {
                            new_effects
                                .voxels_to_write
                                .push((pos, material.to_memory()));
                        }
                        for effect in effects {
                            new_effects.new_effects.push((
                                proj.owner as usize,
                                trigger_player as usize,
                                false,
                                Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]),
                                projectile_dir,
                                effect,
                            ));
                        }
                        for status_effects in status_effects {
                            new_effects
                                .new_status_effects
                                .push((proj.owner as usize, status_effects));
                        }
                    }
                }
            }
        }

        // remove dead projectiles and add new ones
        self.projectiles.retain(|proj| proj.health > 0.0);
        self.projectiles.extend(new_effects.new_projectiles);

        // update voxels
        if is_real_update && new_effects.voxels_to_write.len() > 0 {
            for (pos, material) in new_effects.voxels_to_write {
                vox_compute.queue_voxel_write([pos[0], pos[1], pos[2], material]);
            }
        }

        for (affected_idx, actor_idx, was_headshot, effect_pos, effect_direction, effect) in
            new_effects.new_effects
        {
            let player = self.players.get_mut(affected_idx).unwrap();
            match effect {
                ReferencedEffect::Damage(damage) => {
                    player.adjust_health(-player_stats[affected_idx].damage_taken * damage as f32);
                    if damage > 0 {
                        player.hurtmarkers.push((
                            effect_direction,
                            player_stats[affected_idx].damage_taken * damage as f32,
                            1.0,
                        ));
                    }
                    let actor = self.players.get_mut(actor_idx).unwrap();
                    if was_headshot {
                        actor.hitmarker.1 += damage as f32;
                    } else {
                        actor.hitmarker.0 += damage as f32;
                    }
                }
                ReferencedEffect::Knockback(knockback, direction) => {
                    let knockback = 10.0 * knockback as f32;
                    let knockback_dir = match direction {
                        DirectionCard::None => Vector3::new(0.0, 0.0, 0.0),
                        DirectionCard::Forward => effect_direction,
                        DirectionCard::Up => Vector3::new(0.0, 1.0, 0.0),
                        DirectionCard::Movement => player.movement_direction,
                    };
                    if knockback_dir.magnitude() > 0.0 {
                        player.vel += knockback * (knockback_dir).normalize();
                    } else {
                        player.vel.y += knockback;
                    }
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
        for (player_idx, status_effects) in new_effects.new_status_effects {
            let player: &mut Entity = self.players.get_mut(player_idx).unwrap();
            for status_effect in status_effects.effects {
                match status_effect {
                    ReferencedStatusEffect::Overheal(stacks) => {
                        player.health.push(HealthSection::Overhealth(
                            10.0 * stacks as f32,
                            BaseCard::EFFECT_LENGTH_SCALE * status_effects.duration as f32,
                        ));
                    }
                    _ => {}
                }
                player.status_effects.push(AppliedStatusEffect {
                    effect: status_effect,
                    time_left: BaseCard::EFFECT_LENGTH_SCALE * status_effects.duration as f32,
                })
            }
        }
    }

    fn get_player_proj_collisions(
        &self,
        player_stats: &Vec<PlayerEffectStats>,
        card_manager: &CardManager,
        game_mode: &Box<dyn GameMode>,
        time_step: f32,
    ) -> Vec<(usize, usize, Point3<f32>, Hitsphere)> {
        let mut proj_collisions = Vec::new();
        for (player_idx, player) in self.players.iter().enumerate() {
            if player.respawn_timer > 0.0 || player_stats[player_idx].invincible {
                continue;
            }

            // check piercing invincibility at start to prevent order from mattering
            let player_piercing_invincibility = player.player_piercing_invincibility > 0.0;
            // check for collision with projectiles
            for (proj_idx, proj) in self.projectiles.iter().enumerate() {
                if player_idx as u32 == proj.owner && proj.lifetime < 1.0 && proj.is_from_head == 1
                {
                    continue;
                }
                let proj_card = card_manager.get_referenced_proj(proj.proj_card_idx as usize);

                if proj_card.no_friendly_fire
                    && game_mode.are_friends(proj.owner, player_idx as u32, &self.players)
                {
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
                let projectile_vectors = [
                    projectile_rot.rotate_vector(Vector3::new(1.0, 0.0, 0.0)),
                    projectile_rot.rotate_vector(Vector3::new(0.0, 1.0, 0.0)),
                    projectile_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0)),
                ];
                let projectile_pos = Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]);
                let adjusted_projectile_size = if proj_card.lock_owner.is_some() {
                    Vector3::new(proj.size[0], proj.size[1], proj.size[2])
                } else {
                    Vector3::new(
                        proj.size[0],
                        proj.size[1],
                        proj.size[2] + proj.vel * time_step / 2.0,
                    )
                };
                let mut collision: Option<Hitsphere> = None;
                'outer: for hitsphere in Entity::HITSPHERES.iter().map(|x| Hitsphere {
                    offset: (player.pos + x.offset * player.size).to_vec(),
                    radius: x.radius * player.size,
                    headshot: x.headshot,
                }) {
                    if let Some(prev_collision) = collision.as_ref() {
                        if (projectile_pos - hitsphere.offset).to_vec().magnitude()
                            > (projectile_pos - prev_collision.offset)
                                .to_vec()
                                .magnitude()
                        {
                            continue 'outer;
                        }
                    }
                    for i in 0..3 {
                        if (hitsphere.offset.dot(projectile_vectors[i])
                            - projectile_pos.dot(projectile_vectors[i]))
                        .abs()
                            > adjusted_projectile_size[i] + hitsphere.radius
                        {
                            continue 'outer;
                        }
                    }
                    collision = Some(hitsphere.clone());
                }
                if let Some(collision) = collision {
                    proj_collisions.push((player_idx, proj_idx, projectile_pos, collision));
                }
            }
        }
        proj_collisions
    }

    fn get_player_effect_stats(&self) -> Vec<PlayerEffectStats> {
        self.players.iter().map(Entity::get_effect_stats).collect()
    }

    fn get_collision_pairs(
        &self,
        card_manager: &CardManager,
        time_step: f32,
        game_mode: &Box<dyn GameMode>,
    ) -> Vec<(usize, usize)> {
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
            let adjusted_projectile_1_size = Vector3::new(
                proj1.size[0],
                proj1.size[1],
                proj1.size[2]
                    + if proj1_card.lock_owner.is_none() {
                        proj1.vel * time_step / 2.0
                    } else {
                        0.0
                    },
            );
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
                    pos += adjusted_projectile_1_size[i] * projectile_1_vectors[i] * c[i];
                }
                pos
            })
            .collect::<Vec<_>>();
            'second_proj_loop: for j in i + 1..self.projectiles.len() {
                let proj2 = self.projectiles.get(j).unwrap();

                if proj1.health <= 1.0 && proj2.health <= 1.0 {
                    continue;
                }

                let proj2_card = card_manager.get_referenced_proj(proj2.proj_card_idx as usize);

                if (proj1_card.no_friendly_fire && proj2_card.no_friendly_fire)
                    && game_mode.are_friends(proj1.owner, proj2.owner, &self.players)
                {
                    continue;
                }
                if (proj1_card.no_enemy_fire && proj2_card.no_enemy_fire)
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
                let adjusted_projectile_2_size = Vector3::new(
                    proj2.size[0],
                    proj2.size[1],
                    proj2.size[2]
                        + if proj2_card.lock_owner.is_none() {
                            proj2.vel * time_step / 2.0
                        } else {
                            0.0
                        },
                );

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
                        pos += adjusted_projectile_2_size[i] * projectile_2_vectors[i] * c[i];
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
                            + adjusted_projectile_1_size[i]
                        || max_proj_2
                            < projectile_1_pos.to_vec().dot(projectile_1_vectors[i])
                                - adjusted_projectile_1_size[i]
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
                            + adjusted_projectile_2_size[i]
                        || max_proj_1
                            < projectile_2_pos.to_vec().dot(projectile_2_vectors[i])
                                - adjusted_projectile_2_size[i]
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
        new_effects: &mut NewEffects,
    ) {
        let proj_card = card_manager.get_referenced_proj(self.proj_card_idx as usize);
        let projectile_rot = Quaternion::new(self.dir[3], self.dir[0], self.dir[1], self.dir[2]);
        let projectile_dir = projectile_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0));
        let mut proj_vel = projectile_dir * self.vel;

        let new_projectile_rot: Quaternion<f32> = if let Some(direction) = &proj_card.lock_owner {
            let player_dir = players[self.owner as usize].dir;
            let player_up = players[self.owner as usize].up;
            match direction {
                DirectionCard::Forward => {
                    let proj_pos = players[self.owner as usize].pos
                        + 0.1 * proj_card.speed * player_dir
                        - 0.25 * proj_card.gravity * player_up;
                    for i in 0..3 {
                        self.pos[i] = proj_pos[i];
                    }
                    players[self.owner as usize].rot
                }
                DirectionCard::Up => {
                    let proj_pos = players[self.owner as usize].pos
                        + 0.1 * proj_card.speed * Vector3::new(0.0, 1.0, 0.0)
                        - 0.25 * proj_card.gravity * player_up;
                    for i in 0..3 {
                        self.pos[i] = proj_pos[i];
                    }
                    Quaternion::from_arc(projectile_dir, Vector3::new(0.0, 1.0, 0.0), None)
                        * projectile_rot
                }
                DirectionCard::Movement => {
                    let proj_pos = players[self.owner as usize].pos
                        + 0.1 * proj_card.speed * players[self.owner as usize].movement_direction
                        - 0.25 * proj_card.gravity * player_up;
                    for i in 0..3 {
                        self.pos[i] = proj_pos[i];
                    }
                    Quaternion::from_arc(
                        projectile_dir,
                        players[self.owner as usize].movement_direction,
                        None,
                    ) * projectile_rot
                }
                DirectionCard::None => {
                    let proj_pos = players[self.owner as usize].pos
                        + 0.1 * proj_card.speed * player_dir
                        - 0.25 * proj_card.gravity * player_up;
                    for i in 0..3 {
                        self.pos[i] = proj_pos[i];
                    }
                    projectile_rot
                }
            }
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
                let (proj_effects, vox_effects, effects, status_effects, triggers) = card_manager
                    .get_effects_from_base_card(
                        card_ref,
                        &Point3::new(self.pos[0], self.pos[1], self.pos[2]),
                        &new_projectile_rot,
                        self.owner,
                        false,
                    );
                new_effects.new_projectiles.extend(proj_effects);
                for (pos, material) in vox_effects {
                    new_effects
                        .voxels_to_write
                        .push((pos, material.to_memory()));
                }
                for effect in effects {
                    new_effects.new_effects.push((
                        self.owner as usize,
                        self.owner as usize,
                        false,
                        Point3::new(self.pos[0], self.pos[1], self.pos[2]),
                        projectile_dir,
                        effect,
                    ));
                }
                for status_effects in status_effects {
                    new_effects
                        .new_status_effects
                        .push((self.owner as usize, status_effects));
                }
                new_effects.step_triggers.extend(triggers);
            }
        }

        for (trail_time, trail_card) in proj_card.trail.iter() {
            if self.lifetime % trail_time >= trail_time - time_step {
                let (proj_effects, vox_effects, effects, status_effects, triggers) = card_manager
                    .get_effects_from_base_card(
                        trail_card.clone(),
                        &Point3::new(self.pos[0], self.pos[1], self.pos[2]),
                        &new_projectile_rot,
                        self.owner,
                        false,
                    );
                new_effects.new_projectiles.extend(proj_effects);
                for (pos, material) in vox_effects {
                    new_effects
                        .voxels_to_write
                        .push((pos, material.to_memory()));
                }
                for effect in effects {
                    new_effects.new_effects.push((
                        self.owner as usize,
                        self.owner as usize,
                        false,
                        Point3::new(self.pos[0], self.pos[1], self.pos[2]),
                        projectile_dir,
                        effect,
                    ));
                }
                for status_effects in status_effects {
                    new_effects
                        .new_status_effects
                        .push((self.owner as usize, status_effects));
                }
                new_effects.step_triggers.extend(triggers);
            }
        }
    }
}

pub fn is_inbounds(
    global_pos: Point3<u32>,
    game_state: &GameState,
    game_settings: &GameSettings,
) -> bool {
    (global_pos / CHUNK_SIZE as u32).zip(game_state.start_pos, |a, b| a >= b)
        == Point3::new(true, true, true)
        && (global_pos / CHUNK_SIZE as u32)
            .zip(game_state.start_pos + game_settings.render_size, |a, b| {
                a < b
            })
            == Point3::new(true, true, true)
}

pub fn get_index(
    global_pos: Point3<u32>,
    cpu_chunks: &Vec<Vec<Vec<u32>>>,
    game_state: &GameState,
    game_settings: &GameSettings,
) -> Option<u32> {
    if !is_inbounds(global_pos, game_state, game_settings) {
        return None;
    }
    let chunk_pos = (global_pos / CHUNK_SIZE as u32)
        .zip(Point3::from_vec(game_settings.render_size), |a, b| a % b);
    let pos_in_chunk = global_pos % CHUNK_SIZE as u32;
    let chunk_idx = cpu_chunks[chunk_pos.x as usize][chunk_pos.y as usize][chunk_pos.z as usize];
    let idx_in_chunk = pos_in_chunk.x * CHUNK_SIZE as u32 * CHUNK_SIZE as u32
        + pos_in_chunk.y * CHUNK_SIZE as u32
        + pos_in_chunk.z;
    Some(chunk_idx * CHUNK_SIZE as u32 * CHUNK_SIZE as u32 * CHUNK_SIZE as u32 + idx_in_chunk)
}

#[derive(Debug, Clone)]
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
        cpu_chunks: &Vec<Vec<Vec<u32>>>,
        new_effects: &mut NewEffects,
        game_state: &GameState,
        game_settings: &GameSettings,
        game_mode: &Box<dyn GameMode>,
    ) {
        let size_change = player_stats[player_idx].size - self.size;
        if size_change > 0.0 {
            self.pos += size_change
                * (0.5 * PLAYER_HITBOX_SIZE.y - PLAYER_HITBOX_OFFSET.y)
                * vec3(0.0, 1.0, 0.0);
        }
        self.size = player_stats[player_idx].size;
        if let Some(HealthSection::Health(current, max)) = self.health.get_mut(0).as_mut() {
            *max = player_stats[player_idx].max_health;
            *current = current.min(*max);
        }

        if self.respawn_timer > 0.0 {
            self.respawn_timer -= time_step;
            if self.respawn_timer <= 0.0 {
                self.pos = game_mode.spawn_location(&self);
                self.vel = Vector3::new(0.0, 0.0, 0.0);
                self.health = vec![HealthSection::Health(
                    player_stats[player_idx].max_health,
                    player_stats[player_idx].max_health,
                )];
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
            let mut speed_multiplier = 1.0;
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
                move_vec += Vector3::new(0.0, 0.25, 0.0);
            }
            if action.crouch {
                move_vec -= Vector3::new(0.0, 0.6, 0.0);
                speed_multiplier = 0.5;
            }
            if move_vec.magnitude() > 0.0 {
                move_vec = move_vec.normalize();
            }
            self.movement_direction = move_vec;
            let accel_speed = speed_multiplier
                * if self.collision_vec != Vector3::new(0, 0, 0) {
                    80.0
                } else {
                    18.0
                };
            self.vel += accel_speed
                * Vector3::new(
                    player_stats[player_idx].speed,
                    1.0,
                    player_stats[player_idx].speed,
                )
                .mul_element_wise(move_vec)
                * time_step;

            if action.jump {
                self.vel += player_stats[player_idx].speed
                    * self
                        .collision_vec
                        .zip(Vector3::new(0.3, 13.0, 0.3), |c, m| c as f32 * m);
            }

            for (cooldown_idx, cooldown) in self.abilities.iter_mut().enumerate() {
                if cooldown.remaining_charges > 0 && cooldown.recovery <= 0.0 {
                    for (ability_idx, ability) in cooldown.ability.abilities.iter().enumerate() {
                        if *action
                            .activate_ability
                            .get(cooldown_idx)
                            .map(|cd| cd.get(ability_idx).unwrap_or(&false))
                            .unwrap_or(&false)
                            && !player_stats[player_idx].lockout
                        {
                            cooldown.remaining_charges -= 1;
                            cooldown.recovery = cooldown.value.1[ability_idx];
                            let (proj_effects, vox_effects, effects, status_effects, triggers) =
                                card_manager.get_effects_from_base_card(
                                    ability.0,
                                    &self.pos,
                                    &self.rot,
                                    player_idx as u32,
                                    true,
                                );
                            new_effects.new_projectiles.extend(proj_effects);
                            for (pos, material) in vox_effects {
                                new_effects
                                    .voxels_to_write
                                    .push((pos, material.to_memory()));
                            }
                            for effect in effects {
                                new_effects.new_effects.push((
                                    player_idx, player_idx, false, self.pos, self.dir, effect,
                                ));
                            }
                            for status_effects in status_effects {
                                new_effects
                                    .new_status_effects
                                    .push((player_idx, status_effects));
                            }
                            new_effects.step_triggers.extend(triggers);
                            break;
                        }
                    }
                }
            }
        }
        for ability in self.abilities.iter_mut() {
            if ability.ability.is_reloading {
                if ability.remaining_charges == 0 {
                    ability.cooldown = 0.0;
                    ability.remaining_charges = ability.ability.max_charges;
                    ability.recovery += ability.value.0;
                }
            } else {
                if ability.cooldown > 0.0 && ability.remaining_charges < ability.ability.max_charges
                {
                    ability.cooldown -= time_step;
                } else if ability.remaining_charges < ability.ability.max_charges {
                    ability.cooldown = ability.value.0;
                    ability.remaining_charges += 1;
                }
            }
            if ability.recovery > 0.0 {
                ability.recovery -= time_step;
            }
        }

        self.hitmarker.0 -= 3.0 * (10.0 + self.hitmarker.0) * time_step;
        self.hitmarker.1 -= 3.0 * (10.0 + self.hitmarker.1) * time_step;
        self.hitmarker.0 = self.hitmarker.0.max(0.0);
        self.hitmarker.1 = self.hitmarker.1.max(0.0);
        for hurtmarker in self.hurtmarkers.iter_mut() {
            hurtmarker.2 -= time_step;
        }
        self.hurtmarkers.retain(|hurtmarker| hurtmarker.2 > 0.0);

        //volume effects
        let start_pos =
            self.pos + self.size * PLAYER_HITBOX_OFFSET - self.size * PLAYER_HITBOX_SIZE / 2.0;
        let end_pos =
            self.pos + self.size * PLAYER_HITBOX_OFFSET + self.size * PLAYER_HITBOX_SIZE / 2.0;
        let start_voxel_pos = start_pos.map(|c| c.floor() as u32);
        let iter_counts = end_pos.zip(start_voxel_pos, |a, b| a.floor() as u32 - b + 1);
        let mut nearby_density = 0.0;
        let mut directional_density = Vector3::new(0.0, 0.0, 0.0);
        for x in 0..iter_counts.x {
            for y in 0..iter_counts.y {
                for z in 0..iter_counts.z {
                    let voxel_pos = start_voxel_pos + Vector3::new(x, y, z);
                    let overlapping_volume = voxel_pos.zip(end_pos, |a, b| b.min(a as f32 + 1.0))
                        - voxel_pos.zip(start_pos, |a, b| b.max(a as f32));
                    let overlapping_volume =
                        overlapping_volume.x * overlapping_volume.y * overlapping_volume.z;
                    let material = if is_inbounds(voxel_pos, game_state, game_settings) {
                        let idx = get_index(voxel_pos, cpu_chunks, game_state, game_settings);
                        if let Some(idx) = idx {
                            VoxelMaterial::from_memory(voxel_reader[idx as usize])
                        } else {
                            VoxelMaterial::Unloaded
                        }
                    } else {
                        VoxelMaterial::Unloaded
                    };
                    let density = material.density();
                    nearby_density += overlapping_volume * density;
                    directional_density += overlapping_volume
                        * density
                        * (voxel_pos.map(|c| c as f32 + 0.5)
                            - (self.pos + self.size * PLAYER_HITBOX_OFFSET))
                        / self.size;
                }
            }
        }
        nearby_density /=
            self.size.powi(3) * PLAYER_HITBOX_SIZE.x * PLAYER_HITBOX_SIZE.y * PLAYER_HITBOX_SIZE.z;
        directional_density /=
            self.size.powi(3) * PLAYER_HITBOX_SIZE.x * PLAYER_HITBOX_SIZE.y * PLAYER_HITBOX_SIZE.z;

        self.vel += (PLAYER_DENSITY - nearby_density)
            * player_stats[player_idx].gravity
            * 11.428571428571429
            * time_step;
        if directional_density.magnitude() * time_step > 0.001 {
            self.vel -= 0.5 * directional_density * time_step;
        }
        if self.vel.magnitude() > 0.0 {
            self.vel -= nearby_density * 0.0375 * self.vel * self.vel.magnitude() * time_step
                + 0.2 * self.vel.normalize() * time_step;
        }
        let prev_collision_vec = self.collision_vec.clone();
        self.collision_vec = Vector3::new(0, 0, 0);
        self.collide_player(
            time_step,
            voxel_reader,
            cpu_chunks,
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
        cpu_chunks: &Vec<Vec<Vec<u32>>>,
        prev_collision_vec: Vector3<i32>,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) {
        let collision_corner_offset = self
            .vel
            .map(|c| c.signum())
            .zip(PLAYER_HITBOX_SIZE, |a, b| a * b)
            * 0.5
            * self.size;
        let mut distance_to_move = self.vel * time_step;
        let mut iteration_counter = 0;

        while distance_to_move.magnitude() > 0.0 {
            iteration_counter += 1;

            let player_move_pos =
                self.pos + PLAYER_HITBOX_OFFSET * self.size + collision_corner_offset;
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
            } else if dist_diff > distance_to_move.magnitude() {
                self.pos += distance_to_move;
                break;
            }

            if iteration_counter > 100 {
                println!(
                    "iteration counter exceeded with dtm {:?} and delta {:?}",
                    distance_to_move, delta
                );
                break;
            }

            distance_to_move -= dist_diff * vel_dir;
            'component_loop: for component in 0..3 {
                let mut fake_pos = self.pos;
                fake_pos[component] += dist_diff * vel_dir[component];
                let player_move_pos =
                    fake_pos + PLAYER_HITBOX_OFFSET * self.size + collision_corner_offset;
                if delta[component] <= delta[(component + 1) % 3]
                    && delta[component] <= delta[(component + 2) % 3]
                {
                    let mut start_pos = fake_pos + PLAYER_HITBOX_OFFSET * self.size
                        - 0.5 * self.size * PLAYER_HITBOX_SIZE;
                    start_pos[component] = player_move_pos[component];
                    let x_iter_count = (start_pos[(component + 1) % 3]
                        + self.size * PLAYER_HITBOX_SIZE[(component + 1) % 3])
                        .floor()
                        - (start_pos[(component + 1) % 3]).floor();
                    let z_iter_count = (start_pos[(component + 2) % 3]
                        + self.size * PLAYER_HITBOX_SIZE[(component + 2) % 3])
                        .floor()
                        - (start_pos[(component + 2) % 3]).floor();

                    let mut x_vec = Vector3::new(0.0, 0.0, 0.0);
                    let mut z_vec = Vector3::new(0.0, 0.0, 0.0);
                    x_vec[(component + 1) % 3] = 1.0;
                    z_vec[(component + 2) % 3] = 1.0;
                    for x_iter in 0..=(x_iter_count as u32) {
                        for z_iter in 0..=(z_iter_count as u32) {
                            let pos = start_pos + x_iter as f32 * x_vec + z_iter as f32 * z_vec;
                            let voxel_pos = pos.map(|c| c.floor() as u32);
                            let voxel = if let Some(index) =
                                get_index(voxel_pos, cpu_chunks, game_state, game_settings)
                            {
                                voxel_reader[index as usize]
                            } else {
                                VoxelMaterial::Unloaded.to_memory()
                            };
                            let voxel_material = VoxelMaterial::from_memory(voxel);
                            if !voxel_material.is_passthrough() {
                                if component != 1
                                    && prev_collision_vec[1] == 1
                                    && (pos - start_pos).y < 1.0
                                    && self.can_step_up(
                                        voxel_reader,
                                        cpu_chunks,
                                        component,
                                        player_move_pos,
                                        game_state,
                                        game_settings,
                                    )
                                {
                                    self.pos = fake_pos;
                                    self.pos[1] += 1.0;
                                    continue 'component_loop;
                                }

                                self.vel[component] = 0.0;
                                // apply friction
                                let perp_vel = Vector2::new(
                                    self.vel[(component + 1) % 3],
                                    self.vel[(component + 2) % 3],
                                );
                                if perp_vel.magnitude() > 0.0 {
                                    let friction_factor = voxel_material.get_friction();
                                    let friction = Vector2::new(
                                        (friction_factor * 0.5 * perp_vel.normalize().x
                                            + friction_factor * perp_vel.x)
                                            * time_step,
                                        (friction_factor * 0.5 * perp_vel.normalize().y
                                            + friction_factor * perp_vel.y)
                                            * time_step,
                                    );
                                    if friction.magnitude() > perp_vel.magnitude() {
                                        self.vel[(component + 1) % 3] = 0.0;
                                        self.vel[(component + 2) % 3] = 0.0;
                                    } else {
                                        self.vel[(component + 1) % 3] -= friction.x;
                                        self.vel[(component + 2) % 3] -= friction.y;
                                    }
                                }
                                self.collision_vec[component] = -vel_dir[component].signum() as i32;
                                distance_to_move[component] = 0.0;
                                continue 'component_loop;
                            }
                        }
                    }
                }
                self.pos = fake_pos;
            }
        }
    }

    fn can_step_up(
        &self,
        voxel_reader: &BufferReadGuard<'_, [u32]>,
        cpu_chunks: &Vec<Vec<Vec<u32>>>,
        component: usize,
        player_move_pos: Point3<f32>,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) -> bool {
        let mut start_pos =
            self.pos + PLAYER_HITBOX_OFFSET * self.size - 0.5 * self.size * PLAYER_HITBOX_SIZE;
        start_pos[component] = player_move_pos[component];
        start_pos[1] += 1.0;
        let x_iter_count = (start_pos[(component + 1) % 3]
            + self.size * PLAYER_HITBOX_SIZE[(component + 1) % 3])
            .floor()
            - (start_pos[(component + 1) % 3]).floor();
        let z_iter_count = (start_pos[(component + 2) % 3]
            + self.size * PLAYER_HITBOX_SIZE[(component + 2) % 3])
            .floor()
            - (start_pos[(component + 2) % 3]).floor();

        let mut x_vec = Vector3::new(0.0, 0.0, 0.0);
        let mut z_vec = Vector3::new(0.0, 0.0, 0.0);
        x_vec[(component + 1) % 3] = 1.0;
        z_vec[(component + 2) % 3] = 1.0;
        for x_iter in 0..=(x_iter_count as u32) {
            for z_iter in 0..=(z_iter_count as u32) {
                let pos = start_pos + x_iter as f32 * x_vec + z_iter as f32 * z_vec;
                let voxel_pos = pos.map(|c| c.floor() as u32);
                let voxel = if let Some(index) =
                    get_index(voxel_pos, cpu_chunks, game_state, game_settings)
                {
                    voxel_reader[index as usize]
                } else {
                    VoxelMaterial::Unloaded.to_memory()
                };
                let voxel_material = VoxelMaterial::from_memory(voxel);
                if !voxel_material.is_passthrough() {
                    return false;
                }
            }
        }
        true
    }

    pub fn adjust_health(&mut self, adjustment: f32) {
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

    fn get_effect_stats(&self) -> PlayerEffectStats {
        let mut speed = 1.0;
        let mut damage_taken = 1.0;
        let mut gravity = Vector3::new(0.0, -1.0, 0.0);
        let mut size = 1.0;
        let mut max_health = PLAYER_BASE_MAX_HEALTH;
        let mut invincible = false;
        let mut lockout = false;

        for status_effect in self
            .status_effects
            .iter()
            .map(|e| &e.effect)
            .chain(self.passive_abilities.iter())
        {
            match status_effect {
                ReferencedStatusEffect::DamageOverTime(_) => {
                    // wait for damage taken to be calculated
                }
                ReferencedStatusEffect::Speed(stacks) => {
                    speed *=
                        StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::Speed, *stacks)
                            .get_effect_value();
                }
                ReferencedStatusEffect::IncreaseDamageTaken(stacks) => {
                    damage_taken *= StatusEffect::SimpleStatusEffect(
                        SimpleStatusEffectType::IncreaseDamageTaken,
                        *stacks,
                    )
                    .get_effect_value();
                }
                ReferencedStatusEffect::IncreaseGravity(direction, stacks) => {
                    gravity += StatusEffect::SimpleStatusEffect(
                        SimpleStatusEffectType::IncreaseGravity(direction.clone()),
                        *stacks,
                    )
                    .get_effect_value()
                        * match direction {
                            DirectionCard::Forward => self.dir,
                            DirectionCard::Up => Vector3::new(0.0, 1.0, 0.0),
                            DirectionCard::Movement => {
                                if self.movement_direction.magnitude() == 0.0 {
                                    Vector3::new(0.0, 0.0, 0.0)
                                } else {
                                    self.movement_direction.normalize()
                                }
                            }
                            DirectionCard::None => Vector3::new(0.0, 0.0, 0.0),
                        };
                }
                ReferencedStatusEffect::Overheal(_) => {
                    // managed seperately
                }
                ReferencedStatusEffect::Grow(stacks) => {
                    size *= StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::Grow, *stacks)
                        .get_effect_value();
                }
                ReferencedStatusEffect::IncreaseMaxHealth(stacks) => {
                    max_health += StatusEffect::SimpleStatusEffect(
                        SimpleStatusEffectType::IncreaseMaxHealth,
                        *stacks,
                    )
                    .get_effect_value();
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

        if size > 5.0 {
            size = 5.0;
        }

        PlayerEffectStats {
            speed,
            damage_taken,
            gravity,
            size,
            max_health,
            invincible,
            lockout,
        }
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
