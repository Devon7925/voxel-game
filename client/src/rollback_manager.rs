use core::panic;
use std::{
    collections::{HashMap, VecDeque},
    fs::{self, File},
    io::{BufReader, Lines, Write},
    sync::Arc,
};

use bytemuck::{Pod, Zeroable};
use cgmath::{
    Point3, Quaternion, Vector2,
};
use itertools::Itertools;
use matchbox_socket::{PeerId, PeerState};
use serde::{Deserialize, Serialize};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};
use winit::event::{ElementState, WindowEvent};

use crate::{
    card_system::{
        CardManager, Deck, ReferencedStatusEffect, StateKeybind,
    }, cpu_simulation::{Entity, HealthSection, PlayerAbility, WorldState}, game_manager::GameState, game_modes::GameMode, gui::{GuiElement, GuiState}, networking::{NetworkConnection, NetworkPacket}, settings_manager::{Control, Settings}, voxel_sim_manager::{Collision, Projectile, VoxelComputePipeline}, WindowProperties
};
use voxel_shared::{GameSettings, RoomId};

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
        collisions: Vec<Collision>,
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
        game_mode: &Box<dyn GameMode>,
    ) -> Vec<Collision>;

    fn get_camera(&self) -> Camera;
    fn get_spectate_player(&self) -> Option<Entity>;

    fn get_delta_time(&self) -> f32;
    fn get_rollback_projectiles(&self) -> &Vec<Projectile>;
    fn get_rollback_players(&self) -> Vec<UploadPlayer>;
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

    fn add_player(&mut self, deck: &Deck, card_manager: &mut CardManager, game_mode: &dyn GameMode, entity_metadata: EntityMetaData);
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

impl From<&Entity> for UploadPlayer {
    fn from(player: &Entity) -> Self {
        UploadPlayer {
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
        }
    }
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
        collisions: Vec<Collision>,
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
            Some(collisions)
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
                player_buffer[i] = player.into();
            }
        }
    }

    fn download_projectiles(
        &mut self,
        card_manager: &CardManager,
        vox_compute: &mut VoxelComputePipeline,
        game_settings: &GameSettings,
        game_mode: &Box<dyn GameMode>,
    ) -> Vec<Collision> {
        let (new_projectiles, collisions) =
            vox_compute.download_projectiles(card_manager, game_settings, &mut self.rollback_state, game_mode);
        self.rollback_state.projectiles = new_projectiles;
        collisions
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

    fn get_rollback_players(&self) -> Vec<UploadPlayer> {
        self.rollback_state.players.iter().map(|p| p.into()).collect_vec()
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
        _settings: &GameSettings,
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

                    self.add_player(&cards, card_manager, game_mode.as_ref(), EntityMetaData::Player(
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
    
    fn add_player(&mut self, deck: &Deck, card_manager: &mut CardManager, game_mode: &dyn GameMode, entity_metadata: EntityMetaData) {
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
        self.rollback_state.players.push(new_player);
        self.entity_metadata.push(entity_metadata);
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

        let rollback_state = WorldState::new();
        let entity_metadata = Vec::new();

        let network_connection =
            lobby_id.map(|lobby_id| NetworkConnection::new(settings, &game_settings, lobby_id));

        let mut result = RollbackData {
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
            controls: vec![],
            player_action: PlayerAction::default(),
            player_deck: deck.clone(),
            player_idx_map: HashMap::new(),
            most_future_time_recorded: 0,
            connected_player_count: 1,
            exit_reason: None,
        };
        result.add_player(deck, card_manager, game_mode.as_ref(), EntityMetaData::Player(
            deck.clone(),
            VecDeque::from(vec![
                Action::empty();
                (result.current_time - result.rollback_time + 15) as usize
            ]),
        ));

        let first_player = result.rollback_state.players.get(0).unwrap();
        let controls: Vec<Vec<StateKeybind>> = first_player
            .abilities
            .iter()
            .map(|a| {
                a.ability
                    .abilities
                    .iter()
                    .map(|a| StateKeybind::from(a.1.clone()))
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

        result.controls = controls;
        result.player_action = player_action;

        result
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
                None,
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
        collisions: Vec<Collision>,
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
            Some(collisions),
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
        game_mode: &Box<dyn GameMode>,
    ) -> Vec<Collision> {
        let (new_projectiles, collisions) =
            vox_compute.download_projectiles(card_manager, game_settings, &mut self.state, game_mode);
        self.state.projectiles = new_projectiles;
        collisions
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

    fn get_rollback_players(&self) -> Vec<UploadPlayer> {
        self.state.players.iter().map(|p| p.into()).collect_vec()
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
    
    fn add_player(&mut self, deck: &Deck, card_manager: &mut CardManager, game_mode: &dyn GameMode, entity_metadata: EntityMetaData) {
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
        self.state.players.push(new_player);
        self.entity_metadata.push(entity_metadata);
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

        let state = WorldState::new();
        let entity_metadata = Vec::new();
        let mut actions = VecDeque::new();
        let mut decks = vec![];
        while let Some(line) = replay_lines.next() {
            let Ok(line) = line else {
                continue;
            };
            if let Some(_game_settings_string) = line.strip_prefix("GAME SETTINGS ") {
                panic!("Game settings should have already been handled")
            } else if let Some(deck_string) = line.strip_prefix("PLAYER DECK ") {
                let deck: Deck = ron::de::from_str(deck_string).unwrap();
                decks.push(deck);
            } else if let Some(_time_stamp_string) = line.strip_prefix("TIME ") {
                let actions_string = replay_lines.next().unwrap().unwrap();
                let line_actions: Vec<Action> = ron::de::from_str(&actions_string).unwrap();
                actions.push_back(line_actions);
            }
        }

        println!("Loaded replay with {} actions", actions.len());

        let mut result = ReplayData {
            current_time,
            delta_time: game_settings.delta_time,
            state,
            entity_metadata,
            actions,
            player_buffer,
            projectile_buffer,
        };
        for deck in decks {
            result.add_player(&deck, card_manager, game_mode.as_ref(), EntityMetaData::Player(deck.clone(), VecDeque::new()));
        }
        result
    }
}