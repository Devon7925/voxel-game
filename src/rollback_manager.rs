use std::{
    collections::{HashMap, VecDeque},
    f32::consts::PI,
    fs::File,
    io::{BufRead, BufReader, Write},
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
        BaseCard, CardManager, ReferencedBaseCard, ReferencedBaseCardType, ReferencedEffect,
        ReferencedStatusEffect, ReferencedTrigger, VoxelMaterial,
    },
    gui::{GuiElement, GuiState},
    networking::{NetworkConnection, NetworkPacket},
    projectile_sim_manager::{Projectile, ProjectileComputePipeline},
    settings_manager::{Control, ReplayMode, Settings},
    voxel_sim_manager::VoxelComputePipeline,
    WindowProperties, CHUNK_SIZE, PLAYER_HITBOX_OFFSET, PLAYER_HITBOX_SIZE, RENDER_SIZE,
    SPAWN_LOCATION,
};

#[derive(Clone, Debug)]
pub struct WorldState {
    pub players: Vec<Player>,
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
    );

    fn get_current_state(&self) -> &WorldState;

    fn step(
        &mut self,
        card_manager: &mut CardManager,
        time_step: f32,
        vox_compute: &mut VoxelComputePipeline,
    );

    fn download_projectiles(
        &mut self,
        card_manager: &CardManager,
        projectile_compute: &ProjectileComputePipeline,
        vox_compute: &mut VoxelComputePipeline,
    );

    fn get_camera(&self) -> Camera;
    fn get_spectate_player(&self) -> Option<Player>;

    fn get_delta_time(&self) -> f32;
    fn get_projectiles(&self) -> &Vec<Projectile>;
    fn get_players(&self) -> &Vec<Player>;
    fn player_count(&self) -> usize;

    fn projectile_buffer(&self) -> Subbuffer<[Projectile; 1024]>;
    fn player_buffer(&self) -> Subbuffer<[UploadPlayer; 128]>;

    fn update(&mut self);

    fn process_event(
        &mut self,
        event: &winit::event::WindowEvent,
        settings: &Settings,
        gui_state: &mut GuiState,
        window_props: &WindowProperties,
    );
    fn end_frame(&mut self);

    fn is_sim_behind(&self) -> bool;
}

#[derive(Debug)]
pub struct RollbackData {
    pub current_time: u64,
    pub rollback_time: u64,
    pub delta_time: f32,
    pub rollback_state: WorldState,
    pub cached_current_state: WorldState,
    pub actions: VecDeque<Vec<Option<PlayerAction>>>,
    pub meta_actions: VecDeque<Vec<Option<MetaAction>>>,
    pub projectile_buffer: Subbuffer<[Projectile; 1024]>,
    pub player_buffer: Subbuffer<[UploadPlayer; 128]>,
    replay_file: Option<File>,
    network_connection: NetworkConnection,
    player_action: PlayerAction,
    player_deck: Vec<BaseCard>,
    player_idx_map: HashMap<PeerId, usize>,
    most_future_time_recorded: u64,
}

pub struct ReplayData {
    pub current_time: u64,
    pub delta_time: f32,
    pub state: WorldState,
    pub actions: VecDeque<Vec<Option<PlayerAction>>>,
    pub meta_actions: VecDeque<Vec<Option<MetaAction>>>,
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
    pub activate_ability: Vec<bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MetaAction {
    pub adjust_dt: Option<f32>,
    pub deck_update: Option<Vec<BaseCard>>,
}

#[derive(Clone, Debug)]
pub struct Player {
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
    pub ability: ReferencedBaseCard,
    pub value: f32,
    pub cooldown: f32,
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
            adjust_dt: None,
            deck_update: None,
        }
    }
}

impl Default for Player {
    fn default() -> Self {
        Player {
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

impl PlayerSim for RollbackData {
    fn update_rollback_state(
        &mut self,
        card_manager: &mut CardManager,
        time_step: f32,
        vox_compute: &mut VoxelComputePipeline,
    ) {
        if let Some(replay_file) = self.replay_file.as_mut() {
            write!(replay_file, "\nTIME {}", self.current_time).unwrap();
        }
        self.rollback_time += 1;
        {
            let meta_actions = self.meta_actions.pop_front().unwrap();
            if let Some(replay_file) = self.replay_file.as_mut() {
                replay_file.write_all(b"\n").unwrap();
                ron::ser::to_writer(replay_file, &meta_actions).unwrap();
            }
            for (player_idx, meta_action) in meta_actions.into_iter().enumerate() {
                if let Some(MetaAction {
                    adjust_dt,
                    deck_update,
                }) = meta_action
                {
                    if let Some(adjust_dt) = adjust_dt {
                        self.delta_time = self.delta_time.max(adjust_dt);
                    }
                    if let Some(new_deck) = deck_update {
                        self.rollback_state.players[player_idx].abilities = new_deck
                            .into_iter()
                            .map(|card| PlayerAbility {
                                value: card.get_cooldown(),
                                ability: card_manager.register_base_card(card),
                                cooldown: 0.0,
                            })
                            .collect();
                    }
                }
            }
        }
        if self.rollback_time < 50 {
            return;
        }
        {
            let player_actions = self.actions.pop_front().unwrap();
            assert!(
                player_actions.iter().all(|x| x.is_some()),
                "missing action at timestamp {}",
                self.rollback_time
            );
            if let Some(replay_file) = self.replay_file.as_mut() {
                replay_file.write_all(b"\n").unwrap();
                ron::ser::to_writer(replay_file, &player_actions).unwrap();
            }
            self.rollback_state.step_sim(
                &player_actions,
                true,
                card_manager,
                time_step,
                vox_compute,
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
    ) {
        self.send_action(self.player_action.clone(), 0, self.current_time);
        puffin::profile_function!();
        self.update_rollback_state(card_manager, time_step, vox_compute);
        self.current_time += 1;
        self.actions
            .push_back(vec![None; self.rollback_state.players.len()]);
        self.meta_actions
            .push_back(vec![None; self.rollback_state.players.len()]);
        self.cached_current_state = self.gen_current_state(card_manager, time_step, vox_compute);
        //send projectiles
        let projectile_count = 128.min(self.cached_current_state.projectiles.len());
        {
            let mut projectiles_buffer = self.projectile_buffer.write().unwrap();
            for i in 0..projectile_count {
                let projectile = self.cached_current_state.projectiles.get(i).unwrap();
                projectiles_buffer[i] = projectile.clone();
            }
        }
        //send players
        let player_count = 128.min(self.cached_current_state.players.len());
        {
            let mut player_buffer = self.player_buffer.write().unwrap();
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
    ) {
        self.rollback_state.projectiles =
            projectile_compute.download_projectiles(card_manager, vox_compute);
    }

    fn get_camera(&self) -> Camera {
        let player = self
            .get_spectate_player()
            .unwrap_or_else(|| Player::default());
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

    fn get_players(&self) -> &Vec<Player> {
        &self.rollback_state.players
    }

    fn player_count(&self) -> usize {
        self.rollback_state.players.len()
    }

    fn update(&mut self) {
        let packet_data = NetworkPacket::Action(self.current_time, self.player_action.clone());
        self.network_connection.queue_packet(packet_data);
        let (connection_changes, recieved_packets) =
            self.network_connection.network_update(self.player_count());
        for (peer, state) in connection_changes {
            match state {
                PeerState::Connected => {
                    println!("Peer joined: {:?}", peer);

                    self.player_idx_map.insert(peer, self.player_count());

                    self.player_join(Player {
                        pos: SPAWN_LOCATION,
                        ..Default::default()
                    });

                    {
                        let deck_packet =
                            NetworkPacket::DeckUpdate(self.current_time, self.player_deck.clone());
                        self.network_connection.send_packet(peer, deck_packet);
                    }
                    {
                        let dt_packet =
                            NetworkPacket::DeltatimeUpdate(self.current_time, self.delta_time);
                        self.network_connection.send_packet(peer, dt_packet);
                    }
                }
                PeerState::Disconnected => {
                    println!("Peer left: {:?}", peer);
                }
            }
        }

        for (peer, packet) in recieved_packets {
            let player_idx = self.player_idx_map.get(&peer).unwrap().clone();
            match packet {
                NetworkPacket::Action(time, action) => {
                    self.send_action(action, player_idx, time);
                }
                NetworkPacket::DeckUpdate(time, cards) => {
                    self.send_deck_update(cards, player_idx, time);
                }
                NetworkPacket::DeltatimeUpdate(time, delta_time) => {
                    self.send_dt_update(delta_time, player_idx, time);
                }
            }
        }

        let peers = self.player_idx_map.keys().collect();
        self.network_connection.send_packet_queue(peers);
    }

    fn player_buffer(&self) -> Subbuffer<[UploadPlayer; 128]> {
        self.player_buffer.clone()
    }

    fn projectile_buffer(&self) -> Subbuffer<[Projectile; 1024]> {
        self.projectile_buffer.clone()
    }

    fn get_spectate_player(&self) -> Option<Player> {
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
                for (ability_idx, ability_key) in settings.ability_controls.iter().enumerate() {
                    if let Control::Mouse(mouse_code) = ability_key {
                        if button == mouse_code {
                            self.player_action.activate_ability[ability_idx] =
                                state == &ElementState::Pressed;
                        }
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
                    for (ability_idx, ability_key) in settings.ability_controls.iter().enumerate() {
                        if let Control::Key(key_code) = ability_key {
                            if key == *key_code {
                                self.player_action.activate_ability[ability_idx] =
                                    input.state == ElementState::Pressed;
                            }
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
                                            self.network_connection.queue_packet(
                                                NetworkPacket::DeckUpdate(
                                                    self.current_time,
                                                    self.player_deck.clone(),
                                                ),
                                            );
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
}

impl RollbackData {
    pub fn new(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        settings: &Settings,
        deck: &Vec<BaseCard>,
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

        let mut replay_file = (settings.replay_settings.replay_mode == ReplayMode::Record)
            .then(|| std::fs::File::create(settings.replay_settings.replay_file.clone()).unwrap());

        if let Some(replay_file) = replay_file.as_mut() {
            write!(replay_file, "PLAYER COUNT {}\n", settings.player_count).unwrap();
            write!(replay_file, "PLAYER DECK ").unwrap();
            ron::ser::to_writer(replay_file, &deck).unwrap();
        }
        if let Some(replay_file) = replay_file.as_mut() {
            write!(replay_file, "\nPERSONAL DT {}", settings.delta_time).unwrap();
        }

        let current_time: u64 = 5;
        let rollback_time: u64 = 0;

        let mut rollback_state = WorldState::new();
        let mut actions = VecDeque::from(vec![
            Vec::new();
            (current_time - rollback_time + 15) as usize
        ]);
        let mut meta_actions = VecDeque::from(vec![
            Vec::new();
            (current_time - rollback_time + 15) as usize
        ]);

        let first_player = Player {
            pos: SPAWN_LOCATION,
            abilities: deck
                .iter()
                .map(|card| PlayerAbility {
                    value: card.get_cooldown(),
                    ability: card_manager.register_base_card(card.clone()),
                    cooldown: 0.0,
                })
                .collect(),
            ..Default::default()
        };
        rollback_state.players.push(first_player);
        actions.iter_mut().for_each(|x| {
            x.push(Some(PlayerAction {
                ..Default::default()
            }))
        });
        meta_actions.iter_mut().for_each(|x| x.push(None));

        let network_connection = NetworkConnection::new(settings);

        let player_action = PlayerAction {
            activate_ability: vec![false; settings.ability_controls.len()],
            ..Default::default()
        };

        RollbackData {
            current_time,
            rollback_time,
            delta_time: settings.delta_time,
            rollback_state,
            cached_current_state: WorldState::new(),
            actions,
            meta_actions,
            player_buffer,
            projectile_buffer,
            replay_file,
            network_connection,
            player_action,
            player_deck: deck.clone(),
            player_idx_map: HashMap::new(),
            most_future_time_recorded: 0,
        }
    }

    fn gen_current_state(
        &self,
        card_manager: &CardManager,
        time_step: f32,
        vox_compute: &mut VoxelComputePipeline,
    ) -> WorldState {
        let mut state = self.rollback_state.clone();
        for i in self.rollback_time..self.current_time {
            let actions = self
                .actions
                .get((i - self.rollback_time) as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "cannot access index {} in action deque of length {}",
                        i - self.rollback_time,
                        self.actions.len()
                    )
                });
            state.step_sim(actions, false, card_manager, time_step, vox_compute);
        }
        state
    }

    pub fn send_dt_update(&mut self, delta_time: f32, player_idx: usize, time_stamp: u64) {
        if time_stamp < self.rollback_time {
            println!(
                "cannot send dt update with timestamp {} when rollback time is {}",
                time_stamp, self.rollback_time
            );
            return;
        }
        let time_idx = time_stamp - self.rollback_time;
        let meta_actions_len = self.meta_actions.len();
        let meta_action_frame = self
            .meta_actions
            .get_mut(time_idx as usize)
            .unwrap_or_else(|| {
                panic!(
                "cannot access index {} in action deque of length {} on sending with timestamp {}",
                time_idx, meta_actions_len, time_stamp
            )
            });
        if meta_action_frame[player_idx].is_none() {
            meta_action_frame[player_idx] = Some(MetaAction {
                adjust_dt: Some(delta_time),
                ..Default::default()
            });
        } else {
            meta_action_frame[player_idx].as_mut().unwrap().adjust_dt = Some(delta_time);
        }
    }

    pub fn send_deck_update(
        &mut self,
        new_deck: Vec<BaseCard>,
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
        let meta_actions_len = self.meta_actions.len();
        let meta_action_frame = self
            .meta_actions
            .get_mut(time_idx as usize)
            .unwrap_or_else(|| {
                panic!(
                "cannot access index {} in action deque of length {} on sending with timestamp {}",
                time_idx, meta_actions_len, time_stamp
            )
            });
        if meta_action_frame[player_idx].is_none() {
            meta_action_frame[player_idx] = Some(MetaAction {
                deck_update: Some(new_deck),
                ..Default::default()
            });
        } else {
            meta_action_frame[player_idx].as_mut().unwrap().deck_update = Some(new_deck);
        }
    }

    pub fn send_action(&mut self, action: PlayerAction, player_idx: usize, time_stamp: u64) {
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
        let actions_len = self.actions.len();
        let action_frame = self.actions.get_mut(time_idx as usize).unwrap_or_else(|| {
            panic!(
                "cannot access index {} in action deque of length {} on sending with timestamp {}",
                time_idx, actions_len, time_stamp
            )
        });
        action_frame[player_idx] = Some(action);
    }

    pub fn player_join(&mut self, player: Player) {
        self.rollback_state.players.push(player);
        self.actions.iter_mut().for_each(|x| {
            x.push(Some(PlayerAction {
                ..Default::default()
            }))
        });
        self.meta_actions.iter_mut().for_each(|x| x.push(None));
    }
}

impl PlayerSim for ReplayData {
    fn update_rollback_state(
        &mut self,
        card_manager: &mut CardManager,
        time_step: f32,
        vox_compute: &mut VoxelComputePipeline,
    ) {
        self.current_time += 1;
        {
            let meta_actions = self.meta_actions.pop_front().unwrap();
            for (player_idx, meta_action) in meta_actions.into_iter().enumerate() {
                if let Some(MetaAction {
                    adjust_dt,
                    deck_update,
                }) = meta_action
                {
                    if let Some(adjust_dt) = adjust_dt {
                        self.delta_time = self.delta_time.max(adjust_dt);
                    }
                    if let Some(new_deck) = deck_update {
                        self.state.players[player_idx].abilities = new_deck
                            .into_iter()
                            .map(|card| PlayerAbility {
                                value: card.get_cooldown(),
                                ability: card_manager.register_base_card(card),
                                cooldown: 0.0,
                            })
                            .collect();
                    }
                }
            }
        }
        if self.current_time < 50 {
            return;
        }
        {
            let player_actions = self.actions.pop_front().unwrap();
            assert!(player_actions.iter().all(|x| x.is_some()));
            self.state
                .step_sim(&player_actions, true, card_manager, time_step, vox_compute);
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
    ) {
        puffin::profile_function!();
        self.update_rollback_state(card_manager, time_step, vox_compute);
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
    ) {
        self.state.projectiles = projectile_compute.download_projectiles(card_manager, vox_compute);
    }

    fn get_camera(&self) -> Camera {
        let player = self
            .get_spectate_player()
            .unwrap_or_else(|| Player::default());
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

    fn get_players(&self) -> &Vec<Player> {
        &self.state.players
    }

    fn player_count(&self) -> usize {
        self.state.players.len()
    }

    fn update(&mut self) {}

    fn player_buffer(&self) -> Subbuffer<[UploadPlayer; 128]> {
        self.player_buffer.clone()
    }

    fn projectile_buffer(&self) -> Subbuffer<[Projectile; 1024]> {
        self.projectile_buffer.clone()
    }

    fn get_spectate_player(&self) -> Option<Player> {
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
}

impl ReplayData {
    pub fn new(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        settings: &Settings,
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
        let replay_file =
            std::fs::File::open(settings.replay_settings.replay_file.clone()).unwrap();
        let reader = BufReader::new(replay_file);
        let mut actions = VecDeque::new();
        let mut meta_actions = VecDeque::new();
        let mut delta_time = settings.delta_time;
        let mut lines = reader.lines();
        while let Some(line) = lines.next() {
            let Ok(line) = line else {
                continue;
            };
            if line.starts_with("PLAYER COUNT ") {
            } else if let Some(deck_string) = line.strip_prefix("PLAYER DECK ") {
                let deck: Vec<BaseCard> = ron::de::from_str(deck_string).unwrap();
                state.players.push(Player {
                    pos: SPAWN_LOCATION,
                    abilities: deck
                        .iter()
                        .map(|card| PlayerAbility {
                            value: card.get_cooldown(),
                            ability: card_manager.register_base_card(card.clone()),
                            cooldown: 0.0,
                        })
                        .collect(),
                    ..Default::default()
                });
            } else if let Some(dt_string) = line.strip_prefix("PERSONAL DT ") {
                delta_time = dt_string.parse().unwrap();
            } else if let Some(time_stamp_string) = line.strip_prefix("TIME ") {
                let time_stamp: u64 = time_stamp_string.parse().unwrap();
                let meta_actions_string = lines.next().unwrap().unwrap();
                let line_meta_actions: Vec<Option<MetaAction>> =
                    ron::de::from_str(&meta_actions_string).unwrap();
                meta_actions.push_back(line_meta_actions);
                if time_stamp >= 54 {
                    let actions_string = lines.next().unwrap().unwrap();
                    let line_actions: Vec<Option<PlayerAction>> =
                        ron::de::from_str(&actions_string).unwrap();
                    actions.push_back(line_actions);
                }
            }
        }

        ReplayData {
            current_time,
            delta_time,
            state,
            actions,
            meta_actions,
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
        player_actions: &Vec<Option<PlayerAction>>,
        is_real_update: bool,
        card_manager: &CardManager,
        time_step: f32,
        vox_compute: &mut VoxelComputePipeline,
    ) {
        let voxels = vox_compute.voxels();
        let mut new_projectiles = Vec::new();
        let mut voxels_to_write: Vec<(Point3<i32>, [u32; 2])> = Vec::new();
        let mut new_effects: Vec<(usize, Point3<f32>, Vector3<f32>, ReferencedEffect)> = Vec::new();
        let mut step_triggers: Vec<(ReferencedTrigger, u32)> = Vec::new();

        let player_stats: Vec<PlayerEffectStats> = self
            .players
            .iter_mut()
            .map(|player| {
                let mut speed = 1.0;
                let mut damage_taken = 1.0;
                let mut gravity = 1.0;
                let mut invincible = false;

                for status_effect in player.status_effects.iter_mut() {
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
                        ReferencedStatusEffect::OnHit(_) => {
                            // managed seperately
                        }
                    }
                }
                let mut health_adjustment = 0.0;
                for status_effect in player.status_effects.iter_mut() {
                    match status_effect.effect {
                        ReferencedStatusEffect::DamageOverTime => {
                            health_adjustment += -10.0 * damage_taken * time_step;
                        }
                        ReferencedStatusEffect::HealOverTime => {
                            health_adjustment += 10.0 * damage_taken * time_step;
                        }
                        _ => {}
                    }
                    status_effect.time_left -= time_step;
                }
                if health_adjustment != 0.0 {
                    player.adjust_health(health_adjustment);
                }
                player.status_effects.retain(|x| x.time_left > 0.0);
                PlayerEffectStats {
                    speed,
                    damage_taken,
                    gravity,
                    invincible,
                }
            })
            .collect();

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

        {
            let voxel_reader = voxels.read().unwrap();
            for (player_idx, (player, action)) in self
                .players
                .iter_mut()
                .zip(player_actions.iter())
                .enumerate()
            {
                player.simple_step(
                    time_step,
                    action,
                    &player_stats,
                    player_idx,
                    card_manager,
                    &voxel_reader,
                    &mut new_projectiles,
                    &mut voxels_to_write,
                    &mut new_effects,
                    &mut step_triggers,
                );
            }
        }
        for (player_idx, player) in self.players.iter_mut().enumerate() {
            if player.respawn_timer > 0.0 || player_stats[player_idx].invincible {
                continue;
            }
            // check for collision with projectiles
            for proj in self.projectiles.iter_mut() {
                if player_idx as u32 == proj.owner
                    && proj.lifetime < 1.0
                    && player
                        .abilities
                        .iter()
                        .map(|a| &a.ability)
                        .filter(|a| a.card_type == ReferencedBaseCardType::Projectile)
                        .any(|a| a.card_idx as u32 == proj.proj_card_idx)
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
                            let likely_hit = Player::HITSPHERES
                                .iter()
                                .min_by(|(offset_a, _), (offset_b, _)| {
                                    (player.pos + player.size * offset_a - pos)
                                        .magnitude()
                                        .total_cmp(
                                            &(player.pos + player.size * offset_b - pos)
                                                .magnitude(),
                                        )
                                })
                                .unwrap();

                            if (player.pos + player.size * likely_hit.0 - pos).magnitude()
                                > likely_hit.1 * player.size
                            {
                                continue;
                            }

                            if !proj_card.pierce_players {
                                proj.health = 0.0;
                            } else if player.player_piercing_invincibility > 0.0 {
                                continue;
                            } else {
                                player.player_piercing_invincibility = 0.3;
                            }
                            for card_ref in card_manager
                                .get_referenced_proj(proj.proj_card_idx as usize)
                                .on_hit
                                .clone()
                            {
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
            for j in 0..self.players.len() {
                if i == j {
                    continue;
                }
                let player2 = self.players.get(j).unwrap();
                if 5.0 * (player1.size + player2.size) > (player1.pos - player2.pos).magnitude() {
                    for si in 0..Player::HITSPHERES.len() {
                        for sj in 0..Player::HITSPHERES.len() {
                            let pos1 = player1.pos + player1.size * Player::HITSPHERES[si].0;
                            let pos2 = player2.pos + player2.size * Player::HITSPHERES[sj].0;
                            if (pos1 - pos2).magnitude()
                                < (Player::HITSPHERES[si].1 + Player::HITSPHERES[sj].1)
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
                let player1 = self.players.get(i).unwrap();
                player1
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
                        )
                    })
                    .collect::<Vec<_>>()
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
            for (pos, material) in voxels_to_write {
                vox_compute.queue_update_from_voxel_pos(&[pos.x, pos.y, pos.z]);
                writer[get_index(pos) as usize] = material;
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
                            player
                                .health
                                .push(HealthSection::Overhealth(10.0, duration as f32));
                        }
                        _ => {}
                    }
                    player.status_effects.push(AppliedStatusEffect {
                        effect,
                        time_left: duration as f32,
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
        players: &Vec<Player>,
        card_manager: &CardManager,
        time_step: f32,
        new_projectiles: &mut Vec<Projectile>,
        voxels_to_write: &mut Vec<(Point3<i32>, [u32; 2])>,
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

pub fn get_index(global_pos: Point3<i32>) -> i32 {
    const SIGNED_CHUNK_SIZE: i32 = CHUNK_SIZE as i32;
    let chunk_pos = (global_pos / SIGNED_CHUNK_SIZE)
        .zip(Point3::from(RENDER_SIZE).map(|c| c as i32), |a, b| a % b);
    let pos_in_chunk = global_pos % SIGNED_CHUNK_SIZE;
    let chunk_idx = chunk_pos.x * (RENDER_SIZE[1] as i32) * (RENDER_SIZE[2] as i32)
        + chunk_pos.y * (RENDER_SIZE[2] as i32)
        + chunk_pos.z;
    let idx_in_chunk = pos_in_chunk.x * SIGNED_CHUNK_SIZE * SIGNED_CHUNK_SIZE
        + pos_in_chunk.y * SIGNED_CHUNK_SIZE
        + pos_in_chunk.z;
    return chunk_idx * SIGNED_CHUNK_SIZE * SIGNED_CHUNK_SIZE * SIGNED_CHUNK_SIZE + idx_in_chunk;
}

impl Player {
    const HITSPHERES: [(Vector3<f32>, f32); 6] = [
        (Vector3::new(0.0, 0.0, 0.0), 0.6),
        (Vector3::new(0.0, -1.3, 0.0), 0.6),
        (Vector3::new(0.0, -1.9, 0.0), 0.9),
        (Vector3::new(0.0, -2.6, 0.0), 0.8),
        (Vector3::new(0.0, -3.3, 0.0), 0.6),
        (Vector3::new(0.0, -3.8, 0.0), 0.6),
    ];
    fn simple_step(
        &mut self,
        time_step: f32,
        action: &Option<PlayerAction>,
        player_stats: &Vec<PlayerEffectStats>,
        player_idx: usize,
        card_manager: &CardManager,
        voxel_reader: &BufferReadGuard<'_, [[u32; 2]]>,
        new_projectiles: &mut Vec<Projectile>,
        voxels_to_write: &mut Vec<(Point3<i32>, [u32; 2])>,
        new_effects: &mut Vec<(usize, Point3<f32>, Vector3<f32>, ReferencedEffect)>,
        step_triggers: &mut Vec<(ReferencedTrigger, u32)>,
    ) {
        if self.respawn_timer > 0.0 {
            self.respawn_timer -= time_step;
            if self.respawn_timer <= 0.0 {
                self.pos = SPAWN_LOCATION;
                self.health = vec![HealthSection::Health(100.0, 100.0)];
                self.status_effects.clear();
            }
            return;
        }
        if self.player_piercing_invincibility > 0.0 {
            self.player_piercing_invincibility -= time_step;
        }
        if let Some(action) = action {
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
                    42.0
                } else {
                    18.0
                };
            self.vel += accel_speed * move_vec * time_step;

            if action.jump {
                self.vel += self
                    .collision_vec
                    .zip(Vector3::new(0.3, 13.0, 0.3), |c, m| c as f32 * m);
            }

            for (ability_idx, ability) in self.abilities.iter_mut().enumerate() {
                if ability_idx < action.activate_ability.len()
                    && action.activate_ability[ability_idx]
                    && ability.cooldown <= 0.0
                {
                    ability.cooldown = ability.value;
                    let (proj_effects, vox_effects, effects, triggers) = card_manager
                        .get_effects_from_base_card(
                            ability.ability,
                            &self.pos,
                            &self.rot,
                            player_idx as u32,
                        );
                    new_projectiles.extend(proj_effects);
                    for (pos, material) in vox_effects {
                        voxels_to_write.push((pos, material.to_memory()));
                    }
                    for effect in effects {
                        new_effects.push((player_idx, self.pos, self.dir, effect));
                    }
                    step_triggers.extend(triggers);
                }
            }
        }
        for ability in self.abilities.iter_mut() {
            ability.cooldown -= time_step;
        }
        self.vel.y -= player_stats[player_idx].gravity * 32.0 * time_step;
        if self.vel.magnitude() > 0.0 {
            self.vel -= 0.1 * self.vel * self.vel.magnitude() * time_step
                + 0.2 * self.vel.normalize() * time_step;
        }
        let prev_collision_vec = self.collision_vec.clone();
        self.collision_vec = Vector3::new(0, 0, 0);
        self.collide_player(time_step, voxel_reader, prev_collision_vec);

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
        voxel_reader: &BufferReadGuard<'_, [[u32; 2]]>,
        prev_collision_vec: Vector3<i32>,
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
                    'outer: for x_iter in 0..=(x_iter_count as i32) {
                        for z_iter in 0..=(z_iter_count as i32) {
                            let pos = start_pos
                                + x_dist * x_iter as f32 * x_vec
                                + z_dist * z_iter as f32 * z_vec;
                            let voxel_pos = pos.map(|c| c.floor() as i32);
                            let voxel = voxel_reader[get_index(voxel_pos) as usize];
                            if voxel[0] != 0 {
                                if component != 1
                                    && prev_collision_vec[1] == 1
                                    && (pos - start_pos).y < 1.0
                                    && self.can_step_up(voxel_reader, component, player_move_pos)
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
                                    let friction_factor =
                                        VoxelMaterial::FRICTION_COEFFICIENTS[voxel[0] as usize];
                                    self.vel[(component + 1) % 3] -= (0.5 * perp_vel.normalize().x
                                        + friction_factor * perp_vel.x)
                                        * time_step;
                                    self.vel[(component + 2) % 3] -= (0.5 * perp_vel.normalize().y
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
        voxel_reader: &BufferReadGuard<'_, [[u32; 2]]>,
        component: usize,
        player_move_pos: Point3<f32>,
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
        for x_iter in 0..=(x_iter_count as i32) {
            for z_iter in 0..=(z_iter_count as i32) {
                let pos =
                    start_pos + x_dist * x_iter as f32 * x_vec + z_dist * z_iter as f32 * z_vec;
                let voxel_pos = pos.map(|c| c.floor() as i32);
                let voxel = voxel_reader[get_index(voxel_pos) as usize];
                if voxel[0] != 0 {
                    return false;
                }
            }
        }
        true
    }

    fn adjust_health(&mut self, adjustment: f32) {
        println!("adjusting health by {}", adjustment);
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
