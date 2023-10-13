use std::{collections::VecDeque, f32::consts::PI, fs::File, io::Write, sync::Arc};

use bytemuck::{Pod, Zeroable};
use cgmath::{
    EuclideanSpace, InnerSpace, One, Point3, Quaternion, Rad, Rotation, Rotation3, Vector2, Vector3,
};
use serde::{Deserialize, Serialize};
use vulkano::{
    buffer::{subbuffer::BufferReadGuard, Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
};

use crate::{
    card_system::{
        BaseCard, CardManager, Effect, ReferencedBaseCard, ReferencedBaseCardType, StatusEffect,
        VoxelMaterial,
    },
    projectile_sim_manager::{Projectile, ProjectileComputePipeline},
    settings_manager::Settings,
    voxel_sim_manager::VoxelComputePipeline,
    CHUNK_SIZE, PLAYER_HITBOX_OFFSET, PLAYER_HITBOX_SIZE, RENDER_SIZE, SPAWN_LOCATION,
};

#[derive(Clone, Debug)]
pub struct WorldState {
    pub players: Vec<Player>,
    pub projectiles: Vec<Projectile>,
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
    pub health: f32,
    pub abilities: Vec<PlayerAbility>,
    pub respawn_timer: f32,
    pub collision_vec: Vector3<i32>,
    pub status_effects: Vec<AppliedStatusEffect>,
}

#[derive(Clone, Debug)]
pub struct AppliedStatusEffect {
    pub effect: StatusEffect,
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
            health: 100.0,
            abilities: Vec::new(),
            respawn_timer: 0.0,
            collision_vec: Vector3::new(0, 0, 0),
            status_effects: Vec::new(),
        }
    }
}

impl RollbackData {
    pub fn new(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        settings: &Settings,
        deck: &Vec<BaseCard>,
    ) -> Self {
        let projectile_buffer = Buffer::new_sized(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
        )
        .unwrap();

        let player_buffer = Buffer::new_sized(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
        )
        .unwrap();

        let mut replay_file = settings
            .replay_settings
            .record_replay
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

        RollbackData {
            current_time,
            rollback_time,
            delta_time: settings.delta_time,
            rollback_state: WorldState::new(),
            cached_current_state: WorldState::new(),
            actions: VecDeque::from(vec![
                Vec::new();
                (current_time - rollback_time + 15) as usize
            ]),
            meta_actions: VecDeque::from(vec![
                Vec::new();
                (current_time - rollback_time + 15) as usize
            ]),
            player_buffer,
            projectile_buffer,
            replay_file,
        }
    }

    pub fn projectiles(&self) -> Subbuffer<[Projectile; 1024]> {
        self.projectile_buffer.clone()
    }

    pub fn players(&self) -> Subbuffer<[UploadPlayer; 128]> {
        self.player_buffer.clone()
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
                                value: card.evaluate_value(true),
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
            assert!(player_actions.iter().all(|x| x.is_some()));
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

    fn get_current_state(
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

    pub fn step(
        &mut self,
        card_manager: &mut CardManager,
        time_step: f32,
        vox_compute: &mut VoxelComputePipeline,
    ) {
        puffin::profile_function!();
        self.update_rollback_state(card_manager, time_step, vox_compute);
        self.current_time += 1;
        self.actions
            .push_back(vec![None; self.rollback_state.players.len()]);
        self.meta_actions
            .push_back(vec![None; self.rollback_state.players.len()]);
        self.cached_current_state = self.get_current_state(card_manager, time_step, vox_compute);
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

    pub fn download_projectiles(
        &mut self,
        card_manager: &CardManager,
        projectile_compute: &ProjectileComputePipeline,
        vox_compute: &mut VoxelComputePipeline,
    ) {
        self.rollback_state.projectiles =
            projectile_compute.download_projectiles(card_manager, vox_compute);
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
        for proj in self.projectiles.iter_mut() {
            let projectile_rot =
                Quaternion::new(proj.dir[3], proj.dir[0], proj.dir[1], proj.dir[2]);
            let projectile_dir = projectile_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0));
            let mut proj_vel = projectile_dir * proj.vel;
            let proj_card = card_manager.get_referenced_proj(proj.proj_card_idx as usize);
            proj_vel.y -= proj_card.gravity * time_step;
            for i in 0..3 {
                proj.pos[i] += proj_vel[i] * time_step;
            }
            // recompute vel and rot
            let new_projectile_rot: Quaternion<f32> = if proj_vel.magnitude() < 0.0001 {
                projectile_rot
            } else {
                Quaternion::from_arc(projectile_dir, proj_vel.normalize(), None) * projectile_rot
            };
            proj.dir = [
                new_projectile_rot.v[0],
                new_projectile_rot.v[1],
                new_projectile_rot.v[2],
                new_projectile_rot.s,
            ];
            proj.vel = proj_vel.magnitude();

            proj.lifetime += time_step;
            if proj.lifetime >= proj_card.lifetime {
                proj.health = 0.0;
                for card_ref in card_manager
                    .get_referenced_proj(proj.proj_card_idx as usize)
                    .on_expiry
                    .clone()
                {
                    let proj_rot = proj.dir;
                    let proj_rot =
                        Quaternion::new(proj_rot[3], proj_rot[0], proj_rot[1], proj_rot[2]);
                    let effects = card_manager.get_effects_from_base_card(
                        card_ref,
                        &Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]),
                        &proj_rot,
                        proj.owner,
                    );
                    new_projectiles.extend(effects.0);
                    if is_real_update && effects.1.len() > 0 {
                        let mut writer = voxels.write().unwrap();
                        for (pos, material) in effects.1 {
                            vox_compute.queue_update_from_voxel_pos(&[pos.x, pos.y, pos.z]);
                            writer[get_index(pos) as usize] = material.to_memory();
                        }
                    }
                }
            }
        }
        //collide projectiles <-> projectiles
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

        let mut voxels_to_write: Vec<(i32, [u32; 2])> = Vec::new();

        // handle trails
        for proj in self.projectiles.iter() {
            let proj_card = card_manager.get_referenced_proj(proj.proj_card_idx as usize);
            for (trail_time, trail_card) in proj_card.trail.iter() {
                if proj.lifetime % trail_time >= trail_time - time_step {
                    let proj_rot = proj.dir;
                    let proj_rot = Quaternion::new(proj_rot[3], proj_rot[0], proj_rot[1], proj_rot[2]);
                    let effects = card_manager.get_effects_from_base_card(
                        trail_card.clone(),
                        &Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]),
                        &proj_rot,
                        proj.owner,
                    );
                    new_projectiles.extend(effects.0);
                    if is_real_update && effects.1.len() > 0 {
                        for (pos, material) in effects.1 {
                            vox_compute.queue_update_from_voxel_pos(&[
                                pos.x, pos.y, pos.z,
                            ]);
                            voxels_to_write
                                .push((get_index(pos), material.to_memory()));
                        }
                    }
                } 
            }
        }

        let mut player_stats = vec![];
        for player in self.players.iter_mut() {
            let mut player_speed = 1.0;
            let mut player_damage_taken = 1.0;

            for status_effect in player.status_effects.iter_mut() {
                match status_effect.effect {
                    StatusEffect::DamageOverTime => {
                        // wait for damage taken to be calculated
                    }
                    StatusEffect::HealOverTime => {
                        // wait for damage taken to be calculated
                    }
                    StatusEffect::Speed => {
                        player_speed *= 1.25;
                    }
                    StatusEffect::Slow => {
                        player_speed *= 0.75;
                    }
                    StatusEffect::IncreaceDamageTaken => {
                        player_damage_taken *= 1.25;
                    }
                    StatusEffect::DecreaceDamageTaken => {
                        player_damage_taken *= 0.75;
                    }
                }
            }
            for status_effect in player.status_effects.iter_mut() {
                match status_effect.effect {
                    StatusEffect::DamageOverTime => {
                        player.health -= 10.0 * player_damage_taken * time_step;
                    }
                    StatusEffect::HealOverTime => {
                        player.health += 10.0 * player_damage_taken * time_step;
                    }
                    _ => {}
                }
                status_effect.time_left -= time_step;
            }
            player.status_effects.retain(|x| x.time_left > 0.0);
            player_stats.push((player_speed, player_damage_taken));
        }
        {
            let voxel_reader = voxels.read().unwrap();
            for (player_idx, (player, action)) in self
                .players
                .iter_mut()
                .zip(player_actions.iter())
                .enumerate()
            {
                if player.respawn_timer > 0.0 {
                    player.respawn_timer -= time_step;
                    if player.respawn_timer <= 0.0 {
                        player.pos = SPAWN_LOCATION;
                        player.health = 100.0;
                        player.status_effects.clear();
                    }
                    continue;
                }

                if let Some(action) = action {
                    player.facing[0] = (player.facing[0] - action.aim[0] + 2.0 * PI) % (2.0 * PI);
                    player.facing[1] = (player.facing[1] - action.aim[1])
                        .min(PI / 2.0)
                        .max(-PI / 2.0);
                    player.rot = Quaternion::from_axis_angle(
                        Vector3::new(0.0, 1.0, 0.0),
                        Rad(player.facing[0]),
                    ) * Quaternion::from_axis_angle(
                        Vector3::new(1.0, 0.0, 0.0),
                        Rad(-player.facing[1]),
                    );
                    let horizontal_rot = Quaternion::from_axis_angle(
                        Vector3::new(0.0, 1.0, 0.0),
                        Rad(player.facing[0]),
                    );
                    player.dir = player.rot * Vector3::new(0.0, 0.0, 1.0);
                    player.right = player.rot * Vector3::new(-1.0, 0.0, 0.0);
                    player.up = player.right.cross(player.dir).normalize();
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
                    let accel_speed = player_stats[player_idx].0
                        * if player.collision_vec != Vector3::new(0, 0, 0) {
                            42.0
                        } else {
                            18.0
                        };
                    player.vel += accel_speed * move_vec * time_step;

                    if action.jump {
                        player.vel += player
                            .collision_vec
                            .zip(Vector3::new(0.3, 13.0, 0.3), |c, m| c as f32 * m);
                    }

                    for (ability_idx, ability) in player.abilities.iter_mut().enumerate() {
                        if ability_idx < action.activate_ability.len()
                            && action.activate_ability[ability_idx]
                            && ability.cooldown <= 0.0
                        {
                            ability.cooldown = ability.value;
                            let mut effects = card_manager.get_effects_from_base_card(
                                ability.ability,
                                &player.pos,
                                &player.rot,
                                player_idx as u32,
                            );
                            for proj in effects.0.iter_mut() {
                                let projectile_rot = Quaternion::new(
                                    proj.dir[3],
                                    proj.dir[0],
                                    proj.dir[1],
                                    proj.dir[2],
                                );
                                let projectile_dir =
                                    projectile_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0));
                                let mut proj_vel = projectile_dir * proj.vel;
                                proj_vel += player.vel;
                                // recompute vel and rot
                                let new_projectile_rot: Quaternion<f32> = Quaternion::from_arc(
                                    projectile_dir,
                                    proj_vel.normalize(),
                                    None,
                                ) * projectile_rot;
                                proj.dir = [
                                    new_projectile_rot.v[0],
                                    new_projectile_rot.v[1],
                                    new_projectile_rot.v[2],
                                    new_projectile_rot.s,
                                ];
                                proj.vel = proj_vel.magnitude();
                            }
                            self.projectiles.extend(effects.0);
                            if is_real_update && effects.1.len() > 0 {
                                let mut writer = voxels.write().unwrap();
                                for (pos, material) in effects.1 {
                                    vox_compute.queue_update_from_voxel_pos(&[pos.x, pos.y, pos.z]);
                                    writer[get_index(pos) as usize] = material.to_memory();
                                }
                            }
                            for effect in effects.2 {
                                match effect {
                                    Effect::Damage(damage) => {
                                        player.health -= damage as f32;
                                    }
                                    Effect::Knockback(knockback) => {
                                        let knockback = 10.0 * knockback as f32;
                                        player.vel += knockback * player.dir;
                                    }
                                    Effect::StatusEffect(effect, duration) => {
                                        player.status_effects.push(AppliedStatusEffect {
                                            effect,
                                            time_left: duration as f32,
                                        })
                                    }
                                }
                            }
                        }
                    }
                }
                for ability in player.abilities.iter_mut() {
                    ability.cooldown -= time_step;
                }

                player.vel.y -= 32.0 * time_step;
                if player.vel.magnitude() > 0.0 {
                    player.vel -= 0.1 * player.vel * player.vel.magnitude() * time_step
                        + 0.2 * player.vel.normalize() * time_step;
                }

                let prev_collision_vec = player.collision_vec.clone();
                player.collision_vec = Vector3::new(0, 0, 0);
                collide_player(player, time_step, &voxel_reader, prev_collision_vec);
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
                    let projectile_right =
                        projectile_rot.rotate_vector(Vector3::new(1.0, 0.0, 0.0));
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
                                let hitspheres = [
                                    (Vector3::new(0.0, 0.0, 0.0), 0.6),
                                    (Vector3::new(0.0, -1.3, 0.0), 0.6),
                                    (Vector3::new(0.0, -1.9, 0.0), 0.9),
                                    (Vector3::new(0.0, -2.6, 0.0), 0.8),
                                    (Vector3::new(0.0, -3.3, 0.0), 0.6),
                                    (Vector3::new(0.0, -3.8, 0.0), 0.6),
                                ];
                                let likely_hit = hitspheres
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
                                    < likely_hit.1 * player.size
                                {
                                    proj.health = 0.0;
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
                                        let (on_hit_projectiles, on_hit_voxels, effects) =
                                            card_manager.get_effects_from_base_card(
                                                card_ref,
                                                &Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]),
                                                &proj_rot,
                                                proj.owner,
                                            );
                                        new_projectiles.extend(on_hit_projectiles);
                                        if is_real_update && on_hit_voxels.len() > 0 {
                                            for (pos, material) in on_hit_voxels {
                                                vox_compute.queue_update_from_voxel_pos(&[
                                                    pos.x, pos.y, pos.z,
                                                ]);
                                                voxels_to_write
                                                    .push((get_index(pos), material.to_memory()));
                                            }
                                        }
                                        for effect in effects {
                                            match effect {
                                                Effect::Damage(damage) => {
                                                    player.health -=
                                                        player_stats[player_idx].1 * damage as f32;
                                                }
                                                Effect::Knockback(knockback) => {
                                                    let knockback = 10.0 * knockback as f32;
                                                    let knockback_dir = player.pos
                                                        + player.size * likely_hit.0
                                                        - projectile_pos;
                                                    if knockback_dir.magnitude() > 0.0 {
                                                        player.vel +=
                                                            knockback * (knockback_dir).normalize();
                                                    } else {
                                                        player.vel.y += knockback;
                                                    }
                                                }
                                                Effect::StatusEffect(effect, duration) => player
                                                    .status_effects
                                                    .push(AppliedStatusEffect {
                                                        effect,
                                                        time_left: duration as f32,
                                                    }),
                                            }
                                        }
                                    }
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
            }
        }
        for player in self.players.iter_mut() {
            if player.health <= 0.0 && player.respawn_timer <= 0.0 {
                player.respawn_timer = 5.0;
            } else if player.health > 100.0 {
                player.health = 100.0;
            }
        }
        // remove dead projectiles
        self.projectiles.retain(|proj| proj.health > 0.0);
        self.projectiles.extend(new_projectiles);
        {
            let mut writer = voxels.write().unwrap();
            for (index, material) in voxels_to_write {
                writer[index as usize] = material;
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

fn collide_player(
    player: &mut Player,
    time_step: f32,
    voxel_reader: &BufferReadGuard<'_, [[u32; 2]]>,
    prev_collision_vec: Vector3<i32>,
) {
    let mut player_move_pos = player.pos
        + PLAYER_HITBOX_OFFSET
        + player
            .vel
            .map(|c| c.signum())
            .zip(PLAYER_HITBOX_SIZE, |a, b| a * b)
            * 0.5
            * player.size;
    let mut distance_to_move = player.vel * time_step;
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
            player.pos += distance_to_move;
            player_move_pos += distance_to_move;
            break;
        }

        distance_to_move -= dist_diff * vel_dir;
        player.pos += dist_diff * vel_dir;
        player_move_pos += dist_diff * vel_dir;
        for component in 0..3 {
            if delta[component] <= delta[(component + 1) % 3]
                && delta[component] <= delta[(component + 2) % 3]
            {
                // neccessary because otherwise side plane could hit on ground to prevent walking
                // however this allows clipping when corners would collide
                const HITBOX_SHRINK_FACTOR: f32 = 0.999;
                let x_iter_count =
                    (HITBOX_SHRINK_FACTOR * player.size * PLAYER_HITBOX_SIZE[(component + 1) % 3])
                        .ceil()
                        + 1.0;
                let z_iter_count =
                    (HITBOX_SHRINK_FACTOR * player.size * PLAYER_HITBOX_SIZE[(component + 2) % 3])
                        .ceil()
                        + 1.0;
                let x_dist =
                    (HITBOX_SHRINK_FACTOR * player.size * PLAYER_HITBOX_SIZE[(component + 1) % 3])
                        / x_iter_count;
                let z_dist =
                    (HITBOX_SHRINK_FACTOR * player.size * PLAYER_HITBOX_SIZE[(component + 2) % 3])
                        / z_iter_count;
                let mut start_pos = player.pos + PLAYER_HITBOX_OFFSET
                    - HITBOX_SHRINK_FACTOR * 0.5 * player.size * PLAYER_HITBOX_SIZE;
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
                                && can_step_up(player, voxel_reader, component, player_move_pos)
                            {
                                player.pos[1] += 1.0;
                                player_move_pos[1] += 1.0;
                                break 'outer;
                            }

                            player.pos[component] -= dist_diff * vel_dir[component];
                            player.vel[component] = 0.0;
                            // apply friction
                            let perp_vel = Vector2::new(
                                player.vel[(component + 1) % 3],
                                player.vel[(component + 2) % 3],
                            );
                            if perp_vel.magnitude() > 0.0 {
                                let friction_factor =
                                    VoxelMaterial::FRICTION_COEFFICIENTS[voxel[0] as usize];
                                player.vel[(component + 1) % 3] -= (0.5 * perp_vel.normalize().x
                                    + friction_factor * perp_vel.x)
                                    * time_step;
                                player.vel[(component + 2) % 3] -= (0.5 * perp_vel.normalize().y
                                    + friction_factor * perp_vel.y)
                                    * time_step;
                            }

                            player.collision_vec[component] = -vel_dir[component].signum() as i32;
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
    player: &mut Player,
    voxel_reader: &BufferReadGuard<'_, [[u32; 2]]>,
    component: usize,
    player_move_pos: Point3<f32>,
) -> bool {
    let x_iter_count = (0.99 * player.size * PLAYER_HITBOX_SIZE[(component + 1) % 3]).ceil() + 1.0;
    let z_iter_count = (0.99 * player.size * PLAYER_HITBOX_SIZE[(component + 2) % 3]).ceil() + 1.0;
    let x_dist = (0.99 * player.size * PLAYER_HITBOX_SIZE[(component + 1) % 3]) / x_iter_count;
    let z_dist = (0.99 * player.size * PLAYER_HITBOX_SIZE[(component + 2) % 3]) / z_iter_count;
    let mut start_pos = player.pos + PLAYER_HITBOX_OFFSET
        - 0.99 * 0.5 * player.size * PLAYER_HITBOX_SIZE
        + Vector3::new(0.0, 1.0, 0.0);
    start_pos[component] = player_move_pos[component];

    let mut x_vec = Vector3::new(0.0, 0.0, 0.0);
    let mut z_vec = Vector3::new(0.0, 0.0, 0.0);
    x_vec[(component + 1) % 3] = 1.0;
    z_vec[(component + 2) % 3] = 1.0;
    for x_iter in 0..=(x_iter_count as i32) {
        for z_iter in 0..=(z_iter_count as i32) {
            let pos = start_pos + x_dist * x_iter as f32 * x_vec + z_dist * z_iter as f32 * z_vec;
            let voxel_pos = pos.map(|c| c.floor() as i32);
            let voxel = voxel_reader[get_index(voxel_pos) as usize];
            if voxel[0] != 0 {
                return false;
            }
        }
    }
    true
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
