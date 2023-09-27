use std::{collections::VecDeque, f32::consts::PI, sync::Arc};

use bytemuck::{Pod, Zeroable};
use cgmath::{InnerSpace, One, Point3, Quaternion, Rotation, Vector2, Vector3};
use serde::{Deserialize, Serialize};
use vulkano::{
    buffer::{subbuffer::BufferReadGuard, Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
};

use crate::{
    card_system::{CardManager, ReferencedBaseCard},
    projectile_sim_manager::{Projectile, ProjectileComputePipeline},
    voxel_sim_manager::VoxelComputePipeline,
    CHUNK_SIZE, PLAYER_HITBOX_OFFSET, PLAYER_HITBOX_SIZE, RENDER_SIZE, SPAWN_LOCATION,
};

const ACTIVE_BUTTON: u8 = 1;

#[derive(Clone, Debug)]
pub struct WorldState {
    pub players: Vec<Player>,
    pub projectiles: Vec<Projectile>,
}

#[derive(Clone, Debug)]
pub struct RollbackData {
    pub current_time: u64,
    pub rollback_time: u64,
    pub rollback_state: WorldState,
    pub cached_current_state: WorldState,
    pub actions: VecDeque<Vec<Option<PlayerAction>>>,
    pub projectile_buffer: Subbuffer<[Projectile; 1024]>,
    pub player_buffer: Subbuffer<[UploadPlayer; 128]>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Zeroable, Pod)]
#[repr(C)]
pub struct PlayerAction {
    pub aim: [f32; 2],
    pub forward: u8,
    pub backward: u8,
    pub left: u8,
    pub right: u8,
    pub jump: u8,
    pub crouch: u8,
    pub shoot: u8,
    pub sprint: u8,
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
    pub cards_reference: ReferencedBaseCard,
    pub cards_value: f32,
    pub cooldown: f32,
    pub respawn_timer: f32,
    pub collision_vec: Vector3<i32>,
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
            forward: 0,
            backward: 0,
            left: 0,
            right: 0,
            jump: 0,
            crouch: 0,
            shoot: 0,
            sprint: 0,
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
            cards_reference: ReferencedBaseCard::default(),
            cards_value: 0.0,
            cooldown: 0.0,
            respawn_timer: 0.0,
            collision_vec: Vector3::new(0, 0, 0),
        }
    }
}

impl RollbackData {
    pub fn new(memory_allocator: &Arc<StandardMemoryAllocator>) -> Self {
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

        let current_time: u64 = 5;
        let rollback_time: u64 = 0;

        RollbackData {
            current_time,
            rollback_time,
            rollback_state: WorldState::new(),
            cached_current_state: WorldState::new(),
            actions: VecDeque::from(vec![
                Vec::new();
                (current_time - rollback_time + 10) as usize
            ]),
            player_buffer,
            projectile_buffer,
        }
    }

    pub fn projectiles(&self) -> Subbuffer<[Projectile; 1024]> {
        self.projectile_buffer.clone()
    }

    pub fn players(&self) -> Subbuffer<[UploadPlayer; 128]> {
        self.player_buffer.clone()
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
    }

    fn update_rollback_state(
        &mut self,
        card_manager: &CardManager,
        time_step: f32,
        vox_compute: &mut VoxelComputePipeline,
    ) {
        self.rollback_time += 1;
        if self.rollback_time < 50 {
            return;
        }
        let player_actions = self.actions.pop_front().unwrap();
        assert!(player_actions.iter().all(|x| x.is_some()));
        self.rollback_state
            .step_sim(&player_actions, true, card_manager, time_step, vox_compute);
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
        card_manager: &CardManager,
        time_step: f32,
        vox_compute: &mut VoxelComputePipeline,
    ) {
        self.update_rollback_state(card_manager, time_step, vox_compute);
        self.current_time += 1;
        self.actions
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
                Quaternion::new(proj.dir[3], proj.dir[0], proj.dir[1], proj.dir[2]).conjugate();
            let projectile_dir = projectile_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0));
            let mut proj_vel = projectile_dir * proj.vel;
            let proj_card = card_manager.get_referenced_proj(proj.proj_card_idx as usize);
            proj_vel.y -= 2.0 * (proj_card.gravity as f32) * time_step;
            for i in 0..3 {
                proj.pos[i] += proj_vel[i] * time_step;
            }
            // recompute vel and rot
            let new_projectile_rot =
                Quaternion::look_at(proj_vel.normalize(), Vector3::new(0.0, 1.0, 0.0));
            proj.dir = [
                new_projectile_rot.v[0],
                new_projectile_rot.v[1],
                new_projectile_rot.v[2],
                new_projectile_rot.s,
            ];
            proj.vel = proj_vel.magnitude();

            proj.lifetime += time_step;
            if proj.lifetime >= 3.0 * 1.5f32.powi(proj_card.lifetime) {
                proj.health = 0.0;
            }
        }
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
                }
                continue;
            }
            if let Some(action) = action {
                player.facing[0] =
                    (player.facing[0] - action.aim[0] + 2.0 * PI) % (2.0 * PI);
                player.facing[1] = (player.facing[1] - action.aim[1])
                    .min(PI / 2.0)
                    .max(-PI / 2.0);
                player.dir = Vector3::new(
                    player.facing[0].sin() * player.facing[1].cos(),
                    player.facing[1].sin(),
                    player.facing[0].cos() * player.facing[1].cos(),
                );
                player.rot = Quaternion::look_at(player.dir, Vector3::new(0.0, 1.0, 0.0));
                player.right = player.dir.cross(Vector3::new(0.0, 1.0, 0.0)).normalize();
                player.up = player.right.cross(player.dir).normalize();
                let mut move_vec = Vector3::new(0.0, 0.0, 0.0);
                let player_forward = Vector3::new(player.dir.x, 0.0, player.dir.z).normalize();
                let player_right = Vector3::new(player.right.x, 0.0, player.right.z).normalize();
                if action.forward == ACTIVE_BUTTON {
                    move_vec += player_forward;
                }
                if action.backward == ACTIVE_BUTTON {
                    move_vec -= player_forward;
                }
                if action.left == ACTIVE_BUTTON {
                    move_vec -= player_right;
                }
                if action.right == ACTIVE_BUTTON {
                    move_vec += player_right;
                }
                if action.jump == ACTIVE_BUTTON {
                    move_vec += Vector3::new(0.0, 0.5, 0.0);
                }
                if action.crouch == ACTIVE_BUTTON {
                    move_vec -= Vector3::new(0.0, 0.5, 0.0);
                }
                if move_vec.magnitude() > 0.0 {
                    move_vec = move_vec.normalize();
                }
                let accel_speed = if action.sprint == ACTIVE_BUTTON {
                    1.5
                } else {
                    1.0
                } * if player.collision_vec != Vector3::new(0, 0, 0) {
                    4.0
                } else {
                    1.0
                };
                player.vel += accel_speed * move_vec * time_step;

                if action.jump == ACTIVE_BUTTON {
                    player.vel += player
                        .collision_vec
                        .zip(Vector3::new(0.3, 8.0, 0.3), |c, m| c as f32 * m);
                }

                if action.shoot == ACTIVE_BUTTON && player.cooldown <= 0.0 {
                    player.cooldown = player.cards_value;
                    let effects = card_manager.get_effects_from_base_card(&player.cards_reference, &player.pos, &player.rot, player_idx as u32);
                    self.projectiles.extend(effects.0);
                    if is_real_update && effects.1.len() > 0 {
                        let mut writer = voxels.write().unwrap();
                        for (pos, material) in effects.1 {
                            vox_compute.queue_update_from_voxel_pos(&[pos.x, pos.y, pos.z]);
                            writer[get_index(pos) as usize] = material.to_memory();
                        }
                    }
                }
            }
            player.cooldown -= time_step;

            player.vel.y -= 7.5 * time_step;
            if player.vel.magnitude() > 0.0 {
                player.vel -= 0.1 * player.vel * player.vel.magnitude() * time_step
                    + 0.2 * player.vel.normalize() * time_step;
            }

            let prev_collision_vec = player.collision_vec.clone();
            player.collision_vec = Vector3::new(0, 0, 0);
            collide_player(player, time_step, &voxel_reader, prev_collision_vec);
            // check for collision with projectiles
            for proj in self.projectiles.iter_mut() {
                if player_idx as u32 == proj.owner && proj.lifetime < 1.0 {
                    continue;
                }
                let projectile_rot =
                    Quaternion::new(proj.dir[3], proj.dir[0], proj.dir[1], proj.dir[2]).conjugate();
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
                            let dist = (player.pos - pos).magnitude();
                            if dist < player.size {
                                player.health -= proj.damage;
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
                                    let effects = card_manager.get_effects_from_base_card(
                                        &card_ref,
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
                                break 'outer;
                            }
                        }
                    }
                }
            }
            if player.health <= 0.0 {
                player.respawn_timer = 5.0;
            }
        }
        // remove dead projectiles
        self.projectiles.retain(|proj| proj.health > 0.0);
        self.projectiles.extend(new_projectiles);
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
                                player.vel[(component + 1) % 3] -=
                                    (0.5 * perp_vel.normalize().x + 1.5 * perp_vel.x) * time_step;
                                player.vel[(component + 2) % 3] -=
                                    (0.5 * perp_vel.normalize().y + 1.5 * perp_vel.y) * time_step;
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
