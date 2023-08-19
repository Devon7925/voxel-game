use std::{collections::VecDeque, sync::Arc};

use bytemuck::{Zeroable, Pod};
use cgmath::{Vector3, InnerSpace, Quaternion, Rotation, One, Point3};
use vulkano::{
    buffer::{
        Buffer, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
};

use crate::projectile_sim_manager::{Projectile, ProjectileComputePipeline};

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

#[derive(Clone, Copy, Debug, Zeroable, Pod)]
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
    pub rot: Quaternion<f32>,
    pub size: Vector3<f32>,
    pub vel: Vector3<f32>,
    pub dir: Vector3<f32>,
    pub up: Vector3<f32>,
    pub right: Vector3<f32>,
    pub health: f32,
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
            dir: Vector3::new(0.0, 0.0, 1.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            right: Vector3::new(1.0, 0.0, 0.0),
            rot: Quaternion::one(),
            size: Vector3::new(1.0, 1.0, 1.0),
            vel: Vector3::new(0.0, 0.0, 0.0),
            health: 100.0,
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

        let current_time:u64 = 10;
        let rollback_time:u64 = 0;

        RollbackData {
            current_time,
            rollback_time,
            rollback_state: WorldState::new(),
            cached_current_state: WorldState::new(),
            actions: VecDeque::from(vec![Vec::new(); (current_time - rollback_time + 10) as usize]),
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
            println!("cannot send action with timestamp {} when rollback time is {}", time_stamp, self.rollback_time);
            return;
        }
        let time_idx = time_stamp - self.rollback_time;
        let actions_len = self.actions.len();
        let action_frame = self.actions.get_mut(time_idx as usize).unwrap_or_else(|| panic!("cannot access index {} in action deque of length {} on sending with timestamp {}", time_idx, actions_len, time_stamp));
        action_frame[player_idx] = Some(action);
    }

    pub fn player_join(&mut self, player: Player) {
        self.rollback_state.players.push(player);
        self.actions.iter_mut().for_each(|x| x.push(Some(PlayerAction {
            ..Default::default()
        })));
    }

    fn update_rollback_state(&mut self, time_step: f32) {
        self.rollback_time += 1;
        let player_actions = self.actions.pop_front().unwrap();
        assert!(self.rollback_time < 100 || player_actions.iter().all(|x| x.is_some()));
        self.rollback_state.step_sim(&player_actions, time_step);
    }

    fn get_current_state(&self, time_step: f32) -> WorldState {
        let mut state = self.rollback_state.clone();
        for i in self.rollback_time..self.current_time {
            let actions = self.actions.get((i-self.rollback_time) as usize).unwrap_or_else(|| panic!("cannot access index {} in action deque of length {}", i-self.rollback_time, self.actions.len()));
            state.step_sim(actions, time_step);
        }
        state
    }

    pub fn step(
        &mut self,
        time_step: f32,
    ) {
        self.update_rollback_state(time_step);
        self.current_time += 1;
        self.actions.push_back(vec![None; self.rollback_state.players.len()]);
        self.cached_current_state = self.get_current_state(time_step);
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
                    rot: [player.rot.v[0], player.rot.v[1], player.rot.v[2], player.rot.s],
                    size: [
                        player.size.x,
                        player.size.y,
                        player.size.z,
                        0.0,
                    ],
                    vel: [
                        player.vel.x,
                        player.vel.y,
                        player.vel.z,
                        0.0,
                    ],
                    dir: [
                        player.dir.x,
                        player.dir.y,
                        player.dir.z,
                        0.0,
                    ],
                    up: [
                        player.up.x,
                        player.up.y,
                        player.up.z,
                        0.0,
                    ],
                    right: [
                        player.right.x,
                        player.right.y,
                        player.right.z,
                        0.0,
                    ],
                };
            }
        }
    }

    pub fn download_projectiles(&mut self, projectile_compute: &ProjectileComputePipeline) {
        self.rollback_state.projectiles = projectile_compute.download_projectiles();
    }
}

impl WorldState {
    pub fn new() -> Self {
        WorldState {
            players: Vec::new(),
            projectiles: Vec::new(),
        }
    }

    pub fn step_sim(&mut self, player_actions: &Vec<Option<PlayerAction>>, time_step: f32) {
        for proj in self.projectiles.iter_mut() {
            let projectile_rot = Quaternion::new(proj.dir[3], proj.dir[0], proj.dir[1], proj.dir[2]).conjugate();
            let projectile_dir = projectile_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0));
            for i in 0..3 {
                proj.pos[i] += projectile_dir[i] * proj.vel * time_step;
            }
            proj.lifetime += time_step;
        }
        for (player_idx, (player, action)) in self.players.iter_mut().zip(player_actions.iter()).enumerate() {
            if let Some(action) = action {
                let sensitivity = 0.001;
                player.dir += action.aim[0] * player.right * sensitivity;
                player.dir += -action.aim[1] * player.up * sensitivity;
                player.dir = player.dir.normalize();
                player.rot = Quaternion::look_at(player.dir, player.up);
                player.right = player.dir.cross(player.up).normalize();
                player.up = player.right.cross(player.dir).normalize();
                let mut move_vec = Vector3::new(0.0, 0.0, 0.0);
                if action.forward == ACTIVE_BUTTON {
                    move_vec += player.dir;
                }
                if action.backward == ACTIVE_BUTTON {
                    move_vec -= player.dir;
                }
                if action.left == ACTIVE_BUTTON {
                    move_vec -= player.right;
                }
                if action.right == ACTIVE_BUTTON {
                    move_vec += player.right;
                }
                if action.jump == ACTIVE_BUTTON {
                    move_vec += player.up;
                }
                if action.crouch == ACTIVE_BUTTON {
                    move_vec -= player.up;
                }
                if move_vec.magnitude() > 0.0 {
                    move_vec = move_vec.normalize();
                }
                let accel_speed = if action.sprint == ACTIVE_BUTTON { 1.2 } else { 0.6 };
                player.vel += accel_speed * move_vec * time_step;

                if action.shoot == ACTIVE_BUTTON {
                    self.projectiles.push(Projectile {
                        pos: [player.pos.x, player.pos.y, player.pos.z, 1.0],
                        dir: [player.rot.v[0], player.rot.v[1], player.rot.v[2], player.rot.s],
                        size: [
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                        vel: 6.0,
                        health: 10.0,
                        lifetime: 0.0,
                        owner: player_idx as u32,
                    })
                }
            }
            if player.vel.magnitude() > 0.0 {
                player.vel -= 0.1 * player.vel * player.vel.magnitude() * time_step + 0.1 * player.vel.normalize() * time_step;
            }
            player.pos += player.vel * time_step;
            // check for collision with projectiles
            for proj in self.projectiles.iter_mut() {
                if player_idx as u32 == proj.owner && proj.lifetime < 1.0 {
                    continue;
                }
                let projectile_rot = Quaternion::new(proj.dir[3], proj.dir[0], proj.dir[1], proj.dir[2]).conjugate();
                let projectile_dir = projectile_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0));
                let projectile_right = projectile_rot.rotate_vector(Vector3::new(1.0, 0.0, 0.0));
                let projectile_up = projectile_rot.rotate_vector(Vector3::new(0.0, 1.0, 0.0));
                let projectile_pos = Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]);
                let projectile_size = Vector3::new(proj.size[0], proj.size[1], proj.size[2]);
                
                let grid_iteration_count = (2.0*projectile_size * 2.0_f32.sqrt()).map(|c| c.ceil());
                let grid_dist = 2.0 * projectile_size.zip(grid_iteration_count, |size, count| size/count);

                let start_pos = projectile_pos - projectile_size.x * projectile_right - projectile_size.y * projectile_up - projectile_size.z * projectile_dir;
                'outer: for i in 0..=(grid_iteration_count.x as i32) {
                    for j in 0..=(grid_iteration_count.y as i32) {
                        for k in 0..=(grid_iteration_count.z as i32) {
                            let pos = start_pos + grid_dist.x * i as f32 * projectile_right + grid_dist.y * j as f32 * projectile_up + grid_dist.z * k as f32 * projectile_dir;
                            let dist = (player.pos - pos).magnitude();
                            if dist < player.size.magnitude() {
                                player.health -= 1.0;
                                proj.health = 0.0;
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
        // remove dead projectiles
        self.projectiles.retain(|proj| proj.health > 0.0);
    }
}