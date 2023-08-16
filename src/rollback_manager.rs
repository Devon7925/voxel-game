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
            actions: VecDeque::from(vec![Vec::new(); (current_time - rollback_time + 3) as usize]),
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

    fn update_rollback_state(&mut self) {
        self.rollback_time += 1;
        let player_actions = self.actions.pop_front().unwrap();
        assert!(self.rollback_time < 100 || player_actions.iter().all(|x| x.is_some()));
        self.rollback_state.step_sim(&player_actions);
    }

    fn get_current_state(&self) -> WorldState {
        let mut state = self.rollback_state.clone();
        for i in self.rollback_time..self.current_time {
            let actions = self.actions.get((i-self.rollback_time) as usize).unwrap_or_else(|| panic!("cannot access index {} in action deque of length {}", i-self.rollback_time, self.actions.len()));
            state.step_sim(actions);
        }
        state
    }

    pub fn step(
        &mut self,
    ) {
        self.update_rollback_state();
        self.current_time += 1;
        self.actions.push_back(vec![None; self.rollback_state.players.len()]);
        self.cached_current_state = self.get_current_state();
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

    pub fn step_sim(&mut self, player_actions: &Vec<Option<PlayerAction>>) {
        for proj in self.projectiles.iter_mut() {
            let projectile_dir = quaternion::rotate_vector(quaternion::conj((proj.dir[3], [proj.dir[0], proj.dir[1], proj.dir[2]])), [0.0, 0.0, 1.0]);
            for i in 0..3 {
                proj.pos[i] += projectile_dir[i] * proj.vel;
            }
        }
        for (player, action) in self.players.iter_mut().zip(player_actions.iter()) {
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
                let accel_speed = if action.sprint == ACTIVE_BUTTON { 0.02 } else { 0.01 };
                player.vel += accel_speed * move_vec;

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
                        vel: 0.1,
                        health: 10.0,
                        pad1: 0.1,
                        pad2: 0.1,
                    })
                }
            }
            player.vel *= 0.9;
            player.pos += player.vel;
        }
    }
}