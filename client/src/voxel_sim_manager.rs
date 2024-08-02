// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    app::CreationInterface, card_system::{CardManager, VoxelMaterial}, cpu_simulation::WorldState, game_manager::GameState, game_modes::GameMode, rollback_manager::UploadPlayer, utils::{Direction, QueueMap, QueueSet, VoxelUpdateQueue}, CHUNK_SIZE, MAX_CHUNK_UPDATE_RATE, MAX_VOXEL_UPDATE_RATE, MAX_WORLDGEN_RATE, SUB_CHUNK_COUNT, WORLDGEN_CHUNK_COUNT
};
use bytemuck::{Pod, Zeroable};
use cgmath::{MetricSpace, Point3, Quaternion, Vector3, Zero};
use std::{collections::HashSet, iter, sync::Arc};
use voxel_shared::{GameSettings, WorldGenSettings};
use vulkano::{
    buffer::{
        Buffer, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo,
        layout::{
            PipelineLayoutCreateInfo, PushConstantRange,
        },
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    shader::{ShaderModule, ShaderStages},
    sync::GpuFuture,
};

#[derive(Clone, Copy, Zeroable, Debug, Pod)]
#[repr(C)]
pub struct Projectile {
    pub pos: [f32; 4],
    pub chunk_update_pos: [u32; 4],
    pub dir: [f32; 4],
    pub size: [f32; 4],
    pub vel: f32,
    pub health: f32,
    pub lifetime: f32,
    pub owner: u32,
    pub damage: f32,
    pub proj_card_idx: u32,
    pub wall_bounce: u32,
    pub is_from_head: u32,
    pub should_collide_with_terrain: u32,
    pub _filler0: u32,
    pub _filler1: u32,
    pub _filler2: u32,
}

#[derive(Clone, Copy, Zeroable, Debug, Pod)]
#[repr(C)]
pub struct Collision {
    pub id1: u32,
    pub id2: u32,
    pub properties: u32,
}

/// Pipeline holding double buffered grid & color image. Grids are used to calculate the state, and
/// color image is used to show the output. Because on each step we determine state in parallel, we
/// need to write the output to another grid. Otherwise the state would not be correctly determined
/// as one shader invocation might read data that was just written by another shader invocation.
pub struct VoxelComputePipeline {
    compute_queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,

    compute_proj_pipeline: Arc<ComputePipeline>,
    unload_chunks_pipeline: Arc<ComputePipeline>,
    write_voxels_pipeline: Arc<ComputePipeline>,
    compute_voxel_update_pipeline: Arc<ComputePipeline>,
    compute_worldgen_pipeline: Arc<ComputePipeline>,
    complete_worldgen_pipeline: Arc<ComputePipeline>,
    
    projectile_buffer: Subbuffer<[Projectile; 1024]>,
    upload_projectile_count: usize,

    player_buffer: Subbuffer<[UploadPlayer; 1024]>,
    upload_player_count: usize,

    collision_buffer: Subbuffer<[Collision; 1024]>,

    gpu_chunks: Arc<ImageView>,
    voxel_buffer: Subbuffer<[u32]>,
    cpu_chunks_copy: Vec<Vec<Vec<u32>>>,
    available_chunks: Vec<u32>,
    slice_to_unload: Option<Direction>,

    worldgen_updates: Subbuffer<[[i32; 4]; MAX_WORLDGEN_RATE]>,
    worldgen_results: Subbuffer<
    [u32; MAX_WORLDGEN_RATE
    / WORLDGEN_CHUNK_COUNT
    / WORLDGEN_CHUNK_COUNT
    / WORLDGEN_CHUNK_COUNT],
    >,
    worldgen_update_queue: QueueMap<[u32; 3], bool>,
    oob_worldgens: HashSet<[u32; 3]>,
    last_worldgen_count: usize,
    
    voxel_writes: Subbuffer<[[u32; 4]; MAX_VOXEL_UPDATE_RATE]>,
    voxel_write_queue: QueueSet<[u32; 4]>,
    
    chunk_updates: Subbuffer<[[u32; 4]; MAX_CHUNK_UPDATE_RATE]>,
    chunk_update_queue: VoxelUpdateQueue,
    last_update_priorities: Vec<i32>,
    last_update_count: usize,
}

fn empty_chunk_grid(
    memory_allocator: Arc<StandardMemoryAllocator>,
    game_settings: &GameSettings,
) -> (Vec<Vec<Vec<u32>>>, Arc<ImageView>) {
    let cpu_chunks = (0..game_settings.render_size[0])
        .map(|_| {
            (0..game_settings.render_size[1])
                .map(|_| vec![0; game_settings.render_size[2] as usize])
                .collect()
        })
        .collect();
    let extent = game_settings.render_size.into();
    let image = Image::new(
        memory_allocator,
        ImageCreateInfo {
            image_type: ImageType::Dim3d,
            format: Format::R32_UINT,
            extent,
            usage: ImageUsage::STORAGE,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .unwrap();

    (cpu_chunks, ImageView::new_default(image).unwrap())
}

fn empty_list(
    memory_allocator: Arc<StandardMemoryAllocator>,
    game_settings: &GameSettings,
) -> Subbuffer<[u32]> {
    Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        vec![0; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * game_settings.max_loaded_chunks as usize],
    )
    .unwrap()
}

macro_rules! is_inbounds {
    ($loc:expr, $game_state:expr, $game_settings:expr) => {
        $loc[0] >= $game_state.start_pos[0]
            && $loc[0] < ($game_state.start_pos[0] + $game_settings.render_size[0])
            && $loc[1] >= $game_state.start_pos[1]
            && $loc[1] < ($game_state.start_pos[1] + $game_settings.render_size[1])
            && $loc[2] >= $game_state.start_pos[2]
            && $loc[2] < ($game_state.start_pos[2] + $game_settings.render_size[2])
    };
}

fn get_pipeline(shader: Arc<ShaderModule>, device: Arc<Device>, layout: Arc<PipelineLayout>) -> Arc<ComputePipeline> {
    let cs = shader.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(cs);
    ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap()
}

impl VoxelComputePipeline {
    pub fn new(
        creation_interface: &CreationInterface,
        game_settings: &GameSettings,
    ) -> VoxelComputePipeline {
        let memory_allocator = &creation_interface.memory_allocator;

        let (cpu_chunks_copy, gpu_chunks) =
            empty_chunk_grid(memory_allocator.clone(), game_settings);
        let voxel_buffer = empty_list(memory_allocator.clone(), game_settings);
        // set the unloaded chunk
        {
            let mut voxel_writer = voxel_buffer.write().unwrap();
            for (i, _) in iter::repeat(())
                .enumerate()
                .take((CHUNK_SIZE as usize) * (CHUNK_SIZE as usize) * (CHUNK_SIZE as usize))
            {
                voxel_writer[i] = VoxelMaterial::Unloaded.to_memory();
            }
            for (i, _) in iter::repeat(())
                .enumerate()
                .skip((CHUNK_SIZE as usize) * (CHUNK_SIZE as usize) * (CHUNK_SIZE as usize))
                .take((CHUNK_SIZE as usize) * (CHUNK_SIZE as usize) * (CHUNK_SIZE as usize))
            {
                voxel_writer[i] = VoxelMaterial::UnloadedAir.to_memory();
            }
        }
        let available_chunks = (2..game_settings.max_loaded_chunks).collect();

        let primary_desc_layout = DescriptorSetLayout::new(
            creation_interface.queue.device().clone(),
            DescriptorSetLayoutCreateInfo {
                bindings: [
                    (
                        0,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::COMPUTE,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::StorageImage,
                            )
                        },
                    ),
                    (
                        1,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::COMPUTE,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::StorageBuffer,
                            )
                        },
                    ),
                    (
                        2,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::COMPUTE,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::StorageBuffer,
                            )
                        },
                    ),
                    (
                        3,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::COMPUTE,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::StorageBuffer,
                            )
                        },
                    ),
                    (
                        4,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::COMPUTE,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::StorageBuffer,
                            )
                        },
                    ),
                    (
                        5,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::COMPUTE,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::StorageBuffer,
                            )
                        },
                    ),
                    (
                        6,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::COMPUTE,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::StorageBuffer,
                            )
                        },
                    ),
                    (
                        7,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::COMPUTE,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::StorageBuffer,
                            )
                        },
                    ),
                    (
                        8,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::COMPUTE,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::StorageBuffer,
                            )
                        },
                    ),
                ]
                .into(),
                ..Default::default()
            },
        )
        .unwrap();

        let layout = PipelineLayout::new(
            creation_interface.queue.device().clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![primary_desc_layout.clone()],
                push_constant_ranges: vec![PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    offset: 0,
                    size: std::mem::size_of::<compute_projs_cs::SimData>() as u32,
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let compute_proj_pipeline = get_pipeline(compute_projs_cs::load(creation_interface.queue.device().clone()).unwrap(), creation_interface.queue.device().clone(), layout.clone());
        let unload_chunks_pipeline = get_pipeline(unload_chunks_cs::load(creation_interface.queue.device().clone()).unwrap(), creation_interface.queue.device().clone(), layout.clone());
        let compute_worldgen_pipeline = match game_settings.world_gen {
            WorldGenSettings::Normal => get_pipeline(normal_world_cs::load(creation_interface.queue.device().clone()).unwrap(), creation_interface.queue.device().clone(), layout.clone()),
            WorldGenSettings::Control(_) => get_pipeline(control_world_cs::load(creation_interface.queue.device().clone()).unwrap(), creation_interface.queue.device().clone(), layout.clone()),
        };
        let complete_worldgen_pipeline = get_pipeline(complete_worldgen_cs::load(creation_interface.queue.device().clone()).unwrap(), creation_interface.queue.device().clone(), layout.clone());
        let write_voxels_pipeline = get_pipeline(write_voxels_cs::load(creation_interface.queue.device().clone()).unwrap(), creation_interface.queue.device().clone(), layout.clone());
        let compute_voxel_update_pipeline = get_pipeline(compute_updates_cs::load(creation_interface.queue.device().clone()).unwrap(), creation_interface.queue.device().clone(), layout.clone());

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

        let collision_buffer = Buffer::new_sized(
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

        let chunk_updates = Buffer::new_sized(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        )
        .unwrap();

        let voxel_writes = Buffer::new_sized(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        )
        .unwrap();

        let worldgen_updates = Buffer::new_sized(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        )
        .unwrap();

        let worldgen_results: Subbuffer<
            [u32; MAX_WORLDGEN_RATE
                / WORLDGEN_CHUNK_COUNT
                / WORLDGEN_CHUNK_COUNT
                / WORLDGEN_CHUNK_COUNT],
        > = Buffer::new_sized(
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
        worldgen_results.write().unwrap().fill(1);

        VoxelComputePipeline {
            compute_queue: creation_interface.queue.clone(),
            compute_proj_pipeline,
            unload_chunks_pipeline,
            write_voxels_pipeline,
            compute_voxel_update_pipeline,
            compute_worldgen_pipeline,
            complete_worldgen_pipeline,
            command_buffer_allocator: creation_interface.command_buffer_allocator.clone(),
            descriptor_set_allocator: creation_interface.descriptor_set_allocator.clone(),
            projectile_buffer,
            player_buffer,
            collision_buffer,
            cpu_chunks_copy,
            gpu_chunks,
            voxel_buffer,
            available_chunks,
            chunk_update_queue: VoxelUpdateQueue::with_capacity(
                game_settings.max_update_rate as usize,
            ),
            chunk_updates,
            voxel_write_queue: QueueSet::with_capacity(MAX_VOXEL_UPDATE_RATE),
            voxel_writes,
            slice_to_unload: None,
            worldgen_update_queue: QueueMap::with_capacity(
                game_settings.max_worldgen_rate as usize,
                Box::new(|a, b| a || b),
            ),
            oob_worldgens: HashSet::new(),
            worldgen_updates,
            worldgen_results,
            upload_projectile_count: 0,
            upload_player_count: 0,
            last_update_priorities: Vec::new(),
            last_update_count: 0,
            last_worldgen_count: 0,
        }
    }

    pub fn projectiles(&self) -> Subbuffer<[Projectile; 1024]> {
        self.projectile_buffer.clone()
    }

    pub fn players(&self) -> Subbuffer<[UploadPlayer; 1024]> {
        self.player_buffer.clone()
    }

    pub fn download_projectiles(
        &mut self,
        card_manager: &CardManager,
        game_settings: &GameSettings,
        world_state: &mut WorldState,
        game_mode: &Box<dyn GameMode>,
    ) -> (Vec<Projectile>, Vec<Collision>) {
        let mut projectiles = Vec::new();
        let mut new_voxels = Vec::new();
        let projectiles_buffer = self.projectile_buffer.read().unwrap();
        for i in 0..self.upload_projectile_count {
            let projectile = projectiles_buffer[i];
            if projectile.health == 0.0 && projectile.chunk_update_pos[3] == 1 {
                let proj_card = card_manager.get_referenced_proj(projectile.proj_card_idx as usize);
                {
                    let chunk_location = [
                        (projectile.chunk_update_pos[0] as usize * SUB_CHUNK_COUNT / CHUNK_SIZE)
                            as u32,
                        (projectile.chunk_update_pos[1] as usize * SUB_CHUNK_COUNT / CHUNK_SIZE)
                            as u32,
                        (projectile.chunk_update_pos[2] as usize * SUB_CHUNK_COUNT / CHUNK_SIZE)
                            as u32,
                    ];
                    let chunk = chunk_location.map(|e| e / SUB_CHUNK_COUNT as u32);
                    let chunk_idx = self.get_chunk(chunk, game_settings);
                    if chunk_idx == 0 {
                        self.worldgen_update_queue
                            .push([chunk[0], chunk[1], chunk[2]], false);
                    } else if chunk_idx != 1 {
                        self.chunk_update_queue
                            .push_with_priority(chunk_location, 1);
                    }
                };
                for card_ref in proj_card
                    .on_hit
                    .clone()
                {
                    let proj_rot = projectile.dir;
                    let proj_rot =
                        Quaternion::new(proj_rot[3], proj_rot[0], proj_rot[1], proj_rot[2]);
                    let effects = card_manager.get_effects_from_base_card(
                        card_ref,
                        &Point3::new(projectile.pos[0], projectile.pos[1], projectile.pos[2]),
                        &proj_rot,
                        projectile.owner,
                        false,
                    );
                    projectiles.extend(effects.0);
                    new_voxels.extend(effects.1);
                }
            }
            projectiles.push(projectile);
        }

        if new_voxels.len() > 0 {
            for (pos, material) in new_voxels {
                self.voxel_write_queue
                    .push([pos[0], pos[1], pos[2], material.to_memory()]);
            }
        }

        let collisions_buffer = self.collision_buffer.read().unwrap();
        let mut collisions = vec![];
        for i in 0..1024 {
            if collisions_buffer[i].properties == 0 {
                break;
            }
            let collision = collisions_buffer[i];
            let player_idx = collision.id2 as usize;
            let player = &world_state.players[player_idx];
            let proj = projectiles_buffer[collision.id1 as usize];

            // check piercing invincibility at start to prevent order from mattering
            let player_piercing_invincibility = player.player_piercing_invincibility > 0.0;

            if player_idx as u32 == proj.owner && proj.lifetime < 1.0 && proj.is_from_head == 1
            {
                continue;
            }
            let proj_card = card_manager.get_referenced_proj(proj.proj_card_idx as usize);

            if proj_card.no_friendly_fire
                && game_mode.are_friends(proj.owner, player_idx as u32, &world_state.players)
            {
                continue;
            }
            if proj_card.no_enemy_fire && proj.owner != player_idx as u32 {
                continue;
            }
            if player_piercing_invincibility && proj_card.pierce_players {
                continue;
            }

            collisions.push(collision);
        }

        (projectiles, collisions)
    }

    pub fn cpu_chunks(&self) -> &Vec<Vec<Vec<u32>>> {
        &self.cpu_chunks_copy
    }

    pub fn chunks(&self) -> Arc<ImageView> {
        self.gpu_chunks.clone()
    }

    pub fn voxels(&self) -> Subbuffer<[u32]> {
        self.voxel_buffer.clone()
    }

    pub fn queue_voxel_write(&mut self, write: [u32; 4]) {
        self.voxel_write_queue.push(write);
    }

    pub fn queue_chunk_update(&mut self, queued: [u32; 3], game_settings: &GameSettings) {
        let chunk = queued.map(|e| e / SUB_CHUNK_COUNT as u32);
        let chunk_idx = self.get_chunk(chunk, game_settings);
        if chunk_idx == 0 {
            self.worldgen_update_queue
                .push([chunk[0], chunk[1], chunk[2]], false);
        } else if chunk_idx != 1 {
            self.chunk_update_queue.push_all(queued);
        }
    }

    pub fn queue_update_from_world_pos(
        &mut self,
        queued: &Point3<f32>,
        game_settings: &GameSettings,
    ) {
        let chunk_location = [
            (queued[0].floor() as usize * SUB_CHUNK_COUNT / CHUNK_SIZE) as u32,
            (queued[1].floor() as usize * SUB_CHUNK_COUNT / CHUNK_SIZE) as u32,
            (queued[2].floor() as usize * SUB_CHUNK_COUNT / CHUNK_SIZE) as u32,
        ];
        self.queue_chunk_update(chunk_location, game_settings);
    }

    fn get_chunk(&self, pos: [u32; 3], game_settings: &GameSettings) -> u32 {
        let adj_pos = [0, 1, 2].map(|i| pos[i].rem_euclid(game_settings.render_size[i]));
        self.cpu_chunks_copy[adj_pos[0] as usize][adj_pos[1] as usize][adj_pos[2] as usize]
    }

    fn set_chunk(&mut self, pos: [u32; 3], game_settings: &GameSettings, chunk_idx: u32) {
        let adj_pos = [0, 1, 2].map(|i| pos[i].rem_euclid(game_settings.render_size[i]));
        self.cpu_chunks_copy[adj_pos[0] as usize][adj_pos[1] as usize][adj_pos[2] as usize] =
            chunk_idx;
    }

    pub fn move_start_pos(
        &mut self,
        game_state: &mut GameState,
        direction: Direction,
        game_settings: &GameSettings,
    ) {
        let offset = direction.to_offset();
        game_state.start_pos = game_state.start_pos.zip(Point3::from(offset), |a, b| {
            a.checked_add_signed(b).unwrap()
        });

        let load_idx: u32 = if direction.is_positive() {
            game_settings.render_size[direction.component_index()] - 1
        } else {
            0
        };

        for y_i in 0..game_settings.render_size[(direction.component_index() + 1) % 3] {
            for z_i in 0..game_settings.render_size[(direction.component_index() + 2) % 3] {
                let mut chunk_location = [
                    game_state.start_pos[0],
                    game_state.start_pos[1],
                    game_state.start_pos[2],
                ];
                chunk_location[direction.component_index()] += load_idx;
                chunk_location[(direction.component_index() + 1) % 3] += y_i;
                chunk_location[(direction.component_index() + 2) % 3] += z_i;
                let current_chunk_ref = self.get_chunk(chunk_location, game_settings);
                if current_chunk_ref > 1 {
                    self.available_chunks.push(current_chunk_ref);
                }
                self.set_chunk(chunk_location, game_settings, 0);
                if self.oob_worldgens.contains(&chunk_location) {
                    self.oob_worldgens.remove(&chunk_location);
                    self.worldgen_update_queue.push(chunk_location, false);
                }
            }
        }

        self.slice_to_unload = Some(direction);
    }

    // chunks are represented as u32 with a 1 representing a changed chunk
    // this function will get the locations of those and push updates and then clear the buffer
    pub fn push_updates_from_changed(&mut self, game_settings: &GameSettings, player_locations: &Vec<Point3<f32>>) {
        puffin::profile_function!();
        if self.last_worldgen_count > 0 {
            let worldgen_results = self.worldgen_results.read().unwrap();
            let last_worldgen = self.worldgen_updates.read().unwrap();
            for i in 0..self.last_worldgen_count {
                if worldgen_results[i] == 1 {
                    let last_worldgen_chunk = last_worldgen[8 * i];
                    self.available_chunks
                        .push(last_worldgen_chunk[3].abs() as u32);
                    self.cpu_chunks_copy[((last_worldgen_chunk[0] as u32 / 2)
                        % game_settings.render_size[0])
                        as usize][((last_worldgen_chunk[1] as u32 / 2)
                        % game_settings.render_size[1])
                        as usize][((last_worldgen_chunk[2] as u32 / 2)
                        % game_settings.render_size[2])
                        as usize] = 1;
                    //update neighbors
                    for x in -1..=1 {
                        for y in -1..=1 {
                            for z in -1..=1 {
                                let queued = [
                                    (last_worldgen_chunk[0] / 2 + x) as u32,
                                    (last_worldgen_chunk[1] / 2 + y) as u32,
                                    (last_worldgen_chunk[2] / 2 + z) as u32,
                                ];
                                let chunk = queued.map(|e| e / SUB_CHUNK_COUNT as u32);
                                let chunk_idx = self.get_chunk(chunk, game_settings);
                                if chunk_idx == 0 {
                                    self.worldgen_update_queue
                                        .push([chunk[0], chunk[1], chunk[2]], false);
                                } else if chunk_idx != 1 {
                                    self.chunk_update_queue.push_all(queued);
                                }
                            }
                        }
                    }
                }
            }
        }
        // last component of 1 means the chunk was changed and therefore means it and surrounding chunks need to be updated
        {
            puffin::profile_scope!("count updates");
            let reader = self.chunk_updates.read().unwrap();
            for i in 0..self.last_update_count {
                let read_update = reader[i];
                if read_update[3] & 1 == 1 {
                    let chunk_pos = Point3::new(read_update[0], read_update[1], read_update[2]).map(|c| c * SUB_CHUNK_COUNT as u32 * CHUNK_SIZE as u32).map(|c| c as f32);
                    let min_x = -((read_update[3] as i32 >> 1) & 1);
                    let min_y = -((read_update[3] as i32 >> 2) & 1);
                    let min_z = -((read_update[3] as i32 >> 3) & 1);
                    let max_x = (read_update[3] as i32 >> 4) & 1;
                    let max_y = (read_update[3] as i32 >> 5) & 1;
                    let max_z = (read_update[3] as i32 >> 6) & 1;
                    for x_offset in min_x..=max_x {
                        for y_offset in min_y..=max_y {
                            for z_offset in min_z..=max_z {
                                let distance_weight = player_locations.iter().map(|player_pos| {
                                    let distance = (player_pos - chunk_pos).distance(Vector3::zero());
                                    6.0 - distance
                                }).sum::<f32>();
                                self.chunk_update_queue.push_with_priority(
                                    [
                                        read_update[0].wrapping_add_signed(x_offset),
                                        read_update[1].wrapping_add_signed(y_offset),
                                        read_update[2].wrapping_add_signed(z_offset),
                                    ],
                                    self.last_update_priorities[i] - 1 + distance_weight as i32,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn update_count(&self) -> usize {
        self.chunk_update_queue.len()
    }

    pub fn worldgen_count(&self) -> usize {
        self.worldgen_update_queue.len()
    }

    pub fn worldgen_capacity(&self) -> usize {
        self.available_chunks.len()
    }

    pub fn compute<F>(
        &mut self,
        before_future: F,
        game_state: &mut GameState,
        game_settings: &GameSettings,
        projectiles: &Vec<Projectile>,
        players: Vec<UploadPlayer>,
    ) -> Box<dyn GpuFuture>
    where
        F: GpuFuture + 'static,
    {
        puffin::profile_function!();
        //send projectiles
        self.upload_projectile_count = 1024.min(projectiles.len());
        {
            let mut projectiles_writer = self.projectile_buffer.write().unwrap();
            for i in 0..self.upload_projectile_count {
                let projectile = projectiles.get(i).unwrap();
                projectiles_writer[i] = projectile.clone();
            }
        }
        self.upload_player_count = 1024.min(players.len());
        {
            let mut players_writer = self.player_buffer.write().unwrap();
            for i in 0..self.upload_player_count {
                let player = players.get(i).unwrap();
                players_writer[i] = player.clone();
            }
        }
        {
            let mut collision_writer = self.collision_buffer.write().unwrap();
            for i in 0..1024 {
                collision_writer[i] = Collision {
                    id1: 0,
                    id2: 0,
                    properties: 0,
                };
            }
        }
        //send chunk updates
        let worldgen_update_count = self.load_worldgen_updates(game_state, game_settings);

        //send voxel writes
        let voxel_write_count = self.load_voxel_writes(game_state, game_settings);

        //send chunk updates
        let chunk_update_count = self.load_chunk_updates(game_state, game_settings);

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.as_ref(),
            self.compute_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let primary_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            self.compute_proj_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .unwrap()
                .clone(),
            [
                WriteDescriptorSet::image_view(0, self.chunks()),
                WriteDescriptorSet::buffer(1, self.voxels()),
                WriteDescriptorSet::buffer(2, self.projectiles()),
                WriteDescriptorSet::buffer(3, self.worldgen_updates.clone()),
                WriteDescriptorSet::buffer(4, self.worldgen_results.clone()),
                WriteDescriptorSet::buffer(5, self.chunk_updates.clone()),
                WriteDescriptorSet::buffer(6, self.voxel_writes.clone()),
                WriteDescriptorSet::buffer(7, self.players()),
                WriteDescriptorSet::buffer(8, self.collision_buffer.clone()),
            ],
            [],
        )
        .unwrap();
        let push_constants = compute_projs_cs::SimData {
            render_size: Into::<[u32; 3]>::into(game_settings.render_size).into(),
            start_pos: Into::<[u32; 3]>::into(game_state.start_pos).into(),
            voxel_update_offset: self.chunk_update_queue.queue_set_idx().into(),
            dt: game_settings.delta_time,
            projectile_count: self.upload_projectile_count as u32,
            player_count: self.upload_player_count as u32,
            worldgen_count: self.last_worldgen_count as u32,
            
            unload_index: self.slice_to_unload.map(|direction| if direction.is_positive() {
                game_state.start_pos[direction.component_index()]
                    + game_settings.render_size[direction.component_index()]
                    - 1
            } else {
                game_state.start_pos[direction.component_index()]
            } as i32).unwrap_or(0),
            unload_component: self.slice_to_unload.map(|slice| slice.component_index() as u32).unwrap_or(0),
            voxel_write_count: voxel_write_count as u32,
            worldgen_seed: game_settings.world_gen.get_seed(),
        };

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_proj_pipeline.layout().clone(),
                0,
                primary_set.clone(),
            )
            .unwrap()
            .push_constants(
                self.compute_proj_pipeline.layout().clone(),
                0,
                push_constants,
            )
            .unwrap();

        if self.upload_projectile_count > 0 {
            builder
                .bind_pipeline_compute(self.compute_proj_pipeline.clone())
                .unwrap()
                .dispatch([((self.upload_projectile_count + 127) / 128) as u32, 1, 1])
                .unwrap();
        }

        if let Some(direction_to_unload) = self.slice_to_unload.clone() {
            builder
                .bind_pipeline_compute(self.unload_chunks_pipeline.clone())
                .unwrap()
                .dispatch([
                    game_settings.render_size[(direction_to_unload.component_index() + 1) % 3] / 16,
                    game_settings.render_size[(direction_to_unload.component_index() + 2) % 3] / 16,
                    1,
                ])
                .unwrap();

            self.slice_to_unload = None;
        }
        if worldgen_update_count > 0 {
            builder
                .bind_pipeline_compute(self.compute_worldgen_pipeline.clone())
                .unwrap()
                .dispatch([worldgen_update_count as u32, 1, 1])
                .unwrap()
                .bind_pipeline_compute(self.complete_worldgen_pipeline.clone())
                .unwrap()
                .dispatch([
                    (self.last_worldgen_count as f32 / 256.0).ceil() as u32,
                    1,
                    1,
                ])
                .unwrap();
        };
        if voxel_write_count > 0 {
            builder
                .bind_pipeline_compute(self.write_voxels_pipeline.clone())
                .unwrap()
                .dispatch([(voxel_write_count as f32 / 256.0).ceil() as u32, 1, 1])
                .unwrap();
        };
        if chunk_update_count > 0 {
            builder
                .bind_pipeline_compute(self.compute_voxel_update_pipeline.clone())
                .unwrap()
                .dispatch([chunk_update_count as u32, 1, 1])
                .unwrap();
        };
        let command_buffer = builder.build().unwrap();
        before_future
            .then_execute(self.compute_queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .boxed()
    }

    fn load_chunk_updates(&mut self, game_state: &mut GameState, game_settings: &GameSettings) -> usize {
        let mut chunk_update_count = 0;
        self.last_update_priorities.clear();
        if !self.chunk_update_queue.is_empty() {
            let mut chunk_updates_buffer = self.chunk_updates.write().unwrap();
            self.chunk_update_queue.swap_queue_set();
            while let Some(loc) = self.chunk_update_queue.pop() {
                if is_inbounds!(
                    loc.0.map(|c| c / SUB_CHUNK_COUNT as u32),
                    game_state,
                    game_settings
                ) {
                    let chunk = self.get_chunk(
                        [
                            loc.0[0] / SUB_CHUNK_COUNT as u32,
                            loc.0[1] / SUB_CHUNK_COUNT as u32,
                            loc.0[2] / SUB_CHUNK_COUNT as u32,
                        ],
                        game_settings,
                    );
                    if chunk == 0 {
                        self.worldgen_update_queue.push(
                            [
                                loc.0[0] / SUB_CHUNK_COUNT as u32,
                                loc.0[1] / SUB_CHUNK_COUNT as u32,
                                loc.0[2] / SUB_CHUNK_COUNT as u32,
                            ],
                            false,
                        );
                    } else if chunk != 1 {
                        chunk_updates_buffer[chunk_update_count] =
                            [loc.0[0], loc.0[1], loc.0[2], 0];
                        self.last_update_priorities.push(loc.1);
                        chunk_update_count += 1;
                        if chunk_update_count as u32 >= game_settings.max_update_rate {
                            break;
                        }
                    }
                }
            }
        }
        self.last_update_count = chunk_update_count;
        chunk_update_count
    }
    
    fn load_voxel_writes(&mut self, game_state: &mut GameState, game_settings: &GameSettings) -> usize {
        let mut voxel_write_count = 0;
        {
            let mut delayed_writes = vec![];
            let mut voxel_write_buffer = self.voxel_writes.write().unwrap();
            while let Some(voxel_write) = self.voxel_write_queue.pop() {
                if is_inbounds!(
                    voxel_write.map(|c| c / CHUNK_SIZE as u32),
                    game_state,
                    game_settings
                ) {
                    let voxel_chunk = self.get_chunk(
                        [
                            voxel_write[0] / CHUNK_SIZE as u32,
                            voxel_write[1] / CHUNK_SIZE as u32,
                            voxel_write[2] / CHUNK_SIZE as u32,
                        ],
                        game_settings,
                    );
                    if voxel_chunk > 1 {
                        voxel_write_buffer[voxel_write_count] = voxel_write;
                        voxel_write_count += 1;
                        let pos_in_chunk = [
                            voxel_write[0] % CHUNK_SIZE as u32,
                            voxel_write[1] % CHUNK_SIZE as u32,
                            voxel_write[2] % CHUNK_SIZE as u32,
                        ];
                        let chunk = [
                            voxel_write[0] / CHUNK_SIZE as u32,
                            voxel_write[1] / CHUNK_SIZE as u32,
                            voxel_write[2] / CHUNK_SIZE as u32,
                        ];
                        for offset_x in (if pos_in_chunk[0] == 0 { -1 } else { 0 })
                            ..=(if pos_in_chunk[0] + 1 == CHUNK_SIZE as u32 {
                                1
                            } else {
                                0
                            })
                        {
                            for offset_y in (if pos_in_chunk[1] == 0 { -1 } else { 0 })
                                ..=(if pos_in_chunk[1] + 1 == CHUNK_SIZE as u32 {
                                    1
                                } else {
                                    0
                                })
                            {
                                for offset_z in (if pos_in_chunk[2] == 0 { -1 } else { 0 })
                                    ..=(if pos_in_chunk[2] + 1 == CHUNK_SIZE as u32 {
                                        1
                                    } else {
                                        0
                                    })
                                {
                                    self.chunk_update_queue.push_with_priority(
                                        [
                                            chunk[0].wrapping_add_signed(offset_x),
                                            chunk[1].wrapping_add_signed(offset_y),
                                            chunk[2].wrapping_add_signed(offset_z),
                                        ],
                                        1,
                                    );
                                }
                            }
                        }
                        if voxel_write_count >= MAX_VOXEL_UPDATE_RATE {
                            break;
                        }
                    } else if voxel_chunk == 1 {
                        self.worldgen_update_queue.push(
                            [
                                voxel_write[0] * SUB_CHUNK_COUNT as u32 / CHUNK_SIZE as u32,
                                voxel_write[1] * SUB_CHUNK_COUNT as u32 / CHUNK_SIZE as u32,
                                voxel_write[2] * SUB_CHUNK_COUNT as u32 / CHUNK_SIZE as u32,
                            ],
                            true,
                        );
                        delayed_writes.push(voxel_write);
                    }
                }
            }
            self.voxel_write_queue.extend(delayed_writes);
        }
        voxel_write_count
    }
    
    fn load_worldgen_updates(&mut self, game_state: &mut GameState, game_settings: &GameSettings) -> usize {
        let mut worldgen_update_count = 0;
        if !self.worldgen_update_queue.is_empty() && !self.available_chunks.is_empty() {
            let mut worldgen_updates_buffer = self.worldgen_updates.write().unwrap();
            let mut worldgen_results_buffer = self.worldgen_results.write().unwrap();
            while let Some((loc, forced)) = self.worldgen_update_queue.pop() {
                if is_inbounds!(loc, game_state, game_settings) {
                    let Some(available_chunk_idx) = self.available_chunks.pop() else {
                        self.worldgen_update_queue.push(loc, forced);
                        break;
                    };
                    worldgen_results_buffer[worldgen_update_count
                        / WORLDGEN_CHUNK_COUNT
                        / WORLDGEN_CHUNK_COUNT
                        / WORLDGEN_CHUNK_COUNT] = 1;
                    for i in 0..WORLDGEN_CHUNK_COUNT {
                        for j in 0..WORLDGEN_CHUNK_COUNT {
                            for k in 0..WORLDGEN_CHUNK_COUNT {
                                worldgen_updates_buffer[worldgen_update_count] = [
                                    (WORLDGEN_CHUNK_COUNT * loc[0] as usize + i) as i32,
                                    (WORLDGEN_CHUNK_COUNT * loc[1] as usize + j) as i32,
                                    (WORLDGEN_CHUNK_COUNT * loc[2] as usize + k) as i32,
                                    available_chunk_idx as i32 * if forced { -1 } else { 1 },
                                ];
                                worldgen_update_count += 1;
                            }
                        }
                    }
                    {
                        let adj_pos =
                            [0, 1, 2].map(|i| loc[i].rem_euclid(game_settings.render_size[i]));
                        self.cpu_chunks_copy[adj_pos[0] as usize][adj_pos[1] as usize]
                            [adj_pos[2] as usize] = available_chunk_idx;
                    };
                    for i in 0..SUB_CHUNK_COUNT {
                        for j in 0..SUB_CHUNK_COUNT {
                            for k in 0..SUB_CHUNK_COUNT {
                                self.chunk_update_queue.push_all([
                                    (SUB_CHUNK_COUNT * loc[0] as usize + i) as u32,
                                    (SUB_CHUNK_COUNT * loc[1] as usize + j) as u32,
                                    (SUB_CHUNK_COUNT * loc[2] as usize + k) as u32,
                                ]);
                            }
                        }
                    }
                    if worldgen_update_count as u32 >= game_settings.max_worldgen_rate {
                        break;
                    }
                } else {
                    self.oob_worldgens.insert(loc);
                }
            }
        }
    
        self.last_worldgen_count = worldgen_update_count
            / WORLDGEN_CHUNK_COUNT
            / WORLDGEN_CHUNK_COUNT
            / WORLDGEN_CHUNK_COUNT;
        worldgen_update_count
    }
    
    pub fn ensure_chunk_loaded(&mut self, chunk: Point3<u32>, game_settings: &GameSettings) {
        if self.get_chunk(chunk.into(), game_settings) == 0 {
            self.worldgen_update_queue
                .push([chunk[0], chunk[1], chunk[2]], false);
        }
    }
}

mod compute_projs_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/compute_projectile.glsl",
        include: ["assets/shaders"],
    }
}

mod unload_chunks_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/unload_chunks.glsl",
        include: ["assets/shaders"],
    }
}

mod write_voxels_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/write_voxels.glsl",
        include: ["assets/shaders"],
    }
}

mod compute_updates_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/compute_voxel.glsl",
        include: ["assets/shaders"],
    }
}

mod normal_world_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/maps/open_world.glsl",
        include: ["assets/shaders"],
    }
}

mod control_world_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/maps/control.glsl",
        include: ["assets/shaders"],
    }
}

mod complete_worldgen_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/complete_worldgen.glsl",
        include: ["assets/shaders"],
    }
}
