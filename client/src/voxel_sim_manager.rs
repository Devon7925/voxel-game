// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    app::CreationInterface,
    card_system::{CardManager, VoxelMaterial},
    game_manager::GameState,
    utils::{Direction, QueueMap, QueueSet, VoxelUpdateQueue},
    CHUNK_SIZE, MAX_CHUNK_UPDATE_RATE, MAX_VOXEL_UPDATE_RATE, MAX_WORLDGEN_RATE, SUB_CHUNK_COUNT,
    WORLDGEN_CHUNK_COUNT,
};
use bytemuck::{Pod, Zeroable};
use cgmath::{Point3, Quaternion};
use priority_queue::PriorityQueue;
use std::{collections::HashSet, iter, sync::Arc};
use voxel_shared::GameSettings;
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Queue,
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
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
}

/// Pipeline holding double buffered grid & color image. Grids are used to calculate the state, and
/// color image is used to show the output. Because on each step we determine state in parallel, we
/// need to write the output to another grid. Otherwise the state would not be correctly determined
/// as one shader invocation might read data that was just written by another shader invocation.
pub struct VoxelComputePipeline {
    compute_queue: Arc<Queue>,
    compute_proj_pipeline: Arc<ComputePipeline>,
    unload_chunks_pipeline: Arc<ComputePipeline>,
    write_voxels_pipeline: Arc<ComputePipeline>,
    compute_voxel_update_pipeline: Arc<ComputePipeline>,
    compute_worldgen_pipeline: Arc<ComputePipeline>,
    complete_worldgen_pipeline: Arc<ComputePipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    uniform_buffer: SubbufferAllocator,
    projectile_buffer: Subbuffer<[Projectile; 1024]>,
    cpu_chunks_copy: Vec<Vec<Vec<u32>>>,
    gpu_chunks: Arc<ImageView>,
    voxel_buffer: Subbuffer<[u32]>,
    available_chunks: Vec<u32>,
    slice_to_unload: Option<Direction>,
    chunk_update_queue: VoxelUpdateQueue,
    chunk_updates: Subbuffer<[[u32; 4]; MAX_CHUNK_UPDATE_RATE]>,
    voxel_write_queue: QueueSet<[u32; 4]>,
    voxel_writes: Subbuffer<[[u32; 4]; MAX_VOXEL_UPDATE_RATE]>,
    worldgen_update_queue: QueueMap<[u32; 3], bool>,
    oob_worldgens: HashSet<[u32; 3]>,
    worldgen_updates: Subbuffer<[[i32; 4]; MAX_WORLDGEN_RATE]>,
    worldgen_results: Subbuffer<
        [u32; MAX_WORLDGEN_RATE
            / WORLDGEN_CHUNK_COUNT
            / WORLDGEN_CHUNK_COUNT
            / WORLDGEN_CHUNK_COUNT],
    >,
    upload_projectile_count: usize,
    last_update_priorities: Vec<i32>,
    last_update_count: usize,
    last_worldgen_count: usize,
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
            usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC | ImageUsage::STORAGE,
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
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
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

        let compute_proj_pipeline = {
            let shader = compute_projs_cs::load(creation_interface.queue.device().clone()).unwrap();
            let device = creation_interface.queue.device();
            let cs = shader.entry_point("main").unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let unload_chunks_pipeline = {
            let shader = unload_chunks_cs::load(creation_interface.queue.device().clone()).unwrap();

            let cs = shader.entry_point("main").unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                creation_interface.queue.device().clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(creation_interface.queue.device().clone())
                    .unwrap(),
            )
            .unwrap();

            ComputePipeline::new(
                creation_interface.queue.device().clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let write_voxels_pipeline = {
            let shader = write_voxels_cs::load(creation_interface.queue.device().clone()).unwrap();

            let cs = shader.entry_point("main").unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                creation_interface.queue.device().clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(creation_interface.queue.device().clone())
                    .unwrap(),
            )
            .unwrap();

            ComputePipeline::new(
                creation_interface.queue.device().clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let compute_voxel_update_pipeline = {
            let shader = compute_dists_cs::load(creation_interface.queue.device().clone()).unwrap();

            let cs = shader.entry_point("main").unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                creation_interface.queue.device().clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(creation_interface.queue.device().clone())
                    .unwrap(),
            )
            .unwrap();

            ComputePipeline::new(
                creation_interface.queue.device().clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let compute_worldgen_pipeline = {
            let shader =
                compute_worldgen_cs::load(creation_interface.queue.device().clone()).unwrap();

            let cs = shader.entry_point("main").unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                creation_interface.queue.device().clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(creation_interface.queue.device().clone())
                    .unwrap(),
            )
            .unwrap();

            ComputePipeline::new(
                creation_interface.queue.device().clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let complete_worldgen_pipeline = {
            let shader =
                complete_worldgen_cs::load(creation_interface.queue.device().clone()).unwrap();

            let cs = shader.entry_point("main").unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                creation_interface.queue.device().clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(creation_interface.queue.device().clone())
                    .unwrap(),
            )
            .unwrap();

            ComputePipeline::new(
                creation_interface.queue.device().clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let uniform_buffer = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

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
            uniform_buffer,
            upload_projectile_count: 0,
            last_update_priorities: Vec::new(),
            last_update_count: 0,
            last_worldgen_count: 0,
        }
    }

    pub fn projectiles(&self) -> Subbuffer<[Projectile; 1024]> {
        self.projectile_buffer.clone()
    }

    pub fn upload(&mut self, projectiles: &Vec<Projectile>) {
        //send projectiles
        self.upload_projectile_count = 1024.min(projectiles.len());
        let mut projectiles_buffer = self.projectile_buffer.write().unwrap();
        for i in 0..self.upload_projectile_count {
            let projectile = projectiles.get(i).unwrap();
            projectiles_buffer[i] = projectile.clone();
        }
    }

    pub fn download_projectiles(
        &mut self,
        card_manager: &CardManager,
        game_settings: &GameSettings,
    ) -> Vec<Projectile> {
        let mut projectiles = Vec::new();
        let mut new_voxels = Vec::new();
        let projectiles_buffer = self.projectile_buffer.read().unwrap();
        for i in 0..self.upload_projectile_count {
            let projectile = projectiles_buffer[i];
            if projectile.health == 0.0 && projectile.chunk_update_pos[3] == 1 {
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
                        self.chunk_update_queue.push_with_priority(chunk_location, 1);
                    }
                };
                for card_ref in card_manager
                    .get_referenced_proj(projectile.proj_card_idx as usize)
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
                continue;
            }
            projectiles.push(projectile);
        }

        if new_voxels.len() > 0 {
            for (pos, material) in new_voxels {
                self.voxel_write_queue
                    .push([pos[0], pos[1], pos[2], material.to_memory()]);
            }
        }

        projectiles
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
    pub fn push_updates_from_changed(
        &mut self,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) {
        puffin::profile_function!();
        if self.last_worldgen_count > 0 {
            let worldgen_results = self.worldgen_results.read().unwrap();
            let last_worldgen = self.worldgen_updates.read().unwrap();
            for i in 0..self.last_worldgen_count {
                if worldgen_results[i] == 1 {
                    let last_worldgen_chunk = last_worldgen[8 * i];
                    self.available_chunks.push(last_worldgen_chunk[3].abs() as u32);
                    self.cpu_chunks_copy
                        [((last_worldgen_chunk[0] as u32 / 2) % game_settings.render_size[0]) as usize]
                        [((last_worldgen_chunk[1] as u32 / 2) % game_settings.render_size[1]) as usize]
                        [((last_worldgen_chunk[2] as u32 / 2) % game_settings.render_size[2]) as usize] =
                        1;
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
                    let min_x = -((read_update[3] as i32 >> 1) & 1);
                    let min_y = -((read_update[3] as i32 >> 2) & 1);
                    let min_z = -((read_update[3] as i32 >> 3) & 1);
                    let max_x = (read_update[3] as i32 >> 4) & 1;
                    let max_y = (read_update[3] as i32 >> 5) & 1;
                    let max_z = (read_update[3] as i32 >> 6) & 1;
                    for x_offset in min_x..=max_x {
                        for y_offset in min_y..=max_y {
                            for z_offset in min_z..=max_z {
                                self.chunk_update_queue.push_with_priority(
                                    [
                                        read_update[0].wrapping_add_signed(x_offset),
                                        read_update[1].wrapping_add_signed(y_offset),
                                        read_update[2].wrapping_add_signed(z_offset),
                                    ],
                                    self.last_update_priorities[i] - 1,
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
    ) -> Box<dyn GpuFuture>
    where
        F: GpuFuture + 'static,
    {
        puffin::profile_function!();
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.as_ref(),
            self.compute_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        if self.upload_projectile_count > 0 {
            self.dispatch_projectiles(&mut builder, game_state, game_settings);
        }

        if let Some(direction_to_unload) = self.slice_to_unload.clone() {
            self.dispatch_chunk_unload(
                &mut builder,
                game_state,
                game_settings,
                direction_to_unload,
            );

            self.slice_to_unload = None;
        }
        self.last_worldgen_count = 0;

        if !self.worldgen_update_queue.is_empty() && !self.available_chunks.is_empty() {
            self.dispatch_worldgen(&mut builder, game_state, game_settings);
        };
        if !self.voxel_write_queue.is_empty() {
            self.dispatch_voxel_write(&mut builder, game_state, game_settings);
        };
        if !self.chunk_update_queue.is_empty() {
            self.dispatch_voxel_update(&mut builder, game_state, game_settings);
        };
        let command_buffer = builder.build().unwrap();
        before_future
            .then_execute(self.compute_queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .boxed()
    }

    fn dispatch_projectiles(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) {
        // Resize image if needed.
        let pipeline_layout = self.compute_proj_pipeline.layout();
        let desc_layout = pipeline_layout.set_layouts().get(0).unwrap();

        let uniform_buffer_subbuffer = {
            let uniform_data = compute_projs_cs::SimData {
                render_size: Into::<[u32; 3]>::into(game_settings.render_size).into(),
                start_pos: game_state.start_pos.into(),
                dt: game_settings.delta_time,
                projectile_count: self.upload_projectile_count as u32,
            };

            let uniform_subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *uniform_subbuffer.write().unwrap() = uniform_data;

            uniform_subbuffer
        };
        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            desc_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.chunks()),
                WriteDescriptorSet::buffer(1, self.voxels()),
                WriteDescriptorSet::buffer(2, self.projectile_buffer.clone()),
                WriteDescriptorSet::buffer(3, uniform_buffer_subbuffer),
            ],
            [],
        )
        .unwrap();
        builder
            .bind_pipeline_compute(self.compute_proj_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline_layout.clone(), 0, set)
            .unwrap()
            .dispatch([((self.upload_projectile_count + 127) / 128) as u32, 1, 1])
            .unwrap();
    }

    /// Builds the command for a dispatch.
    fn dispatch_chunk_unload(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        game_state: &GameState,
        game_settings: &GameSettings,
        direction: Direction,
    ) {
        // Resize image if needed.
        let pipeline_layout = self.unload_chunks_pipeline.layout();
        let desc_layout = pipeline_layout.set_layouts().get(0).unwrap();

        let slice_uniform_buffer_subbuffer = {
            let uniform_data = unload_chunks_cs::SliceData {
                index: if direction.is_positive() {
                    game_state.start_pos[direction.component_index()]
                        + game_settings.render_size[direction.component_index()]
                        - 1
                } else {
                    game_state.start_pos[direction.component_index()]
                } as i32,
                component: direction.component_index() as u32,
            };

            let uniform_subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *uniform_subbuffer.write().unwrap() = uniform_data;

            uniform_subbuffer
        };

        let sim_uniform_buffer_subbuffer = {
            let uniform_data = unload_chunks_cs::SimData {
                render_size: Into::<[u32; 3]>::into(game_settings.render_size).into(),
                start_pos: game_state.start_pos.into(),
            };

            let uniform_subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *uniform_subbuffer.write().unwrap() = uniform_data;

            uniform_subbuffer
        };

        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            desc_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.gpu_chunks.clone()),
                WriteDescriptorSet::buffer(1, slice_uniform_buffer_subbuffer),
                WriteDescriptorSet::buffer(2, sim_uniform_buffer_subbuffer),
            ],
            [],
        )
        .unwrap();
        builder
            .bind_pipeline_compute(self.unload_chunks_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline_layout.clone(), 0, set)
            .unwrap()
            .dispatch([
                game_settings.render_size[(direction.component_index() + 1) % 3] / 16,
                game_settings.render_size[(direction.component_index() + 2) % 3] / 16,
                1,
            ])
            .unwrap();
    }

    /// Builds the command for a dispatch.
    fn dispatch_worldgen(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) {
        // Resize image if needed.
        let pipeline_layout = self.compute_worldgen_pipeline.layout();
        let desc_layout = pipeline_layout.set_layouts().get(0).unwrap();

        //send chunk updates
        let mut worldgen_update_count = 0;
        {
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

        let uniform_buffer_subbuffer = {
            let uniform_data = compute_worldgen_cs::SimData {
                render_size: Into::<[u32; 3]>::into(game_settings.render_size).into(),
                start_pos: game_state.start_pos.into(),
                count: self.last_worldgen_count as u32,
            };

            let uniform_subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *uniform_subbuffer.write().unwrap() = uniform_data;

            uniform_subbuffer
        };

        let compute_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            desc_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, self.voxel_buffer.clone()),
                WriteDescriptorSet::buffer(1, self.worldgen_updates.clone()),
                WriteDescriptorSet::buffer(2, uniform_buffer_subbuffer),
                WriteDescriptorSet::buffer(3, self.worldgen_results.clone()),
            ],
            [],
        )
        .unwrap();
        builder
            .bind_pipeline_compute(self.compute_worldgen_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline_layout.clone(),
                0,
                compute_set,
            )
            .unwrap()
            .dispatch([worldgen_update_count as u32, 1, 1])
            .unwrap();

        let pipeline_layout = self.complete_worldgen_pipeline.layout();
        let desc_layout = pipeline_layout.set_layouts().get(0).unwrap();
        let complete_uniform_buffer_subbuffer = {
            let uniform_data = complete_worldgen_cs::SimData {
                render_size: Into::<[u32; 3]>::into(game_settings.render_size).into(),
                start_pos: game_state.start_pos.into(),
                count: self.last_worldgen_count as u32,
            };

            let uniform_subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *uniform_subbuffer.write().unwrap() = uniform_data;

            uniform_subbuffer
        };

        let complete_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            desc_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.gpu_chunks.clone()),
                WriteDescriptorSet::buffer(1, self.worldgen_updates.clone()),
                WriteDescriptorSet::buffer(2, complete_uniform_buffer_subbuffer),
                WriteDescriptorSet::buffer(3, self.worldgen_results.clone()),
            ],
            [],
        )
        .unwrap();
        builder
            .bind_pipeline_compute(self.complete_worldgen_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline_layout.clone(),
                0,
                complete_set,
            )
            .unwrap()
            .dispatch([
                (self.last_worldgen_count as f32 / 256.0).ceil() as u32,
                1,
                1,
            ])
            .unwrap();
    }

    fn dispatch_voxel_write(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) {
        // Resize image if needed.
        let pipeline_layout = self.write_voxels_pipeline.layout();
        let desc_layout = pipeline_layout.set_layouts().get(0).unwrap();

        //send chunk updates
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
                        self.worldgen_update_queue.push([
                            voxel_write[0] * SUB_CHUNK_COUNT as u32 / CHUNK_SIZE as u32,
                            voxel_write[1] * SUB_CHUNK_COUNT as u32 / CHUNK_SIZE as u32,
                            voxel_write[2] * SUB_CHUNK_COUNT as u32 / CHUNK_SIZE as u32,
                        ], true);
                        delayed_writes.push(voxel_write);
                    }
                }
            }
            self.voxel_write_queue.extend(delayed_writes);
        }

        let uniform_buffer_subbuffer = {
            let uniform_data = write_voxels_cs::SimData {
                render_size: Into::<[u32; 3]>::into(game_settings.render_size).into(),
                start_pos: Into::<[u32; 3]>::into(game_state.start_pos).into(),
                count: voxel_write_count as u32,
            };

            let uniform_subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *uniform_subbuffer.write().unwrap() = uniform_data;

            uniform_subbuffer
        };

        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            desc_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.gpu_chunks.clone()),
                WriteDescriptorSet::buffer(1, self.voxel_buffer.clone()),
                WriteDescriptorSet::buffer(2, self.voxel_writes.clone()),
                WriteDescriptorSet::buffer(3, uniform_buffer_subbuffer),
            ],
            [],
        )
        .unwrap();
        builder
            .bind_pipeline_compute(self.write_voxels_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline_layout.clone(), 0, set)
            .unwrap()
            .dispatch([(voxel_write_count as f32 / 256.0).ceil() as u32, 1, 1])
            .unwrap();
    }

    fn dispatch_voxel_update(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) {
        // Resize image if needed.
        let pipeline_layout = self.compute_voxel_update_pipeline.layout();
        let desc_layout = pipeline_layout.set_layouts().get(0).unwrap();

        self.last_update_priorities.clear();

        //send chunk updates
        let mut chunk_update_count = 0;
        {
            let mut chunk_updates_buffer = self.chunk_updates.write().unwrap();
            self.chunk_update_queue.swap_queue_set();
            while let Some(loc) = self.chunk_update_queue.pop() {
                if is_inbounds!(loc.0.map(|c| c / SUB_CHUNK_COUNT as u32), game_state, game_settings) {
                    let chunk = self.get_chunk(
                        [
                            loc.0[0] / SUB_CHUNK_COUNT as u32,
                            loc.0[1] / SUB_CHUNK_COUNT as u32,
                            loc.0[2] / SUB_CHUNK_COUNT as u32,
                        ],
                        game_settings,
                    );
                    if chunk == 0 {
                        self.worldgen_update_queue.push([
                            loc.0[0] / SUB_CHUNK_COUNT as u32,
                            loc.0[1] / SUB_CHUNK_COUNT as u32,
                            loc.0[2] / SUB_CHUNK_COUNT as u32,
                        ], false);
                    } else if chunk != 1 {
                        chunk_updates_buffer[chunk_update_count] = [loc.0[0], loc.0[1], loc.0[2], 0];
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

        let uniform_buffer_subbuffer = {
            let uniform_data = compute_dists_cs::SimData {
                render_size: Into::<[u32; 3]>::into(game_settings.render_size).into(),
                start_pos: Into::<[u32; 3]>::into(game_state.start_pos).into(),
                update_offset: self.chunk_update_queue.queue_set_idx().into(),
            };

            let uniform_subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *uniform_subbuffer.write().unwrap() = uniform_data;

            uniform_subbuffer
        };

        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            desc_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.gpu_chunks.clone()),
                WriteDescriptorSet::buffer(1, self.voxel_buffer.clone()),
                WriteDescriptorSet::buffer(2, self.chunk_updates.clone()),
                WriteDescriptorSet::buffer(3, uniform_buffer_subbuffer),
            ],
            [],
        )
        .unwrap();
        builder
            .bind_pipeline_compute(self.compute_voxel_update_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline_layout.clone(), 0, set)
            .unwrap()
            .dispatch([chunk_update_count as u32, 1, 1])
            .unwrap();
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

mod compute_dists_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/compute_voxel.glsl",
        include: ["assets/shaders"],
    }
}

mod compute_worldgen_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/compute_worldgen.glsl",
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
