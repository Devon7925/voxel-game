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
    card_system::VoxelMaterial,
    game_manager::GameState,
    utils::{Direction, QueueSet, VoxelUpdateQueue},
    CHUNK_SIZE, MAX_CHUNK_UPDATE_RATE, MAX_WORLDGEN_RATE, SUB_CHUNK_COUNT, WORLDGEN_CHUNK_COUNT,
};
use cgmath::Point3;
use std::{iter, sync::Arc};
use voxel_shared::GameSettings;
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferUsage, PrimaryAutoCommandBuffer,
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

/// Pipeline holding double buffered grid & color image. Grids are used to calculate the state, and
/// color image is used to show the output. Because on each step we determine state in parallel, we
/// need to write the output to another grid. Otherwise the state would not be correctly determined
/// as one shader invocation might read data that was just written by another shader invocation.
pub struct VoxelComputePipeline {
    compute_queue: Arc<Queue>,
    unload_chunks_pipeline: Arc<ComputePipeline>,
    compute_life_pipeline: Arc<ComputePipeline>,
    compute_worldgen_pipeline: Arc<ComputePipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    cpu_chunks_copy: Vec<Vec<Vec<u32>>>,
    gpu_chunks: Arc<ImageView>,
    voxel_buffer: Subbuffer<[u32]>,
    available_chunks: Vec<u32>,
    slice_to_unload: Option<Direction>,
    chunk_update_queue: VoxelUpdateQueue,
    chunk_updates: Subbuffer<[[u32; 4]; MAX_CHUNK_UPDATE_RATE]>,
    worldgen_update_queue: QueueSet<[u32; 3]>,
    worldgen_updates: Subbuffer<[[u32; 4]; MAX_WORLDGEN_RATE]>,
    uniform_buffer: SubbufferAllocator,
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
        vec![0; (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * game_settings.max_loaded_chunks) as usize],
    )
    .unwrap()
}

impl VoxelComputePipeline {
    pub fn new(
        creation_interface: &CreationInterface,
        game_settings: &GameSettings,
    ) -> VoxelComputePipeline {
        let memory_allocator = &creation_interface.memory_allocator;

        let (cpu_chunks_copy, chunk_buffer) =
            empty_chunk_grid(memory_allocator.clone(), game_settings);
        let voxel_buffer = empty_list(memory_allocator.clone(), game_settings);
        // set the unloaded chunk
        {
            let mut voxel_writer = voxel_buffer.write().unwrap();
            for (i, _) in iter::repeat(())
                .take((CHUNK_SIZE as usize) * (CHUNK_SIZE as usize) * (CHUNK_SIZE as usize))
                .enumerate()
            {
                voxel_writer[i] = VoxelMaterial::Unloaded.to_memory();
            }
        }
        let available_chunks = (1..game_settings.max_loaded_chunks).collect();

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

        let compute_life_pipeline = {
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

        let uniform_buffer = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

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

        let worldgen_updates = Buffer::new_sized(
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

        VoxelComputePipeline {
            compute_queue: creation_interface.queue.clone(),
            unload_chunks_pipeline,
            compute_life_pipeline,
            compute_worldgen_pipeline,
            command_buffer_allocator: creation_interface.command_buffer_allocator.clone(),
            descriptor_set_allocator: creation_interface.descriptor_set_allocator.clone(),
            cpu_chunks_copy,
            gpu_chunks: chunk_buffer,
            voxel_buffer,
            available_chunks,
            chunk_update_queue: VoxelUpdateQueue::with_capacity(
                game_settings.max_update_rate as usize,
            ),
            chunk_updates,
            slice_to_unload: None,
            worldgen_update_queue: QueueSet::with_capacity(
                game_settings.max_worldgen_rate as usize,
            ),
            worldgen_updates,
            uniform_buffer,
            last_update_count: 0,
        }
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

    pub fn queue_update(&mut self, queued: [u32; 3], game_settings: &GameSettings) {
        let chunk = queued.map(|e| e / SUB_CHUNK_COUNT);
        if self.get_chunk(chunk, game_settings) == 0 {
            self.worldgen_update_queue
                .push([chunk[0], chunk[1], chunk[2]]);
        } else {
            self.chunk_update_queue.push_all(queued);
        }
    }

    pub fn queue_update_from_world_pos(
        &mut self,
        queued: &Point3<f32>,
        game_settings: &GameSettings,
    ) {
        let chunk_location = [
            queued[0].floor() as u32 * SUB_CHUNK_COUNT / CHUNK_SIZE,
            queued[1].floor() as u32 * SUB_CHUNK_COUNT / CHUNK_SIZE,
            queued[2].floor() as u32 * SUB_CHUNK_COUNT / CHUNK_SIZE,
        ];
        self.queue_update(chunk_location, game_settings);
    }

    pub fn queue_update_from_voxel_pos(&mut self, queued: &[u32; 3], game_settings: &GameSettings) {
        let chunk_location = [
            queued[0] * SUB_CHUNK_COUNT / CHUNK_SIZE,
            queued[1] * SUB_CHUNK_COUNT / CHUNK_SIZE,
            queued[2] * SUB_CHUNK_COUNT / CHUNK_SIZE,
        ];
        self.queue_update(chunk_location, game_settings);
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
                    //todo fix
                    game_state.start_pos[0],
                    game_state.start_pos[1],
                    game_state.start_pos[2],
                ];
                chunk_location[direction.component_index()] += load_idx;
                chunk_location[(direction.component_index() + 1) % 3] += y_i;
                chunk_location[(direction.component_index() + 2) % 3] += z_i;
                let current_chunk_ref = self.get_chunk(chunk_location, game_settings);
                if current_chunk_ref != 0 {
                    self.available_chunks.push(current_chunk_ref);
                }
                self.set_chunk(chunk_location, game_settings, 0);
            }
        }

        self.slice_to_unload = Some(direction);

        //remove queued updates that are now out of bounds
        self.worldgen_update_queue.keep_if(|loc| {
            loc[0] >= game_state.start_pos[0]
                && loc[0] < (game_state.start_pos[0] + game_settings.render_size[0])
                && loc[1] >= game_state.start_pos[1]
                && loc[1] < (game_state.start_pos[1] + game_settings.render_size[1])
                && loc[2] >= game_state.start_pos[2]
                && loc[2] < (game_state.start_pos[2] + game_settings.render_size[2])
        });
        self.chunk_update_queue.keep_if(|loc| {
            loc[0] >= SUB_CHUNK_COUNT * game_state.start_pos[0]
                && loc[0]
                    < SUB_CHUNK_COUNT * (game_state.start_pos[0] + game_settings.render_size[0])
                && loc[1] >= SUB_CHUNK_COUNT * game_state.start_pos[1]
                && loc[1]
                    < SUB_CHUNK_COUNT * (game_state.start_pos[1] + game_settings.render_size[1])
                && loc[2] >= SUB_CHUNK_COUNT * game_state.start_pos[2]
                && loc[2]
                    < SUB_CHUNK_COUNT * (game_state.start_pos[2] + game_settings.render_size[2])
        });
    }

    // chunks are represented as u32 with a 1 representing a changed chunk
    // this function will get the locations of those and push updates and then clear the buffer
    pub fn push_updates_from_changed(&mut self, game_state: &GameState, game_settings: &GameSettings) {
        puffin::profile_function!();
        // last component of 1 means the chunk was changed and therefore means it and surrounding chunks need to be updated
        let mut updates_todo = QueueSet::new();
        {
            puffin::profile_scope!("count updates");
            let reader = self.chunk_updates.read().unwrap();
            for i in 0..self.last_update_count {
                let read_update = reader[i];
                if read_update[3] == 1 {
                    for x_offset in -1..=1 {
                        for y_offset in -1..=1 {
                            for z_offset in -1..=1 {
                                updates_todo.push([
                                    read_update[0].wrapping_add_signed(x_offset),
                                    read_update[1].wrapping_add_signed(y_offset),
                                    read_update[2].wrapping_add_signed(z_offset),
                                ]);
                            }
                        }
                    }
                }
            }
        }
        for update in updates_todo.into_iter() {
            let chunk = update.map(|e| e / SUB_CHUNK_COUNT);
            let is_chunk_loaded = {
                let is_chunk_loaded = self.get_chunk(chunk, game_settings) != 0;
                is_chunk_loaded
            };
            if !is_chunk_loaded {
                //check if chunk is inbounds
                if chunk[0] >= game_state.start_pos[0]
                    && chunk[0] < (game_state.start_pos[0] + game_settings.render_size[0])
                    && chunk[1] >= game_state.start_pos[1]
                    && chunk[1] < (game_state.start_pos[1] + game_settings.render_size[1])
                    && chunk[2] >= game_state.start_pos[2]
                    && chunk[2] < (game_state.start_pos[2] + game_settings.render_size[2])
                {
                    self.worldgen_update_queue
                        .push([chunk[0], chunk[1], chunk[2]]);
                }
            } else {
                self.chunk_update_queue.push_all(update);
            }
        }
    }

    fn queue_chunk_update(&mut self, chunk_location: [u32; 3], game_settings: &GameSettings) {
        for chunk_i in 0..SUB_CHUNK_COUNT {
            for chunk_j in 0..SUB_CHUNK_COUNT {
                for chunk_k in 0..SUB_CHUNK_COUNT {
                    let i = chunk_location[0] * SUB_CHUNK_COUNT + chunk_i;
                    let j = chunk_location[1] * SUB_CHUNK_COUNT + chunk_j;
                    let k = chunk_location[2] * SUB_CHUNK_COUNT + chunk_k;
                    self.queue_update([i, j, k], game_settings);
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
        let early_pipeline = if let Some(direction_to_unload) = self.slice_to_unload.clone() {
            let mut builder = AutoCommandBufferBuilder::primary(
                self.command_buffer_allocator.as_ref(),
                self.compute_queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            self.dispatch_chunk_unload(&mut builder, game_state, game_settings, direction_to_unload);

            let command_buffer = builder.build().unwrap();
            self.slice_to_unload = None;
            before_future
                .then_execute(self.compute_queue.clone(), command_buffer)
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .boxed()
        } else {
            before_future.boxed()
        };
        let mid_pipeline = if self.worldgen_update_queue.is_empty() || self.available_chunks.is_empty() {
            early_pipeline.boxed()
        } else {
            let mut builder = AutoCommandBufferBuilder::primary(
                self.command_buffer_allocator.as_ref(),
                self.compute_queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            // Dispatch will mutate the builder adding commands which won't be sent before we build the
            // command buffer after dispatches. This will minimize the commands we send to the GPU. For
            // example, we could be doing tens of dispatches here depending on our needs. Maybe we
            // wanted to simulate 10 steps at a time...

            // First compute the next state.
            self.dispatch_worldgen(&mut builder, game_state, game_settings);

            let command_buffer = builder.build().unwrap();
            early_pipeline
                .then_execute(self.compute_queue.clone(), command_buffer)
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .boxed()
        };
        let finished_pipeline = if self.chunk_update_queue.is_empty() {
            mid_pipeline.boxed()
        } else {
            let mut builder = AutoCommandBufferBuilder::primary(
                self.command_buffer_allocator.as_ref(),
                self.compute_queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            // Dispatch will mutate the builder adding commands which won't be sent before we build the
            // command buffer after dispatches. This will minimize the commands we send to the GPU. For
            // example, we could be doing tens of dispatches here depending on our needs. Maybe we
            // wanted to simulate 10 steps at a time...

            // First compute the next state.
            self.dispatch_voxel_update(&mut builder, game_state, game_settings);

            let command_buffer = builder.build().unwrap();
            mid_pipeline
                .then_execute(self.compute_queue.clone(), command_buffer)
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .boxed()
        };

        finished_pipeline
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

            let subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
        };

        let sim_uniform_buffer_subbuffer = {
            let uniform_data = unload_chunks_cs::SimData {
                render_size: Into::<[u32; 3]>::into(game_settings.render_size).into(),
                start_pos: game_state.start_pos.into(),
            };

            let subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
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
                game_settings.render_size[(direction.component_index() + 1) % 3]/16,
                game_settings.render_size[(direction.component_index() + 2) % 3]/16,
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
            while let Some(loc) = self.worldgen_update_queue.pop() {
                if loc[0] >= game_state.start_pos[0]
                    && loc[0] < (game_state.start_pos[0] + game_settings.render_size[0])
                    && loc[1] >= game_state.start_pos[1]
                    && loc[1] < (game_state.start_pos[1] + game_settings.render_size[1])
                    && loc[2] >= game_state.start_pos[2]
                    && loc[2] < (game_state.start_pos[2] + game_settings.render_size[2])
                {
                    let Some(available_chunk_idx) = self.available_chunks.pop() else {
                        break;
                    };
                    for i in 0..WORLDGEN_CHUNK_COUNT {
                        for j in 0..WORLDGEN_CHUNK_COUNT {
                            for k in 0..WORLDGEN_CHUNK_COUNT {
                                worldgen_updates_buffer[worldgen_update_count] = [
                                    WORLDGEN_CHUNK_COUNT * loc[0] + i,
                                    WORLDGEN_CHUNK_COUNT * loc[1] + j,
                                    WORLDGEN_CHUNK_COUNT * loc[2] + k,
                                    available_chunk_idx,
                                ];
                                worldgen_update_count += 1;
                            }
                        }
                    }
                    {
                        let adj_pos = [0, 1, 2].map(|i| loc[i].rem_euclid(game_settings.render_size[i]));
                        self.cpu_chunks_copy[adj_pos[0] as usize][adj_pos[1] as usize][adj_pos[2] as usize] =
                            available_chunk_idx;
                    };
                    for i in 0..SUB_CHUNK_COUNT {
                        for j in 0..SUB_CHUNK_COUNT {
                            for k in 0..SUB_CHUNK_COUNT {
                                self.chunk_update_queue.push_all([
                                    SUB_CHUNK_COUNT * loc[0] + i,
                                    SUB_CHUNK_COUNT * loc[1] + j,
                                    SUB_CHUNK_COUNT * loc[2] + k,
                                ]);
                            }
                        }
                    }
                    if worldgen_update_count as u32 >= game_settings.max_worldgen_rate {
                        break;
                    }
                }
            }
        }

        let uniform_buffer_subbuffer = {
            let uniform_data = compute_worldgen_cs::SimData {
                render_size: Into::<[u32; 3]>::into(game_settings.render_size).into(),
                start_pos: game_state.start_pos.into(),
            };

            let subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
        };

        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            desc_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.gpu_chunks.clone()),
                WriteDescriptorSet::buffer(1, self.voxel_buffer.clone()),
                WriteDescriptorSet::buffer(2, self.worldgen_updates.clone()),
                WriteDescriptorSet::buffer(3, uniform_buffer_subbuffer),
            ],
            [],
        )
        .unwrap();
        builder
            .bind_pipeline_compute(self.compute_worldgen_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline_layout.clone(), 0, set)
            .unwrap()
            .dispatch([worldgen_update_count as u32, 1, 1])
            .unwrap();
    }

    fn dispatch_voxel_update(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) {
        // Resize image if needed.
        let pipeline_layout = self.compute_life_pipeline.layout();
        let desc_layout = pipeline_layout.set_layouts().get(0).unwrap();

        //send chunk updates
        let mut chunk_update_count = 0;
        {
            let mut chunk_updates_buffer = self.chunk_updates.write().unwrap();
            self.chunk_update_queue.swap_queue_set();
            while let Some(loc) = self.chunk_update_queue.pop() {
                if loc[0] >= SUB_CHUNK_COUNT * game_state.start_pos[0]
                    && loc[0]
                        < SUB_CHUNK_COUNT * (game_state.start_pos[0] + game_settings.render_size[0])
                    && loc[1] >= SUB_CHUNK_COUNT * game_state.start_pos[1]
                    && loc[1]
                        < SUB_CHUNK_COUNT * (game_state.start_pos[1] + game_settings.render_size[1])
                    && loc[2] >= SUB_CHUNK_COUNT * game_state.start_pos[2]
                    && loc[2]
                        < SUB_CHUNK_COUNT * (game_state.start_pos[2] + game_settings.render_size[2])
                {
                    chunk_updates_buffer[chunk_update_count] = [loc[0], loc[1], loc[2], 0];
                    chunk_update_count += 1;
                    if chunk_update_count as u32 >= game_settings.max_update_rate {
                        break;
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

            let subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
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
            .bind_pipeline_compute(self.compute_life_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline_layout.clone(), 0, set)
            .unwrap()
            .dispatch([chunk_update_count as u32, 1, 1])
            .unwrap();
    }
}

mod unload_chunks_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/unload_chunks.glsl",
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
