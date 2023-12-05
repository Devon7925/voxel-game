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
    game_manager::{GameSettings, GameState},
    utils::QueueSet,
    world_gen::WorldGen,
    CHUNK_SIZE, SUB_CHUNK_COUNT,
};
use noise::{Add, Constant, Multiply, NoiseFn, OpenSimplex, ScalePoint};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;
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
    compute_life_pipeline: Arc<ComputePipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    voxel_buffer: Subbuffer<[u32]>,
    chunk_update_queue: QueueSet<[u32; 3]>,
    chunk_updates: Subbuffer<[[u32; 4]; 256]>,
    uniform_buffer: SubbufferAllocator,
    world_gen: WorldGen,
    last_update_count: usize,
}

fn empty_grid(memory_allocator: Arc<StandardMemoryAllocator>, game_settings: &GameSettings) -> Subbuffer<[u32]> {
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
        vec![
            0;
            (CHUNK_SIZE
                * CHUNK_SIZE
                * CHUNK_SIZE
                * game_settings.render_size[0]
                * game_settings.render_size[1]
                * game_settings.render_size[2]) as usize
        ],
    )
    .unwrap()
}

impl VoxelComputePipeline {
    pub fn new(creation_interface: &CreationInterface, game_settings: &GameSettings) -> VoxelComputePipeline {
        let memory_allocator = &creation_interface.memory_allocator;

        // generate based on simplex noise
        const SCALE: f64 = 0.04;
        let world_density: Box<dyn NoiseFn<f64, 3> + Sync> = Box::new(Add::new(
            Add::new(
                ScalePoint::new(OpenSimplex::new(10)).set_scale(SCALE),
                Multiply::new(
                    ScalePoint::new(OpenSimplex::new(10)).set_scale(SCALE / 40.0),
                    Constant::new(5.0),
                ),
            ),
            Constant::new(-1.0),
        ));
        let pillar_density: Box<dyn NoiseFn<f64, 3> + Sync> = Box::new(
            ScalePoint::new(OpenSimplex::new(11))
                .set_scale(SCALE * 1.5)
                .set_y_scale(0.0),
        );
        let world_gen = WorldGen::new(world_density, pillar_density);

        let voxel_buffer = empty_grid(memory_allocator.clone(), game_settings);

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
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        )
        .unwrap();

        VoxelComputePipeline {
            compute_queue: creation_interface.queue.clone(),
            compute_life_pipeline,
            command_buffer_allocator: creation_interface.command_buffer_allocator.clone(),
            descriptor_set_allocator: creation_interface.descriptor_set_allocator.clone(),
            voxel_buffer,
            chunk_update_queue: QueueSet::with_capacity(256),
            chunk_updates,
            uniform_buffer,
            world_gen,
            last_update_count: 0,
        }
    }

    pub fn voxels(&self) -> Subbuffer<[u32]> {
        self.voxel_buffer.clone()
    }

    pub fn queue_updates(&mut self, queued: &[[u32; 3]]) {
        for item in queued {
            self.chunk_update_queue.push(*item);
        }
    }

    pub fn queue_update_from_world_pos(&mut self, queued: &[f32; 3]) {
        let chunk_location = [
            queued[0].floor() as u32 * SUB_CHUNK_COUNT / CHUNK_SIZE,
            queued[1].floor() as u32 * SUB_CHUNK_COUNT / CHUNK_SIZE,
            queued[2].floor() as u32 * SUB_CHUNK_COUNT / CHUNK_SIZE,
        ];
        self.chunk_update_queue.push(chunk_location);
    }

    pub fn queue_update_from_voxel_pos(&mut self, queued: &[u32; 3]) {
        let chunk_location = [
            queued[0] * SUB_CHUNK_COUNT / CHUNK_SIZE,
            queued[1] * SUB_CHUNK_COUNT / CHUNK_SIZE,
            queued[2] * SUB_CHUNK_COUNT / CHUNK_SIZE,
        ];
        self.chunk_update_queue.push(chunk_location);
    }

    fn get_chunk_idx(&self, pos: [u32; 3], game_settings: &GameSettings) -> usize {
        let adj_pos = [0, 1, 2].map(|i| pos[i].rem_euclid(game_settings.render_size[i]));
        (adj_pos[0] * game_settings.render_size[1] * game_settings.render_size[2]
            + adj_pos[1] * game_settings.render_size[2]
            + adj_pos[2]) as usize
    }

    fn write_chunk(&mut self, chunk_location: [u32; 3], game_settings: &GameSettings) {
        let mut chunk_buffer = self.voxel_buffer.write().unwrap();
        let chunk_idx = self.get_chunk_idx(chunk_location, game_settings);
        for (i, vox) in self
            .world_gen
            .gen_chunk(chunk_location)
            .into_iter()
            .enumerate()
        {
            chunk_buffer[chunk_idx
                * (CHUNK_SIZE as usize)
                * (CHUNK_SIZE as usize)
                * (CHUNK_SIZE as usize)
                + i] = vox;
        }
        for i in 0..SUB_CHUNK_COUNT {
            for j in 0..SUB_CHUNK_COUNT {
                for k in 0..SUB_CHUNK_COUNT {
                    self.chunk_update_queue.push([
                        chunk_location[0] * SUB_CHUNK_COUNT + i as u32,
                        chunk_location[1] * SUB_CHUNK_COUNT + j as u32,
                        chunk_location[2] * SUB_CHUNK_COUNT + k as u32,
                    ]);
                }
            }
        }
    }

    pub fn move_start_pos(&mut self, game_state: &mut GameState, offset: [i32; 3], game_settings: &GameSettings) {
        game_state.start_pos = [0, 1, 2].map(|i| game_state.start_pos[i].checked_add_signed(offset[i]).unwrap());

        let [load_range_x, load_range_y, load_range_z] = [0, 1, 2].map(|i| {
            if offset[i] > 0 {
                (game_settings.render_size[i] as i32) - offset[i]..=(game_settings.render_size[i] as i32) - 1
            } else {
                0..=-offset[i] - 1
            }
        });

        for x_offset in load_range_x.clone() {
            for y_i in 0..game_settings.render_size[1] {
                for z_i in 0..game_settings.render_size[2] {
                    let chunk_location = [
                        game_state.start_pos[0].wrapping_add_signed(x_offset),
                        game_state.start_pos[1] + y_i,
                        game_state.start_pos[2] + z_i,
                    ];
                    self.write_chunk(chunk_location, game_settings);
                }
            }
        }

        for x_i in 0..game_settings.render_size[0] {
            if load_range_x.contains(&(x_i as i32)) {
                continue;
            }
            for y_offset in load_range_y.clone() {
                for z_i in 0..game_settings.render_size[2] {
                    let chunk_location = [
                        game_state.start_pos[0] + x_i,
                        game_state.start_pos[1].wrapping_add_signed(y_offset),
                        game_state.start_pos[2] + z_i,
                    ];
                    self.write_chunk(chunk_location, game_settings);
                }
            }
        }

        for x_i in 0..game_settings.render_size[0] {
            if load_range_x.contains(&(x_i as i32)) {
                continue;
            }
            for y_i in 0..game_settings.render_size[1] {
                if load_range_y.contains(&(y_i as i32)) {
                    continue;
                }
                for z_offset in load_range_z.clone() {
                    let chunk_location = [
                        game_state.start_pos[0] + x_i,
                        game_state.start_pos[1] + y_i,
                        game_state.start_pos[2].wrapping_add_signed(z_offset),
                    ];
                    self.write_chunk(chunk_location, game_settings);
                }
            }
        }
    }

    pub fn load_chunks(&mut self, start_pos: [u32; 3], game_settings: &GameSettings) {
        let mut chunk_buffer = self.voxel_buffer.write().unwrap();
        for x_i in 0..game_settings.render_size[0] {
            for y_i in 0..game_settings.render_size[1] {
                let chunks: Vec<Vec<u32>> = (0..game_settings.render_size[2])
                    .into_par_iter()
                    .map(|z_i| {
                        let chunk_location = [
                            start_pos[0] + x_i,
                            start_pos[1] + y_i,
                            start_pos[2] + z_i,
                        ];
                        self.world_gen.gen_chunk(chunk_location)
                    })
                    .collect();
                for (z_i, chunk) in chunks.into_iter().enumerate() {
                    let chunk_location = [
                        start_pos[0] + x_i,
                        start_pos[1] + y_i,
                        start_pos[2] + z_i as u32,
                    ];
                    let chunk_idx = self.get_chunk_idx(chunk_location, game_settings);
                    for (i, vox) in chunk.into_iter().enumerate() {
                        chunk_buffer[chunk_idx
                            * (CHUNK_SIZE as usize)
                            * (CHUNK_SIZE as usize)
                            * (CHUNK_SIZE as usize)
                            + i] = vox;
                    }
                    for i in 0..SUB_CHUNK_COUNT {
                        for j in 0..SUB_CHUNK_COUNT {
                            for k in 0..SUB_CHUNK_COUNT {
                                self.chunk_update_queue.push([
                                    chunk_location[0] * SUB_CHUNK_COUNT + i,
                                    chunk_location[1] * SUB_CHUNK_COUNT + j,
                                    chunk_location[2] * SUB_CHUNK_COUNT + k,
                                ]);
                            }
                        }
                    }
                }
            }
        }
    }

    // chunks are represented as u32 with a 1 representing a changed chunk
    // this function will get the locations of those and push updates and then clear the buffer
    pub fn push_updates_from_changed(&mut self) {
        puffin::profile_function!();
        let reader = self.chunk_updates.read().unwrap();
        // last component of 1 means the chunk was changed and therefore means it and surrounding chunks need to be updated
        for i in 0..self.last_update_count {
            let read_update = reader[i];
            if read_update[3] == 1 {
                for x_offset in -1..=1 {
                    for y_offset in -1..=1 {
                        for z_offset in -1..=1 {
                            self.chunk_update_queue.push([
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

    fn queue_chunk_update(&mut self, chunk_location: [u32; 3]) {
        for chunk_i in 0..SUB_CHUNK_COUNT {
            for chunk_j in 0..SUB_CHUNK_COUNT {
                for chunk_k in 0..SUB_CHUNK_COUNT {
                    let i = chunk_location[0] * SUB_CHUNK_COUNT + chunk_i;
                    let j = chunk_location[1] * SUB_CHUNK_COUNT + chunk_j;
                    let k = chunk_location[2] * SUB_CHUNK_COUNT + chunk_k;
                    self.chunk_update_queue.push([i, j, k]);
                }
            }
        }
    }

    pub fn update_count(&self) -> usize {
        self.chunk_update_queue.len()
    }

    pub fn compute<F>(
        &mut self,
        before_future: F,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) -> Box<dyn GpuFuture>
    where
        F: GpuFuture + 'static,
    {
        puffin::profile_function!();
        if self.chunk_update_queue.is_empty() {
            return before_future.boxed();
        }
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
        self.dispatch(&mut builder, game_state, game_settings);

        let command_buffer = builder.build().unwrap();
        let finished = before_future
            .then_execute(self.compute_queue.clone(), command_buffer)
            .unwrap();
        let after_pipeline = finished.then_signal_fence_and_flush().unwrap().boxed();

        after_pipeline
    }

    /// Builds the command for a dispatch.
    fn dispatch(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) {
        // Resize image if needed.
        let pipeline_layout = self.compute_life_pipeline.layout();
        let desc_layout = pipeline_layout.set_layouts().get(0).unwrap();

        let uniform_buffer_subbuffer = {
            let uniform_data = compute_dists_cs::SimData {
                render_size: game_settings.render_size.into(),
                start_pos: game_state.start_pos.into(),
            };

            let subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
        };

        //send chunk updates
        let mut chunk_update_count = 0;
        {
            let mut chunk_updates_buffer = self.chunk_updates.write().unwrap();
            while let Some(loc) = self.chunk_update_queue.pop() {
                if loc[0] >= SUB_CHUNK_COUNT * game_state.start_pos[0]
                    && loc[0]
                        < SUB_CHUNK_COUNT
                            * (game_state.start_pos[0] + game_settings.render_size[0])
                    && loc[1] >= SUB_CHUNK_COUNT * game_state.start_pos[1]
                    && loc[1]
                        < SUB_CHUNK_COUNT
                            * (game_state.start_pos[1] + game_settings.render_size[1])
                    && loc[2] >= SUB_CHUNK_COUNT * game_state.start_pos[2]
                    && loc[2]
                        < SUB_CHUNK_COUNT
                            * (game_state.start_pos[2] + game_settings.render_size[2])
                {
                    chunk_updates_buffer[chunk_update_count] = [loc[0], loc[1], loc[2], 0];
                    chunk_update_count += 1;
                    if chunk_update_count >= 256 {
                        break;
                    }
                }
            }
        }
        self.last_update_count = chunk_update_count;

        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            desc_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, self.voxel_buffer.clone()),
                WriteDescriptorSet::buffer(1, self.chunk_updates.clone()),
                WriteDescriptorSet::buffer(2, uniform_buffer_subbuffer),
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

mod compute_dists_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/compute_voxel.glsl",
        include: ["assets/shaders"],
    }
}
