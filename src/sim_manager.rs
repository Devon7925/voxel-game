// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{app::App, world_gen::WorldGen, SimData, CHUNK_SIZE, RENDER_SIZE};
use noise::{NoiseFn, OpenSimplex, ScalePoint, Multiply, Add, Constant};
use std::{collections::VecDeque, sync::Arc};
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
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryUsage},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::GpuFuture,
};

/// Pipeline holding double buffered grid & color image. Grids are used to calculate the state, and
/// color image is used to show the output. Because on each step we determine state in parallel, we
/// need to write the output to another grid. Otherwise the state would not be correctly determined
/// as one shader invocation might read data that was just written by another shader invocation.
pub struct DistanceComputePipeline {
    compute_queue: Arc<Queue>,
    compute_life_pipeline: Arc<ComputePipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    life_a: Subbuffer<[[u32; 2]]>,
    life_b: Subbuffer<[[u32; 2]]>,
    is_a_in_buffer: bool,
    chunk_update_queue: VecDeque<[i32; 3]>,
    chunk_updates: Subbuffer<[[i32; 4]; 128]>,
    uniform_buffer: SubbufferAllocator,
    world_gen: WorldGen,
}

fn empty_grid(
    memory_allocator: &impl MemoryAllocator,
) -> Subbuffer<[[u32; 2]]> {
    Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        (0..RENDER_SIZE[0])
            .flat_map(|chunk_x| {
                if chunk_x % 3 == 0 {
                    println!("Generating chunk_x: {}", chunk_x);
                }
                (0..RENDER_SIZE[1])
                    .flat_map(|_chunk_y| {
                        (0..RENDER_SIZE[2])
                            .flat_map(|_chunk_z| {
                                vec![[0, 0x11111111]; (CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE) as usize]
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

impl DistanceComputePipeline {
    pub fn new(app: &App, compute_queue: Arc<Queue>) -> DistanceComputePipeline {
        let memory_allocator = app.context.memory_allocator();

        // generate based on simplex noise
        const SCALE: f64 = 0.2;
        let noise: Box<dyn NoiseFn<f64, 3>> =
            Box::new(Add::new(
                Add::new(
                    ScalePoint::new(OpenSimplex::new(10)).set_scale(SCALE),
                    Multiply::new(
                        ScalePoint::new(OpenSimplex::new(10)).set_scale(SCALE/40.0).set_y_scale(SCALE/12.0),
                        Constant::new(5.0),
                    ),
                ),
                Constant::new(-1.0),
            ));
        let world_gen = WorldGen::new(noise);

        let life_a = empty_grid(memory_allocator);
        let life_b = empty_grid(memory_allocator);

        let compute_life_pipeline = {
            let shader = compute_dists_cs::load(compute_queue.device().clone()).unwrap();
            ComputePipeline::new(
                compute_queue.device().clone(),
                shader.entry_point("main").unwrap(),
                &(),
                None,
                |_| {},
            )
            .unwrap()
        };

        let uniform_buffer = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
        );

        let chunk_updates = Buffer::new_sized(
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

        DistanceComputePipeline {
            compute_queue,
            compute_life_pipeline,
            command_buffer_allocator: app.command_buffer_allocator.clone(),
            descriptor_set_allocator: app.descriptor_set_allocator.clone(),
            life_a,
            life_b,
            chunk_update_queue: VecDeque::new(),
            chunk_updates,
            uniform_buffer,
            world_gen,
            is_a_in_buffer: true,
        }
    }

    pub fn voxels(&self) -> Subbuffer<[[u32; 2]]> {
        if self.is_a_in_buffer {
            self.life_b.clone()
        } else {
            self.life_a.clone()
        }
    }

    pub fn queue_updates(&mut self, queued: &[[i32; 3]]) {
        self.chunk_update_queue.extend(queued);
    }

    fn get_chunk_idx(&self, pos: [i32; 3]) -> usize {
        let adj_pos = [0, 1, 2].map(|i| pos[i].rem_euclid(RENDER_SIZE[i] as i32));
        (adj_pos[0] * (RENDER_SIZE[1] as i32) * (RENDER_SIZE[2] as i32)
            + adj_pos[1] * (RENDER_SIZE[2] as i32)
            + adj_pos[2]) as usize
    }

    fn write_chunk(&mut self, chunk_location: [i32; 3]) {
        let mut a_chunk_buffer = self.life_a.write().unwrap();
        let mut b_chunk_buffer = self.life_b.write().unwrap();
        let chunk_idx = self.get_chunk_idx(chunk_location);
        for (i, vox) in self
            .world_gen
            .gen_chunk(chunk_location)
            .into_iter()
            .enumerate()
        {
            a_chunk_buffer[chunk_idx
                * (CHUNK_SIZE as usize)
                * (CHUNK_SIZE as usize)
                * (CHUNK_SIZE as usize)
                + i] = vox;
            b_chunk_buffer[chunk_idx
                * (CHUNK_SIZE as usize)
                * (CHUNK_SIZE as usize)
                * (CHUNK_SIZE as usize)
                + i] = vox;
        }
        for i in 0..CHUNK_SIZE / 8 {
            for j in 0..CHUNK_SIZE / 8 {
                for k in 0..CHUNK_SIZE / 8 {
                    self.chunk_update_queue.push_back([
                        chunk_location[0] * (CHUNK_SIZE / 8) as i32 + i as i32,
                        chunk_location[1] * (CHUNK_SIZE / 8) as i32 + j as i32,
                        chunk_location[2] * (CHUNK_SIZE / 8) as i32 + k as i32,
                    ]);
                }
            }
        }
    }

    pub fn move_start_pos(&mut self, sim_data: &mut SimData, offset: [i32; 3]) {
        sim_data.start_pos = [0, 1, 2].map(|i| sim_data.start_pos[i] + offset[i]);
        
        let [load_range_x, load_range_y, load_range_z] = [0, 1, 2].map(|i| if offset[i] > 0 {
            (RENDER_SIZE[i] as i32) - offset[i]..=(RENDER_SIZE[i] as i32) - 1
        } else {
            0..=-offset[i]-1
        });

        for x_offset in load_range_x.clone() {
            for y_i in 0..RENDER_SIZE[1] {
                for z_i in 0..RENDER_SIZE[2] {
                    let chunk_location = [
                        sim_data.start_pos[0] + x_offset,
                        sim_data.start_pos[1] + y_i as i32,
                        sim_data.start_pos[2] + z_i as i32,
                    ];
                    self.write_chunk(chunk_location);
                }
            }
        }

        for x_i in 0..RENDER_SIZE[0] as i32 {
            if load_range_x.contains(&(x_i as i32)) {
                continue;
            }
            for y_offset in load_range_y.clone() {
                for z_i in 0..RENDER_SIZE[2] {
                    let chunk_location = [
                        sim_data.start_pos[0] + x_i as i32,
                        sim_data.start_pos[1] + y_offset,
                        sim_data.start_pos[2] + z_i as i32,
                    ];
                    self.write_chunk(chunk_location);
                }
            }
        }

        for x_i in 0..RENDER_SIZE[0] {
            if load_range_x.contains(&(x_i as i32)) {
                continue;
            }
            for y_i in 0..RENDER_SIZE[1] {
                if load_range_y.contains(&(y_i as i32)) {
                    continue;
                }
                for z_offset in load_range_z.clone() {
                    let chunk_location = [
                        sim_data.start_pos[0] + x_i as i32,
                        sim_data.start_pos[1] + y_i as i32,
                        sim_data.start_pos[2] + z_offset,
                    ];
                    self.write_chunk(chunk_location);
                }
            }
        }
    }

    pub fn load_chunks(&mut self, sim_data: &mut SimData) {
        for x_i in 0..RENDER_SIZE[0] {
            for y_i in 0..RENDER_SIZE[1] {
                for z_i in 0..RENDER_SIZE[2] {
                    let chunk_location = [
                        sim_data.start_pos[0] + x_i as i32,
                        sim_data.start_pos[1] + y_i as i32,
                        sim_data.start_pos[2] + z_i as i32,
                    ];
                    self.write_chunk(chunk_location);
                }
            }
        }
    }

    pub fn compute(
        &mut self,
        before_future: Box<dyn GpuFuture>,
        sim_data: &mut SimData,
    ) -> Box<dyn GpuFuture> {
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.compute_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Dispatch will mutate the builder adding commands which won't be sent before we build the
        // command buffer after dispatches. This will minimize the commands we send to the GPU. For
        // example, we could be doing tens of dispatches here depending on our needs. Maybe we
        // wanted to simulate 10 steps at a time...

        // First compute the next state.
        self.dispatch(&mut builder, sim_data);

        let command_buffer = builder.build().unwrap();
        let finished = before_future
            .then_execute(self.compute_queue.clone(), command_buffer)
            .unwrap();
        let after_pipeline = finished.then_signal_fence_and_flush().unwrap().boxed();

        self.is_a_in_buffer = !self.is_a_in_buffer;

        after_pipeline
    }

    /// Builds the command for a dispatch.
    fn dispatch(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        sim_data: &mut SimData,
    ) {
        // Resize image if needed.
        let pipeline_layout = self.compute_life_pipeline.layout();
        let desc_layout = pipeline_layout.set_layouts().get(0).unwrap();

        let uniform_buffer_subbuffer = {
            let uniform_data = compute_dists_cs::SimData {
                max_dist: sim_data.max_dist.into(),
                render_size: sim_data.render_size.into(),
                start_pos: sim_data.start_pos.into(),
            };

            let subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
        };

        //send chunk updates
        let chunk_update_count = 128.min(self.chunk_update_queue.len());
        {
            let mut chunk_updates_buffer = self.chunk_updates.write().unwrap();
            for i in 0..chunk_update_count {
                let new_update = self.chunk_update_queue.pop_front().unwrap();
                chunk_updates_buffer[i] = [new_update[0], new_update[1], new_update[2], 0];
            }
        }

        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            desc_layout.clone(),
            [
                WriteDescriptorSet::buffer(
                    0,
                    if self.is_a_in_buffer {
                        self.life_a.clone()
                    } else {
                        self.life_b.clone()
                    },
                ),
                WriteDescriptorSet::buffer(
                    1,
                    if self.is_a_in_buffer {
                        self.life_b.clone()
                    } else {
                        self.life_a.clone()
                    },
                ),
                WriteDescriptorSet::buffer(2, self.chunk_updates.clone()),
                WriteDescriptorSet::buffer(3, uniform_buffer_subbuffer),
            ],
        )
        .unwrap();
        builder
            .bind_pipeline_compute(self.compute_life_pipeline.clone())
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline_layout.clone(), 0, set)
            .dispatch([chunk_update_count as u32, 1, 1])
            .unwrap();
    }
}

mod compute_dists_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/comp.glsl",
        include: ["src"],
    }
}
