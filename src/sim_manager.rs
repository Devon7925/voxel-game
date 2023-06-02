// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{app::App, SimData, GRID_SIZE};
use noise::{NoiseFn, OpenSimplex};
use std::{sync::Arc, collections::VecDeque};
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
    life_a: Subbuffer<[[u32;2]]>,
    life_b: Subbuffer<[[u32;2]]>,
    chunk_update_queue: VecDeque<[i32;3]>,
    chunk_updates: Subbuffer<[[i32;3];128]>,
    uniform_buffer: SubbufferAllocator,
}

fn rand_grid(memory_allocator: &impl MemoryAllocator, size: [u32; 3]) -> Subbuffer<[[u32;2]]> {
    // generate based on simplex noise
    let noise = OpenSimplex::new(10);
    const SCALE:f64 = 0.1;
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
        (0..size[0])
            .flat_map(|x| {
                if x % 10 == 0 {
                    println!("Generating x: {}", x);
                }
                (0..size[1])
                    .flat_map(|y| {
                        (0..size[1])
                            .map(|z| {
                                if noise.get([x as f64 * SCALE, z as f64 * SCALE])*20.0 > (y as f64 - size[1] as f64 / 2.0) {
                                    [1, 0x00000000]
                                } else {
                                    [0, 0x11111111]
                                }
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
        let life_a = rand_grid(memory_allocator, GRID_SIZE);
        let life_b = life_a.clone();

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
            }
        ).unwrap();

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
        }
    }

    pub fn voxels(&self) -> Subbuffer<[[u32;2]]> {
        self.life_b.clone()
    }

    pub fn queue_updates(&mut self, queued: &[[i32;3]]) {
        self.chunk_update_queue.extend(queued);
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

        sim_data.is_a_in_buffer = !sim_data.is_a_in_buffer;

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
                grid_size: sim_data.grid_size.into(),
                is_a_in_buffer: sim_data.is_a_in_buffer.into(),
                start_pos: sim_data.start_pos.into(),
            };

            let subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
        };

        //send chunk updates
        {
            let mut chunk_updates_buffer = self.chunk_updates.write().unwrap();
            for i in 0..127.max(self.chunk_update_queue.len()-1) {
                chunk_updates_buffer[i] = self.chunk_update_queue.pop_front().unwrap();
            }
        }

        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            desc_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, self.life_a.clone()),
                WriteDescriptorSet::buffer(1, self.life_b.clone()),
                WriteDescriptorSet::buffer(2, self.chunk_updates.clone()),
                WriteDescriptorSet::buffer(3, uniform_buffer_subbuffer),
            ],
        )
        .unwrap();
        builder
            .bind_pipeline_compute(self.compute_life_pipeline.clone())
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline_layout.clone(), 0, set)
            .dispatch([self.chunk_update_queue.len() as u32, 1, 1])
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