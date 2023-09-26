// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    app::VulkanoInterface, card_system::CardManager, voxel_sim_manager::VoxelComputePipeline,
    SimData,
};
use bytemuck::{Pod, Zeroable};
use cgmath::{Quaternion, Vector3};
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
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::GpuFuture,
};

/// Pipeline holding double buffered grid & color image. Grids are used to calculate the state, and
/// color image is used to show the output. Because on each step we determine state in parallel, we
/// need to write the output to another grid. Otherwise the state would not be correctly determined
/// as one shader invocation might read data that was just written by another shader invocation.
pub struct ProjectileComputePipeline {
    compute_queue: Arc<Queue>,
    compute_proj_pipeline: Arc<ComputePipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    uniform_buffer: SubbufferAllocator,
    pub projectile_buffer: Subbuffer<[Projectile; 1024]>,
    pub upload_projectile_count: usize,
}

#[derive(Clone, Copy, Zeroable, Debug, Pod)]
#[repr(C)]
pub struct Projectile {
    pub pos: [f32; 4],
    pub chunk_update_pos: [i32; 4],
    pub dir: [f32; 4],
    pub size: [f32; 4],
    pub vel: f32,
    pub health: f32,
    pub lifetime: f32,
    pub owner: u32,
    pub damage: f32,
    pub proj_card_idx: u32,
    pub _filler2: f32,
    pub _filler3: f32,
}

impl ProjectileComputePipeline {
    pub fn new(app: &VulkanoInterface, compute_queue: Arc<Queue>) -> ProjectileComputePipeline {
        let memory_allocator = &app.memory_allocator;

        let compute_proj_pipeline = {
            let shader = compute_projs_cs::load(compute_queue.device().clone()).unwrap();
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

        ProjectileComputePipeline {
            compute_queue,
            compute_proj_pipeline,
            command_buffer_allocator: app.command_buffer_allocator.clone(),
            descriptor_set_allocator: app.descriptor_set_allocator.clone(),
            uniform_buffer,
            projectile_buffer,
            upload_projectile_count: 0,
        }
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

    pub fn compute<F>(
        &mut self,
        before_future: F,
        voxel_data: &VoxelComputePipeline,
        sim_data: &mut SimData,
        time_step: f32,
    ) -> Box<dyn GpuFuture>
    where
        F: GpuFuture + 'static,
    {
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
        self.dispatch(&mut builder, voxel_data, sim_data, time_step);

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
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        voxel_data: &VoxelComputePipeline,
        sim_data: &mut SimData,
        time_step: f32,
    ) {
        // Resize image if needed.
        let pipeline_layout = self.compute_proj_pipeline.layout();
        let desc_layout = pipeline_layout.set_layouts().get(0).unwrap();

        let uniform_buffer_subbuffer = {
            let uniform_data = compute_projs_cs::SimData {
                max_dist: sim_data.max_dist.into(),
                render_size: sim_data.render_size.into(),
                start_pos: sim_data.start_pos.into(),
                dt: time_step,
            };

            let subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
        };
        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            desc_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, voxel_data.voxels()),
                WriteDescriptorSet::buffer(1, self.projectile_buffer.clone()),
                WriteDescriptorSet::buffer(2, uniform_buffer_subbuffer),
            ],
        )
        .unwrap();
        builder
            .bind_pipeline_compute(self.compute_proj_pipeline.clone())
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline_layout.clone(), 0, set)
            .dispatch([((self.upload_projectile_count + 127) / 128) as u32, 1, 1])
            .unwrap();
    }

    pub fn projectiles(&self) -> Subbuffer<[Projectile; 1024]> {
        self.projectile_buffer.clone()
    }

    pub fn download_projectiles(
        &self,
        card_manager: &CardManager,
        vox_compute: &mut VoxelComputePipeline,
    ) -> Vec<Projectile> {
        let mut projectiles = Vec::new();
        let projectiles_buffer = self.projectile_buffer.read().unwrap();
        for i in 0..self.upload_projectile_count {
            let projectile = projectiles_buffer[i];
            if projectile.health == 0.0 {
                vox_compute.queue_update_from_voxel_pos(&[
                    projectile.chunk_update_pos[0],
                    projectile.chunk_update_pos[1],
                    projectile.chunk_update_pos[2],
                ]);
                for card_ref in card_manager
                    .get_referenced_proj(projectile.proj_card_idx as usize)
                    .on_hit.clone()
                {
                    let proj_rot = projectile.dir;
                    let proj_rot = Quaternion::new(proj_rot[3], proj_rot[0], proj_rot[1], proj_rot[2]);
                    projectiles.extend(card_manager.get_projectiles_from_base_card(&card_ref, &Vector3::new(projectile.pos[0], projectile.pos[1], projectile.pos[2]), &proj_rot, projectile.owner))
                }
                continue;
            }
            projectiles.push(projectile);
        }
        projectiles
    }
}

mod compute_projs_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/compute_projectile.glsl",
        include: ["assets/shaders"],
    }
}
