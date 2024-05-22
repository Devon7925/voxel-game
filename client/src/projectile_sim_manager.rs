// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    app::CreationInterface, card_system::CardManager, game_manager::GameState, rollback_manager::{get_index, PlayerSim}, voxel_sim_manager::VoxelComputePipeline
};
use bytemuck::{Pod, Zeroable};
use cgmath::{Point3, Quaternion};
use core::panic;
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
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::GpuFuture,
};
use voxel_shared::GameSettings;

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

impl ProjectileComputePipeline {
    pub fn new(creation_interface: &CreationInterface) -> ProjectileComputePipeline {
        let memory_allocator = &creation_interface.memory_allocator;

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

        ProjectileComputePipeline {
            compute_queue: creation_interface.queue.clone(),
            compute_proj_pipeline,
            command_buffer_allocator: creation_interface.command_buffer_allocator.clone(),
            descriptor_set_allocator: creation_interface.descriptor_set_allocator.clone(),
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
        game_state: &GameState,
        game_settings: &GameSettings,
        rollback_data: &Box<dyn PlayerSim>,
        voxel_compute: &VoxelComputePipeline,
    ) -> Box<dyn GpuFuture>
    where
        F: GpuFuture + 'static,
    {
        puffin::profile_function!();
        if self.upload_projectile_count == 0 {
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
        self.dispatch(&mut builder, game_state, game_settings, rollback_data, voxel_compute);

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
        rollback_data: &Box<dyn PlayerSim>,
        voxel_compute: &VoxelComputePipeline,
    ) {
        // Resize image if needed.
        let pipeline_layout = self.compute_proj_pipeline.layout();
        let desc_layout = pipeline_layout.set_layouts().get(0).unwrap();

        let uniform_buffer_subbuffer = {
            let uniform_data = compute_projs_cs::SimData {
                render_size:  Into::<[u32;3]>::into(game_settings.render_size).into(),
                start_pos: game_state.start_pos.into(),
                dt: rollback_data.get_delta_time().into(),
                projectile_count: self.upload_projectile_count as u32,
            };

            let subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
        };
        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            desc_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, voxel_compute.chunks()),
                WriteDescriptorSet::buffer(1, voxel_compute.voxels()),
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

    pub fn projectiles(&self) -> Subbuffer<[Projectile; 1024]> {
        self.projectile_buffer.clone()
    }

    pub fn download_projectiles(
        &self,
        card_manager: &CardManager,
        vox_compute: &mut VoxelComputePipeline,
        game_state: &GameState,
        game_settings: &GameSettings
    ) -> Vec<Projectile> {
        let mut projectiles = Vec::new();
        let mut new_voxels = Vec::new();
        let projectiles_buffer = self.projectile_buffer.read().unwrap();
        for i in 0..self.upload_projectile_count {
            let projectile = projectiles_buffer[i];
            if projectile.health == 0.0 && projectile.chunk_update_pos[3] == 1 {
                vox_compute.queue_update_from_voxel_pos(&[
                    projectile.chunk_update_pos[0],
                    projectile.chunk_update_pos[1],
                    projectile.chunk_update_pos[2],
                ], game_settings);
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
            let voxels = vox_compute.voxels();
            let mut writer = voxels.write().unwrap();
            for (pos, material) in new_voxels {
                vox_compute.queue_update_from_voxel_pos(&[pos.x, pos.y, pos.z], game_settings);
                let Some(index) = get_index(pos, &vox_compute.cpu_chunks(), game_state, game_settings) else {
                    panic!("Voxel pos out of bounds");
                };
                writer[index as usize] = material.to_memory();
            }
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
