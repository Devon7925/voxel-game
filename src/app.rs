// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    sim_manager::DistanceComputePipeline, render_pass::RenderPassPlaceOverFrame,
    WINDOW_HEIGHT, WINDOW_WIDTH, rollback_manager::RollbackData,
};
use std::sync::Arc;
use vulkano::{
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::allocator::StandardDescriptorSetAllocator, device::Queue,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::event_loop::EventLoop;

pub struct VulkanoInterface {
    pub context: VulkanoContext,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

pub struct RenderPipeline {
    pub windows: VulkanoWindows,
    pub vulkano_interface: VulkanoInterface,
    pub compute: DistanceComputePipeline,
    pub rollback_data: RollbackData,
    pub place_over_frame: RenderPassPlaceOverFrame,
}

impl RenderPipeline {
    pub fn new(event_loop: &EventLoop<()>) -> RenderPipeline {
        let context = VulkanoContext::new(VulkanoConfig::default());
        let mut windows = VulkanoWindows::default();
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device().clone(),
            Default::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            context.device().clone(),
        ));

        let compute_queue: Arc<Queue> = context.graphics_queue().clone();
        let gfx_queue: Arc<Queue> = context.graphics_queue().clone();
        // Create windows & pipelines.
        windows.create_window(
            event_loop,
            &context,
            &WindowDescriptor {
                width: WINDOW_WIDTH,
                height: WINDOW_HEIGHT,
                title: "Voxel Raymarcher".to_string(),
                ..Default::default()
            },
            |_| {},
        );
        let swapchain_format = windows
            .get_primary_renderer()
            .unwrap()
            .swapchain_format();
        let vulkano_interface = VulkanoInterface {
            context,
            command_buffer_allocator: command_buffer_allocator,
            descriptor_set_allocator: descriptor_set_allocator,
        };

        RenderPipeline {
            windows,
            compute: DistanceComputePipeline::new(&vulkano_interface, compute_queue),
            rollback_data: RollbackData::new(&vulkano_interface.context),
            place_over_frame: RenderPassPlaceOverFrame::new(&vulkano_interface, gfx_queue, swapchain_format),
            vulkano_interface,
        }
    }
}