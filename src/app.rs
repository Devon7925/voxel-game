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
    WINDOW_HEIGHT, WINDOW_WIDTH,
};
use std::{collections::HashMap, sync::Arc};
use vulkano::{
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::allocator::StandardDescriptorSetAllocator, device::Queue, format::Format,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{event_loop::EventLoop, window::WindowId};

pub struct RenderPipeline {
    pub compute: DistanceComputePipeline,
    pub place_over_frame: RenderPassPlaceOverFrame,
}

impl RenderPipeline {
    pub fn new(
        app: &App,
        compute_queue: Arc<Queue>,
        gfx_queue: Arc<Queue>,
        swapchain_format: Format,
    ) -> RenderPipeline {
        RenderPipeline {
            compute: DistanceComputePipeline::new(app, compute_queue),
            place_over_frame: RenderPassPlaceOverFrame::new(app, gfx_queue, swapchain_format),
        }
    }
}

pub struct App {
    pub context: VulkanoContext,
    pub windows: VulkanoWindows,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub pipelines: HashMap<WindowId, RenderPipeline>,
}

impl App {
    pub fn open(&mut self, event_loop: &EventLoop<()>) {
        // Create windows & pipelines.
        let id1 = self.windows.create_window(
            event_loop,
            &self.context,
            &WindowDescriptor {
                width: WINDOW_WIDTH,
                height: WINDOW_HEIGHT,
                title: "Voxel Raymarcher".to_string(),
                ..Default::default()
            },
            |_| {},
        );
        self.pipelines.insert(
            id1,
            RenderPipeline::new(
                self,
                // Use same queue.. for synchronization.
                self.context.graphics_queue().clone(),
                self.context.graphics_queue().clone(),
                self.windows
                    .get_primary_renderer()
                    .unwrap()
                    .swapchain_format(),
            ),
        );
    }
}

impl Default for App {
    fn default() -> Self {
        let context = VulkanoContext::new(VulkanoConfig::default());
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device().clone(),
            Default::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            context.device().clone(),
        ));

        App {
            context,
            windows: VulkanoWindows::default(),
            command_buffer_allocator,
            descriptor_set_allocator,
            pipelines: HashMap::new(),
        }
    }
}