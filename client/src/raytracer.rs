// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use cgmath::Matrix4;
use std::sync::Arc;
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferInheritanceInfo, CommandBufferUsage, SecondaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Queue,
    image::view::ImageView,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{
                AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
            },
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::Subpass,
};

use crate::{
    game_manager::Game, multipass_system::LightingVertex, settings_manager::GraphicsSettings,
};

pub struct PointLightingSystem {
    gfx_queue: Arc<Queue>,
    vertex_buffer: Subbuffer<[LightingVertex]>,
    subpass: Subpass,
    pipeline: Arc<GraphicsPipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    uniform_buffer: SubbufferAllocator,
}

impl PointLightingSystem {
    /// Initializes the point lighting system.
    pub fn new(
        gfx_queue: Arc<Queue>,
        subpass: Subpass,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> PointLightingSystem {
        // TODO: vulkano doesn't allow us to draw without a vertex buffer, otherwise we could
        //       hard-code these values in the shader
        let vertices = [
            LightingVertex {
                position: [-1.0, -1.0],
            },
            LightingVertex {
                position: [-1.0, 3.0],
            },
            LightingVertex {
                position: [3.0, -1.0],
            },
        ];
        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .expect("failed to create buffer");

        let uniform_buffer = SubbufferAllocator::new(
            memory_allocator,
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let pipeline = {
            let device = gfx_queue.device();
            let vs = vs::load(device.clone())
                .expect("failed to create shader module")
                .entry_point("main")
                .expect("shader entry point not found");
            let fs = fs::load(device.clone())
                .expect("failed to create shader module")
                .entry_point("main")
                .expect("shader entry point not found");

            let vertex_input_state = LightingVertex::per_vertex()
                .definition(&vs.info().input_interface)
                .unwrap();
            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend {
                                color_blend_op: BlendOp::Add,
                                src_color_blend_factor: BlendFactor::One,
                                dst_color_blend_factor: BlendFactor::One,
                                alpha_blend_op: BlendOp::Max,
                                src_alpha_blend_factor: BlendFactor::One,
                                dst_alpha_blend_factor: BlendFactor::One,
                            }),
                            ..Default::default()
                        },
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.clone().into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        PointLightingSystem {
            gfx_queue,
            vertex_buffer,
            subpass,
            pipeline,
            command_buffer_allocator,
            descriptor_set_allocator,
            uniform_buffer,
        }
    }

    fn create_desc_set(&self, game: &Game) -> Arc<PersistentDescriptorSet> {
        let layout = self.pipeline.layout().set_layouts().get(1).unwrap();
        let sim_uniform_buffer_subbuffer = {
            let uniform_data = fs::SimData {
                render_size: Into::<[u32; 3]>::into(game.game_settings.render_size).into(),
                projectile_count: (game.rollback_data.get_render_projectiles().len() as u32).into(),
                start_pos: game.game_state.start_pos.into(),
            };

            let uniform_subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *uniform_subbuffer.write().unwrap() = uniform_data;

            uniform_subbuffer
        };

        PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::image_view(0, game.voxel_compute.chunks().clone()),
                WriteDescriptorSet::buffer(1, game.voxel_compute.voxels().clone()),
                WriteDescriptorSet::buffer(2, sim_uniform_buffer_subbuffer),
                WriteDescriptorSet::buffer(3, game.rollback_data.visable_player_buffer()),
                WriteDescriptorSet::buffer(4, game.rollback_data.visable_projectile_buffer()),
            ],
            [],
        )
        .unwrap()
    }

    /// Builds a secondary command buffer that applies a point lighting.
    ///
    /// This secondary command buffer will read `depth_input` and rebuild the world position of the
    /// pixel currently being processed (modulo rounding errors). It will then compare this
    /// position with `position`, and process the lighting based on the distance and orientation
    /// (similar to the directional lighting system).
    ///
    /// It then writes the output to the current framebuffer with additive blending (in other words
    /// the value will be added to the existing value in the framebuffer, and not replace the
    /// existing value).
    ///
    /// Note that in a real-world application, you probably want to pass additional parameters
    /// such as some way to indicate the distance at which the lighting decrease. In this example
    /// this value is hardcoded in the shader.
    ///
    /// - `viewport_dimensions` contains the dimensions of the current framebuffer.
    /// - `color_input` is an image containing the albedo of each object of the scene. It is the
    ///   result of the deferred pass.
    /// - `normals_input` is an image containing the normals of each object of the scene. It is the
    ///   result of the deferred pass.
    /// - `depth_input` is an image containing the depth value of each pixel of the scene. It is
    ///   the result of the deferred pass.
    /// - `screen_to_world` is a matrix that turns coordinates from framebuffer space into world
    ///   space. This matrix is used alongside with `depth_input` to determine the world
    ///   coordinates of each pixel being processed.
    /// - `position` is the position of the spot light in world coordinates.
    /// - `color` is the color of the light.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        viewport_dimensions: [u32; 2],
        color_input: Arc<ImageView>,
        normals_input: Arc<ImageView>,
        depth_input: Arc<ImageView>,
        screen_to_world: Matrix4<f32>,
        game: &Game,
        graphics_settings: &GraphicsSettings,
    ) -> Arc<SecondaryAutoCommandBuffer> {
        let push_constants = fs::PushConstants {
            screen_to_world: screen_to_world.into(),
            aspect_ratio: viewport_dimensions[0] as f32 / viewport_dimensions[1] as f32,
            time: game.rollback_data.get_current_time() as f32 * game.rollback_data.get_delta_time(),
            primary_ray_dist: graphics_settings.primary_ray_dist.into(),
            shadow_ray_dist: graphics_settings.shadow_ray_dist.into(),
            reflection_ray_dist: graphics_settings.reflection_ray_dist.into(),
            ao_ray_dist: graphics_settings.ao_ray_dist.into(),
            vertical_resolution: viewport_dimensions[1],
        };

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let image_descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::image_view(0, color_input),
                WriteDescriptorSet::image_view(1, normals_input),
                WriteDescriptorSet::image_view(2, depth_input),
            ],
            [],
        )
        .unwrap();

        let data_descriptor_set = self.create_desc_set(game);

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
            depth_range: 0.0..=1.0,
        };

        let mut builder = AutoCommandBufferBuilder::secondary(
            self.command_buffer_allocator.as_ref(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..Default::default()
            },
        )
        .unwrap();
        builder
            .set_viewport(0, [viewport].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                (image_descriptor_set, data_descriptor_set),
            )
            .unwrap()
            .push_constants(self.pipeline.layout().clone(), 0, push_constants)
            .unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .unwrap()
            .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();
        builder.build().unwrap()
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(location = 0) in vec2 position;
            layout(location = 0) out vec2 v_screen_coords;

            void main() {
                v_screen_coords = position;
                gl_Position = vec4(position, 0.0, 1.0);
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "assets/shaders/frag.glsl",
        include: ["assets/shaders"],
    }
}
