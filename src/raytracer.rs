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
    image::ImageViewAbstract,
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{AttachmentBlend, BlendFactor, BlendOp, ColorBlendState},
            input_assembly::InputAssemblyState,
            vertex_input::Vertex,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::Subpass,
};

use crate::{
    multipass_system::LightingVertex, rollback_manager::RollbackData,
    settings_manager::GraphicsSettings, SimData,
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
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            vertices,
        )
        .expect("failed to create buffer");

        let uniform_buffer = SubbufferAllocator::new(
            memory_allocator,
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
        );

        let pipeline = {
            let vs = vs::load(gfx_queue.device().clone()).expect("failed to create shader module");
            let fs = fs::load(gfx_queue.device().clone()).expect("failed to create shader module");

            GraphicsPipeline::start()
                .vertex_input_state(LightingVertex::per_vertex())
                .vertex_shader(vs.entry_point("main").unwrap(), ())
                .input_assembly_state(InputAssemblyState::new())
                .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
                .fragment_shader(fs.entry_point("main").unwrap(), ())
                .color_blend_state(ColorBlendState::new(subpass.num_color_attachments()).blend(
                    AttachmentBlend {
                        color_op: BlendOp::Add,
                        color_source: BlendFactor::One,
                        color_destination: BlendFactor::One,
                        alpha_op: BlendOp::Max,
                        alpha_source: BlendFactor::One,
                        alpha_destination: BlendFactor::One,
                    },
                ))
                .render_pass(subpass.clone())
                .build(gfx_queue.device().clone())
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

    fn create_desc_set(
        &self,
        voxels: Subbuffer<[[u32; 2]]>,
        sim_data: &mut SimData,
        rollback_manager: &RollbackData,
    ) -> Arc<PersistentDescriptorSet> {
        let layout = self.pipeline.layout().set_layouts().get(1).unwrap();
        let sim_uniform_buffer_subbuffer = {
            let uniform_data = fs::SimData {
                render_size: sim_data.render_size.into(),
                projectile_count: (rollback_manager.cached_current_state.projectiles.len() as u32)
                    .into(),
                max_dist: sim_data.max_dist.into(),
                start_pos: sim_data.start_pos.into(),
            };

            let subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
        };

        PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, voxels.clone()),
                WriteDescriptorSet::buffer(1, sim_uniform_buffer_subbuffer),
                WriteDescriptorSet::buffer(2, rollback_manager.players()),
                WriteDescriptorSet::buffer(3, rollback_manager.projectiles()),
            ],
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
        color_input: Arc<dyn ImageViewAbstract + 'static>,
        normals_input: Arc<dyn ImageViewAbstract + 'static>,
        depth_input: Arc<dyn ImageViewAbstract + 'static>,
        screen_to_world: Matrix4<f32>,
        voxels: Subbuffer<[[u32; 2]]>,
        rollback_manager: &RollbackData,
        sim_data: &mut SimData,
        graphics_settings: &GraphicsSettings,
    ) -> SecondaryAutoCommandBuffer {
        let push_constants = fs::PushConstants {
            screen_to_world: screen_to_world.into(),
            aspect_ratio: viewport_dimensions[0] as f32 / viewport_dimensions[1] as f32,
            primary_ray_dist: graphics_settings.primary_ray_dist.into(),
            transparency_ray_dist: graphics_settings.transparency_ray_dist.into(),
            shadow_ray_dist: graphics_settings.shadow_ray_dist.into(),
            transparent_shadow_ray_dist: graphics_settings.transparent_shadow_ray_dist.into(),
            ao_ray_dist: graphics_settings.ao_ray_dist.into(),
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
        )
        .unwrap();

        let data_descriptor_set = self.create_desc_set(voxels, sim_data, rollback_manager);

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
            depth_range: 0.0..1.0,
        };

        let mut builder = AutoCommandBufferBuilder::secondary(
            &self.command_buffer_allocator,
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..Default::default()
            },
        )
        .unwrap();
        builder
            .set_viewport(0, [viewport])
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                (image_descriptor_set, data_descriptor_set),
            )
            .push_constants(self.pipeline.layout().clone(), 0, push_constants)
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
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
