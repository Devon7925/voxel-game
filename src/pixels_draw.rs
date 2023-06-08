// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{app::App, CamData, SimData};
use std::sync::Arc;
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferInheritanceInfo, CommandBufferUsage, SecondaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Queue,
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            vertex_input::Vertex,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::Subpass,
};

/// Vertex for textured quads.
#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct TexturedVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub tex_coords: [f32; 2],
}

pub fn textured_quad(width: f32, height: f32) -> (Vec<TexturedVertex>, Vec<u32>) {
    (
        vec![
            TexturedVertex {
                position: [-(width / 2.0), -(height / 2.0)],
                tex_coords: [-1.0, 1.0],
            },
            TexturedVertex {
                position: [-(width / 2.0), height / 2.0],
                tex_coords: [-1.0, -1.0],
            },
            TexturedVertex {
                position: [width / 2.0, height / 2.0],
                tex_coords: [1.0, -1.0],
            },
            TexturedVertex {
                position: [width / 2.0, -(height / 2.0)],
                tex_coords: [1.0, 1.0],
            },
        ],
        vec![0, 2, 1, 0, 3, 2],
    )
}

/// A subpass pipeline that fills a quad over the frame.
pub struct PixelsDrawPipeline {
    gfx_queue: Arc<Queue>,
    subpass: Subpass,
    pipeline: Arc<GraphicsPipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    vertices: Subbuffer<[TexturedVertex]>,
    indices: Subbuffer<[u32]>,
    uniform_buffer: SubbufferAllocator,
}

impl PixelsDrawPipeline {
    pub fn new(app: &App, gfx_queue: Arc<Queue>, subpass: Subpass) -> PixelsDrawPipeline {
        let (vertices, indices) = textured_quad(2.0, 2.0);
        let memory_allocator = app.context.memory_allocator();
        let vertex_buffer = Buffer::from_iter(
            memory_allocator,
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
        .unwrap();
        let index_buffer = Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            indices,
        )
        .unwrap();

        let uniform_buffer = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
        );

        let pipeline = {
            let vs = vs::load(gfx_queue.device().clone()).expect("failed to create shader module");
            let fs = fs::load(gfx_queue.device().clone()).expect("failed to create shader module");
            GraphicsPipeline::start()
                .vertex_input_state(TexturedVertex::per_vertex())
                .vertex_shader(vs.entry_point("main").unwrap(), ())
                .input_assembly_state(InputAssemblyState::new())
                .fragment_shader(fs.entry_point("main").unwrap(), ())
                .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
                .render_pass(subpass.clone())
                .build(gfx_queue.device().clone())
                .unwrap()
        };

        PixelsDrawPipeline {
            gfx_queue,
            subpass,
            pipeline,
            command_buffer_allocator: app.command_buffer_allocator.clone(),
            descriptor_set_allocator: app.descriptor_set_allocator.clone(),
            vertices: vertex_buffer,
            indices: index_buffer,
            uniform_buffer,
        }
    }

    fn create_desc_set(
        &self,
        voxels: Subbuffer<[[u32; 2]]>,
        cam_data: &CamData,
        sim_data: &mut SimData,
    ) -> Arc<PersistentDescriptorSet> {
        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();

        let cam_uniform_buffer_subbuffer = {
            let uniform_data = fs::CamData {
                pos: [cam_data.pos.x, cam_data.pos.y, cam_data.pos.z].into(),
                dir: [cam_data.dir.x, cam_data.dir.y, cam_data.dir.z].into(),
                up: [cam_data.up.x, cam_data.up.y, cam_data.up.z].into(),
                right: [cam_data.right.x, cam_data.right.y, cam_data.right.z].into(),
            };

            let subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
        };

        let sim_uniform_buffer_subbuffer = {
            let uniform_data = fs::SimData {
                render_size: sim_data.render_size.into(),
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
                WriteDescriptorSet::buffer(1, cam_uniform_buffer_subbuffer),
                WriteDescriptorSet::buffer(2, sim_uniform_buffer_subbuffer),
            ],
        )
        .unwrap()
    }

    /// Draws input `image` over a quad of size -1.0 to 1.0.
    pub fn draw(
        &self,
        viewport_dimensions: [u32; 2],
        voxels: Subbuffer<[[u32; 2]]>,
        cam_data: &CamData,
        sim_data: &mut SimData,
    ) -> SecondaryAutoCommandBuffer {
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
        let desc_set = self.create_desc_set(voxels, cam_data, sim_data);
        builder
            .set_viewport(
                0,
                [Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
                    depth_range: 0.0..1.0,
                }],
            )
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                desc_set,
            )
            .bind_vertex_buffers(0, self.vertices.clone())
            .bind_index_buffer(self.indices.clone())
            .draw_indexed(self.indices.len() as u32, 1, 0, 0, 0)
            .unwrap();
        builder.build().unwrap()
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450
            layout(location=0) in vec2 position;
            layout(location=1) in vec2 tex_coords;

            layout(location = 0) out vec2 f_tex_coords;

            void main() {
                gl_Position =  vec4(position, 0.0, 1.0);
                f_tex_coords = tex_coords;
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/frag.glsl",
        include: ["src"],
    }
}