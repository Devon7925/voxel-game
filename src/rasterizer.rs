use bytemuck::{Pod, Zeroable};
use cgmath::{InnerSpace, Matrix4, Point3, Quaternion, Rad, Rotation3, Vector3};
use obj::{load_obj, Obj};
use std::{fs::File, io::BufReader, sync::Arc};
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
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
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

use crate::rollback_manager::WorldState;

pub struct RasterizerSystem {
    gfx_queue: Arc<Queue>,
    proj_vertex_buffer: Subbuffer<[TrianglePos]>,
    proj_normal_buffer: Subbuffer<[TriangleNormal]>,
    proj_instance_data: Subbuffer<[ProjectileInstanceData; 1024]>,
    player_vertex_buffer: Subbuffer<[TrianglePos]>,
    player_normal_buffer: Subbuffer<[TriangleNormal]>,
    player_instance_data: Subbuffer<[ProjectileInstanceData; 1024]>,
    subpass: Subpass,
    pipeline: Arc<GraphicsPipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    uniform_buffer: SubbufferAllocator,
}

/// The vertex type that describes the unique data per instance.
#[derive(Vertex, Clone, Copy, Zeroable, Debug, Pod)]
#[repr(C)]
struct ProjectileInstanceData {
    #[format(R32G32B32_SFLOAT)]
    instance_position: [f32; 3],
    #[format(R32G32B32A32_SFLOAT)]
    instance_rotation: [f32; 4],
    #[format(R32G32B32_SFLOAT)]
    instance_scale: [f32; 3],
}

impl RasterizerSystem {
    /// Initializes a triangle drawing system.
    pub fn new(
        gfx_queue: Arc<Queue>,
        subpass: Subpass,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> RasterizerSystem {
        let (proj_vertices, proj_normals) = cube_vecs();
        let proj_vertex_buffer = Buffer::from_iter(
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
            proj_vertices,
        )
        .expect("failed to create buffer");

        let proj_normal_buffer = Buffer::from_iter(
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
            proj_normals,
        )
        .expect("failed to create buffer");

        let proj_instance_data = Buffer::new_sized(
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
        )
        .expect("failed to create buffer");

        let (player_vertices, player_normals) = load_obj_to_vecs("assets/player.obj");
        let player_vertex_buffer = Buffer::from_iter(
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
            player_vertices,
        )
        .expect("failed to create buffer");

        let player_normal_buffer = Buffer::from_iter(
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
            player_normals,
        )
        .expect("failed to create buffer");

        let player_instance_data = Buffer::new_sized(
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
        )
        .expect("failed to create buffer");

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
            let vertex_input_state = [
                TrianglePos::per_vertex(),
                TriangleNormal::per_vertex(),
                ProjectileInstanceData::per_instance(),
            ]
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
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    }),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.clone().into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        let uniform_buffer = SubbufferAllocator::new(
            memory_allocator,
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        RasterizerSystem {
            gfx_queue,
            proj_vertex_buffer,
            proj_normal_buffer,
            proj_instance_data,
            subpass,
            pipeline,
            command_buffer_allocator,
            descriptor_set_allocator,
            uniform_buffer,
            player_vertex_buffer,
            player_normal_buffer,
            player_instance_data,
        }
    }

    /// Builds a secondary command buffer that draws the triangle on the current subpass.
    pub fn draw(
        &self,
        viewport_dimensions: [u32; 2],
        view_matrix: Matrix4<f32>,
        world_state: &WorldState,
    ) -> Arc<SecondaryAutoCommandBuffer> {
        let uniform_buffer_subbuffer = {
            let aspect_ratio = viewport_dimensions[0] as f32 / viewport_dimensions[1] as f32;
            let proj =
                cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.1, 100.0);

            let uniform_data = vs::Data {
                view: view_matrix.into(),
                proj: proj.into(),
            };

            let subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
        };

        let mut projectile_writer = self.proj_instance_data.write().unwrap();
        for (i, projectile) in world_state
            .projectiles
            .iter()
            .filter(|proj| {
                let proj_pos = Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]);
                let proj_max_size = proj
                    .size
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                proj.owner > 0
                    || proj.lifetime > 0.3
                    || (proj_pos - world_state.players[0].pos).magnitude() > 0.5 + proj_max_size
            })
            .enumerate()
        {
            projectile_writer[i].instance_position =
                [projectile.pos[0], projectile.pos[1], projectile.pos[2]];
            projectile_writer[i].instance_rotation = projectile.dir;
            projectile_writer[i].instance_scale =
                [projectile.size[0], projectile.size[1], projectile.size[2]];
        }

        let mut player_writer = self.player_instance_data.write().unwrap();
        let mut player_buffer_idx = 0;
        for player in world_state.players.iter().skip(1) {
            if player.get_health_stats().0 <= 0.0 {
                continue;
            }
            player_writer[player_buffer_idx].instance_position =
                [player.pos[0], player.pos[1], player.pos[2]];
            let render_rotation =
                Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), Rad(player.facing[0]));
            player_writer[player_buffer_idx].instance_rotation = [
                render_rotation.v[0],
                render_rotation.v[1],
                render_rotation.v[2],
                render_rotation.s,
            ];
            player_writer[player_buffer_idx].instance_scale =
                [player.size, player.size, player.size];
            player_buffer_idx += 1;
        }

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();

        let proj_color_uniform_buffer_subbuffer = {
            let uniform_data = fs::Material {
                material_color: [0.0, 0.0, 0.0, 0.0],
            };

            let subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
        };
        let proj_descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer.clone()),
                WriteDescriptorSet::buffer(1, proj_color_uniform_buffer_subbuffer),
            ],
            [],
        )
        .unwrap();
        let player_color_uniform_buffer_subbuffer = {
            let uniform_data = fs::Material {
                material_color: [1.0, 0.0, 0.0, 0.0],
            };

            let subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
        };
        let player_descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer),
                WriteDescriptorSet::buffer(1, player_color_uniform_buffer_subbuffer),
            ],
            [],
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::secondary(
            self.command_buffer_allocator.as_ref(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..Default::default()
            },
        )
        .unwrap();
        builder
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                proj_descriptor_set,
            )
            .unwrap()
            .bind_vertex_buffers(
                0,
                (
                    self.proj_vertex_buffer.clone(),
                    self.proj_normal_buffer.clone(),
                    self.proj_instance_data.clone(),
                ),
            )
            .unwrap()
            .draw(
                self.proj_vertex_buffer.len() as u32,
                world_state.projectiles.len() as u32,
                0,
                0,
            )
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                player_descriptor_set,
            )
            .unwrap()
            .bind_vertex_buffers(
                0,
                (
                    self.player_vertex_buffer.clone(),
                    self.player_normal_buffer.clone(),
                    self.player_instance_data.clone(),
                ),
            )
            .unwrap()
            .draw(
                self.player_vertex_buffer.len() as u32,
                player_buffer_idx as u32,
                0,
                0,
            )
            .unwrap();
        builder.build().unwrap()
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct TrianglePos {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct TriangleNormal {
    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],
}

fn load_obj_to_vecs(path: &str) -> (Vec<TrianglePos>, Vec<TriangleNormal>) {
    let input = BufReader::new(File::open(path).unwrap());
    let model: Obj = load_obj(input).unwrap();
    let mut index_iter = model.indices.iter();
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    while let Some(first_idx) = index_iter.next() {
        let second_idx = index_iter.next().unwrap();
        let third_idx = index_iter.next().unwrap();
        let first_vertex = model.vertices.get(*first_idx as usize).unwrap();
        let second_vertex = model.vertices.get(*second_idx as usize).unwrap();
        let third_vertex = model.vertices.get(*third_idx as usize).unwrap();
        positions.push(TrianglePos {
            position: first_vertex.position,
        });
        positions.push(TrianglePos {
            position: second_vertex.position,
        });
        positions.push(TrianglePos {
            position: third_vertex.position,
        });
        normals.push(TriangleNormal {
            normal: first_vertex.normal,
        });
        normals.push(TriangleNormal {
            normal: second_vertex.normal,
        });
        normals.push(TriangleNormal {
            normal: third_vertex.normal,
        });
    }
    (positions, normals)
}

fn cube_vecs() -> (Vec<TrianglePos>, Vec<TriangleNormal>) {
    (
        vec![
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
        .iter()
        .map(|&[x, y, z]| TrianglePos {
            position: [x, y, z],
        })
        .collect(),
        vec![
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
        .iter()
        .map(|&[x, y, z]| TriangleNormal { normal: [x, y, z] })
        .collect(),
    )
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;

            layout(location = 2) in vec3 instance_position;
            layout(location = 3) in vec4 instance_rotation;
            layout(location = 4) in vec3 instance_scale;

            layout(location = 0) out vec3 v_normal;

            layout(set = 0, binding = 0) uniform Data {
                mat4 view;
                mat4 proj;
            } uniforms;

            vec3 quat_transform(vec4 q, vec3 v) {
                return v + 2.*cross( q.xyz, cross( q.xyz, v ) + q.w*v ); 
            }

            vec4 quat_inverse(vec4 q) {
                return vec4(-q.xyz, q.w) / dot(q, q);
            }

            void main() {
                v_normal = -quat_transform(instance_rotation, normal);
                vec3 instance_vertex_pos = quat_transform(instance_rotation, instance_scale * position) - instance_position;
                gl_Position = uniforms.proj * (uniforms.view * vec4(instance_vertex_pos, 1.0));
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450

            layout(location = 0) in vec3 v_normal;

            layout(location = 0) out vec4 f_color;
            layout(location = 1) out vec3 f_normal;

            layout(set = 0, binding = 1) uniform Material {
                vec4 material_color;
            } material;

            void main() {
                f_color = material.material_color;
                f_normal = v_normal;
            }
        ",
    }
}
