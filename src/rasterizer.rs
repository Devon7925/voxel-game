use std::{sync::Arc, fs::File, io::BufReader};
use bytemuck::{Zeroable, Pod};
use cgmath::{Rad, Matrix4, SquareMatrix};
use obj::{Obj, load_obj};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer, allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo}},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferInheritanceInfo, CommandBufferUsage, SecondaryAutoCommandBuffer,
    },
    device::Queue,
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            vertex_input::Vertex,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::Subpass, descriptor_set::{PersistentDescriptorSet, allocator::StandardDescriptorSetAllocator, WriteDescriptorSet},
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
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            proj_vertices,
        )
        .expect("failed to create buffer");

        let proj_normal_buffer = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            proj_normals,
        )
        .expect("failed to create buffer");

        let proj_instance_data = Buffer::new_sized(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
        )
        .expect("failed to create buffer");

        let (player_vertices, player_normals) = load_obj_to_vecs("assets/player.obj");
        let player_vertex_buffer = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            player_vertices,
        )
        .expect("failed to create buffer");

        let player_normal_buffer = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            player_normals,
        )
        .expect("failed to create buffer");

        let player_instance_data = Buffer::new_sized(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
        )
        .expect("failed to create buffer");

        let pipeline = {
            let vs = vs::load(gfx_queue.device().clone()).expect("failed to create shader module");
            let fs = fs::load(gfx_queue.device().clone()).expect("failed to create shader module");

            GraphicsPipeline::start()
                .vertex_input_state([TrianglePos::per_vertex(), TriangleNormal::per_vertex(), ProjectileInstanceData::per_instance()])
                .vertex_shader(vs.entry_point("main").unwrap(), ())
                .input_assembly_state(InputAssemblyState::new())
                .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
                .fragment_shader(fs.entry_point("main").unwrap(), ())
                .depth_stencil_state(DepthStencilState::simple_depth_test())
                .render_pass(subpass.clone())
                .build(gfx_queue.device().clone())
                .unwrap()
        };

        let uniform_buffer = SubbufferAllocator::new(
            memory_allocator,
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
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
    pub fn draw(&self, viewport_dimensions: [u32; 2], view_matrix: Matrix4<f32>, world_state: &WorldState) -> SecondaryAutoCommandBuffer {

        let uniform_buffer_subbuffer = {
            let model_matrix = Matrix4::identity();

            let aspect_ratio =
                viewport_dimensions[0] as f32 / viewport_dimensions[1] as f32;
            let proj = cgmath::perspective(
                Rad(std::f32::consts::FRAC_PI_2),
                aspect_ratio,
                0.01,
                100.0,
            );

            let uniform_data = vs::Data {
                world: model_matrix.into(),
                view: view_matrix.into(),
                proj: proj.into(),
            };

            let subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
        };

        let mut projectile_writer = self.proj_instance_data.write().unwrap();
        for (i, projectile) in world_state.projectiles.iter().enumerate() {
            projectile_writer[i].instance_position = [-projectile.pos[0], -projectile.pos[1], -projectile.pos[2]];
            projectile_writer[i].instance_rotation = projectile.dir;
            projectile_writer[i].instance_scale = [projectile.size[0], projectile.size[1], projectile.size[2]];
        }

        let mut player_writer = self.player_instance_data.write().unwrap();
        let mut player_buffer_idx = 0;
        for player in world_state.players.iter().skip(1) {
            if player.health <= 0.0 {
                continue;
            }
            player_writer[player_buffer_idx].instance_position = [-player.pos[0], -player.pos[1], -player.pos[2]];
            player_writer[player_buffer_idx].instance_rotation = [player.rot.v[0], player.rot.v[1], player.rot.v[2], player.rot.s];
            player_writer[player_buffer_idx].instance_scale = [player.size, player.size, player.size];
            player_buffer_idx += 1;
        }

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer)],
        )
        .unwrap();

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
                descriptor_set,
            )
            .bind_vertex_buffers(0, (self.proj_vertex_buffer.clone(), self.proj_normal_buffer.clone(), self.proj_instance_data.clone()))
            .draw(self.proj_vertex_buffer.len() as u32, world_state.projectiles.len() as u32, 0, 0)
            .unwrap()
            .bind_vertex_buffers(0, (self.player_vertex_buffer.clone(), self.player_normal_buffer.clone(), self.player_instance_data.clone()))
            .draw(self.player_vertex_buffer.len() as u32, player_buffer_idx as u32, 0, 0)
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
            [-1.0, -1.0, -1.0], [-1.0, -1.0,  1.0], [ 1.0, -1.0, -1.0],
            [ 1.0, -1.0,  1.0], [-1.0, -1.0,  1.0], [ 1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0], [-1.0, -1.0,  1.0], [-1.0,  1.0, -1.0],
            [-1.0,  1.0,  1.0], [-1.0, -1.0,  1.0], [-1.0,  1.0, -1.0],
            [-1.0, -1.0, -1.0], [-1.0,  1.0, -1.0], [ 1.0, -1.0, -1.0],
            [ 1.0,  1.0, -1.0], [-1.0,  1.0, -1.0], [ 1.0, -1.0, -1.0],

            [ 1.0,  1.0,  1.0], [ 1.0,  1.0, -1.0], [-1.0,  1.0,  1.0],
            [-1.0,  1.0, -1.0], [ 1.0,  1.0, -1.0], [-1.0,  1.0,  1.0],
            [ 1.0,  1.0,  1.0], [ 1.0,  1.0, -1.0], [ 1.0, -1.0,  1.0],
            [ 1.0, -1.0, -1.0], [ 1.0,  1.0, -1.0], [ 1.0, -1.0,  1.0],
            [ 1.0,  1.0,  1.0], [ 1.0, -1.0,  1.0], [-1.0,  1.0,  1.0],
            [-1.0, -1.0,  1.0], [ 1.0, -1.0,  1.0], [-1.0,  1.0,  1.0],
        ].iter().map(|&[x, y, z]| TrianglePos { position: [x, y, z] }).collect(),
        vec![
            [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
            [0.0,  1.0, 0.0], [0.0,  1.0, 0.0], [0.0,  1.0, 0.0],
            [0.0,  1.0, 0.0], [0.0,  1.0, 0.0], [0.0,  1.0, 0.0],
            [ 1.0, 0.0, 0.0], [ 1.0, 0.0, 0.0], [ 1.0, 0.0, 0.0],
            [ 1.0, 0.0, 0.0], [ 1.0, 0.0, 0.0], [ 1.0, 0.0, 0.0],
            [0.0, 0.0,  1.0], [0.0, 0.0,  1.0], [0.0, 0.0,  1.0],
            [0.0, 0.0,  1.0], [0.0, 0.0,  1.0], [0.0, 0.0,  1.0],
        ].iter().map(|&[x, y, z]| TriangleNormal { normal: [x, y, z] }).collect()
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
                mat4 world;
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
                mat4 worldview = uniforms.view * uniforms.world;
                vec4 proj_rot_quaternion = quat_inverse(instance_rotation);
                v_normal = -transpose(inverse(mat3(uniforms.world))) * quat_transform(proj_rot_quaternion, normal);
                vec3 instance_vertex_pos = quat_transform(proj_rot_quaternion, instance_scale * position) + instance_position;
                gl_Position = uniforms.proj * worldview * vec4(instance_vertex_pos, 1.0);
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

            void main() {
                f_color = vec4(1.0, 1.0, 1.0, 1.0);
                f_normal = v_normal;
            }
        ",
    }
}