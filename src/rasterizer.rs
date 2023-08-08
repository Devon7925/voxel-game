use std::{sync::Arc, fs::File, io::BufReader};
use cgmath::{Rad, Matrix4, Vector3};
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

pub struct RasterizerSystem {
    gfx_queue: Arc<Queue>,
    vertex_buffer: Subbuffer<[TrianglePos]>,
    normals_buffer: Subbuffer<[TriangleNormal]>,
    subpass: Subpass,
    pipeline: Arc<GraphicsPipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    uniform_buffer: SubbufferAllocator,
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
        let (vertices, normals) = load_obj_to_vecs("assets/player.obj");
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

        let normal_buffer = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            normals,
        )
        .expect("failed to create buffer");

        let pipeline = {
            let vs = vs::load(gfx_queue.device().clone()).expect("failed to create shader module");
            let fs = fs::load(gfx_queue.device().clone()).expect("failed to create shader module");

            GraphicsPipeline::start()
                .vertex_input_state([TrianglePos::per_vertex(), TriangleNormal::per_vertex()])
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
            vertex_buffer,
            normals_buffer: normal_buffer,
            subpass,
            pipeline,
            command_buffer_allocator,
            descriptor_set_allocator,
            uniform_buffer
        }
    }

    /// Builds a secondary command buffer that draws the triangle on the current subpass.
    pub fn draw(&self, viewport_dimensions: [u32; 2], view_matrix: Matrix4<f32>) -> SecondaryAutoCommandBuffer {

        let uniform_buffer_subbuffer = {
            let model_matrix = Matrix4::from_translation(Vector3::new(-1664.0, -1664.0, -1664.0));

            // note: this teapot was meant for OpenGL where the origin is at the lower left
            //       instead the origin is at the upper left in Vulkan, so we reverse the Y axis
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
            .bind_vertex_buffers(0, (self.vertex_buffer.clone(), self.normals_buffer.clone()))
            .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
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

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;

            layout(location = 0) out vec3 v_normal;

            layout(set = 0, binding = 0) uniform Data {
                mat4 world;
                mat4 view;
                mat4 proj;
            } uniforms;

            void main() {
                mat4 worldview = uniforms.view * uniforms.world;
                v_normal = -transpose(inverse(mat3(uniforms.world))) * normal;
                gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
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