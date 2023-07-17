use crate::{app::App, SimData};
use std::sync::Arc;
use vulkano::{
    buffer::{
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
use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, Zeroable, Debug, Pod)]
#[repr(C)]
pub struct Projectile {
    pub pos: [f32; 4],
    pub dir: [f32; 4],
    pub size: [f32; 4],
    pub vel: [f32; 4],
}

pub struct ProjectilePipeline {
    compute_queue: Arc<Queue>,
    compute_life_pipeline: Arc<ComputePipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    projectiles: Vec<Projectile>,
    projectile_buffer: Subbuffer<[Projectile; 128]>,
}

impl ProjectilePipeline {
    pub fn new(app: &App, compute_queue: Arc<Queue>) -> ProjectilePipeline {
        let memory_allocator = app.context.memory_allocator();

        let compute_life_pipeline = {
            let shader = compute_projectiles_cs::load(compute_queue.device().clone()).unwrap();
            ComputePipeline::new(
                compute_queue.device().clone(),
                shader.entry_point("main").unwrap(),
                &(),
                None,
                |_| {},
            )
            .unwrap()
        };

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

        ProjectilePipeline {
            compute_queue,
            compute_life_pipeline,
            command_buffer_allocator: app.command_buffer_allocator.clone(),
            descriptor_set_allocator: app.descriptor_set_allocator.clone(),
            projectiles: Vec::new(),
            projectile_buffer,
        }
    }

    pub fn push_projectile(&mut self, projectile: Projectile) {
        self.projectiles.push(projectile);
    }

    pub fn projectiles(&self) -> Subbuffer<[Projectile; 128]> {
        self.projectile_buffer.clone()
    }

    pub fn compute(
        &mut self,
        before_future: Box<dyn GpuFuture>,
        sim_data: &mut SimData,
    ) -> Box<dyn GpuFuture> {
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
        self.dispatch(&mut builder, sim_data);

        let command_buffer = builder.build().unwrap();
        let finished = before_future
            .then_execute(self.compute_queue.clone(), command_buffer)
            .unwrap();
        let after_pipeline = finished.then_signal_fence_and_flush().unwrap().boxed();

        let projectile_count = 128.min(self.projectiles.len());
        {
            let projectiles_buffer = self.projectile_buffer.read().unwrap();
            for i in 0..projectile_count {
                self.projectiles[i] = projectiles_buffer[i].clone();
            }
        }

        after_pipeline
    }

    /// Builds the command for a dispatch.
    fn dispatch(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        _sim_data: &mut SimData,
    ) {
        // Resize image if needed.
        let pipeline_layout = self.compute_life_pipeline.layout();
        let desc_layout = pipeline_layout.set_layouts().get(0).unwrap();

        //send projectiles
        let projectile_count = 128.min(self.projectiles.len());
        {
            let mut projectiles_buffer = self.projectile_buffer.write().unwrap();
            for i in 0..projectile_count {
                let projectile = self.projectiles.get(i).unwrap();
                projectiles_buffer[i] = projectile.clone();
            }
        }

        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            desc_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, self.projectile_buffer.clone()),
            ],
        )
        .unwrap();
        builder
            .bind_pipeline_compute(self.compute_life_pipeline.clone())
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline_layout.clone(), 0, set)
            .dispatch([projectile_count as u32, 1, 1])
            .unwrap();
    }
}

mod compute_projectiles_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/projectiles.glsl",
        include: ["src"],
    }
}