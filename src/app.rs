// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    multipass_system::FrameSystem,
    rasterizer::RasterizerSystem,
    settings_manager::Settings,
    WINDOW_HEIGHT, WINDOW_WIDTH, game_manager::Game,
};
use std::sync::Arc;
use vulkano::{
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::{
        StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo, QueueFlags,
    },
    image::{view::ImageView, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    VulkanLibrary,
};

use winit::{
    dpi::PhysicalSize,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

pub struct VulkanoInterface {
    pub swapchain: Arc<Swapchain>,
    pub images: Vec<Arc<ImageView>>,
    pub surface: Arc<Surface>,
    pub device: Arc<Device>,
    pub frame_system: FrameSystem,
    pub rasterizer_system: RasterizerSystem,
    pub creation_interface: CreationInterface,
}

pub struct CreationInterface {
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub queue: Arc<Queue>,
}

pub struct RenderPipeline {
    pub vulkano_interface: VulkanoInterface,
    pub settings: Settings,
    pub game: Option<Game>
}

impl RenderPipeline {
    pub fn new(
        event_loop: &EventLoop<()>,
        settings: Settings,
    ) -> RenderPipeline {
        // Basic initialization. See the triangle example if you want more details about this.

        let library = VulkanLibrary::new().unwrap();
        let required_extensions = Surface::required_extensions(event_loop);
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .unwrap();

        let window = Arc::new(
            WindowBuilder::new()
                .with_inner_size(PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
                .with_title("Voxel game")
                .build(&event_loop)
                .unwrap(),
        );
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();
        let queue = queues.next().unwrap();

        let (swapchain, images) = {
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();
            let image_format = device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0;
            let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

            let (swapchain, images) = Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count,
                    image_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap();
            let images = images
                .into_iter()
                .map(|image| ImageView::new_default(image).unwrap())
                .collect::<Vec<_>>();
            (swapchain, images)
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo {
                secondary_buffer_count: 32,
                ..Default::default()
            },
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        // Here is the basic initialization for the deferred system.
        let frame_system = FrameSystem::new(
            queue.clone(),
            surface.clone(),
            event_loop,
            swapchain.image_format(),
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
        );
        let rasterizer_system = RasterizerSystem::new(
            queue.clone(),
            frame_system.deferred_subpass(),
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            descriptor_set_allocator.clone(),
        );

        let compute_queue: Arc<Queue> = queue.clone();

        let vulkano_interface = VulkanoInterface {
            swapchain,
            images,
            surface,
            device,
            frame_system,
            rasterizer_system,
            creation_interface: CreationInterface {
                memory_allocator,
                command_buffer_allocator,
                descriptor_set_allocator,
                queue: compute_queue,
            },
        };

        RenderPipeline {
            vulkano_interface,
            settings,
            game: None
        }
    }
}
