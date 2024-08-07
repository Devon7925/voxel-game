// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.
use cgmath::{Matrix4, SquareMatrix};
use egui_winit_vulkano::{
    egui::{
        self, epaint, vec2, Align2, Color32, Margin, Order, Rect, RichText, Stroke, Vec2,
    },
    Gui, GuiConfig,
};
use rfd::FileDialog;
use std::{fs, sync::Arc};
use vulkano::{
    buffer::BufferContents,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SecondaryCommandBufferAbstract,
        SubpassBeginInfo, SubpassContents,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::Queue,
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
    memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator},
    pipeline::graphics::vertex_input::Vertex,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::Surface,
    sync::GpuFuture,
};
use winit::event_loop::EventLoop;

use crate::{
    app::CreationInterface, game_manager::Game, gui::{card_editor, cooldown, healthbar, horizontal_centerer, vertical_centerer, GuiElement}, raytracer::PointLightingSystem, settings_manager::Settings, utils::recurse_files, GuiState
};
use voxel_shared::RoomId;

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct LightingVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
}

/// System that contains the necessary facilities for rendering a single frame.
pub struct FrameSystem {
    // Queue to use to render everything.
    gfx_queue: Arc<Queue>,

    // Render pass used for the drawing. See the `new` method for the actual render pass content.
    // We need to keep it in `FrameSystem` because we may want to recreate the intermediate buffers
    // in of a change in the dimensions.
    render_pass: Arc<RenderPass>,

    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,

    // Intermediate render target that will contain the albedo of each pixel of the scene.
    diffuse_buffer: Arc<ImageView>,
    // Intermediate render target that will contain the normal vector in world coordinates of each
    // pixel of the scene.
    // The normal vector is the vector perpendicular to the surface of the object at this point.
    normals_buffer: Arc<ImageView>,
    // Intermediate render target that will contain the depth of each pixel of the scene.
    // This is a traditional depth buffer. `0.0` means "near", and `1.0` means "far".
    depth_buffer: Arc<ImageView>,

    // Will allow us to add an ambient lighting to a scene during the second subpass.
    ambient_lighting_system: PointLightingSystem,
    pub gui: Gui,
}

impl FrameSystem {
    /// Initializes the frame system.
    ///
    /// Should be called at initialization, as it can take some time to build.
    ///
    /// - `gfx_queue` is the queue that will be used to perform the main rendering.
    /// - `final_output_format` is the format of the image that will later be passed to the
    ///   `frame()` method. We need to know that in advance. If that format ever changes, we have
    ///   to create a new `FrameSystem`.
    pub fn new(
        gfx_queue: Arc<Queue>,
        surface: Arc<Surface>,
        event_loop: &EventLoop<()>,
        final_output_format: Format,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    ) -> FrameSystem {
        // Creating the render pass.
        //
        // The render pass has two subpasses. In the first subpass, we draw all the objects of the
        // scene. Note that it is not the `FrameSystem` that is responsible for the drawing,
        // instead it only provides an API that allows the user to do so.
        //
        // The drawing of the objects will write to the `diffuse`, `normals` and `depth`
        // attachments.
        //
        // Then in the second subpass, we read these three attachments as input attachments and
        // draw to `final_color`. Each draw operation performed in this second subpass has its
        // value added to `final_color` and not replaced, thanks to blending.
        //
        // > **Warning**: If the red, green or blue component of the final image goes over `1.0`
        // > then it will be clamped. For example a pixel of `[2.0, 1.0, 1.0]` (which is red) will
        // > be clamped to `[1.0, 1.0, 1.0]` (which is white) instead of being converted to
        // > `[1.0, 0.5, 0.5]` as desired. In a real-life application you want to use an additional
        // > intermediate image with a floating-point format, then perform additional passes to
        // > convert all the colors in the correct range. These techniques are known as HDR and
        // > tone mapping.
        //
        // Input attachments are a special kind of way to read images. You can only read from them
        // from a fragment shader, and you can only read the pixel corresponding to the pixel
        // currently being processed by the fragment shader. If you want to read from attachments
        // but can't deal with these restrictions, then you should create multiple render passes
        // instead.
        let render_pass = vulkano::ordered_passes_renderpass!(
            gfx_queue.device().clone(),
            attachments: {
                // The image that will contain the final rendering (in this example the swapchain
                // image, but it could be another image).
                final_color: {
                    format: final_output_format,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                // Will be bound to `self.diffuse_buffer`.
                diffuse: {
                    format: Format::A2B10G10R10_UNORM_PACK32,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },
                // Will be bound to `self.normals_buffer`.
                normals: {
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },
                // Will be bound to `self.depth_buffer`.
                depth: {
                    format: Format::D32_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },
            },
            passes: [
                // Write to the diffuse, normals and depth attachments.
                {
                    color: [diffuse, normals],
                    depth_stencil: {depth},
                    input: [],
                },
                // Apply lighting by reading these three attachments and writing to `final_color`.
                {
                    color: [final_color],
                    depth_stencil: {},
                    input: [diffuse, normals, depth],
                },
            ],
        )
        .unwrap();

        // For now we create three temporary images with a dimension of 1 by 1 pixel. These images
        // will be replaced the first time we call `frame()`.
        let diffuse_buffer = ImageView::new_default(
            Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::A2B10G10R10_UNORM_PACK32,
                    extent: [1, 1, 1],
                    usage: ImageUsage::TRANSIENT_ATTACHMENT
                        | ImageUsage::INPUT_ATTACHMENT
                        | ImageUsage::COLOR_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();
        let normals_buffer = ImageView::new_default(
            Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R16G16B16A16_SFLOAT,
                    extent: [1, 1, 1],
                    usage: ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();
        let depth_buffer = ImageView::new_default(
            Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::D32_SFLOAT,
                    extent: [1, 1, 1],
                    usage: ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            gfx_queue.device().clone(),
            Default::default(),
        ));

        // Initialize the three lighting systems. Note that we need to pass to them the subpass
        // where they will be executed.
        let lighting_subpass = Subpass::from(render_pass.clone(), 1).unwrap();
        let ambient_lighting_system = PointLightingSystem::new(
            gfx_queue.clone(),
            lighting_subpass.clone(),
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            descriptor_set_allocator,
        );
        let gui = Gui::new_with_subpass(
            &event_loop,
            surface.clone(),
            gfx_queue.clone(),
            lighting_subpass,
            Format::B8G8R8A8_UNORM,
            GuiConfig {
                is_overlay: true,
                ..Default::default()
            },
        );

        FrameSystem {
            gfx_queue,
            render_pass,
            memory_allocator,
            command_buffer_allocator,
            diffuse_buffer,
            normals_buffer,
            depth_buffer,
            ambient_lighting_system,
            gui,
        }
    }

    /// Returns the subpass of the render pass where the rendering should write info to gbuffers.
    ///
    /// Has two outputs: the diffuse color (3 components) and the normals in world coordinates
    /// (3 components). Also has a depth attachment.
    ///
    /// This method is necessary in order to initialize the pipelines that will draw the objects
    /// of the scene.
    #[inline]
    pub fn deferred_subpass(&self) -> Subpass {
        Subpass::from(self.render_pass.clone(), 0).unwrap()
    }

    /// Starts drawing a new frame.
    ///
    /// - `before_future` is the future after which the main rendering should be executed.
    /// - `final_image` is the image we are going to draw to.
    /// - `world_to_framebuffer` is the matrix that will be used to convert from 3D coordinates in
    ///   the world into 2D coordinates on the framebuffer.
    pub fn frame<F>(
        &mut self,
        before_future: F,
        final_image: Arc<ImageView>,
        world_to_framebuffer: Matrix4<f32>,
    ) -> Frame
    where
        F: GpuFuture + 'static,
    {
        puffin::profile_function!();
        // First of all we recreate `self.diffuse_buffer`, `self.normals_buffer` and
        // `self.depth_buffer` if their dimensions doesn't match the dimensions of the final image.
        let extent = final_image.image().extent();
        if self.diffuse_buffer.image().extent() != extent {
            // Note that we create "transient" images here. This means that the content of the
            // image is only defined when within a render pass. In other words you can draw to
            // them in a subpass then read them in another subpass, but as soon as you leave the
            // render pass their content becomes undefined.
            self.diffuse_buffer = ImageView::new_default(
                Image::new(
                    self.memory_allocator.clone(),
                    ImageCreateInfo {
                        extent,
                        format: Format::A2B10G10R10_UNORM_PACK32,
                        usage: ImageUsage::TRANSIENT_ATTACHMENT
                            | ImageUsage::INPUT_ATTACHMENT
                            | ImageUsage::COLOR_ATTACHMENT,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap(),
            )
            .unwrap();
            self.normals_buffer = ImageView::new_default(
                Image::new(
                    self.memory_allocator.clone(),
                    ImageCreateInfo {
                        extent,
                        format: Format::R16G16B16A16_SFLOAT,
                        usage: ImageUsage::TRANSIENT_ATTACHMENT
                            | ImageUsage::INPUT_ATTACHMENT
                            | ImageUsage::COLOR_ATTACHMENT,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap(),
            )
            .unwrap();
            self.depth_buffer = ImageView::new_default(
                Image::new(
                    self.memory_allocator.clone(),
                    ImageCreateInfo {
                        extent,
                        format: Format::D32_SFLOAT,
                        usage: ImageUsage::TRANSIENT_ATTACHMENT
                            | ImageUsage::INPUT_ATTACHMENT
                            | ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap(),
            )
            .unwrap();
        }

        // Build the framebuffer. The image must be attached in the same order as they were defined
        // with the `ordered_passes_renderpass!` macro.
        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![
                    final_image.clone(),
                    self.diffuse_buffer.clone(),
                    self.normals_buffer.clone(),
                    self.depth_buffer.clone(),
                ],
                ..Default::default()
            },
        )
        .unwrap();

        // Start the command buffer builder that will be filled throughout the frame handling.
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.as_ref(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        command_buffer_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some([0.0, 0.0, 0.0, 0.0].into()),
                        Some([0.0, 0.0, 0.0, 0.0].into()),
                        Some([0.0, 0.0, 0.0, 0.0].into()),
                        Some(1.0f32.into()),
                    ],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassBeginInfo {
                    contents: SubpassContents::SecondaryCommandBuffers,
                    ..Default::default()
                },
            )
            .unwrap();

        Frame {
            system: self,
            before_main_cb_future: Some(Box::new(before_future)),
            framebuffer,
            num_pass: 0,
            command_buffer_builder: Some(command_buffer_builder),
            world_to_framebuffer,
        }
    }
}

/// Represents the active process of rendering a frame.
///
/// This struct mutably borrows the `FrameSystem`.
pub struct Frame<'a> {
    // The `FrameSystem`.
    system: &'a mut FrameSystem,

    // The active pass we are in. This keeps track of the step we are in.
    // - If `num_pass` is 0, then we haven't start anything yet.
    // - If `num_pass` is 1, then we have finished drawing all the objects of the scene.
    // - If `num_pass` is 2, then we have finished applying lighting.
    // - Otherwise the frame is finished.
    // In a more complex application you can have dozens of passes, in which case you probably
    // don't want to document them all here.
    num_pass: u8,

    // Future to wait upon before the main rendering.
    before_main_cb_future: Option<Box<dyn GpuFuture>>,
    // Framebuffer that was used when starting the render pass.
    framebuffer: Arc<Framebuffer>,
    // The command buffer builder that will be built during the lifetime of this object.
    command_buffer_builder: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
    // Matrix that was passed to `frame()`.
    world_to_framebuffer: Matrix4<f32>,
}

impl<'a> Frame<'a> {
    /// Returns an enumeration containing the next pass of the rendering.
    pub fn next_pass<'f>(&'f mut self) -> Option<Pass<'f, 'a>> {
        // This function reads `num_pass` increments its value, and returns a struct corresponding
        // to that pass that the user will be able to manipulate in order to customize the pass.
        match {
            let current_pass = self.num_pass;
            self.num_pass += 1;
            current_pass
        } {
            0 => {
                // If we are in the pass 0 then we haven't start anything yet.
                // We already called `begin_render_pass` (in the `frame()` method), and that's the
                // state we are in.
                // We return an object that will allow the user to draw objects on the scene.
                Some(Pass::Deferred(DrawPass { frame: self }))
            }

            1 => {
                // If we are in pass 1 then we have finished drawing the objects on the scene.
                // Going to the next subpass.
                self.command_buffer_builder
                    .as_mut()
                    .unwrap()
                    .next_subpass(
                        Default::default(),
                        SubpassBeginInfo {
                            contents: SubpassContents::SecondaryCommandBuffers,
                            ..Default::default()
                        },
                    )
                    .unwrap();

                // And returning an object that will allow the user to apply lighting to the scene.
                Some(Pass::Lighting(LightingPass { frame: self }))
            }

            2 => {
                // If we are in pass 2 then we have finished applying lighting.
                // We take the builder, call `end_render_pass()`, and then `build()` it to obtain
                // an actual command buffer.
                self.command_buffer_builder
                    .as_mut()
                    .unwrap()
                    .end_render_pass(Default::default())
                    .unwrap();
                let command_buffer = self.command_buffer_builder.take().unwrap().build().unwrap();

                // Extract `before_main_cb_future` and append the command buffer execution to it.
                let after_main_cb = self
                    .before_main_cb_future
                    .take()
                    .unwrap()
                    .then_execute(self.system.gfx_queue.clone(), command_buffer)
                    .unwrap();
                // We obtain `after_main_cb`, which we give to the user.
                Some(Pass::Finished(Box::new(after_main_cb)))
            }

            // If the pass is over 2 then the frame is in the finished state and can't do anything
            // more.
            _ => None,
        }
    }
}

/// Struct provided to the user that allows them to customize or handle the pass.
pub enum Pass<'f, 's: 'f> {
    /// We are in the pass where we draw objects on the scene. The `DrawPass` allows the user to
    /// draw the objects.
    Deferred(DrawPass<'f, 's>),

    /// We are in the pass where we add lighting to the scene. The `LightingPass` allows the user
    /// to add light sources.
    Lighting(LightingPass<'f, 's>),

    /// The frame has been fully prepared, and here is the future that will perform the drawing
    /// on the image.
    Finished(Box<dyn GpuFuture>),
}

/// Allows the user to draw objects on the scene.
pub struct DrawPass<'f, 's: 'f> {
    frame: &'f mut Frame<'s>,
}

impl<'f, 's: 'f> DrawPass<'f, 's> {
    /// Appends a command that executes a secondary command buffer that performs drawing.
    pub fn execute(&mut self, command_buffer: Arc<dyn SecondaryCommandBufferAbstract>) {
        self.frame
            .command_buffer_builder
            .as_mut()
            .unwrap()
            .execute_commands(command_buffer)
            .unwrap();
    }

    /// Returns the dimensions in pixels of the viewport.
    pub fn viewport_dimensions(&self) -> [u32; 2] {
        self.frame.framebuffer.extent()
    }

    /// Returns the 4x4 matrix that turns world coordinates into 2D coordinates on the framebuffer.
    #[allow(dead_code)]
    pub fn world_to_framebuffer_matrix(&self) -> Matrix4<f32> {
        self.frame.world_to_framebuffer
    }
}

/// Allows the user to apply lighting on the scene.
pub struct LightingPass<'f, 's: 'f> {
    frame: &'f mut Frame<'s>,
}

impl<'f, 's: 'f> LightingPass<'f, 's> {
    /// Applies a spot lighting to the scene.
    ///
    /// All the objects will be colored with an intensity varying between `[0, 0, 0]` and `color`,
    /// depending on their distance with `position`. Objects that aren't facing `position` won't
    /// receive any light.
    pub fn raytrace(&mut self, game: &Game, settings: &Settings) {
        puffin::profile_function!();
        let command_buffer = {
            self.frame.system.ambient_lighting_system.draw(
                self.frame.framebuffer.extent(),
                self.frame.system.diffuse_buffer.clone(),
                self.frame.system.normals_buffer.clone(),
                self.frame.system.depth_buffer.clone(),
                self.frame.world_to_framebuffer.invert().unwrap(),
                game,
                &settings.graphics_settings,
            )
        };

        self.frame
            .command_buffer_builder
            .as_mut()
            .unwrap()
            .execute_commands(command_buffer)
            .unwrap();
    }

    pub fn gui(
        &mut self,
        game: &mut Option<Game>,
        gui_state: &mut GuiState,
        settings: &Settings,
        creation_interface: &CreationInterface,
    ) {
        puffin::profile_function!();
        self.frame.system.gui.immediate_ui(|gui| {
            let ctx = gui.context();
            // Fill egui UI layout here
            if let Some(game) = game {
                let spectate_player = &game.rollback_data.get_spectate_player();

                if let Some(spectate_player) = spectate_player {
                    let corner_offset = 10.0;

                    if gui_state.menu_stack.is_empty() {
                        egui::Area::new("crosshair")
                            .anchor(Align2::LEFT_TOP, (0.0, 0.0))
                            .show(&ctx, |ui| {
                                let center = ui.available_rect_before_wrap().center();

                                if spectate_player.hitmarker.0 + spectate_player.hitmarker.1 > 0.0 {
                                    let hitmarker_size = 0.5 * spectate_player.hitmarker.0;
                                    let head_hitmarker_size = 0.5
                                        * (spectate_player.hitmarker.0
                                            + spectate_player.hitmarker.1);
                                    let hitmarker_thickness = 1.5;
                                    let head_hitmarker_color = Color32::RED;
                                    let hitmarker_color = Color32::from_additive_luminance(255);
                                    ui.painter().add(epaint::Shape::line_segment(
                                        [
                                            center
                                                + vec2(-head_hitmarker_size, -head_hitmarker_size),
                                            center + vec2(head_hitmarker_size, head_hitmarker_size),
                                        ],
                                        Stroke::new(hitmarker_thickness, head_hitmarker_color),
                                    ));
                                    ui.painter().add(epaint::Shape::line_segment(
                                        [
                                            center
                                                + vec2(-head_hitmarker_size, head_hitmarker_size),
                                            center
                                                + vec2(head_hitmarker_size, -head_hitmarker_size),
                                        ],
                                        Stroke::new(hitmarker_thickness, head_hitmarker_color),
                                    ));
                                    ui.painter().add(epaint::Shape::line_segment(
                                        [
                                            center + vec2(-hitmarker_size, -hitmarker_size),
                                            center + vec2(hitmarker_size, hitmarker_size),
                                        ],
                                        Stroke::new(hitmarker_thickness, hitmarker_color),
                                    ));
                                    ui.painter().add(epaint::Shape::line_segment(
                                        [
                                            center + vec2(-hitmarker_size, hitmarker_size),
                                            center + vec2(hitmarker_size, -hitmarker_size),
                                        ],
                                        Stroke::new(hitmarker_thickness, hitmarker_color),
                                    ));
                                }

                                let thickness = 1.0;
                                let color = Color32::from_additive_luminance(255);
                                let crosshair_size = 10.0;

                                ui.painter().add(epaint::Shape::line_segment(
                                    [
                                        center + vec2(-crosshair_size, 0.0),
                                        center + vec2(crosshair_size, 0.0),
                                    ],
                                    Stroke::new(thickness, color),
                                ));
                                ui.painter().add(epaint::Shape::line_segment(
                                    [
                                        center + vec2(0.0, -crosshair_size),
                                        center + vec2(0.0, crosshair_size),
                                    ],
                                    Stroke::new(thickness, color),
                                ));

                                //draw hurtmarkers
                                for (hurt_direction, hurt_size, remaining_marker_duration) in
                                    spectate_player.hurtmarkers.iter()
                                {
                                    let hurtmarker_color = Color32::RED
                                        .gamma_multiply(remaining_marker_duration / 1.5);
                                    let hurtmarker_size = 1.2 * hurt_size.sqrt();
                                    let transformed_hurt_angle = spectate_player.facing[0]
                                        - (-hurt_direction.z).atan2(hurt_direction.x);
                                    let hurtmarker_center = center
                                        + vec2(
                                            transformed_hurt_angle.cos(),
                                            transformed_hurt_angle.sin(),
                                        ) * 50.0;
                                    ui.painter().circle_filled(
                                        hurtmarker_center,
                                        hurtmarker_size,
                                        hurtmarker_color,
                                    );
                                }
                            });
                    }

                    healthbar(corner_offset, &ctx, spectate_player);

                    let respawn_time = spectate_player.respawn_timer;
                    if respawn_time > 0.0 {
                        egui::Area::new("respawn")
                            .anchor(Align2::LEFT_TOP, Vec2::new(corner_offset, corner_offset))
                            .show(&ctx, |ui| {
                                ui.label(RichText::new("You have died").color(Color32::WHITE));
                                ui.label(
                                    RichText::new(format!("Respawn in {}", respawn_time))
                                        .color(Color32::WHITE),
                                );
                            });
                    }

                    
                    egui::Area::new("game overlay")
                    .anchor(Align2::CENTER_TOP, Vec2::new(0.0, 0.0))
                    .show(&ctx, |ui| {
                        game.game_mode.overlay(ui, &game.rollback_data);
                    });

                    egui::Area::new("cooldowns")
                        .anchor(
                            Align2::RIGHT_BOTTOM,
                            Vec2::new(-corner_offset, -corner_offset),
                        )
                        .show(&ctx, |ui| {
                            for ability in spectate_player.abilities.iter() {
                                for ability_idx in 0..ability.ability.abilities.len() {
                                    ui.add(cooldown(ability, ability_idx));
                                }
                            }
                        });
                }
            }
            match gui_state.menu_stack.last() {
                Some(&GuiElement::MainMenu) => {
                    egui::Area::new("main menu")
                        .anchor(Align2::LEFT_TOP, Vec2::new(0.0, 0.0))
                        .show(&ctx, |ui| {
                            let menu_size = Rect::from_center_size(
                                ui.available_rect_before_wrap().center(),
                                ui.available_rect_before_wrap().size(),
                            );

                            ui.allocate_ui_at_rect(menu_size, |ui| {
                                ui.painter().rect_filled(
                                    ui.available_rect_before_wrap(),
                                    0.0,
                                    Color32::BLACK,
                                );
                                vertical_centerer(ui, |ui| {
                                    ui.vertical_centered(|ui| {
                                        if ui.button("Singleplayer").clicked() {
                                            gui_state.menu_stack.push(GuiElement::SingleplayerMenu);
                                        }
                                        if ui.button("Multiplayer").clicked() {
                                            gui_state.menu_stack.push(GuiElement::MultiplayerMenu);
                                        }
                                        if ui.button("Deck Picker").clicked() {
                                            gui_state.menu_stack.push(GuiElement::DeckPicker);
                                        }
                                        if ui.button("Play Replay").clicked() {
                                            let mut replay_folder_path =
                                                std::env::current_dir().unwrap();
                                            replay_folder_path.push(
                                                settings.replay_settings.replay_folder.clone(),
                                            );
                                            let file = FileDialog::new()
                                                .add_filter("replay", &["replay"])
                                                .set_directory(replay_folder_path)
                                                .pick_file();
                                            if let Some(file) = file {
                                                gui_state.menu_stack.pop();
                                                *game = Some(Game::from_replay(
                                                    file.as_path(),
                                                    creation_interface,
                                                ));
                                                gui_state.game_just_started = true;
                                            }
                                        }
                                        if ui.button("Card Editor").clicked() {
                                            gui_state.menu_stack.push(GuiElement::CardEditor);
                                        }
                                        if ui.button("Exit to Desktop").clicked() {
                                            gui_state.should_exit = true;
                                        }
                                    });
                                });
                            });
                        });
                }
                Some(&GuiElement::SingleplayerMenu) => {
                    egui::Area::new("singleplayer menu")
                        .anchor(Align2::LEFT_TOP, Vec2::new(0.0, 0.0))
                        .show(&ctx, |ui| {
                            let menu_size = Rect::from_center_size(
                                ui.available_rect_before_wrap().center(),
                                ui.available_rect_before_wrap().size(),
                            );

                            ui.allocate_ui_at_rect(menu_size, |ui| {
                                ui.painter().rect_filled(
                                    ui.available_rect_before_wrap(),
                                    0.0,
                                    Color32::BLACK,
                                );
                                vertical_centerer(ui, |ui| {
                                    ui.vertical_centered(|ui| {
                                        for preset in settings.preset_settings.iter() {
                                            if ui.button(&preset.name).clicked() {
                                                gui_state.menu_stack.clear();
                                                *game = Some(Game::new(
                                                    settings,
                                                    preset.clone(),
                                                    &gui_state.gui_deck,
                                                    creation_interface,
                                                    None,
                                                ));
                                                gui_state.game_just_started = true;
                                            }
                                        }
                                        if ui.button("Back").clicked() {
                                            gui_state.menu_stack.pop();
                                        }
                                    });
                                });
                            });
                        });
                }
                Some(&GuiElement::EscMenu) => {
                    egui::Area::new("menu")
                        .anchor(Align2::LEFT_TOP, Vec2::new(0.0, 0.0))
                        .show(&ctx, |ui| {
                            ui.painter().rect_filled(
                                ui.available_rect_before_wrap(),
                                0.0,
                                Color32::BLACK.gamma_multiply(0.5),
                            );

                            let menu_size = Rect::from_center_size(
                                ui.available_rect_before_wrap().center(),
                                egui::vec2(300.0, 300.0),
                            );

                            ui.allocate_ui_at_rect(menu_size, |ui| {
                                ui.painter().rect_filled(
                                    ui.available_rect_before_wrap(),
                                    0.0,
                                    Color32::BLACK,
                                );
                                ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                                    ui.label(RichText::new("Menu").color(Color32::WHITE));
                                    if ui.button("Card Editor").clicked() {
                                        gui_state.menu_stack.push(GuiElement::CardEditor);
                                    }
                                    if let Some(game) = game {
                                        if game.game_mode.has_mode_gui() {
                                            if ui.button("Mode configuration").clicked() {
                                                gui_state.menu_stack.push(GuiElement::ModeGui);
                                            }
                                        }
                                    }
                                    if ui.button("Leave Game").clicked() {
                                        if let Some(game) = game {
                                            game.rollback_data.leave_game();
                                        }
                                        gui_state.menu_stack.clear();
                                        gui_state.menu_stack.push(GuiElement::MainMenu);
                                        *game = None;
                                    }
                                    if ui.button("Exit to Desktop").clicked() {
                                        gui_state.should_exit = true;
                                    }
                                });
                            });
                        });
                }
                Some(&GuiElement::CardEditor) => {
                    card_editor(&ctx, gui_state, game);
                }

                Some(&GuiElement::MultiplayerMenu) => {
                    egui::Area::new("multiplayer menu")
                        .anchor(Align2::LEFT_TOP, Vec2::new(0.0, 0.0))
                        .show(&ctx, |ui| {
                            let menu_size = Rect::from_center_size(
                                ui.available_rect_before_wrap().center(),
                                ui.available_rect_before_wrap().size(),
                            );

                            ui.allocate_ui_at_rect(menu_size, |ui| {
                                ui.painter().rect_filled(
                                    ui.available_rect_before_wrap(),
                                    0.0,
                                    Color32::BLACK,
                                );
                                vertical_centerer(ui, |ui| {
                                    ui.vertical_centered(|ui| {
                                        if ui.button("Host").clicked() {
                                            let client = reqwest::blocking::Client::new();
                                            let new_lobby_response = client
                                                .post(format!(
                                                    "http://{}create_lobby",
                                                    settings.remote_url.clone()
                                                ))
                                                .json(&settings.create_lobby_settings)
                                                .send();
                                            let new_lobby_response = match new_lobby_response {
                                                Ok(new_lobby_response) => new_lobby_response,
                                                Err(e) => {
                                                    println!("error creating lobby: {:?}", e);
                                                    gui_state.errors.push(
                                                        format!("Error creating lobby {}", e)
                                                            .to_string(),
                                                    );
                                                    return;
                                                }
                                            };
                                            let new_lobby_id = new_lobby_response.json::<String>();
                                            let new_lobby_id = match new_lobby_id {
                                                Ok(new_lobby_id) => new_lobby_id,
                                                Err(e) => {
                                                    println!("error creating lobby: {:?}", e);
                                                    gui_state.errors.push(
                                                        format!("Error creating lobby {}", e)
                                                            .to_string(),
                                                    );
                                                    return;
                                                }
                                            };
                                            println!("new lobby id: {}", new_lobby_id);
                                            *game = Some(Game::new(
                                                settings,
                                                settings.create_lobby_settings.clone(),
                                                &gui_state.gui_deck,
                                                creation_interface,
                                                Some(RoomId(new_lobby_id)),
                                            ));
                                            gui_state.menu_stack.push(GuiElement::LobbyQueue);
                                            gui_state.game_just_started = true;
                                        }
                                        if ui.button("Join").clicked() {
                                            gui_state.lobby_browser.update(settings);
                                            gui_state.menu_stack.push(GuiElement::LobbyBrowser);
                                        }
                                        if ui.button("Back").clicked() {
                                            gui_state.menu_stack.pop();
                                        }
                                    });
                                });
                            });
                        });
                }
                Some(&GuiElement::LobbyBrowser) => {
                    use egui_extras::{Column, TableBuilder};
                    egui::Area::new("lobby browser")
                        .anchor(Align2::LEFT_TOP, Vec2::new(0.0, 0.0))
                        .show(&ctx, |ui| {
                            let menu_size = Rect::from_center_size(
                                ui.available_rect_before_wrap().center(),
                                ui.available_rect_before_wrap().size(),
                            );
                            let lobby_list = match gui_state.lobby_browser.get_lobbies()
                            {
                                Ok(lobby_list) => lobby_list,
                                Err(err) => {
                                    gui_state.errors.push(format!(
                                        "Error getting lobbies: {}",
                                        err
                                    ));
                                    vec![]
                                }
                            };

                            ui.allocate_ui_at_rect(menu_size, |ui| {
                                ui.painter().rect_filled(
                                    ui.available_rect_before_wrap(),
                                    0.0,
                                    Color32::BLACK,
                                );
                                vertical_centerer(ui, |ui| {
                                    ui.vertical_centered(|ui| {
                                        horizontal_centerer(ui, |ui| {
                                            ui.vertical_centered(|ui| {
                                                let available_height = ui.available_height();
                                                let table = TableBuilder::new(ui)
                                                    .striped(true)
                                                    .resizable(false)
                                                    .cell_layout(egui::Layout::left_to_right(
                                                        egui::Align::Center,
                                                    ))
                                                    .column(Column::auto())
                                                    .column(Column::auto())
                                                    .column(Column::auto())
                                                    .column(Column::auto())
                                                    .column(Column::auto())
                                                    .max_scroll_height(available_height);

                                                table
                                                    .header(20.0, |mut header| {
                                                        header.col(|ui| {
                                                            ui.strong("Name");
                                                        });
                                                        header.col(|ui| {
                                                            ui.strong("Mode");
                                                        });
                                                        header.col(|ui| {
                                                            ui.strong("Map");
                                                        });
                                                        header.col(|ui| {
                                                            ui.strong("Players");
                                                        });
                                                        header.col(|ui| {
                                                            ui.strong("");
                                                        });
                                                    })
                                                    .body(|mut body| {
                                                        for lobby in lobby_list.iter() {
                                                            body.row(20.0, |mut row| {
                                                                row.col(|ui| {
                                                                    ui.label(lobby.name.clone());
                                                                });
                                                                row.col(|ui| {
                                                                    ui.label(
                                                                        lobby
                                                                            .settings
                                                                            .game_mode
                                                                            .get_name(),
                                                                    );
                                                                });
                                                                row.col(|ui| {
                                                                    ui.label(
                                                                        lobby
                                                                            .settings
                                                                            .world_gen
                                                                            .get_name(),
                                                                    );
                                                                });
                                                                row.col(|ui| {
                                                                    ui.label(format!(
                                                                        "{}/{}",
                                                                        lobby.player_count,
                                                                        lobby.settings.player_count
                                                                    ));
                                                                });
                                                                row.col(|ui| {
                                                                    if ui.button("Join").clicked() {
                                                                        gui_state
                                                                            .menu_stack
                                                                            .clear();
                                                                        *game = Some(Game::new(
                                                                            settings,
                                                                            lobby.settings.clone(),
                                                                            &gui_state.gui_deck,
                                                                            creation_interface,
                                                                            Some(
                                                                                lobby
                                                                                    .lobby_id
                                                                                    .clone(),
                                                                            ),
                                                                        ));
                                                                        gui_state.menu_stack.push(
                                                                            GuiElement::LobbyQueue,
                                                                        );
                                                                        gui_state
                                                                            .game_just_started =
                                                                            true;
                                                                    }
                                                                });
                                                            });
                                                        }
                                                    });
                                            });
                                        });
                                        if lobby_list.is_empty() {
                                            ui.label("No lobbies found");
                                        }
                                        if ui.button("Back").clicked() {
                                            gui_state.menu_stack.pop();
                                        }
                                    });
                                });
                            });
                        });
                }
                Some(&GuiElement::LobbyQueue) => {
                    egui::Area::new("lobby queue")
                        .anchor(Align2::LEFT_TOP, Vec2::new(0.0, 0.0))
                        .show(&ctx, |ui| {
                            let menu_size = Rect::from_center_size(
                                ui.available_rect_before_wrap().center(),
                                ui.available_rect_before_wrap().size(),
                            );

                            ui.allocate_ui_at_rect(menu_size, |ui| {
                                ui.painter().rect_filled(
                                    ui.available_rect_before_wrap(),
                                    0.0,
                                    Color32::BLACK,
                                );
                                vertical_centerer(ui, |ui| {
                                    ui.vertical_centered(|ui| {
                                        ui.label("Waiting for players to join...");
                                        if let Some(game) = game {
                                            ui.label(format!(
                                                "Players: {}/{}",
                                                game.rollback_data.player_count(),
                                                game.game_settings.player_count
                                            ));
                                        }
                                        if ui.button("Back").clicked() {
                                            gui_state.menu_stack.pop();
                                            *game = None;
                                        }
                                    });
                                });
                            });
                        });
                }
                Some(&GuiElement::ModeGui) => {
                    egui::Area::new("mode gui")
                        .anchor(Align2::LEFT_TOP, Vec2::new(0.0, 0.0))
                        .show(&ctx, |ui| {
                            let menu_size = Rect::from_center_size(
                                ui.available_rect_before_wrap().center(),
                                ui.available_rect_before_wrap().size(),
                            );

                            ui.allocate_ui_at_rect(menu_size, |ui| {
                                ui.painter().rect_filled(
                                    ui.available_rect_before_wrap(),
                                    0.0,
                                    Color32::BLACK,
                                );
                                vertical_centerer(ui, |ui| {
                                    ui.vertical_centered(|ui| {
                                        if let Some(game) = game {
                                            game.game_mode.mode_gui(ui, &mut game.rollback_data);
                                        }
                                        if ui.button("Back").clicked() {
                                            gui_state.menu_stack.pop();
                                            *game = None;
                                        }
                                    });
                                });
                            });
                        });
                }
                Some(GuiElement::DeckPicker) => {
                    egui::Area::new("deck picker")
                        .anchor(Align2::LEFT_TOP, Vec2::new(0.0, 0.0))
                        .show(&ctx, |ui| {
                            let menu_size = Rect::from_center_size(
                                ui.available_rect_before_wrap().center(),
                                ui.available_rect_before_wrap().size(),
                            );

                            ui.allocate_ui_at_rect(menu_size, |ui| {
                                ui.painter().rect_filled(
                                    ui.available_rect_before_wrap(),
                                    0.0,
                                    Color32::BLACK,
                                );
                                vertical_centerer(ui, |ui| {
                                    ui.vertical_centered(|ui| {
                                        let Ok(decks) = recurse_files(settings.card_dir.clone()) else {
                                            panic!("Cannot read directory {}", settings.card_dir);
                                        };
                                        for deck in decks {
                                            if ui.button(deck.file_name().unwrap().to_str().unwrap()).clicked() {
                                                gui_state.gui_deck = ron::from_str(fs::read_to_string(deck.as_path()).unwrap().as_str()).unwrap();
                                                gui_state.menu_stack.pop();
                                                gui_state.menu_stack.push(GuiElement::CardEditor);
                                            }
                                        }
                                        if ui.button("Back").clicked() {
                                            gui_state.menu_stack.pop();
                                        }
                                    });
                                });
                            });
                        });
                }
                None => {}
            }
            let corner_offset = 10.0;
            egui::Area::new("errors")
                .order(Order::Foreground)
                .anchor(
                    Align2::RIGHT_BOTTOM,
                    Vec2::new(-corner_offset, -corner_offset),
                )
                .show(&ctx, |ui| {
                    let mut errors_to_remove = vec![];
                    for (err_idx, error) in gui_state.errors.iter().enumerate() {
                        egui::Frame {
                            inner_margin: Margin {
                                left: 4.0,
                                right: 4.0,
                                top: 4.0,
                                bottom: 4.0,
                            },
                            ..Default::default()
                        }
                        .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
                        .fill(Color32::BLACK)
                        .rounding(ui.visuals().widgets.noninteractive.rounding)
                        .show(ui, |ui| {
                            ui.style_mut().wrap = Some(false);
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new(error).color(egui::Color32::WHITE));
                                if ui
                                    .button("X")
                                    .on_hover_cursor(egui::CursorIcon::PointingHand)
                                    .clicked()
                                {
                                    errors_to_remove.push(err_idx);
                                }
                            });
                        });
                    }
                    for err_idx in errors_to_remove.iter().rev() {
                        gui_state.errors.remove(*err_idx);
                    }
                });
        });
        let cb = self
            .frame
            .system
            .gui
            .draw_on_subpass_image(self.frame.framebuffer.extent());
        self.frame
            .command_buffer_builder
            .as_mut()
            .unwrap()
            .execute_commands(cb)
            .unwrap();
    }
}
