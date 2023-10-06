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
    egui::{self, emath, epaint, pos2, Align2, Color32, Rect, Rounding, Stroke, Vec2, RichText},
    Gui, GuiConfig,
};
use std::sync::Arc;
use vulkano::{
    buffer::{BufferContents, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SecondaryCommandBufferAbstract,
        SubpassContents,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::Queue,
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageAccess, ImageUsage, ImageViewAbstract},
    memory::allocator::StandardMemoryAllocator,
    pipeline::graphics::vertex_input::Vertex,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::Surface,
    sync::GpuFuture,
};
use winit::event_loop::EventLoop;

use crate::{raytracer::PointLightingSystem, rollback_manager::RollbackData, SimData, GuiState, settings_manager::Settings, gui::{cooldown, drop_target, drag_source, GuiElement, draw_base_card}};

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
    diffuse_buffer: Arc<ImageView<AttachmentImage>>,
    // Intermediate render target that will contain the normal vector in world coordinates of each
    // pixel of the scene.
    // The normal vector is the vector perpendicular to the surface of the object at this point.
    normals_buffer: Arc<ImageView<AttachmentImage>>,
    // Intermediate render target that will contain the depth of each pixel of the scene.
    // This is a traditional depth buffer. `0.0` means "near", and `1.0` means "far".
    depth_buffer: Arc<ImageView<AttachmentImage>>,

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
                    load: Clear,
                    store: Store,
                    format: final_output_format,
                    samples: 1,
                },
                // Will be bound to `self.diffuse_buffer`.
                diffuse: {
                    load: Clear,
                    store: DontCare,
                    format: Format::A2B10G10R10_UNORM_PACK32,
                    samples: 1,
                },
                // Will be bound to `self.normals_buffer`.
                normals: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: 1,
                },
                // Will be bound to `self.depth_buffer`.
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D32_SFLOAT,
                    samples: 1,
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
            AttachmentImage::with_usage(
                &memory_allocator,
                [1, 1],
                Format::A2B10G10R10_UNORM_PACK32,
                ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
            )
            .unwrap(),
        )
        .unwrap();
        let normals_buffer = ImageView::new_default(
            AttachmentImage::with_usage(
                &memory_allocator,
                [1, 1],
                Format::R16G16B16A16_SFLOAT,
                ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
            )
            .unwrap(),
        )
        .unwrap();
        let depth_buffer = ImageView::new_default(
            AttachmentImage::with_usage(
                &memory_allocator,
                [1, 1],
                Format::D32_SFLOAT,
                ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
            )
            .unwrap(),
        )
        .unwrap();

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            gfx_queue.device().clone(),
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
        final_image: Arc<dyn ImageViewAbstract + 'static>,
        world_to_framebuffer: Matrix4<f32>,
    ) -> Frame
    where
        F: GpuFuture + 'static,
    {
        // First of all we recreate `self.diffuse_buffer`, `self.normals_buffer` and
        // `self.depth_buffer` if their dimensions doesn't match the dimensions of the final image.
        let img_dims = final_image.image().dimensions().width_height();
        if self.diffuse_buffer.image().dimensions().width_height() != img_dims {
            // Note that we create "transient" images here. This means that the content of the
            // image is only defined when within a render pass. In other words you can draw to
            // them in a subpass then read them in another subpass, but as soon as you leave the
            // render pass their content becomes undefined.
            self.diffuse_buffer = ImageView::new_default(
                AttachmentImage::with_usage(
                    &self.memory_allocator,
                    img_dims,
                    Format::A2B10G10R10_UNORM_PACK32,
                    ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
                )
                .unwrap(),
            )
            .unwrap();
            self.normals_buffer = ImageView::new_default(
                AttachmentImage::with_usage(
                    &self.memory_allocator,
                    img_dims,
                    Format::R16G16B16A16_SFLOAT,
                    ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
                )
                .unwrap(),
            )
            .unwrap();
            self.depth_buffer = ImageView::new_default(
                AttachmentImage::with_usage(
                    &self.memory_allocator,
                    img_dims,
                    Format::D32_SFLOAT,
                    ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
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
            &self.command_buffer_allocator,
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
                SubpassContents::SecondaryCommandBuffers,
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
    command_buffer_builder: Option<
        AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, Arc<StandardCommandBufferAllocator>>,
    >,
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
                    .next_subpass(SubpassContents::SecondaryCommandBuffers)
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
                    .end_render_pass()
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
    pub fn execute<C>(&mut self, command_buffer: C)
    where
        C: SecondaryCommandBufferAbstract + 'static,
    {
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
    pub fn raytrace(
        &mut self,
        voxels: Subbuffer<[[u32; 2]]>,
        rollback_manager: &RollbackData,
        sim_data: &mut SimData,
        gui_state: &mut GuiState,
        settings: &Settings,
    ) {
        let command_buffer = {
            self.frame.system.ambient_lighting_system.draw(
                self.frame.framebuffer.extent(),
                self.frame.system.diffuse_buffer.clone(),
                self.frame.system.normals_buffer.clone(),
                self.frame.system.depth_buffer.clone(),
                self.frame.world_to_framebuffer.invert().unwrap(),
                voxels,
                rollback_manager,
                sim_data,
                &settings.graphics_settings
            )
        };

        self.frame
            .command_buffer_builder
            .as_mut()
            .unwrap()
            .execute_commands(command_buffer)
            .unwrap();
        self.frame.system.gui.immediate_ui(|gui| {
            let ctx = gui.context();
            let screen_size = ctx.screen_rect().size();
            // Fill egui UI layout here
            egui::Area::new("crosshair").show(&ctx, |ui| {
                let center = screen_size / 2.0;
                let thickness = 1.0;
                let color = Color32::from_additive_luminance(255);
                let crosshair_size = 10.0;

                ui.painter().add(epaint::Shape::line_segment(
                    [
                        pos2(-crosshair_size, 0.0) + center,
                        pos2(crosshair_size, 0.0) + center,
                    ],
                    Stroke::new(thickness, color),
                ));
                ui.painter().add(epaint::Shape::line_segment(
                    [
                        pos2(0.0, -crosshair_size) + center,
                        pos2(0.0, crosshair_size) + center,
                    ],
                    Stroke::new(thickness, color),
                ));
            });

            let corner_offset = 10.0;
            egui::Area::new("healthbar")
                .anchor(
                    Align2::LEFT_BOTTOM,
                    Vec2::new(corner_offset, -corner_offset),
                )
                .show(&ctx, |ui| {
                    let thickness = 1.0;
                    let color = Color32::from_additive_luminance(255);
                    let player_health =
                        rollback_manager.cached_current_state.players[0].health as f32;
                    let player_max_health = 100.0;

                    ui.label(RichText::new(format!("{} / {}", player_health, player_max_health)).color(Color32::WHITE));
                    let desired_size = egui::vec2(200.0, 30.0);
                    let (_id, rect) = ui.allocate_space(desired_size);

                    let to_screen = emath::RectTransform::from_to(
                        Rect::from_x_y_ranges(0.0..=1.0, 0.0..=1.0),
                        rect,
                    );

                    let healthbar_size = Rect::from_min_max(
                        to_screen * pos2(0.0, 0.0),
                        to_screen * pos2(1.0, 1.0),
                    );
                    let health_size = Rect::from_min_max(
                        to_screen * pos2(0.0, 0.0),
                        to_screen * pos2(
                            player_health / player_max_health,
                            1.0,
                        ),
                    );

                    ui.painter().add(epaint::Shape::rect_stroke(
                        healthbar_size,
                        Rounding::none(),
                        Stroke::new(thickness, color),
                    ));
                    ui.painter().add(epaint::Shape::rect_filled(
                        health_size,
                        Rounding::none(),
                        color,
                    ));
                });
            
            let respawn_time =
                rollback_manager.cached_current_state.players[0].respawn_timer;
            if respawn_time > 0.0 {
                egui::Area::new("respawn")
                    .anchor(
                        Align2::LEFT_TOP,
                        Vec2::new(corner_offset, corner_offset),
                    )
                    .show(&ctx, |ui| {
                        ui.label(RichText::new("You have died").color(Color32::WHITE)); 
                        ui.label(RichText::new(format!("Respawn in {}", respawn_time)).color(Color32::WHITE)); 
                    });
            }

            egui::Area::new("cooldowns")
                .anchor(
                    Align2::RIGHT_BOTTOM,
                    Vec2::new(-corner_offset, -corner_offset),
                )
                .show(&ctx, |ui| {
                    for (ability_idx, ability) in rollback_manager.cached_current_state.players[0].abilities.iter().enumerate() {
                        ui.add(cooldown(format!("{}", settings.ability_controls[ability_idx]).as_str(), ability));
                    }
                });
            match gui_state.menu_stack.last() {
                Some(&GuiElement::MainMenu) => {
                    egui::Area::new("menu")
                    .anchor(
                        Align2::LEFT_TOP,
                        Vec2::new(0.0, 0.0),
                    )
                    .show(&ctx, |ui| {
                        ui.painter().rect_filled(
                            ui.available_rect_before_wrap(),
                            0.0,
                            Color32::BLACK.gamma_multiply(0.5),
                        );

                        let menu_size = Rect::from_center_size(ui.available_rect_before_wrap().center(), egui::vec2(300.0, 300.0));
                        
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
                                if ui.button("Exit to Desktop").clicked() {
                                    gui_state.should_exit = true;
                                }
                                if ui.button("Close").clicked() {
                                    gui_state.menu_stack.pop();
                                }
                            });
                        });
                    });
                }
                Some(&GuiElement::CardEditor) => {
                    egui::Area::new("card editor")
                    .anchor(
                        Align2::LEFT_TOP,
                        Vec2::new(0.0, 0.0),
                    )
                    .show(&ctx, |ui| {
                        ui.painter().rect_filled(
                            ui.available_rect_before_wrap(),
                            0.0,
                            Color32::BLACK.gamma_multiply(0.5),
                        );

                        let menu_size = Rect::from_center_size(ui.available_rect_before_wrap().center(), ui.available_rect_before_wrap().size() * egui::vec2(0.75, 0.75));
                        
                        ui.allocate_ui_at_rect(menu_size, |ui| {
                            ui.painter().rect_filled(
                                ui.available_rect_before_wrap(),
                                0.0,
                                Color32::BLACK,
                            );
                            ui.with_layout(egui::Layout::top_down(egui::Align::LEFT), |ui| {
                                ui.label(RichText::new("Card Editor").color(Color32::WHITE));
                                
                                let id_source = "my_drag_and_drop_demo";
                                // let mut source_path = None;
                                // let mut drop_path = None;

                                for (ability_idx, card) in gui_state.gui_cards.iter().enumerate() {
                                    ui.horizontal_top(|ui| {
                                        {
                                            let desired_size = ui.spacing().interact_size.y * egui::vec2(3.0, 3.0);
                                            let (rect, _response) = ui.allocate_exact_size(desired_size, egui::Sense::click());
                                            let font = egui::FontId::proportional(24.0);
                                            ui.painter().rect_filled(rect, 0.0, Color32::LIGHT_GRAY);
                                            ui.painter().text(
                                                rect.center(),
                                                Align2::CENTER_CENTER,
                                                format!("{}", settings.ability_controls[ability_idx]).as_str(),
                                                font,
                                                Color32::BLACK,
                                            );
                                        }
                                        draw_base_card(ui, &card);
                                    });
                                }
                                // for card in gui_state.gui_cards.iter() {
                                //     let can_accept_what_is_being_dragged = true; // We accept anything being dragged (for now) ¯\_(ツ)_/¯
                                //     let response = drop_target(ui, can_accept_what_is_being_dragged, |ui| {
                                //         ui.set_min_size(egui::vec2(64.0, 100.0));
                                //         for (row_idx, item) in column.iter().enumerate() {
                                //             let item_id = egui::Id::new(id_source).with(col_idx).with(row_idx);
                                //             drag_source(ui, item_id, |ui| {
                                //                 ui.add(Label::new(item).sense(Sense::click()));
                                //             });

                                //             if ui.memory(|mem| mem.is_being_dragged(item_id)) {
                                //                 source_col_row = Some((col_idx, row_idx));
                                //             }
                                //         }
                                //     })
                                //     .response;

                                //     let is_being_dragged = ui.memory(|mem| mem.is_anything_being_dragged());
                                //     if is_being_dragged && can_accept_what_is_being_dragged && response.hovered() {
                                //         drop_col = Some(col_idx);
                                //     }
                                // }

                                // if let Some((source_col, source_row)) = source_col_row {
                                //     if let Some(drop_col) = drop_col {
                                //         if ui.input(|i| i.pointer.any_released()) {
                                //             // do the drop:
                                //             let item = columns[source_col].remove(source_row);
                                //             columns[drop_col].push(item);
                                //         }
                                //     }
                                // }
                            });
                        });
                    });
                }
                None => {}
            }
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
