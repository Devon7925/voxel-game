use std::collections::VecDeque;

use egui_winit_vulkano::egui::{
    self, emath, epaint, pos2, text::LayoutJob, vec2, Align, Align2, Color32, CursorIcon, FontId,
    Id, InnerResponse, Label, LayerId, Order, Rect, Rgba, RichText, Rounding, ScrollArea, Sense,
    Shape, Stroke, TextFormat, Ui, Vec2,
};
use itertools::Itertools;

use crate::{
    card_system::{
        Ability, BaseCard, Cooldown, CooldownModifier, DraggableCard, Effect, Keybind,
        MultiCastModifier, ProjectileModifier, ProjectileModifierType, ReferencedStatusEffect,
        SimpleCooldownModifier, StatusEffect, VoxelMaterial,
    },
    lobby_browser::LobbyBrowser,
    rollback_manager::{AppliedStatusEffect, Entity, HealthSection, PlayerAbility},
    settings_manager::Control,
    utils::{translate_egui_key_code, translate_egui_pointer_button},
};

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum GuiElement {
    EscMenu,
    CardEditor,
    MainMenu,
    MultiplayerMenu,
    LobbyBrowser,
    LobbyQueue,
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum PaletteState {
    ProjectileModifiers,
    BaseCards,
    AdvancedProjectileModifiers,
    MultiCastModifiers,
    CooldownModifiers,
    Materials,
    Effects,
    StatusEffects,
    Dock,
}

pub struct GuiState {
    pub menu_stack: Vec<GuiElement>,
    pub errors: Vec<String>,
    pub gui_cards: Vec<Cooldown>,
    pub dock_cards: Vec<DraggableCard>,
    pub cooldown_cache_refresh_delay: f32,
    pub palette_state: PaletteState,
    pub lobby_browser: LobbyBrowser,
    pub should_exit: bool,
    pub game_just_started: bool,
}

// Helper function to center arbitrary widgets. It works by measuring the width of the widgets after rendering, and
// then using that offset on the next frame.
pub fn vertical_centerer(ui: &mut Ui, add_contents: impl FnOnce(&mut Ui)) {
    ui.vertical(|ui| {
        let id = ui.id().with("_v_centerer");
        let last_height: Option<f32> = ui.memory_mut(|mem| mem.data.get_temp(id));
        if let Some(last_height) = last_height {
            ui.add_space((ui.available_height() - last_height) / 2.0);
        }
        let res = ui
            .scope(|ui| {
                add_contents(ui);
            })
            .response;
        let height = res.rect.height();
        ui.memory_mut(|mem| mem.data.insert_temp(id, height));

        // Repaint if height changed
        match last_height {
            None => ui.ctx().request_repaint(),
            Some(last_height) if last_height != height => ui.ctx().request_repaint(),
            Some(_) => {}
        }
    });
}

pub fn horizontal_centerer(ui: &mut Ui, add_contents: impl FnOnce(&mut Ui)) {
    ui.horizontal(|ui| {
        let id = ui.id().with("_h_centerer");
        let last_width: Option<f32> = ui.memory_mut(|mem| mem.data.get_temp(id));
        if let Some(last_width) = last_width {
            ui.add_space((ui.available_width() - last_width) / 2.0);
        }
        let res = ui
            .scope(|ui| {
                add_contents(ui);
            })
            .response;
        let width = res.rect.width();
        ui.memory_mut(|mem| mem.data.insert_temp(id, width));

        // Repaint if height changed
        match last_width {
            None => ui.ctx().request_repaint(),
            Some(last_width) if last_width != width => ui.ctx().request_repaint(),
            Some(_) => {}
        }
    });
}

fn cooldown_ui(ui: &mut egui::Ui, ability: &PlayerAbility, ability_idx: usize) -> egui::Response {
    let desired_size = ui.spacing().interact_size.y * egui::vec2(3.0, 3.0);
    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());

    if ui.is_rect_visible(rect) {
        let font = egui::FontId::proportional(24.0);
        if ability.cooldown > ability.ability.add_charge as f32 * ability.value.0 {
            ui.painter().rect_filled(rect, 0.0, Color32::DARK_GRAY);
            ui.painter().text(
                rect.center(),
                Align2::CENTER_CENTER,
                format!(
                    "{}",
                    (ability.cooldown - ability.ability.add_charge as f32 * ability.value.0).ceil()
                        as i32
                ),
                font.clone(),
                Color32::WHITE,
            );
            return response;
        }
        ui.painter().rect_filled(rect, 0.0, Color32::LIGHT_GRAY);
        {
            let keybind = &ability.ability.abilities[ability_idx].1;
            if let Some(key) = keybind.get_simple_representation() {
                ui.painter().text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    format!("{}", key),
                    font,
                    Color32::BLACK,
                );
            }
        }
        if ability.ability.add_charge > 0 {
            let font = egui::FontId::proportional(12.0);
            let charge_count = ((1 + ability.ability.add_charge) as f32
                - ability.cooldown / ability.value.0)
                .floor() as i32;
            ui.painter()
                .circle_filled(rect.right_top(), 8.0, Color32::GRAY);
            ui.painter().text(
                rect.right_top(),
                Align2::CENTER_CENTER,
                format!("{}", charge_count),
                font,
                Color32::BLACK,
            );
        }
    }

    response
}

pub fn cooldown<'a>(ability: &'a PlayerAbility, ability_idx: usize) -> impl egui::Widget + 'a {
    move |ui: &mut egui::Ui| cooldown_ui(ui, ability, ability_idx)
}

pub fn drag_source(ui: &mut Ui, id: Id, dragable: bool, body: impl FnOnce(&mut Ui)) {
    let is_being_dragged = ui.memory(|mem| mem.is_being_dragged(id));

    if !is_being_dragged || !dragable {
        //load from previous frame
        let prev_frame_area: Option<Rect> = ui.data(|d| d.get_temp(id));
        if let Some(area) = prev_frame_area {
            if dragable {
                // Check for drags:
                let response = ui.interact(area, id, Sense::drag());
                if response.hovered() {
                    ui.ctx().set_cursor_icon(CursorIcon::Grab);
                }
            }
        }
        let response = ui.scope(body).response;
        //store for next frame
        ui.data_mut(|d| d.insert_temp(id, response.rect));
    } else {
        ui.ctx().set_cursor_icon(CursorIcon::Grabbing);

        // Paint the body to a new layer:
        let layer_id = LayerId::new(Order::Tooltip, id);
        let response = ui.with_layer_id(layer_id, body).response;

        // Now we move the visuals of the body to where the mouse is.
        // Normally you need to decide a location for a widget first,
        // because otherwise that widget cannot interact with the mouse.
        // However, a dragged component cannot be interacted with anyway
        // (anything with `Order::Tooltip` always gets an empty [`Response`])
        // So this is fine!

        if let Some(pointer_pos) = ui.ctx().pointer_interact_pos() {
            let delta = pointer_pos - response.rect.center();
            ui.ctx().translate_layer(layer_id, delta);
        }
    }
}

pub fn darken(color: Color32, factor: f32) -> Color32 {
    let mut color = Rgba::from(color);
    for i in 0..3 {
        color[i] = color[i] * factor;
    }
    color.into()
}

pub fn drop_target<R>(
    ui: &mut Ui,
    can_accept_what_is_being_dragged: bool,
    body: impl FnOnce(&mut Ui) -> R,
) -> InnerResponse<R> {
    let is_being_dragged = ui.memory(|mem| mem.is_anything_being_dragged());

    let margin = egui::Vec2::splat(4.0);

    let outer_rect_bounds = ui.available_rect_before_wrap();
    let inner_rect = outer_rect_bounds.shrink2(margin);
    let where_to_put_background = ui.painter().add(Shape::Noop);
    let mut content_ui = ui.child_ui(inner_rect, *ui.layout());
    let ret = body(&mut content_ui);
    let outer_rect = Rect::from_min_max(outer_rect_bounds.min, content_ui.min_rect().max + margin);
    let (rect, response) = ui.allocate_at_least(outer_rect.size(), Sense::hover());

    let style = if is_being_dragged && can_accept_what_is_being_dragged && response.hovered() {
        ui.visuals().widgets.active
    } else {
        ui.visuals().widgets.inactive
    };

    let mut fill_color = style.bg_fill;
    let mut stroke = style.bg_stroke;
    if is_being_dragged && !can_accept_what_is_being_dragged {
        fill_color = ui.visuals().gray_out(fill_color);
        stroke.color = ui.visuals().gray_out(stroke.color);
    }

    ui.painter().set(
        where_to_put_background,
        epaint::RectShape::new(rect, style.rounding, fill_color, stroke),
    );

    InnerResponse::new(ret, response)
}

#[derive(Debug)]
pub enum DragableType {
    ProjectileModifier,
    MultiCastModifier,
    CooldownModifier,
    BaseCard,
}

#[derive(Debug)]
pub enum DropableType {
    MultiCastBaseCard,
    BaseNone,
    BaseProjectile,
    Cooldown,
    Palette,
}

#[derive(Debug)]
pub enum ModificationType {
    Add,
    Remove,
}

pub fn is_valid_drag(from: &DragableType, to: &DropableType) -> bool {
    match (from, to) {
        (DragableType::ProjectileModifier, DropableType::BaseProjectile) => true,
        (DragableType::MultiCastModifier, DropableType::MultiCastBaseCard) => true,
        (DragableType::BaseCard, DropableType::MultiCastBaseCard) => true,
        (DragableType::BaseCard, DropableType::BaseNone) => true,
        (DragableType::CooldownModifier, DropableType::Cooldown) => true,
        (DragableType::BaseCard, DropableType::Cooldown) => true,
        (_, DropableType::Palette) => true,
        _ => false,
    }
}

pub fn draw_cooldown(
    ui: &mut Ui,
    cooldown: &mut Cooldown,
    path: &mut VecDeque<u32>,
    source_path: &mut Option<(VecDeque<u32>, DragableType)>,
    drop_path: &mut Option<(VecDeque<u32>, DropableType)>,
    modify_path: &mut Option<(VecDeque<u32>, ModificationType)>,
    total_impact: f32,
) {
    let can_accept_what_is_being_dragged = true;
    let id_source = "my_drag_and_drop_demo";
    let cooldown_value = cooldown.get_cooldown_recovery(total_impact);
    let Cooldown {
        abilities,
        modifiers,
    } = cooldown;
    ui.visuals_mut().widgets.active.rounding = Rounding::from(CARD_UI_ROUNDING);
    ui.visuals_mut().widgets.inactive.rounding = Rounding::from(CARD_UI_ROUNDING);
    ui.visuals_mut().override_text_color = Some(Color32::WHITE);
    ui.visuals_mut().widgets.inactive.bg_stroke = Stroke::new(0.5, Color32::from_rgb(255, 0, 255));
    let response = drop_target(ui, can_accept_what_is_being_dragged, |ui| {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.add_space(CARD_UI_SPACING);
                ui.label("Cooldown");
                ui.label(format!("{:.2}s", cooldown_value.0))
                    .on_hover_text(format!(
                        "Recoveries: {}",
                        cooldown_value
                            .1
                            .iter()
                            .map(|v| format!("{:.2}s", v))
                            .join(", ")
                    ));
                path.push_back(0);
                for (mod_idx, modifier) in modifiers.iter().enumerate() {
                    path.push_back(mod_idx as u32);
                    DraggableCard::CooldownModifier(modifier.clone()).draw_draggable(
                        ui,
                        path,
                        source_path,
                        drop_path,
                        modify_path,
                    );
                    path.pop_back();
                }
                path.pop_back();
                ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                    if ui.button("X").clicked() {
                        *modify_path = Some((path.clone(), ModificationType::Remove));
                    }
                    ui.add_space(CARD_UI_SPACING);
                });
            });
            path.push_back(1);
            for (ability_idx, mut ability) in abilities.iter_mut().enumerate() {
                path.push_back(ability_idx as u32);
                ui.horizontal(|ui| {
                    draw_keybind(ui, &mut ability);
                    draw_base_card(ui, &ability.card, path, source_path, drop_path, modify_path);
                });
                path.pop_back();
            }
            path.pop_back();

            path.push_back(0);
            for (modifier_idx, _modifier) in modifiers.iter().enumerate() {
                path.push_back(modifier_idx as u32);
                let item_id = egui::Id::new(id_source).with(path.clone());
                if source_path.is_none() && ui.memory(|mem| mem.is_being_dragged(item_id)) {
                    *source_path = Some((path.clone(), DragableType::CooldownModifier));
                }
                path.pop_back();
            }
            path.pop_back();
        });
    })
    .response;

    if drop_path.is_none() {
        let is_being_dragged = ui.memory(|mem| mem.is_anything_being_dragged());
        if is_being_dragged && can_accept_what_is_being_dragged && response.hovered() {
            *drop_path = Some((path.clone(), DropableType::Cooldown));
        }
    }
}

fn draw_keybind(ui: &mut Ui, ability: &mut Ability) {
    let desired_size = ui.spacing().interact_size.y * egui::vec2(3.0, 3.0);
    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());

    if ability.is_keybind_selected {
        ui.ctx().input(|input| {
            for key in input.events.iter() {
                if let egui::Event::Key { key, pressed, .. } = key {
                    if *pressed {
                        ability.keybind =
                            Keybind::Pressed(Control::Key(translate_egui_key_code(*key)));
                        ability.is_keybind_selected = false;
                    }
                }
                if let egui::Event::PointerButton {
                    button, pressed, ..
                } = key
                {
                    if *pressed {
                        ability.keybind = Keybind::Pressed(Control::Mouse(
                            translate_egui_pointer_button(*button),
                        ));
                        ability.is_keybind_selected = false;
                    }
                }
            }
        });
    } else if response.clicked() {
        ability.is_keybind_selected = true;
    }

    if ui.is_rect_visible(rect) {
        let font = egui::FontId::proportional(24.0);

        let fill_color = if ability.is_keybind_selected {
            Color32::DARK_GRAY
        } else {
            Color32::LIGHT_GRAY
        };
        ui.painter().rect_filled(rect, 0.0, fill_color);
        {
            if let Some(key) = ability.keybind.get_simple_representation() {
                ui.painter().text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    format!("{}", key),
                    font,
                    Color32::BLACK,
                );
            }
        }
    }
}

const CARD_UI_SPACING: f32 = 3.0;
const CARD_UI_ROUNDING: f32 = 3.0;
impl DraggableCard {
    pub fn get_type(&self) -> DragableType {
        match self {
            DraggableCard::BaseCard(_) => DragableType::BaseCard,
            DraggableCard::CooldownModifier(_) => DragableType::CooldownModifier,
            DraggableCard::MultiCastModifier(_) => DragableType::MultiCastModifier,
            DraggableCard::ProjectileModifier(_) => DragableType::ProjectileModifier,
        }
    }

    pub fn draw_draggable(
        &self,
        ui: &mut Ui,
        path: &mut VecDeque<u32>,
        source_path: &mut Option<(VecDeque<u32>, DragableType)>,
        dest_path: &mut Option<(VecDeque<u32>, DropableType)>,
        modify_path: &mut Option<(VecDeque<u32>, ModificationType)>,
    ) {
        let id_source = "my_drag_and_drop_demo";
        match self {
            DraggableCard::BaseCard(card) => {
                draw_base_card(ui, card, path, source_path, dest_path, modify_path);
            }
            DraggableCard::CooldownModifier(modifier) => {
                let item_id = egui::Id::new(id_source).with(path.clone());
                let hover_text = modifier.get_hover_text();
                match modifier {
                    CooldownModifier::SimpleCooldownModifier(
                        SimpleCooldownModifier::AddCharge,
                        v,
                    ) => add_hoverable_basic_modifer(
                        ui,
                        item_id,
                        "Add Charge",
                        *v,
                        hover_text,
                        modify_path,
                        path,
                    ),
                    CooldownModifier::SimpleCooldownModifier(
                        SimpleCooldownModifier::AddCooldown,
                        v,
                    ) => add_hoverable_basic_modifer(
                        ui,
                        item_id,
                        "Add Cooldown",
                        *v,
                        hover_text,
                        modify_path,
                        path,
                    ),
                    CooldownModifier::SimpleCooldownModifier(
                        SimpleCooldownModifier::MultiplyImpact,
                        v,
                    ) => add_hoverable_basic_modifer(
                        ui,
                        item_id,
                        "Multiply Impact",
                        *v,
                        hover_text,
                        modify_path,
                        path,
                    ),
                }
            }
            DraggableCard::MultiCastModifier(modifier) => {
                let item_id = egui::Id::new(id_source).with(path.clone());
                let hover_text = modifier.get_hover_text();
                match modifier {
                    MultiCastModifier::Spread(v) => add_hoverable_basic_modifer(
                        ui,
                        item_id,
                        "Spread",
                        *v,
                        hover_text,
                        modify_path,
                        path,
                    ),
                    MultiCastModifier::Duplication(v) => add_hoverable_basic_modifer(
                        ui,
                        item_id,
                        "Duplication",
                        *v,
                        hover_text,
                        modify_path,
                        path,
                    ),
                }
            }
            DraggableCard::ProjectileModifier(modifier) => {
                let item_id = egui::Id::new(id_source).with(path.clone());
                let mut advanced_modifier = false;
                match modifier {
                    ProjectileModifier::SimpleModify(ty, v) => {
                        let name = match ty {
                            ProjectileModifierType::Gravity => "Gravity",
                            ProjectileModifierType::Health => "Health",
                            ProjectileModifierType::Height => "Height",
                            ProjectileModifierType::Length => "Length",
                            ProjectileModifierType::Lifetime => "Lifetime",
                            ProjectileModifierType::Speed => "Speed",
                            ProjectileModifierType::Width => "Width",
                        };
                        add_hoverable_basic_modifer(
                            ui,
                            item_id,
                            name,
                            *v,
                            modifier.get_hover_text(),
                            modify_path,
                            path,
                        )
                    }
                    ProjectileModifier::NoEnemyFire => add_hoverable_basic_modifer(
                        ui,
                        item_id,
                        "No Enemy Fire",
                        "",
                        modifier.get_hover_text(),
                        modify_path,
                        path,
                    ),
                    ProjectileModifier::FriendlyFire => add_hoverable_basic_modifer(
                        ui,
                        item_id,
                        "Friendly Fire",
                        "",
                        modifier.get_hover_text(),
                        modify_path,
                        path,
                    ),
                    ProjectileModifier::LockToOwner => add_hoverable_basic_modifer(
                        ui,
                        item_id,
                        "Lock To Owner",
                        "",
                        modifier.get_hover_text(),
                        modify_path,
                        path,
                    ),
                    ProjectileModifier::PiercePlayers => add_hoverable_basic_modifer(
                        ui,
                        item_id,
                        "Pierce Players",
                        "",
                        modifier.get_hover_text(),
                        modify_path,
                        path,
                    ),
                    ProjectileModifier::WallBounce => add_hoverable_basic_modifer(
                        ui,
                        item_id,
                        "Wall Bounce",
                        "",
                        modifier.get_hover_text(),
                        modify_path,
                        path,
                    ),
                    modifier if modifier.is_advanced() => {
                        advanced_modifier = true;
                    }
                    _ => panic!("Invalid State"),
                }
                if advanced_modifier {
                    ui.horizontal(|ui| {
                        ui.add_space(CARD_UI_SPACING);
                        match modifier {
                            ProjectileModifier::OnHit(base_card) => {
                                drag_source(ui, item_id, true, |ui| {
                                    ui.label("On Hit");
                                    path.push_back(0);
                                    draw_base_card(
                                        ui,
                                        base_card,
                                        path,
                                        source_path,
                                        dest_path,
                                        modify_path,
                                    );
                                    path.pop_back();
                                });
                            }
                            ProjectileModifier::OnHeadshot(base_card) => {
                                drag_source(ui, item_id, true, |ui| {
                                    ui.label("On Headshot");
                                    path.push_back(0);
                                    draw_base_card(
                                        ui,
                                        base_card,
                                        path,
                                        source_path,
                                        dest_path,
                                        modify_path,
                                    );
                                    path.pop_back();
                                });
                            }
                            ProjectileModifier::OnExpiry(base_card) => {
                                drag_source(ui, item_id, true, |ui| {
                                    ui.label("On Expiry");
                                    path.push_back(0);
                                    draw_base_card(
                                        ui,
                                        base_card,
                                        path,
                                        source_path,
                                        dest_path,
                                        modify_path,
                                    );
                                    path.pop_back();
                                });
                            }
                            ProjectileModifier::OnTrigger(id, base_card) => {
                                drag_source(ui, item_id, true, |ui| {
                                    add_basic_modifer(ui, "On Trigger", *id, modify_path, path);
                                    path.push_back(0);
                                    draw_base_card(
                                        ui,
                                        base_card,
                                        path,
                                        source_path,
                                        dest_path,
                                        modify_path,
                                    );
                                    path.pop_back();
                                });
                            }
                            ProjectileModifier::Trail(frequency, base_card) => {
                                drag_source(ui, item_id, true, |ui| {
                                    add_basic_modifer(ui, "Trail", *frequency, modify_path, path);
                                    path.push_back(0);
                                    draw_base_card(
                                        ui,
                                        base_card,
                                        path,
                                        source_path,
                                        dest_path,
                                        modify_path,
                                    );
                                    path.pop_back();
                                });
                            }
                            _ => panic!("Invalid State"),
                        }
                    });
                }
            }
        }
    }
}

pub fn draw_base_card(
    ui: &mut Ui,
    card: &BaseCard,
    path: &mut VecDeque<u32>,
    source_path: &mut Option<(VecDeque<u32>, DragableType)>,
    dest_path: &mut Option<(VecDeque<u32>, DropableType)>,
    modify_path: &mut Option<(VecDeque<u32>, ModificationType)>,
) {
    let id_source = "my_drag_and_drop_demo";

    let item_id = egui::Id::new(id_source).with(path.clone());
    let is_draggable = !matches!(card, BaseCard::None | BaseCard::Palette(_, _));
    let can_accept_what_is_being_dragged = true; // We accept anything being dragged (for now) ¯\_(ツ)_/¯
    drag_source(ui, item_id, is_draggable, |ui| match card {
        BaseCard::Projectile(modifiers) => {
            ui.vertical(|ui| {
                ui.visuals_mut().widgets.inactive.bg_stroke = Stroke::new(0.5, Color32::WHITE);
                ui.visuals_mut().widgets.inactive.bg_fill =
                    darken(ui.visuals_mut().widgets.inactive.bg_stroke.color, 0.25);

                let response = drop_target(ui, can_accept_what_is_being_dragged, |ui| {
                    let mut advanced_modifiers = vec![];
                    ui.horizontal(|ui| {
                        ui.add_space(CARD_UI_SPACING);
                        ui.vertical(|ui| {
                            ui.add_space(CARD_UI_SPACING);
                            ui.horizontal_wrapped(|ui| {
                                ui.label("Create Projectile");
                                for (modifier_idx, modifier) in modifiers.iter().enumerate() {
                                    if modifier.is_advanced() {
                                        advanced_modifiers.push((modifier_idx, modifier));
                                        continue;
                                    }
                                    path.push_back(modifier_idx as u32);
                                    DraggableCard::ProjectileModifier(modifier.clone())
                                        .draw_draggable(
                                            ui,
                                            path,
                                            source_path,
                                            dest_path,
                                            modify_path,
                                        );
                                    path.pop_back();
                                }
                            });

                            for (modifier_idx, modifier) in advanced_modifiers.into_iter() {
                                path.push_back(modifier_idx as u32);
                                DraggableCard::ProjectileModifier(modifier.clone()).draw_draggable(
                                    ui,
                                    path,
                                    source_path,
                                    dest_path,
                                    modify_path,
                                );
                                path.pop_back();
                            }
                            ui.add_space(CARD_UI_SPACING);
                        });
                        ui.add_space(CARD_UI_SPACING);
                    });

                    for (modifier_idx, _modifier) in modifiers.iter().enumerate() {
                        path.push_back(modifier_idx as u32);
                        let item_id = egui::Id::new(id_source).with(path.clone());
                        if source_path.is_none() && ui.memory(|mem| mem.is_being_dragged(item_id)) {
                            *source_path = Some((path.clone(), DragableType::ProjectileModifier));
                        }
                        path.pop_back();
                    }
                })
                .response;

                if dest_path.is_none() {
                    let is_being_dragged = ui.memory(|mem| mem.is_anything_being_dragged());
                    if is_being_dragged && can_accept_what_is_being_dragged && response.hovered() {
                        *dest_path = Some((path.clone(), DropableType::BaseProjectile));
                    }
                }
            });
        }
        BaseCard::MultiCast(cards, modifiers) => {
            ui.visuals_mut().widgets.inactive.bg_stroke = Stroke::new(0.5, Color32::YELLOW);
            ui.visuals_mut().widgets.inactive.bg_fill =
                darken(ui.visuals_mut().widgets.inactive.bg_stroke.color, 0.25);
            let response = drop_target(ui, can_accept_what_is_being_dragged, |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.add_space(CARD_UI_SPACING);
                        ui.label("Multicast");
                        path.push_back(0);
                        for (mod_idx, modifier) in modifiers.iter().enumerate() {
                            path.push_back(mod_idx as u32);
                            DraggableCard::MultiCastModifier(modifier.clone()).draw_draggable(
                                ui,
                                path,
                                source_path,
                                dest_path,
                                modify_path,
                            );
                            path.pop_back();
                        }
                        path.pop_back();
                        ui.add_space(CARD_UI_SPACING);
                    });
                    path.push_back(1);
                    for (card_idx, card) in cards.iter().enumerate() {
                        path.push_back(card_idx as u32);
                        draw_base_card(ui, card, path, source_path, dest_path, modify_path);
                        path.pop_back();
                    }
                    path.pop_back();
                });

                path.push_back(0);
                for (modifier_idx, _modifier) in modifiers.iter().enumerate() {
                    path.push_back(modifier_idx as u32);
                    let item_id = egui::Id::new(id_source).with(path.clone());
                    if source_path.is_none() && ui.memory(|mem| mem.is_being_dragged(item_id)) {
                        *source_path = Some((path.clone(), DragableType::MultiCastModifier));
                    }
                    path.pop_back();
                }
                path.pop_back();
            })
            .response;

            if dest_path.is_none() {
                let is_being_dragged = ui.memory(|mem| mem.is_anything_being_dragged());
                if is_being_dragged && can_accept_what_is_being_dragged && response.hovered() {
                    *dest_path = Some((path.clone(), DropableType::MultiCastBaseCard));
                }
            }
        }
        BaseCard::CreateMaterial(mat) => {
            let where_to_put_background = ui.painter().add(Shape::Noop);
            ui.vertical(|ui| {
                ui.add_space(CARD_UI_SPACING);
                ui.horizontal(|ui| {
                    ui.add_space(CARD_UI_SPACING);
                    ui.label("Create Material");
                    ui.label(format!("{:?}", mat));
                    ui.add_space(CARD_UI_SPACING);
                });
                ui.add_space(CARD_UI_SPACING);
            });

            let color = Color32::BLUE;
            ui.painter().set(
                where_to_put_background,
                epaint::RectShape::new(
                    ui.min_rect(),
                    CARD_UI_ROUNDING,
                    darken(color, 0.25),
                    Stroke::new(1.0, color),
                ),
            );
        }
        BaseCard::Effect(effect) => {
            let where_to_put_background = ui.painter().add(Shape::Noop);
            ui.vertical(|ui| {
                ui.add_space(CARD_UI_SPACING);
                ui.horizontal(|ui| {
                    ui.add_space(CARD_UI_SPACING);
                    ui.label("Apply Effect");
                    match effect {
                        Effect::Damage(v) => add_basic_modifer(ui, "Damage", *v, modify_path, path),
                        Effect::Knockback(v) => {
                            add_basic_modifer(ui, "Knockback", *v, modify_path, path)
                        }
                        Effect::Cleanse => add_basic_modifer(ui, "Cleanse", "", modify_path, path),
                        Effect::Teleport => {
                            add_basic_modifer(ui, "Teleport", "", modify_path, path)
                        }
                        Effect::StatusEffect(e, t) => {
                            if let StatusEffect::OnHit(base_card) = e {
                                add_basic_modifer(ui, "On Hit", *t, modify_path, path);
                                path.push_back(0);
                                draw_base_card(
                                    ui,
                                    base_card,
                                    path,
                                    source_path,
                                    dest_path,
                                    modify_path,
                                );
                                path.pop_back();
                            } else {
                                let effect_name = match e {
                                    StatusEffect::DamageOverTime => "Damage Over Time",
                                    StatusEffect::HealOverTime => "Heal Over Time",
                                    StatusEffect::DecreaceDamageTaken => "Decreace Damage Taken",
                                    StatusEffect::IncreaceDamageTaken => "Increace Damage Taken",
                                    StatusEffect::Slow => "Slow",
                                    StatusEffect::Speed => "Speed Up",
                                    StatusEffect::DecreaceGravity => "Decreace Gravity",
                                    StatusEffect::IncreaceGravity => "Increace Gravity",
                                    StatusEffect::Overheal => "Overheal",
                                    StatusEffect::Invincibility => "Invincibility",
                                    StatusEffect::Trapped => "Trapped",
                                    StatusEffect::Lockout => "Lockout",
                                    StatusEffect::OnHit(_base_card) => {
                                        panic!("OnHit should be handled above")
                                    }
                                };
                                add_basic_modifer(ui, effect_name, *t, modify_path, path)
                            }
                        }
                    }
                    ui.add_space(CARD_UI_SPACING);
                });
                ui.add_space(CARD_UI_SPACING);
            });

            let color = Color32::RED;
            ui.painter().set(
                where_to_put_background,
                epaint::RectShape::new(
                    ui.min_rect(),
                    CARD_UI_ROUNDING,
                    darken(color, 0.25),
                    Stroke::new(1.0, color),
                ),
            );
        }
        BaseCard::Trigger(id) => {
            let where_to_put_background = ui.painter().add(Shape::Noop);
            ui.vertical(|ui| {
                ui.add_space(CARD_UI_SPACING);
                ui.horizontal(|ui| {
                    ui.add_space(CARD_UI_SPACING);
                    add_basic_modifer(ui, "Trigger", *id, modify_path, path);
                    ui.add_space(CARD_UI_SPACING);
                });
                ui.add_space(CARD_UI_SPACING);
            });

            let color = Color32::from_rgb(0, 255, 255);
            ui.painter().set(
                where_to_put_background,
                epaint::RectShape::new(
                    ui.min_rect(),
                    CARD_UI_ROUNDING,
                    darken(color, 0.25),
                    Stroke::new(1.0, color),
                ),
            );
        }
        BaseCard::None => {
            ui.visuals_mut().widgets.inactive.bg_stroke = Stroke::new(1.0, Color32::GREEN);
            ui.visuals_mut().widgets.inactive.bg_fill =
                darken(ui.visuals_mut().widgets.inactive.bg_stroke.color, 0.25);
            let response = drop_target(ui, can_accept_what_is_being_dragged, |ui| {
                ui.horizontal(|ui| {
                    ui.add_space(CARD_UI_SPACING);
                    ui.label("None");
                    ui.add_space(CARD_UI_SPACING);
                });
                ui.add_space(CARD_UI_SPACING);
            })
            .response;

            if dest_path.is_none() {
                let is_being_dragged = ui.memory(|mem| mem.is_anything_being_dragged());
                if is_being_dragged && can_accept_what_is_being_dragged && response.hovered() {
                    *dest_path = Some((path.clone(), DropableType::BaseNone));
                }
            }
        }
        BaseCard::Palette(palette_cards, is_vertical) => {
            ui.visuals_mut().widgets.inactive.bg_stroke = Stroke::new(1.0, Color32::GRAY);
            ui.visuals_mut().widgets.inactive.bg_fill = Color32::BLACK;
            let response = drop_target(ui, can_accept_what_is_being_dragged, |ui| {
                ui.set_min_size(vec2(200.0, 40.0));
                ui.add_space(CARD_UI_SPACING);
                let layout = if *is_vertical {
                    egui::Layout::top_down(egui::Align::LEFT)
                } else {
                    egui::Layout::left_to_right(egui::Align::LEFT).with_main_wrap(true)
                };
                ui.with_layout(layout, |ui| {
                    for (card_idx, card) in palette_cards.iter().enumerate() {
                        path.push_back(card_idx as u32);
                        card.draw_draggable(ui, path, source_path, dest_path, modify_path);
                        path.pop_back();
                    }
                });

                for (card_idx, card) in palette_cards.iter().enumerate() {
                    path.push_back(card_idx as u32);
                    let item_id = egui::Id::new(id_source).with(path.clone());
                    if source_path.is_none() && ui.memory(|mem| mem.is_being_dragged(item_id)) {
                        *source_path = Some((path.clone(), card.get_type()));
                    }
                    path.pop_back();
                }
                ui.add_space(CARD_UI_SPACING);
            })
            .response;

            if dest_path.is_none() {
                let is_being_dragged = ui.memory(|mem| mem.is_anything_being_dragged());
                if is_being_dragged && can_accept_what_is_being_dragged && response.hovered() {
                    *dest_path = Some((path.clone(), DropableType::Palette));
                }
            }
        }
    });
    if is_draggable && source_path.is_none() && ui.memory(|mem| mem.is_being_dragged(item_id)) {
        *source_path = Some((path.clone(), DragableType::BaseCard));
    }
}

pub fn add_basic_modifer(
    ui: &mut Ui,
    name: &str,
    count: impl std::fmt::Display,
    modify_path: &mut Option<(VecDeque<u32>, ModificationType)>,
    path: &mut VecDeque<u32>,
) {
    let mut job = LayoutJob::default();
    job.append(
        name,
        0.0,
        TextFormat {
            color: Color32::WHITE,
            ..Default::default()
        },
    );
    job.append(
        format!("{}", count).as_str(),
        0.0,
        TextFormat {
            font_id: FontId::proportional(7.0),
            valign: Align::TOP,
            color: Color32::WHITE,
            ..Default::default()
        },
    );
    let widget = if ui.input(|i| i.modifiers.ctrl) {
        Label::new(job).sense(Sense::click())
    } else {
        Label::new(job)
    };
    let response = ui.add(widget);

    if response.clicked() {
        if modify_path.is_none() {
            let modification_type = if ui.input(|i| i.modifiers.shift) {
                ModificationType::Remove
            } else {
                ModificationType::Add
            };
            *modify_path = Some((path.clone(), modification_type));
        }
    }
}

pub fn add_hoverable_basic_modifer(
    ui: &mut Ui,
    id: Id,
    name: &str,
    count: impl std::fmt::Display,
    hover_text: String,
    modify_path: &mut Option<(VecDeque<u32>, ModificationType)>,
    path: &mut VecDeque<u32>,
) {
    ui.style_mut().wrap = Some(false);
    let mut job = LayoutJob::default();
    job.append(
        name,
        0.0,
        TextFormat {
            color: Color32::WHITE,
            ..Default::default()
        },
    );
    job.append(
        format!("{}", count).as_str(),
        0.0,
        TextFormat {
            font_id: FontId::proportional(7.0),
            valign: Align::TOP,
            color: Color32::WHITE,
            ..Default::default()
        },
    );
    let is_being_dragged = ui.memory(|mem| mem.is_being_dragged(id));

    if !is_being_dragged {
        //load from previous frame
        let prev_frame_area: Option<Rect> = ui.data(|d| d.get_temp(id));
        if let Some(area) = prev_frame_area {
            // Check for drags:
            let response = ui.interact(area, id, Sense::drag());
            if response.hovered() {
                ui.ctx().set_cursor_icon(CursorIcon::Grab);
            }
        }
        let widget = if ui.input(|i| i.modifiers.ctrl) {
            Label::new(job).sense(Sense::click())
        } else {
            Label::new(job)
        };
        let response = ui.add(widget).on_hover_text(hover_text);

        if response.clicked() {
            if modify_path.is_none() {
                let modification_type = if ui.input(|i| i.modifiers.shift) {
                    ModificationType::Remove
                } else {
                    ModificationType::Add
                };
                *modify_path = Some((path.clone(), modification_type));
            }
        }
        //store for next frame
        ui.data_mut(|d| d.insert_temp(id, response.rect));
    } else {
        ui.ctx().set_cursor_icon(CursorIcon::Grabbing);

        // Paint the body to a new layer:
        let layer_id = LayerId::new(Order::Tooltip, id);
        let response = ui
            .with_layer_id(layer_id, |ui| {
                ui.add(Label::new(job)).on_hover_text(hover_text);
            })
            .response;

        // Now we move the visuals of the body to where the mouse is.
        // Normally you need to decide a location for a widget first,
        // because otherwise that widget cannot interact with the mouse.
        // However, a dragged component cannot be interacted with anyway
        // (anything with `Order::Tooltip` always gets an empty [`Response`])
        // So this is fine!

        if let Some(pointer_pos) = ui.ctx().pointer_interact_pos() {
            let delta = pointer_pos - response.rect.center();
            ui.ctx().translate_layer(layer_id, delta);
        }
    }
}

pub fn card_editor(ctx: &egui::Context, gui_state: &mut GuiState) {
    egui::Area::new("card editor")
        .anchor(Align2::LEFT_TOP, Vec2::new(0.0, 0.0))
        .show(&ctx, |ui| {
            ui.painter().rect_filled(
                ui.available_rect_before_wrap(),
                0.0,
                Color32::BLACK.gamma_multiply(0.5),
            );

            let menu_size = Rect::from_center_size(
                ui.available_rect_before_wrap().center(),
                ui.available_rect_before_wrap().size() * egui::vec2(0.75, 0.75),
            );

            ui.allocate_ui_at_rect(menu_size, |ui| {
                ui.painter()
                    .rect_filled(ui.available_rect_before_wrap(), 0.0, Color32::BLACK);
                ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .scroll_bar_visibility(
                        egui::scroll_area::ScrollBarVisibility::VisibleWhenNeeded,
                    )
                    .drag_to_scroll(false)
                    .show(ui, |ui| {
                        ui.with_layout(egui::Layout::top_down(egui::Align::LEFT), |ui| {
                            ui.set_clip_rect(menu_size);

                            ui.horizontal_wrapped(|ui| {
                                ui.label(RichText::new("Card Editor").color(Color32::WHITE));
                                if ui.button("Export to Clipboard").clicked() {
                                    let export = ron::to_string(&gui_state.gui_cards).unwrap();
                                    ui.output_mut(|o| o.copied_text = export);
                                }

                                if ui.button("Import from Clipboard").clicked() {
                                    let mut clipboard = clippers::Clipboard::get();
                                    let import: Option<Vec<Cooldown>> = match clipboard.read() {
                                        Some(clippers::ClipperData::Text(text)) => {
                                            let clipboard_parse = ron::from_str(text.as_str());
                                            if let Err(e) = &clipboard_parse {
                                                println!("Failed to parse clipboard: {}", e);
                                            }
                                            clipboard_parse.ok()
                                        }
                                        _ => {
                                            println!("Failed to import from clipboard");
                                            None
                                        }
                                    };
                                    if let Some(import) = import {
                                        gui_state.gui_cards = import;
                                    }
                                }

                                if ui.button("Clear Dock").clicked() {
                                    gui_state.dock_cards = vec![];
                                }
                            });

                            ui.horizontal_wrapped(|ui| {
                                ui.selectable_value(
                                    &mut gui_state.palette_state,
                                    PaletteState::ProjectileModifiers,
                                    "Projectile Modifiers",
                                );
                                ui.selectable_value(
                                    &mut gui_state.palette_state,
                                    PaletteState::BaseCards,
                                    "Base Cards",
                                );
                                ui.selectable_value(
                                    &mut gui_state.palette_state,
                                    PaletteState::AdvancedProjectileModifiers,
                                    "Advanced Projectile Modifiers",
                                );
                                ui.selectable_value(
                                    &mut gui_state.palette_state,
                                    PaletteState::MultiCastModifiers,
                                    "Multicast Modifiers",
                                );
                                ui.selectable_value(
                                    &mut gui_state.palette_state,
                                    PaletteState::CooldownModifiers,
                                    "Cooldown Modifiers",
                                );
                                ui.selectable_value(
                                    &mut gui_state.palette_state,
                                    PaletteState::Effects,
                                    "Effects",
                                );
                                ui.selectable_value(
                                    &mut gui_state.palette_state,
                                    PaletteState::StatusEffects,
                                    "Status Effects",
                                );
                                ui.selectable_value(
                                    &mut gui_state.palette_state,
                                    PaletteState::Materials,
                                    "Materials",
                                );
                                ui.selectable_value(
                                    &mut gui_state.palette_state,
                                    PaletteState::Dock,
                                    "Dock",
                                );
                            });

                            let mut source_path = None;
                            let mut drop_path = None;
                            let mut modify_path = None;
                            let mut palette_card = BaseCard::Palette(
                                match gui_state.palette_state {
                                    PaletteState::ProjectileModifiers => vec![
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::SimpleModify(
                                                ProjectileModifierType::Gravity,
                                                -1,
                                            ),
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::SimpleModify(
                                                ProjectileModifierType::Gravity,
                                                1,
                                            ),
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::SimpleModify(
                                                ProjectileModifierType::Health,
                                                -1,
                                            ),
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::SimpleModify(
                                                ProjectileModifierType::Health,
                                                1,
                                            ),
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::SimpleModify(
                                                ProjectileModifierType::Length,
                                                -1,
                                            ),
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::SimpleModify(
                                                ProjectileModifierType::Length,
                                                1,
                                            ),
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::SimpleModify(
                                                ProjectileModifierType::Width,
                                                -1,
                                            ),
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::SimpleModify(
                                                ProjectileModifierType::Width,
                                                1,
                                            ),
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::SimpleModify(
                                                ProjectileModifierType::Height,
                                                -1,
                                            ),
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::SimpleModify(
                                                ProjectileModifierType::Height,
                                                1,
                                            ),
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::SimpleModify(
                                                ProjectileModifierType::Speed,
                                                -1,
                                            ),
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::SimpleModify(
                                                ProjectileModifierType::Speed,
                                                1,
                                            ),
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::SimpleModify(
                                                ProjectileModifierType::Lifetime,
                                                -1,
                                            ),
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::SimpleModify(
                                                ProjectileModifierType::Lifetime,
                                                1,
                                            ),
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::NoEnemyFire,
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::FriendlyFire,
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::LockToOwner,
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::PiercePlayers,
                                        ),
                                    ],
                                    PaletteState::BaseCards => vec![
                                        DraggableCard::BaseCard(BaseCard::Projectile(vec![])),
                                        DraggableCard::BaseCard(BaseCard::MultiCast(
                                            vec![],
                                            vec![],
                                        )),
                                        DraggableCard::BaseCard(BaseCard::Trigger(0)),
                                    ],
                                    PaletteState::AdvancedProjectileModifiers => vec![
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::OnHit(BaseCard::None),
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::OnHeadshot(BaseCard::None),
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::OnExpiry(BaseCard::None),
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::OnTrigger(0, BaseCard::None),
                                        ),
                                        DraggableCard::ProjectileModifier(
                                            ProjectileModifier::Trail(1, BaseCard::None),
                                        ),
                                    ],
                                    PaletteState::MultiCastModifiers => vec![
                                        DraggableCard::MultiCastModifier(
                                            MultiCastModifier::Spread(1),
                                        ),
                                        DraggableCard::MultiCastModifier(
                                            MultiCastModifier::Duplication(1),
                                        ),
                                    ],
                                    PaletteState::CooldownModifiers => vec![
                                        DraggableCard::CooldownModifier(
                                            CooldownModifier::SimpleCooldownModifier(
                                                SimpleCooldownModifier::AddCharge,
                                                1,
                                            ),
                                        ),
                                        DraggableCard::CooldownModifier(
                                            CooldownModifier::SimpleCooldownModifier(
                                                SimpleCooldownModifier::AddCooldown,
                                                1,
                                            ),
                                        ),
                                        DraggableCard::CooldownModifier(
                                            CooldownModifier::SimpleCooldownModifier(
                                                SimpleCooldownModifier::MultiplyImpact,
                                                1,
                                            ),
                                        ),
                                    ],
                                    PaletteState::Effects => vec![
                                        DraggableCard::BaseCard(BaseCard::Effect(Effect::Damage(
                                            1,
                                        ))),
                                        DraggableCard::BaseCard(BaseCard::Effect(Effect::Damage(
                                            -1,
                                        ))),
                                        DraggableCard::BaseCard(BaseCard::Effect(
                                            Effect::Knockback(1),
                                        )),
                                        DraggableCard::BaseCard(BaseCard::Effect(
                                            Effect::Knockback(-1),
                                        )),
                                        DraggableCard::BaseCard(BaseCard::Effect(Effect::Cleanse)),
                                        DraggableCard::BaseCard(BaseCard::Effect(Effect::Teleport)),
                                    ],
                                    PaletteState::StatusEffects => vec![
                                        DraggableCard::BaseCard(BaseCard::Effect(
                                            Effect::StatusEffect(StatusEffect::DamageOverTime, 1),
                                        )),
                                        DraggableCard::BaseCard(BaseCard::Effect(
                                            Effect::StatusEffect(StatusEffect::HealOverTime, 1),
                                        )),
                                        DraggableCard::BaseCard(BaseCard::Effect(
                                            Effect::StatusEffect(
                                                StatusEffect::DecreaceDamageTaken,
                                                1,
                                            ),
                                        )),
                                        DraggableCard::BaseCard(BaseCard::Effect(
                                            Effect::StatusEffect(
                                                StatusEffect::IncreaceDamageTaken,
                                                1,
                                            ),
                                        )),
                                        DraggableCard::BaseCard(BaseCard::Effect(
                                            Effect::StatusEffect(StatusEffect::DecreaceGravity, 1),
                                        )),
                                        DraggableCard::BaseCard(BaseCard::Effect(
                                            Effect::StatusEffect(StatusEffect::IncreaceGravity, 1),
                                        )),
                                        DraggableCard::BaseCard(BaseCard::Effect(
                                            Effect::StatusEffect(StatusEffect::Speed, 1),
                                        )),
                                        DraggableCard::BaseCard(BaseCard::Effect(
                                            Effect::StatusEffect(StatusEffect::Slow, 1),
                                        )),
                                        DraggableCard::BaseCard(BaseCard::Effect(
                                            Effect::StatusEffect(StatusEffect::Overheal, 1),
                                        )),
                                        DraggableCard::BaseCard(BaseCard::Effect(
                                            Effect::StatusEffect(StatusEffect::Invincibility, 1),
                                        )),
                                        DraggableCard::BaseCard(BaseCard::Effect(
                                            Effect::StatusEffect(
                                                StatusEffect::OnHit(Box::new(BaseCard::None)),
                                                1,
                                            ),
                                        )),
                                    ],
                                    PaletteState::Materials => vec![
                                        DraggableCard::BaseCard(BaseCard::CreateMaterial(
                                            VoxelMaterial::Grass,
                                        )),
                                        DraggableCard::BaseCard(BaseCard::CreateMaterial(
                                            VoxelMaterial::Dirt,
                                        )),
                                        DraggableCard::BaseCard(BaseCard::CreateMaterial(
                                            VoxelMaterial::Stone,
                                        )),
                                        DraggableCard::BaseCard(BaseCard::CreateMaterial(
                                            VoxelMaterial::Ice,
                                        )),
                                        DraggableCard::BaseCard(BaseCard::CreateMaterial(
                                            VoxelMaterial::Water,
                                        )),
                                    ],
                                    PaletteState::Dock => gui_state.dock_cards.clone(),
                                },
                                match gui_state.palette_state {
                                    PaletteState::ProjectileModifiers => false,
                                    PaletteState::BaseCards => false,
                                    PaletteState::AdvancedProjectileModifiers => true,
                                    PaletteState::MultiCastModifiers => false,
                                    PaletteState::CooldownModifiers => false,
                                    PaletteState::Effects => true,
                                    PaletteState::StatusEffects => true,
                                    PaletteState::Materials => true,
                                    PaletteState::Dock => true,
                                },
                            );

                            ui.scope(|ui| {
                                ui.visuals_mut().override_text_color = Some(Color32::WHITE);
                                draw_base_card(
                                    ui,
                                    &palette_card,
                                    &mut vec![0].into(),
                                    &mut source_path,
                                    &mut drop_path,
                                    &mut modify_path,
                                );
                            });

                            let total_impact = gui_state
                                .gui_cards
                                .iter()
                                .map(|card| card.get_impact_multiplier())
                                .sum::<f32>();

                            if gui_state.cooldown_cache_refresh_delay <= 0.0 {
                                for cooldown in gui_state.gui_cards.iter_mut() {
                                    if cooldown.generate_cooldown_cache() {
                                        gui_state.cooldown_cache_refresh_delay = 0.5;
                                    }
                                }
                            }

                            for (ability_idx, mut cooldown) in
                                gui_state.gui_cards.iter_mut().enumerate()
                            {
                                ui.horizontal_top(|ui| {
                                    draw_cooldown(
                                        ui,
                                        &mut cooldown,
                                        &mut vec![ability_idx as u32 + 1].into(),
                                        &mut source_path,
                                        &mut drop_path,
                                        &mut modify_path,
                                        total_impact,
                                    );
                                });
                            }

                            if ui
                                .button("Add Cooldown")
                                .on_hover_text("Add a new cooldown")
                                .clicked()
                            {
                                gui_state.gui_cards.push(Cooldown::empty());
                            }

                            if let Some((modify_path, modify_type)) = modify_path.as_mut() {
                                let modify_action_idx = modify_path.pop_front().unwrap() as usize;
                                if modify_path.is_empty() {
                                    assert!(matches!(modify_type, ModificationType::Remove));
                                    gui_state.gui_cards.remove(modify_action_idx - 1);
                                } else {
                                    gui_state.gui_cards[modify_action_idx - 1]
                                        .modify_from_path(&mut modify_path.clone(), modify_type);
                                }
                            }
                            if let Some((source_path, source_type)) = source_path.as_mut() {
                                if let Some((drop_path, drop_type)) = drop_path.as_mut() {
                                    let source_action_idx =
                                        source_path.pop_front().unwrap() as usize;
                                    let drop_action_idx = drop_path.pop_front().unwrap() as usize;
                                    if ui.input(|i| i.pointer.any_released())
                                        && (is_valid_drag(source_type, drop_type)
                                            || drop_action_idx == 0)
                                    {
                                        // do the drop:
                                        let item = if source_action_idx == 0 {
                                            palette_card.take_from_path(source_path)
                                        } else {
                                            gui_state.gui_cards[source_action_idx - 1]
                                                .take_from_path(&mut source_path.clone())
                                        };
                                        if drop_action_idx > 0 {
                                            gui_state.gui_cards[drop_action_idx - 1]
                                                .insert_to_path(drop_path, item);
                                        } else if matches!(
                                            gui_state.palette_state,
                                            PaletteState::Dock
                                        ) {
                                            gui_state.dock_cards.push(item);
                                        }
                                        if source_action_idx > 0 {
                                            gui_state.gui_cards[source_action_idx - 1]
                                                .cleanup(source_path);
                                        }
                                    }
                                }
                            }
                        });
                    });
            });
        });
}

pub fn healthbar(corner_offset: f32, ctx: &egui::Context, spectate_player: &Entity) {
    egui::Area::new("healthbar")
        .anchor(
            Align2::LEFT_BOTTOM,
            Vec2::new(corner_offset, -corner_offset),
        )
        .show(ctx, |ui| {
            let thickness = 1.0;
            let color = Color32::from_additive_luminance(255);
            let (player_health, player_max_health) = spectate_player.get_health_stats();

            for AppliedStatusEffect { effect, time_left } in spectate_player.status_effects.iter() {
                let effect_name = match effect {
                    ReferencedStatusEffect::DamageOverTime => "Damage Over Time",
                    ReferencedStatusEffect::HealOverTime => "Heal Over Time",
                    ReferencedStatusEffect::Slow => "Slow",
                    ReferencedStatusEffect::Speed => "Speed",
                    ReferencedStatusEffect::DecreaceDamageTaken => "Decreace Damage Taken",
                    ReferencedStatusEffect::IncreaceDamageTaken => "Increase Damage Taken",
                    ReferencedStatusEffect::DecreaceGravity => "Decreace Gravity",
                    ReferencedStatusEffect::IncreaceGravity => "Increase Gravity",
                    ReferencedStatusEffect::Invincibility => "Invincibility",
                    ReferencedStatusEffect::Overheal => "Overheal",
                    ReferencedStatusEffect::OnHit(_) => "On Player Hit",
                    ReferencedStatusEffect::Trapped => "Trapped",
                    ReferencedStatusEffect::Lockout => "Lockout",
                };
                ui.label(
                    RichText::new(format!("{}: {:.1}s", effect_name, time_left))
                        .color(Color32::WHITE),
                );
            }

            ui.label(
                RichText::new(format!("{} / {}", player_health, player_max_health))
                    .color(Color32::WHITE),
            );
            let desired_size = egui::vec2(200.0, 30.0);
            let (_id, rect) = ui.allocate_space(desired_size);

            let to_screen =
                emath::RectTransform::from_to(Rect::from_x_y_ranges(0.0..=1.0, 0.0..=1.0), rect);

            let healthbar_size =
                Rect::from_min_max(to_screen * pos2(0.0, 0.0), to_screen * pos2(1.0, 1.0));
            let mut health_rendered = 0.0;
            for health_section in spectate_player.health.iter() {
                let (health_size, health_color) = match health_section {
                    HealthSection::Health(current, _max) => {
                        let prev_health_rendered = health_rendered;
                        health_rendered += current;
                        (
                            Rect::from_min_max(
                                to_screen * pos2(prev_health_rendered / player_max_health, 0.0),
                                to_screen * pos2(health_rendered / player_max_health, 1.0),
                            ),
                            Color32::WHITE,
                        )
                    }
                    HealthSection::Overhealth(current, _time) => {
                        let prev_health_rendered = health_rendered;
                        health_rendered += current;
                        (
                            Rect::from_min_max(
                                to_screen * pos2(prev_health_rendered / player_max_health, 0.0),
                                to_screen * pos2(health_rendered / player_max_health, 1.0),
                            ),
                            Color32::GREEN,
                        )
                    }
                };
                ui.painter().add(epaint::Shape::rect_filled(
                    health_size,
                    Rounding::ZERO,
                    health_color,
                ));
            }

            ui.painter().add(epaint::Shape::rect_stroke(
                healthbar_size,
                Rounding::ZERO,
                Stroke::new(thickness, color),
            ));
        });
}
