use std::{collections::VecDeque, ops::{Add, Mul, Sub}};

use egui_winit_vulkano::egui::{
    self, emath::{self, Numeric}, epaint::{self, PathShape}, pos2, text::LayoutJob, vec2, Align2, Color32, CursorIcon, DragValue, FontId, Id, InnerResponse, Label, LayerId, Order, Pos2, Rect, Rgba, RichText, Rounding, ScrollArea, Sense, Shape, Stroke, TextFormat, TextStyle, Ui, Vec2
};
use itertools::Itertools;

use crate::{
    card_system::{
        Ability, BaseCard, Cooldown, CooldownModifier, Deck, DirectionCard, DragableCard, Effect,
        Keybind, MultiCastModifier, PassiveCard, ProjectileModifier, ReferencedStatusEffect,
        SignedSimpleCooldownModifier, SimpleCooldownModifier, SimpleProjectileModifierType,
        SimpleStatusEffectType, StatusEffect, VoxelMaterial,
    },
    lobby_browser::LobbyBrowser,
    rollback_manager::{AppliedStatusEffect, Entity, HealthSection, PlayerAbility},
    settings_manager::Control,
    utils::{translate_egui_key_code, translate_egui_pointer_button},
};

const ID_SOURCE: &str = "card_editor";

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
    StatusEffects,
    Directions,
    Dock,
}

pub struct GuiState {
    pub menu_stack: Vec<GuiElement>,
    pub errors: Vec<String>,
    pub gui_deck: Deck,
    pub dock_cards: Vec<DragableCard>,
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

const TAU: f32 = std::f32::consts::TAU;
pub fn lerp<T>(start: T, end: T, t: f32) -> T
where
    T: Add<T, Output = T> + Sub<T, Output = T> + Mul<f32, Output = T> + Copy,
{
    (end - start) * t.clamp(0.0, 1.0) + start
}

fn get_arc_points(
    start: f32,
    center: Pos2,
    radius: f32,
    value: f32,
    max_arc_distance: f32,
) -> Vec<Pos2> {
    let start_turns: f32 = start;
    let end_turns = start_turns + value;

    let points = (value.abs() / max_arc_distance).ceil() as usize;
    let points = points.max(1);
    (0..=points)
        .map(|i| {
            let t = i as f32 / (points - 1) as f32;
            let angle = lerp(start_turns * TAU, end_turns * TAU, t);
            let x = radius * angle.cos();
            let y = -radius * angle.sin();
            pos2(x, y) + center.to_vec2()
        })
        .collect()
}

fn get_arc_shape(
    start: f32,
    center: Pos2,
    radius: f32,
    value: f32,
    max_arc_distance: f32,
    stroke: Stroke,
) -> Shape {
    Shape::Path(PathShape {
        points: get_arc_points(
            start,
            center,
            radius,
            value,
            0.03,
        ),
        closed: false,
        fill: Color32::TRANSPARENT,
        stroke,
    })
}

fn cooldown_ui(ui: &mut egui::Ui, ability: &PlayerAbility, ability_idx: usize) -> egui::Response {
    let desired_size = ui.spacing().interact_size.y * egui::vec2(3.0, 3.0);
    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());
    let recovery_bar_rect = rect.with_max_y(rect.min.y + 10.0).with_max_x(rect.min.x + rect.width() * ability.recovery / ability.value.1[ability_idx]);

    if ui.is_rect_visible(rect) {
        let font = egui::FontId::proportional(24.0);
        if ability.cooldown > ability.ability.add_charge as f32 * ability.value.0 {
            ui.painter().rect_filled(rect, 5.0, Color32::DARK_GRAY);
            if ability.recovery > 0.0 {
                ui.painter().rect_filled(recovery_bar_rect, 5.0, Color32::GREEN);
            }
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
        ui.painter().rect_filled(rect, 5.0, Color32::LIGHT_GRAY);
        if ability.recovery > 0.0 {
            ui.painter().rect_filled(recovery_bar_rect, 5.0, Color32::GREEN);
        }
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
            let to_next_charge = 1.0 - (ability.cooldown / ability.value.0) % 1.0;
            ui.painter()
                .circle_filled(rect.right_top(), 8.0, Color32::GRAY);
            ui.painter().add(get_arc_shape(0.0, rect.right_top(), 8.0, to_next_charge, 0.03, Stroke::new(1.0, Color32::BLACK)));
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
        let mut size = vec2(0.0, 0.0);
        if let Some(area) = prev_frame_area {
            if dragable {
                // Check for drags:
                let response = ui.interact(area, id, Sense::drag());
                if response.hovered() {
                    ui.ctx().set_cursor_icon(CursorIcon::Grab);
                }
            }
            size.x = area.size().x;
        }
        if ui.available_size_before_wrap().x < size.x {
            ui.end_row();
        }
        let response = ui.scope(body).response;
        //store for next frame
        ui.data_mut(|d| d.insert_temp(id, response.rect.shrink(CARD_UI_SPACING)));
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
    StatusEffect,
    BaseCard,
    Direction,
}

#[derive(Debug)]
pub enum DropableType {
    MultiCastBaseCard,
    BaseNone,
    BaseProjectile,
    BaseStatusEffects,
    Cooldown,
    Direction,
    Palette,
}

#[derive(Debug)]
pub enum ModificationType {
    Add,
    Remove,
    Other,
}

pub fn is_valid_drag(from: &DragableType, to: &DropableType) -> bool {
    match (from, to) {
        (DragableType::ProjectileModifier, DropableType::BaseProjectile) => true,
        (DragableType::StatusEffect, DropableType::BaseStatusEffects) => true,
        (DragableType::MultiCastModifier, DropableType::MultiCastBaseCard) => true,
        (DragableType::BaseCard, DropableType::MultiCastBaseCard) => true,
        (DragableType::BaseCard, DropableType::BaseNone) => true,
        (DragableType::CooldownModifier, DropableType::Cooldown) => true,
        (DragableType::BaseCard, DropableType::Cooldown) => true,
        (DragableType::Direction, DropableType::Direction) => true,
        (_, DropableType::Palette) => true,
        _ => false,
    }
}

impl DrawableCard for Cooldown {
    fn draw(
        &mut self,
        ui: &mut Ui,
        path: &mut VecDeque<u32>,
        source_path: &mut Option<(VecDeque<u32>, DragableType)>,
        drop_path: &mut Option<(VecDeque<u32>, DropableType)>,
        modify_path: &mut Option<(VecDeque<u32>, ModificationType)>,
    ) {
        let can_accept_what_is_being_dragged = true;
        let Cooldown {
            abilities,
            modifiers,
            cooldown_value,
        } = self;
        ui.visuals_mut().widgets.active.rounding = Rounding::from(CARD_UI_ROUNDING);
        ui.visuals_mut().widgets.inactive.rounding = Rounding::from(CARD_UI_ROUNDING);
        ui.visuals_mut().override_text_color = Some(Color32::WHITE);
        ui.visuals_mut().widgets.inactive.bg_stroke =
            Stroke::new(0.5, Color32::from_rgb(255, 0, 255));
        let response = drop_target(ui, can_accept_what_is_being_dragged, |ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.add_space(CARD_UI_SPACING);
                    ui.label("Cooldown");
                    if let Some(cooldown_value) = cooldown_value {
                        ui.label(format!("{:.2}s", cooldown_value.0))
                            .on_hover_text(format!(
                                "Recoveries: {}",
                                cooldown_value
                                    .1
                                    .iter()
                                    .map(|v| format!("{:.2}s", v))
                                    .join(", ")
                            ));
                    }
                    path.push_back(0);
                    for (mod_idx, modifier) in modifiers.iter_mut().enumerate() {
                        path.push_back(mod_idx as u32);
                        modifier.draw(ui, path, source_path, drop_path, modify_path);
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
                        ability
                            .card
                            .draw(ui, path, source_path, drop_path, modify_path);
                    });
                    path.pop_back();
                }
                path.pop_back();

                path.push_back(0);
                for (modifier_idx, _modifier) in modifiers.iter().enumerate() {
                    path.push_back(modifier_idx as u32);
                    let item_id = egui::Id::new(ID_SOURCE).with(path.clone());
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

    fn modify_from_path(&mut self, path: &mut VecDeque<u32>, modification_type: ModificationType) {
        self.cooldown_value = None;
        let type_idx = path.pop_front().unwrap() as usize;
        if type_idx == 0 {
            let idx = path.pop_front().unwrap() as usize;
            self.modifiers[idx].modify_from_path(path, modification_type);
        } else if type_idx == 1 {
            let idx = path.pop_front().unwrap() as usize;
            self.abilities[idx]
                .card
                .modify_from_path(path, modification_type);
            self.abilities[idx].invalidate_cooldown_cache();
        } else {
            panic!("Invalid state");
        }
    }

    fn take_from_path(&mut self, path: &mut VecDeque<u32>) -> DragableCard {
        self.cooldown_value = None;
        let type_idx = path.pop_front().unwrap() as usize;
        if type_idx == 0 {
            let idx = path.pop_front().unwrap() as usize;
            self.modifiers[idx].take_from_path(path)
        } else if type_idx == 1 {
            let idx = path.pop_front().unwrap() as usize;
            if path.is_empty() {
                let ability_card = self.abilities[idx].card.clone();
                self.abilities[idx].card = BaseCard::None;
                self.abilities[idx].invalidate_cooldown_cache();
                DragableCard::BaseCard(ability_card)
            } else {
                let result = self.abilities[idx].card.take_from_path(path);
                self.abilities[idx].invalidate_cooldown_cache();
                result
            }
        } else {
            panic!("Invalid state");
        }
    }

    fn insert_to_path(&mut self, path: &mut VecDeque<u32>, item: DragableCard) {
        self.cooldown_value = None;
        if path.is_empty() {
            if let DragableCard::BaseCard(item) = item {
                self.abilities.push(Ability {
                    card: item,
                    ..Default::default()
                });
            } else if let DragableCard::CooldownModifier(modifier_item) = item {
                let mut combined = false;
                match modifier_item.clone() {
                    CooldownModifier::SimpleCooldownModifier(last_type, last_s) => {
                        for modifier in self.modifiers.iter_mut() {
                            match modifier {
                                CooldownModifier::SimpleCooldownModifier(current_type, s)
                                    if *current_type == last_type =>
                                {
                                    *s += last_s;
                                    combined = true;
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                    CooldownModifier::SignedSimpleCooldownModifier(last_type, last_s) => {
                        for modifier in self.modifiers.iter_mut() {
                            match modifier {
                                CooldownModifier::SignedSimpleCooldownModifier(current_type, s)
                                    if *current_type == last_type =>
                                {
                                    *s += last_s;
                                    combined = true;
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }

                if !combined {
                    self.modifiers.push(modifier_item.clone());
                }
            } else {
                panic!("Invalid state")
            }
        } else {
            assert!(path.pop_front().unwrap() == 1);
            let idx = path.pop_front().unwrap() as usize;
            self.abilities[idx].card.insert_to_path(path, item);
            self.abilities[idx].invalidate_cooldown_cache();
        }
    }

    fn cleanup(&mut self, path: &mut VecDeque<u32>) {
        if path.is_empty() {
            return;
        }
        let idx_type = path.pop_front().unwrap();
        if idx_type == 0 {
            let idx = path.pop_front().unwrap() as usize;
            assert!(path.is_empty());
            match self.modifiers[idx] {
                CooldownModifier::None => {
                    self.modifiers.remove(idx);
                }
                CooldownModifier::SimpleCooldownModifier(_, s) => {
                    if s == 0 {
                        self.modifiers.remove(idx);
                    }
                }
                CooldownModifier::SignedSimpleCooldownModifier(_, s) => {
                    if s == 0 {
                        self.modifiers.remove(idx);
                    }
                }
                _ => {}
            }
        } else if idx_type == 1 {
            let idx = path.pop_front().unwrap() as usize;
            if path.is_empty() {
                if matches!(self.abilities[idx].card, BaseCard::None) && self.abilities.len() > 1 {
                    self.abilities.remove(idx);
                }
            } else {
                self.abilities[idx].card.cleanup(path);
                self.abilities[idx].invalidate_cooldown_cache();
            }
        } else {
            panic!("Invalid state");
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
trait DrawableCard {
    fn draw(
        &mut self,
        ui: &mut Ui,
        path: &mut VecDeque<u32>,
        source_path: &mut Option<(VecDeque<u32>, DragableType)>,
        dest_path: &mut Option<(VecDeque<u32>, DropableType)>,
        modify_path: &mut Option<(VecDeque<u32>, ModificationType)>,
    );

    fn modify_from_path(&mut self, path: &mut VecDeque<u32>, modification_type: ModificationType);
    fn take_from_path(&mut self, path: &mut VecDeque<u32>) -> DragableCard;
    fn insert_to_path(&mut self, path: &mut VecDeque<u32>, item: DragableCard);
    fn cleanup(&mut self, path: &mut VecDeque<u32>);
}

impl DrawableCard for CooldownModifier {
    fn draw(
        &mut self,
        ui: &mut Ui,
        path: &mut VecDeque<u32>,
        _source_path: &mut Option<(VecDeque<u32>, DragableType)>,
        _dest_path: &mut Option<(VecDeque<u32>, DropableType)>,
        modify_path: &mut Option<(VecDeque<u32>, ModificationType)>,
    ) {
        let item_id = egui::Id::new(ID_SOURCE).with(path.clone());
        let hover_text = self.get_hover_text();
        let name = self.get_name();
        match self {
            CooldownModifier::SimpleCooldownModifier(_, ref mut v) => draw_modifier(
                ui,
                item_id,
                name,
                Some(v),
                hover_text,
                true,
                modify_path,
                path,
            ),
            CooldownModifier::SignedSimpleCooldownModifier(_, ref mut v) => draw_modifier(
                ui,
                item_id,
                name,
                Some(v),
                hover_text,
                true,
                modify_path,
                path,
            ),
            CooldownModifier::None | CooldownModifier::Reloading => draw_modifier(
                ui,
                item_id,
                name,
                None::<&mut u32>,
                hover_text,
                true,
                modify_path,
                path,
            ),
        }
    }

    fn modify_from_path(&mut self, path: &mut VecDeque<u32>, modification_type: ModificationType) {
        assert!(path.is_empty());
        match self {
            CooldownModifier::SimpleCooldownModifier(_, v) => match modification_type {
                ModificationType::Add => *v += 1,
                ModificationType::Remove => {
                    if *v > 1 {
                        *v -= 1
                    }
                }
                ModificationType::Other => {}
            },
            CooldownModifier::SignedSimpleCooldownModifier(_, v) => match modification_type {
                ModificationType::Add => *v += 1,
                ModificationType::Remove => *v -= 1,
                ModificationType::Other => {}
            },
            _ => {}
        }
    }

    fn take_from_path(&mut self, path: &mut VecDeque<u32>) -> DragableCard {
        assert!(path.is_empty());
        let modifier = self.clone();
        *self = CooldownModifier::None;
        DragableCard::CooldownModifier(modifier)
    }

    fn insert_to_path(&mut self, _path: &mut VecDeque<u32>, _item: DragableCard) {}

    fn cleanup(&mut self, _path: &mut VecDeque<u32>) {}
}

impl DrawableCard for PassiveCard {
    fn draw(
        &mut self,
        ui: &mut Ui,
        path: &mut VecDeque<u32>,
        source_path: &mut Option<(VecDeque<u32>, DragableType)>,
        dest_path: &mut Option<(VecDeque<u32>, DropableType)>,
        modify_path: &mut Option<(VecDeque<u32>, ModificationType)>,
    ) {
        let can_accept_what_is_being_dragged = true; // We accept anything being dragged (for now) ¯\_(ツ)_/¯
        ui.vertical(|ui| {
            ui.visuals_mut().widgets.inactive.bg_stroke = Stroke::new(0.5, Color32::KHAKI);
            ui.visuals_mut().widgets.inactive.bg_fill =
                darken(ui.visuals_mut().widgets.inactive.bg_stroke.color, 0.25);

            let response = drop_target(ui, can_accept_what_is_being_dragged, |ui| {
                let mut advanced_effects = vec![];
                ui.horizontal(|ui| {
                    ui.add_space(CARD_UI_SPACING);
                    ui.vertical(|ui| {
                        ui.add_space(CARD_UI_SPACING);
                        ui.horizontal_wrapped(|ui| {
                            draw_label(
                                ui,
                                "Passive Status Effects",
                                format!("Apply status effects passively"),
                                modify_path,
                                path,
                            );
                            for (effect_idx, effect) in self.passive_effects.iter_mut().enumerate()
                            {
                                if effect.is_advanced() {
                                    advanced_effects.push((effect_idx, effect));
                                    continue;
                                }
                                path.push_back(effect_idx as u32);
                                effect.draw(ui, path, source_path, dest_path, modify_path);
                                path.pop_back();
                            }
                        });

                        for (modifier_idx, modifier) in advanced_effects.into_iter() {
                            path.push_back(modifier_idx as u32);
                            modifier.draw(ui, path, source_path, dest_path, modify_path);
                            path.pop_back();
                        }
                        ui.add_space(CARD_UI_SPACING);
                    });
                    ui.add_space(CARD_UI_SPACING);
                });

                for (modifier_idx, _modifier) in self.passive_effects.iter().enumerate() {
                    path.push_back(modifier_idx as u32);
                    let item_id = egui::Id::new(ID_SOURCE).with(path.clone());
                    if source_path.is_none() && ui.memory(|mem| mem.is_being_dragged(item_id)) {
                        *source_path = Some((path.clone(), DragableType::StatusEffect));
                    }
                    path.pop_back();
                }
            })
            .response;

            if dest_path.is_none() {
                let is_being_dragged = ui.memory(|mem| mem.is_anything_being_dragged());
                if is_being_dragged && can_accept_what_is_being_dragged && response.hovered() {
                    *dest_path = Some((path.clone(), DropableType::BaseStatusEffects));
                }
            }
        });
    }

    fn modify_from_path(&mut self, path: &mut VecDeque<u32>, modification_type: ModificationType) {
        if path.is_empty() {
        } else {
            let effect_idx = path.pop_front().unwrap() as usize;
            self.passive_effects[effect_idx].modify_from_path(path, modification_type);
        }
    }

    fn take_from_path(&mut self, path: &mut VecDeque<u32>) -> DragableCard {
        let Some(effect_idx) = path.pop_front() else {
            panic!("Invalid state: path is empty");
        };
        if path.is_empty() {
            let effect = self.passive_effects[effect_idx as usize].clone();
            self.passive_effects[effect_idx as usize] = StatusEffect::None;
            return DragableCard::StatusEffect(effect);
        }
        self.passive_effects
            .get_mut(effect_idx as usize)
            .unwrap()
            .take_from_path(path)
    }

    fn insert_to_path(&mut self, path: &mut VecDeque<u32>, item: DragableCard) {
        if path.is_empty() {
            let DragableCard::StatusEffect(item) = item else {
                panic!("Invalid state")
            };
            if let StatusEffect::SimpleStatusEffect(last_ty, last_s) = item.clone() {
                let mut combined = false;
                for effect in self.passive_effects.iter_mut() {
                    match effect {
                        StatusEffect::SimpleStatusEffect(ty, s) => {
                            if last_ty == *ty {
                                *s += last_s;
                                combined = true;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                if !combined {
                    self.passive_effects.push(item.clone());
                }
            } else {
                self.passive_effects.push(item.clone());
            }

            self.passive_effects.retain(|effect| match effect {
                StatusEffect::SimpleStatusEffect(_, s) => *s != 0,
                _ => true,
            });
        } else {
            let idx = path.pop_front().unwrap() as usize;
            assert!(path.pop_front().unwrap() == 0);
            self.passive_effects[idx].insert_to_path(path, item);
        }
    }

    fn cleanup(&mut self, path: &mut VecDeque<u32>) {
        if path.len() <= 1 {
            self.passive_effects.retain(|effect| match effect {
                StatusEffect::None => false,
                _ => true,
            });
        } else {
            let idx = path.pop_front().unwrap() as usize;
            assert!(path.pop_front().unwrap() == 0);
            match self.passive_effects[idx] {
                StatusEffect::OnHit(ref mut card) => card.cleanup(path),
                StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::IncreaseGravity(_), _) => {
                }
                ref invalid => panic!(
                    "Invalid state: cannot follow path {} into {:?}",
                    idx, invalid
                ),
            }
        }
    }
}

impl DrawableCard for MultiCastModifier {
    fn draw(
        &mut self,
        ui: &mut Ui,
        path: &mut VecDeque<u32>,
        _source_path: &mut Option<(VecDeque<u32>, DragableType)>,
        _dest_path: &mut Option<(VecDeque<u32>, DropableType)>,
        modify_path: &mut Option<(VecDeque<u32>, ModificationType)>,
    ) {
        let item_id = egui::Id::new(ID_SOURCE).with(path.clone());
        let hover_text = self.get_hover_text();
        let name = self.get_name();
        match self {
            MultiCastModifier::None => draw_label(ui, &name, hover_text, modify_path, path),
            MultiCastModifier::Spread(ref mut v) | MultiCastModifier::Duplication(ref mut v) => {
                draw_modifier(
                    ui,
                    item_id,
                    name,
                    Some(v),
                    hover_text,
                    true,
                    modify_path,
                    path,
                )
            }
        }
    }

    fn modify_from_path(&mut self, path: &mut VecDeque<u32>, modification_type: ModificationType) {
        assert!(path.is_empty());
        match self {
            MultiCastModifier::None => {}
            MultiCastModifier::Spread(ref mut value) => match modification_type {
                ModificationType::Add => *value += 1,
                ModificationType::Remove => {
                    if *value > 1 {
                        *value -= 1
                    }
                }
                ModificationType::Other => {}
            },
            MultiCastModifier::Duplication(ref mut value) => match modification_type {
                ModificationType::Add => *value += 1,
                ModificationType::Remove => {
                    if *value > 1 {
                        *value -= 1
                    }
                }
                ModificationType::Other => {}
            },
        }
    }

    fn take_from_path(&mut self, path: &mut VecDeque<u32>) -> DragableCard {
        assert!(path.is_empty());
        let multicast_modifier = self.clone();
        *self = MultiCastModifier::None;
        DragableCard::MultiCastModifier(multicast_modifier)
    }

    fn insert_to_path(&mut self, _path: &mut VecDeque<u32>, _item: DragableCard) {}

    fn cleanup(&mut self, _path: &mut VecDeque<u32>) {}
}

impl DrawableCard for ProjectileModifier {
    fn draw(
        &mut self,
        ui: &mut Ui,
        path: &mut VecDeque<u32>,
        source_path: &mut Option<(VecDeque<u32>, DragableType)>,
        dest_path: &mut Option<(VecDeque<u32>, DropableType)>,
        modify_path: &mut Option<(VecDeque<u32>, ModificationType)>,
    ) {
        let item_id = egui::Id::new(ID_SOURCE).with(path.clone());
        let mut advanced_modifier = false;
        let hover_text = self.get_hover_text();
        let name = self.get_name();
        match self {
            ProjectileModifier::SimpleModify(_, ref mut v) => draw_modifier(
                ui,
                item_id,
                name,
                Some(v),
                hover_text,
                true,
                modify_path,
                path,
            ),
            ProjectileModifier::NoEnemyFire
            | ProjectileModifier::FriendlyFire
            | ProjectileModifier::LockToOwner
            | ProjectileModifier::PiercePlayers
            | ProjectileModifier::WallBounce
            | ProjectileModifier::None => draw_modifier(
                ui,
                item_id,
                name,
                None::<&mut u32>,
                hover_text,
                true,
                modify_path,
                path,
            ),
            ref modifier if modifier.is_advanced() => {
                advanced_modifier = true;
            }
            _ => panic!("Invalid State"),
        }
        if advanced_modifier {
            let hover_text = self.get_hover_text();
            let name = self.get_name();
            ui.horizontal(|ui| {
                ui.add_space(CARD_UI_SPACING);
                match self {
                    ProjectileModifier::OnHit(ref mut base_card)
                    | ProjectileModifier::OnHeadshot(ref mut base_card)
                    | ProjectileModifier::OnExpiry(ref mut base_card) => {
                        drag_source(ui, item_id, true, |ui| {
                            draw_modifier(
                                ui,
                                item_id,
                                name,
                                None::<&mut u32>,
                                hover_text,
                                false,
                                modify_path,
                                path,
                            );
                            path.push_back(0);
                            base_card.draw(ui, path, source_path, dest_path, modify_path);
                            path.pop_back();
                        });
                    }
                    ProjectileModifier::OnTrigger(frequency, ref mut base_card)
                    | ProjectileModifier::Trail(frequency, ref mut base_card) => {
                        drag_source(ui, item_id, true, |ui| {
                            draw_modifier(
                                ui,
                                item_id,
                                name,
                                Some(frequency),
                                hover_text,
                                false,
                                modify_path,
                                path,
                            );
                            path.push_back(0);
                            base_card.draw(ui, path, source_path, dest_path, modify_path);
                            path.pop_back();
                        });
                    }
                    _ => panic!("Invalid State"),
                }
            });
        }
    }

    fn modify_from_path(&mut self, path: &mut VecDeque<u32>, modification_type: ModificationType) {
        if path.is_empty() {
            match self {
                ProjectileModifier::SimpleModify(_type, ref mut value) => match modification_type {
                    ModificationType::Add => *value += 1,
                    ModificationType::Remove => *value -= 1,
                    ModificationType::Other => {}
                },
                ProjectileModifier::Trail(ref mut frequency, _card) => match modification_type {
                    ModificationType::Add => *frequency += 1,
                    ModificationType::Remove => {
                        if *frequency > 1 {
                            *frequency -= 1
                        }
                    }
                    ModificationType::Other => {}
                },
                ProjectileModifier::OnTrigger(ref mut id, _card) => match modification_type {
                    ModificationType::Add => *id += 1,
                    ModificationType::Remove => {
                        if *id > 0 {
                            *id -= 1
                        }
                    }
                    ModificationType::Other => {}
                },
                ProjectileModifier::FriendlyFire
                | ProjectileModifier::LockToOwner
                | ProjectileModifier::NoEnemyFire
                | ProjectileModifier::PiercePlayers
                | ProjectileModifier::OnHeadshot(_)
                | ProjectileModifier::OnHit(_)
                | ProjectileModifier::OnExpiry(_)
                | ProjectileModifier::WallBounce
                | ProjectileModifier::None => {}
            }
        } else {
            assert!(path.pop_front().unwrap() == 0);
            match self {
                ProjectileModifier::OnHit(ref mut card) => {
                    card.modify_from_path(path, modification_type)
                }
                ProjectileModifier::OnHeadshot(ref mut card) => {
                    card.modify_from_path(path, modification_type)
                }
                ProjectileModifier::OnExpiry(ref mut card) => {
                    card.modify_from_path(path, modification_type)
                }
                ProjectileModifier::OnTrigger(_, ref mut card) => {
                    card.modify_from_path(path, modification_type)
                }
                ProjectileModifier::Trail(_freqency, ref mut card) => {
                    card.modify_from_path(path, modification_type)
                }
                _ => panic!("Invalid state"),
            }
        }
    }

    fn take_from_path(&mut self, path: &mut VecDeque<u32>) -> DragableCard {
        if path.is_empty() {
            let value = self.clone();
            *self = ProjectileModifier::None;
            DragableCard::ProjectileModifier(value)
        } else {
            assert!(path.pop_front().unwrap() == 0);
            if path.is_empty() {
                let card_ref = match self {
                    ProjectileModifier::OnHit(ref mut card)
                    | ProjectileModifier::OnHeadshot(ref mut card)
                    | ProjectileModifier::OnExpiry(ref mut card)
                    | ProjectileModifier::OnTrigger(_, ref mut card)
                    | ProjectileModifier::Trail(_, ref mut card) => card,
                    invalid_take_modifier => panic!(
                        "Invalid state: cannot take from {:?}",
                        invalid_take_modifier
                    ),
                };
                let result = DragableCard::BaseCard(card_ref.clone());
                *card_ref = BaseCard::None;
                result
            } else {
                match self {
                    ProjectileModifier::OnHit(ref mut card)
                    | ProjectileModifier::OnHeadshot(ref mut card)
                    | ProjectileModifier::OnExpiry(ref mut card)
                    | ProjectileModifier::OnTrigger(_, ref mut card)
                    | ProjectileModifier::Trail(_, ref mut card) => card.take_from_path(path),
                    _ => panic!("Invalid state"),
                }
            }
        }
    }

    fn insert_to_path(&mut self, path: &mut VecDeque<u32>, item: DragableCard) {
        assert!(path.pop_front().unwrap() == 0);
        match self {
            ProjectileModifier::OnHit(ref mut card)
            | ProjectileModifier::OnHeadshot(ref mut card)
            | ProjectileModifier::OnExpiry(ref mut card)
            | ProjectileModifier::OnTrigger(_, ref mut card)
            | ProjectileModifier::Trail(_, ref mut card) => card.insert_to_path(path, item),
            _ => panic!("Invalid state"),
        }
    }

    fn cleanup(&mut self, path: &mut VecDeque<u32>) {
        assert!(path.pop_front().unwrap() == 0);
        match self {
            ProjectileModifier::OnHit(ref mut card)
            | ProjectileModifier::OnHeadshot(ref mut card)
            | ProjectileModifier::OnExpiry(ref mut card)
            | ProjectileModifier::OnTrigger(_, ref mut card)
            | ProjectileModifier::Trail(_, ref mut card) => card.cleanup(path),
            ref invalid => panic!(
                "Invalid state: cannot follow path {:?} into {:?}",
                path, invalid
            ),
        }
    }
}

impl DrawableCard for StatusEffect {
    fn draw(
        &mut self,
        ui: &mut Ui,
        path: &mut VecDeque<u32>,
        source_path: &mut Option<(VecDeque<u32>, DragableType)>,
        dest_path: &mut Option<(VecDeque<u32>, DropableType)>,
        modify_path: &mut Option<(VecDeque<u32>, ModificationType)>,
    ) {
        let item_id = egui::Id::new(ID_SOURCE).with(path.clone());
        let mut advanced_effect = false;
        let name = self.get_name();
        let hover_text = self.get_hover_text();
        match self {
            StatusEffect::SimpleStatusEffect(
                SimpleStatusEffectType::IncreaseGravity(direction),
                v,
            ) => {
                drag_source(ui, item_id, true, |ui| {
                    draw_modifier(
                        ui,
                        item_id,
                        name,
                        Some(v),
                        hover_text,
                        false,
                        modify_path,
                        path,
                    );
                    path.push_back(0);
                    direction.draw(ui, path, source_path, dest_path, modify_path);
                    path.pop_back();
                });
            }
            StatusEffect::SimpleStatusEffect(_, ref mut v) => draw_modifier(
                ui,
                item_id,
                name,
                Some(v),
                hover_text,
                true,
                modify_path,
                path,
            ),
            StatusEffect::Invincibility
            | StatusEffect::Lockout
            | StatusEffect::Trapped
            | StatusEffect::Stun
            | StatusEffect::None => draw_modifier(
                ui,
                item_id,
                name,
                None::<&mut u32>,
                hover_text,
                true,
                modify_path,
                path,
            ),
            ref modifier if modifier.is_advanced() => {
                advanced_effect = true;
            }
            _ => panic!("Invalid State"),
        }
        if advanced_effect {
            ui.horizontal(|ui| {
                ui.add_space(CARD_UI_SPACING);
                let hover_text = self.get_hover_text();
                match self {
                    StatusEffect::OnHit(base_card) => {
                        drag_source(ui, item_id, true, |ui| {
                            draw_label(ui, "On Hit", hover_text, modify_path, path);
                            path.push_back(0);
                            base_card.draw(ui, path, source_path, dest_path, modify_path);
                            path.pop_back();
                        });
                    }
                    _ => panic!("Invalid State"),
                }
            });
        }
    }

    fn modify_from_path(&mut self, path: &mut VecDeque<u32>, modification_type: ModificationType) {
        match self {
            StatusEffect::SimpleStatusEffect(_, ref mut stacks) => match modification_type {
                ModificationType::Add => *stacks += 1,
                ModificationType::Remove => *stacks -= 1,
                ModificationType::Other => {}
            },
            StatusEffect::None
            | StatusEffect::Invincibility
            | StatusEffect::Trapped
            | StatusEffect::Lockout
            | StatusEffect::Stun => {}
            StatusEffect::OnHit(ref mut card) => {
                assert!(path.pop_front().unwrap() == 0);
                card.modify_from_path(path, modification_type)
            }
        }
    }

    fn take_from_path(&mut self, path: &mut VecDeque<u32>) -> DragableCard {
        match self {
            StatusEffect::OnHit(card) => {
                assert!(path.pop_front().unwrap() == 0);
                card.take_from_path(path)
            }
            StatusEffect::SimpleStatusEffect(
                SimpleStatusEffectType::IncreaseGravity(ref mut direction),
                _,
            ) => {
                assert!(path.pop_front().unwrap() == 0);
                direction.take_from_path(path)
            }
            _ => panic!("Invalid state"),
        }
    }

    fn insert_to_path(&mut self, path: &mut VecDeque<u32>, item: DragableCard) {
        match self {
            StatusEffect::OnHit(ref mut card) => card.insert_to_path(path, item),
            StatusEffect::SimpleStatusEffect(
                SimpleStatusEffectType::IncreaseGravity(direction),
                _,
            ) => {
                direction.insert_to_path(path, item);
            }
            _ => panic!("Invalid state"),
        }
    }

    fn cleanup(&mut self, path: &mut VecDeque<u32>) {
        assert!(path.pop_front().unwrap() == 0);
        match self {
            StatusEffect::OnHit(ref mut card) => card.cleanup(path),
            StatusEffect::SimpleStatusEffect(
                SimpleStatusEffectType::IncreaseGravity(direction),
                _,
            ) => {
                direction.cleanup(path);
            }
            _ => {}
        }
    }
}

impl DrawableCard for DirectionCard {
    fn draw(
        &mut self,
        ui: &mut Ui,
        path: &mut VecDeque<u32>,
        source_path: &mut Option<(VecDeque<u32>, DragableType)>,
        dest_path: &mut Option<(VecDeque<u32>, DropableType)>,
        modify_path: &mut Option<(VecDeque<u32>, ModificationType)>,
    ) {
        let item_id = egui::Id::new(ID_SOURCE).with(path.clone());
        let is_draggable = !matches!(self, DirectionCard::None);
        let can_accept_what_is_being_dragged = true; // We accept anything being dragged (for now) ¯\_(ツ)_/¯
        drag_source(ui, item_id, is_draggable, |ui| {
            ui.visuals_mut().widgets.inactive.bg_stroke = Stroke::new(0.5, Color32::TRANSPARENT);
            ui.visuals_mut().widgets.inactive.bg_fill = Color32::TRANSPARENT;
            let response = drop_target(ui, can_accept_what_is_being_dragged, |ui| {
                let where_to_put_background = ui.painter().add(Shape::Noop);
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        let name = match self {
                            DirectionCard::None => "None",
                            DirectionCard::Up => "Up",
                            DirectionCard::Forward => "Forward",
                            DirectionCard::Movement => "Movement",
                        };
                        draw_label(ui, name, name.to_string(), modify_path, path);
                        ui.add_space(CARD_UI_SPACING);
                    });
                });

                let color = Color32::from_rgb(100, 255, 150);
                let min_rect = ui.min_rect().expand(CARD_UI_SPACING / 2.0);
                ui.painter().set(
                    where_to_put_background,
                    epaint::PathShape::convex_polygon(
                        vec![
                            min_rect.left_bottom(),
                            min_rect.left_top(),
                            min_rect.right_top() - vec2(10.0, 0.0),
                            min_rect.right_top().lerp(min_rect.right_bottom(), 0.5),
                            min_rect.right_bottom() - vec2(10.0, 0.0),
                        ],
                        darken(color, 0.25),
                        Stroke::new(1.0, color),
                    ),
                );
            })
            .response;

            if dest_path.is_none() {
                let is_being_dragged = ui.memory(|mem| mem.is_anything_being_dragged());
                if is_being_dragged && can_accept_what_is_being_dragged && response.hovered() {
                    *dest_path = Some((path.clone(), DropableType::Direction));
                }
            }
        });

        if source_path.is_none() && ui.memory(|mem| mem.is_being_dragged(item_id)) {
            *source_path = Some((path.clone(), DragableType::Direction));
        }
    }

    fn modify_from_path(
        &mut self,
        _path: &mut VecDeque<u32>,
        _modification_type: ModificationType,
    ) {
    }

    fn take_from_path(&mut self, path: &mut VecDeque<u32>) -> DragableCard {
        assert!(path.is_empty());
        let taken_direction = self.clone();
        *self = DirectionCard::None;
        DragableCard::Direction(taken_direction)
    }

    fn insert_to_path(&mut self, path: &mut VecDeque<u32>, item: DragableCard) {
        assert!(path.is_empty());
        if let DragableCard::Direction(new_direction) = item {
            *self = new_direction;
        } else {
            panic!("Invalid state");
        }
    }

    fn cleanup(&mut self, _path: &mut VecDeque<u32>) {}
}

impl DrawableCard for BaseCard {
    fn draw(
        &mut self,
        ui: &mut Ui,
        path: &mut VecDeque<u32>,
        source_path: &mut Option<(VecDeque<u32>, DragableType)>,
        dest_path: &mut Option<(VecDeque<u32>, DropableType)>,
        modify_path: &mut Option<(VecDeque<u32>, ModificationType)>,
    ) {
        let item_id = egui::Id::new(ID_SOURCE).with(path.clone());
        let is_draggable = !matches!(self, BaseCard::None | BaseCard::Palette(_));
        let can_accept_what_is_being_dragged = true; // We accept anything being dragged (for now) ¯\_(ツ)_/¯
        drag_source(ui, item_id, is_draggable, |ui| match self {
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
                                    for (modifier_idx, modifier) in modifiers.iter_mut().enumerate()
                                    {
                                        if modifier.is_advanced() {
                                            advanced_modifiers.push((modifier_idx, modifier));
                                            continue;
                                        }
                                        path.push_back(modifier_idx as u32);
                                        modifier.draw(
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
                                    modifier.draw(ui, path, source_path, dest_path, modify_path);
                                    path.pop_back();
                                }
                                ui.add_space(CARD_UI_SPACING);
                            });
                            ui.add_space(CARD_UI_SPACING);
                        });

                        for (modifier_idx, _modifier) in modifiers.iter().enumerate() {
                            path.push_back(modifier_idx as u32);
                            let item_id = egui::Id::new(ID_SOURCE).with(path.clone());
                            if source_path.is_none()
                                && ui.memory(|mem| mem.is_being_dragged(item_id))
                            {
                                *source_path =
                                    Some((path.clone(), DragableType::ProjectileModifier));
                            }
                            path.pop_back();
                        }
                    })
                    .response;

                    if dest_path.is_none() {
                        let is_being_dragged = ui.memory(|mem| mem.is_anything_being_dragged());
                        if is_being_dragged
                            && can_accept_what_is_being_dragged
                            && response.hovered()
                        {
                            *dest_path = Some((path.clone(), DropableType::BaseProjectile));
                        }
                    }
                });
            }
            BaseCard::MultiCast(cards, modifiers) => {
                ui.vertical(|ui| {
                    ui.visuals_mut().widgets.inactive.bg_stroke = Stroke::new(0.5, Color32::YELLOW);
                    ui.visuals_mut().widgets.inactive.bg_fill =
                        darken(ui.visuals_mut().widgets.inactive.bg_stroke.color, 0.25);
                    let response = drop_target(ui, can_accept_what_is_being_dragged, |ui| {
                        ui.horizontal(|ui| {
                            ui.add_space(CARD_UI_SPACING);
                            ui.vertical(|ui| {
                                ui.add_space(CARD_UI_SPACING);
                                ui.horizontal_wrapped(|ui| {
                                    ui.label("Multicast");
                                    path.push_back(0);
                                    for (mod_idx, modifier) in modifiers.iter_mut().enumerate() {
                                        path.push_back(mod_idx as u32);
                                        modifier.draw(
                                            ui,
                                            path,
                                            source_path,
                                            dest_path,
                                            modify_path,
                                        );
                                        path.pop_back();
                                    }
                                    path.pop_back();
                                });
                                path.push_back(1);
                                for (card_idx, card) in cards.iter_mut().enumerate() {
                                    path.push_back(card_idx as u32);
                                    card.draw(ui, path, source_path, dest_path, modify_path);
                                    path.pop_back();
                                }
                                path.pop_back();
                                ui.add_space(CARD_UI_SPACING);
                            });
                            ui.add_space(CARD_UI_SPACING);
                        });

                        path.push_back(0);
                        for (modifier_idx, _modifier) in modifiers.iter().enumerate() {
                            path.push_back(modifier_idx as u32);
                            let item_id = egui::Id::new(ID_SOURCE).with(path.clone());
                            if source_path.is_none()
                                && ui.memory(|mem| mem.is_being_dragged(item_id))
                            {
                                *source_path =
                                    Some((path.clone(), DragableType::MultiCastModifier));
                            }
                            path.pop_back();
                        }
                        path.pop_back();
                    })
                    .response;

                    if dest_path.is_none() {
                        let is_being_dragged = ui.memory(|mem| mem.is_anything_being_dragged());
                        if is_being_dragged
                            && can_accept_what_is_being_dragged
                            && response.hovered()
                        {
                            *dest_path = Some((path.clone(), DropableType::MultiCastBaseCard));
                        }
                    }
                });
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
                let name = effect.get_name();
                let hover_text = effect.get_hover_text();
                ui.vertical(|ui| {
                    ui.add_space(CARD_UI_SPACING);
                    ui.horizontal(|ui| {
                        ui.add_space(CARD_UI_SPACING);
                        ui.label("Apply Effect");
                        match effect {
                            Effect::Damage(ref mut v) => draw_modifier(
                                ui,
                                item_id,
                                name,
                                Some(v),
                                hover_text,
                                false,
                                modify_path,
                                path,
                            ),
                            Effect::Knockback(v, direction) => {
                                draw_modifier(
                                    ui,
                                    item_id,
                                    name,
                                    Some(v),
                                    hover_text,
                                    false,
                                    modify_path,
                                    path,
                                );
                                path.push_back(0);
                                direction.draw(ui, path, source_path, dest_path, modify_path);
                                path.pop_back();
                            }
                            Effect::Cleanse | Effect::Teleport => draw_modifier(
                                ui,
                                item_id,
                                name,
                                None::<&mut u32>,
                                hover_text,
                                false,
                                modify_path,
                                path,
                            ),
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
            BaseCard::StatusEffects(duration, effects) => {
                let hover_text = format!(
                    "Apply status effects for a duration of {}s",
                    *duration as f32 * BaseCard::EFFECT_LENGTH_SCALE
                );
                ui.vertical(|ui| {
                    ui.visuals_mut().widgets.inactive.bg_stroke =
                        Stroke::new(0.5, Color32::LIGHT_RED);
                    ui.visuals_mut().widgets.inactive.bg_fill =
                        darken(ui.visuals_mut().widgets.inactive.bg_stroke.color, 0.25);

                    let response = drop_target(ui, can_accept_what_is_being_dragged, |ui| {
                        let mut advanced_effects = vec![];
                        ui.horizontal(|ui| {
                            ui.add_space(CARD_UI_SPACING);
                            ui.vertical(|ui| {
                                ui.add_space(CARD_UI_SPACING);
                                ui.horizontal_wrapped(|ui| {
                                    draw_modifier(
                                        ui,
                                        item_id,
                                        "Apply Status Effects".to_string(),
                                        Some(duration),
                                        hover_text,
                                        false,
                                        modify_path,
                                        path,
                                    );
                                    for (effect_idx, effect) in effects.iter_mut().enumerate() {
                                        if effect.is_advanced() {
                                            advanced_effects.push((effect_idx, effect));
                                            continue;
                                        }
                                        path.push_back(effect_idx as u32);
                                        effect.draw(ui, path, source_path, dest_path, modify_path);
                                        path.pop_back();
                                    }
                                });

                                for (modifier_idx, modifier) in advanced_effects.into_iter() {
                                    path.push_back(modifier_idx as u32);
                                    modifier.draw(ui, path, source_path, dest_path, modify_path);
                                    path.pop_back();
                                }
                                ui.add_space(CARD_UI_SPACING);
                            });
                            ui.add_space(CARD_UI_SPACING);
                        });

                        for (modifier_idx, _modifier) in effects.iter().enumerate() {
                            path.push_back(modifier_idx as u32);
                            let item_id = egui::Id::new(ID_SOURCE).with(path.clone());
                            if source_path.is_none()
                                && ui.memory(|mem| mem.is_being_dragged(item_id))
                            {
                                *source_path = Some((path.clone(), DragableType::StatusEffect));
                            }
                            path.pop_back();
                        }
                    })
                    .response;

                    if dest_path.is_none() {
                        let is_being_dragged = ui.memory(|mem| mem.is_anything_being_dragged());
                        if is_being_dragged
                            && can_accept_what_is_being_dragged
                            && response.hovered()
                        {
                            *dest_path = Some((path.clone(), DropableType::BaseStatusEffects));
                        }
                    }
                });
            }
            BaseCard::Trigger(id) => {
                let where_to_put_background = ui.painter().add(Shape::Noop);
                ui.vertical(|ui| {
                    ui.add_space(CARD_UI_SPACING);
                    ui.horizontal(|ui| {
                        ui.add_space(CARD_UI_SPACING);
                        draw_modifier(
                            ui,
                            item_id,
                            "Trigger".to_string(),
                            Some(id),
                            "Cause on trigger events to activate".to_string(),
                            false,
                            modify_path,
                            path,
                        );
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
            BaseCard::Palette(palette_cards) => {
                ui.visuals_mut().widgets.inactive.bg_stroke = Stroke::new(1.0, Color32::GRAY);
                ui.visuals_mut().widgets.inactive.bg_fill = Color32::BLACK;
                let response = drop_target(ui, can_accept_what_is_being_dragged, |ui| {
                    ui.set_min_size(vec2(200.0, 40.0));
                    ui.add_space(CARD_UI_SPACING);
                    ui.horizontal_wrapped(|ui| {
                        ui.set_row_height(ui.spacing().interact_size.y);
                        for (card_idx, card) in palette_cards.iter_mut().enumerate() {
                            path.push_back(card_idx as u32);
                            card.draw_draggable(ui, path, source_path, dest_path, modify_path);
                            path.pop_back();
                        }
                    });

                    for (card_idx, card) in palette_cards.iter().enumerate() {
                        path.push_back(card_idx as u32);
                        let item_id = egui::Id::new(ID_SOURCE).with(path.clone());
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

    fn modify_from_path(&mut self, path: &mut VecDeque<u32>, modification_type: ModificationType) {
        match self {
            BaseCard::Projectile(modifiers) => {
                let idx = path.pop_front().unwrap() as usize;
                modifiers[idx].modify_from_path(path, modification_type);
            }
            BaseCard::MultiCast(cards, modifiers) => {
                let type_idx = path.pop_front().unwrap() as usize;
                if type_idx == 0 {
                    let idx = path.pop_front().unwrap() as usize;
                    modifiers[idx].modify_from_path(path, modification_type);
                } else if type_idx == 1 {
                    let idx = path.pop_front().unwrap() as usize;
                    cards[idx].modify_from_path(path, modification_type)
                } else {
                    panic!("Invalid state");
                }
            }
            BaseCard::CreateMaterial(_) => panic!("Invalid state"),
            BaseCard::StatusEffects(duration, effects) => {
                if path.is_empty() {
                    match modification_type {
                        ModificationType::Add => *duration += 1,
                        ModificationType::Remove => {
                            if *duration > 1 {
                                *duration -= 1
                            }
                        }
                        ModificationType::Other => {}
                    }
                } else {
                    let effect_idx = path.pop_front().unwrap() as usize;
                    effects[effect_idx].modify_from_path(path, modification_type);
                }
            }
            BaseCard::Effect(effect) => {
                assert!(path.is_empty());
                match effect {
                    Effect::Damage(ref mut damage) => match modification_type {
                        ModificationType::Add => *damage += 1,
                        ModificationType::Remove => *damage -= 1,
                        ModificationType::Other => {}
                    },
                    Effect::Knockback(ref mut knockback, _) => match modification_type {
                        ModificationType::Add => *knockback += 1,
                        ModificationType::Remove => *knockback -= 1,
                        ModificationType::Other => {}
                    },
                    Effect::Cleanse => {}
                    Effect::Teleport => {}
                }
            }
            BaseCard::Trigger(ref mut id) => match modification_type {
                ModificationType::Add => *id += 1,
                ModificationType::Remove => {
                    if *id > 0 {
                        *id -= 1
                    }
                }
                ModificationType::Other => {}
            },
            BaseCard::None => panic!("Invalid state"),
            BaseCard::Palette(..) => {}
        }
    }

    fn take_from_path(&mut self, path: &mut VecDeque<u32>) -> DragableCard {
        if path.is_empty() {
            let result = DragableCard::BaseCard(self.clone());
            *self = BaseCard::None;
            return result;
        }
        match self {
            BaseCard::Projectile(modifiers) => {
                let idx = path.pop_front().unwrap() as usize;
                modifiers[idx].take_from_path(path)
            }
            BaseCard::MultiCast(cards, modifiers) => {
                let type_idx = path.pop_front().unwrap() as usize;
                if type_idx == 0 {
                    let idx = path.pop_front().unwrap() as usize;
                    modifiers[idx].take_from_path(path)
                } else if type_idx == 1 {
                    let idx = path.pop_front().unwrap() as usize;
                    if path.is_empty() {
                        let value = cards[idx].clone();
                        cards[idx] = BaseCard::None;
                        DragableCard::BaseCard(value)
                    } else {
                        cards[idx].take_from_path(path)
                    }
                } else {
                    panic!("Invalid state");
                }
            }
            BaseCard::StatusEffects(_, effects) => {
                let Some(effect_idx) = path.pop_front() else {
                    panic!("Invalid state: path is empty");
                };
                if path.is_empty() {
                    let effect = effects[effect_idx as usize].clone();
                    effects[effect_idx as usize] = StatusEffect::None;
                    return DragableCard::StatusEffect(effect);
                }
                effects
                    .get_mut(effect_idx as usize)
                    .unwrap()
                    .take_from_path(path)
            }
            BaseCard::Palette(cards) => {
                let card_idx = path.pop_front().unwrap() as usize;
                cards[card_idx].clone()
            }
            BaseCard::Effect(Effect::Knockback(_, direction)) => {
                assert!(path.pop_front().unwrap() == 0);
                let value = DragableCard::Direction(direction.clone());
                *direction = DirectionCard::None;
                value
            }
            invalid_take @ (BaseCard::CreateMaterial(_)
            | BaseCard::None
            | BaseCard::Trigger(_)
            | BaseCard::Effect(_)) => panic!("Invalid state: cannot take from {:?}", invalid_take),
        }
    }

    fn insert_to_path(&mut self, path: &mut VecDeque<u32>, item: DragableCard) {
        match self {
            BaseCard::Projectile(modifiers) => {
                if path.is_empty() {
                    let DragableCard::ProjectileModifier(item) = item else {
                        panic!("Invalid state")
                    };
                    if let ProjectileModifier::SimpleModify(last_ty, last_s) = item.clone() {
                        let mut combined = false;
                        for modifier in modifiers.iter_mut() {
                            match modifier {
                                ProjectileModifier::SimpleModify(ty, s) => {
                                    if last_ty == *ty {
                                        *s += last_s;
                                        combined = true;
                                        break;
                                    }
                                }
                                _ => {}
                            }
                        }
                        if !combined {
                            modifiers.push(item.clone());
                        }
                    } else {
                        modifiers.push(item.clone());
                    }

                    modifiers.retain(|modifier| match modifier {
                        ProjectileModifier::SimpleModify(_, s) => *s != 0,
                        _ => true,
                    });
                } else {
                    let idx = path.pop_front().unwrap() as usize;
                    modifiers[idx].insert_to_path(path, item);
                }
            }
            BaseCard::MultiCast(cards, modifiers) => {
                if path.is_empty() {
                    if let DragableCard::BaseCard(item) = item {
                        cards.push(item);
                    } else if let DragableCard::MultiCastModifier(modifier_item) = item {
                        let mut combined = false;
                        match modifier_item.clone() {
                            MultiCastModifier::None => {}
                            MultiCastModifier::Duplication(last_s) => {
                                for modifier in modifiers.iter_mut() {
                                    match modifier {
                                        MultiCastModifier::Duplication(s) => {
                                            *s += last_s;
                                            combined = true;
                                            break;
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            MultiCastModifier::Spread(last_s) => {
                                for modifier in modifiers.iter_mut() {
                                    match modifier {
                                        MultiCastModifier::Spread(s) => {
                                            *s += last_s;
                                            combined = true;
                                            break;
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }

                        if !combined {
                            modifiers.push(modifier_item.clone());
                        }
                    } else {
                        panic!("Invalid state")
                    }
                } else {
                    assert!(path.pop_front().unwrap() == 1);
                    let idx = path.pop_front().unwrap() as usize;
                    cards[idx].insert_to_path(path, item)
                }
            }
            BaseCard::StatusEffects(_, effects) => {
                if path.is_empty() {
                    let DragableCard::StatusEffect(item) = item else {
                        panic!("Invalid state")
                    };
                    if let StatusEffect::SimpleStatusEffect(last_ty, last_s) = item.clone() {
                        let mut combined = false;
                        for effect in effects.iter_mut() {
                            match effect {
                                StatusEffect::SimpleStatusEffect(ty, s) => {
                                    if last_ty == *ty {
                                        *s += last_s;
                                        combined = true;
                                        break;
                                    }
                                }
                                _ => {}
                            }
                        }
                        if !combined {
                            effects.push(item.clone());
                        }
                    } else {
                        effects.push(item.clone());
                    }

                    effects.retain(|effect| match effect {
                        StatusEffect::SimpleStatusEffect(_, s) => *s != 0,
                        _ => true,
                    });
                } else {
                    let idx = path.pop_front().unwrap() as usize;
                    assert!(path.pop_front().unwrap() == 0);
                    effects[idx].insert_to_path(path, item);
                }
            }
            BaseCard::Effect(Effect::Knockback(_, ref mut direction)) => {
                assert!(path.pop_front().unwrap() == 0);
                if let DragableCard::Direction(new_direction) = item {
                    *direction = new_direction;
                } else {
                    panic!("Invalid state")
                }
            }
            BaseCard::None => {
                assert!(
                    path.is_empty(),
                    "Invalid state: should not have nonempty path {:?} when inserting into None",
                    path
                );
                let DragableCard::BaseCard(item) = item else {
                    panic!("Invalid state")
                };
                *self = item;
            }
            c => panic!("Invalid state: Could not insert into {:?}", c),
        }
    }

    fn cleanup(&mut self, path: &mut VecDeque<u32>) {
        match self {
            BaseCard::Projectile(modifiers) => {
                if path.len() <= 1 {
                    modifiers.retain(|modifier| match modifier {
                        ProjectileModifier::None => false,
                        _ => true,
                    });
                } else {
                    let idx = path.pop_front().unwrap() as usize;
                    modifiers[idx].cleanup(path);
                }
            }
            BaseCard::MultiCast(cards, modifiers) => {
                if path.is_empty() {
                    cards.retain(|card| !matches!(card, BaseCard::None));
                } else {
                    let idx_type = path.pop_front().unwrap();
                    if idx_type == 0 {
                        let idx = path.pop_front().unwrap() as usize;
                        assert!(path.is_empty());
                        match modifiers[idx] {
                            MultiCastModifier::None => {
                                modifiers.remove(idx);
                            }
                            MultiCastModifier::Duplication(s) => {
                                if s == 0 {
                                    modifiers.remove(idx);
                                }
                            }
                            MultiCastModifier::Spread(s) => {
                                if s == 0 {
                                    modifiers.remove(idx);
                                }
                            }
                        }
                    } else if idx_type == 1 {
                        let idx = path.pop_front().unwrap() as usize;
                        if path.is_empty() {
                            if matches!(cards[idx], BaseCard::None) {
                                cards.remove(idx);
                            }
                        } else {
                            cards[idx].cleanup(path);
                        }
                    } else {
                        panic!("Invalid state");
                    }
                }
            }
            BaseCard::StatusEffects(_, effects) => {
                if path.len() <= 1 {
                    effects.retain(|effect| match effect {
                        StatusEffect::None => false,
                        _ => true,
                    });
                } else {
                    let idx = path.pop_front().unwrap() as usize;
                    assert!(path.pop_front().unwrap() == 0);
                    match effects[idx] {
                        StatusEffect::OnHit(ref mut card) => card.cleanup(path),
                        StatusEffect::SimpleStatusEffect(
                            SimpleStatusEffectType::IncreaseGravity(_),
                            _,
                        ) => {}
                        ref invalid => panic!(
                            "Invalid state: cannot follow path {} into {:?}",
                            idx, invalid
                        ),
                    }
                }
            }
            BaseCard::None => {
                assert!(path.is_empty(), "Invalid state");
            }
            BaseCard::Effect(Effect::Knockback(_, _)) => {}
            c => panic!("Invalid state: Could cleanup {:?}", c),
        }
    }
}

impl DragableCard {
    pub fn get_type(&self) -> DragableType {
        match self {
            DragableCard::BaseCard(_) => DragableType::BaseCard,
            DragableCard::CooldownModifier(_) => DragableType::CooldownModifier,
            DragableCard::MultiCastModifier(_) => DragableType::MultiCastModifier,
            DragableCard::ProjectileModifier(_) => DragableType::ProjectileModifier,
            DragableCard::StatusEffect(_) => DragableType::StatusEffect,
            DragableCard::Direction(_) => DragableType::Direction,
        }
    }

    pub fn draw_draggable(
        &mut self,
        ui: &mut Ui,
        path: &mut VecDeque<u32>,
        source_path: &mut Option<(VecDeque<u32>, DragableType)>,
        dest_path: &mut Option<(VecDeque<u32>, DropableType)>,
        modify_path: &mut Option<(VecDeque<u32>, ModificationType)>,
    ) {
        match self {
            DragableCard::BaseCard(card) => {
                card.draw(ui, path, source_path, dest_path, modify_path)
            }
            DragableCard::CooldownModifier(modifier) => {
                modifier.draw(ui, path, source_path, dest_path, modify_path)
            }
            DragableCard::MultiCastModifier(modifier) => {
                modifier.draw(ui, path, source_path, dest_path, modify_path)
            }
            DragableCard::ProjectileModifier(modifier) => {
                modifier.draw(ui, path, source_path, dest_path, modify_path)
            }
            DragableCard::StatusEffect(effect) => {
                effect.draw(ui, path, source_path, dest_path, modify_path)
            }
            DragableCard::Direction(direction) => {
                direction.draw(ui, path, source_path, dest_path, modify_path)
            }
        }
    }
}

pub fn draw_label(
    ui: &mut Ui,
    name: &str,
    hover_text: String,
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
}

pub fn draw_modifier<T: std::fmt::Display + Numeric + Copy>(
    ui: &mut Ui,
    id: Id,
    name: String,
    count: Option<&mut T>,
    hover_text: String,
    handle_drag: bool,
    modify_path: &mut Option<(VecDeque<u32>, ModificationType)>,
    path: &mut VecDeque<u32>,
) {
    ui.style_mut().wrap = Some(false);
    let mut job = LayoutJob::default();
    job.append(
        &name,
        0.0,
        TextFormat {
            color: Color32::WHITE,
            ..Default::default()
        },
    );
    let add_contents = |ui: &mut Ui| {
        ui.scope(|ui| {
            ui.style_mut().spacing.interact_size.x = 0.0;
            ui.style_mut().spacing.interact_size.y = 0.0;
            ui.style_mut().spacing.item_spacing.x = 0.0;
            ui.style_mut().spacing.button_padding.x = 1.0;
            ui.style_mut().visuals.widgets.inactive.bg_stroke = Stroke::new(0.0, Color32::WHITE);
            ui.style_mut().visuals.widgets.inactive.bg_fill = Color32::TRANSPARENT;
            ui.style_mut().visuals.widgets.inactive.weak_bg_fill = Color32::TRANSPARENT;
            ui.style_mut().visuals.widgets.active.bg_fill = Color32::TRANSPARENT;
            ui.style_mut().visuals.widgets.active.weak_bg_fill = Color32::TRANSPARENT;
            ui.style_mut().visuals.widgets.hovered.weak_bg_fill = Color32::TRANSPARENT;
            ui.style_mut().text_styles = [
                (
                    TextStyle::Heading,
                    FontId::new(30.0, egui::FontFamily::Proportional),
                ),
                (
                    TextStyle::Name("Heading2".into()),
                    FontId::new(25.0, egui::FontFamily::Proportional),
                ),
                (
                    TextStyle::Name("Context".into()),
                    FontId::new(23.0, egui::FontFamily::Proportional),
                ),
                (
                    TextStyle::Body,
                    FontId::new(18.0, egui::FontFamily::Proportional),
                ),
                (
                    TextStyle::Monospace,
                    FontId::new(14.0, egui::FontFamily::Proportional),
                ),
                (
                    TextStyle::Button,
                    FontId::new(7.0, egui::FontFamily::Proportional),
                ),
                (
                    TextStyle::Small,
                    FontId::new(10.0, egui::FontFamily::Proportional),
                ),
            ]
            .into();
            let response = ui.add(Label::new(job)).on_hover_text(hover_text);
            if let Some(count) = count {
                ui.vertical(|ui| {
                    if ui.add(DragValue::new(count).speed(0.1)).changed() {
                        if modify_path.is_none() {
                            *modify_path = Some((path.clone(), ModificationType::Other));
                        }
                    }
                });
            }
            response
        })
        .response
    };
    let is_being_dragged = ui.memory(|mem| mem.is_being_dragged(id));

    if !is_being_dragged || !handle_drag {
        let prev_frame_area: Option<Rect> = ui.data(|d| d.get_temp(id));
        let mut size = vec2(0.0, 0.0);
        //load from previous frame
        if let Some(area) = prev_frame_area {
            if handle_drag {
                // Check for drags:
                let response = ui.interact(area, id, Sense::drag());
                if response.hovered() {
                    ui.ctx().set_cursor_icon(CursorIcon::Grab);
                }
            }
            size.x = area.size().x;
        }
        if ui.available_size_before_wrap().x < size.x {
            ui.end_row();
        }
        let response = ui.scope(add_contents).response;

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
        ui.data_mut(|d| d.insert_temp(id, response.rect.shrink(CARD_UI_SPACING)));
    } else {
        ui.ctx().set_cursor_icon(CursorIcon::Grabbing);

        // Paint the body to a new layer:
        let layer_id = LayerId::new(Order::Tooltip, id);
        let response = ui.with_layer_id(layer_id, add_contents).response;

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

                let selector_height = ui
                    .scope(|ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(RichText::new("Card Editor").color(Color32::WHITE));
                            if ui.button("Export to Clipboard").clicked() {
                                let export = ron::to_string(&gui_state.gui_deck).unwrap();
                                ui.output_mut(|o| o.copied_text = export);
                            }

                            if ui.button("Import from Clipboard").clicked() {
                                let mut clipboard = clippers::Clipboard::get();
                                let import: Option<Deck> = match clipboard.read() {
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
                                    gui_state.gui_deck = import;
                                }
                            }

                            if ui.button("Clear Dock").clicked() {
                                gui_state.dock_cards = vec![];
                            }
                        });

                        ui.horizontal_wrapped(|ui| {
                            ui.selectable_value(
                                &mut gui_state.palette_state,
                                PaletteState::BaseCards,
                                "Base Cards",
                            );
                            ui.selectable_value(
                                &mut gui_state.palette_state,
                                PaletteState::Materials,
                                "Materials",
                            );
                            ui.selectable_value(
                                &mut gui_state.palette_state,
                                PaletteState::ProjectileModifiers,
                                "Projectile Modifiers",
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
                                PaletteState::StatusEffects,
                                "Status Effects",
                            );
                            ui.selectable_value(
                                &mut gui_state.palette_state,
                                PaletteState::CooldownModifiers,
                                "Cooldown Modifiers",
                            );
                            ui.selectable_value(
                                &mut gui_state.palette_state,
                                PaletteState::Directions,
                                "Directions",
                            );
                            ui.selectable_value(
                                &mut gui_state.palette_state,
                                PaletteState::Dock,
                                "Dock",
                            );
                        })
                    })
                    .response
                    .rect
                    .height();

                let mut source_path = None;
                let mut dest_path = None;
                let mut modify_path = None;
                let mut palette_card = BaseCard::Palette(match gui_state.palette_state {
                    PaletteState::ProjectileModifiers => vec![
                        DragableCard::ProjectileModifier(ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Gravity,
                            1,
                        )),
                        DragableCard::ProjectileModifier(ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Health,
                            1,
                        )),
                        DragableCard::ProjectileModifier(ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Length,
                            1,
                        )),
                        DragableCard::ProjectileModifier(ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Width,
                            1,
                        )),
                        DragableCard::ProjectileModifier(ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Height,
                            1,
                        )),
                        DragableCard::ProjectileModifier(ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Size,
                            1,
                        )),
                        DragableCard::ProjectileModifier(ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Speed,
                            1,
                        )),
                        DragableCard::ProjectileModifier(ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Lifetime,
                            1,
                        )),
                        DragableCard::ProjectileModifier(ProjectileModifier::NoEnemyFire),
                        DragableCard::ProjectileModifier(ProjectileModifier::FriendlyFire),
                        DragableCard::ProjectileModifier(ProjectileModifier::LockToOwner),
                        DragableCard::ProjectileModifier(ProjectileModifier::PiercePlayers),
                    ],
                    PaletteState::BaseCards => vec![
                        DragableCard::BaseCard(BaseCard::Projectile(vec![])),
                        DragableCard::BaseCard(BaseCard::MultiCast(vec![], vec![])),
                        DragableCard::BaseCard(BaseCard::Trigger(0)),
                        DragableCard::BaseCard(BaseCard::Effect(Effect::Damage(1))),
                        DragableCard::BaseCard(BaseCard::Effect(Effect::Knockback(
                            1,
                            DirectionCard::None,
                        ))),
                        DragableCard::BaseCard(BaseCard::Effect(Effect::Cleanse)),
                        DragableCard::BaseCard(BaseCard::Effect(Effect::Teleport)),
                        DragableCard::BaseCard(BaseCard::StatusEffects(1, vec![])),
                    ],
                    PaletteState::AdvancedProjectileModifiers => vec![
                        DragableCard::ProjectileModifier(ProjectileModifier::OnHit(BaseCard::None)),
                        DragableCard::ProjectileModifier(ProjectileModifier::OnHeadshot(
                            BaseCard::None,
                        )),
                        DragableCard::ProjectileModifier(ProjectileModifier::OnExpiry(
                            BaseCard::None,
                        )),
                        DragableCard::ProjectileModifier(ProjectileModifier::OnTrigger(
                            0,
                            BaseCard::None,
                        )),
                        DragableCard::ProjectileModifier(ProjectileModifier::Trail(
                            1,
                            BaseCard::None,
                        )),
                    ],
                    PaletteState::MultiCastModifiers => vec![
                        DragableCard::MultiCastModifier(MultiCastModifier::Spread(1)),
                        DragableCard::MultiCastModifier(MultiCastModifier::Duplication(1)),
                    ],
                    PaletteState::CooldownModifiers => vec![
                        DragableCard::CooldownModifier(CooldownModifier::SimpleCooldownModifier(
                            SimpleCooldownModifier::AddCharge,
                            1,
                        )),
                        DragableCard::CooldownModifier(CooldownModifier::SimpleCooldownModifier(
                            SimpleCooldownModifier::AddCooldown,
                            1,
                        )),
                        DragableCard::CooldownModifier(
                            CooldownModifier::SignedSimpleCooldownModifier(
                                SignedSimpleCooldownModifier::DecreaseCooldown,
                                1,
                            ),
                        ),
                        DragableCard::CooldownModifier(CooldownModifier::Reloading),
                    ],
                    PaletteState::StatusEffects => vec![
                        DragableCard::StatusEffect(StatusEffect::SimpleStatusEffect(
                            SimpleStatusEffectType::DamageOverTime,
                            1,
                        )),
                        DragableCard::StatusEffect(StatusEffect::SimpleStatusEffect(
                            SimpleStatusEffectType::IncreaseDamageTaken,
                            1,
                        )),
                        DragableCard::StatusEffect(StatusEffect::SimpleStatusEffect(
                            SimpleStatusEffectType::IncreaseGravity(DirectionCard::None),
                            1,
                        )),
                        DragableCard::StatusEffect(StatusEffect::SimpleStatusEffect(
                            SimpleStatusEffectType::Speed,
                            1,
                        )),
                        DragableCard::StatusEffect(StatusEffect::SimpleStatusEffect(
                            SimpleStatusEffectType::Overheal,
                            1,
                        )),
                        DragableCard::StatusEffect(StatusEffect::SimpleStatusEffect(
                            SimpleStatusEffectType::Grow,
                            1,
                        )),
                        DragableCard::StatusEffect(StatusEffect::SimpleStatusEffect(
                            SimpleStatusEffectType::IncreaseMaxHealth,
                            1,
                        )),
                        DragableCard::StatusEffect(StatusEffect::Invincibility),
                        DragableCard::StatusEffect(StatusEffect::Trapped),
                        DragableCard::StatusEffect(StatusEffect::Lockout),
                        DragableCard::StatusEffect(StatusEffect::Stun),
                        DragableCard::StatusEffect(StatusEffect::OnHit(Box::new(BaseCard::None))),
                    ],
                    PaletteState::Materials => vec![
                        DragableCard::BaseCard(BaseCard::CreateMaterial(VoxelMaterial::Grass)),
                        DragableCard::BaseCard(BaseCard::CreateMaterial(VoxelMaterial::Dirt)),
                        DragableCard::BaseCard(BaseCard::CreateMaterial(VoxelMaterial::Stone)),
                        DragableCard::BaseCard(BaseCard::CreateMaterial(VoxelMaterial::Ice)),
                        DragableCard::BaseCard(BaseCard::CreateMaterial(VoxelMaterial::Water)),
                    ],
                    PaletteState::Directions => vec![
                        DragableCard::Direction(DirectionCard::Up),
                        DragableCard::Direction(DirectionCard::Forward),
                        DragableCard::Direction(DirectionCard::Movement),
                    ],
                    PaletteState::Dock => gui_state.dock_cards.clone(),
                });

                let palette_height = ui
                    .scope(|ui| {
                        ui.visuals_mut().override_text_color = Some(Color32::WHITE);
                        palette_card.draw(
                            ui,
                            &mut vec![0].into(),
                            &mut source_path,
                            &mut dest_path,
                            &mut modify_path,
                        );
                    })
                    .response
                    .rect
                    .height();
                let scroll_area = Rect::from_min_max(
                    menu_size.min + vec2(0.0, palette_height + selector_height),
                    menu_size.max,
                );
                ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .scroll_bar_visibility(
                        egui::scroll_area::ScrollBarVisibility::VisibleWhenNeeded,
                    )
                    .drag_to_scroll(false)
                    .show(ui, |ui| {
                        ui.with_layout(egui::Layout::top_down(egui::Align::LEFT), |ui| {
                            ui.set_clip_rect(scroll_area);
                            let total_impact = gui_state.gui_deck.get_total_impact();

                            if gui_state.cooldown_cache_refresh_delay <= 0.0 {
                                for cooldown in gui_state.gui_deck.cooldowns.iter_mut() {
                                    if cooldown.generate_cooldown_cache() {
                                        gui_state.cooldown_cache_refresh_delay = 0.5;
                                    }
                                }
                            }
                            ui.horizontal_top(|ui| {
                                gui_state.gui_deck.passive.draw(
                                    ui,
                                    &mut vec![1].into(),
                                    &mut source_path,
                                    &mut dest_path,
                                    &mut modify_path,
                                );
                            });

                            for (ability_idx, cooldown) in
                                gui_state.gui_deck.cooldowns.iter_mut().enumerate()
                            {
                                if cooldown.cooldown_value.is_none() {
                                    cooldown.cooldown_value =
                                        Some(cooldown.get_cooldown_recovery(total_impact));
                                }
                                ui.horizontal_top(|ui| {
                                    cooldown.draw(
                                        ui,
                                        &mut vec![ability_idx as u32 + 2].into(),
                                        &mut source_path,
                                        &mut dest_path,
                                        &mut modify_path,
                                    );
                                });
                            }

                            if ui
                                .button("Add Cooldown")
                                .on_hover_text("Add a new cooldown")
                                .clicked()
                            {
                                gui_state.gui_deck.cooldowns.push(Cooldown::empty());
                            }

                            if let Some((mut modify_path, modification_type)) = modify_path {
                                for cooldown in gui_state.gui_deck.cooldowns.iter_mut() {
                                    cooldown.cooldown_value = None;
                                }
                                let modify_action_idx = modify_path.pop_front().unwrap() as usize;
                                if modify_action_idx == 1 {
                                    gui_state.gui_deck.passive.modify_from_path(
                                        &mut modify_path.clone(),
                                        modification_type,
                                    );
                                } else if modify_path.is_empty() {
                                    if matches!(modification_type, ModificationType::Remove) {
                                        gui_state.gui_deck.cooldowns.remove(modify_action_idx - 2);
                                    }
                                } else if modify_action_idx > 1 {
                                    gui_state.gui_deck.cooldowns[modify_action_idx - 2]
                                        .modify_from_path(
                                            &mut modify_path.clone(),
                                            modification_type,
                                        );
                                }
                            }
                            if let Some((source_path, source_type)) = source_path.as_mut() {
                                if let Some((drop_path, drop_type)) = dest_path.as_mut() {
                                    for cooldown in gui_state.gui_deck.cooldowns.iter_mut() {
                                        cooldown.cooldown_value = None;
                                    }
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
                                        } else if source_action_idx == 1 {
                                            gui_state
                                                .gui_deck
                                                .passive
                                                .take_from_path(&mut source_path.clone())
                                        } else {
                                            gui_state.gui_deck.cooldowns[source_action_idx - 2]
                                                .take_from_path(&mut source_path.clone())
                                        };
                                        if drop_action_idx == 1 {
                                            gui_state
                                                .gui_deck
                                                .passive
                                                .insert_to_path(drop_path, item);
                                        } else if drop_action_idx > 1 {
                                            gui_state.gui_deck.cooldowns[drop_action_idx - 2]
                                                .insert_to_path(drop_path, item);
                                        } else if matches!(
                                            gui_state.palette_state,
                                            PaletteState::Dock
                                        ) {
                                            gui_state.dock_cards.push(item);
                                        }
                                        if source_action_idx == 1 {
                                            gui_state.gui_deck.passive.cleanup(source_path);
                                        } else if source_action_idx > 1 {
                                            gui_state.gui_deck.cooldowns[source_action_idx - 2]
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
                    ReferencedStatusEffect::DamageOverTime(stacks) => {
                        format!("Damage Over Time {}", stacks)
                    }
                    ReferencedStatusEffect::Speed(stacks) => format!("Speed {}", stacks),
                    ReferencedStatusEffect::IncreaseDamageTaken(stacks) => {
                        format!("Increase Damage Taken {}", stacks)
                    }
                    ReferencedStatusEffect::IncreaseGravity(direction, stacks) => {
                        format!("Increase Gravity {} {}", direction, stacks)
                    }
                    ReferencedStatusEffect::Overheal(stacks) => format!("Overheal {}", stacks),
                    ReferencedStatusEffect::Grow(stacks) => format!("Grow {}", stacks),
                    ReferencedStatusEffect::IncreaseMaxHealth(stacks) => {
                        format!("Increase Max Health {}", stacks)
                    }
                    ReferencedStatusEffect::Invincibility => "Invincibility".to_string(),
                    ReferencedStatusEffect::Trapped => "Trapped".to_string(),
                    ReferencedStatusEffect::Lockout => "Lockout".to_string(),
                    ReferencedStatusEffect::OnHit(_) => "On Player Hit".to_string(),
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
