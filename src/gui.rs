use std::collections::VecDeque;

use egui_winit_vulkano::egui::{
    self, epaint, text::LayoutJob, Align, Align2, Color32, CursorIcon, FontId, Id, InnerResponse,
    LayerId, Order, Rect, Rounding, Sense, Shape, Stroke, TextFormat, Ui,
};

use crate::{
    card_system::{
        BaseCard, Cooldown, CooldownModifier, Effect, MultiCastModifier, ProjectileModifier,
        ProjectileModifierType, StatusEffect,
    },
    rollback_manager::PlayerAbility,
};

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum GuiElement {
    EscMenu,
    CardEditor,
    MainMenu,
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum PaletteState {
    ProjectileModifiers,
    BaseCards,
    AdvancedProjectileModifiers,
    MultiCastModifiers,
}

pub struct GuiState {
    pub menu_stack: Vec<GuiElement>,
    pub gui_cards: Vec<Cooldown>,
    pub in_game: bool,
    pub should_exit: bool,
    pub palette_state: PaletteState,
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

pub fn drag_source(ui: &mut Ui, id: Id, body: impl FnOnce(&mut Ui)) {
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

pub enum DragableType {
    ProjectileModifier,
    MultiCastModifier,
    CooldownModifier,
    BaseCard,
}

pub enum DropableType {
    MultiCastBaseCard,
    BaseNone,
    BaseProjectile,
    Cooldown,
}

pub fn is_valid_drag(from: &DragableType, to: &DropableType) -> bool {
    match (from, to) {
        (DragableType::ProjectileModifier, DropableType::BaseProjectile) => true,
        (DragableType::MultiCastModifier, DropableType::MultiCastBaseCard) => true,
        (DragableType::BaseCard, DropableType::MultiCastBaseCard) => true,
        (DragableType::BaseCard, DropableType::BaseNone) => true,
        (DragableType::CooldownModifier, DropableType::Cooldown) => true,
        (DragableType::BaseCard, DropableType::Cooldown) => true,
        _ => false,
    }
}

pub fn draw_cooldown(
    ui: &mut Ui,
    cooldown: &Cooldown,
    path: &mut VecDeque<u32>,
    source_path: &mut Option<(VecDeque<u32>, DragableType)>,
    drop_path: &mut Option<(VecDeque<u32>, DropableType)>,
) {
    let can_accept_what_is_being_dragged = true;
    let id_source = "my_drag_and_drop_demo";
    let Cooldown {
        abilities,
        modifiers,
    } = cooldown;
    let response = drop_target(ui, can_accept_what_is_being_dragged, |ui| {
        ui.horizontal(|ui| {
            ui.add_space(CARD_UI_SPACING);
            ui.label("Ability");
            path.push_back(0);
            for (mod_idx, modifier) in modifiers.iter().enumerate() {
                path.push_back(mod_idx as u32);
                let item_id = egui::Id::new(id_source).with(path.clone());
                drag_source(ui, item_id, |ui| match modifier {
                    CooldownModifier::AddCharge(v) => add_basic_modifer(ui, "Add charge", *v),
                });
                path.pop_back();
            }
            path.pop_back();
            ui.add_space(CARD_UI_SPACING);
        });
        path.push_back(1);
        ui.vertical(|ui| {
            for (ability_idx, ability) in abilities.iter().enumerate() {
                path.push_back(ability_idx as u32);
                draw_base_card(ui, &ability.card, path, source_path, drop_path);
                path.pop_back();
            }
        });
        path.pop_back();
        ui.painter().rect_stroke(
            ui.min_rect(),
            CARD_UI_ROUNDING,
            Stroke::new(1.0, Color32::YELLOW),
        );

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
    })
    .response;

    if drop_path.is_none() {
        let is_being_dragged = ui.memory(|mem| mem.is_anything_being_dragged());
        if is_being_dragged && can_accept_what_is_being_dragged && response.hovered() {
            *drop_path = Some((path.clone(), DropableType::Cooldown));
        }
    }
}

const CARD_UI_SPACING: f32 = 3.0;
const CARD_UI_ROUNDING: f32 = 3.0;
pub fn draw_base_card(
    ui: &mut Ui,
    card: &BaseCard,
    path: &mut VecDeque<u32>,
    source_path: &mut Option<(VecDeque<u32>, DragableType)>,
    dest_path: &mut Option<(VecDeque<u32>, DropableType)>,
) {
    let id_source = "my_drag_and_drop_demo";

    let item_id = egui::Id::new(id_source).with(path.clone());
    let can_accept_what_is_being_dragged = true; // We accept anything being dragged (for now) ¯\_(ツ)_/¯
    drag_source(ui, item_id, |ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(1000.0, 0.0),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| {
                ui.add_space(CARD_UI_SPACING);
                match card {
                    BaseCard::Projectile(modifiers) => {
                        ui.visuals_mut().widgets.active.rounding = Rounding::from(CARD_UI_ROUNDING);
                        ui.visuals_mut().widgets.inactive.bg_stroke =
                            Stroke::new(0.5, Color32::WHITE);
                        let response = drop_target(ui, can_accept_what_is_being_dragged, |ui| {
                            ui.horizontal(|ui| {
                                ui.add_space(CARD_UI_SPACING);
                                ui.label("Create Projectile");
                                for (modifier_idx, modifier) in modifiers.iter().enumerate() {
                                    path.push_back(modifier_idx as u32);
                                    let item_id = egui::Id::new(id_source).with(path.clone());

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
                                            )
                                        }
                                        ProjectileModifier::NoEnemyFire => {
                                            add_hoverable_basic_modifer(
                                                ui,
                                                item_id,
                                                "No Enemy Fire",
                                                "",
                                                modifier.get_hover_text(),
                                            )
                                        }
                                        ProjectileModifier::FriendlyFire => {
                                            add_hoverable_basic_modifer(
                                                ui,
                                                item_id,
                                                "Friendly Fire",
                                                "",
                                                modifier.get_hover_text(),
                                            )
                                        }
                                        ProjectileModifier::LockToOwner => {
                                            add_hoverable_basic_modifer(
                                                ui,
                                                item_id,
                                                "Lock To Owner",
                                                "",
                                                modifier.get_hover_text(),
                                            )
                                        }
                                        ProjectileModifier::PiercePlayers => {
                                            add_hoverable_basic_modifer(
                                                ui,
                                                item_id,
                                                "Pierce Players",
                                                "",
                                                modifier.get_hover_text(),
                                            )
                                        }
                                        ProjectileModifier::WallBounce => {
                                            add_hoverable_basic_modifer(
                                                ui,
                                                item_id,
                                                "Wall Bounce",
                                                "",
                                                modifier.get_hover_text(),
                                            )
                                        }
                                        ProjectileModifier::OnExpiry(_)
                                        | ProjectileModifier::OnHit(_)
                                        | ProjectileModifier::OnTrigger(_, _) => {}
                                        ProjectileModifier::Trail(_, _) => {}
                                    }

                                    path.pop_back();
                                }
                                ui.add_space(CARD_UI_SPACING);
                            });

                            for (modifier_idx, modifier) in modifiers.iter().enumerate() {
                                path.push_back(modifier_idx as u32);
                                let item_id = egui::Id::new(id_source).with(path.clone());
                                ui.horizontal_top(|ui| {
                                    ui.add_space(CARD_UI_SPACING);
                                    match modifier {
                                        ProjectileModifier::OnHit(base_card) => {
                                            drag_source(ui, item_id, |ui| {
                                                ui.label("On Hit");
                                                path.push_back(0);
                                                draw_base_card(
                                                    ui,
                                                    base_card,
                                                    path,
                                                    source_path,
                                                    dest_path,
                                                );
                                                path.pop_back();
                                            });
                                        }
                                        ProjectileModifier::OnExpiry(base_card) => {
                                            drag_source(ui, item_id, |ui| {
                                                ui.label("On Expiry");
                                                path.push_back(0);
                                                draw_base_card(
                                                    ui,
                                                    base_card,
                                                    path,
                                                    source_path,
                                                    dest_path,
                                                );
                                                path.pop_back();
                                            });
                                        }
                                        ProjectileModifier::OnTrigger(id, base_card) => {
                                            drag_source(ui, item_id, |ui| {
                                                ui.label(format!("On Trigger {}", id));
                                                path.push_back(0);
                                                draw_base_card(
                                                    ui,
                                                    base_card,
                                                    path,
                                                    source_path,
                                                    dest_path,
                                                );
                                                path.pop_back();
                                            });
                                        }
                                        ProjectileModifier::Trail(frequency, base_card) => {
                                            drag_source(ui, item_id, |ui| {
                                                let mut job = LayoutJob::default();
                                                job.append(
                                                    "Trail",
                                                    0.0,
                                                    TextFormat {
                                                        ..Default::default()
                                                    },
                                                );
                                                job.append(
                                                    format!("{}", frequency).as_str(),
                                                    0.0,
                                                    TextFormat {
                                                        font_id: FontId::proportional(7.0),
                                                        valign: Align::TOP,
                                                        ..Default::default()
                                                    },
                                                );
                                                ui.label(job);
                                                path.push_back(0);
                                                draw_base_card(
                                                    ui,
                                                    base_card,
                                                    path,
                                                    source_path,
                                                    dest_path,
                                                );
                                                path.pop_back();
                                            });
                                        }
                                        _ => {}
                                    }
                                    ui.add_space(CARD_UI_SPACING);
                                });
                                if ui.memory(|mem| mem.is_being_dragged(item_id)) {
                                    *source_path =
                                        Some((path.clone(), DragableType::ProjectileModifier));
                                }
                                path.pop_back();
                            }
                            for (modifier_idx, _modifier) in modifiers.iter().enumerate() {
                                path.push_back(modifier_idx as u32);
                                let item_id = egui::Id::new(id_source).with(path.clone());
                                if source_path.is_none()
                                    && ui.memory(|mem| mem.is_being_dragged(item_id))
                                {
                                    *source_path =
                                        Some((path.clone(), DragableType::ProjectileModifier));
                                }
                                path.pop_back();
                            }
                            ui.add_space(CARD_UI_SPACING);
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
                    }
                    BaseCard::MultiCast(cards, modifiers) => {
                        let response = drop_target(ui, can_accept_what_is_being_dragged, |ui| {
                            ui.horizontal(|ui| {
                                ui.add_space(CARD_UI_SPACING);
                                ui.label("Multicast");
                                path.push_back(0);
                                for (mod_idx, modifier) in modifiers.iter().enumerate() {
                                    path.push_back(mod_idx as u32);
                                    let item_id = egui::Id::new(id_source).with(path.clone());
                                    drag_source(ui, item_id, |ui| match modifier {
                                        MultiCastModifier::Spread(v) => {
                                            add_basic_modifer(ui, "Spread", *v)
                                        }
                                        MultiCastModifier::Duplication(v) => {
                                            add_basic_modifer(ui, "Duplication", *v)
                                        }
                                    });
                                    path.pop_back();
                                }
                                path.pop_back();
                                ui.add_space(CARD_UI_SPACING);
                            });
                            path.push_back(1);
                            for (card_idx, card) in cards.iter().enumerate() {
                                path.push_back(card_idx as u32);
                                draw_base_card(ui, card, path, source_path, dest_path);
                                path.pop_back();
                            }
                            path.pop_back();
                            ui.painter().rect_stroke(
                                ui.min_rect(),
                                CARD_UI_ROUNDING,
                                Stroke::new(1.0, Color32::YELLOW),
                            );

                            path.push_back(0);
                            for (modifier_idx, _modifier) in modifiers.iter().enumerate() {
                                path.push_back(modifier_idx as u32);
                                let item_id = egui::Id::new(id_source).with(path.clone());
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
                    }
                    BaseCard::CreateMaterial(mat) => {
                        ui.horizontal(|ui| {
                            ui.add_space(CARD_UI_SPACING);
                            ui.label("Create Material");
                            ui.label(format!("{:?}", mat));
                            ui.add_space(CARD_UI_SPACING);
                        });
                        ui.add_space(CARD_UI_SPACING);
                        ui.painter().rect_stroke(
                            ui.min_rect(),
                            CARD_UI_ROUNDING,
                            Stroke::new(1.0, Color32::BLUE),
                        );
                    }
                    BaseCard::Effect(effect) => {
                        ui.horizontal(|ui| {
                            ui.add_space(CARD_UI_SPACING);
                            ui.label("Apply Effect");
                            match effect {
                                Effect::Damage(v) => add_basic_modifer(ui, "Damage", *v),
                                Effect::Knockback(v) => add_basic_modifer(ui, "Knockback", *v),
                                Effect::Cleanse => add_basic_modifer(ui, "Cleanse", ""),
                                Effect::Teleport => add_basic_modifer(ui, "Teleport", ""),
                                Effect::StatusEffect(e, t) => {
                                    if let StatusEffect::OnHit(base_card) = e {
                                        ui.label("On Hit");
                                        draw_base_card(ui, base_card, path, source_path, dest_path)
                                    } else {
                                        let effect_name = match e {
                                            StatusEffect::DamageOverTime => "Damage Over Time",
                                            StatusEffect::HealOverTime => "Heal Over Time",
                                            StatusEffect::DecreaceDamageTaken => {
                                                "Decreace Damage Taken"
                                            }
                                            StatusEffect::IncreaceDamageTaken => {
                                                "Increace Damage Taken"
                                            }
                                            StatusEffect::Slow => "Slow",
                                            StatusEffect::Speed => "Speed Up",
                                            StatusEffect::DecreaceGravity => "Decreace Gravity",
                                            StatusEffect::IncreaceGravity => "Increace Gravity",
                                            StatusEffect::Overheal => "Overheal",
                                            StatusEffect::Invincibility => "Invincibility",
                                            StatusEffect::OnHit(_base_card) => {
                                                panic!("OnHit should be handled above")
                                            }
                                        };
                                        add_basic_modifer(ui, effect_name, *t)
                                    }
                                }
                            }
                            ui.add_space(CARD_UI_SPACING);
                        });
                        ui.add_space(CARD_UI_SPACING);
                        ui.painter().rect_stroke(
                            ui.min_rect(),
                            CARD_UI_ROUNDING,
                            Stroke::new(1.0, Color32::RED),
                        );
                    }
                    BaseCard::Trigger(id) => {
                        ui.horizontal(|ui| {
                            ui.add_space(CARD_UI_SPACING);
                            ui.label(format!("Trigger {}", id));
                            ui.add_space(CARD_UI_SPACING);
                        });
                        ui.add_space(CARD_UI_SPACING);
                        ui.painter().rect_stroke(
                            ui.min_rect(),
                            CARD_UI_ROUNDING,
                            Stroke::new(1.0, Color32::GOLD),
                        );
                    }
                    BaseCard::None => {
                        ui.visuals_mut().widgets.active.rounding = Rounding::from(CARD_UI_ROUNDING);
                        ui.visuals_mut().widgets.inactive.bg_stroke =
                            Stroke::new(1.0, Color32::GREEN);
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
                            if is_being_dragged
                                && can_accept_what_is_being_dragged
                                && response.hovered()
                            {
                                *dest_path = Some((path.clone(), DropableType::BaseNone));
                            }
                        }
                    }
                }
            },
        );
    });
    if source_path.is_none() && ui.memory(|mem| mem.is_being_dragged(item_id)) {
        *source_path = Some((path.clone(), DragableType::BaseCard));
    }
}

pub fn add_basic_modifer(ui: &mut Ui, name: &str, count: impl std::fmt::Display) {
    let mut job = LayoutJob::default();
    job.append(
        name,
        0.0,
        TextFormat {
            ..Default::default()
        },
    );
    job.append(
        format!("{}", count).as_str(),
        0.0,
        TextFormat {
            font_id: FontId::proportional(7.0),
            valign: Align::TOP,
            ..Default::default()
        },
    );
    ui.label(job);
}

pub fn add_hoverable_basic_modifer(
    ui: &mut Ui,
    item_id: Id,
    name: &str,
    count: impl std::fmt::Display,
    hover_text: String,
) {
    drag_source(ui, item_id, |ui| {
        let mut job = LayoutJob::default();
        job.append(
            name,
            0.0,
            TextFormat {
                ..Default::default()
            },
        );
        job.append(
            format!("{}", count).as_str(),
            0.0,
            TextFormat {
                font_id: FontId::proportional(7.0),
                valign: Align::TOP,
                ..Default::default()
            },
        );
        ui.label(job).on_hover_text(hover_text);
    });
}
