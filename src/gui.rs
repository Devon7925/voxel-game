use egui_winit_vulkano::egui::{
    self, epaint, Align, Align2, Color32, CursorIcon, FontId, Id, InnerResponse, LayerId, Order,
    Rect, Sense, Shape, Stroke, TextFormat, Ui,
};

use crate::{
    card_system::{BaseCard, Effect, ProjectileModifier},
    rollback_manager::PlayerAbility,
};

pub enum GuiElement {
    MainMenu,
    CardEditor,
}

pub struct GuiState {
    pub menu_stack: Vec<GuiElement>,
    pub gui_cards: Vec<BaseCard>,
    pub should_exit: bool,
}

fn cooldown_ui(ui: &mut egui::Ui, button: &str, ability: &PlayerAbility) -> egui::Response {
    let desired_size = ui.spacing().interact_size.y * egui::vec2(3.0, 3.0);
    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());

    if ui.is_rect_visible(rect) {
        let font = egui::FontId::proportional(24.0);
        if ability.cooldown > 0.0 {
            ui.painter().rect_filled(rect, 0.0, Color32::DARK_GRAY);
            ui.painter().text(
                rect.center(),
                Align2::CENTER_CENTER,
                format!("{}", ability.cooldown.ceil() as i32),
                font,
                Color32::WHITE,
            );
        } else {
            ui.painter().rect_filled(rect, 0.0, Color32::LIGHT_GRAY);
            ui.painter().text(
                rect.center(),
                Align2::CENTER_CENTER,
                format!("{}", button),
                font,
                Color32::BLACK,
            );
        }
    }

    response
}

pub fn cooldown<'a>(button: &'a str, ability: &'a PlayerAbility) -> impl egui::Widget + 'a {
    move |ui: &mut egui::Ui| cooldown_ui(ui, button, ability)
}

pub fn drag_source(ui: &mut Ui, id: Id, body: impl FnOnce(&mut Ui)) {
    let is_being_dragged = ui.memory(|mem| mem.is_being_dragged(id));

    if !is_being_dragged {
        let response = ui.scope(body).response;

        // Check for drags:
        let response = ui.interact(response.rect, id, Sense::drag());
        if response.hovered() {
            ui.ctx().set_cursor_icon(CursorIcon::Grab);
        }
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

    let mut fill = style.bg_fill;
    let mut stroke = style.bg_stroke;
    if is_being_dragged && !can_accept_what_is_being_dragged {
        fill = ui.visuals().gray_out(fill);
        stroke.color = ui.visuals().gray_out(stroke.color);
    }

    ui.painter().set(
        where_to_put_background,
        epaint::RectShape {
            rounding: style.rounding,
            fill,
            stroke,
            rect,
        },
    );

    InnerResponse::new(ret, response)
}

const CARD_UI_SPACING: f32 = 3.0;
const CARD_UI_ROUNDING: f32 = 3.0;
pub fn draw_base_card(ui: &mut Ui, card: &BaseCard) {
    ui.allocate_ui_with_layout(
        egui::vec2(1000.0, 0.0),
        egui::Layout::top_down(egui::Align::LEFT),
        |ui| {
            ui.add_space(CARD_UI_SPACING);
            match card {
                BaseCard::Projectile(modifiers) => {
                    ui.horizontal(|ui| {
                        ui.add_space(CARD_UI_SPACING);
                        ui.label("Projectile");
                        for modifier in modifiers {
                            match modifier {
                                ProjectileModifier::Gravity(v) => {
                                    add_basic_modifer(ui, "Gravity", *v)
                                }
                                ProjectileModifier::Health(v) => {
                                    add_basic_modifer(ui, "Health", *v)
                                }
                                ProjectileModifier::Height(v) => {
                                    add_basic_modifer(ui, "Height", *v)
                                }
                                ProjectileModifier::Length(v) => {
                                    add_basic_modifer(ui, "Length", *v)
                                }
                                ProjectileModifier::Lifetime(v) => {
                                    add_basic_modifer(ui, "Lifetime", *v)
                                }
                                ProjectileModifier::NoEnemyFire => {
                                    ui.label("No Enemy Fire");
                                }
                                ProjectileModifier::NoFriendlyFire => {
                                    ui.label("No Friendly Fire");
                                }
                                ProjectileModifier::Speed(v) => add_basic_modifer(ui, "Speed", *v),
                                ProjectileModifier::Width(v) => add_basic_modifer(ui, "Width", *v),
                                _ => {}
                            }
                        }
                        ui.add_space(CARD_UI_SPACING);
                    });

                    for modifier in modifiers {
                        ui.horizontal_top(|ui| {
                            ui.add_space(CARD_UI_SPACING);
                            match modifier {
                                ProjectileModifier::OnExpiry(base_card) => {
                                    ui.label("On Expiry");
                                    draw_base_card(ui, base_card)
                                }
                                ProjectileModifier::OnHit(base_card) => {
                                    ui.label("On Hit");
                                    draw_base_card(ui, base_card)
                                }
                                _ => {}
                            }
                            ui.add_space(CARD_UI_SPACING);
                        });
                    }
                    ui.add_space(CARD_UI_SPACING);
                    ui.painter().rect_stroke(
                        ui.min_rect(),
                        CARD_UI_ROUNDING,
                        Stroke::new(1.0, Color32::WHITE),
                    );
                }
                BaseCard::MultiCast(cards) => {
                    ui.label("Multi");
                    for card in cards {
                        draw_base_card(ui, card);
                    }
                    ui.painter().rect_stroke(
                        ui.min_rect(),
                        CARD_UI_ROUNDING,
                        Stroke::new(1.0, Color32::YELLOW),
                    );
                }
                BaseCard::CreateMaterial(mat) => {
                    ui.horizontal(|ui| {
                        ui.add_space(CARD_UI_SPACING);
                        ui.label("Material");
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
                        ui.label("Effect");
                        match effect {
                            Effect::Damage(v) => add_basic_modifer(ui, "Damage", *v),
                            Effect::Knockback(v) => add_basic_modifer(ui, "Knockback", *v),
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
            }
        },
    );
}

pub fn add_basic_modifer(ui: &mut Ui, name: &str, count: impl std::fmt::Display) {
    use egui::text::LayoutJob;
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
