use egui_winit_vulkano::egui::{self, Color32, Align2};

use crate::rollback_manager::PlayerAbility;

fn cooldown_ui(ui: &mut egui::Ui, button: &str, ability: &PlayerAbility) -> egui::Response {
    let desired_size = ui.spacing().interact_size.y * egui::vec2(3.0, 3.0);
    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());

    if ui.is_rect_visible(rect) {
        let font = egui::FontId::proportional(24.0);
        if ability.cooldown > 0.0 {
            ui.painter().rect_filled(
                rect,
                0.0,
                Color32::DARK_GRAY,
            );
            ui.painter().text(rect.center(), Align2::CENTER_CENTER, format!("{}", ability.cooldown.ceil() as i32), font, Color32::WHITE);
        } else {
            ui.painter().rect_filled(
                rect,
                0.0,
                Color32::LIGHT_GRAY,
            );
            ui.painter().text(rect.center(), Align2::CENTER_CENTER, format!("{}", button), font, Color32::BLACK);
        }
    }

    response
}

pub fn cooldown<'a>(button: &'a str, ability: &'a PlayerAbility) -> impl egui::Widget + 'a {
    move |ui: &mut egui::Ui| cooldown_ui(ui, button, ability)
}