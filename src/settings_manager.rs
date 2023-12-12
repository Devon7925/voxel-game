use std::fmt::Display;

use serde::{Deserialize, Serialize};
use winit::event::{MouseButton, VirtualKeyCode};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Settings {
    pub local_url: String,
    pub remote_url: String,
    pub card_file: String,
    pub fullscreen_toggle: VirtualKeyCode,
    pub movement_controls: ControlSettings,
    pub graphics_settings: GraphicsSettings,
    pub replay_settings: ReplaySettings,
    pub delta_time: f32,
    pub do_profiling: bool,
    pub crash_log: String,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub enum Control {
    Key(VirtualKeyCode),
    Mouse(MouseButton),
}

impl Display for Control {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Control::Key(key) => write!(f, "{:?}", key),
            Control::Mouse(MouseButton::Left) => write!(f, "↖"),
            Control::Mouse(MouseButton::Middle) => write!(f, "⬆"),
            Control::Mouse(MouseButton::Right) => write!(f, "↗"),
            Control::Mouse(button) => write!(f, "{:?}", button),
        }
    }
}

impl Settings {
    pub fn from_string(yaml_string: &str) -> Self {
        serde_yaml::from_str(yaml_string).unwrap()
    }

    pub fn to_string(&self) -> String {
        serde_yaml::to_string(self).unwrap()
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ControlSettings {
    pub forward: Control,
    pub backward: Control,
    pub left: Control,
    pub right: Control,
    pub jump: Control,
    pub crouch: Control,
    pub sensitivity: f32,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct GraphicsSettings {
    pub primary_ray_dist: u32,
    pub transparency_ray_dist: u32,
    pub shadow_ray_dist: u32,
    pub transparent_shadow_ray_dist: u32,
    pub ao_ray_dist: u32,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ReplaySettings {
    pub replay_folder: String,
    pub record_replay: bool,
}
