use serde::{Deserialize, Serialize};
use winit::event::VirtualKeyCode;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Settings {
    pub local_url: String,
    pub remote_url: String,
    pub is_remote: bool,
    pub player_count: u32,
    pub movement_controls: ControlSettings,
    pub ability_controls: Vec<VirtualKeyCode>,
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
    pub forward: VirtualKeyCode,
    pub back: VirtualKeyCode,
    pub left: VirtualKeyCode,
    pub right: VirtualKeyCode,
    pub jump: VirtualKeyCode,
    pub crouch: VirtualKeyCode,
    pub sprint: VirtualKeyCode,
    pub sensitivity: f32,
}