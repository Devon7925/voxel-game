use serde::{Deserialize, Serialize};
use winit::event::{MouseButton, VirtualKeyCode};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Settings {
    pub local_url: String,
    pub remote_url: String,
    pub is_remote: bool,
    pub player_count: u32,
    pub fullscreen_toggle: VirtualKeyCode,
    pub movement_controls: ControlSettings,
    pub ability_controls: Vec<Control>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum Control {
    Key(VirtualKeyCode),
    Mouse(MouseButton),
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
    pub sprint: Control,
    pub sensitivity: f32,
}
