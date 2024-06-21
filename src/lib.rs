use cgmath::{Point3, Vector3};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum WorldGenSettings {
    Normal,
    Control,
}
impl WorldGenSettings {
    pub fn get_name(&self) -> &str {
        match self {
            WorldGenSettings::Normal => "Normal",
            WorldGenSettings::Control => "Control",
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum GameModeSettings {
    PracticeRange{spawn_location: Point3<f32>},
    Explorer{spawn_location: Point3<f32>},
    FFA,
    Control,
}
impl GameModeSettings {
    pub fn get_name(&self) -> &str {
        match self {
            GameModeSettings::PracticeRange{..} => "Practice Range",
            GameModeSettings::Explorer{..} => "Explorer",
            GameModeSettings::FFA => "FFA",
            GameModeSettings::Control => "Control",
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GameSettings {
    pub name: String,
    pub delta_time: f32,
    pub is_remote: bool,
    pub rollback_buffer_size: u32,
    pub player_count: u32,
    pub render_size: Vector3<u32>,
    pub max_loaded_chunks: u32,
    pub max_worldgen_rate: u32,
    pub max_update_rate: u32,
    pub world_gen: WorldGenSettings,
    pub game_mode: GameModeSettings,
}

#[derive(Debug, Deserialize, Serialize, Default, Clone, PartialEq, Eq, Hash)]
pub struct RoomId(pub String);

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Lobby {
    pub name: String,
    pub player_count: u32,
    pub lobby_id: RoomId,
    pub settings: GameSettings,
}
