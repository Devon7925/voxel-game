use cgmath::Point3;
use egui_winit_vulkano::egui::Ui;
use voxel_shared::GameModeSettings;

use crate::{game_manager::Game, gui::EditMode, rollback_manager::{Entity, PlayerSim}, RESPAWN_TIME};

pub trait GameMode {
    fn are_friends(&self, player1: u32, player2: u32, entities: &Vec<Entity>) -> bool;
    fn spawn_location(&self, entity: &Entity) -> Point3<f32>;
    fn get_initial_center(&self) -> Point3<f32>;
    fn fixed_center(&self) -> bool;
    fn deck_swapping(&self, player: &Entity) -> EditMode;
    fn has_mode_gui(&self) -> bool { false }
    fn mode_gui(&mut self, ui: &mut Ui, sim: &mut Box<dyn PlayerSim>) {}
    fn send_action(&self, player_idx: usize, action: String, entities: &mut Vec<Entity>) {}
}

struct PracticeRangeMode {
    spawn_location: Point3<f32>,
}

struct ExplorerMode {
    spawn_location: Point3<f32>,
}

struct FFAMode;
struct ControlMode;

pub fn game_mode_from_type(game_mode: GameModeSettings) -> Box<dyn GameMode> {
    match game_mode {
        GameModeSettings::PracticeRange { spawn_location } => {
            Box::new(PracticeRangeMode { spawn_location })
        }
        GameModeSettings::Explorer { spawn_location } => Box::new(ExplorerMode { spawn_location }),
        GameModeSettings::FFA => Box::new(FFAMode),
        GameModeSettings::Control => Box::new(ControlMode),
    }
}

impl GameMode for PracticeRangeMode {
    fn are_friends(&self, _player1: u32, _player2: u32, entities: &Vec<Entity>) -> bool {
        true
    }

    fn spawn_location(&self, _entity: &Entity) -> Point3<f32> {
        self.spawn_location
    }

    fn fixed_center(&self) -> bool {
        true
    }

    fn get_initial_center(&self) -> Point3<f32> {
        self.spawn_location
    }

    fn deck_swapping(&self, _player: &Entity) -> EditMode {
        EditMode::FullEditing
    }
}

impl GameMode for ExplorerMode {
    fn are_friends(&self, _player1: u32, _player2: u32, entities: &Vec<Entity>) -> bool {
        true
    }

    fn spawn_location(&self, _entity: &Entity) -> Point3<f32> {
        self.spawn_location
    }

    fn fixed_center(&self) -> bool {
        false
    }

    fn get_initial_center(&self) -> Point3<f32> {
        self.spawn_location
    }
    
    fn deck_swapping(&self, _player: &Entity) -> EditMode {
        EditMode::FullEditing
    }
}

impl GameMode for FFAMode {
    fn are_friends(&self, player1: u32, player2: u32, entities: &Vec<Entity>) -> bool {
        player1 == player2
    }

    fn spawn_location(&self, _entity: &Entity) -> Point3<f32> {
        todo!()
    }

    fn fixed_center(&self) -> bool {
        false
    }

    fn get_initial_center(&self) -> Point3<f32> {
        todo!()
    }
    
    fn deck_swapping(&self, _player: &Entity) -> EditMode {
        EditMode::Readonly
    }
}

const SPAWN_ROOM_OFFSET: i32 = 150;
impl GameMode for ControlMode {
    fn are_friends(&self, player1: u32, player2: u32, entities: &Vec<Entity>) -> bool {
        let player1_team = entities.get(player1 as usize).map(|p| p.gamemode_data.get(0).unwrap_or(&0)).unwrap_or(&0);
        let player2_team = entities.get(player2 as usize).map(|p| p.gamemode_data.get(0).unwrap_or(&0)).unwrap_or(&0);
        player1_team == player2_team
    }

    fn spawn_location(&self, entity: &Entity) -> Point3<f32> {
        let player_team = entity.gamemode_data.get(0).unwrap_or(&0);
        match player_team {
            0 => Point3::new(10000.0, 1805.0, 10000.0),
            1 => Point3::new(10000.0 - SPAWN_ROOM_OFFSET as f32 - 2.0, 1805.0, 10000.0),
            2 => Point3::new(10000.0 + SPAWN_ROOM_OFFSET as f32 + 2.0, 1805.0, 10000.0),
            _ => panic!("Invalid Team"),
        }
    }

    fn fixed_center(&self) -> bool {
        true
    }

    fn get_initial_center(&self) -> Point3<f32> {
        Point3::new(10000.0, 1836.0, 10000.0)
    }
    
    fn deck_swapping(&self, entity: &Entity) -> EditMode {
        let team = entity.gamemode_data.get(0).unwrap_or(&0);
        match team {
            0 => EditMode::Readonly,
            1 => if entity.pos.x - 10000.0 < -SPAWN_ROOM_OFFSET as f32 {
                EditMode::FullEditing
            } else {
                EditMode::Readonly
            },
            2 => if entity.pos.x - 10000.0 > SPAWN_ROOM_OFFSET as f32 {
                EditMode::FullEditing
            } else {
                EditMode::Readonly
            },
            _ => panic!("Invalid Team"),
        }
    }

    fn send_action(&self, player_idx: usize, action: String, entities: &mut Vec<Entity>) {
        let Some(entity) = entities.get_mut(player_idx) else {
            panic!("Could not get player at index {}", player_idx);
        };
        if let Some(first) = entity.gamemode_data.get_mut(0) {
            *first = action.parse::<u32>().unwrap();
        } else {
            entity.gamemode_data.push(action.parse::<u32>().unwrap());
        }
        entity.respawn_timer = RESPAWN_TIME;
    }

    fn has_mode_gui(&self) -> bool {
        true
    }
    fn mode_gui(&mut self, ui: &mut Ui, sim: &mut Box<dyn PlayerSim>) {
        if ui.button("Switch to Spectators").clicked() {
            sim.send_gamemode_packet("0".to_string());
        };
        if ui.button("Switch to Team 1").clicked() {
            sim.send_gamemode_packet("1".to_string());
        };
        if ui.button("Switch to Team 2").clicked() {
            sim.send_gamemode_packet("2".to_string());
        };
    }
}
