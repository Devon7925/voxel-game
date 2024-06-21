use cgmath::Point3;
use voxel_shared::GameModeSettings;

use crate::{gui::EditMode, rollback_manager::Entity};

pub trait GameMode {
    fn are_friends(&self, player1: usize, player2: usize) -> bool;
    fn spawn_location(&self, player_idx: usize) -> Point3<f32>;
    fn get_initial_center(&self) -> Point3<f32>;
    fn fixed_center(&self) -> bool;
    fn deck_swapping(&self, player: &Entity) -> EditMode;
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
    fn are_friends(&self, _player1: usize, _player2: usize) -> bool {
        true
    }

    fn spawn_location(&self, _player_idx: usize) -> Point3<f32> {
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
    fn are_friends(&self, _player1: usize, _player2: usize) -> bool {
        true
    }

    fn spawn_location(&self, _player_idx: usize) -> Point3<f32> {
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
    fn are_friends(&self, player1: usize, player2: usize) -> bool {
        player1 == player2
    }

    fn spawn_location(&self, _player_idx: usize) -> Point3<f32> {
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

impl GameMode for ControlMode {
    fn are_friends(&self, _player1: usize, _player2: usize) -> bool {
        todo!()
    }

    fn spawn_location(&self, _player_idx: usize) -> Point3<f32> {
        todo!()
    }

    fn fixed_center(&self) -> bool {
        true
    }

    fn get_initial_center(&self) -> Point3<f32> {
        todo!()
    }
    
    fn deck_swapping(&self, player: &Entity) -> EditMode {
        EditMode::Readonly
    }
}
