use std::ops::RangeInclusive;

use cgmath::Point3;
use egui_winit_vulkano::egui::{Slider, Ui};
use voxel_shared::GameModeSettings;

use crate::{
    card_system::{CardManager, Deck},
    cpu_simulation::Entity,
    gui::EditMode,
    rollback_manager::{EntityMetaData, PlayerSim},
    voxel_sim_manager::Projectile,
    RESPAWN_TIME,
};

pub trait GameMode {
    fn are_friends(&self, player1: u32, player2: u32, entities: &Vec<Entity>) -> bool;
    fn spawn_location(&self, entity: &Entity) -> Point3<f32>;

    fn player_mode(&self, entity: &Entity) -> PlayerMode {
        if entity.respawn_timer > 0.0 {
            PlayerMode::Spectator
        } else {
            PlayerMode::Normal
        }
    }

    fn get_initial_center(&self) -> Point3<f32>;
    fn fixed_center(&self) -> bool;
    fn deck_swapping(&self, player: &Entity) -> EditMode;
    fn has_mode_gui(&self) -> bool {
        false
    }
    fn mode_gui(&mut self, _ui: &mut Ui, _sim: &mut Box<dyn PlayerSim>) {}
    fn overlay(&self, _ui: &mut Ui, _sim: &Box<dyn PlayerSim>) {}
    fn send_action(&self, _player_idx: usize, _action: String, _entities: &mut Vec<Entity>) {}
    fn initialize(
        &mut self,
        _player_sim: &mut Box<dyn PlayerSim>,
        _card_manager: &mut CardManager,
    ) {
    }
    fn update(
        &mut self,
        _entities: &mut Vec<Entity>,
        _projectiles: &mut Vec<Projectile>,
        _delta_time: f32,
    ) {
    }
    fn cooldowns_reset_on_deck_swap(&self) -> bool {
        false
    }
}

pub enum PlayerMode {
    Normal,
    Spectator,
}

impl PlayerMode {
    pub fn has_entity_collison(&self) -> bool {
        match self {
            PlayerMode::Normal => true,
            PlayerMode::Spectator => false,
        }
    }

    pub fn has_world_collison(&self) -> bool {
        match self {
            PlayerMode::Normal => true,
            PlayerMode::Spectator => false,
        }
    }

    pub fn can_interact(&self) -> bool {
        match self {
            PlayerMode::Normal => true,
            PlayerMode::Spectator => false,
        }
    }
}

struct PracticeRangeMode {
    spawn_location: Point3<f32>,
}

struct ExplorerMode {
    spawn_location: Point3<f32>,
}

struct FFAMode;
struct ControlMode {
    team_1_score: f32,
    team_2_score: f32,
    capture_progress: f32,
}

impl Default for ControlMode {
    fn default() -> Self {
        Self {
            team_1_score: 0.0,
            team_2_score: 0.0,
            capture_progress: 0.0,
        }
    }
}

pub fn game_mode_from_type(game_mode: GameModeSettings) -> Box<dyn GameMode> {
    match game_mode {
        GameModeSettings::PracticeRange { spawn_location } => {
            Box::new(PracticeRangeMode { spawn_location })
        }
        GameModeSettings::Explorer { spawn_location } => Box::new(ExplorerMode { spawn_location }),
        GameModeSettings::FFA => Box::new(FFAMode),
        GameModeSettings::Control => Box::new(ControlMode::default()),
    }
}

impl GameMode for PracticeRangeMode {
    fn are_friends(&self, player1: u32, player2: u32, _entities: &Vec<Entity>) -> bool {
        player1 == player2
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

    fn cooldowns_reset_on_deck_swap(&self) -> bool {
        true
    }

    fn initialize(&mut self, player_sim: &mut Box<dyn PlayerSim>, card_manager: &mut CardManager) {
        player_sim.add_player(
            &Deck::empty(),
            card_manager,
            self,
            EntityMetaData::TrainingBot,
        );
    }
}

impl GameMode for ExplorerMode {
    fn are_friends(&self, _player1: u32, _player2: u32, _entities: &Vec<Entity>) -> bool {
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
    fn are_friends(&self, player1: u32, player2: u32, _entities: &Vec<Entity>) -> bool {
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
        let player1_team = entities
            .get(player1 as usize)
            .map(|p| p.gamemode_data.get(0).unwrap_or(&0))
            .unwrap_or(&0);
        let player2_team = entities
            .get(player2 as usize)
            .map(|p| p.gamemode_data.get(0).unwrap_or(&0))
            .unwrap_or(&0);
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

    fn player_mode(&self, entity: &Entity) -> PlayerMode {
        let player_team = entity.gamemode_data.get(0).unwrap_or(&0);
        if entity.respawn_timer > 0.0 || *player_team == 0 {
            PlayerMode::Spectator
        } else {
            PlayerMode::Normal
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
            1 => {
                if entity.pos.x - 10000.0 < -SPAWN_ROOM_OFFSET as f32 {
                    EditMode::FullEditing
                } else {
                    EditMode::Readonly
                }
            }
            2 => {
                if entity.pos.x - 10000.0 > SPAWN_ROOM_OFFSET as f32 {
                    EditMode::FullEditing
                } else {
                    EditMode::Readonly
                }
            }
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

    fn overlay(&self, ui: &mut Ui, _sim: &Box<dyn PlayerSim>) {
        ui.label(format!("{}", self.team_1_score));
        ui.add(Slider::new(
            &mut self.capture_progress.clone(),
            RangeInclusive::new(-1.0, 1.0),
        ));
        ui.label(format!("{}", self.team_2_score));
    }

    fn update(
        &mut self,
        entities: &mut Vec<Entity>,
        projectiles: &mut Vec<Projectile>,
        delta_time: f32,
    ) {
        let mut team_1_capturers = 0;
        let mut team_2_capturers = 0;
        for entity in entities.iter_mut() {
            if entity.respawn_timer > 0.0 {
                continue;
            }
            let entity_team = entity.gamemode_data.get(0).unwrap_or(&0);
            let vec_from_point = entity.pos - Point3::new(10000.0, 1800.0, 10000.0);
            match entity_team {
                1 => {
                    if entity.pos.x - 10000.0 < -SPAWN_ROOM_OFFSET as f32 {
                        entity.adjust_health(25.0 * delta_time);
                    }
                    if vec_from_point.y > 0.0
                        && vec_from_point.map(|c| c.abs()).x < 11.0
                        && vec_from_point.map(|c| c.abs()).z < 11.0
                    {
                        team_1_capturers += 1;
                    }
                }
                2 => {
                    if entity.pos.x - 10000.0 > SPAWN_ROOM_OFFSET as f32 {
                        entity.adjust_health(25.0 * delta_time);
                    }
                    if vec_from_point.y > 0.0
                        && vec_from_point.map(|c| c.abs()).x < 11.0
                        && vec_from_point.map(|c| c.abs()).z < 11.0
                    {
                        team_2_capturers += 1;
                    }
                }
                _ => {}
            }
            if entity.pos.y < 1800.0 - 14.0 {
                entity.adjust_health(-50.0 * delta_time);
            }
        }
        for proj in projectiles.iter_mut() {
            let entity_team = entities
                .get(proj.owner as usize)
                .unwrap()
                .gamemode_data
                .get(0)
                .unwrap_or(&0);
            match entity_team {
                1 => {
                    if proj.pos[0] - 10000.0 > SPAWN_ROOM_OFFSET as f32 {
                        proj.health = 0.0;
                    }
                }
                2 => {
                    if proj.pos[0] - 10000.0 < -SPAWN_ROOM_OFFSET as f32 {
                        proj.health = 0.0;
                    }
                }
                _ => {}
            }
        }
        if team_1_capturers > 0 && team_2_capturers == 0 {
            self.capture_progress -= 0.5 * delta_time;
        }
        if team_2_capturers > 0 && team_1_capturers == 0 {
            self.capture_progress += 0.5 * delta_time;
        }
        self.capture_progress = self.capture_progress.clamp(-1.0, 1.0);
        if self.capture_progress > 0.0 {
            self.team_2_score += delta_time;
        } else if self.capture_progress < 0.0 {
            self.team_1_score += delta_time;
        }
    }
}
