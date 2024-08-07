use std::{
    io::{BufRead, BufReader},
    path::Path,
};

use cgmath::{EuclideanSpace, Point3};

use crate::{
    app::CreationInterface,
    card_system::{CardManager, Deck},
    game_modes::{game_mode_from_type, GameMode},
    rollback_manager::{PlayerSim, ReplayData, RollbackData},
    settings_manager::Settings,
    voxel_sim_manager::VoxelComputePipeline,
    CHUNK_SIZE,
};
use voxel_shared::{GameSettings, RoomId};

pub struct Game {
    pub voxel_compute: VoxelComputePipeline,
    pub rollback_data: Box<dyn PlayerSim>,
    pub card_manager: CardManager,
    pub game_state: GameState,
    pub game_settings: GameSettings,
    pub game_mode: Box<dyn GameMode>,
    pub has_started: bool,
}
pub struct GameState {
    pub start_pos: Point3<u32>,
    pub players_center: Point3<f32>,
}

impl Game {
    pub fn new(
        settings: &Settings,
        game_settings: GameSettings,
        deck: &Deck,
        creation_interface: &CreationInterface,
        lobby_id: Option<RoomId>,
    ) -> Self {
        let mut game_mode = game_mode_from_type(game_settings.game_mode.clone());
        let game_state = GameState {
            start_pos: game_mode.get_initial_center().zip(
                Point3::from_vec(game_settings.render_size),
                |spawn, size| spawn as u32 / CHUNK_SIZE as u32 - size / 2,
            ),
            players_center: game_mode.get_initial_center().into(),
        };
        let mut card_manager = CardManager::default();

        let mut rollback_data: Box<dyn PlayerSim> = Box::new(RollbackData::new(
            &creation_interface.memory_allocator,
            &settings,
            &game_settings,
            deck,
            &mut card_manager,
            lobby_id,
            &game_mode,
        ));

        let mut voxel_compute = VoxelComputePipeline::new(creation_interface, &game_settings);

        voxel_compute.queue_update_from_world_pos(&game_mode.get_initial_center(), &game_settings);

        game_mode.initialize(&mut rollback_data, &mut card_manager);

        Game {
            voxel_compute,
            rollback_data,
            card_manager,
            game_state,
            game_settings,
            game_mode,
            has_started: false,
        }
    }

    pub fn from_replay(replay_file: &Path, creation_interface: &CreationInterface) -> Self {
        let replay_file = std::fs::File::open(replay_file).unwrap();
        let reader = BufReader::new(replay_file);
        let mut replay_lines = reader.lines();
        let game_settings: GameSettings = 'game_settings: {
            while let Some(line) = replay_lines.next() {
                let Ok(line) = line else {
                    continue;
                };
                if let Some(game_settings_string) = line.strip_prefix("GAME SETTINGS ") {
                    break 'game_settings ron::de::from_str(game_settings_string).unwrap();
                }
            }
            panic!("No game settings found in replay file");
        };
        let mut game_mode = game_mode_from_type(game_settings.game_mode.clone());

        let game_state = GameState {
            start_pos: game_mode.get_initial_center().zip(
                Point3::from_vec(game_settings.render_size),
                |spawn, size| spawn as u32 / CHUNK_SIZE as u32 - size / 2,
            ),
            players_center: game_mode.get_initial_center().into(),
        };
        let mut card_manager = CardManager::default();

        let mut rollback_data: Box<dyn PlayerSim> = Box::new(ReplayData::new(
            &creation_interface.memory_allocator,
            &game_settings,
            &mut replay_lines,
            &mut card_manager,
            &game_mode,
        ));

        let mut voxel_compute = VoxelComputePipeline::new(creation_interface, &game_settings);

        voxel_compute.queue_update_from_world_pos(&game_mode.get_initial_center(), &game_settings);

        game_mode.initialize(&mut rollback_data, &mut card_manager);

        Game {
            voxel_compute,
            rollback_data,
            card_manager,
            game_state,
            game_settings,
            game_mode,
            has_started: false,
        }
    }
}
