use std::{io::{BufReader, BufRead}, path::Path};

use cgmath::{EuclideanSpace, Point3, Vector3};
use serde::{Deserialize, Serialize};

use crate::{
    app::CreationInterface,
    card_system::{CardManager, Cooldown},
    projectile_sim_manager::ProjectileComputePipeline,
    rollback_manager::{PlayerSim, ReplayData, RollbackData},
    settings_manager::Settings,
    voxel_sim_manager::VoxelComputePipeline,
    CHUNK_SIZE,
};

pub struct Game {
    pub voxel_compute: VoxelComputePipeline,
    pub projectile_compute: ProjectileComputePipeline,
    pub rollback_data: Box<dyn PlayerSim>,
    pub card_manager: CardManager,
    pub game_state: GameState,
    pub game_settings: GameSettings,
}

#[derive(Serialize, Deserialize)]
pub enum WorldGenSettings {
    Normal,
    PracticeRange,
}

#[derive(Serialize, Deserialize)]
pub struct GameSettings {
    pub is_remote: bool,
    pub player_count: u32,
    pub render_size: Vector3<u32>,
    pub spawn_location: Point3<f32>,
    pub max_loaded_chunks: u32,
    pub world_gen: WorldGenSettings,
    pub fixed_center: bool,
}
pub struct GameState {
    pub start_pos: Point3<u32>,
    pub players_center: Point3<f32>,
}

impl Game {
    pub fn new(
        settings: &Settings,
        game_settings: GameSettings,
        deck: &Vec<Cooldown>,
        creation_interface: &CreationInterface,
    ) -> Self {
        let game_state = GameState {
            start_pos: game_settings.spawn_location.zip(
                Point3::from_vec(game_settings.render_size),
                |spawn, size| spawn as u32 / CHUNK_SIZE - size / 2,
            ),
            players_center: game_settings.spawn_location.into(),
        };
        let mut card_manager = CardManager::default();

        let rollback_data: Box<dyn PlayerSim> = Box::new(RollbackData::new(
            &creation_interface.memory_allocator,
            &settings,
            &game_settings,
            deck,
            &mut card_manager,
        ));

        let mut voxel_compute = VoxelComputePipeline::new(creation_interface, &game_settings);
        let projectile_compute = ProjectileComputePipeline::new(creation_interface);

        voxel_compute.queue_update_from_world_pos(&game_settings.spawn_location, &game_settings);

        Game {
            voxel_compute,
            projectile_compute,
            rollback_data,
            card_manager,
            game_state,
            game_settings,
        }
    }

    pub fn from_replay(
        settings: &Settings,
        replay_file: &Path,
        creation_interface: &CreationInterface,
    ) -> Self {
        
        let replay_file =
            std::fs::File::open(replay_file).unwrap();
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
        
        let game_state = GameState {
            start_pos: game_settings.spawn_location.zip(
                Point3::from_vec(game_settings.render_size),
                |spawn, size| spawn as u32 / CHUNK_SIZE - size / 2,
            ),
            players_center: game_settings.spawn_location.into(),
        };
        let mut card_manager = CardManager::default();

        let rollback_data: Box<dyn PlayerSim> = Box::new(ReplayData::new(
            &creation_interface.memory_allocator,
            &settings,
            &game_settings,
            &mut replay_lines,
            &mut card_manager,
        ));

        let mut voxel_compute = VoxelComputePipeline::new(creation_interface, &game_settings);
        let projectile_compute = ProjectileComputePipeline::new(creation_interface);

        voxel_compute.queue_update_from_world_pos(&game_settings.spawn_location, &game_settings);

        Game {
            voxel_compute,
            projectile_compute,
            rollback_data,
            card_manager,
            game_state,
            game_settings,
        }
    }
}
