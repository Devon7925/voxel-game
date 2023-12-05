use serde::{Serialize, Deserialize};

use crate::{
    card_system::{CardManager, Cooldown},
    projectile_sim_manager::ProjectileComputePipeline,
    rollback_manager::{PlayerSim, ReplayData, RollbackData},
    settings_manager::{ReplayMode, Settings},
    voxel_sim_manager::VoxelComputePipeline,
    app::CreationInterface, CHUNK_SIZE,
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
    pub render_size: [u32; 3],
    pub spawn_location: [f32; 3],
    pub world_gen: WorldGenSettings,
}
pub struct GameState {
    pub start_pos: [u32; 3],
}

impl Game {
    pub fn new(
        settings: &Settings,
        game_settings: GameSettings,
        deck: &Vec<Cooldown>,
        creation_interface: &CreationInterface,
    ) -> Self {
        let game_state = GameState {
            start_pos: [
                game_settings.spawn_location[0] as u32 / CHUNK_SIZE - game_settings.render_size[0] / 2,
                game_settings.spawn_location[1] as u32 / CHUNK_SIZE - game_settings.render_size[1] / 2,
                game_settings.spawn_location[2] as u32 / CHUNK_SIZE - game_settings.render_size[2] / 2,
            ],
        };
        let mut card_manager = CardManager::default();

        let rollback_data: Box<dyn PlayerSim> =
            if settings.replay_settings.replay_mode == ReplayMode::Playback {
                Box::new(ReplayData::new(
                    &creation_interface.memory_allocator,
                    &settings,
                    &mut card_manager,
                ))
            } else {
                Box::new(RollbackData::new(
                    &creation_interface.memory_allocator,
                    &settings,
                    &game_settings,
                    deck,
                    &mut card_manager,
                ))
            };

        let mut voxel_compute = VoxelComputePipeline::new(creation_interface, &game_settings);
        let projectile_compute =
            ProjectileComputePipeline::new(creation_interface);

        voxel_compute.load_chunks(game_state.start_pos, &game_settings);

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
