use crate::{
    card_system::{CardManager, Cooldown},
    projectile_sim_manager::ProjectileComputePipeline,
    rollback_manager::{PlayerSim, ReplayData, RollbackData},
    settings_manager::{ReplayMode, Settings},
    voxel_sim_manager::VoxelComputePipeline,
    app::CreationInterface, FIRST_START_POS,
};

pub struct Game {
    pub voxel_compute: VoxelComputePipeline,
    pub projectile_compute: ProjectileComputePipeline,
    pub rollback_data: Box<dyn PlayerSim>,
    pub card_manager: CardManager,
    pub game_state: GameState,
    pub game_settings: GameSettings,
}

pub enum WorldGenSettings {
    Normal,
    PracticeRange,
}

pub struct GameSettings {
    pub is_remote: bool,
    pub player_count: u32,
    pub render_size: [u32; 3],
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
            start_pos: FIRST_START_POS,
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

        let mut voxel_compute = VoxelComputePipeline::new(creation_interface);
        let projectile_compute =
            ProjectileComputePipeline::new(creation_interface);

        voxel_compute.load_chunks(game_state.start_pos);

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
