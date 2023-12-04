use noise::NoiseFn;

use crate::{CHUNK_SIZE, card_system::VoxelMaterial};

pub struct WorldGen {
    world_density: Box<dyn NoiseFn<f64, 3> + Sync>,
    pillar_density: Box<dyn NoiseFn<f64, 3> + Sync>,
}

impl WorldGen {
    pub fn new(world_density: Box<dyn NoiseFn<f64, 3> + Sync>, pillar_density: Box<dyn NoiseFn<f64, 3> + Sync>) -> Self {
        Self {
            world_density,
            pillar_density,
        }
    }
    pub fn gen_chunk(&self, chunk_location: [u32; 3]) -> Vec<u32> {
        (0..CHUNK_SIZE)
            .flat_map(|x| {
                (0..CHUNK_SIZE)
                    .flat_map(|y| {
                        (0..CHUNK_SIZE)
                            .map(|z| {
                                let true_pos = [
                                    (chunk_location[0] * (CHUNK_SIZE as u32) + (x as u32)) as f64,
                                    (chunk_location[1] * (CHUNK_SIZE as u32) + (y as u32)) as f64,
                                    (chunk_location[2] * (CHUNK_SIZE as u32) + (z as u32)) as f64,
                                ];
                                let density = self.world_density.get(true_pos) - ((true_pos[1] - 1800.0) / 15.0);
                                let pillar_density = self.pillar_density.get(true_pos) - ((true_pos[1] - 1800.0) / 80.0);
                                if pillar_density > 0.2 { 
                                    VoxelMaterial::Stone
                                } else if density > 0.3 { 
                                    VoxelMaterial::Stone
                                } else if density > 0.1 {
                                    VoxelMaterial::Dirt
                                } else if density > 0.0 {
                                    VoxelMaterial::Grass
                                } else if true_pos[1] <= 1796.0 {
                                    VoxelMaterial::Ice
                                } else {
                                    VoxelMaterial::Air
                                }.to_memory()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }
}