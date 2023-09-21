use noise::NoiseFn;

use crate::CHUNK_SIZE;

pub struct WorldGen {
    world_density: Box<dyn NoiseFn<f64, 3>>,
}

impl WorldGen {
    pub fn new(noise: Box<dyn NoiseFn<f64, 3>>) -> Self {
        Self {
            world_density: noise,
        }
    }
    pub fn gen_chunk(&self, chunk_location: [i32; 3]) -> Vec<[u32; 2]> {
        (0..CHUNK_SIZE)
            .flat_map(|x| {
                (0..CHUNK_SIZE)
                    .flat_map(|y| {
                        (0..CHUNK_SIZE)
                            .map(|z| {
                                let true_pos = [
                                    (chunk_location[0] * (CHUNK_SIZE as i32) + (x as i32)) as f64,
                                    (chunk_location[1] * (CHUNK_SIZE as i32) + (y as i32)) as f64,
                                    (chunk_location[2] * (CHUNK_SIZE as i32) + (z as i32)) as f64,
                                ];
                                let density = self.world_density.get(true_pos) - ((true_pos[1] - 1800.0) / 50.0);
                                if density > 0.5 { 
                                    [1, 0x00000000]
                                } else if density > 0.1 {
                                    [3, 0x00000000]
                                } else if density > 0.0 {
                                    [4, 0x00000000]
                                } else {
                                    [0, 0x11111111]
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }
}