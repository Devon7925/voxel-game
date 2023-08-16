use noise::NoiseFn;

use crate::{CHUNK_SIZE};

pub struct WorldGen {
    noise: Box<dyn NoiseFn<f64, 3>>,
}

impl WorldGen {
    pub fn new(noise: Box<dyn NoiseFn<f64, 3>>) -> Self {
        Self {
            noise,
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
                                if self.noise.get(true_pos) * 20.0
                                    > 0.0
                                {
                                    [1, 0x0000000A]
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