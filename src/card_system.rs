use cgmath::{Quaternion, Vector3, Point3};
use serde::{Deserialize, Serialize};

use crate::projectile_sim_manager::Projectile;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum BaseCard {
    Projectile(Vec<ProjectileModifier>),
    MultiCast(Vec<BaseCard>),
    CreateMaterial(VoxelMaterial),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum ProjectileModifier {
    Damage(i32),
    Speed(i32),
    Size(i32),
    Lifetime(i32),
    Gravity(i32),
    OnHit(BaseCard),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum VoxelMaterial {
    Stone,
    Dirt,
    Grass,
    Air,
    Ice,
    Glass,
}

impl VoxelMaterial {
    pub const FRICTION_COEFFICIENTS: [f32; 8] = [0.0, 1.5, 0.0, 1.5, 1.5, 0.0, 0.1, 1.0];
    pub fn to_memory(&self) -> [u32; 2] {
        match self {
            VoxelMaterial::Air => [0, 0x11111111],
            VoxelMaterial::Stone => [1, 0x00000000],
            VoxelMaterial::Dirt => [3, 0x00000000],
            VoxelMaterial::Grass => [4, 0x00000000],
            VoxelMaterial::Ice => [6, 0x00000000],
            VoxelMaterial::Glass => [7, 0x00000000],
        }
    }
}

impl BaseCard {
    pub fn from_string(ron_string: &str) -> Self {
        ron::from_str(ron_string).unwrap()
    }

    pub fn vec_from_string(ron_string: &str) -> Vec<Self> {
        ron::from_str(ron_string).unwrap()
    }

    pub fn to_string(&self) -> String {
        ron::to_string(self).unwrap()
    }

    pub fn evaluate_value(&self) -> f32 {
        match self {
            BaseCard::Projectile(modifiers) => {
                let mut hit_value = 0.0;
                let mut speed = 0;
                let mut size = 0;
                let mut lifetime = 0;
                let mut gravity = 0;
                for modifier in modifiers {
                    match modifier {
                        ProjectileModifier::Damage(d) => hit_value += *d as f32,
                        ProjectileModifier::Speed(s) => speed += s,
                        ProjectileModifier::Size(s) => size += s,
                        ProjectileModifier::Lifetime(l) => lifetime += l,
                        ProjectileModifier::Gravity(g) => gravity += g,
                        ProjectileModifier::OnHit(card) => hit_value += card.evaluate_value(),
                    }
                }
                0.002
                    * hit_value as f32
                    * (1.0 + 1.5f32.powi(speed) * 1.5f32.powi(lifetime))
                    * (1.0 + 1.25f32.powi(size))
            }
            BaseCard::MultiCast(cards) => {
                cards.iter().map(|card| card.evaluate_value()).sum::<f32>()
            }
            BaseCard::CreateMaterial(material) => match material {
                VoxelMaterial::Air => 0.0,
                VoxelMaterial::Stone => 10.0,
                VoxelMaterial::Dirt => 5.0,
                VoxelMaterial::Grass => 5.0,
                VoxelMaterial::Ice => 20.0,
                VoxelMaterial::Glass => 15.0,
            },
        }
    }
}

impl Default for BaseCard {
    fn default() -> Self {
        BaseCard::MultiCast(vec![])
    }
}

pub struct ProjStats {
    pub damage: i32,
    pub speed: i32,
    pub size: i32,
    pub idx: u32,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum ReferencedBaseCardType {
    Projectile,
    MultiCast,
    CreateMaterial,
    None,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ReferencedBaseCard {
    card_type: ReferencedBaseCardType,
    card_idx: usize,
}

impl Default for ReferencedBaseCard {
    fn default() -> Self {
        ReferencedBaseCard {
            card_type: ReferencedBaseCardType::None,
            card_idx: 0,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ReferencedProjectile {
    pub damage: i32,
    pub speed: i32,
    pub size: i32,
    pub lifetime: i32,
    pub gravity: i32,
    pub on_hit: Vec<ReferencedBaseCard>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ReferencedMulticast {
    pub sub_cards: Vec<ReferencedBaseCard>,
}

pub struct CardManager {
    pub referenced_multicasts: Vec<ReferencedMulticast>,
    pub referenced_projs: Vec<ReferencedProjectile>,
    pub referenced_material_creators: Vec<VoxelMaterial>,
}

impl Default for CardManager {
    fn default() -> Self {
        CardManager {
            referenced_multicasts: vec![],
            referenced_projs: vec![],
            referenced_material_creators: vec![],
        }
    }
}

impl CardManager {
    pub fn register_base_card(&mut self, card: BaseCard) -> ReferencedBaseCard {
        let value = card.evaluate_value();
        match card {
            BaseCard::Projectile(modifiers) => {
                let mut damage = 0;
                let mut speed = 0;
                let mut size = 0;
                let mut lifetime = 0;
                let mut gravity = 0;
                let mut on_hit = Vec::new();
                for modifier in modifiers {
                    match modifier {
                        ProjectileModifier::Damage(d) => damage += d,
                        ProjectileModifier::Speed(s) => speed += s,
                        ProjectileModifier::Size(s) => size += s,
                        ProjectileModifier::Lifetime(l) => lifetime += l,
                        ProjectileModifier::Gravity(g) => gravity += g,
                        ProjectileModifier::OnHit(card) => {
                            on_hit.push(self.register_base_card(card))
                        }
                    }
                }
                self.referenced_projs.push(ReferencedProjectile {
                    damage,
                    speed,
                    size,
                    lifetime,
                    gravity,
                    on_hit,
                });

                ReferencedBaseCard {
                    card_type: ReferencedBaseCardType::Projectile,
                    card_idx: self.referenced_projs.len() - 1,
                }
            }
            BaseCard::MultiCast(cards) => {
                let mut referenced_multicast = ReferencedMulticast {
                    sub_cards: Vec::new(),
                };
                for card in cards {
                    referenced_multicast
                        .sub_cards
                        .push(self.register_base_card(card));
                }
                self.referenced_multicasts.push(referenced_multicast);

                ReferencedBaseCard {
                    card_type: ReferencedBaseCardType::MultiCast,
                    card_idx: self.referenced_multicasts.len() - 1,
                }
            }
            BaseCard::CreateMaterial(material) => {
                self.referenced_material_creators.push(material);
                ReferencedBaseCard {
                    card_type: ReferencedBaseCardType::CreateMaterial,
                    card_idx: self.referenced_material_creators.len() - 1,
                }
            }
        }
    }

    pub fn get_single_refs_from_basecard(
        &self,
        reference: &ReferencedBaseCard,
    ) -> Vec<ReferencedBaseCard> {
        match reference {
            ReferencedBaseCard {
                card_type: ReferencedBaseCardType::MultiCast,
                card_idx,
                ..
            } => self.referenced_multicasts[*card_idx]
                .sub_cards
                .iter()
                .map(|card| self.get_single_refs_from_basecard(card))
                .flatten()
                .collect(),
            ReferencedBaseCard {
                card_type: ReferencedBaseCardType::None,
                ..
            } => {
                panic!("Cannot get projs from None card")
            }
            card_reference => vec![card_reference.clone()],
        }
    }

    pub fn get_effects_from_base_card(
        &self,
        card: &ReferencedBaseCard,
        pos: &Point3<f32>,
        rot: &Quaternion<f32>,
        player_idx: u32,
    ) -> (Vec<Projectile>, Vec<(Point3<i32>, VoxelMaterial)>) {
        let mut projectiles = vec![];
        let mut new_voxels = vec![];
        for reference in self.get_single_refs_from_basecard(card) {
            match reference {
                ReferencedBaseCard {
                    card_type: ReferencedBaseCardType::Projectile,
                    card_idx,
                    ..
                } => {
                    let proj_stats = self.get_referenced_proj(card_idx);
                    let proj_size = 1.25f32.powi(proj_stats.size);
                    let proj_speed = 3.0 * 1.5f32.powi(proj_stats.speed);
                    let proj_damage = proj_stats.damage as f32;
                    projectiles.push(Projectile {
                        pos: [pos.x, pos.y, pos.z, 1.0],
                        chunk_update_pos: [0, 0, 0, 0],
                        dir: [rot.v[0], rot.v[1], rot.v[2], rot.s],
                        size: [proj_size, proj_size, proj_size, 1.0],
                        vel: proj_speed,
                        health: 10.0,
                        lifetime: 0.0,
                        owner: player_idx,
                        damage: proj_damage,
                        proj_card_idx: card_idx as u32,
                        _filler2: 0.0,
                        _filler3: 0.0,
                    });
                }
                ReferencedBaseCard {
                    card_type: ReferencedBaseCardType::CreateMaterial,
                    card_idx,
                    ..
                } => {
                    let material = &self.referenced_material_creators[card_idx];
                    new_voxels.push((pos.cast::<i32>().unwrap(), material.clone()));
                }
                _ => panic!("Invalid state")
            }
        }
        (projectiles, new_voxels)
    }

    pub fn get_referenced_proj(&self, idx: usize) -> &ReferencedProjectile {
        &self.referenced_projs[idx]
    }
}
