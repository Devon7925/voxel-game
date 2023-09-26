use cgmath::{Vector3, Quaternion};
use serde::{Deserialize, Serialize};

use crate::projectile_sim_manager::Projectile;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum BaseCard {
    Projectile(Vec<ProjectileModifier>),
    MultiCast(Vec<BaseCard>),
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

impl BaseCard {
    pub fn from_string(ron_string: &str) -> Self {
        ron::from_str(ron_string).unwrap()
    }

    pub fn to_string(&self) -> String {
        ron::to_string(self).unwrap()
    }

    pub fn evaluate_value(&self) -> f32 {
        match self {
            BaseCard::Projectile(modifiers) => {
                let mut damage = 0;
                let mut speed = 0;
                let mut size = 0;
                let mut lifetime = 0;
                let mut gravity = 0;
                let mut value = 0.0;
                for modifier in modifiers {
                    match modifier {
                        ProjectileModifier::Damage(d) => damage += d,
                        ProjectileModifier::Speed(s) => speed += s,
                        ProjectileModifier::Size(s) => size += s,
                        ProjectileModifier::Lifetime(l) => lifetime += l,
                        ProjectileModifier::Gravity(g) => gravity += g,
                        ProjectileModifier::OnHit(card) => value += card.evaluate_value(),
                    }
                }
                value += if damage > 0 {
                    0.002
                        * damage as f32
                        * (1.0 + 1.5f32.powi(speed) * 1.5f32.powi(lifetime))
                        * (1.0 + 1.25f32.powi(size))
                } else {
                    -0.02 * damage as f32
                };
                value
            }
            BaseCard::MultiCast(cards) => {
                cards.iter().map(|card| card.evaluate_value()).sum::<f32>()
            }
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
    pub value: f32,
    pub on_hit: Vec<ReferencedBaseCard>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ReferencedMulticast {
    pub sub_cards: Vec<ReferencedBaseCard>,
    pub value: f32,
}

pub struct CardManager {
    pub referenced_multicasts: Vec<ReferencedMulticast>,
    pub referenced_projs: Vec<ReferencedProjectile>,
}

impl Default for CardManager {
    fn default() -> Self {
        CardManager {
            referenced_multicasts: vec![],
            referenced_projs: vec![],
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
                    value,
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
                    value,
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
        }
    }

    pub fn get_value(&self, reference: &ReferencedBaseCard) -> f32 {
        match reference.card_type {
            ReferencedBaseCardType::Projectile => self.referenced_projs[reference.card_idx].value,
            ReferencedBaseCardType::MultiCast => {
                self.referenced_multicasts[reference.card_idx].value
            }
            ReferencedBaseCardType::None => panic!("Cannot get value of None card"),
        }
    }

    pub fn get_proj_refs_from_basecard(&self, reference: &ReferencedBaseCard) -> Vec<u32> {
        match reference {
            ReferencedBaseCard {
                card_type: ReferencedBaseCardType::Projectile,
                card_idx,
            } => {
                vec![*card_idx as u32]
            }
            ReferencedBaseCard {
                card_type: ReferencedBaseCardType::MultiCast,
                card_idx,
            } => self.referenced_multicasts[*card_idx]
                .sub_cards
                .iter()
                .map(|card| self.get_proj_refs_from_basecard(card))
                .flatten()
                .collect(),
            ReferencedBaseCard {
                card_type: ReferencedBaseCardType::None,
                card_idx: _,
            } => {
                panic!("Cannot get projs from None card")
            }
        }
    }

    pub fn get_projectiles_from_base_card(&self, card: &ReferencedBaseCard, pos: &Vector3<f32>, rot: &Quaternion<f32>, player_idx: u32) -> Vec<Projectile> {
        self.get_proj_refs_from_basecard(card)
            .iter()
            .map(|idx| (idx, self.get_referenced_proj(*idx as usize)))
            .map(|(proj_card_idx, proj_stats)| {
                let proj_size = 1.25f32.powi(proj_stats.size);
                let proj_speed = 3.0 * 1.5f32.powi(proj_stats.speed);
                let proj_damage = proj_stats.damage as f32;
                Projectile {
                    pos: pos.extend(1.0).into(),
                    chunk_update_pos: [0, 0, 0, 0],
                    dir: [
                        rot.v[0],
                        rot.v[1],
                        rot.v[2],
                        rot.s,
                    ],
                    size: [proj_size, proj_size, proj_size, 1.0],
                    vel: proj_speed,
                    health: 10.0,
                    lifetime: 0.0,
                    owner: player_idx,
                    damage: proj_damage,
                    proj_card_idx: *proj_card_idx,
                    _filler2: 0.0,
                    _filler3: 0.0,
                }
            })
            .collect()
    }

    pub fn get_referenced_proj(&self, idx: usize) -> &ReferencedProjectile {
        &self.referenced_projs[idx]
    }
}
