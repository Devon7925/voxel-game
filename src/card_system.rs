use cgmath::{Point3, Quaternion, Rotation3, Rad};
use serde::{Deserialize, Serialize};

use crate::projectile_sim_manager::Projectile;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum BaseCard {
    Projectile(Vec<ProjectileModifier>),
    MultiCast(Vec<BaseCard>, Vec<MultiCastModifier>),
    CreateMaterial(VoxelMaterial),
    Effect(Effect),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum ProjectileModifier {
    Speed(i32),
    Length(i32),
    Width(i32),
    Height(i32),
    Lifetime(i32),
    Gravity(i32),
    Health(i32),
    NoFriendlyFire,
    NoEnemyFire,
    OnHit(BaseCard),
    OnExpiry(BaseCard),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum MultiCastModifier {
    Spread(u32),
    Duplication(u32),
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

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum Effect {
    Damage(i32),
    Knockback(i32),
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

    pub fn evaluate_value(&self, consider_hit_probability: bool) -> f32 {
        match self {
            BaseCard::Projectile(modifiers) => {
                let mut hit_value = 0.0;
                let mut speed = 0;
                let mut length = 0;
                let mut width = 0;
                let mut height = 0;
                let mut lifetime = 0;
                let mut gravity = 0;
                let mut health = 0;
                let mut friendly_fire = true;
                let mut enemy_fire = true;
                for modifier in modifiers {
                    match modifier {
                        ProjectileModifier::Speed(s) => speed += s,
                        ProjectileModifier::Length(s) => length += s,
                        ProjectileModifier::Width(s) => width += s,
                        ProjectileModifier::Height(s) => height += s,
                        ProjectileModifier::Lifetime(l) => lifetime += l,
                        ProjectileModifier::Gravity(g) => gravity += g,
                        ProjectileModifier::Health(g) => health += g,
                        ProjectileModifier::NoFriendlyFire => friendly_fire = false,
                        ProjectileModifier::NoEnemyFire => enemy_fire = false,
                        ProjectileModifier::OnHit(card) => hit_value += card.evaluate_value(false),
                        ProjectileModifier::OnExpiry(card) => hit_value += card.evaluate_value(false),
                    }
                }
                if consider_hit_probability {
                    0.002
                        * hit_value
                        * (1.0 + 1.5f32.powi(speed) * 1.5f32.powi(lifetime))
                        * (1.0 + 1.25f32.powi(width) * 1.25f32.powi(height) + 1.25f32.powi(length))
                        * (1.0 + 1.25f32.powi(health))
                    + 0.02
                        * 1.5f32.powi(lifetime)
                        * (1.0 + 1.25f32.powi(width) * 1.25f32.powi(height) + 1.25f32.powi(length))
                        * (1.0 + 1.25f32.powi(health))
                        * if friendly_fire { 1.0 } else { 2.0 }
                } else {
                    hit_value
                    * (1.0 + 1.25f32.powi(health))
                    + 0.02
                        * 1.5f32.powi(lifetime)
                        * (1.0 + 1.25f32.powi(width) * 1.25f32.powi(height) + 1.25f32.powi(length))
                        * (1.0 + 1.25f32.powi(health))
                        * if friendly_fire { 1.0 } else { 2.0 }
                }
            }
            BaseCard::MultiCast(cards, modifiers) => {
                let mut value = cards.iter().map(|card| card.evaluate_value(consider_hit_probability)).sum::<f32>();
                for modifier in modifiers.iter() {
                    match modifier {
                        MultiCastModifier::Duplication(duplication) => {
                            value *= 2f32.powi(*duplication as i32);
                        }
                        MultiCastModifier::Spread(spread) => {
                            value *= 0.5 + 0.5f32.powi(*spread as i32);
                        }
                    }
                }
                value
            }
            BaseCard::CreateMaterial(material) => match material {
                VoxelMaterial::Air => 0.0,
                VoxelMaterial::Stone => 10.0,
                VoxelMaterial::Dirt => 5.0,
                VoxelMaterial::Grass => 5.0,
                VoxelMaterial::Ice => 20.0,
                VoxelMaterial::Glass => 15.0,
            },
            BaseCard::Effect(effect) => match effect {
                Effect::Damage(damage) => (*damage as f32).abs(),
                Effect::Knockback(knockback) => (*knockback as f32).abs(),
            },
        }
    }

    pub fn is_reasonable(&self) -> bool {
        match self {
            BaseCard::Projectile(modifiers) => {
                for modifier in modifiers {
                    match modifier {
                        ProjectileModifier::Speed(s) => {
                            if *s > 15 {
                                return false;
                            }
                        }
                        ProjectileModifier::Length(s) => {
                            if *s > 15 {
                                return false;
                            }
                        }
                        ProjectileModifier::Width(s) => {
                            if *s > 15 {
                                return false;
                            }
                        }
                        ProjectileModifier::Height(s) => {
                            if *s > 15 {
                                return false;
                            }
                        }
                        ProjectileModifier::Gravity(g) => {
                            if *g > 15 {
                                return false;
                            }
                        }
                        ProjectileModifier::OnHit(card) => {
                            if !card.is_reasonable() {
                                return false;
                            }
                        }
                        ProjectileModifier::OnExpiry(card) => {
                            if !card.is_reasonable() {
                                return false;
                            }
                        }
                        _ => {}
                    }
                }
            }
            BaseCard::MultiCast(cards, modifiers) => {
                if !cards.iter().all(|card| card.is_reasonable()) {
                    return false;
                }
                for modifier in modifiers.iter() {
                    match modifier {
                        MultiCastModifier::Duplication(duplication) => {
                            if *duplication > 12 {
                                return false;
                            }
                        }
                        _ => {}
                    }
                }
            }
            BaseCard::CreateMaterial(_) => {}
            BaseCard::Effect(effect) => match effect {
                Effect::Damage(damage) => {
                    if damage.abs() >= 1024 {
                        return false;
                    }
                }
                Effect::Knockback(knockback) => {
                    if knockback.abs() > 20 {
                        return false;
                    }
                }
            },
        }
        return true;
    }
}

impl Default for BaseCard {
    fn default() -> Self {
        BaseCard::MultiCast(vec![], vec![])
    }
}

impl ProjectileModifier {
    pub fn get_hover_text(&self) -> String {
        match self {
            ProjectileModifier::Speed(_) => format!("Speed (+50% per) {}b/s", self.get_effect_value()),
            ProjectileModifier::Length(_) => format!("Length (+25% per) {}", self.get_effect_value()),
            ProjectileModifier::Width(_) => format!("Width (+25% per) {}", self.get_effect_value()),
            ProjectileModifier::Height(_) => format!("Height (+25% per) {}", self.get_effect_value()),
            ProjectileModifier::Lifetime(_) => format!("Lifetime (+50% per) {}s", self.get_effect_value()),
            ProjectileModifier::Gravity(_) => format!("Gravity (+2 per) {}b/s/s", self.get_effect_value()),
            ProjectileModifier::Health(_) => format!("Entity Health (+50% per) {}", self.get_effect_value()),
            ProjectileModifier::NoFriendlyFire => format!("Prevents hitting friendly entities"),
            ProjectileModifier::NoEnemyFire => format!("Prevents hitting enemy entities"),
            ProjectileModifier::OnHit(card) => format!("On Hit {}", card.to_string()),
            ProjectileModifier::OnExpiry(card) => format!("On Expiry {}", card.to_string()),
        }
    }

    pub fn get_effect_value(&self) -> f32 {
        match self {
            ProjectileModifier::Speed(s) => 24.0 * 1.5f32.powi(*s),
            ProjectileModifier::Length(s) => 1.25f32.powi(*s),
            ProjectileModifier::Width(s) => 1.25f32.powi(*s),
            ProjectileModifier::Height(s) => 1.25f32.powi(*s),
            ProjectileModifier::Lifetime(s) => 3.0 * 1.5f32.powi(*s),
            ProjectileModifier::Gravity(s) => 2.0 * (*s as f32),
            ProjectileModifier::Health(s) => 10.0 * 1.5f32.powi(*s),
            ProjectileModifier::NoFriendlyFire => panic!(),
            ProjectileModifier::NoEnemyFire => panic!(),
            ProjectileModifier::OnHit(_) => panic!(),
            ProjectileModifier::OnExpiry(_) => panic!(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq)]
pub enum ReferencedBaseCardType {
    Projectile,
    MultiCast,
    CreateMaterial,
    Effect,
    None,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq)]
pub struct ReferencedBaseCard {
    pub card_type: ReferencedBaseCardType,
    pub card_idx: usize,
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
    pub speed: f32,
    pub length: f32,
    pub width: f32,
    pub height: f32,
    pub lifetime: f32,
    pub gravity: f32,
    pub health: f32,
    pub no_friendly_fire: bool,
    pub no_enemy_fire: bool,
    pub on_hit: Vec<ReferencedBaseCard>,
    pub on_expiry: Vec<ReferencedBaseCard>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ReferencedMulticast {
    pub sub_cards: Vec<ReferencedBaseCard>,
    pub spread: u32,
    pub duplication: u32,
}

pub struct CardManager {
    pub referenced_multicasts: Vec<ReferencedMulticast>,
    pub referenced_projs: Vec<ReferencedProjectile>,
    pub referenced_material_creators: Vec<VoxelMaterial>,
    pub referenced_effects: Vec<Effect>,
}

impl Default for CardManager {
    fn default() -> Self {
        CardManager {
            referenced_multicasts: vec![],
            referenced_projs: vec![],
            referenced_material_creators: vec![],
            referenced_effects: vec![],
        }
    }
}

impl CardManager {
    pub fn register_base_card(&mut self, card: BaseCard) -> ReferencedBaseCard {
        match card {
            BaseCard::Projectile(modifiers) => {
                let mut damage = 0;
                let mut speed = 0;
                let mut length = 0;
                let mut width = 0;
                let mut height = 0;
                let mut lifetime = 0;
                let mut gravity = 0;
                let mut health = 0;
                let mut no_friendly_fire = false;
                let mut no_enemy_fire = false;
                let mut on_hit = Vec::new();
                let mut on_expiry = Vec::new();
                for modifier in modifiers {
                    match modifier {
                        ProjectileModifier::Speed(s) => speed += s,
                        ProjectileModifier::Length(s) => length += s,
                        ProjectileModifier::Width(s) => width += s,
                        ProjectileModifier::Height(s) => height += s,
                        ProjectileModifier::Lifetime(l) => lifetime += l,
                        ProjectileModifier::Gravity(g) => gravity += g,
                        ProjectileModifier::Health(g) => health += g,
                        ProjectileModifier::NoFriendlyFire => no_friendly_fire = true,
                        ProjectileModifier::NoEnemyFire => no_enemy_fire = true,
                        ProjectileModifier::OnHit(card) => {
                            if let BaseCard::Effect(Effect::Damage(proj_damage)) = card {
                                damage += proj_damage;
                            }
                            on_hit.push(self.register_base_card(card))
                        }
                        ProjectileModifier::OnExpiry(card) => {
                            if let BaseCard::Effect(Effect::Damage(proj_damage)) = card {
                                damage += proj_damage;
                            }
                            on_expiry.push(self.register_base_card(card))
                        }
                    }
                }
                self.referenced_projs.push(ReferencedProjectile {
                    damage,
                    speed: ProjectileModifier::Speed(speed).get_effect_value(),
                    length: ProjectileModifier::Length(length).get_effect_value(),
                    width: ProjectileModifier::Width(width).get_effect_value(),
                    height: ProjectileModifier::Height(height).get_effect_value(),
                    lifetime: ProjectileModifier::Lifetime(lifetime).get_effect_value(),
                    gravity: ProjectileModifier::Gravity(gravity).get_effect_value(),
                    health: ProjectileModifier::Health(health).get_effect_value(),
                    no_friendly_fire,
                    no_enemy_fire,
                    on_hit,
                    on_expiry,
                });

                ReferencedBaseCard {
                    card_type: ReferencedBaseCardType::Projectile,
                    card_idx: self.referenced_projs.len() - 1,
                }
            }
            BaseCard::MultiCast(cards, modifiers) => {
                let mut spread = 0;
                let mut duplication = 0;
                for modifier in modifiers {
                    match modifier {
                        MultiCastModifier::Spread(s) => spread += s,
                        MultiCastModifier::Duplication(d) => duplication += d,
                    }
                }
                let mut referenced_multicast = ReferencedMulticast {
                    sub_cards: Vec::new(),
                    spread,
                    duplication,
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
            BaseCard::Effect(effect) => {
                self.referenced_effects.push(effect);
                ReferencedBaseCard {
                    card_type: ReferencedBaseCardType::Effect,
                    card_idx: self.referenced_effects.len() - 1,
                }
            }
        }
    }

    pub fn get_effects_from_base_card(
        &self,
        card: ReferencedBaseCard,
        pos: &Point3<f32>,
        rot: &Quaternion<f32>,
        player_idx: u32,
    ) -> (
        Vec<Projectile>,
        Vec<(Point3<i32>, VoxelMaterial)>,
        Vec<Effect>,
    ) {
        let mut projectiles = vec![];
        let mut new_voxels = vec![];
        let mut effects = vec![];
        match card {
            ReferencedBaseCard {
                card_type: ReferencedBaseCardType::Projectile,
                card_idx,
                ..
            } => {
                let proj_stats = self.get_referenced_proj(card_idx);
                let proj_damage = proj_stats.damage as f32;
                projectiles.push(Projectile {
                    pos: [pos.x, pos.y, pos.z, 1.0],
                    chunk_update_pos: [0, 0, 0, 0],
                    dir: [rot.v[0], rot.v[1], rot.v[2], rot.s],
                    size: [proj_stats.width, proj_stats.height, proj_stats.length, 1.0],
                    vel: proj_stats.speed,
                    health: proj_stats.health,
                    lifetime: 0.0,
                    owner: player_idx,
                    damage: proj_damage,
                    proj_card_idx: card_idx as u32,
                    _filler2: 0.0,
                    _filler3: 0.0,
                });
            }
            ReferencedBaseCard {
                card_type: ReferencedBaseCardType::MultiCast,
                card_idx,
                ..
            } => {
                let multicast = &self.referenced_multicasts[card_idx];
                let mut individual_sub_projectiles = vec![];
                let mut sub_projectiles = vec![];
                let mut sub_voxels = vec![];
                let mut sub_effects = vec![];
                for sub_card in multicast.sub_cards.iter() {
                    let (sub_sub_projectiles, sub_sub_voxels, sub_sub_effects) =
                        self.get_effects_from_base_card(*sub_card, pos, rot, player_idx);
                    individual_sub_projectiles.extend(sub_sub_projectiles);
                    sub_voxels.extend(sub_sub_voxels);
                    sub_effects.extend(sub_sub_effects);
                }
                let spread:f32 = multicast.spread as f32 / 15.0;
                let count = 2u32.pow(multicast.duplication);
                let rotation_factor = 2.4;
                for i in 0..count {
                    for sub_projectile in individual_sub_projectiles.iter() {
                        let mut new_sub_projectile = sub_projectile.clone();
                        let x_rot = spread*(i as f32 / count as f32).sqrt()*(rotation_factor*(i as f32)).cos();
                        let y_rot = spread*(i as f32 / count as f32).sqrt()*(rotation_factor*(i as f32)).sin();
                        let new_rot = rot
                            * Quaternion::from_axis_angle([0.0, 1.0, 0.0].into(), Rad(x_rot))
                            * Quaternion::from_axis_angle([1.0, 0.0, 0.0].into(), Rad(y_rot));
                        
                        new_sub_projectile.dir = [new_rot.v[0], new_rot.v[1], new_rot.v[2], new_rot.s];
                        sub_projectiles.push(new_sub_projectile);
                    }
                }
                projectiles.extend(sub_projectiles);
                new_voxels.extend(sub_voxels);
                effects.extend(sub_effects);
            }
            ReferencedBaseCard {
                card_type: ReferencedBaseCardType::CreateMaterial,
                card_idx,
                ..
            } => {
                let material = &self.referenced_material_creators[card_idx];
                new_voxels.push((pos.cast::<i32>().unwrap(), material.clone()));
            }
            ReferencedBaseCard {
                card_type: ReferencedBaseCardType::Effect,
                card_idx,
                ..
            } => {
                let effect = &self.referenced_effects[card_idx];
                effects.push(effect.clone());
            }
            _ => panic!("Invalid state"),
        }
        (projectiles, new_voxels, effects)
    }

    pub fn get_referenced_proj(&self, idx: usize) -> &ReferencedProjectile {
        &self.referenced_projs[idx]
    }
}
