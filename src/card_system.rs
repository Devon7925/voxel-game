use std::collections::VecDeque;

use cgmath::{Point3, Quaternion, Rad, Rotation3};
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
    SimpleModify(ProjectileModifierType, i32),
    FriendlyFire,
    NoEnemyFire,
    OnHit(BaseCard),
    OnExpiry(BaseCard),
    Trail(u32, BaseCard),
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub enum ProjectileModifierType {
    Speed,
    Length,
    Width,
    Height,
    Lifetime,
    Gravity,
    Health,
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
    StatusEffect(StatusEffect, u32),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum StatusEffect {
    Speed,
    Slow,
    DamageOverTime,
    HealOverTime,
    IncreaceDamageTaken,
    DecreaceDamageTaken,
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

    pub fn evaluate_value(&self, is_direct_shot: bool) -> f32 {
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
                let mut friendly_fire = false;
                let mut enemy_fire = true;
                let mut trail_value = 0.0;
                for modifier in modifiers {
                    match modifier {
                        ProjectileModifier::SimpleModify(ProjectileModifierType::Speed, s) => {
                            speed += s
                        }
                        ProjectileModifier::SimpleModify(ProjectileModifierType::Length, s) => {
                            length += s
                        }
                        ProjectileModifier::SimpleModify(ProjectileModifierType::Width, s) => {
                            width += s
                        }
                        ProjectileModifier::SimpleModify(ProjectileModifierType::Height, s) => {
                            height += s
                        }
                        ProjectileModifier::SimpleModify(ProjectileModifierType::Lifetime, l) => {
                            lifetime += l
                        }
                        ProjectileModifier::SimpleModify(ProjectileModifierType::Gravity, g) => {
                            gravity += g
                        }
                        ProjectileModifier::SimpleModify(ProjectileModifierType::Health, g) => {
                            health += g
                        }
                        ProjectileModifier::FriendlyFire => friendly_fire = true,
                        ProjectileModifier::NoEnemyFire => enemy_fire = false,
                        ProjectileModifier::OnHit(card) => hit_value += card.evaluate_value(false),
                        ProjectileModifier::OnExpiry(card) => {
                            hit_value += card.evaluate_value(false)
                        }
                        ProjectileModifier::Trail(freq, card) => {
                            trail_value += card.evaluate_value(false) * *freq as f32
                        }
                    }
                }
                let speed = ProjectileModifier::SimpleModify(ProjectileModifierType::Speed, speed)
                    .get_effect_value();
                let length =
                    ProjectileModifier::SimpleModify(ProjectileModifierType::Length, length)
                        .get_effect_value();
                let width = ProjectileModifier::SimpleModify(ProjectileModifierType::Width, width)
                    .get_effect_value();
                let height =
                    ProjectileModifier::SimpleModify(ProjectileModifierType::Height, height)
                        .get_effect_value();
                let lifetime =
                    ProjectileModifier::SimpleModify(ProjectileModifierType::Lifetime, lifetime)
                        .get_effect_value();
                let _gravity =
                    ProjectileModifier::SimpleModify(ProjectileModifierType::Gravity, gravity)
                        .get_effect_value();
                let health =
                    ProjectileModifier::SimpleModify(ProjectileModifierType::Health, health)
                        .get_effect_value();
                let trail_value = trail_value * lifetime;

                trail_value
                    + if is_direct_shot {
                        0.002
                            * hit_value
                            * (1.0 + speed.abs() * lifetime).sqrt()
                            * (1.0 + width * height + length)
                            * (1.0 + health)
                            + 0.02
                                * lifetime
                                * (1.0 + width * height + length)
                                * (1.0 + health)
                                * if friendly_fire { 1.0 } else { 2.0 }
                    } else {
                        hit_value * (1.0 + health)
                            + 0.02
                                * lifetime
                                * (1.0 + width * height + length)
                                * (1.0 + health)
                                * if friendly_fire { 1.0 } else { 2.0 }
                    }
            }
            BaseCard::MultiCast(cards, modifiers) => {
                let mut value = cards
                    .iter()
                    .map(|card| card.evaluate_value(is_direct_shot))
                    .sum::<f32>();
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
                Effect::Knockback(knockback) => 0.3 * (*knockback as f32).abs(),
                Effect::StatusEffect(effect_type, duration) => match effect_type {
                    StatusEffect::Speed => 0.5 * (*duration as f32),
                    StatusEffect::Slow => (if is_direct_shot {-1.0} else {1.0}) * 0.5 * (*duration as f32),
                    StatusEffect::DamageOverTime => (if is_direct_shot {-1.0} else {1.0}) * 7.0 * (*duration as f32),
                    StatusEffect::HealOverTime => 7.0 * (*duration as f32),
                    StatusEffect::IncreaceDamageTaken => 5.0 * (*duration as f32),
                    StatusEffect::DecreaceDamageTaken => 5.0 * (*duration as f32),
                },
            },
        }
    }

    pub fn is_reasonable(&self) -> bool {
        match self {
            BaseCard::Projectile(modifiers) => {
                for modifier in modifiers {
                    match modifier {
                        ProjectileModifier::SimpleModify(ProjectileModifierType::Speed, _) => {
                            if modifier.get_effect_value().abs() > 200.0 {
                                return false;
                            }
                        }
                        ProjectileModifier::SimpleModify(_, s) => {
                            if *s > 15 {
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
                        ProjectileModifier::Trail(_, card) => {
                            if !card.is_reasonable() {
                                return false;
                            }
                        }
                        ProjectileModifier::NoEnemyFire => {}
                        ProjectileModifier::FriendlyFire => {}
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
                Effect::StatusEffect(_, duration) => {
                    if *duration > 15 {
                        return false;
                    }
                }
            },
        }
        return true;
    }

    pub fn take_modifier(&mut self, path: &mut VecDeque<u32>) -> ProjectileModifier {
        match self {
            BaseCard::Projectile(modifiers) => {
                let idx = path.pop_front().unwrap() as usize;
                if path.is_empty() {
                    let value = modifiers[idx].clone();
                    modifiers[idx] =
                        ProjectileModifier::SimpleModify(ProjectileModifierType::Speed, 0);
                    value
                } else {
                    match modifiers[idx] {
                        ProjectileModifier::OnHit(ref mut card) => card.take_modifier(path),
                        ProjectileModifier::OnExpiry(ref mut card) => card.take_modifier(path),
                        _ => panic!("Invalid state"),
                    }
                }
            }
            BaseCard::MultiCast(cards, _modifiers) => {
                let idx = path.pop_front().unwrap() as usize;
                if path.is_empty() {
                    panic!("Invalid state")
                } else {
                    cards[idx].take_modifier(path)
                }
            }
            _ => panic!("Invalid state"),
        }
    }

    pub fn insert_modifier(&mut self, path: &mut VecDeque<u32>, item: ProjectileModifier) {
        match self {
            BaseCard::Projectile(modifiers) => {
                if path.is_empty() {
                    modifiers.push(item);
                } else {
                    let idx = path.pop_front().unwrap() as usize;
                    match modifiers[idx] {
                        ProjectileModifier::OnHit(ref mut card) => card.insert_modifier(path, item),
                        ProjectileModifier::OnExpiry(ref mut card) => {
                            card.insert_modifier(path, item)
                        }
                        _ => panic!("Invalid state"),
                    }
                }
            }
            BaseCard::MultiCast(cards, _modifiers) => {
                if path.is_empty() {
                    panic!("Invalid state")
                } else {
                    let idx = path.pop_front().unwrap() as usize;
                    cards[idx].insert_modifier(path, item)
                }
            }
            _ => panic!("Invalid state"),
        }
    }

    pub fn cleanup(&mut self) {
        match self {
            BaseCard::Projectile(modifiers) => {
                // cleanup sub cards
                for modifier in modifiers.iter_mut() {
                    match modifier {
                        ProjectileModifier::OnHit(card) => card.cleanup(),
                        ProjectileModifier::OnExpiry(card) => card.cleanup(),
                        _ => {}
                    }
                }
                // combine modifiers
                let mut new_modifiers = Vec::new();
                for modifier in modifiers.iter() {
                    match modifier {
                        ProjectileModifier::SimpleModify(ty, s) => {
                            if let Some(ProjectileModifier::SimpleModify(last_ty, last_s)) =
                                new_modifiers.last_mut()
                            {
                                if *last_ty == *ty {
                                    *last_s += s;
                                    if *last_s == 0 {
                                        new_modifiers.pop();
                                    }
                                } else if *s != 0 {
                                    new_modifiers
                                        .push(ProjectileModifier::SimpleModify(ty.clone(), *s));
                                }
                            } else if *s != 0 {
                                new_modifiers
                                    .push(ProjectileModifier::SimpleModify(ty.clone(), *s));
                            }
                        }
                        uncombinable => new_modifiers.push(uncombinable.clone()),
                    }
                }
                *modifiers = new_modifiers;
            }
            BaseCard::MultiCast(cards, modifiers) => {
                modifiers.retain(|modifier| match modifier {
                    MultiCastModifier::Spread(s) => *s != 0,
                    MultiCastModifier::Duplication(d) => *d != 0,
                });
                for card in cards.iter_mut() {
                    card.cleanup();
                }
            }
            _ => {}
        }
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
            ProjectileModifier::SimpleModify(ProjectileModifierType::Speed, _) => {
                format!("Speed (+8 per) {}b/s", self.get_effect_value())
            }
            ProjectileModifier::SimpleModify(ProjectileModifierType::Length, _) => {
                format!("Length (+25% per) {}", self.get_effect_value())
            }
            ProjectileModifier::SimpleModify(ProjectileModifierType::Width, _) => {
                format!("Width (+25% per) {}", self.get_effect_value())
            }
            ProjectileModifier::SimpleModify(ProjectileModifierType::Height, _) => {
                format!("Height (+25% per) {}", self.get_effect_value())
            }
            ProjectileModifier::SimpleModify(ProjectileModifierType::Lifetime, _) => {
                format!("Lifetime (+50% per) {}s", self.get_effect_value())
            }
            ProjectileModifier::SimpleModify(ProjectileModifierType::Gravity, _) => {
                format!("Gravity (+2 per) {}b/s/s", self.get_effect_value())
            }
            ProjectileModifier::SimpleModify(ProjectileModifierType::Health, _) => {
                format!("Entity Health (+50% per) {}", self.get_effect_value())
            }
            ProjectileModifier::FriendlyFire => format!("Prevents hitting friendly entities"),
            ProjectileModifier::NoEnemyFire => format!("Prevents hitting enemy entities"),
            ProjectileModifier::OnHit(card) => format!("On Hit {}", card.to_string()),
            ProjectileModifier::OnExpiry(card) => format!("On Expiry {}", card.to_string()),
            ProjectileModifier::Trail(freq, card) => format!("Trail {}: {}", freq, card.to_string()),
        }
    }

    pub fn get_effect_value(&self) -> f32 {
        match self {
            ProjectileModifier::SimpleModify(ProjectileModifierType::Speed, s) => {
                8.0 * (*s as f32 + 3.0)
            }
            ProjectileModifier::SimpleModify(ProjectileModifierType::Length, s) => 1.25f32.powi(*s),
            ProjectileModifier::SimpleModify(ProjectileModifierType::Width, s) => 1.25f32.powi(*s),
            ProjectileModifier::SimpleModify(ProjectileModifierType::Height, s) => 1.25f32.powi(*s),
            ProjectileModifier::SimpleModify(ProjectileModifierType::Lifetime, s) => {
                3.0 * 1.5f32.powi(*s)
            }
            ProjectileModifier::SimpleModify(ProjectileModifierType::Gravity, s) => {
                2.0 * (*s as f32)
            }
            ProjectileModifier::SimpleModify(ProjectileModifierType::Health, s) => {
                10.0 * 1.5f32.powi(*s)
            }
            ProjectileModifier::FriendlyFire => panic!(),
            ProjectileModifier::NoEnemyFire => panic!(),
            ProjectileModifier::OnHit(_) => panic!(),
            ProjectileModifier::OnExpiry(_) => panic!(),
            ProjectileModifier::Trail(_, _) => panic!(),
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
    pub trail: Vec<(f32, ReferencedBaseCard)>,
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
                let mut friendly_fire = false;
                let mut no_enemy_fire = false;
                let mut on_hit = Vec::new();
                let mut on_expiry = Vec::new();
                let mut trail = Vec::new();
                for modifier in modifiers {
                    match modifier {
                        ProjectileModifier::SimpleModify(ProjectileModifierType::Speed, s) => {
                            speed += s
                        }
                        ProjectileModifier::SimpleModify(ProjectileModifierType::Length, s) => {
                            length += s
                        }
                        ProjectileModifier::SimpleModify(ProjectileModifierType::Width, s) => {
                            width += s
                        }
                        ProjectileModifier::SimpleModify(ProjectileModifierType::Height, s) => {
                            height += s
                        }
                        ProjectileModifier::SimpleModify(ProjectileModifierType::Lifetime, l) => {
                            lifetime += l
                        }
                        ProjectileModifier::SimpleModify(ProjectileModifierType::Gravity, g) => {
                            gravity += g
                        }
                        ProjectileModifier::SimpleModify(ProjectileModifierType::Health, g) => {
                            health += g
                        }
                        ProjectileModifier::FriendlyFire => friendly_fire = true,
                        ProjectileModifier::NoEnemyFire => no_enemy_fire = true,
                        ProjectileModifier::OnHit(card) => {
                            if let BaseCard::Effect(Effect::Damage(proj_damage)) = card {
                                damage += proj_damage;
                            }
                            on_hit.push(self.register_base_card(card))
                        }
                        ProjectileModifier::OnExpiry(card) => {
                            on_expiry.push(self.register_base_card(card))
                        }
                        ProjectileModifier::Trail(freq, card) => {
                            trail.push((1.0/(freq as f32), self.register_base_card(card)))
                        }
                    }
                }
                self.referenced_projs.push(ReferencedProjectile {
                    damage,
                    speed: ProjectileModifier::SimpleModify(ProjectileModifierType::Speed, speed)
                        .get_effect_value(),
                    length: ProjectileModifier::SimpleModify(
                        ProjectileModifierType::Length,
                        length,
                    )
                    .get_effect_value(),
                    width: ProjectileModifier::SimpleModify(ProjectileModifierType::Width, width)
                        .get_effect_value(),
                    height: ProjectileModifier::SimpleModify(
                        ProjectileModifierType::Height,
                        height,
                    )
                    .get_effect_value(),
                    lifetime: ProjectileModifier::SimpleModify(
                        ProjectileModifierType::Lifetime,
                        lifetime,
                    )
                    .get_effect_value(),
                    gravity: ProjectileModifier::SimpleModify(
                        ProjectileModifierType::Gravity,
                        gravity,
                    )
                    .get_effect_value(),
                    health: ProjectileModifier::SimpleModify(
                        ProjectileModifierType::Health,
                        health,
                    )
                    .get_effect_value(),
                    no_friendly_fire: !friendly_fire,
                    no_enemy_fire,
                    on_hit,
                    on_expiry,
                    trail,
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
                let spread: f32 = multicast.spread as f32 / 15.0;
                let count = 2u32.pow(multicast.duplication);
                let rotation_factor = 2.4;
                for i in 0..count {
                    for sub_projectile in individual_sub_projectiles.iter() {
                        let sub_rot = Quaternion::from(sub_projectile.dir);
                        let mut new_sub_projectile = sub_projectile.clone();
                        let x_rot = spread
                            * (i as f32 / count as f32).sqrt()
                            * (rotation_factor * (i as f32)).cos();
                        let y_rot = spread
                            * (i as f32 / count as f32).sqrt()
                            * (rotation_factor * (i as f32)).sin();
                        let new_rot = sub_rot
                            * Quaternion::from_axis_angle([0.0, 1.0, 0.0].into(), Rad(x_rot))
                            * Quaternion::from_axis_angle([1.0, 0.0, 0.0].into(), Rad(y_rot));

                        new_sub_projectile.dir =
                            [new_rot.v[0], new_rot.v[1], new_rot.v[2], new_rot.s];
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
