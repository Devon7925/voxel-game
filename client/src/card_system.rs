use std::collections::{HashMap, VecDeque};

use cgmath::{Point3, Quaternion, Rad, Rotation3};
use serde::{Deserialize, Serialize};

use crate::{gui::ModificationType, voxel_sim_manager::Projectile, settings_manager::Control};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Cooldown {
    pub modifiers: Vec<CooldownModifier>,
    pub abilities: Vec<Ability>,
    pub cached_cooldown: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum CooldownModifier {
    SimpleCooldownModifier(SimpleCooldownModifier, u32),
}

impl CooldownModifier {
    pub fn get_hover_text(&self) -> String {
        match self {
            CooldownModifier::SimpleCooldownModifier(SimpleCooldownModifier::AddCharge, s) => format!("Add {} charges", s),
            CooldownModifier::SimpleCooldownModifier(SimpleCooldownModifier::AddCooldown, s) => format!("Increace cooldown by {}s ({} per)", SimpleCooldownModifier::ADD_COOLDOWN_AMOUNT*(*s as f32), SimpleCooldownModifier::ADD_COOLDOWN_AMOUNT),
            CooldownModifier::SimpleCooldownModifier(SimpleCooldownModifier::MultiplyImpact, s) => format!("Multiply impact by {}, this lowers the cooldown of this abiliy, but increaces the cooldown of all other abilities", s),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub enum SimpleCooldownModifier {
    AddCharge,
    AddCooldown,
    MultiplyImpact,
}

impl SimpleCooldownModifier {
    const ADD_COOLDOWN_AMOUNT: f32 = 0.5;
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Ability {
    pub card: BaseCard,
    pub keybind: Keybind,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum Keybind {
    Pressed(Control),
    OnPressed(Control),
    OnReleased(Control),
    IsOnGround,
    And(Box<Keybind>, Box<Keybind>),
    Or(Box<Keybind>, Box<Keybind>),
    Not(Box<Keybind>),
    True,
}

impl Keybind {
    pub fn get_simple_representation(&self) -> Option<String> {
        match self {
            Keybind::Pressed(control) => Some(format!("{}", control)),
            Keybind::OnPressed(control) => Some(format!("{}", control)),
            Keybind::OnReleased(control) => Some(format!("{}", control)),
            Keybind::IsOnGround => None,
            Keybind::And(_, _) => None,
            Keybind::Or(_, _) => None,
            Keybind::Not(_) => None,
            Keybind::True => Some("⨀".to_string()),
        }
    }
}

impl Cooldown {
    pub fn vec_from_string(ron_string: &str) -> Vec<Self> {
        ron::from_str(ron_string).unwrap()
    }

    pub fn is_reasonable(&self) -> bool {
        self.abilities
            .iter()
            .all(|ability| ability.card.is_reasonable())
    }

    pub fn get_and_cache_cooldown(&mut self, total_impact: f32) -> f32 {
        if let Some(cooldown) = self.cached_cooldown {
            return cooldown;
        }
        let cooldown = self.get_cooldown_recovery(total_impact).0;
        self.cached_cooldown = Some(cooldown);
        cooldown
    }

    const GLOBAL_COOLDOWN_MULTIPLIER: f32 = 0.3;
    pub fn get_cooldown_recovery(&self, total_impact: f32) -> (f32, Vec<f32>) {
        let ability_values: Vec<f32> = self
            .abilities
            .iter()
            .map(|ability| ability.card.get_cooldown())
            .collect();
        let sum: f32 = ability_values.iter().sum();
        let max: f32 = ability_values
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            .clone();
        let mut ability_charges = 0;
        let mut added_cooldown = 0;
        let mut impact_multiplier = 1.0;
        for modifier in self.modifiers.iter() {
            match modifier {
                CooldownModifier::SimpleCooldownModifier(SimpleCooldownModifier::AddCharge, s) => {
                    ability_charges += s;
                }
                CooldownModifier::SimpleCooldownModifier(
                    SimpleCooldownModifier::AddCooldown,
                    s,
                ) => {
                    added_cooldown += s;
                }
                CooldownModifier::SimpleCooldownModifier(
                    SimpleCooldownModifier::MultiplyImpact,
                    s,
                ) => {
                    impact_multiplier += 0.25 * *s as f32;
                }
            }
        }
        let cooldown = SimpleCooldownModifier::ADD_COOLDOWN_AMOUNT * added_cooldown as f32
            + Self::GLOBAL_COOLDOWN_MULTIPLIER * (sum + 2.0 * max) / 3.0
                * (1.0 + 0.25 * (ability_charges as f32 + 1.0).ln())
                / (impact_multiplier / total_impact);
        let recovery = ability_values
            .iter()
            .map(|val| {
                Self::GLOBAL_COOLDOWN_MULTIPLIER * 0.5 * added_cooldown as f32
                    + val * (1.0 - (-(ability_charges as f32 / 10.0)).exp())
                        / (impact_multiplier / total_impact)
            })
            .collect();
        (cooldown, recovery)
    }

    pub fn get_impact_multiplier(&self) -> f32 {
        let mut impact_multiplier = 1.0;
        for modifier in self.modifiers.iter() {
            match modifier {
                CooldownModifier::SimpleCooldownModifier(
                    SimpleCooldownModifier::MultiplyImpact,
                    s,
                ) => {
                    impact_multiplier += 0.25 * *s as f32;
                }
                _ => {}
            }
        }
        impact_multiplier
    }

    pub fn modify_from_path(
        &mut self,
        path: &mut VecDeque<u32>,
        modification_type: &ModificationType,
    ) {
        self.cached_cooldown = None;
        let type_idx = path.pop_front().unwrap() as usize;
        if type_idx == 0 {
            let idx = path.pop_front().unwrap() as usize;
            assert!(path.is_empty());
            match &mut self.modifiers[idx] {
                CooldownModifier::SimpleCooldownModifier(_, v) => match modification_type {
                    ModificationType::Add => *v += 1,
                    ModificationType::Remove => {
                        if *v > 1 {
                            *v -= 1
                        }
                    }
                },
            }
        } else if type_idx == 1 {
            let idx = path.pop_front().unwrap() as usize;
            self.abilities[idx]
                .card
                .modify_from_path(path, modification_type)
        } else {
            panic!("Invalid state");
        }
    }

    pub fn take_from_path(&mut self, path: &mut VecDeque<u32>) -> DraggableCard {
        self.cached_cooldown = None;
        let type_idx = path.pop_front().unwrap() as usize;
        if type_idx == 0 {
            let idx = path.pop_front().unwrap() as usize;
            assert!(path.is_empty());
            let modifier = self.modifiers[idx].clone();
            self.modifiers[idx] =
                CooldownModifier::SimpleCooldownModifier(SimpleCooldownModifier::AddCharge, 0);
            DraggableCard::CooldownModifier(modifier)
        } else if type_idx == 1 {
            let idx = path.pop_front().unwrap() as usize;
            if path.is_empty() {
                let ability_card = self.abilities[idx].card.clone();
                self.abilities[idx].card = BaseCard::None;
                DraggableCard::BaseCard(ability_card)
            } else {
                self.abilities[idx].card.take_from_path(path)
            }
        } else {
            panic!("Invalid state");
        }
    }

    pub fn insert_to_path(&mut self, path: &mut VecDeque<u32>, item: DraggableCard) {
        self.cached_cooldown = None;
        if path.is_empty() {
            if let DraggableCard::BaseCard(item) = item {
                self.abilities.push(Ability {
                    card: item,
                    keybind: Keybind::Not(Box::new(Keybind::True)),
                });
            } else if let DraggableCard::CooldownModifier(modifier_item) = item {
                let mut combined = false;
                match modifier_item.clone() {
                    CooldownModifier::SimpleCooldownModifier(last_type, last_s) => {
                        for modifier in self.modifiers.iter_mut() {
                            match modifier {
                                CooldownModifier::SimpleCooldownModifier(current_type, s)
                                    if *current_type == last_type =>
                                {
                                    *s += last_s;
                                    combined = true;
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                }

                if !combined {
                    self.modifiers.push(modifier_item.clone());
                }
            } else {
                panic!("Invalid state")
            }
        } else {
            assert!(path.pop_front().unwrap() == 1);
            let idx = path.pop_front().unwrap() as usize;
            self.abilities[idx].card.insert_to_path(path, item)
        }
    }

    pub fn cleanup(&mut self, path: &mut VecDeque<u32>) {
        self.cached_cooldown = None;
        if path.is_empty() {
            return;
        }
        let idx_type = path.pop_front().unwrap();
        if idx_type == 0 {
            let idx = path.pop_front().unwrap() as usize;
            assert!(path.is_empty());
            match self.modifiers[idx] {
                CooldownModifier::SimpleCooldownModifier(_, s) => {
                    if s == 0 {
                        self.modifiers.remove(idx);
                    }
                }
            }
        } else if idx_type == 1 {
            let idx = path.pop_front().unwrap() as usize;
            if path.is_empty() {
                // do nothing
            } else {
                self.abilities[idx].card.cleanup(path);
            }
        } else {
            panic!("Invalid state");
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum BaseCard {
    Projectile(Vec<ProjectileModifier>),
    MultiCast(Vec<BaseCard>, Vec<MultiCastModifier>),
    CreateMaterial(VoxelMaterial),
    Effect(Effect),
    Trigger(u32),
    Palette(Vec<DraggableCard>, bool),
    None,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum ProjectileModifier {
    SimpleModify(ProjectileModifierType, i32),
    FriendlyFire,
    NoEnemyFire,
    OnHit(BaseCard),
    OnHeadshot(BaseCard),
    OnExpiry(BaseCard),
    OnTrigger(u32, BaseCard),
    Trail(u32, BaseCard),
    LockToOwner,
    PiercePlayers,
    WallBounce,
}

impl ProjectileModifier {
    pub fn is_advanced(&self) -> bool {
        match self {
            ProjectileModifier::SimpleModify(_, _) => false,
            ProjectileModifier::FriendlyFire => false,
            ProjectileModifier::NoEnemyFire => false,
            ProjectileModifier::OnHit(_) => true,
            ProjectileModifier::OnHeadshot(_) => true,
            ProjectileModifier::OnExpiry(_) => true,
            ProjectileModifier::OnTrigger(_, _) => true,
            ProjectileModifier::Trail(_, _) => true,
            ProjectileModifier::LockToOwner => false,
            ProjectileModifier::PiercePlayers => false,
            ProjectileModifier::WallBounce => false,
        }
    }
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
impl MultiCastModifier {
    pub fn get_hover_text(&self) -> String {
        match self {
            MultiCastModifier::Spread(s) => format!("Increace spread by {}", s),
            MultiCastModifier::Duplication(s) => {
                format!("Create {} copies of the projectile", 2_u32.pow(*s))
            }
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum VoxelMaterial {
    Air,
    Stone,
    Unloaded,
    Dirt,
    Grass,
    Projectile,
    Ice,
    Glass,
    Player,
    UnloadedAir,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum Effect {
    Cleanse,
    Teleport,
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
    IncreaceGravity,
    DecreaceGravity,
    Overheal,
    Invincibility,
    Trapped,
    Lockout,
    OnHit(Box<BaseCard>),
}

impl VoxelMaterial {
    pub const FRICTION_COEFFICIENTS: [f32; 10] = [0.0, 5.0, 0.0, 5.0, 5.0, 0.0, 0.1, 1.0, 0.0, 0.0];
    pub fn to_memory(&self) -> u32 {
        match self {
            VoxelMaterial::Air => 0 << 24,
            VoxelMaterial::Stone => 1 << 24,
            VoxelMaterial::Unloaded => 2 << 24,
            VoxelMaterial::Dirt => 3 << 24,
            VoxelMaterial::Grass => 4 << 24,
            VoxelMaterial::Projectile => 5 << 24,
            VoxelMaterial::Ice => 6 << 24,
            VoxelMaterial::Glass => 7 << 24,
            VoxelMaterial::Player => 8 << 24,
            VoxelMaterial::UnloadedAir => 9 << 24,
        }
    }
}

#[derive(Debug, Clone)]
struct CardValue {
    damage: f32,
    generic: f32,
    range_probabilities: [f32; 10],
}

fn convolve_range_probabilities(a_probs: [f32; 10], b_props: [f32; 10]) -> [f32; 10] {
    let mut new_range_probabilities = [0.0; 10];
    for a_idx in 0..10 {
        for b_idx in 0..10 {
            if a_idx + b_idx >= 10 {
                break;
            }
            new_range_probabilities[a_idx + b_idx] += a_probs[a_idx] * b_props[b_idx];
        }
    }
    new_range_probabilities
}

const SCALE: i32 = 10;
fn gen_cooldown_for_ttk(accuracy: f32, damage: f32, goal_ttk: f32) -> f32 {
    let mut healing = 128;
    let mut delta = healing;
    while delta > 1 {
        delta /= 2;
        if get_avg_ttk(
            accuracy,
            (damage * SCALE as f32) as i32,
            healing,
            SCALE * 100,
            100,
            &mut HashMap::new(),
        ) * healing as f32
            / 128.0
            > goal_ttk
        {
            healing -= delta;
        } else {
            healing += delta;
        }
    }
    healing as f32 / SCALE as f32 / 12.8
}

fn get_avg_ttk(
    accuracy: f32,
    damage: i32,
    healing: i32,
    current_health: i32,
    iterations: usize,
    table: &mut HashMap<i32, (f32, usize)>,
) -> f32 {
    if current_health <= healing {
        0.0
    } else if iterations == 0 {
        0.0
    } else if current_health > 100 * SCALE {
        get_avg_ttk(accuracy, damage, healing, 100 * SCALE, iterations, table)
    } else {
        if let Some((cached_result, cached_iterations)) = table.get(&current_health) {
            if cached_iterations >= &iterations {
                return *cached_result;
            }
        }
        let result = 1.0
            + accuracy
                * get_avg_ttk(
                    accuracy,
                    damage,
                    healing,
                    current_health - damage + healing,
                    iterations - 1,
                    table,
                )
            + (1.0 - accuracy)
                * get_avg_ttk(
                    accuracy,
                    damage,
                    healing,
                    current_health + healing,
                    iterations - 1,
                    table,
                );
        table.insert(current_health, (result, iterations));
        result
    }
}

impl BaseCard {
    pub const EFFECT_LENGTH_SCALE: f32 = 0.5;
    pub fn from_string(ron_string: &str) -> Self {
        ron::from_str(ron_string).unwrap()
    }

    pub fn to_string(&self) -> String {
        ron::to_string(self).unwrap()
    }

    pub fn get_cooldown(&self) -> f32 {
        let card_value = self.evaluate_value(true);
        card_value
            .iter()
            .map(|card_value| {
                if card_value.damage == 0.0 {
                    return card_value.generic;
                }
                let damage_value: f32 = card_value
                    .range_probabilities
                    .iter()
                    .map(|accuracy| gen_cooldown_for_ttk(*accuracy, card_value.damage, 3.5))
                    .sum::<f32>()
                    / card_value.range_probabilities.len() as f32;
                damage_value + card_value.generic
            })
            .sum::<f32>()
    }

    fn evaluate_value(&self, is_direct: bool) -> Vec<CardValue> {
        const RANGE_PROBABILITIES_SCALE: f32 = 10.0;
        match self {
            BaseCard::Projectile(modifiers) => {
                let mut hit_value = vec![];
                let mut headshot_value = vec![];
                let mut expiry_value = vec![];
                let mut trigger_value = vec![];
                let mut trail_value = vec![];
                let mut speed = 0;
                let mut length = 0;
                let mut width = 0;
                let mut height = 0;
                let mut lifetime = 0;
                let mut gravity = 0;
                let mut health = 0;
                let mut friendly_fire = false;
                let mut enemy_fire = true;
                let mut pierce_players = false;
                let mut wall_bounce = false;
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
                        ProjectileModifier::WallBounce => wall_bounce = true,
                        ProjectileModifier::OnHit(card) => {
                            hit_value.extend(card.evaluate_value(false))
                        }
                        ProjectileModifier::OnHeadshot(card) => {
                            headshot_value.extend(card.evaluate_value(false))
                        }
                        ProjectileModifier::OnExpiry(card) => {
                            expiry_value.extend(card.evaluate_value(is_direct));
                        }
                        ProjectileModifier::OnTrigger(_id, card) => {
                            trigger_value.extend(card.evaluate_value(is_direct));
                        }
                        ProjectileModifier::Trail(freq, card) => {
                            trail_value.extend(
                                card.evaluate_value(false)
                                    .into_iter()
                                    .map(|v| (v, *freq as f32)),
                            );
                        }
                        ProjectileModifier::LockToOwner => {}
                        ProjectileModifier::PiercePlayers => pierce_players = true,
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

                let range_probabilities = core::array::from_fn(|idx| {
                    if RANGE_PROBABILITIES_SCALE * idx as f32 > speed.abs() * lifetime {
                        return 0.0;
                    }
                    (0.06
                        * (1.0 + speed.abs() / idx as f32).sqrt()
                        * (1.0 + width * height + length)
                        * (1.0 + health))
                        .min(1.0)
                });
                let headshot_range_probabilities = core::array::from_fn(|idx| {
                    if RANGE_PROBABILITIES_SCALE * idx as f32 > speed.abs() * lifetime {
                        return 0.0;
                    }
                    (0.02
                        * (1.0 + speed.abs() / idx as f32).sqrt()
                        * (1.0 + width * height + length)
                        * (1.0 + health))
                        .min(1.0)
                });
                let mut value = vec![];
                if enemy_fire {
                    value.push(CardValue {
                        damage: 0.0,
                        generic: 0.02
                            * lifetime
                            * (width * height + length * height + length * width)
                            * (1.0 + health)
                            * if friendly_fire { 1.0 } else { 2.0 },
                        range_probabilities,
                    });
                }
                value.extend(trail_value.into_iter().flat_map(|(value, freq)| {
                    (1..(freq * lifetime) as usize).map(move |idx| {
                        let dist = idx as f32 / freq * speed;
                        let int_dist = (dist / RANGE_PROBABILITIES_SCALE) as usize;
                        let trail_range_probabilities =
                            core::array::from_fn(|idx| if idx == int_dist { 1.0 } else { 0.0 });
                        CardValue {
                            damage: value.damage,
                            generic: value.generic,
                            range_probabilities: convolve_range_probabilities(
                                trail_range_probabilities,
                                value.range_probabilities,
                            ),
                        }
                    })
                }));
                value.extend(hit_value.into_iter().map(|hit_value| CardValue {
                    damage: hit_value.damage,
                    generic: hit_value.generic,
                    range_probabilities: convolve_range_probabilities(
                        range_probabilities,
                        hit_value.range_probabilities,
                    ),
                }));
                value.extend(headshot_value.into_iter().map(|headshot_value| CardValue {
                    damage: headshot_value.damage,
                    generic: headshot_value.generic,
                    range_probabilities: convolve_range_probabilities(
                        headshot_range_probabilities,
                        headshot_value.range_probabilities,
                    ),
                }));
                let expiry_range_probabilities: [f32; 10] = core::array::from_fn(|idx| {
                    if idx == (lifetime * speed / RANGE_PROBABILITIES_SCALE) as usize {
                        1.0
                    } else {
                        0.0
                    }
                });
                value.extend(expiry_value.into_iter().map(|expiry_value| CardValue {
                    damage: expiry_value.damage,
                    generic: expiry_value.generic,
                    range_probabilities: convolve_range_probabilities(
                        expiry_range_probabilities,
                        expiry_value.range_probabilities,
                    ),
                }));
                let trigger_range_probabilities: [f32; 10] = core::array::from_fn(|idx| {
                    if idx <= (lifetime * speed / RANGE_PROBABILITIES_SCALE) as usize {
                        1.0
                    } else {
                        0.0
                    }
                });
                value.extend(trigger_value.into_iter().map(|trigger_value| CardValue {
                    damage: trigger_value.damage,
                    generic: trigger_value.generic,
                    range_probabilities: convolve_range_probabilities(
                        trigger_range_probabilities,
                        trigger_value.range_probabilities,
                    ),
                }));

                value
            }
            BaseCard::MultiCast(cards, modifiers) => {
                let mut duplication = 1;
                let mut spread = 0;
                for modifier in modifiers.iter() {
                    match modifier {
                        MultiCastModifier::Duplication(mod_duplication) => {
                            duplication *= 2_u32.pow(*mod_duplication);
                        }
                        MultiCastModifier::Spread(mod_spread) => {
                            spread += *mod_spread;
                        }
                    }
                }
                cards
                    .iter()
                    .flat_map(|card| card.evaluate_value(is_direct))
                    .flat_map(|card_value| {
                        let mut values = vec![card_value.clone()];
                        if duplication > 1 {
                            values.push(CardValue {
                                damage: (duplication - 1) as f32 * card_value.damage,
                                generic: (duplication - 1) as f32 * card_value.generic,
                                range_probabilities: if spread > 0 {
                                    let mut i = -1;
                                    card_value.range_probabilities.map(|prob| {
                                        i += 1;
                                        if i == 0 {
                                            return prob;
                                        }
                                        prob / (i as f32 * spread as f32).powi(2)
                                    })
                                } else {
                                    card_value.range_probabilities
                                },
                            });
                        };
                        values
                    })
                    .collect()
            }
            BaseCard::CreateMaterial(material) => {
                let material_value = match material {
                    VoxelMaterial::Air => 0.0,
                    VoxelMaterial::Stone => 1.0,
                    VoxelMaterial::Unloaded => panic!("Invalid state"),
                    VoxelMaterial::Dirt => 0.5,
                    VoxelMaterial::Grass => 0.5,
                    VoxelMaterial::Projectile => panic!("Invalid state"),
                    VoxelMaterial::Ice => 2.0,
                    VoxelMaterial::Glass => 1.5,
                    VoxelMaterial::Player => panic!("Invalid state"),
                    VoxelMaterial::UnloadedAir => panic!("Invalid state"),
                };
                vec![CardValue {
                    damage: 0.0,
                    generic: material_value,
                    range_probabilities: core::array::from_fn(
                        |idx| if idx == 0 { 1.0 } else { 0.0 },
                    ),
                }]
            }
            BaseCard::Effect(effect) => match effect {
                Effect::Damage(damage) => {
                    if *damage > 0 {
                        vec![CardValue {
                            damage: *damage as f32,
                            generic: 0.0,
                            range_probabilities: core::array::from_fn(|idx| {
                                if idx == 0 {
                                    1.0
                                } else {
                                    0.0
                                }
                            }),
                        }]
                    } else {
                        vec![CardValue {
                            damage: 0.0,
                            generic: -*damage as f32,
                            range_probabilities: core::array::from_fn(|idx| {
                                if idx == 0 {
                                    1.0
                                } else {
                                    0.0
                                }
                            }),
                        }]
                    }
                }
                Effect::Knockback(knockback) => vec![CardValue {
                    damage: 0.0,
                    generic: 0.3 * (*knockback as f32).abs(),
                    range_probabilities: core::array::from_fn(
                        |idx| if idx == 0 { 1.0 } else { 0.0 },
                    ),
                }],
                Effect::Cleanse => vec![CardValue {
                    damage: 0.0,
                    generic: 7.0,
                    range_probabilities: core::array::from_fn(
                        |idx| if idx == 0 { 1.0 } else { 0.0 },
                    ),
                }],
                Effect::Teleport => vec![CardValue {
                    damage: 0.0,
                    generic: 12.0,
                    range_probabilities: core::array::from_fn(
                        |idx| if idx == 0 { 1.0 } else { 0.0 },
                    ),
                }],
                Effect::StatusEffect(effect_type, duration) => match effect_type {
                    StatusEffect::Speed => vec![CardValue {
                        damage: 0.0,
                        generic: Self::EFFECT_LENGTH_SCALE * 0.5 * (*duration as f32),
                        range_probabilities: core::array::from_fn(
                            |idx| if idx == 0 { 1.0 } else { 0.0 },
                        ),
                    }],
                    StatusEffect::Slow => vec![CardValue {
                        damage: 0.0,
                        generic: Self::EFFECT_LENGTH_SCALE
                            * (if is_direct { -1.0 } else { 1.0 })
                            * 0.3
                            * (*duration as f32),
                        range_probabilities: core::array::from_fn(
                            |idx| if idx == 0 { 1.0 } else { 0.0 },
                        ),
                    }],
                    StatusEffect::DamageOverTime => {
                        if is_direct {
                            vec![CardValue {
                                damage: 0.0,
                                generic: Self::EFFECT_LENGTH_SCALE * -1.0 * (*duration as f32),
                                range_probabilities: core::array::from_fn(|idx| {
                                    if idx == 0 {
                                        1.0
                                    } else {
                                        0.0
                                    }
                                }),
                            }]
                        } else {
                            vec![CardValue {
                                damage: Self::EFFECT_LENGTH_SCALE * 7.0 * (*duration as f32),
                                generic: 0.0,
                                range_probabilities: core::array::from_fn(|idx| {
                                    if idx == 0 {
                                        1.0
                                    } else {
                                        0.0
                                    }
                                }),
                            }]
                        }
                    }
                    StatusEffect::HealOverTime => vec![CardValue {
                        damage: 0.0,
                        generic: Self::EFFECT_LENGTH_SCALE * 1.0 * (*duration as f32),
                        range_probabilities: core::array::from_fn(
                            |idx| if idx == 0 { 1.0 } else { 0.0 },
                        ),
                    }],
                    StatusEffect::IncreaceDamageTaken => vec![CardValue {
                        damage: 0.0,
                        generic: Self::EFFECT_LENGTH_SCALE * 5.0 * (*duration as f32),
                        range_probabilities: core::array::from_fn(
                            |idx| if idx == 0 { 1.0 } else { 0.0 },
                        ),
                    }],
                    StatusEffect::DecreaceDamageTaken => vec![CardValue {
                        damage: 0.0,
                        generic: Self::EFFECT_LENGTH_SCALE * 5.0 * (*duration as f32),
                        range_probabilities: core::array::from_fn(
                            |idx| if idx == 0 { 1.0 } else { 0.0 },
                        ),
                    }],
                    StatusEffect::IncreaceGravity => vec![CardValue {
                        damage: 0.0,
                        generic: Self::EFFECT_LENGTH_SCALE * 0.5 * (*duration as f32),
                        range_probabilities: core::array::from_fn(
                            |idx| if idx == 0 { 1.0 } else { 0.0 },
                        ),
                    }],
                    StatusEffect::DecreaceGravity => vec![CardValue {
                        damage: 0.0,
                        generic: Self::EFFECT_LENGTH_SCALE * 0.5 * (*duration as f32),
                        range_probabilities: core::array::from_fn(
                            |idx| if idx == 0 { 1.0 } else { 0.0 },
                        ),
                    }],
                    StatusEffect::Overheal => vec![CardValue {
                        damage: 0.0,
                        generic: 5.0
                            * (2.0 - (-(Self::EFFECT_LENGTH_SCALE * *duration as f32)).exp()),
                        range_probabilities: core::array::from_fn(
                            |idx| if idx == 0 { 1.0 } else { 0.0 },
                        ),
                    }],
                    StatusEffect::Invincibility => vec![CardValue {
                        damage: 0.0,
                        generic: Self::EFFECT_LENGTH_SCALE * 10.0 * (*duration as f32),
                        range_probabilities: core::array::from_fn(
                            |idx| if idx == 0 { 1.0 } else { 0.0 },
                        ),
                    }],
                    StatusEffect::Trapped => vec![CardValue {
                        damage: 0.0,
                        generic: Self::EFFECT_LENGTH_SCALE
                            * (if is_direct { -1.0 } else { 1.0 })
                            * 5.0
                            * (*duration as f32),
                        range_probabilities: core::array::from_fn(
                            |idx| if idx == 0 { 1.0 } else { 0.0 },
                        ),
                    }],
                    StatusEffect::Lockout => vec![CardValue {
                        damage: 0.0,
                        generic: Self::EFFECT_LENGTH_SCALE
                            * (if is_direct { -1.0 } else { 1.0 })
                            * 5.0
                            * (*duration as f32),
                        range_probabilities: core::array::from_fn(
                            |idx| if idx == 0 { 1.0 } else { 0.0 },
                        ),
                    }],
                    StatusEffect::OnHit(card) => {
                        let range_probabilities: [f32; 10] = core::array::from_fn(|idx| {
                            (0.1 * (9.0 - idx as f32)
                                * (Self::EFFECT_LENGTH_SCALE * *duration as f32))
                                .min(1.0)
                        });
                        let hit_value = card.evaluate_value(false);
                        hit_value
                            .iter()
                            .map(|hit_value| CardValue {
                                damage: hit_value.damage,
                                generic: hit_value.generic,
                                range_probabilities: convolve_range_probabilities(
                                    range_probabilities,
                                    hit_value.range_probabilities,
                                ),
                            })
                            .collect()
                    }
                },
            },
            BaseCard::Trigger(_id) => vec![],
            BaseCard::None => vec![],
            BaseCard::Palette(..) => panic!("Invalid state"),
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
                        ProjectileModifier::OnHeadshot(card) => {
                            if !card.is_reasonable() {
                                return false;
                            }
                        }
                        ProjectileModifier::OnExpiry(card) => {
                            if !card.is_reasonable() {
                                return false;
                            }
                        }
                        ProjectileModifier::OnTrigger(_, card) => {
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
                        ProjectileModifier::LockToOwner => {}
                        ProjectileModifier::PiercePlayers => {}
                        ProjectileModifier::WallBounce => {}
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
                    if knockback.abs() > 40 {
                        return false;
                    }
                }
                Effect::StatusEffect(_, duration) => {
                    if *duration > 15 {
                        return false;
                    }
                }
                Effect::Cleanse => {}
                Effect::Teleport => {}
            },
            BaseCard::Trigger(_) => {}
            BaseCard::None => {}
            BaseCard::Palette(..) => panic!("Invalid state"),
        }
        return true;
    }

    pub fn modify_from_path(
        &mut self,
        path: &mut VecDeque<u32>,
        modification_type: &ModificationType,
    ) {
        match self {
            BaseCard::Projectile(modifiers) => {
                let idx = path.pop_front().unwrap() as usize;
                if path.is_empty() {
                    match modifiers.get_mut(idx).unwrap() {
                        ProjectileModifier::SimpleModify(_type, ref mut value) => {
                            match modification_type {
                                ModificationType::Add => *value += 1,
                                ModificationType::Remove => *value -= 1,
                            }
                        }
                        ProjectileModifier::Trail(ref mut frequency, _card) => {
                            match modification_type {
                                ModificationType::Add => *frequency += 1,
                                ModificationType::Remove => {
                                    if *frequency > 1 {
                                        *frequency -= 1
                                    }
                                }
                            }
                        }
                        ProjectileModifier::OnTrigger(ref mut id, _card) => match modification_type
                        {
                            ModificationType::Add => *id += 1,
                            ModificationType::Remove => {
                                if *id > 0 {
                                    *id -= 1
                                }
                            }
                        },
                        _ => panic!("Invalid State"),
                    }
                } else {
                    assert!(path.pop_front().unwrap() == 0);
                    match modifiers[idx] {
                        ProjectileModifier::OnHit(ref mut card) => {
                            card.modify_from_path(path, modification_type)
                        }
                        ProjectileModifier::OnHeadshot(ref mut card) => {
                            card.modify_from_path(path, modification_type)
                        }
                        ProjectileModifier::OnExpiry(ref mut card) => {
                            card.modify_from_path(path, modification_type)
                        }
                        ProjectileModifier::OnTrigger(_, ref mut card) => {
                            card.modify_from_path(path, modification_type)
                        }
                        ProjectileModifier::Trail(_freqency, ref mut card) => {
                            card.modify_from_path(path, modification_type)
                        }
                        _ => panic!("Invalid state"),
                    }
                }
            }
            BaseCard::MultiCast(cards, modifiers) => {
                let type_idx = path.pop_front().unwrap() as usize;
                if type_idx == 0 {
                    let idx = path.pop_front().unwrap() as usize;
                    assert!(path.is_empty());
                    match modifiers[idx] {
                        MultiCastModifier::Spread(ref mut value) => match modification_type {
                            ModificationType::Add => *value += 1,
                            ModificationType::Remove => {
                                if *value > 1 {
                                    *value -= 1
                                }
                            }
                        },
                        MultiCastModifier::Duplication(ref mut value) => match modification_type {
                            ModificationType::Add => *value += 1,
                            ModificationType::Remove => {
                                if *value > 1 {
                                    *value -= 1
                                }
                            }
                        },
                    }
                } else if type_idx == 1 {
                    let idx = path.pop_front().unwrap() as usize;
                    cards[idx].modify_from_path(path, modification_type)
                } else {
                    panic!("Invalid state");
                }
            }
            BaseCard::CreateMaterial(_) => panic!("Invalid state"),
            BaseCard::Effect(Effect::StatusEffect(StatusEffect::OnHit(card), duration)) => {
                if path.is_empty() {
                    match modification_type {
                        ModificationType::Add => *duration += 1,
                        ModificationType::Remove => {
                            if *duration > 1 {
                                *duration -= 1
                            }
                        }
                    }
                } else {
                    assert!(path.pop_front().unwrap() == 0);
                    card.modify_from_path(path, modification_type)
                }
            }
            BaseCard::Effect(effect) => {
                assert!(path.is_empty());
                match effect {
                    Effect::Damage(ref mut damage) => match modification_type {
                        ModificationType::Add => *damage += 1,
                        ModificationType::Remove => *damage -= 1,
                    },
                    Effect::Knockback(ref mut knockback) => match modification_type {
                        ModificationType::Add => *knockback += 1,
                        ModificationType::Remove => *knockback -= 1,
                    },
                    Effect::StatusEffect(_, ref mut duration) => match modification_type {
                        ModificationType::Add => *duration += 1,
                        ModificationType::Remove => {
                            if *duration > 1 {
                                *duration -= 1
                            }
                        }
                    },
                    Effect::Cleanse => {}
                    Effect::Teleport => {}
                }
            }
            BaseCard::Trigger(ref mut id) => match modification_type {
                ModificationType::Add => *id += 1,
                ModificationType::Remove => {
                    if *id > 0 {
                        *id -= 1
                    }
                }
            },
            BaseCard::None => panic!("Invalid state"),
            BaseCard::Palette(..) => {}
        }
    }

    pub fn take_from_path(&mut self, path: &mut VecDeque<u32>) -> DraggableCard {
        if path.is_empty() {
            let result = DraggableCard::BaseCard(self.clone());
            *self = BaseCard::None;
            return result;
        }
        match self {
            BaseCard::Projectile(modifiers) => {
                let idx = path.pop_front().unwrap() as usize;
                if path.is_empty() {
                    let value = modifiers[idx].clone();
                    modifiers[idx] =
                        ProjectileModifier::SimpleModify(ProjectileModifierType::Speed, 0);
                    DraggableCard::ProjectileModifier(value)
                } else {
                    assert!(path.pop_front().unwrap() == 0);
                    if path.is_empty() {
                        let card_ref = match modifiers.get_mut(idx).unwrap() {
                            ProjectileModifier::OnHit(ref mut card) => card,
                            ProjectileModifier::OnHeadshot(ref mut card) => card,
                            ProjectileModifier::OnExpiry(ref mut card) => card,
                            ProjectileModifier::OnTrigger(_id, ref mut card) => card,
                            ProjectileModifier::Trail(_freqency, ref mut card) => card,
                            invalid_take_modifier => panic!(
                                "Invalid state: cannot take from {:?}",
                                invalid_take_modifier
                            ),
                        };
                        let result = DraggableCard::BaseCard(card_ref.clone());
                        *card_ref = BaseCard::None;
                        result
                    } else {
                        match modifiers[idx] {
                            ProjectileModifier::OnHit(ref mut card) => card.take_from_path(path),
                            ProjectileModifier::OnHeadshot(ref mut card) => {
                                card.take_from_path(path)
                            }
                            ProjectileModifier::OnExpiry(ref mut card) => card.take_from_path(path),
                            ProjectileModifier::OnTrigger(_, ref mut card) => {
                                card.take_from_path(path)
                            }
                            ProjectileModifier::Trail(_freqency, ref mut card) => {
                                card.take_from_path(path)
                            }
                            _ => panic!("Invalid state"),
                        }
                    }
                }
            }
            BaseCard::MultiCast(cards, modifiers) => {
                let type_idx = path.pop_front().unwrap() as usize;
                if type_idx == 0 {
                    let idx = path.pop_front().unwrap() as usize;
                    assert!(path.is_empty());
                    let multicast_modifier = modifiers[idx].clone();
                    modifiers[idx] = MultiCastModifier::Spread(0);
                    DraggableCard::MultiCastModifier(multicast_modifier)
                } else if type_idx == 1 {
                    let idx = path.pop_front().unwrap() as usize;
                    if path.is_empty() {
                        let value = cards[idx].clone();
                        cards[idx] = BaseCard::None;
                        DraggableCard::BaseCard(value)
                    } else {
                        cards[idx].take_from_path(path)
                    }
                } else {
                    panic!("Invalid state");
                }
            }
            BaseCard::Effect(Effect::StatusEffect(StatusEffect::OnHit(card), _)) => {
                assert!(path.pop_front().unwrap() == 0);
                card.take_from_path(path)
            }
            BaseCard::Palette(cards, _) => {
                let card_idx = path.pop_front().unwrap() as usize;
                cards[card_idx].clone()
            }
            invalid_take @ (BaseCard::CreateMaterial(_)
            | BaseCard::None
            | BaseCard::Trigger(_)
            | BaseCard::Effect(_)) => panic!("Invalid state: cannot take from {:?}", invalid_take),
        }
    }

    pub fn insert_to_path(&mut self, path: &mut VecDeque<u32>, item: DraggableCard) {
        match self {
            BaseCard::Projectile(modifiers) => {
                if path.is_empty() {
                    let DraggableCard::ProjectileModifier(item) = item else {
                        panic!("Invalid state")
                    };
                    if let ProjectileModifier::SimpleModify(last_ty, last_s) = item.clone() {
                        let mut combined = false;
                        for modifier in modifiers.iter_mut() {
                            match modifier {
                                ProjectileModifier::SimpleModify(ty, s) => {
                                    if last_ty == *ty {
                                        *s += last_s;
                                        combined = true;
                                        break;
                                    }
                                }
                                _ => {}
                            }
                        }
                        if !combined {
                            modifiers.push(item.clone());
                        }
                    } else {
                        modifiers.push(item.clone());
                    }

                    modifiers.retain(|modifier| match modifier {
                        ProjectileModifier::SimpleModify(_, s) => *s != 0,
                        _ => true,
                    });
                } else {
                    let idx = path.pop_front().unwrap() as usize;
                    assert!(path.pop_front().unwrap() == 0);
                    match modifiers[idx] {
                        ProjectileModifier::OnHit(ref mut card) => card.insert_to_path(path, item),
                        ProjectileModifier::OnHeadshot(ref mut card) => {
                            card.insert_to_path(path, item)
                        }
                        ProjectileModifier::OnExpiry(ref mut card) => {
                            card.insert_to_path(path, item)
                        }
                        ProjectileModifier::OnTrigger(_id, ref mut card) => {
                            card.insert_to_path(path, item)
                        }
                        ProjectileModifier::Trail(_freqency, ref mut card) => {
                            card.insert_to_path(path, item)
                        }
                        _ => panic!("Invalid state"),
                    }
                }
            }
            BaseCard::MultiCast(cards, modifiers) => {
                if path.is_empty() {
                    if let DraggableCard::BaseCard(item) = item {
                        cards.push(item);
                    } else if let DraggableCard::MultiCastModifier(modifier_item) = item {
                        let mut combined = false;
                        match modifier_item.clone() {
                            MultiCastModifier::Duplication(last_s) => {
                                for modifier in modifiers.iter_mut() {
                                    match modifier {
                                        MultiCastModifier::Duplication(s) => {
                                            *s += last_s;
                                            combined = true;
                                            break;
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            MultiCastModifier::Spread(last_s) => {
                                for modifier in modifiers.iter_mut() {
                                    match modifier {
                                        MultiCastModifier::Spread(s) => {
                                            *s += last_s;
                                            combined = true;
                                            break;
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }

                        if !combined {
                            modifiers.push(modifier_item.clone());
                        }
                    } else {
                        panic!("Invalid state")
                    }
                } else {
                    assert!(path.pop_front().unwrap() == 1);
                    let idx = path.pop_front().unwrap() as usize;
                    cards[idx].insert_to_path(path, item)
                }
            }
            BaseCard::Effect(Effect::StatusEffect(StatusEffect::OnHit(card), _)) => {
                assert!(path.pop_front().unwrap() == 0);
                card.insert_to_path(path, item);
            }
            BaseCard::None => {
                assert!(
                    path.is_empty(),
                    "Invalid state: should not have nonempty path {:?} when inserting into None",
                    path
                );
                let DraggableCard::BaseCard(item) = item else {
                    panic!("Invalid state")
                };
                *self = item;
            }
            _ => panic!("Invalid state"),
        }
    }

    pub fn cleanup(&mut self, path: &mut VecDeque<u32>) {
        match self {
            BaseCard::Projectile(modifiers) => {
                if path.len() <= 1 {
                    modifiers.retain(|modifier| match modifier {
                        ProjectileModifier::SimpleModify(_, s) => *s != 0,
                        _ => true,
                    });
                } else {
                    let idx = path.pop_front().unwrap() as usize;
                    assert!(path.pop_front().unwrap() == 0);
                    match modifiers[idx] {
                        ProjectileModifier::OnHit(ref mut card) => card.cleanup(path),
                        ProjectileModifier::OnHeadshot(ref mut card) => card.cleanup(path),
                        ProjectileModifier::OnExpiry(ref mut card) => card.cleanup(path),
                        ProjectileModifier::OnTrigger(_, ref mut card) => card.cleanup(path),
                        ProjectileModifier::Trail(_freqency, ref mut card) => card.cleanup(path),
                        ref invalid => panic!(
                            "Invalid state: cannot follow path {} into {:?}",
                            idx, invalid
                        ),
                    }
                }
            }
            BaseCard::MultiCast(cards, modifiers) => {
                if path.is_empty() {
                    cards.retain(|card| !matches!(card, BaseCard::None));
                } else {
                    let idx_type = path.pop_front().unwrap();
                    if idx_type == 0 {
                        let idx = path.pop_front().unwrap() as usize;
                        assert!(path.is_empty());
                        match modifiers[idx] {
                            MultiCastModifier::Duplication(s) => {
                                if s == 0 {
                                    modifiers.remove(idx);
                                }
                            }
                            MultiCastModifier::Spread(s) => {
                                if s == 0 {
                                    modifiers.remove(idx);
                                }
                            }
                        }
                    } else if idx_type == 1 {
                        let idx = path.pop_front().unwrap() as usize;
                        if path.is_empty() {
                            if matches!(cards[idx], BaseCard::None) {
                                cards.remove(idx);
                            }
                        } else {
                            cards[idx].cleanup(path);
                        }
                    } else {
                        panic!("Invalid state");
                    }
                }
            }
            BaseCard::Effect(Effect::StatusEffect(StatusEffect::OnHit(card), _)) => {
                assert!(path.pop_front().unwrap() == 0);
                card.cleanup(path);
            }
            BaseCard::None => {
                assert!(path.is_empty(), "Invalid state");
            }
            _ => panic!("Invalid state"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DraggableCard {
    ProjectileModifier(ProjectileModifier),
    MultiCastModifier(MultiCastModifier),
    CooldownModifier(CooldownModifier),
    BaseCard(BaseCard),
}

impl Default for BaseCard {
    fn default() -> Self {
        BaseCard::None
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
            ProjectileModifier::OnHeadshot(card) => format!("On Headshot {}", card.to_string()),
            ProjectileModifier::OnExpiry(card) => format!("On Expiry {}", card.to_string()),
            ProjectileModifier::OnTrigger(id, card) => {
                format!("On trigger {} {}", id, card.to_string())
            }
            ProjectileModifier::Trail(freq, card) => {
                format!("Trail {}: {}", freq, card.to_string())
            }
            ProjectileModifier::LockToOwner => {
                format!("Locks the projectile's position to the player's position")
            }
            ProjectileModifier::PiercePlayers => {
                format!(
                    "Allows the projectile to pierce players, potentially hitting multiple players"
                )
            }
            ProjectileModifier::WallBounce => {
                format!("Allows the projectile to bounce off walls")
            }
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
            ProjectileModifier::OnHeadshot(_) => panic!(),
            ProjectileModifier::OnExpiry(_) => panic!(),
            ProjectileModifier::OnTrigger(_, _) => panic!(),
            ProjectileModifier::Trail(_, _) => panic!(),
            ProjectileModifier::LockToOwner => panic!(),
            ProjectileModifier::PiercePlayers => panic!(),
            ProjectileModifier::WallBounce => panic!(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum StateKeybind {
    Pressed(Control, bool),
    OnPressed(Control, bool),
    OnReleased(Control, bool),
    IsOnGround(bool),
    And(Box<StateKeybind>, Box<StateKeybind>),
    Or(Box<StateKeybind>, Box<StateKeybind>),
    Not(Box<StateKeybind>),
    True,
}

impl StateKeybind {
    pub fn update(&mut self, control: &Control, state: bool) {
        match self {
            StateKeybind::Pressed(c, s) => {
                if *c == *control {
                    *s = state;
                }
            }
            StateKeybind::OnPressed(c, s) => {
                if *c == *control && state {
                    *s = true;
                }
            }
            StateKeybind::OnReleased(c, s) => {
                if *c == *control && !state {
                    *s = true;
                }
            }
            StateKeybind::IsOnGround(_) => {}
            StateKeybind::And(a, b) => {
                a.as_mut().update(control, state);
                b.as_mut().update(control, state);
            }
            StateKeybind::Or(a, b) => {
                a.as_mut().update(control, state);
                b.as_mut().update(control, state);
            }
            StateKeybind::Not(a) => {
                a.as_mut().update(control, state);
            }
            StateKeybind::True => {}
        }
    }

    pub fn update_on_ground(&mut self, state: bool) {
        match self {
            StateKeybind::IsOnGround(s) => {
                *s = state;
            }
            StateKeybind::And(a, b) => {
                a.as_mut().update_on_ground(state);
                b.as_mut().update_on_ground(state);
            }
            StateKeybind::Or(a, b) => {
                a.as_mut().update_on_ground(state);
                b.as_mut().update_on_ground(state);
            }
            StateKeybind::Not(a) => {
                a.as_mut().update_on_ground(state);
            }
            _ => {}
        }
    }

    pub fn get_simple_representation(&self) -> Option<String> {
        match self {
            StateKeybind::Pressed(control, _) => Some(format!("{}", control)),
            StateKeybind::OnPressed(control, _) => Some(format!("{}", control)),
            StateKeybind::OnReleased(control, _) => Some(format!("{}", control)),
            StateKeybind::IsOnGround(_) => None,
            StateKeybind::And(_, _) => None,
            StateKeybind::Or(_, _) => None,
            StateKeybind::Not(_) => None,
            StateKeybind::True => Some("⨀".to_string()),
        }
    }

    pub fn get_state(&self) -> bool {
        match self {
            StateKeybind::Pressed(_, s) => *s,
            StateKeybind::OnPressed(_, s) => *s,
            StateKeybind::OnReleased(_, s) => *s,
            StateKeybind::IsOnGround(s) => *s,
            StateKeybind::And(a, b) => a.get_state() && b.get_state(),
            StateKeybind::Or(a, b) => a.get_state() || b.get_state(),
            StateKeybind::Not(a) => !a.get_state(),
            StateKeybind::True => true,
        }
    }

    pub fn clear(&mut self) {
        match self {
            StateKeybind::Pressed(_, _) => {}
            StateKeybind::OnPressed(_, s) => *s = false,
            StateKeybind::OnReleased(_, s) => *s = false,
            StateKeybind::IsOnGround(s) => *s = false,
            StateKeybind::And(a, b) => {
                a.as_mut().clear();
                b.as_mut().clear();
            }
            StateKeybind::Or(a, b) => {
                a.as_mut().clear();
                b.as_mut().clear();
            }
            StateKeybind::Not(a) => {
                a.as_mut().clear();
            }
            StateKeybind::True => {}
        }
    }
}

impl From<Keybind> for StateKeybind {
    fn from(keybind: Keybind) -> Self {
        match keybind {
            Keybind::Pressed(control) => StateKeybind::Pressed(control, false),
            Keybind::OnPressed(control) => StateKeybind::OnPressed(control, false),
            Keybind::OnReleased(control) => StateKeybind::OnReleased(control, false),
            Keybind::IsOnGround => StateKeybind::IsOnGround(false),
            Keybind::And(a, b) => StateKeybind::And(Box::new((*a).into()), Box::new((*b).into())),
            Keybind::Or(a, b) => StateKeybind::Or(Box::new((*a).into()), Box::new((*b).into())),
            Keybind::Not(a) => StateKeybind::Not(Box::new((*a).into())),
            Keybind::True => StateKeybind::True,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ReferencedCooldown {
    pub add_charge: u32,
    pub add_cooldown: u32,
    pub abilities: Vec<(ReferencedBaseCard, Keybind)>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq)]
pub enum ReferencedBaseCardType {
    Projectile,
    MultiCast,
    CreateMaterial,
    Effect,
    Trigger,
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
    pub lock_owner: bool,
    pub pierce_players: bool,
    pub wall_bounce: bool,
    pub on_hit: Vec<ReferencedBaseCard>,
    pub on_headshot: Vec<ReferencedBaseCard>,
    pub on_expiry: Vec<ReferencedBaseCard>,
    pub on_trigger: Vec<(u32, ReferencedBaseCard)>,
    pub trail: Vec<(f32, ReferencedBaseCard)>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ReferencedMulticast {
    pub sub_cards: Vec<ReferencedBaseCard>,
    pub spread: u32,
    pub duplication: u32,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum ReferencedEffect {
    Damage(i32),
    Knockback(i32),
    Cleanse,
    Teleport,
    StatusEffect(ReferencedStatusEffect, u32),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum ReferencedStatusEffect {
    Speed,
    Slow,
    DamageOverTime,
    HealOverTime,
    IncreaceDamageTaken,
    DecreaceDamageTaken,
    IncreaceGravity,
    DecreaceGravity,
    Overheal,
    Invincibility,
    Trapped,
    Lockout,
    OnHit(ReferencedBaseCard),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ReferencedTrigger(pub u32);

pub struct CardManager {
    pub referenced_multicasts: Vec<ReferencedMulticast>,
    pub referenced_projs: Vec<ReferencedProjectile>,
    pub referenced_material_creators: Vec<VoxelMaterial>,
    pub referenced_effects: Vec<ReferencedEffect>,
    pub referenced_triggers: Vec<ReferencedTrigger>,
}

impl Default for CardManager {
    fn default() -> Self {
        CardManager {
            referenced_multicasts: vec![],
            referenced_projs: vec![],
            referenced_material_creators: vec![],
            referenced_effects: vec![],
            referenced_triggers: vec![],
        }
    }
}

impl CardManager {
    pub fn register_cooldown(&mut self, cooldown: Cooldown) -> ReferencedCooldown {
        let mut abilities = Vec::new();
        for ability in cooldown.abilities {
            abilities.push((
                self.register_base_card(ability.card),
                ability.keybind.into(),
            ));
        }
        let mut add_charge = 0;
        let mut add_cooldown = 0;
        for modifier in cooldown.modifiers {
            match modifier {
                CooldownModifier::SimpleCooldownModifier(SimpleCooldownModifier::AddCharge, c) => {
                    add_charge += c
                }
                CooldownModifier::SimpleCooldownModifier(
                    SimpleCooldownModifier::AddCooldown,
                    c,
                ) => add_cooldown += c,
                CooldownModifier::SimpleCooldownModifier(
                    SimpleCooldownModifier::MultiplyImpact,
                    _c,
                ) => {}
            }
        }
        ReferencedCooldown {
            add_charge,
            add_cooldown,
            abilities,
        }
    }

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
                let mut on_headshot = Vec::new();
                let mut on_expiry = Vec::new();
                let mut on_trigger = Vec::new();
                let mut trail = Vec::new();
                let mut lock_owner = false;
                let mut pierce_players = false;
                let mut wall_bounce = false;
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
                        ProjectileModifier::OnHeadshot(card) => {
                            on_headshot.push(self.register_base_card(card))
                        }
                        ProjectileModifier::OnExpiry(card) => {
                            on_expiry.push(self.register_base_card(card))
                        }
                        ProjectileModifier::OnTrigger(id, card) => {
                            on_trigger.push((id, self.register_base_card(card)))
                        }
                        ProjectileModifier::Trail(freq, card) => {
                            trail.push((1.0 / (freq as f32), self.register_base_card(card)))
                        }
                        ProjectileModifier::LockToOwner => lock_owner = true,
                        ProjectileModifier::PiercePlayers => pierce_players = true,
                        ProjectileModifier::WallBounce => wall_bounce = true,
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
                    lock_owner,
                    pierce_players,
                    wall_bounce,
                    on_hit,
                    on_headshot,
                    on_expiry,
                    on_trigger,
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
                let referenced_effect = match effect {
                    Effect::Damage(damage) => ReferencedEffect::Damage(damage),
                    Effect::Knockback(knockback) => ReferencedEffect::Knockback(knockback),
                    Effect::Cleanse => ReferencedEffect::Cleanse,
                    Effect::Teleport => ReferencedEffect::Teleport,
                    Effect::StatusEffect(status, duration) => {
                        let referenced_status = match status {
                            StatusEffect::Speed => ReferencedStatusEffect::Speed,
                            StatusEffect::Slow => ReferencedStatusEffect::Slow,
                            StatusEffect::DamageOverTime => ReferencedStatusEffect::DamageOverTime,
                            StatusEffect::HealOverTime => ReferencedStatusEffect::HealOverTime,
                            StatusEffect::IncreaceDamageTaken => {
                                ReferencedStatusEffect::IncreaceDamageTaken
                            }
                            StatusEffect::DecreaceDamageTaken => {
                                ReferencedStatusEffect::DecreaceDamageTaken
                            }
                            StatusEffect::IncreaceGravity => {
                                ReferencedStatusEffect::IncreaceGravity
                            }
                            StatusEffect::DecreaceGravity => {
                                ReferencedStatusEffect::DecreaceGravity
                            }
                            StatusEffect::Overheal => ReferencedStatusEffect::Overheal,
                            StatusEffect::Invincibility => ReferencedStatusEffect::Invincibility,
                            StatusEffect::Trapped => ReferencedStatusEffect::Trapped,
                            StatusEffect::Lockout => ReferencedStatusEffect::Lockout,
                            StatusEffect::OnHit(card) => {
                                ReferencedStatusEffect::OnHit(self.register_base_card(*card))
                            }
                        };
                        ReferencedEffect::StatusEffect(referenced_status, duration)
                    }
                };
                self.referenced_effects.push(referenced_effect);
                ReferencedBaseCard {
                    card_type: ReferencedBaseCardType::Effect,
                    card_idx: self.referenced_effects.len() - 1,
                }
            }
            BaseCard::Trigger(id) => {
                self.referenced_triggers.push(ReferencedTrigger(id));
                ReferencedBaseCard {
                    card_type: ReferencedBaseCardType::Trigger,
                    card_idx: self.referenced_triggers.len() - 1,
                }
            }
            BaseCard::None => ReferencedBaseCard {
                card_type: ReferencedBaseCardType::None,
                card_idx: 0,
            },
            BaseCard::Palette(..) => panic!("Invalid state"),
        }
    }

    pub fn get_effects_from_base_card(
        &self,
        card: ReferencedBaseCard,
        pos: &Point3<f32>,
        rot: &Quaternion<f32>,
        player_idx: u32,
        is_from_head: bool,
    ) -> (
        Vec<Projectile>,
        Vec<(Point3<u32>, VoxelMaterial)>,
        Vec<ReferencedEffect>,
        Vec<(ReferencedTrigger, u32)>,
    ) {
        let mut projectiles = vec![];
        let mut new_voxels = vec![];
        let mut effects = vec![];
        let mut triggers = vec![];
        match card {
            ReferencedBaseCard {
                card_type: ReferencedBaseCardType::Projectile,
                card_idx,
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
                    wall_bounce: if proj_stats.wall_bounce { 1 } else { 0 },
                    is_from_head: if is_from_head { 1 } else { 0 },
                });
            }
            ReferencedBaseCard {
                card_type: ReferencedBaseCardType::MultiCast,
                card_idx,
            } => {
                let multicast = &self.referenced_multicasts[card_idx];
                let mut individual_sub_projectiles = vec![];
                let mut sub_projectiles = vec![];
                let mut sub_voxels = vec![];
                let mut sub_effects = vec![];
                let mut sub_triggers = vec![];
                for sub_card in multicast.sub_cards.iter() {
                    let (sub_sub_projectiles, sub_sub_voxels, sub_sub_effects, sub_sub_triggers) =
                        self.get_effects_from_base_card(
                            *sub_card,
                            pos,
                            rot,
                            player_idx,
                            is_from_head,
                        );
                    individual_sub_projectiles.extend(sub_sub_projectiles);
                    sub_voxels.extend(sub_sub_voxels);
                    sub_effects.extend(sub_sub_effects);
                    sub_triggers.extend(sub_sub_triggers);
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
                triggers.extend(sub_triggers);
            }
            ReferencedBaseCard {
                card_type: ReferencedBaseCardType::CreateMaterial,
                card_idx,
            } => {
                let material = &self.referenced_material_creators[card_idx];
                new_voxels.push((pos.cast::<u32>().unwrap(), material.clone()));
            }
            ReferencedBaseCard {
                card_type: ReferencedBaseCardType::Effect,
                card_idx,
            } => {
                let effect = &self.referenced_effects[card_idx];
                effects.push(effect.clone());
            }
            ReferencedBaseCard {
                card_type: ReferencedBaseCardType::Trigger,
                card_idx,
            } => {
                let trigger = self.referenced_triggers[card_idx].clone();
                triggers.push((trigger, player_idx));
            }
            ReferencedBaseCard {
                card_type: ReferencedBaseCardType::None,
                ..
            } => {}
        }
        (projectiles, new_voxels, effects, triggers)
    }

    pub fn get_referenced_proj(&self, idx: usize) -> &ReferencedProjectile {
        &self.referenced_projs[idx]
    }
}
