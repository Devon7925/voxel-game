use crate::{settings_manager::Control, voxel_sim_manager::Projectile, PLAYER_BASE_MAX_HEALTH};
use cgmath::{Point3, Quaternion, Rad, Rotation3};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Deck {
    pub cooldowns: Vec<Cooldown>,
    pub passive: PassiveCard,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PassiveCard {
    pub passive_effects: Vec<StatusEffect>,
}

impl Deck {
    pub fn get_total_impact(&self) -> f32 {
        let passive_value =
            BaseCard::StatusEffects(1, self.passive.passive_effects.clone()).get_cooldown();
        if passive_value >= 0.5 {
            return f32::MAX;
        }
        self.cooldowns
            .iter()
            .map(|cooldown| cooldown.get_impact_multiplier())
            .sum::<f32>()
            / (1.0 - 2.0 * passive_value)
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Cooldown {
    pub modifiers: Vec<CooldownModifier>,
    pub abilities: Vec<Ability>,
    #[serde(skip_serializing, default)]
    pub cooldown_value: Option<(f32, Vec<f32>)>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum CooldownModifier {
    None,
    SignedSimpleCooldownModifier(SignedSimpleCooldownModifier, i32),
    SimpleCooldownModifier(SimpleCooldownModifier, u32),
    Reloading,
}

impl CooldownModifier {
    pub fn get_hover_text(&self) -> String {
        match self {
            CooldownModifier::None => "".to_string(),
            CooldownModifier::SimpleCooldownModifier(SimpleCooldownModifier::AddCharge, s) => format!("Add {} charges", s),
            CooldownModifier::SimpleCooldownModifier(SimpleCooldownModifier::AddCooldown, s) => format!("Increase cooldown by {}s ({} per)", SimpleCooldownModifier::ADD_COOLDOWN_AMOUNT*(*s as f32), SimpleCooldownModifier::ADD_COOLDOWN_AMOUNT),
            CooldownModifier::SignedSimpleCooldownModifier(SignedSimpleCooldownModifier::DecreaseCooldown, _) => format!("Multiply impact by {}, this lowers the cooldown of this abiliy, but increases the cooldown of all other abilities", self.get_effect_value()),
            CooldownModifier::Reloading => "For multicharge cooldowns, only start cooldown once all charges are depleted".to_string(),
        }
    }

    pub fn get_effect_value(&self) -> f32 {
        match self {
            CooldownModifier::None => 0.0,
            CooldownModifier::SimpleCooldownModifier(SimpleCooldownModifier::AddCharge, s) => {
                *s as f32
            }
            CooldownModifier::SimpleCooldownModifier(SimpleCooldownModifier::AddCooldown, s) => {
                *s as f32
            }
            CooldownModifier::SignedSimpleCooldownModifier(
                SignedSimpleCooldownModifier::DecreaseCooldown,
                s,
            ) => 1.25f32.powi(*s),
            CooldownModifier::Reloading => 1.0,
        }
    }

    pub fn get_name(&self) -> String {
        match self {
            CooldownModifier::None => "None",
            CooldownModifier::SimpleCooldownModifier(SimpleCooldownModifier::AddCharge, _) => {
                "Add charge"
            }
            CooldownModifier::SimpleCooldownModifier(SimpleCooldownModifier::AddCooldown, _) => {
                "Add cooldown"
            }
            CooldownModifier::SignedSimpleCooldownModifier(
                SignedSimpleCooldownModifier::DecreaseCooldown,
                _,
            ) => "Decrease cooldown",
            CooldownModifier::Reloading => "Reloading",
        }
        .to_string()
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub enum SignedSimpleCooldownModifier {
    DecreaseCooldown,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub enum SimpleCooldownModifier {
    AddCharge,
    AddCooldown,
}

impl SimpleCooldownModifier {
    const ADD_COOLDOWN_AMOUNT: f32 = 0.5;
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Ability {
    pub card: BaseCard,
    pub keybind: Keybind,
    #[serde(skip_serializing, default)]
    pub cached_cooldown: Option<f32>,
    #[serde(skip_serializing, default)]
    pub is_cache_valid: bool,
    #[serde(skip_serializing, default)]
    pub is_keybind_selected: bool,
}

impl Default for Ability {
    fn default() -> Self {
        Ability {
            card: BaseCard::None,
            keybind: Keybind::Not(Box::new(Keybind::True)),
            cached_cooldown: None,
            is_cache_valid: false,
            is_keybind_selected: false,
        }
    }
}

impl Ability {
    pub fn invalidate_cooldown_cache(&mut self) {
        self.is_cache_valid = false;
    }

    pub fn get_cooldown(&self) -> f32 {
        if let Some(cooldown) = self.cached_cooldown {
            return cooldown;
        }
        self.card.get_cooldown()
    }
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
    pub fn empty() -> Self {
        Cooldown {
            modifiers: vec![],
            abilities: vec![Ability::default()],
            cooldown_value: None,
        }
    }

    pub fn get_unreasonable_reason(&self) -> Option<String> {
        Some(
            self.abilities
                .iter()
                .filter_map(|ability| ability.card.get_unreasonable_reason())
                .join(", "),
        )
        .filter(|reason| !reason.is_empty())
    }

    pub fn generate_cooldown_cache(&mut self) -> bool {
        let mut has_anything_changed = false;
        for ability in self
            .abilities
            .iter_mut()
            .filter(|ability| !ability.is_cache_valid)
        {
            ability.cached_cooldown = Some(ability.card.get_cooldown());
            ability.is_cache_valid = true;
            has_anything_changed = true;
        }
        has_anything_changed
    }

    pub fn get_cooldown_recovery(&self, total_impact: f32) -> (f32, Vec<f32>) {
        let ability_values: Vec<f32> = self
            .abilities
            .iter()
            .map(|ability| ability.get_cooldown())
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
        let mut reloading = false;
        for modifier in self.modifiers.iter() {
            match modifier {
                CooldownModifier::None => {}
                CooldownModifier::SimpleCooldownModifier(SimpleCooldownModifier::AddCharge, s) => {
                    ability_charges += s;
                }
                CooldownModifier::SimpleCooldownModifier(
                    SimpleCooldownModifier::AddCooldown,
                    s,
                ) => {
                    added_cooldown += s;
                }
                CooldownModifier::SignedSimpleCooldownModifier(
                    SignedSimpleCooldownModifier::DecreaseCooldown,
                    _,
                ) => {
                    impact_multiplier *= modifier.get_effect_value();
                }
                CooldownModifier::Reloading => {
                    reloading = true;
                }
            }
        }
        let recovery: Vec<f32> = ability_values
            .iter()
            .map(|val| {
                SimpleCooldownModifier::ADD_COOLDOWN_AMOUNT * added_cooldown as f32
                    + val * (0.5 + 0.5 * (1.0 - (-(ability_charges as f32 / 5.0)).exp()))
                        / (impact_multiplier / total_impact)
            })
            .collect();
        let max_recovery = recovery
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            .clone();
        let mut cooldown = SimpleCooldownModifier::ADD_COOLDOWN_AMOUNT * added_cooldown as f32
            + (sum + 2.0 * max) / 3.0
                * (1.0 + 0.75 * (1.0 - (-(ability_charges as f32 / 5.0)).exp()))
                / (impact_multiplier / total_impact);
        if reloading && ability_charges > 0 {
            cooldown = 0.55 * (1 + ability_charges) as f32 * (cooldown - max_recovery);
        }
        (cooldown, recovery)
    }

    pub fn get_impact_multiplier(&self) -> f32 {
        let mut impact_multiplier = 1.0;
        for modifier in self.modifiers.iter() {
            match modifier {
                CooldownModifier::SignedSimpleCooldownModifier(
                    SignedSimpleCooldownModifier::DecreaseCooldown,
                    _,
                ) => {
                    impact_multiplier *= modifier.get_effect_value();
                }
                _ => {}
            }
        }
        impact_multiplier
    }

    pub fn ability_charges(&self) -> u32 {
        self.modifiers
            .iter()
            .map(|modifier| match modifier {
                CooldownModifier::SimpleCooldownModifier(SimpleCooldownModifier::AddCharge, s) => {
                    *s
                }
                _ => 0,
            })
            .sum::<u32>()
            + 1
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum BaseCard {
    Projectile(Vec<ProjectileModifier>),
    MultiCast(Vec<BaseCard>, Vec<MultiCastModifier>),
    CreateMaterial(VoxelMaterial),
    Effect(Effect),
    StatusEffects(u32, Vec<StatusEffect>),
    Trigger(u32),
    Palette(Vec<DragableCard>),
    None,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum ProjectileModifier {
    None,
    SimpleModify(SimpleProjectileModifierType, i32),
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

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub enum SimpleProjectileModifierType {
    Speed,
    Length,
    Width,
    Height,
    Size,
    Lifetime,
    Gravity,
    Health,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum MultiCastModifier {
    None,
    Spread(u32),
    Duplication(u32),
}
impl MultiCastModifier {
    pub fn get_hover_text(&self) -> String {
        match self {
            MultiCastModifier::None => "".to_string(),
            MultiCastModifier::Spread(s) => format!("Increase spread by {}", s),
            MultiCastModifier::Duplication(s) => {
                format!("Create {} copies of the projectile", 2_u32.pow(*s))
            }
        }
    }

    pub fn get_name(&self) -> String {
        match self {
            MultiCastModifier::None => "None",
            MultiCastModifier::Spread(_) => "Spread",
            MultiCastModifier::Duplication(_) => "Duplication",
        }
        .to_string()
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
    Water,
    Player,
    UnloadedAir,
    Wood,
    Leaf,
    Unbreakable,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub enum DirectionCard {
    None,
    Forward,
    Up,
    Movement,
}

impl std::fmt::Display for DirectionCard {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            DirectionCard::None => write!(f, "None"),
            DirectionCard::Forward => write!(f, "Forward"),
            DirectionCard::Up => write!(f, "Up"),
            DirectionCard::Movement => write!(f, "Movement"),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum Effect {
    Cleanse,
    Teleport,
    Damage(i32),
    Knockback(i32, DirectionCard),
}
impl Effect {
    pub fn get_name(&self) -> String {
        match self {
            Effect::Cleanse => "Cleanse",
            Effect::Teleport => "Teleport",
            Effect::Damage(_) => "Damage",
            Effect::Knockback(_, _) => "Knockback",
        }
        .to_string()
    }

    pub fn get_hover_text(&self) -> String {
        match self {
            Effect::Cleanse => "Clear status effects".to_string(),
            Effect::Teleport => {
                "Teleport the active player to where this was activated".to_string()
            }
            Effect::Damage(damage) => format!("Deal {} damage to anything", damage),
            Effect::Knockback(knockback, direction) => format!(
                "Apply an impulse {} in the {} direction",
                knockback, direction
            ),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum StatusEffect {
    None,
    SimpleStatusEffect(SimpleStatusEffectType, i32),
    Invincibility,
    Trapped,
    Lockout,
    Stun,
    OnHit(Box<BaseCard>),
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub enum SimpleStatusEffectType {
    Speed,
    DamageOverTime,
    IncreaseDamageTaken,
    IncreaseGravity(DirectionCard),
    Overheal,
    Grow,
    IncreaseMaxHealth,
}

impl StatusEffect {
    pub fn get_effect_value(&self) -> f32 {
        match self {
            StatusEffect::None => 0.0,
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::Speed, s) => 1.25f32.powi(*s),
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::DamageOverTime, s) => {
                10.0 * *s as f32
            }
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::IncreaseDamageTaken, s) => {
                1.25f32.powi(*s)
            }
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::IncreaseGravity(_), s) => {
                0.5 * *s as f32
            }
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::Overheal, s) => {
                10.0 * *s as f32
            }
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::Grow, s) => 1.25f32.powi(*s),
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::IncreaseMaxHealth, s) => {
                0.1 * PLAYER_BASE_MAX_HEALTH * *s as f32
            }
            StatusEffect::Invincibility => 0.0,
            StatusEffect::Trapped => 0.0,
            StatusEffect::Lockout => 0.0,
            StatusEffect::Stun => 0.0,
            StatusEffect::OnHit(_card) => 0.0,
        }
    }

    pub fn is_advanced(&self) -> bool {
        match self {
            StatusEffect::None => false,
            StatusEffect::SimpleStatusEffect(_, _) => false,
            StatusEffect::Invincibility => false,
            StatusEffect::Trapped => false,
            StatusEffect::Lockout => false,
            StatusEffect::Stun => false,
            StatusEffect::OnHit(_) => true,
        }
    }

    pub fn get_hover_text(&self) -> String {
        match self {
            StatusEffect::None => "".to_string(),
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::Speed, _) => {
                format!("Speed (25% per) {}%", self.get_effect_value() * 100.0)
            }
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::DamageOverTime, _) => format!(
                "Damage over time (10 dps per) {}dps",
                self.get_effect_value()
            ),
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::IncreaseDamageTaken, _) => {
                format!(
                    "Increase damage taken (25% per) {}%",
                    self.get_effect_value() * 100.0
                )
            }
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::IncreaseGravity(_), _) => {
                format!(
                    "Increase gravity (0.5x normal gravity per) {}b/s/s in the chosen direction",
                    self.get_effect_value()
                )
            }
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::Overheal, _) => {
                format!("Overheal (10 per) {}", self.get_effect_value())
            }
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::Grow, _) => {
                format!("Grow (25% per) {}%", self.get_effect_value() * 100.0)
            }
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::IncreaseMaxHealth, _) => {
                format!("Increase max health (10% per) {}%", self.get_effect_value())
            }
            StatusEffect::Invincibility => "Invincibility".to_string(),
            StatusEffect::Trapped => "Trapped".to_string(),
            StatusEffect::Lockout => "Lockout".to_string(),
            StatusEffect::Stun => "Stun".to_string(),
            StatusEffect::OnHit(card) => format!("On hit {}", card.to_string()),
        }
    }

    fn get_unreasonable_reason(&self) -> Option<String> {
        match self {
            StatusEffect::None => None,
            StatusEffect::SimpleStatusEffect(_, stacks) => {
                if stacks.abs() > 20 {
                    Some(format!("Too many effect stacks ({} > 20)", stacks.abs()))
                } else {
                    None
                }
            }
            StatusEffect::Invincibility => None,
            StatusEffect::Trapped => None,
            StatusEffect::Lockout => None,
            StatusEffect::Stun => None,
            StatusEffect::OnHit(card) => card.get_unreasonable_reason(),
        }
    }

    pub fn get_name(&self) -> String {
        match self {
            StatusEffect::None => "None",
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::Speed, _) => "Speed",
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::DamageOverTime, _) => {
                "Damage over time"
            }
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::IncreaseDamageTaken, _) => {
                "Increase damage taken"
            }
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::IncreaseGravity(_), _) => {
                "Increase gravity"
            }
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::Overheal, _) => "Overheal",
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::Grow, _) => "Grow",
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::IncreaseMaxHealth, _) => {
                "Increase max health"
            }
            StatusEffect::Invincibility => "Invincibility",
            StatusEffect::Trapped => "Trapped",
            StatusEffect::Lockout => "Lockout",
            StatusEffect::Stun => "Stun",
            StatusEffect::OnHit(_) => "On hit",
        }
        .to_string()
    }
}

impl VoxelMaterial {
    pub fn from_memory(memory: u32) -> Self {
        match memory >> 24 {
            0 => VoxelMaterial::Air,
            1 => VoxelMaterial::Stone,
            2 => VoxelMaterial::Unloaded,
            3 => VoxelMaterial::Dirt,
            4 => VoxelMaterial::Grass,
            5 => VoxelMaterial::Projectile,
            6 => VoxelMaterial::Ice,
            7 => VoxelMaterial::Water,
            8 => VoxelMaterial::Player,
            9 => VoxelMaterial::UnloadedAir,
            10 => VoxelMaterial::Wood,
            11 => VoxelMaterial::Leaf,
            12 => VoxelMaterial::Unbreakable,
            _ => panic!("Invalid state"),
        }
    }

    pub fn get_material_idx(&self) -> u32 {
        match self {
            VoxelMaterial::Air => 0,
            VoxelMaterial::Stone => 1,
            VoxelMaterial::Unloaded => 2,
            VoxelMaterial::Dirt => 3,
            VoxelMaterial::Grass => 4,
            VoxelMaterial::Projectile => 5,
            VoxelMaterial::Ice => 6,
            VoxelMaterial::Water => 7,
            VoxelMaterial::Player => 8,
            VoxelMaterial::UnloadedAir => 9,
            VoxelMaterial::Wood => 10,
            VoxelMaterial::Leaf => 11,
            VoxelMaterial::Unbreakable => 12,
        }
    }

    pub fn to_memory(&self) -> u32 {
        self.get_material_idx() << 24
    }

    pub fn is_passthrough(&self) -> bool {
        matches!(
            self,
            VoxelMaterial::Air | VoxelMaterial::Water | VoxelMaterial::UnloadedAir
        )
    }

    pub fn density(&self) -> f32 {
        match self {
            VoxelMaterial::Air => 1.0,
            VoxelMaterial::Stone => 1.0,
            VoxelMaterial::Unloaded => 1.0,
            VoxelMaterial::Dirt => 1.0,
            VoxelMaterial::Grass => 1.0,
            VoxelMaterial::Projectile => panic!("Invalid state"),
            VoxelMaterial::Ice => 1.0,
            VoxelMaterial::Water => 4.4,
            VoxelMaterial::Player => panic!("Invalid state"),
            VoxelMaterial::UnloadedAir => 1.0,
            VoxelMaterial::Wood => 1.0,
            VoxelMaterial::Leaf => 1.0,
            VoxelMaterial::Unbreakable => 1.0,
        }
    }

    pub fn get_friction(&self) -> f32 {
        match self {
            VoxelMaterial::Air => 0.0,
            VoxelMaterial::Stone => 5.0,
            VoxelMaterial::Unloaded => 0.0,
            VoxelMaterial::Dirt => 5.0,
            VoxelMaterial::Grass => 5.0,
            VoxelMaterial::Projectile => 0.0,
            VoxelMaterial::Ice => 0.1,
            VoxelMaterial::Water => 1.0,
            VoxelMaterial::Player => 0.0,
            VoxelMaterial::UnloadedAir => 0.0,
            VoxelMaterial::Wood => 5.0,
            VoxelMaterial::Leaf => 4.0,
            VoxelMaterial::Unbreakable => 5.0,
        }
    }
}

#[derive(Debug, Clone)]
struct CardValue {
    damage: f32,
    generic: f32,
    range_probabilities: [f32; 15],
}

fn convolve_range_probabilities<const COUNT: usize>(
    a_probs: [f32; COUNT],
    b_props: [f32; COUNT],
) -> [f32; COUNT] {
    let mut new_range_probabilities = [0.0; COUNT];
    for a_idx in 0..COUNT {
        for b_idx in 0..COUNT {
            if a_idx + b_idx >= COUNT {
                break;
            }
            new_range_probabilities[a_idx + b_idx] += a_probs[a_idx] * b_props[b_idx];
        }
    }
    new_range_probabilities
}

const DAMAGE_CALCULATION_FLOAT_SCALE: f32 = 5.0;
const SCALED_PLAYER_BASE_MAX_HEALTH: i32 =
    (PLAYER_BASE_MAX_HEALTH * DAMAGE_CALCULATION_FLOAT_SCALE) as i32;
const TIME_TO_FIRST_SHOT: f32 = 0.5;
const HEALING_RATE: f32 = 12.8;
fn gen_cooldown_for_ttk(damage_profile: Vec<(f32, f32)>, goal_ttk: f32) -> f32 {
    let minimum_damage = damage_profile.get(0).unwrap().0;
    if minimum_damage >= PLAYER_BASE_MAX_HEALTH {
        return 120.0;
    }
    let mut healing = 128;
    let mut delta = healing;
    while delta > 1 {
        delta /= 2;
        if (get_avg_ttk(
            &damage_profile,
            healing,
            SCALED_PLAYER_BASE_MAX_HEALTH,
            100,
            &mut HashMap::new(),
        ) - TIME_TO_FIRST_SHOT)
            * healing as f32
            / HEALING_RATE
            / DAMAGE_CALCULATION_FLOAT_SCALE
            > goal_ttk
        {
            healing -= delta;
        } else {
            healing += delta;
        }
    }
    healing as f32 / DAMAGE_CALCULATION_FLOAT_SCALE / HEALING_RATE
}

fn get_avg_ttk(
    damage_profile: &Vec<(f32, f32)>,
    healing: i32,
    current_health: i32,
    iterations: usize,
    table: &mut HashMap<i32, (f32, usize)>,
) -> f32 {
    if current_health <= healing {
        0.0
    } else if iterations == 0 {
        0.0
    } else if current_health > SCALED_PLAYER_BASE_MAX_HEALTH {
        get_avg_ttk(
            damage_profile,
            healing,
            SCALED_PLAYER_BASE_MAX_HEALTH,
            iterations,
            table,
        )
    } else {
        if let Some((cached_result, cached_iterations)) = table.get(&current_health) {
            if cached_iterations >= &iterations {
                return *cached_result;
            }
        }
        let result = 1.0
            + (0..damage_profile.len())
                .map(|idx| {
                    damage_profile[idx].1
                        * get_avg_ttk(
                            damage_profile,
                            healing,
                            current_health
                                - (damage_profile[idx].0 * DAMAGE_CALCULATION_FLOAT_SCALE) as i32
                                + healing,
                            iterations - 1,
                            table,
                        )
                })
                .sum::<f32>();
        table.insert(current_health, (result, iterations));
        result
    }
}

fn error_function(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * x);
    1.0 - (0.254829592 * t - 0.284496736 * t * t + 1.421413741 * t.powi(3)
        - 1.453152027 * t.powi(4)
        + 1.061405429 * t.powi(5))
        * (-x * x).exp()
}

fn normal_pdf(x: f32, mu: f32, sigma: f32) -> f32 {
    (-(x - mu).powi(2) / (2.0 * sigma.powi(2))).exp()
        / (sigma * (2.0 * std::f32::consts::PI).sqrt())
}

const RANGE_PROBABILITIES_SCALE: f32 = 5.0;
impl BaseCard {
    pub const EFFECT_LENGTH_SCALE: f32 = 0.5;
    pub fn from_string(ron_string: &str) -> Self {
        ron::from_str(ron_string).unwrap()
    }

    pub fn to_string(&self) -> String {
        ron::to_string(self).unwrap()
    }

    pub fn get_cooldown(&self) -> f32 {
        let card_values = self.evaluate_value(true);
        let generic_value = card_values
            .iter()
            .map(|card_value| card_value.generic)
            .sum::<f32>();
        if card_values
            .iter()
            .all(|card_value| card_value.damage == 0.0)
        {
            return generic_value;
        }

        let ranged_damage_profiles: Vec<Vec<(f32, f32)>> = (0..15)
            .map(|idx| {
                let mut damage_profile = vec![(0.0, 1.0)];
                for card_value in card_values.iter() {
                    let damage = card_value.damage;
                    if damage == 0.0 {
                        continue;
                    }
                    let range_probability = card_value.range_probabilities[idx].min(1.0);
                    if range_probability == 0.0 {
                        continue;
                    }
                    let mut new_damage_profile = vec![];
                    for (profile_damage, profile_probability) in damage_profile.iter_mut() {
                        if range_probability < 1.0 {
                            new_damage_profile.push((
                                *profile_damage,
                                *profile_probability * (1.0 - range_probability),
                            ));
                        }
                        new_damage_profile.push((
                            *profile_damage + damage,
                            *profile_probability * range_probability,
                        ));
                    }
                    damage_profile = new_damage_profile;
                }
                damage_profile.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                damage_profile
                    .chunk_by(|(a, _), (b, _)| a == b)
                    .map(|group| {
                        let damage = group.first().unwrap().0;
                        let probability = group.iter().map(|(_, p)| p).sum::<f32>();
                        (damage, probability)
                    })
                    .collect()
            })
            .collect();

        let range_cds = ranged_damage_profiles
            .into_iter()
            .map(|damage_profile| gen_cooldown_for_ttk(damage_profile, 3.5))
            .collect_vec();
        let average_cd = range_cds.iter().sum::<f32>() / range_cds.len() as f32;
        let std_cd = (range_cds
            .iter()
            .map(|cd| (cd - average_cd).powi(2))
            .sum::<f32>()
            / range_cds.len() as f32)
            .sqrt();
        let range_cd_weights = range_cds
            .iter()
            .map(|cd| ((cd - average_cd) / std_cd).exp())
            .collect_vec();
        let range_cd_weights_sum = range_cd_weights.iter().sum::<f32>();
        const RANGE_CONTROL: f32 = 1.0 / 20.0;
        let range_cd_weights = range_cd_weights
            .iter()
            .enumerate()
            .map(|(idx, weight)| {
                RANGE_CONTROL * weight / range_cd_weights_sum
                    + (1.0 - RANGE_CONTROL)
                        * RANGE_PROBABILITIES_SCALE
                        * normal_pdf(idx as f32 * RANGE_PROBABILITIES_SCALE, 25.0, 15.0)
            })
            .collect_vec();
        let damage_value = range_cds
            .iter()
            .zip(range_cd_weights.iter())
            .map(|(cd, weight)| cd * weight)
            .sum::<f32>()
            / range_cd_weights.iter().sum::<f32>();
        damage_value + generic_value
    }

    fn evaluate_value(&self, is_direct: bool) -> Vec<CardValue> {
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
                        ProjectileModifier::None => {}
                        ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Speed,
                            s,
                        ) => speed += s,
                        ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Length,
                            s,
                        ) => length += s,
                        ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Width,
                            s,
                        ) => width += s,
                        ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Height,
                            s,
                        ) => height += s,
                        ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Size, s) => {
                            length += s;
                            width += s;
                            height += s;
                        }
                        ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Lifetime,
                            l,
                        ) => lifetime += l,
                        ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Gravity,
                            g,
                        ) => gravity += g,
                        ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Health,
                            g,
                        ) => health += g,
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
                let speed =
                    ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Speed, speed)
                        .get_effect_value();
                let length =
                    ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Length, length)
                        .get_effect_value();
                let width =
                    ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Width, width)
                        .get_effect_value();
                let height =
                    ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Height, height)
                        .get_effect_value();
                let lifetime = ProjectileModifier::SimpleModify(
                    SimpleProjectileModifierType::Lifetime,
                    lifetime,
                )
                .get_effect_value();
                let _gravity = ProjectileModifier::SimpleModify(
                    SimpleProjectileModifierType::Gravity,
                    gravity,
                )
                .get_effect_value();
                let health =
                    ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Health, health)
                        .get_effect_value();
                let range_prob_evaluator = |idx: usize, target_width: f32, target_height: f32| {
                    let distance = idx as f32 * RANGE_PROBABILITIES_SCALE;
                    let time_traveled = distance / speed;
                    let aim_std = 0.3 + 0.2 * time_traveled;
                    let max_size = (target_width + length)
                        .max(target_width + width)
                        .max(target_height + height);
                    if distance > speed.abs() * lifetime + max_size {
                        return 0.0;
                    }
                    let mut result = 0.0;
                    if speed.abs() > 0.0 && idx > 0 {
                        let x_aim_area =
                            (target_width + width + (target_width + length) / speed.abs())
                                / distance;
                        let y_aim_area =
                            (target_height + height + (target_width + length) / speed.abs())
                                / distance;
                        result += error_function(x_aim_area / aim_std)
                            * error_function(y_aim_area / aim_std);
                    }

                    if distance
                        <= (target_width + length)
                            .min(target_width + width)
                            .min(target_height + height)
                    {
                        result += 0.1 * (target_width + length) * (target_width + width)
                            + (target_height + height);
                    } else {
                        if distance <= (target_width + length) {
                            result += 8.0 * (target_width + width)
                                + (target_height + height)
                                    / (4.0 * std::f32::consts::PI * distance * distance);
                        }
                        if distance <= (target_width + width) {
                            result += 8.0 * (target_width + length)
                                + (target_height + height)
                                    / (4.0 * std::f32::consts::PI * distance * distance);
                        }
                        if distance <= (target_height + height) {
                            result += 8.0 * (target_width + length)
                                + (target_width + width)
                                    / (4.0 * std::f32::consts::PI * distance * distance);
                        }
                    }
                    result.min(2.0)
                };

                let range_probabilities = core::array::from_fn(|idx| {
                    let target_width = 1.0;
                    let target_height = 3.0;
                    range_prob_evaluator(idx, target_width, target_height)
                });
                let headshot_range_probabilities = core::array::from_fn(|idx| {
                    let target_width = 0.5;
                    let target_height = 0.5;
                    range_prob_evaluator(idx, target_width, target_height)
                });
                let mut value = vec![];
                if enemy_fire && health > 1.0 {
                    value.push(CardValue {
                        damage: 0.0,
                        generic: 0.0035
                            * lifetime
                            * (width * height + length * height + length * width).sqrt()
                            * (30.0 + health)
                            * if friendly_fire { 1.0 } else { 2.0 },
                        range_probabilities: core::array::from_fn(
                            |idx| if idx == 0 { 1.0 } else { 0.0 },
                        ),
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
                let expiry_range_probabilities: [f32; 15] = core::array::from_fn(|idx| {
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
                let trigger_range_probabilities: [f32; 15] = core::array::from_fn(|idx| {
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
                let mut duplication = 0;
                let mut spread = 0;
                for modifier in modifiers.iter() {
                    match modifier {
                        MultiCastModifier::None => {}
                        MultiCastModifier::Duplication(mod_duplication) => {
                            duplication += *mod_duplication;
                        }
                        MultiCastModifier::Spread(mod_spread) => {
                            spread += *mod_spread;
                        }
                    }
                }
                let duplicate_amount = 2u32.pow(duplication);
                cards
                    .iter()
                    .flat_map(|card| card.evaluate_value(is_direct))
                    .flat_map(|card_value| {
                        let mut values = vec![card_value.clone()];
                        for dup_idx in 1..duplicate_amount {
                            let offset = dup_idx as f32 * spread as f32 / duplicate_amount as f32;
                            values.push(CardValue {
                                damage: card_value.damage,
                                generic: card_value.generic,
                                range_probabilities: if spread > 0 {
                                    let mut i = -1;
                                    card_value.range_probabilities.map(|prob| {
                                        i += 1;
                                        if i == 0 {
                                            return prob;
                                        }
                                        prob / (RANGE_PROBABILITIES_SCALE * i as f32 * offset)
                                            .powi(2)
                                    })
                                } else {
                                    card_value.range_probabilities
                                },
                            });
                        }
                        values
                    })
                    .collect()
            }
            BaseCard::CreateMaterial(material) => {
                let material_value = match material {
                    VoxelMaterial::Air => 0.0,
                    VoxelMaterial::Stone => 0.0875,
                    VoxelMaterial::Unloaded => panic!("Invalid state"),
                    VoxelMaterial::Dirt => 0.0525,
                    VoxelMaterial::Grass => 0.0525,
                    VoxelMaterial::Projectile => panic!("Invalid state"),
                    VoxelMaterial::Ice => 0.175,
                    VoxelMaterial::Water => 0.0525,
                    VoxelMaterial::Player => panic!("Invalid state"),
                    VoxelMaterial::UnloadedAir => panic!("Invalid state"),
                    VoxelMaterial::Wood => 0.08,
                    VoxelMaterial::Leaf => 0.045,
                    VoxelMaterial::Unbreakable => panic!("Invalid state"),
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
                        if is_direct {
                            vec![CardValue {
                                damage: 0.0,
                                generic: -0.35 * *damage as f32,
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
                        }
                    } else {
                        vec![CardValue {
                            damage: 0.0,
                            generic: -0.35 * *damage as f32,
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
                Effect::Knockback(knockback, _) => vec![CardValue {
                    damage: 0.0,
                    generic: 0.1 * (*knockback as f32).abs(),
                    range_probabilities: core::array::from_fn(
                        |idx| if idx == 0 { 1.0 } else { 0.0 },
                    ),
                }],
                Effect::Cleanse => vec![CardValue {
                    damage: 0.0,
                    generic: 2.4,
                    range_probabilities: core::array::from_fn(
                        |idx| if idx == 0 { 1.0 } else { 0.0 },
                    ),
                }],
                Effect::Teleport => vec![CardValue {
                    damage: 0.0,
                    generic: 4.0,
                    range_probabilities: core::array::from_fn(
                        |idx| if idx == 0 { 1.0 } else { 0.0 },
                    ),
                }],
            },
            BaseCard::StatusEffects(duration, effects) => {
                let true_duration = Self::EFFECT_LENGTH_SCALE * *duration as f32;
                let mut result = vec![];
                for effect in effects.iter() {
                    result.extend(match effect {
                        StatusEffect::None => vec![],
                        StatusEffect::SimpleStatusEffect(effect_type, stacks) => {
                            match effect_type {
                                &SimpleStatusEffectType::Speed => vec![CardValue {
                                    damage: 0.0,
                                    generic: if stacks < &0 {
                                        0.1 * (1.0 - effect.get_effect_value())
                                            * true_duration
                                            * (if is_direct { -1.0 } else { 1.0 })
                                    } else {
                                        0.17 * true_duration * effect.get_effect_value()
                                    },
                                    range_probabilities: core::array::from_fn(|idx| {
                                        if idx == 0 {
                                            1.0
                                        } else {
                                            0.0
                                        }
                                    }),
                                }],
                                SimpleStatusEffectType::DamageOverTime => {
                                    if is_direct {
                                        vec![CardValue {
                                            damage: 0.0,
                                            generic: -0.17
                                                * effect.get_effect_value()
                                                * true_duration,
                                            range_probabilities: core::array::from_fn(|idx| {
                                                if idx == 0 {
                                                    1.0
                                                } else {
                                                    0.0
                                                }
                                            }),
                                        }]
                                    } else if stacks > &0 {
                                        vec![CardValue {
                                            damage: true_duration * effect.get_effect_value() * 0.9,
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
                                            generic: 0.06
                                                * effect.get_effect_value().abs()
                                                * true_duration,
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
                                SimpleStatusEffectType::IncreaseDamageTaken => vec![CardValue {
                                    damage: 0.0,
                                    generic: 0.35
                                        * true_duration
                                        * effect
                                            .get_effect_value()
                                            .max(1.0 / effect.get_effect_value()),
                                    range_probabilities: core::array::from_fn(|idx| {
                                        if idx == 0 {
                                            1.0
                                        } else {
                                            0.0
                                        }
                                    }),
                                }],
                                SimpleStatusEffectType::IncreaseGravity(_) => vec![CardValue {
                                    damage: 0.0,
                                    generic: 0.17 * true_duration * effect.get_effect_value().abs(),
                                    range_probabilities: core::array::from_fn(|idx| {
                                        if idx == 0 {
                                            1.0
                                        } else {
                                            0.0
                                        }
                                    }),
                                }],
                                SimpleStatusEffectType::Overheal => vec![CardValue {
                                    damage: 0.0,
                                    generic: 0.35
                                        * (1.0 - (-true_duration).exp())
                                        * effect.get_effect_value(),
                                    range_probabilities: core::array::from_fn(|idx| {
                                        if idx == 0 {
                                            1.0
                                        } else {
                                            0.0
                                        }
                                    }),
                                }],
                                SimpleStatusEffectType::Grow => vec![CardValue {
                                    damage: 0.0,
                                    generic: (if is_direct && stacks > &0 { -0.5 } else { 1.0 })
                                        * 0.1
                                        * true_duration
                                        * effect.get_effect_value(),
                                    range_probabilities: core::array::from_fn(|idx| {
                                        if idx == 0 {
                                            1.0
                                        } else {
                                            0.0
                                        }
                                    }),
                                }],
                                SimpleStatusEffectType::IncreaseMaxHealth => vec![CardValue {
                                    damage: 0.0,
                                    generic: (if is_direct && stacks < &0 { -0.5 } else { 1.0 })
                                        * 0.0035
                                        * true_duration
                                        * effect.get_effect_value().abs(),
                                    range_probabilities: core::array::from_fn(|idx| {
                                        if idx == 0 {
                                            1.0
                                        } else {
                                            0.0
                                        }
                                    }),
                                }],
                            }
                        }
                        StatusEffect::Invincibility => vec![CardValue {
                            damage: 0.0,
                            generic: 3.5 * true_duration,
                            range_probabilities: core::array::from_fn(|idx| {
                                if idx == 0 {
                                    1.0
                                } else {
                                    0.0
                                }
                            }),
                        }],
                        StatusEffect::Trapped => vec![CardValue {
                            damage: 0.0,
                            generic: (if is_direct { -1.0 } else { 1.0 })
                                * 0.35
                                * true_duration.powi(2),
                            range_probabilities: core::array::from_fn(|idx| {
                                if idx == 0 {
                                    1.0
                                } else {
                                    0.0
                                }
                            }),
                        }],
                        StatusEffect::Lockout => vec![CardValue {
                            damage: 0.0,
                            generic: (if is_direct { -1.0 } else { 1.0 })
                                * 0.35
                                * true_duration.powi(2),
                            range_probabilities: core::array::from_fn(|idx| {
                                if idx == 0 {
                                    1.0
                                } else {
                                    0.0
                                }
                            }),
                        }],
                        StatusEffect::Stun => vec![CardValue {
                            damage: 0.0,
                            generic: (if is_direct { -1.0 } else { 1.0 })
                                * 0.7
                                * true_duration.powi(2),
                            range_probabilities: core::array::from_fn(|idx| {
                                if idx == 0 {
                                    1.0
                                } else {
                                    0.0
                                }
                            }),
                        }],
                        StatusEffect::OnHit(card) => {
                            let range_probabilities: [f32; 15] = core::array::from_fn(|idx| {
                                (0.1 * (10.0 - idx as f32 / true_duration)).min(1.0)
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
                    });
                }
                result
            }
            BaseCard::Trigger(_id) => vec![],
            BaseCard::None => vec![],
            BaseCard::Palette(..) => panic!("Invalid state"),
        }
    }

    pub fn get_unreasonable_reason(&self) -> Option<String> {
        match self {
            BaseCard::Projectile(modifiers) => {
                for modifier in modifiers {
                    match modifier {
                        ProjectileModifier::None => {}
                        ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Speed,
                            _,
                        ) => {
                            let speed = modifier.get_effect_value().abs();
                            if speed > 400.0 {
                                return Some(format!(
                                    "Projectile speed too high ({} > 400)",
                                    speed
                                ));
                            }
                        }
                        ProjectileModifier::SimpleModify(_, s) => {
                            if *s > 15 {
                                return Some(format!("Projectile modifier too high ({} > 15)", s));
                            }
                        }
                        ProjectileModifier::OnHit(card) => {
                            if let Some(reason) = card.get_unreasonable_reason() {
                                return Some(reason);
                            }
                        }
                        ProjectileModifier::OnHeadshot(card) => {
                            if let Some(reason) = card.get_unreasonable_reason() {
                                return Some(reason);
                            }
                        }
                        ProjectileModifier::OnExpiry(card) => {
                            if let Some(reason) = card.get_unreasonable_reason() {
                                return Some(reason);
                            }
                        }
                        ProjectileModifier::OnTrigger(_, card) => {
                            if let Some(reason) = card.get_unreasonable_reason() {
                                return Some(reason);
                            }
                        }
                        ProjectileModifier::Trail(_, card) => {
                            if let Some(reason) = card.get_unreasonable_reason() {
                                return Some(reason);
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
                let mut unreasonable_reasons = vec![];
                for card in cards {
                    if let Some(reason) = card.get_unreasonable_reason() {
                        unreasonable_reasons.push(reason);
                    }
                }
                for modifier in modifiers.iter() {
                    match modifier {
                        MultiCastModifier::Duplication(duplication) => {
                            if *duplication > 12 {
                                return Some(format!(
                                    "Multicast duplication too high ({} > 12)",
                                    duplication
                                ));
                            }
                        }
                        _ => {}
                    }
                }
                if unreasonable_reasons.len() > 0 {
                    return Some(unreasonable_reasons.join(", "));
                }
            }
            BaseCard::CreateMaterial(material) => match material {
                VoxelMaterial::Air
                | VoxelMaterial::Stone
                | VoxelMaterial::Dirt
                | VoxelMaterial::Grass
                | VoxelMaterial::Ice
                | VoxelMaterial::Water
                | VoxelMaterial::Wood
                | VoxelMaterial::Leaf => {}
                VoxelMaterial::Projectile
                | VoxelMaterial::Unloaded
                | VoxelMaterial::Player
                | VoxelMaterial::UnloadedAir
                | VoxelMaterial::Unbreakable => {
                    return Some(format!("Invalid Material {:?}", material));
                }
            },
            BaseCard::Effect(effect) => match effect {
                Effect::Damage(damage) => {
                    if damage.abs() >= 1024 {
                        return Some(format!("Damage too high ({} > 1024)", damage.abs()));
                    }
                }
                Effect::Knockback(knockback, _) => {
                    if knockback.abs() > 40 {
                        return Some(format!("Knockback too high ({} > 40)", knockback.abs()));
                    }
                }
                Effect::Cleanse => {}
                Effect::Teleport => {}
            },
            BaseCard::StatusEffects(duration, effects) => {
                let mut unreasonable_reason = vec![];
                if *duration > 15 {
                    unreasonable_reason.push(format!(
                        "Status effect duration too high ({} > 15)",
                        duration
                    ));
                }
                for effect in effects.iter() {
                    if let Some(reason) = effect.get_unreasonable_reason() {
                        unreasonable_reason.push(reason);
                    }
                }
                if unreasonable_reason.len() > 0 {
                    return Some(unreasonable_reason.join(", "));
                }
            }
            BaseCard::Trigger(_) => {}
            BaseCard::None => {}
            BaseCard::Palette(..) => panic!("Invalid state"),
        }
        return None;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DragableCard {
    ProjectileModifier(ProjectileModifier),
    MultiCastModifier(MultiCastModifier),
    CooldownModifier(CooldownModifier),
    StatusEffect(StatusEffect),
    BaseCard(BaseCard),
    Direction(DirectionCard),
}

impl Default for BaseCard {
    fn default() -> Self {
        BaseCard::None
    }
}

impl ProjectileModifier {
    pub fn is_advanced(&self) -> bool {
        match self {
            ProjectileModifier::None => false,
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

    pub fn get_hover_text(&self) -> String {
        match self {
            ProjectileModifier::None => format!(""),
            ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Speed, _) => {
                format!("Speed (+8 per) {}b/s", self.get_effect_value())
            }
            ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Length, _) => {
                format!("Length (+25% per) {}", self.get_effect_value())
            }
            ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Width, _) => {
                format!("Width (+25% per) {}", self.get_effect_value())
            }
            ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Height, _) => {
                format!("Height (+25% per) {}", self.get_effect_value())
            }
            ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Size, _) => {
                format!("Size (+25% per) {}", self.get_effect_value())
            }
            ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Lifetime, _) => {
                format!("Lifetime (+50% per) {}s", self.get_effect_value())
            }
            ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Gravity, _) => {
                format!("Gravity (+2 per) {}b/s/s", self.get_effect_value())
            }
            ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Health, _) => {
                format!("Projectile Health (+50% per) {}", self.get_effect_value())
            }
            ProjectileModifier::FriendlyFire => format!("Allows hitting friendly entities"),
            ProjectileModifier::NoEnemyFire => format!("Prevents hitting enemy entities"),
            ProjectileModifier::OnHit(_) => format!("On hit activate the following card"),
            ProjectileModifier::OnHeadshot(_) => format!("On headshot activate the following card"),
            ProjectileModifier::OnExpiry(_) => format!("On expiry activate the following card"),
            ProjectileModifier::OnTrigger(id, _) => {
                format!("When {} triggers, activate the following card", id)
            }
            ProjectileModifier::Trail(freq, _) => {
                format!("Every {}s, activate the following card", 1.0 / *freq as f32)
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
            ProjectileModifier::None => 0.0,
            ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Speed, s) => {
                8.0 * (*s as f32 + 5.0)
            }
            ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Length, s) => {
                1.25f32.powi(*s)
            }
            ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Width, s) => {
                1.25f32.powi(*s)
            }
            ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Height, s) => {
                1.25f32.powi(*s)
            }
            ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Size, s) => {
                1.25f32.powi(*s)
            }
            ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Lifetime, s) => {
                3.0 * 1.5f32.powi(*s)
            }
            ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Gravity, s) => {
                2.0 * (*s as f32)
            }
            ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Health, s) => {
                1.5f32.powi(*s)
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

    pub fn get_name(&self) -> String {
        match self {
            ProjectileModifier::None => "",
            ProjectileModifier::SimpleModify(ty, _) => match ty {
                SimpleProjectileModifierType::Speed => "Speed",
                SimpleProjectileModifierType::Length => "Length",
                SimpleProjectileModifierType::Width => "Width",
                SimpleProjectileModifierType::Height => "Height",
                SimpleProjectileModifierType::Size => "Size",
                SimpleProjectileModifierType::Lifetime => "Lifetime",
                SimpleProjectileModifierType::Gravity => "Gravity",
                SimpleProjectileModifierType::Health => "Health",
            },
            ProjectileModifier::FriendlyFire => "Friendly Fire",
            ProjectileModifier::NoEnemyFire => "No Enemy Fire",
            ProjectileModifier::OnHit(_) => "On Hit",
            ProjectileModifier::OnHeadshot(_) => "On Headshot",
            ProjectileModifier::OnExpiry(_) => "On Expiry",
            ProjectileModifier::OnTrigger(_, _) => "On Trigger",
            ProjectileModifier::Trail(_, _) => "Trail",
            ProjectileModifier::LockToOwner => "Lock To Owner",
            ProjectileModifier::PiercePlayers => "Pierce Players",
            ProjectileModifier::WallBounce => "Wall Bounce",
        }
        .to_string()
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
    pub max_charges: u32,
    pub add_cooldown: u32,
    pub abilities: Vec<(ReferencedBaseCard, Keybind)>,
    pub is_reloading: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq)]
pub enum ReferencedBaseCardType {
    Projectile,
    MultiCast,
    CreateMaterial,
    Effect,
    StatusEffects,
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
    Knockback(i32, DirectionCard),
    Cleanse,
    Teleport,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ReferencedStatusEffects {
    pub duration: u32,
    pub effects: Vec<ReferencedStatusEffect>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum ReferencedStatusEffect {
    Speed(i32),
    DamageOverTime(i32),
    IncreaseDamageTaken(i32),
    IncreaseGravity(DirectionCard, i32),
    Overheal(i32),
    Grow(i32),
    IncreaseMaxHealth(i32),
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
    pub referenced_status_effects: Vec<ReferencedStatusEffects>,
    pub referenced_triggers: Vec<ReferencedTrigger>,
}

impl Default for CardManager {
    fn default() -> Self {
        CardManager {
            referenced_multicasts: vec![],
            referenced_projs: vec![],
            referenced_material_creators: vec![],
            referenced_effects: vec![],
            referenced_status_effects: vec![],
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
        let mut max_charges = 1;
        let mut add_cooldown = 0;
        let mut is_reloading = false;
        for modifier in cooldown.modifiers {
            match modifier {
                CooldownModifier::None => {}
                CooldownModifier::SimpleCooldownModifier(SimpleCooldownModifier::AddCharge, c) => {
                    max_charges += c
                }
                CooldownModifier::SimpleCooldownModifier(
                    SimpleCooldownModifier::AddCooldown,
                    c,
                ) => add_cooldown += c,
                CooldownModifier::SignedSimpleCooldownModifier(
                    SignedSimpleCooldownModifier::DecreaseCooldown,
                    _c,
                ) => {}
                CooldownModifier::Reloading => is_reloading = true,
            }
        }
        ReferencedCooldown {
            max_charges,
            add_cooldown,
            abilities,
            is_reloading,
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
                        ProjectileModifier::None => {}
                        ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Speed,
                            s,
                        ) => speed += s,
                        ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Length,
                            s,
                        ) => length += s,
                        ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Width,
                            s,
                        ) => width += s,
                        ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Height,
                            s,
                        ) => height += s,
                        ProjectileModifier::SimpleModify(SimpleProjectileModifierType::Size, s) => {
                            length += s;
                            width += s;
                            height += s;
                        }
                        ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Lifetime,
                            l,
                        ) => lifetime += l,
                        ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Gravity,
                            g,
                        ) => gravity += g,
                        ProjectileModifier::SimpleModify(
                            SimpleProjectileModifierType::Health,
                            g,
                        ) => health += g,
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
                    speed: ProjectileModifier::SimpleModify(
                        SimpleProjectileModifierType::Speed,
                        speed,
                    )
                    .get_effect_value(),
                    length: ProjectileModifier::SimpleModify(
                        SimpleProjectileModifierType::Length,
                        length,
                    )
                    .get_effect_value(),
                    width: ProjectileModifier::SimpleModify(
                        SimpleProjectileModifierType::Width,
                        width,
                    )
                    .get_effect_value(),
                    height: ProjectileModifier::SimpleModify(
                        SimpleProjectileModifierType::Height,
                        height,
                    )
                    .get_effect_value(),
                    lifetime: ProjectileModifier::SimpleModify(
                        SimpleProjectileModifierType::Lifetime,
                        lifetime,
                    )
                    .get_effect_value(),
                    gravity: ProjectileModifier::SimpleModify(
                        SimpleProjectileModifierType::Gravity,
                        gravity,
                    )
                    .get_effect_value(),
                    health: ProjectileModifier::SimpleModify(
                        SimpleProjectileModifierType::Health,
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
                        MultiCastModifier::None => {}
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
                    Effect::Knockback(knockback, direction) => {
                        ReferencedEffect::Knockback(knockback, direction)
                    }
                    Effect::Cleanse => ReferencedEffect::Cleanse,
                    Effect::Teleport => ReferencedEffect::Teleport,
                };
                self.referenced_effects.push(referenced_effect);
                ReferencedBaseCard {
                    card_type: ReferencedBaseCardType::Effect,
                    card_idx: self.referenced_effects.len() - 1,
                }
            }
            BaseCard::StatusEffects(duration, effects) => {
                let mut referenced_status_effects = ReferencedStatusEffects {
                    duration,
                    effects: vec![],
                };
                for effect in effects {
                    referenced_status_effects
                        .effects
                        .extend(self.register_status_effect(effect));
                }
                self.referenced_status_effects
                    .push(referenced_status_effects);
                ReferencedBaseCard {
                    card_type: ReferencedBaseCardType::StatusEffects,
                    card_idx: self.referenced_status_effects.len() - 1,
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

    pub fn register_status_effect(&mut self, effect: StatusEffect) -> Vec<ReferencedStatusEffect> {
        match effect {
            StatusEffect::None => panic!("Invalid state"),
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::Speed, stacks) => {
                vec![ReferencedStatusEffect::Speed(stacks)]
            }
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::DamageOverTime, stacks) => {
                vec![ReferencedStatusEffect::DamageOverTime(stacks)]
            }
            StatusEffect::SimpleStatusEffect(
                SimpleStatusEffectType::IncreaseDamageTaken,
                stacks,
            ) => vec![ReferencedStatusEffect::IncreaseDamageTaken(stacks)],
            StatusEffect::SimpleStatusEffect(
                SimpleStatusEffectType::IncreaseGravity(direction),
                stacks,
            ) => vec![ReferencedStatusEffect::IncreaseGravity(direction, stacks)],
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::Overheal, stacks) => {
                vec![ReferencedStatusEffect::Overheal(stacks)]
            }
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::Grow, stacks) => {
                vec![ReferencedStatusEffect::Grow(stacks)]
            }
            StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::IncreaseMaxHealth, stacks) => {
                vec![ReferencedStatusEffect::IncreaseMaxHealth(stacks)]
            }
            StatusEffect::Invincibility => vec![ReferencedStatusEffect::Invincibility],
            StatusEffect::Trapped => vec![ReferencedStatusEffect::Trapped],
            StatusEffect::Lockout => vec![ReferencedStatusEffect::Lockout],
            StatusEffect::Stun => vec![
                ReferencedStatusEffect::Trapped,
                ReferencedStatusEffect::Lockout,
            ],
            StatusEffect::OnHit(card) => {
                vec![ReferencedStatusEffect::OnHit(
                    self.register_base_card(*card),
                )]
            }
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
        Vec<ReferencedStatusEffects>,
        Vec<(ReferencedTrigger, u32)>,
    ) {
        let mut projectiles = vec![];
        let mut new_voxels = vec![];
        let mut effects = vec![];
        let mut status_effects = vec![];
        let mut triggers = vec![];
        match card {
            ReferencedBaseCard {
                card_type: ReferencedBaseCardType::Projectile,
                card_idx,
            } => {
                let proj_stats = self.get_referenced_proj(card_idx);
                let proj_damage = proj_stats.damage as f32;
                let should_collide_with_terrain = !proj_stats.lock_owner || !proj_stats.on_hit.is_empty() || !proj_stats.on_headshot.is_empty();
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
                    should_collide_with_terrain: if should_collide_with_terrain { 1 } else { 0 },
                    _filler0: 0,
                    _filler1: 0,
                    _filler2: 0,
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
                let mut sub_status_effects = vec![];
                let mut sub_triggers = vec![];
                for sub_card in multicast.sub_cards.iter() {
                    let (
                        sub_sub_projectiles,
                        sub_sub_voxels,
                        sub_sub_effects,
                        sub_sub_status_effects,
                        sub_sub_triggers,
                    ) = self.get_effects_from_base_card(
                        *sub_card,
                        pos,
                        rot,
                        player_idx,
                        is_from_head,
                    );
                    individual_sub_projectiles.extend(sub_sub_projectiles);
                    sub_voxels.extend(sub_sub_voxels);
                    sub_effects.extend(sub_sub_effects);
                    sub_status_effects.extend(sub_sub_status_effects);
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
                status_effects.extend(sub_status_effects);
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
                card_type: ReferencedBaseCardType::StatusEffects,
                card_idx,
            } => {
                let effect = &self.referenced_status_effects[card_idx];
                status_effects.push(effect.clone());
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
        (projectiles, new_voxels, effects, status_effects, triggers)
    }

    pub fn get_referenced_proj(&self, idx: usize) -> &ReferencedProjectile {
        &self.referenced_projs[idx]
    }
}
