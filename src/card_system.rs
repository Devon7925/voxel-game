use serde::{Deserialize, Serialize};

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
}

pub struct ProjStats {
    pub damage: i32,
    pub speed: i32,
    pub size: i32,
    pub lifetime: i32,
    pub gravity: i32,
}

impl BaseCard {
    pub fn from_string(ron_string: &str) -> Self {
        ron::from_str(ron_string).unwrap()
    }

    pub fn to_string(&self) -> String {
        ron::to_string(self).unwrap()
    }

    pub fn get_proj_stats(&self) -> Vec<ProjStats> {
        match self {
            BaseCard::Projectile(modifiers) => {
                let mut stats = Vec::new();
                let mut damage = 0;
                let mut speed = 0;
                let mut size = 0;
                let mut lifetime = 0;
                let mut gravity = 0;
                for modifier in modifiers {
                    match modifier {
                        ProjectileModifier::Damage(d) => damage += d,
                        ProjectileModifier::Speed(s) => speed += s,
                        ProjectileModifier::Size(s) => size += s,
                        ProjectileModifier::Lifetime(l) => lifetime += l,
                        ProjectileModifier::Gravity(g) => gravity += g,
                    }
                }
                stats.push(ProjStats {
                    damage,
                    speed,
                    size,
                    lifetime,
                    gravity,
                });
                stats
            }
            BaseCard::MultiCast(cards) => {
                let mut stats = Vec::new();
                for card in cards {
                    stats.append(&mut card.get_proj_stats());
                }
                stats
            }
        }
    }

    pub fn evaluate_value(&self) -> f32 {
        match self {
            BaseCard::Projectile(modifiers) => {
                let mut damage = 0;
                let mut speed = 0;
                let mut size = 0;
                let mut lifetime = 0;
                let mut gravity = 0;
                for modifier in modifiers {
                    match modifier {
                        ProjectileModifier::Damage(d) => damage += d,
                        ProjectileModifier::Speed(s) => speed += s,
                        ProjectileModifier::Size(s) => size += s,
                        ProjectileModifier::Lifetime(l) => lifetime += l,
                        ProjectileModifier::Gravity(g) => gravity += g,
                    }
                }
                if damage > 0 {
                    0.002 * damage as f32 * (1.0 + 1.5f32.powi(speed) * 1.5f32.powi(lifetime)) * (1.0 + 1.25f32.powi(size))
                } else {
                    -0.02 * damage as f32
                }
            }
            BaseCard::MultiCast(cards) => cards.iter().map(|card| card.evaluate_value()).sum::<f32>(),
        }
    }
}

impl Default for BaseCard {
    fn default() -> Self {
        BaseCard::MultiCast(vec![])
    }
}
