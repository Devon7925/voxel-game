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
}

pub struct ProjStats {
    pub damage: i32,
    pub speed: i32,
    pub size: i32,
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
                for modifier in modifiers {
                    match modifier {
                        ProjectileModifier::Damage(d) => damage += d,
                        ProjectileModifier::Speed(s) => speed += s,
                        ProjectileModifier::Size(s) => size += s,
                    }
                }
                stats.push(ProjStats {
                    damage,
                    speed,
                    size,
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
                for modifier in modifiers {
                    match modifier {
                        ProjectileModifier::Damage(d) => damage += d,
                        ProjectileModifier::Speed(s) => speed += s,
                        ProjectileModifier::Size(s) => size += s,
                    }
                }
                if damage > 0 {
                    0.01 * damage as f32 * 1.5f32.powi(speed) * (1.0 + 1.25f32.powi(size))
                } else {
                    -0.05 * damage as f32
                }
            }
            BaseCard::MultiCast(cards) => {
                let mut value = 0.0;
                for card in cards {
                    value += card.evaluate_value();
                }
                value
            }
        }
    }
}

impl Default for BaseCard {
    fn default() -> Self {
        BaseCard::MultiCast(vec![])
    }
}
