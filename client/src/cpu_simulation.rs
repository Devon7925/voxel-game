use std::f32::consts::PI;

use cgmath::{
    vec3, ElementWise, EuclideanSpace, InnerSpace, One, Point3, Quaternion, Rad, Rotation,
    Rotation3, Vector3,
};
use itertools::Itertools;

use crate::{
    card_system::{
        BaseCard, CardManager, DirectionCard, ReferencedCooldown, ReferencedEffect,
        ReferencedStatusEffect, ReferencedStatusEffects, ReferencedTrigger, SimpleStatusEffectType,
        StatusEffect,
    },
    game_modes::GameMode,
    rollback_manager::Action,
    voxel_sim_manager::{Collision, Projectile, VoxelComputePipeline},
    PLAYER_BASE_MAX_HEALTH, PLAYER_HITBOX_OFFSET, PLAYER_HITBOX_SIZE,
    RESPAWN_TIME,
};

#[derive(Clone, Debug)]
pub struct WorldState {
    pub players: Vec<Entity>,
    pub projectiles: Vec<Projectile>,
}

#[derive(Clone, Debug)]
pub struct Entity {
    pub pos: Point3<f32>,
    pub facing: [f32; 2],
    pub rot: Quaternion<f32>,
    pub size: f32,
    pub vel: Vector3<f32>,
    pub dir: Vector3<f32>,
    pub up: Vector3<f32>,
    pub right: Vector3<f32>,
    pub health: Vec<HealthSection>,
    pub abilities: Vec<PlayerAbility>,
    pub passive_abilities: Vec<ReferencedStatusEffect>,
    pub collision_vec: Vector3<i32>,
    pub movement_direction: Vector3<f32>,
    pub status_effects: Vec<AppliedStatusEffect>,
    pub hitmarker: (f32, f32),
    pub hurtmarkers: Vec<(Vector3<f32>, f32, f32)>,
    pub gamemode_data: Vec<u32>,
    pub respawn_timer: f32,
    pub player_piercing_invincibility: f32,
    pub on_hit_passive_cooldown: f32,
}

#[derive(Clone, Debug)]
pub enum HealthSection {
    Health(f32, f32),
    Overhealth(f32, f32),
}

#[derive(Clone, Debug)]
pub struct AppliedStatusEffect {
    pub effect: ReferencedStatusEffect,
    pub time_left: f32,
}

#[derive(Clone, Debug)]
pub struct PlayerAbility {
    pub ability: ReferencedCooldown,
    pub value: (f32, Vec<f32>),
    pub cooldown: f32,
    pub recovery: f32,
    pub remaining_charges: u32,
}

pub struct PlayerEffectStats {
    pub speed: f32,
    pub damage_taken: f32,
    pub gravity: Vector3<f32>,
    pub size: f32,
    pub max_health: f32,
    pub invincible: bool,
    pub lockout: bool,
}

impl Default for Entity {
    fn default() -> Self {
        Entity {
            pos: Point3::new(0.0, 0.0, 0.0),
            facing: [0.0, 0.0],
            dir: Vector3::new(0.0, 0.0, 1.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            right: Vector3::new(1.0, 0.0, 0.0),
            rot: Quaternion::one(),
            size: 1.0,
            vel: Vector3::new(0.0, 0.0, 0.0),
            health: vec![HealthSection::Health(
                PLAYER_BASE_MAX_HEALTH,
                PLAYER_BASE_MAX_HEALTH,
            )],
            abilities: Vec::new(),
            passive_abilities: Vec::new(),
            collision_vec: Vector3::new(0, 0, 0),
            movement_direction: Vector3::new(0.0, 0.0, 0.0),
            status_effects: Vec::new(),
            hitmarker: (0.0, 0.0),
            hurtmarkers: Vec::new(),
            gamemode_data: Vec::new(),
            respawn_timer: 0.0,
            player_piercing_invincibility: 0.0,
            on_hit_passive_cooldown: 0.0,
        }
    }
}
struct NewEffects {
    new_projectiles: Vec<Projectile>,
    voxels_to_write: Vec<(Point3<u32>, u32)>,
    new_effects: Vec<(
        usize,
        usize,
        bool,
        Point3<f32>,
        Vector3<f32>,
        ReferencedEffect,
    )>,
    new_status_effects: Vec<(usize, ReferencedStatusEffects)>,
    step_triggers: Vec<(ReferencedTrigger, u32)>,
}

impl Default for NewEffects {
    fn default() -> Self {
        NewEffects {
            new_projectiles: Vec::new(),
            voxels_to_write: Vec::new(),
            new_effects: Vec::new(),
            new_status_effects: Vec::new(),
            step_triggers: Vec::new(),
        }
    }
}

impl WorldState {
    pub fn new() -> Self {
        WorldState {
            players: Vec::new(),
            projectiles: Vec::new(),
        }
    }

    pub fn step_sim(
        &mut self,
        player_actions: Vec<Action>,
        is_real_update: bool,
        card_manager: &CardManager,
        time_step: f32,
        vox_compute: &mut VoxelComputePipeline,
        game_mode: &Box<dyn GameMode>,
        collisions: Option<Vec<Collision>>,
    ) {
        let mut new_effects = NewEffects::default();

        let player_stats: Vec<PlayerEffectStats> = self.get_player_effect_stats();

        for (player, player_stats) in self.players.iter_mut().zip(player_stats.iter()) {
            let mut health_adjustment = 0.0;
            for status_effect in player
                .status_effects
                .iter()
                .map(|e| &e.effect)
                .chain(player.passive_abilities.iter())
            {
                match status_effect {
                    ReferencedStatusEffect::DamageOverTime(stacks) => {
                        health_adjustment +=
                            -10.0 * player_stats.damage_taken * *stacks as f32 * time_step;
                    }
                    ReferencedStatusEffect::Overheal(stacks) => {
                        player.health.push(HealthSection::Overhealth(
                            10.0 * *stacks as f32 * time_step / BaseCard::EFFECT_LENGTH_SCALE,
                            BaseCard::EFFECT_LENGTH_SCALE,
                        ));
                    }
                    _ => {}
                }
            }
            for status_effect in player.status_effects.iter_mut() {
                status_effect.time_left -= time_step;
            }
            if health_adjustment != 0.0 {
                player.adjust_health(health_adjustment);
            }
            player.status_effects.retain(|x| x.time_left > 0.0);
        }

        for (player_idx, (player, entity_action)) in self
            .players
            .iter_mut()
            .zip(player_actions.into_iter())
            .enumerate()
        {
            player.simple_step(
                time_step,
                entity_action,
                &player_stats,
                player_idx,
                card_manager,
                &mut new_effects,
                game_mode,
            );
        }

        self.projectiles.iter_mut().for_each(|proj| {
            proj.simple_step(&self.players, card_manager, time_step, &mut new_effects)
        });

        let collision_pairs = self.get_collision_pairs(card_manager, time_step, game_mode);

        for (i, j) in collision_pairs.iter() {
            let damage_1 = self.projectiles.get(*i).unwrap().damage;
            let damage_2 = self.projectiles.get(*j).unwrap().damage;
            {
                let proj1_mut = self.projectiles.get_mut(*j).unwrap();
                proj1_mut.health -= damage_2;
                proj1_mut.health -= damage_1;
            }
            {
                let proj2_mut = self.projectiles.get_mut(*i).unwrap();
                proj2_mut.health -= damage_1;
                proj2_mut.health -= damage_2;
            }
            // trigger on hit effects
            {
                let proj = self.projectiles.get(*i).unwrap();
                let hit_cards = card_manager
                    .get_referenced_proj(proj.proj_card_idx as usize)
                    .on_hit
                    .iter()
                    .map(|x| (x.clone(), false))
                    .collect::<Vec<_>>();
                for (card_ref, _was_headshot) in hit_cards {
                    let proj_rot = proj.dir;
                    let proj_rot =
                        Quaternion::new(proj_rot[3], proj_rot[0], proj_rot[1], proj_rot[2]);
                    let (on_hit_projectiles, on_hit_voxels, _effects, _status_effects, triggers) =
                        card_manager.get_effects_from_base_card(
                            card_ref,
                            &Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]),
                            &proj_rot,
                            proj.owner,
                            false,
                        );
                    new_effects.new_projectiles.extend(on_hit_projectiles);
                    for (pos, material) in on_hit_voxels {
                        new_effects
                            .voxels_to_write
                            .push((pos, material.to_memory()));
                    }
                    new_effects.step_triggers.extend(triggers);
                }
            }
            {
                let proj = self.projectiles.get(*j).unwrap();
                let hit_cards = card_manager
                    .get_referenced_proj(proj.proj_card_idx as usize)
                    .on_hit
                    .iter()
                    .map(|x| (x.clone(), false))
                    .collect::<Vec<_>>();
                for (card_ref, _was_headshot) in hit_cards {
                    let proj_rot = proj.dir;
                    let proj_rot =
                        Quaternion::new(proj_rot[3], proj_rot[0], proj_rot[1], proj_rot[2]);
                    let (on_hit_projectiles, on_hit_voxels, _effects, _status_effects, triggers) =
                        card_manager.get_effects_from_base_card(
                            card_ref,
                            &Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]),
                            &proj_rot,
                            proj.owner,
                            false,
                        );
                    new_effects.new_projectiles.extend(on_hit_projectiles);
                    for (pos, material) in on_hit_voxels {
                        new_effects
                            .voxels_to_write
                            .push((pos, material.to_memory()));
                    }
                    new_effects.step_triggers.extend(triggers);
                }
            }
        }

        let proj_collisions = if is_real_update {
            if let Some(collisions) = collisions {
                collisions.iter().map(|collision| {
                    let proj = self.projectiles[collision.id1 as usize];
                    let projectile_rot = Quaternion::new(proj.dir[3], proj.dir[0], proj.dir[1], proj.dir[2]);
                    let projectile_dir = projectile_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0));
                    let projectile_pos = Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]) + (collision.when - proj.vel * time_step) * projectile_dir;
                    (collision.id2 as usize, collision.id1 as usize, projectile_pos, self.players[collision.id2 as usize].pos, collision.properties > 1)
                }).collect_vec()
            } else {
                panic!("No provided collisions for real update");
            }
        } else {
            self.get_player_proj_collisions(&player_stats, card_manager, game_mode, time_step)
        };

        proj_collisions
            .iter()
            .filter(|(_player_idx, proj_idx, damage_source_location, damage_end_location, _headshot)| {
                let vec_start = damage_source_location.to_vec();
                let vec_end = damage_end_location.to_vec();
                self.projectiles
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| {
                        collision_pairs.contains(&&(*proj_idx.min(idx), *proj_idx.max(idx)))
                    })
                    .all(|(_, proj2)| {
                        let mut adj_vec_start = vec_start;
                        let mut adj_vec_end = vec_end;
                        adj_vec_start -= Vector3::new(proj2.pos[0], proj2.pos[1], proj2.pos[2]);
                        adj_vec_end -= Vector3::new(proj2.pos[0], proj2.pos[1], proj2.pos[2]);
                        let proj2_rot =
                            Quaternion::new(proj2.dir[3], proj2.dir[0], proj2.dir[1], proj2.dir[2]);
                        let proj2_rot_inv = proj2_rot.invert();
                        adj_vec_start = proj2_rot_inv.rotate_vector(adj_vec_start);
                        adj_vec_end = proj2_rot_inv.rotate_vector(adj_vec_end);
                        adj_vec_start.div_assign_element_wise(Vector3::new(
                            proj2.size[0],
                            proj2.size[1],
                            proj2.size[2],
                        ));
                        adj_vec_end.div_assign_element_wise(Vector3::new(
                            proj2.size[0],
                            proj2.size[1],
                            proj2.size[2],
                        ));

                        let vec_dir = adj_vec_end - adj_vec_start;
                        let (t_min, t_max) = (0..3)
                            .map(|i| {
                                if vec_dir[i] == 0.0 {
                                    if adj_vec_start[i].abs() <= 1.0 {
                                        (
                                            f32::NEG_INFINITY,
                                            f32::INFINITY
                                        )
                                    } else {
                                        (
                                            f32::NEG_INFINITY,
                                            f32::NEG_INFINITY
                                        )
                                    }
                                } else {
                                    (
                                        (-1.0 - adj_vec_start[i]) / vec_dir[i],
                                        (1.0 - adj_vec_start[i]) / vec_dir[i],
                                    )
                                }
                            })
                            .map(|t| (t.0.min(t.1), t.0.max(t.1)))
                            .reduce(|(t_min1, t_max1), (t_min2, t_max2)| {
                                (t_min1.max(t_min2), t_max1.min(t_max2))
                            })
                            .unwrap();
                        !(t_min < 1.0 && t_max > 0.0 && t_min < t_max)
                    })
            })
            .collect_vec()
            .iter()
            .for_each(|(player_idx, proj_idx, start_pos, end_pos, headshot)| {
                let player = self.players.get_mut(*player_idx).unwrap();
                let proj = self.projectiles.get_mut(*proj_idx).unwrap();
                let proj_card = card_manager.get_referenced_proj(proj.proj_card_idx as usize);

                let projectile_rot =
                    Quaternion::new(proj.dir[3], proj.dir[0], proj.dir[1], proj.dir[2]);
                let projectile_vectors = [
                    projectile_rot.rotate_vector(Vector3::new(1.0, 0.0, 0.0)),
                    projectile_rot.rotate_vector(Vector3::new(0.0, 1.0, 0.0)),
                    projectile_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0)),
                ];
                let projectile_pos = Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]);

                if !proj_card.pierce_players {
                    proj.health = 0.0;
                } else {
                    player.player_piercing_invincibility = 0.3;
                }
                let mut hit_cards = card_manager
                    .get_referenced_proj(proj.proj_card_idx as usize)
                    .on_hit
                    .iter()
                    .map(|x| (x.clone(), false))
                    .collect::<Vec<_>>();
                if *headshot {
                    hit_cards.extend(
                        card_manager
                            .get_referenced_proj(proj.proj_card_idx as usize)
                            .on_headshot
                            .iter()
                            .map(|x| (x.clone(), true)),
                    );
                }
                for (card_ref, was_headshot) in hit_cards {
                    let proj_rot = proj.dir;
                    let proj_rot =
                        Quaternion::new(proj_rot[3], proj_rot[0], proj_rot[1], proj_rot[2]);
                    let (on_hit_projectiles, on_hit_voxels, effects, status_effects, triggers) =
                        card_manager.get_effects_from_base_card(
                            card_ref,
                            &start_pos,
                            &proj_rot,
                            proj.owner,
                            false,
                        );
                    new_effects.new_projectiles.extend(on_hit_projectiles);
                    for (pos, material) in on_hit_voxels {
                        new_effects
                            .voxels_to_write
                            .push((pos, material.to_memory()));
                    }
                    for effect in effects {
                        new_effects.new_effects.push((
                            *player_idx,
                            proj.owner as usize,
                            was_headshot,
                            *start_pos,
                            ((end_pos.to_vec() - projectile_pos.to_vec()).normalize()
                                + projectile_vectors[2] * proj.vel)
                                .normalize(),
                            effect,
                        ));
                    }
                    for status_effects in status_effects {
                        new_effects
                            .new_status_effects
                            .push((*player_idx, status_effects));
                    }
                    new_effects.step_triggers.extend(triggers);
                }
            });

        let player_player_collision_pairs =
            self.get_player_player_collisions(&player_stats, game_mode);
        for (i, j) in player_player_collision_pairs {
            let player1_pos = self.players.get(i).unwrap().pos;
            let player2_pos = self.players.get(j).unwrap().pos;
            let hit_effects = {
                let player1 = self.players.get_mut(i).unwrap();
                let hit_effects = player1
                    .status_effects
                    .iter()
                    .filter_map(|effect| match effect {
                        AppliedStatusEffect {
                            effect: ReferencedStatusEffect::OnHit(hit_card),
                            time_left: _,
                        } => Some(hit_card),
                        _ => None,
                    })
                    .chain(
                        (player1.on_hit_passive_cooldown <= 0.0)
                            .then_some(player1.passive_abilities.iter().filter_map(|effect| {
                                match effect {
                                    ReferencedStatusEffect::OnHit(hit_card) => Some(hit_card),
                                    _ => None,
                                }
                            }))
                            .into_iter()
                            .flatten(),
                    )
                    .map(|hit_effect| {
                        card_manager.get_effects_from_base_card(
                            *hit_effect,
                            &player1.pos,
                            &player1.rot,
                            i as u32,
                            false,
                        )
                    })
                    .collect::<Vec<_>>();

                player1.status_effects.retain(|effect| match effect {
                    AppliedStatusEffect {
                        effect: ReferencedStatusEffect::OnHit(_),
                        time_left: _,
                    } => false,
                    _ => true,
                });
                if player1.on_hit_passive_cooldown <= 0.0 {
                    player1.on_hit_passive_cooldown = 0.5;
                }
                hit_effects
            };
            for (on_hit_projectiles, on_hit_voxels, effects, status_effects, triggers) in
                hit_effects
            {
                new_effects.new_projectiles.extend(on_hit_projectiles);
                for (pos, material) in on_hit_voxels {
                    new_effects
                        .voxels_to_write
                        .push((pos, material.to_memory()));
                }
                for effect in effects {
                    new_effects.new_effects.push((
                        j,
                        i,
                        false,
                        player1_pos,
                        player2_pos - player1_pos,
                        effect,
                    ));
                }
                for status_effects in status_effects {
                    new_effects.new_status_effects.push((j, status_effects));
                }
                new_effects.step_triggers.extend(triggers);
            }
        }

        for player in self.players.iter_mut() {
            if player.get_health_stats().0 <= 0.0 && player.respawn_timer <= 0.0 {
                player.respawn_timer = RESPAWN_TIME;
            }
        }

        for (ReferencedTrigger(trigger_id), trigger_player) in new_effects.step_triggers {
            for proj in self.projectiles.iter_mut() {
                if proj.owner != trigger_player {
                    continue;
                }
                let proj_card = card_manager
                    .get_referenced_proj(proj.proj_card_idx as usize)
                    .clone();
                let projectile_rot =
                    Quaternion::new(proj.dir[3], proj.dir[0], proj.dir[1], proj.dir[2]);
                let projectile_dir = projectile_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0));
                for (proj_trigger_id, on_trigger) in proj_card.on_trigger {
                    if proj_trigger_id == trigger_id {
                        proj.health = 0.0;
                        let (proj_effects, vox_effects, effects, status_effects, _) = card_manager
                            .get_effects_from_base_card(
                                on_trigger,
                                &Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]),
                                &projectile_rot,
                                proj.owner,
                                false,
                            );
                        new_effects.new_projectiles.extend(proj_effects);
                        for (pos, material) in vox_effects {
                            new_effects
                                .voxels_to_write
                                .push((pos, material.to_memory()));
                        }
                        for effect in effects {
                            new_effects.new_effects.push((
                                proj.owner as usize,
                                trigger_player as usize,
                                false,
                                Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]),
                                projectile_dir,
                                effect,
                            ));
                        }
                        for status_effects in status_effects {
                            new_effects
                                .new_status_effects
                                .push((proj.owner as usize, status_effects));
                        }
                    }
                }
            }
        }

        // remove dead projectiles and add new ones
        self.projectiles.retain(|proj| proj.health > 0.0);
        self.projectiles.extend(new_effects.new_projectiles);

        // update voxels
        if is_real_update && new_effects.voxels_to_write.len() > 0 {
            for (pos, material) in new_effects.voxels_to_write {
                vox_compute.queue_voxel_write([pos[0], pos[1], pos[2], material]);
            }
        }

        for (affected_idx, actor_idx, was_headshot, effect_pos, effect_direction, effect) in
            new_effects.new_effects
        {
            let player = self.players.get_mut(affected_idx).unwrap();
            if game_mode.has_immunity(player, &player_stats[affected_idx]) {
                continue;
            }
            match effect {
                ReferencedEffect::Damage(damage) => {
                    player.adjust_health(-player_stats[affected_idx].damage_taken * damage as f32);
                    if damage > 0 {
                        player.hurtmarkers.push((
                            effect_direction,
                            player_stats[affected_idx].damage_taken * damage as f32,
                            1.0,
                        ));
                    }
                    let actor = self.players.get_mut(actor_idx).unwrap();
                    if is_real_update {
                        if was_headshot {
                            actor.hitmarker.1 += damage as f32;
                        } else {
                            actor.hitmarker.0 += damage as f32;
                        }
                    }
                }
                ReferencedEffect::Knockback(knockback, direction) => {
                    let knockback = 10.0 * knockback as f32;
                    let knockback_dir = match direction {
                        DirectionCard::None => Vector3::new(0.0, 0.0, 0.0),
                        DirectionCard::Forward => effect_direction,
                        DirectionCard::Up => Vector3::new(0.0, 1.0, 0.0),
                        DirectionCard::Movement => player.movement_direction,
                    };
                    if knockback_dir.magnitude() > 0.0 {
                        player.vel += knockback * (knockback_dir).normalize();
                    } else {
                        player.vel.y += knockback;
                    }
                }
                ReferencedEffect::Cleanse => {
                    player.status_effects.clear();
                }
                ReferencedEffect::Teleport => {
                    player.pos = effect_pos;
                    player.pos.y +=
                        player.size * (PLAYER_HITBOX_SIZE[1] / 2.0 - PLAYER_HITBOX_OFFSET[1]);
                    player.vel = Vector3::new(0.0, 0.0, 0.0);
                }
            }
        }
        for (player_idx, status_effects) in new_effects.new_status_effects {
            let player: &mut Entity = self.players.get_mut(player_idx).unwrap();
            for status_effect in status_effects.effects {
                match status_effect {
                    ReferencedStatusEffect::Overheal(stacks) => {
                        player.health.push(HealthSection::Overhealth(
                            10.0 * stacks as f32,
                            BaseCard::EFFECT_LENGTH_SCALE * status_effects.duration as f32,
                        ));
                    }
                    _ => {}
                }
                player.status_effects.push(AppliedStatusEffect {
                    effect: status_effect,
                    time_left: BaseCard::EFFECT_LENGTH_SCALE * status_effects.duration as f32,
                })
            }
        }
    }

    fn get_player_player_collisions(
        &mut self,
        player_stats: &Vec<PlayerEffectStats>,
        game_mode: &Box<dyn GameMode>,
    ) -> Vec<(usize, usize)> {
        let mut player_player_collision_pairs: Vec<(usize, usize)> = vec![];
        for i in 0..self.players.len() {
            let player1 = self.players.get(i).unwrap();
            if game_mode.has_immunity(player1, &player_stats[i]) {
                continue;
            }
            for j in 0..self.players.len() {
                if game_mode.are_friends(i as u32, j as u32, &self.players) {
                    continue;
                }
                let player2 = self.players.get(j).unwrap();
                if game_mode.has_immunity(player2, &player_stats[j]) {
                    continue;
                }
                if 5.0 * (player1.size + player2.size) > (player1.pos - player2.pos).magnitude() {
                    for si in 0..Entity::HITSPHERES.len() {
                        for sj in 0..Entity::HITSPHERES.len() {
                            let pos1 = player1.pos + player1.size * Entity::HITSPHERES[si].offset;
                            let pos2 = player2.pos + player2.size * Entity::HITSPHERES[sj].offset;
                            if (pos1 - pos2).magnitude()
                                < (Entity::HITSPHERES[si].radius + Entity::HITSPHERES[sj].radius)
                                    * (player1.size + player2.size)
                            {
                                player_player_collision_pairs.push((i, j));
                            }
                        }
                    }
                }
            }
        }
        player_player_collision_pairs
    }

    fn get_player_proj_collisions(
        &self,
        player_stats: &Vec<PlayerEffectStats>,
        card_manager: &CardManager,
        game_mode: &Box<dyn GameMode>,
        time_step: f32,
    ) -> Vec<(usize, usize, Point3<f32>, Point3<f32>, bool)> {
        let mut proj_collisions = Vec::new();
        for (player_idx, player) in self.players.iter().enumerate() {
            if game_mode.has_immunity(player, &player_stats[player_idx]) {
                continue;
            }

            // check piercing invincibility at start to prevent order from mattering
            let player_piercing_invincibility = player.player_piercing_invincibility > 0.0;
            // check for collision with projectiles
            for (proj_idx, proj) in self.projectiles.iter().enumerate() {
                if player_idx as u32 == proj.owner && proj.lifetime < 1.0 && proj.is_from_head == 1
                {
                    continue;
                }
                let proj_card = card_manager.get_referenced_proj(proj.proj_card_idx as usize);

                if proj_card.no_friendly_fire
                    && game_mode.are_friends(proj.owner, player_idx as u32, &self.players)
                {
                    continue;
                }
                if proj_card.no_enemy_fire && proj.owner != player_idx as u32 {
                    continue;
                }
                if player_piercing_invincibility && proj_card.pierce_players {
                    continue;
                }

                let projectile_rot =
                    Quaternion::new(proj.dir[3], proj.dir[0], proj.dir[1], proj.dir[2]);
                let projectile_vectors = [
                    projectile_rot.rotate_vector(Vector3::new(1.0, 0.0, 0.0)),
                    projectile_rot.rotate_vector(Vector3::new(0.0, 1.0, 0.0)),
                    projectile_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0)),
                ];
                let projectile_pos = Point3::new(proj.pos[0], proj.pos[1], proj.pos[2]);
                let adjusted_projectile_size = if proj_card.lock_owner.is_some() {
                    Vector3::new(proj.size[0], proj.size[1], proj.size[2])
                } else {
                    Vector3::new(
                        proj.size[0],
                        proj.size[1],
                        proj.size[2] + proj.vel * time_step / 2.0,
                    )
                };
                let mut collision: Option<Hitsphere> = None;
                'outer: for hitsphere in Entity::HITSPHERES.iter().map(|x| Hitsphere {
                    offset: (player.pos + x.offset * player.size).to_vec(),
                    radius: x.radius * player.size,
                    headshot: x.headshot,
                }) {
                    if let Some(prev_collision) = collision.as_ref() {
                        if (projectile_pos - hitsphere.offset).to_vec().magnitude()
                            > (projectile_pos - prev_collision.offset)
                                .to_vec()
                                .magnitude()
                        {
                            continue 'outer;
                        }
                    }
                    for i in 0..3 {
                        if (hitsphere.offset.dot(projectile_vectors[i])
                            - projectile_pos.dot(projectile_vectors[i]))
                        .abs()
                            > adjusted_projectile_size[i] + hitsphere.radius
                        {
                            continue 'outer;
                        }
                    }
                    collision = Some(hitsphere.clone());
                }
                if let Some(collision) = collision {
                    proj_collisions.push((player_idx, proj_idx, projectile_pos, player.pos, collision.headshot));
                }
            }
        }
        proj_collisions
    }

    fn get_player_effect_stats(&self) -> Vec<PlayerEffectStats> {
        self.players.iter().map(Entity::get_effect_stats).collect()
    }

    fn get_collision_pairs(
        &self,
        card_manager: &CardManager,
        time_step: f32,
        game_mode: &Box<dyn GameMode>,
    ) -> Vec<(usize, usize)> {
        let mut collision_pairs = Vec::new();
        for i in 0..self.projectiles.len() {
            let proj1 = self.projectiles.get(i).unwrap();
            let proj1_card = card_manager.get_referenced_proj(proj1.proj_card_idx as usize);
            let projectile_1_rot =
                Quaternion::new(proj1.dir[3], proj1.dir[0], proj1.dir[1], proj1.dir[2]);
            let projectile_1_vectors = [
                projectile_1_rot.rotate_vector(Vector3::new(1.0, 0.0, 0.0)),
                projectile_1_rot.rotate_vector(Vector3::new(0.0, 1.0, 0.0)),
                projectile_1_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0)),
            ];
            let projectile_1_pos = Point3::new(proj1.pos[0], proj1.pos[1], proj1.pos[2]);
            let adjusted_projectile_1_size = Vector3::new(
                proj1.size[0],
                proj1.size[1],
                proj1.size[2]
                    + if proj1_card.lock_owner.is_none() {
                        proj1.vel * time_step / 2.0
                    } else {
                        0.0
                    },
            );
            let projectile_1_coords = vec![
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
            ]
            .iter()
            .map(|c| {
                let mut pos = projectile_1_pos;
                for i in 0..3 {
                    pos += adjusted_projectile_1_size[i] * projectile_1_vectors[i] * c[i];
                }
                pos
            })
            .collect::<Vec<_>>();
            'second_proj_loop: for j in i + 1..self.projectiles.len() {
                let proj2 = self.projectiles.get(j).unwrap();

                if proj1.health <= 1.0 && proj2.health <= 1.0 {
                    continue;
                }

                let proj2_card = card_manager.get_referenced_proj(proj2.proj_card_idx as usize);

                if (proj1_card.no_friendly_fire && proj2_card.no_friendly_fire)
                    && game_mode.are_friends(proj1.owner, proj2.owner, &self.players)
                {
                    continue;
                }
                if (proj1_card.no_enemy_fire && proj2_card.no_enemy_fire)
                    && proj1.owner != proj2.owner
                {
                    continue;
                }

                let projectile_2_rot =
                    Quaternion::new(proj2.dir[3], proj2.dir[0], proj2.dir[1], proj2.dir[2]);
                let projectile_2_vectors = [
                    projectile_2_rot.rotate_vector(Vector3::new(1.0, 0.0, 0.0)),
                    projectile_2_rot.rotate_vector(Vector3::new(0.0, 1.0, 0.0)),
                    projectile_2_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0)),
                ];
                let projectile_2_pos = Point3::new(proj2.pos[0], proj2.pos[1], proj2.pos[2]);
                let adjusted_projectile_2_size = Vector3::new(
                    proj2.size[0],
                    proj2.size[1],
                    proj2.size[2]
                        + if proj2_card.lock_owner.is_none() {
                            proj2.vel * time_step / 2.0
                        } else {
                            0.0
                        },
                );

                let projectile_2_coords = vec![
                    [-1.0, -1.0, -1.0],
                    [-1.0, -1.0, 1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, 1.0, 1.0],
                    [1.0, -1.0, -1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, -1.0],
                    [1.0, 1.0, 1.0],
                ]
                .iter()
                .map(|c| {
                    let mut pos = projectile_2_pos;
                    for i in 0..3 {
                        pos += adjusted_projectile_2_size[i] * projectile_2_vectors[i] * c[i];
                    }
                    pos
                })
                .collect::<Vec<_>>();

                // sat collision detection
                for i in 0..3 {
                    let (min_proj_2, max_proj_2) = projectile_2_coords
                        .iter()
                        .map(|c| c.to_vec().dot(projectile_1_vectors[i]))
                        .fold((f32::INFINITY, f32::NEG_INFINITY), |acc, x| {
                            (acc.0.min(x), acc.1.max(x))
                        });
                    if min_proj_2
                        > projectile_1_pos.to_vec().dot(projectile_1_vectors[i])
                            + adjusted_projectile_1_size[i]
                        || max_proj_2
                            < projectile_1_pos.to_vec().dot(projectile_1_vectors[i])
                                - adjusted_projectile_1_size[i]
                    {
                        continue 'second_proj_loop;
                    }
                }
                for i in 0..3 {
                    let (min_proj_1, max_proj_1) = projectile_1_coords
                        .iter()
                        .map(|c| c.to_vec().dot(projectile_2_vectors[i]))
                        .fold((f32::INFINITY, f32::NEG_INFINITY), |acc, x| {
                            (acc.0.min(x), acc.1.max(x))
                        });
                    if min_proj_1
                        > projectile_2_pos.to_vec().dot(projectile_2_vectors[i])
                            + adjusted_projectile_2_size[i]
                        || max_proj_1
                            < projectile_2_pos.to_vec().dot(projectile_2_vectors[i])
                                - adjusted_projectile_2_size[i]
                    {
                        continue 'second_proj_loop;
                    }
                }
                // collision detected
                collision_pairs.push((i, j));
            }
        }
        collision_pairs
    }
}

impl Projectile {
    fn simple_step(
        &mut self,
        players: &Vec<Entity>,
        card_manager: &CardManager,
        time_step: f32,
        new_effects: &mut NewEffects,
    ) {
        let proj_card = card_manager.get_referenced_proj(self.proj_card_idx as usize);
        let projectile_rot = Quaternion::new(self.dir[3], self.dir[0], self.dir[1], self.dir[2]);
        let projectile_dir = projectile_rot.rotate_vector(Vector3::new(0.0, 0.0, 1.0));
        let mut proj_vel = projectile_dir * self.vel;

        let new_projectile_rot: Quaternion<f32> = if let Some(direction) = &proj_card.lock_owner {
            let player_dir = players[self.owner as usize].dir;
            let player_up = players[self.owner as usize].up;
            match direction {
                DirectionCard::Forward => {
                    let proj_pos = players[self.owner as usize].pos
                        + 0.1 * proj_card.speed * player_dir
                        - 0.25 * proj_card.gravity * player_up;
                    for i in 0..3 {
                        self.pos[i] = proj_pos[i];
                    }
                    players[self.owner as usize].rot
                }
                DirectionCard::Up => {
                    let proj_pos = players[self.owner as usize].pos
                        + 0.1 * proj_card.speed * Vector3::new(0.0, 1.0, 0.0)
                        - 0.25 * proj_card.gravity * player_up;
                    for i in 0..3 {
                        self.pos[i] = proj_pos[i];
                    }
                    Quaternion::from_arc(projectile_dir, Vector3::new(0.0, 1.0, 0.0), None)
                        * projectile_rot
                }
                DirectionCard::Movement => {
                    let proj_pos = players[self.owner as usize].pos
                        + 0.1 * proj_card.speed * players[self.owner as usize].movement_direction
                        - 0.25 * proj_card.gravity * player_up;
                    for i in 0..3 {
                        self.pos[i] = proj_pos[i];
                    }
                    Quaternion::from_arc(
                        projectile_dir,
                        players[self.owner as usize].movement_direction,
                        None,
                    ) * projectile_rot
                }
                DirectionCard::None => {
                    let proj_pos = players[self.owner as usize].pos
                        + 0.1 * proj_card.speed * player_dir
                        - 0.25 * proj_card.gravity * player_up;
                    for i in 0..3 {
                        self.pos[i] = proj_pos[i];
                    }
                    projectile_rot
                }
            }
        } else {
            proj_vel.y -= proj_card.gravity * time_step;
            for i in 0..3 {
                self.pos[i] += proj_vel[i] * time_step;
            }
            // recompute vel and rot
            if proj_vel.magnitude() < 0.0001 {
                projectile_rot
            } else {
                Quaternion::from_arc(projectile_dir, proj_vel.normalize(), None) * projectile_rot
            }
        };
        self.dir = [
            new_projectile_rot.v[0],
            new_projectile_rot.v[1],
            new_projectile_rot.v[2],
            new_projectile_rot.s,
        ];
        self.vel = proj_vel.magnitude();

        self.lifetime += time_step;
        if self.lifetime >= proj_card.lifetime {
            self.health = 0.0;
            for card_ref in card_manager
                .get_referenced_proj(self.proj_card_idx as usize)
                .on_expiry
                .clone()
            {
                let (proj_effects, vox_effects, effects, status_effects, triggers) = card_manager
                    .get_effects_from_base_card(
                        card_ref,
                        &Point3::new(self.pos[0], self.pos[1], self.pos[2]),
                        &new_projectile_rot,
                        self.owner,
                        false,
                    );
                new_effects.new_projectiles.extend(proj_effects);
                for (pos, material) in vox_effects {
                    new_effects
                        .voxels_to_write
                        .push((pos, material.to_memory()));
                }
                for effect in effects {
                    new_effects.new_effects.push((
                        self.owner as usize,
                        self.owner as usize,
                        false,
                        Point3::new(self.pos[0], self.pos[1], self.pos[2]),
                        projectile_dir,
                        effect,
                    ));
                }
                for status_effects in status_effects {
                    new_effects
                        .new_status_effects
                        .push((self.owner as usize, status_effects));
                }
                new_effects.step_triggers.extend(triggers);
            }
        }

        for (trail_time, trail_card) in proj_card.trail.iter() {
            if self.lifetime % trail_time >= trail_time - time_step {
                let (proj_effects, vox_effects, effects, status_effects, triggers) = card_manager
                    .get_effects_from_base_card(
                        trail_card.clone(),
                        &Point3::new(self.pos[0], self.pos[1], self.pos[2]),
                        &new_projectile_rot,
                        self.owner,
                        false,
                    );
                new_effects.new_projectiles.extend(proj_effects);
                for (pos, material) in vox_effects {
                    new_effects
                        .voxels_to_write
                        .push((pos, material.to_memory()));
                }
                for effect in effects {
                    new_effects.new_effects.push((
                        self.owner as usize,
                        self.owner as usize,
                        false,
                        Point3::new(self.pos[0], self.pos[1], self.pos[2]),
                        projectile_dir,
                        effect,
                    ));
                }
                for status_effects in status_effects {
                    new_effects
                        .new_status_effects
                        .push((self.owner as usize, status_effects));
                }
                new_effects.step_triggers.extend(triggers);
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Hitsphere {
    offset: Vector3<f32>,
    radius: f32,
    headshot: bool,
}

impl Entity {
    const HITSPHERES: [Hitsphere; 6] = [
        Hitsphere {
            offset: Vector3::new(0.0, 0.0, 0.0),
            radius: 0.6,
            headshot: true,
        },
        Hitsphere {
            offset: Vector3::new(0.0, -1.3, 0.0),
            radius: 0.6,
            headshot: false,
        },
        Hitsphere {
            offset: Vector3::new(0.0, -1.9, 0.0),
            radius: 0.9,
            headshot: false,
        },
        Hitsphere {
            offset: Vector3::new(0.0, -2.6, 0.0),
            radius: 0.8,
            headshot: false,
        },
        Hitsphere {
            offset: Vector3::new(0.0, -3.3, 0.0),
            radius: 0.6,
            headshot: false,
        },
        Hitsphere {
            offset: Vector3::new(0.0, -3.8, 0.0),
            radius: 0.6,
            headshot: false,
        },
    ];

    fn simple_step(
        &mut self,
        time_step: f32,
        action: Action,
        player_stats: &Vec<PlayerEffectStats>,
        player_idx: usize,
        card_manager: &CardManager,
        new_effects: &mut NewEffects,
        game_mode: &Box<dyn GameMode>,
    ) {
        let size_change = player_stats[player_idx].size - self.size;
        if size_change > 0.0 {
            self.pos += size_change
                * (0.5 * PLAYER_HITBOX_SIZE.y - PLAYER_HITBOX_OFFSET.y)
                * vec3(0.0, 1.0, 0.0);
        }
        self.size = player_stats[player_idx].size;
        if let Some(HealthSection::Health(current, max)) = self.health.get_mut(0).as_mut() {
            *max = player_stats[player_idx].max_health;
            *current = current.min(*max);
        }

        if self.respawn_timer > 0.0 {
            self.respawn_timer -= time_step;
            if self.respawn_timer <= 0.0 {
                self.pos = game_mode.spawn_location(&self);
                self.vel = Vector3::new(0.0, 0.0, 0.0);
                self.health = vec![HealthSection::Health(
                    player_stats[player_idx].max_health,
                    player_stats[player_idx].max_health,
                )];
                self.status_effects.clear();
            }
        }
        if self.player_piercing_invincibility > 0.0 {
            self.player_piercing_invincibility -= time_step;
        }
        if self.on_hit_passive_cooldown > 0.0 {
            self.on_hit_passive_cooldown -= time_step;
        }
        if let Some(action) = action.primary_action {
            self.facing[0] = (self.facing[0] - action.aim[0] + 2.0 * PI) % (2.0 * PI);
            self.facing[1] = (self.facing[1] - action.aim[1])
                .min(PI / 2.0)
                .max(-PI / 2.0);
            self.rot =
                Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), Rad(self.facing[0]))
                    * Quaternion::from_axis_angle(
                        Vector3::new(1.0, 0.0, 0.0),
                        Rad(-self.facing[1]),
                    );
            let horizontal_rot =
                Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), Rad(self.facing[0]));
            self.dir = self.rot * Vector3::new(0.0, 0.0, 1.0);
            self.right = self.rot * Vector3::new(-1.0, 0.0, 0.0);
            self.up = self.right.cross(self.dir).normalize();
            let mut move_vec = Vector3::new(0.0, 0.0, 0.0);
            let player_forward = horizontal_rot * Vector3::new(0.0, 0.0, 1.0);
            let player_right = horizontal_rot * Vector3::new(-1.0, 0.0, 0.0);
            let mut speed_multiplier = 1.0;
            if action.forward {
                move_vec += player_forward;
            }
            if action.backward {
                move_vec -= player_forward;
            }
            if action.left {
                move_vec -= player_right;
            }
            if action.right {
                move_vec += player_right;
            }
            if action.jump {
                move_vec += Vector3::new(0.0, 0.25, 0.0);
            }
            if action.crouch {
                move_vec -= Vector3::new(0.0, 0.6, 0.0);
                speed_multiplier = 0.5;
            }
            if move_vec.magnitude() > 0.0 {
                move_vec = move_vec.normalize();
            }
            self.movement_direction = move_vec;
            let accel_speed = speed_multiplier
                * if self.collision_vec != Vector3::new(0, 0, 0) {
                    80.0
                } else {
                    18.0
                };
            self.vel += accel_speed
                * Vector3::new(
                    player_stats[player_idx].speed,
                    1.0,
                    player_stats[player_idx].speed,
                )
                .mul_element_wise(move_vec)
                * time_step;

            if action.jump {
                self.vel += player_stats[player_idx].speed
                    * self
                        .collision_vec
                        .zip(Vector3::new(0.3, 13.0, 0.3), |c, m| c as f32 * m);
            }

            if game_mode.player_mode(self).can_interact() {
                for (cooldown_idx, cooldown) in self.abilities.iter_mut().enumerate() {
                    if cooldown.remaining_charges > 0 && cooldown.recovery <= 0.0 {
                        for (ability_idx, ability) in cooldown.ability.abilities.iter().enumerate()
                        {
                            if *action
                                .activate_ability
                                .get(cooldown_idx)
                                .map(|cd| cd.get(ability_idx).unwrap_or(&false))
                                .unwrap_or(&false)
                                && !player_stats[player_idx].lockout
                            {
                                cooldown.remaining_charges -= 1;
                                cooldown.recovery = cooldown.value.1[ability_idx];
                                let (proj_effects, vox_effects, effects, status_effects, triggers) =
                                    card_manager.get_effects_from_base_card(
                                        ability.0,
                                        &self.pos,
                                        &self.rot,
                                        player_idx as u32,
                                        true,
                                    );
                                new_effects.new_projectiles.extend(proj_effects);
                                for (pos, material) in vox_effects {
                                    new_effects
                                        .voxels_to_write
                                        .push((pos, material.to_memory()));
                                }
                                for effect in effects {
                                    new_effects.new_effects.push((
                                        player_idx, player_idx, false, self.pos, self.dir, effect,
                                    ));
                                }
                                for status_effects in status_effects {
                                    new_effects
                                        .new_status_effects
                                        .push((player_idx, status_effects));
                                }
                                new_effects.step_triggers.extend(triggers);
                                break;
                            }
                        }
                    }
                }
            }
        }
        for ability in self.abilities.iter_mut() {
            if ability.ability.is_reloading {
                if ability.remaining_charges == 0 {
                    ability.cooldown = 0.0;
                    ability.remaining_charges = ability.ability.max_charges;
                    ability.recovery += ability.value.0;
                }
            } else {
                if ability.cooldown > 0.0 && ability.remaining_charges < ability.ability.max_charges
                {
                    ability.cooldown -= time_step;
                } else if ability.remaining_charges < ability.ability.max_charges {
                    ability.cooldown = ability.value.0;
                    ability.remaining_charges += 1;
                }
            }
            if ability.recovery > 0.0 {
                ability.recovery -= time_step;
            }
        }

        self.hitmarker.0 -= 3.0 * (10.0 + self.hitmarker.0) * time_step;
        self.hitmarker.1 -= 3.0 * (10.0 + self.hitmarker.1) * time_step;
        self.hitmarker.0 = self.hitmarker.0.max(0.0);
        self.hitmarker.1 = self.hitmarker.1.max(0.0);
        for hurtmarker in self.hurtmarkers.iter_mut() {
            hurtmarker.2 -= time_step;
        }
        self.hurtmarkers.retain(|hurtmarker| hurtmarker.2 > 0.0);

        for health_section in self.health.iter_mut() {
            match health_section {
                HealthSection::Overhealth(_health, duration) => {
                    *duration -= time_step;
                }
                _ => {}
            }
        }
        self.health.retain(|health_section| match health_section {
            HealthSection::Health(_health, _max_health) => true,
            HealthSection::Overhealth(health, duration) => *health > 0.0 && *duration > 0.0,
        });
    }

    pub fn adjust_health(&mut self, adjustment: f32) {
        if adjustment > 0.0 {
            let mut healing_left = adjustment;
            let mut health_idx = 0;
            while healing_left > 0.0 {
                let health_section = &mut self.health[health_idx];
                match health_section {
                    HealthSection::Health(current, max) => {
                        let health_to_add = (*max - *current).min(healing_left);
                        *current += health_to_add;
                        healing_left -= health_to_add;
                    }
                    HealthSection::Overhealth(_current, _duration) => {
                        // overhealth is not affected by healing
                    }
                }
                health_idx += 1;
                if health_idx >= self.health.len() {
                    break;
                }
            }
        } else {
            let mut damage_left = -adjustment;
            let mut health_idx = self.health.len() - 1;
            while damage_left > 0.0 {
                let health_section = &mut self.health[health_idx];
                match health_section {
                    HealthSection::Health(current, _) => {
                        let health_to_remove = (*current).min(damage_left);
                        *current -= health_to_remove;
                        damage_left -= health_to_remove;
                    }
                    HealthSection::Overhealth(current, _duration) => {
                        let health_to_remove = (*current).min(damage_left);
                        *current -= health_to_remove;
                        damage_left -= health_to_remove;
                    }
                }
                if health_idx == 0 {
                    break;
                }
                health_idx -= 1;
            }
        }
    }

    pub fn get_health_stats(&self) -> (f32, f32) {
        let mut current_health = 0.0;
        let mut max_health = 0.0;
        for health_section in self.health.iter() {
            match health_section {
                HealthSection::Health(current, max) => {
                    current_health += *current;
                    max_health += *max;
                }
                HealthSection::Overhealth(current, _duration) => {
                    current_health += *current;
                    max_health += *current;
                }
            }
        }
        (current_health, max_health)
    }

    pub fn get_effect_stats(&self) -> PlayerEffectStats {
        let mut speed = 1.0;
        let mut damage_taken = 1.0;
        let mut gravity = Vector3::new(0.0, -1.0, 0.0);
        let mut size = 1.0;
        let mut max_health = PLAYER_BASE_MAX_HEALTH;
        let mut invincible = false;
        let mut lockout = false;

        for status_effect in self
            .status_effects
            .iter()
            .map(|e| &e.effect)
            .chain(self.passive_abilities.iter())
        {
            match status_effect {
                ReferencedStatusEffect::DamageOverTime(_) => {
                    // wait for damage taken to be calculated
                }
                ReferencedStatusEffect::Speed(stacks) => {
                    speed *=
                        StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::Speed, *stacks)
                            .get_effect_value();
                }
                ReferencedStatusEffect::IncreaseDamageTaken(stacks) => {
                    damage_taken *= StatusEffect::SimpleStatusEffect(
                        SimpleStatusEffectType::IncreaseDamageTaken,
                        *stacks,
                    )
                    .get_effect_value();
                }
                ReferencedStatusEffect::IncreaseGravity(direction, stacks) => {
                    gravity += StatusEffect::SimpleStatusEffect(
                        SimpleStatusEffectType::IncreaseGravity(direction.clone()),
                        *stacks,
                    )
                    .get_effect_value()
                        * match direction {
                            DirectionCard::Forward => self.dir,
                            DirectionCard::Up => Vector3::new(0.0, 1.0, 0.0),
                            DirectionCard::Movement => {
                                if self.movement_direction.magnitude() == 0.0 {
                                    Vector3::new(0.0, 0.0, 0.0)
                                } else {
                                    self.movement_direction.normalize()
                                }
                            }
                            DirectionCard::None => Vector3::new(0.0, 0.0, 0.0),
                        };
                }
                ReferencedStatusEffect::Overheal(_) => {
                    // managed seperately
                }
                ReferencedStatusEffect::Grow(stacks) => {
                    size *= StatusEffect::SimpleStatusEffect(SimpleStatusEffectType::Grow, *stacks)
                        .get_effect_value();
                }
                ReferencedStatusEffect::IncreaseMaxHealth(stacks) => {
                    max_health += StatusEffect::SimpleStatusEffect(
                        SimpleStatusEffectType::IncreaseMaxHealth,
                        *stacks,
                    )
                    .get_effect_value();
                }
                ReferencedStatusEffect::Invincibility => {
                    invincible = true;
                }
                ReferencedStatusEffect::Trapped => {
                    speed *= 0.0;
                }
                ReferencedStatusEffect::Lockout => {
                    lockout = true;
                }
                ReferencedStatusEffect::OnHit(_) => {
                    // managed seperately
                }
            }
        }

        if size > 5.0 {
            damage_taken *= size - 4.0;
            size = 5.0;
        }

        PlayerEffectStats {
            speed,
            damage_taken,
            gravity,
            size,
            max_health,
            invincible,
            lockout,
        }
    }
}
