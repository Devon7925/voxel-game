use std::f32::consts::PI;

use cgmath::{
    vec3, ElementWise, EuclideanSpace, InnerSpace, One, Point3, Quaternion, Rad, Rotation,
    Rotation3, Vector2, Vector3,
};
use itertools::Itertools;

use vulkano::buffer::subbuffer::BufferReadGuard;

use crate::{
    card_system::{
        BaseCard, CardManager, DirectionCard, ReferencedCooldown, ReferencedEffect,
        ReferencedStatusEffect, ReferencedStatusEffects, ReferencedTrigger, SimpleStatusEffectType,
        StatusEffect, VoxelMaterial,
    },
    game_manager::GameState,
    game_modes::GameMode,
    rollback_manager::Action,
    voxel_sim_manager::{Projectile, VoxelComputePipeline},
    CHUNK_SIZE, PLAYER_BASE_MAX_HEALTH, PLAYER_DENSITY, PLAYER_HITBOX_OFFSET, PLAYER_HITBOX_SIZE,
    RESPAWN_TIME,
};
use voxel_shared::GameSettings;

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
        game_state: &GameState,
        game_settings: &GameSettings,
        game_mode: &Box<dyn GameMode>,
    ) {
        let voxels = vox_compute.voxels();
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

        {
            let voxel_reader = voxels.read().unwrap();
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
                    &voxel_reader,
                    &vox_compute.cpu_chunks(),
                    &mut new_effects,
                    game_state,
                    game_settings,
                    game_mode,
                );
            }
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

        let proj_collisions =
            self.get_player_proj_collisions(&player_stats, card_manager, game_mode, time_step);

        proj_collisions
            .iter()
            .filter(|(_, proj_idx, damage_source_location, collision)| {
                let vec_start = damage_source_location.to_vec();
                let vec_end = collision.offset;
                self.projectiles
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| {
                        collision_pairs.contains(&(*proj_idx.min(idx), *proj_idx.max(idx)))
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
                                (
                                    (-1.0 - adj_vec_start[i]) / vec_dir[i],
                                    (1.0 - adj_vec_start[i]) / vec_dir[i],
                                )
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
            .for_each(|(player_idx, proj_idx, _, collision)| {
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
                if collision.headshot {
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
                    for effect in effects {
                        new_effects.new_effects.push((
                            *player_idx,
                            proj.owner as usize,
                            was_headshot,
                            Point3::from_vec(collision.offset),
                            ((collision.offset - projectile_pos.to_vec()).normalize()
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

        let mut player_player_collision_pairs: Vec<(usize, usize)> = vec![];
        for i in 0..self.players.len() {
            let player1 = self.players.get(i).unwrap();
            if player1.respawn_timer > 0.0 || player_stats[i].invincible {
                continue;
            }
            for j in 0..self.players.len() {
                if game_mode.are_friends(i as u32, j as u32, &self.players) {
                    continue;
                }
                let player2 = self.players.get(j).unwrap();
                if player2.respawn_timer > 0.0 || player_stats[j].invincible {
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
        for (i, j) in player_player_collision_pairs {
            let player1_pos = self.players.get(i).unwrap().pos;
            let player2_pos = self.players.get(j).unwrap().pos;
            let hit_effects =
                {
                    let player1 = self.players.get_mut(i).unwrap();
                    let hit_effects =
                        player1
                            .status_effects
                            .iter()
                            .filter_map(|effect| match effect {
                                AppliedStatusEffect {
                                    effect: ReferencedStatusEffect::OnHit(hit_card),
                                    time_left: _,
                                } => Some(hit_card),
                                _ => None,
                            })
                            .chain((player1.on_hit_passive_cooldown <= 0.0).then_some(
                                player1.passive_abilities.iter().filter_map(
                                    |effect| match effect {
                                        ReferencedStatusEffect::OnHit(hit_card) => Some(hit_card),
                                        _ => None,
                                    },
                                ),
                            ).into_iter().flatten())
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
                    if was_headshot {
                        actor.hitmarker.1 += damage as f32;
                    } else {
                        actor.hitmarker.0 += damage as f32;
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

    fn get_player_proj_collisions(
        &self,
        player_stats: &Vec<PlayerEffectStats>,
        card_manager: &CardManager,
        game_mode: &Box<dyn GameMode>,
        time_step: f32,
    ) -> Vec<(usize, usize, Point3<f32>, Hitsphere)> {
        let mut proj_collisions = Vec::new();
        for (player_idx, player) in self.players.iter().enumerate() {
            if player.respawn_timer > 0.0 || player_stats[player_idx].invincible {
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
                    proj_collisions.push((player_idx, proj_idx, projectile_pos, collision));
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

pub fn is_inbounds(
    global_pos: Point3<u32>,
    game_state: &GameState,
    game_settings: &GameSettings,
) -> bool {
    (global_pos / CHUNK_SIZE as u32).zip(game_state.start_pos, |a, b| a >= b)
        == Point3::new(true, true, true)
        && (global_pos / CHUNK_SIZE as u32)
            .zip(game_state.start_pos + game_settings.render_size, |a, b| {
                a < b
            })
            == Point3::new(true, true, true)
}

pub fn get_index(
    global_pos: Point3<u32>,
    cpu_chunks: &Vec<Vec<Vec<u32>>>,
    game_state: &GameState,
    game_settings: &GameSettings,
) -> Option<u32> {
    if !is_inbounds(global_pos, game_state, game_settings) {
        return None;
    }
    let chunk_pos = (global_pos / CHUNK_SIZE as u32)
        .zip(Point3::from_vec(game_settings.render_size), |a, b| a % b);
    let pos_in_chunk = global_pos % CHUNK_SIZE as u32;
    let chunk_idx = cpu_chunks[chunk_pos.x as usize][chunk_pos.y as usize][chunk_pos.z as usize];
    let idx_in_chunk = pos_in_chunk.x * CHUNK_SIZE as u32 * CHUNK_SIZE as u32
        + pos_in_chunk.y * CHUNK_SIZE as u32
        + pos_in_chunk.z;
    Some(chunk_idx * CHUNK_SIZE as u32 * CHUNK_SIZE as u32 * CHUNK_SIZE as u32 + idx_in_chunk)
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
        voxel_reader: &BufferReadGuard<'_, [u32]>,
        cpu_chunks: &Vec<Vec<Vec<u32>>>,
        new_effects: &mut NewEffects,
        game_state: &GameState,
        game_settings: &GameSettings,
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
            return;
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

            for (cooldown_idx, cooldown) in self.abilities.iter_mut().enumerate() {
                if cooldown.remaining_charges > 0 && cooldown.recovery <= 0.0 {
                    for (ability_idx, ability) in cooldown.ability.abilities.iter().enumerate() {
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

        //volume effects
        let start_pos =
            self.pos + self.size * PLAYER_HITBOX_OFFSET - self.size * PLAYER_HITBOX_SIZE / 2.0;
        let end_pos =
            self.pos + self.size * PLAYER_HITBOX_OFFSET + self.size * PLAYER_HITBOX_SIZE / 2.0;
        let start_voxel_pos = start_pos.map(|c| c.floor() as u32);
        let iter_counts = end_pos.zip(start_voxel_pos, |a, b| a.floor() as u32 - b + 1);
        let mut nearby_density = 0.0;
        let mut directional_density = Vector3::new(0.0, 0.0, 0.0);
        for x in 0..iter_counts.x {
            for y in 0..iter_counts.y {
                for z in 0..iter_counts.z {
                    let voxel_pos = start_voxel_pos + Vector3::new(x, y, z);
                    let overlapping_volume = voxel_pos.zip(end_pos, |a, b| b.min(a as f32 + 1.0))
                        - voxel_pos.zip(start_pos, |a, b| b.max(a as f32));
                    let overlapping_volume =
                        overlapping_volume.x * overlapping_volume.y * overlapping_volume.z;
                    let material = if is_inbounds(voxel_pos, game_state, game_settings) {
                        let idx = get_index(voxel_pos, cpu_chunks, game_state, game_settings);
                        if let Some(idx) = idx {
                            VoxelMaterial::from_memory(voxel_reader[idx as usize])
                        } else {
                            VoxelMaterial::Unloaded
                        }
                    } else {
                        VoxelMaterial::Unloaded
                    };
                    let density = material.density();
                    nearby_density += overlapping_volume * density;
                    directional_density += overlapping_volume
                        * density
                        * (voxel_pos.map(|c| c as f32 + 0.5)
                            - (self.pos + self.size * PLAYER_HITBOX_OFFSET))
                        / self.size;
                }
            }
        }
        nearby_density /=
            self.size.powi(3) * PLAYER_HITBOX_SIZE.x * PLAYER_HITBOX_SIZE.y * PLAYER_HITBOX_SIZE.z;
        directional_density /=
            self.size.powi(3) * PLAYER_HITBOX_SIZE.x * PLAYER_HITBOX_SIZE.y * PLAYER_HITBOX_SIZE.z;

        self.vel += (PLAYER_DENSITY - nearby_density)
            * player_stats[player_idx].gravity
            * 11.428571428571429
            * time_step;
        if directional_density.magnitude() * time_step > 0.001 {
            self.vel -= 0.5 * directional_density * time_step;
        }
        if self.vel.magnitude() > 0.0 {
            self.vel -= nearby_density * 0.0375 * self.vel * self.vel.magnitude() * time_step
                + 0.2 * self.vel.normalize() * time_step;
        }
        let prev_collision_vec = self.collision_vec.clone();
        self.collision_vec = Vector3::new(0, 0, 0);
        self.collide_player(
            time_step,
            voxel_reader,
            cpu_chunks,
            prev_collision_vec,
            game_state,
            game_settings,
        );

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

    fn collide_player(
        &mut self,
        time_step: f32,
        voxel_reader: &BufferReadGuard<'_, [u32]>,
        cpu_chunks: &Vec<Vec<Vec<u32>>>,
        prev_collision_vec: Vector3<i32>,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) {
        let collision_corner_offset = self
            .vel
            .map(|c| c.signum())
            .zip(PLAYER_HITBOX_SIZE, |a, b| a * b)
            * 0.5
            * self.size;
        let mut distance_to_move = self.vel * time_step;
        let mut iteration_counter = 0;

        while distance_to_move.magnitude() > 0.0 {
            iteration_counter += 1;

            let player_move_pos =
                self.pos + PLAYER_HITBOX_OFFSET * self.size + collision_corner_offset;
            let vel_dir = distance_to_move.normalize();
            let delta = ray_box_dist(player_move_pos, vel_dir);
            let mut dist_diff = delta.x.min(delta.y).min(delta.z);
            if dist_diff == 0.0 {
                dist_diff = distance_to_move.magnitude();
                if delta.x != 0.0 {
                    dist_diff = dist_diff.min(delta.x);
                }
                if delta.y != 0.0 {
                    dist_diff = dist_diff.min(delta.y);
                }
                if delta.z != 0.0 {
                    dist_diff = dist_diff.min(delta.z);
                }
            } else if dist_diff > distance_to_move.magnitude() {
                self.pos += distance_to_move;
                break;
            }

            if iteration_counter > 100 {
                println!(
                    "iteration counter exceeded with dtm {:?} and delta {:?}",
                    distance_to_move, delta
                );
                break;
            }

            distance_to_move -= dist_diff * vel_dir;
            'component_loop: for component in 0..3 {
                let mut fake_pos = self.pos;
                fake_pos[component] += dist_diff * vel_dir[component];
                let player_move_pos =
                    fake_pos + PLAYER_HITBOX_OFFSET * self.size + collision_corner_offset;
                if delta[component] <= delta[(component + 1) % 3]
                    && delta[component] <= delta[(component + 2) % 3]
                {
                    let mut start_pos = fake_pos + PLAYER_HITBOX_OFFSET * self.size
                        - 0.5 * self.size * PLAYER_HITBOX_SIZE;
                    start_pos[component] = player_move_pos[component];
                    let x_iter_count = (start_pos[(component + 1) % 3]
                        + self.size * PLAYER_HITBOX_SIZE[(component + 1) % 3])
                        .floor()
                        - (start_pos[(component + 1) % 3]).floor();
                    let z_iter_count = (start_pos[(component + 2) % 3]
                        + self.size * PLAYER_HITBOX_SIZE[(component + 2) % 3])
                        .floor()
                        - (start_pos[(component + 2) % 3]).floor();

                    let mut x_vec = Vector3::new(0.0, 0.0, 0.0);
                    let mut z_vec = Vector3::new(0.0, 0.0, 0.0);
                    x_vec[(component + 1) % 3] = 1.0;
                    z_vec[(component + 2) % 3] = 1.0;
                    for x_iter in 0..=(x_iter_count as u32) {
                        for z_iter in 0..=(z_iter_count as u32) {
                            let pos = start_pos + x_iter as f32 * x_vec + z_iter as f32 * z_vec;
                            let voxel_pos = pos.map(|c| c.floor() as u32);
                            let voxel = if let Some(index) =
                                get_index(voxel_pos, cpu_chunks, game_state, game_settings)
                            {
                                voxel_reader[index as usize]
                            } else {
                                VoxelMaterial::Unloaded.to_memory()
                            };
                            let voxel_material = VoxelMaterial::from_memory(voxel);
                            if !voxel_material.is_passthrough() {
                                if component != 1
                                    && prev_collision_vec[1] == 1
                                    && (pos - start_pos).y < 1.0
                                    && self.can_step_up(
                                        voxel_reader,
                                        cpu_chunks,
                                        component,
                                        player_move_pos,
                                        game_state,
                                        game_settings,
                                    )
                                {
                                    self.pos = fake_pos;
                                    self.pos[1] += 1.0;
                                    continue 'component_loop;
                                }

                                self.vel[component] = 0.0;
                                // apply friction
                                let perp_vel = Vector2::new(
                                    self.vel[(component + 1) % 3],
                                    self.vel[(component + 2) % 3],
                                );
                                if perp_vel.magnitude() > 0.0 {
                                    let friction_factor = voxel_material.get_friction();
                                    let friction = Vector2::new(
                                        (friction_factor * 0.5 * perp_vel.normalize().x
                                            + friction_factor * perp_vel.x)
                                            * time_step,
                                        (friction_factor * 0.5 * perp_vel.normalize().y
                                            + friction_factor * perp_vel.y)
                                            * time_step,
                                    );
                                    if friction.magnitude() > perp_vel.magnitude() {
                                        self.vel[(component + 1) % 3] = 0.0;
                                        self.vel[(component + 2) % 3] = 0.0;
                                    } else {
                                        self.vel[(component + 1) % 3] -= friction.x;
                                        self.vel[(component + 2) % 3] -= friction.y;
                                    }
                                }
                                self.collision_vec[component] = -vel_dir[component].signum() as i32;
                                distance_to_move[component] = 0.0;
                                continue 'component_loop;
                            }
                        }
                    }
                }
                self.pos = fake_pos;
            }
        }
    }

    fn can_step_up(
        &self,
        voxel_reader: &BufferReadGuard<'_, [u32]>,
        cpu_chunks: &Vec<Vec<Vec<u32>>>,
        component: usize,
        player_move_pos: Point3<f32>,
        game_state: &GameState,
        game_settings: &GameSettings,
    ) -> bool {
        let mut start_pos =
            self.pos + PLAYER_HITBOX_OFFSET * self.size - 0.5 * self.size * PLAYER_HITBOX_SIZE;
        start_pos[component] = player_move_pos[component];
        start_pos[1] += 1.0;
        let x_iter_count = (start_pos[(component + 1) % 3]
            + self.size * PLAYER_HITBOX_SIZE[(component + 1) % 3])
            .floor()
            - (start_pos[(component + 1) % 3]).floor();
        let z_iter_count = (start_pos[(component + 2) % 3]
            + self.size * PLAYER_HITBOX_SIZE[(component + 2) % 3])
            .floor()
            - (start_pos[(component + 2) % 3]).floor();

        let mut x_vec = Vector3::new(0.0, 0.0, 0.0);
        let mut z_vec = Vector3::new(0.0, 0.0, 0.0);
        x_vec[(component + 1) % 3] = 1.0;
        z_vec[(component + 2) % 3] = 1.0;
        for x_iter in 0..=(x_iter_count as u32) {
            for z_iter in 0..=(z_iter_count as u32) {
                let pos = start_pos + x_iter as f32 * x_vec + z_iter as f32 * z_vec;
                let voxel_pos = pos.map(|c| c.floor() as u32);
                let voxel = if let Some(index) =
                    get_index(voxel_pos, cpu_chunks, game_state, game_settings)
                {
                    voxel_reader[index as usize]
                } else {
                    VoxelMaterial::Unloaded.to_memory()
                };
                let voxel_material = VoxelMaterial::from_memory(voxel);
                if !voxel_material.is_passthrough() {
                    return false;
                }
            }
        }
        true
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

fn ray_box_dist(pos: Point3<f32>, ray: Vector3<f32>) -> Vector3<f32> {
    let vmin = pos.map(|c| c.floor());
    let vmax = pos.map(|c| c.ceil());
    let norm_min_diff: Vector3<f32> =
        (vmin - pos).zip(ray, |n, d| if d == 0.0 { 2.0 } else { n / d });
    let norm_max_diff: Vector3<f32> =
        (vmax - pos).zip(ray, |n, d| if d == 0.0 { 2.0 } else { n / d });
    return norm_min_diff.zip(norm_max_diff, |min_diff, max_diff| min_diff.max(max_diff));
}
