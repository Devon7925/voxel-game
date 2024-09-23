#version 450
#include <common.glsl>

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, r16ui) uniform uimage3D chunks;
layout(set = 0, binding = 1, r32ui) uniform uimage3D voxels;
layout(set = 0, binding = 2) buffer Projectiles {
    Projectile projectiles[];
};
layout(set = 0, binding = 7) buffer Players {
    Player players[];
};
layout(set = 0, binding = 8) buffer Collisions {
    Collision collisions[];
};

layout(push_constant) uniform SimData {
    uvec3 render_size;
    uvec3 start_pos;
    uvec3 voxel_update_offset;
    float dt;
    uint projectile_count;
    uint player_count;
    uint worldgen_count;
    int unload_index;
    uint unload_component;
    uint voxel_write_count;
    uint worldgen_seed;
} sim_data;

ivec3 get_index(uvec3 global_pos) {
    uvec4 indicies = get_indicies(global_pos, sim_data.render_size);
    uint z = imageLoad(chunks, ivec3(indicies.xyz)).x;
    uint y = z / 1024;
    z = z % 1024;
    uint x = y / 1024;
    y = y % 1024;
    return ivec3(x * CHUNK_SIZE, y * CHUNK_SIZE, z * CHUNK_SIZE) + ivec3(global_pos % CHUNK_SIZE);
}

uint get_data_unchecked(uvec3 global_pos) {
    return imageLoad(voxels, get_index(global_pos)).x;
}

uint get_data(uvec3 global_pos) {
    uvec3 start_offset = CHUNK_SIZE * sim_data.start_pos;
    if (any(lessThan(global_pos, start_offset))) return MAT_OOB << 24;
    uvec3 rel_pos = global_pos - start_offset;
    if (any(greaterThanEqual(rel_pos, CHUNK_SIZE * sim_data.render_size))) return MAT_OOB << 24;
    return get_data_unchecked(global_pos);
}

void set_data(uvec3 global_pos, uint data) {
    uvec3 start_offset = CHUNK_SIZE * sim_data.start_pos;
    if (any(lessThan(global_pos, start_offset))) return;
    uvec3 rel_pos = global_pos - start_offset;
    if (any(greaterThanEqual(rel_pos, CHUNK_SIZE * sim_data.render_size))) return;
    imageStore(voxels, get_index(global_pos), uvec4(data, 0, 0, 0));
}

vec3 ray_box_dist(vec3 pos, vec3 ray, vec3 vmin, vec3 vmax) {
    vec3 normMinDiff = (vmin - pos) / ray;
    vec3 normMaxDiff = (vmax - pos) / ray;
    return max(normMinDiff, normMaxDiff);
}

bool can_step_up(
    Player player,
    uint component,
    vec3 player_move_pos
) {
    vec3 start_pos =
        player.pos.xyz + PLAYER_HITBOX_OFFSET * player.size - 0.5 * player.size * PLAYER_HITBOX_SIZE;
    start_pos[component] = player_move_pos[component];
    start_pos[1] += 1.0;
    int x_iter_count = int(floor(start_pos[(component + 1) % 3]
                    + player.size * PLAYER_HITBOX_SIZE[(component + 1) % 3])
                - floor(start_pos[(component + 1) % 3]));
    int z_iter_count = int(floor(start_pos[(component + 2) % 3]
                    + player.size * PLAYER_HITBOX_SIZE[(component + 2) % 3])
                - floor(start_pos[(component + 2) % 3]));

    vec3 x_vec = vec3(0.0);
    vec3 z_vec = vec3(0.0);
    x_vec[(component + 1) % 3] = 1.0;
    z_vec[(component + 2) % 3] = 1.0;
    for (int x_iter = 0; x_iter <= x_iter_count; x_iter++) {
        for (int z_iter = 0; z_iter <= z_iter_count; z_iter++) {
            vec3 pos = start_pos + float(x_iter) * x_vec + float(z_iter) * z_vec;
            uvec3 voxel_pos = uvec3(pos);
            uint voxel = get_data(voxel_pos);
            uint voxel_material = voxel >> 24;
            if (!physics_properties[voxel_material].is_fluid) {
                return false;
            }
        }
    }
    return true;
}

void collide_player(
    inout Player player,
    vec3 prev_collision_vec
) {
    vec3 collision_corner_offset = sign(player.vel.xyz) * PLAYER_HITBOX_SIZE
            * 0.5
            * player.size;
    vec3 distance_to_move = player.vel.xyz * sim_data.dt;
    uint iteration_counter = 0;

    while (dot(distance_to_move, distance_to_move) > 0.0001) {
        iteration_counter += 1;

        vec3 player_move_pos =
            player.pos.xyz + PLAYER_HITBOX_OFFSET * player.size + collision_corner_offset;
        vec3 vel_dir = normalize(distance_to_move);
        vec3 delta = ray_box_dist(player_move_pos, vel_dir, floor(player_move_pos), ceil(player_move_pos));
        if (isnan(delta.x)) {
            delta.x = 2.0;
        }
        if (isnan(delta.y)) {
            delta.y = 2.0;
        }
        if (isnan(delta.z)) {
            delta.z = 2.0;
        }
        float dist_diff = min(delta.x, min(delta.y, delta.z));
        if (dist_diff == 0.0) {
            dist_diff = length(distance_to_move);
            if (delta.x != 0.0) {
                dist_diff = min(dist_diff, delta.x);
            }
            if (delta.y != 0.0) {
                dist_diff = min(dist_diff, delta.y);
            }
            if (delta.z != 0.0) {
                dist_diff = min(dist_diff, delta.z);
            }
        } else if (dist_diff > length(distance_to_move)) {
            player.pos.xyz += distance_to_move;
            break;
        }

        if (iteration_counter > 100) {
            break;
        }

        distance_to_move -= dist_diff * vel_dir;
        for (uint component = 0; component < 3; component++) {
            vec3 fake_pos = player.pos.xyz;
            fake_pos[component] += dist_diff * vel_dir[component];
            vec3 player_move_pos =
                fake_pos + PLAYER_HITBOX_OFFSET * player.size + collision_corner_offset;
            if (delta[component] <= delta[(component + 1) % 3]
                    && delta[component] <= delta[(component + 2) % 3]) {
                vec3 start_pos = fake_pos + PLAYER_HITBOX_OFFSET * player.size
                        - 0.5 * player.size * PLAYER_HITBOX_SIZE;
                start_pos[component] = player_move_pos[component];
                int x_iter_count = int(floor(start_pos[(component + 1) % 3]
                                + player.size * PLAYER_HITBOX_SIZE[(component + 1) % 3])
                            - floor(start_pos[(component + 1) % 3]));
                int z_iter_count = int(floor(start_pos[(component + 2) % 3]
                                + player.size * PLAYER_HITBOX_SIZE[(component + 2) % 3])
                            - floor(start_pos[(component + 2) % 3]));

                vec3 x_vec = vec3(0);
                vec3 z_vec = vec3(0);
                x_vec[(component + 1) % 3] = 1.0;
                z_vec[(component + 2) % 3] = 1.0;
                bool found_position = false;
                for (int x_iter = 0; x_iter <= x_iter_count; x_iter++) {
                    for (int z_iter = 0; z_iter <= z_iter_count; z_iter++) {
                        vec3 pos = start_pos + x_iter * x_vec + z_iter * z_vec;
                        uvec3 voxel_pos = uvec3(pos);
                        uint voxel = get_data(voxel_pos);
                        uint voxel_material = voxel >> 24;
                        if (!physics_properties[voxel_material].is_fluid) {
                            if (component != 1
                                    && prev_collision_vec[1] == 1
                                    && (pos - start_pos).y < 1.0
                                    && can_step_up(
                                        player,
                                        component,
                                        player_move_pos
                                    ))
                            {
                                player.pos.xyz = fake_pos;
                                player.pos[1] += 1.0;
                                found_position = true;
                                break;
                            }

                            player.vel[component] = 0.0;
                            // apply friction
                            vec2 perp_vel = vec2(
                                    player.vel[(component + 1) % 3],
                                    player.vel[(component + 2) % 3]
                                );
                            if (length(perp_vel) > 0.0) {
                                float friction_factor = physics_properties[voxel_material].friction;
                                vec2 friction = vec2((friction_factor * 0.5 * normalize(perp_vel).x
                                            + friction_factor * perp_vel.x)
                                            * sim_data.dt,
                                        (friction_factor * 0.5 * normalize(perp_vel).y
                                            + friction_factor * perp_vel.y)
                                            * sim_data.dt
                                    );
                                if (length(friction) > length(perp_vel)) {
                                    player.vel[(component + 1) % 3] = 0.0;
                                    player.vel[(component + 2) % 3] = 0.0;
                                } else {
                                    player.vel[(component + 1) % 3] -= friction.x;
                                    player.vel[(component + 2) % 3] -= friction.y;
                                }
                            }
                            player.collision_vec[component] = int(-sign(vel_dir[component]));
                            distance_to_move[component] = 0.0;
                            found_position = true;
                            break;
                        }
                    }
                    if (found_position) {
                        break;
                    }
                }
                if (found_position) {
                    continue;
                }
            }
            player.pos.xyz = fake_pos;
        }
    }
}

void main() {
    uint entity_idx = gl_WorkGroupSize.x * gl_WorkGroupID.x + gl_LocalInvocationID.x;
    if (entity_idx >= sim_data.player_count) return;
    Player player = players[entity_idx];

    //volume effects
    vec3 start_pos =
        player.pos.xyz + player.size * PLAYER_HITBOX_OFFSET - player.size * PLAYER_HITBOX_SIZE / 2.0;
    vec3 end_pos =
        player.pos.xyz + player.size * PLAYER_HITBOX_OFFSET + player.size * PLAYER_HITBOX_SIZE / 2.0;
    uvec3 start_voxel_pos = uvec3(start_pos);
    uvec3 iter_counts = uvec3(end_pos) - start_voxel_pos + uvec3(1);
    float nearby_density = 0.0;
    vec3 directional_density = vec3(0.0);
    for (int x = 0; x < iter_counts.x; x++) {
        for (int y = 0; y < iter_counts.y; y++) {
            for (int z = 0; z < iter_counts.z; z++) {
                uvec3 voxel_pos = start_voxel_pos + uvec3(x, y, z);
                vec3 overlapping_dimensions = min(end_pos, vec3(voxel_pos) + vec3(1.0)) - max(start_pos, vec3(voxel_pos));
                float overlapping_volume =
                    overlapping_dimensions.x * overlapping_dimensions.y * overlapping_dimensions.z;
                uint voxel = get_data(voxel_pos);
                float density = physics_properties[voxel >> 24].density;
                nearby_density += overlapping_volume * density;
                directional_density += overlapping_volume
                        * density
                        * ((vec3(voxel_pos) + vec3(0.5))
                            - (player.pos.xyz + player.size * PLAYER_HITBOX_OFFSET))
                        / player.size;
            }
        }
    }
    nearby_density /=
        player.size * player.size * player.size * PLAYER_HITBOX_SIZE.x * PLAYER_HITBOX_SIZE.y * PLAYER_HITBOX_SIZE.z;
    directional_density /=
        player.size * player.size * player.size * PLAYER_HITBOX_SIZE.x * PLAYER_HITBOX_SIZE.y * PLAYER_HITBOX_SIZE.z;

    if (player.has_world_collision == 1) {
        player.vel.xyz += (PLAYER_DENSITY - nearby_density)
                * player.gravity
                * 11.428571428571429
                * sim_data.dt;
        if (length(directional_density) * sim_data.dt > 0.001) {
            player.vel.xyz -= 0.5 * directional_density * sim_data.dt;
        }
        if (dot(player.vel.xyz, player.vel.xyz) > 0.0) {
            player.vel.xyz -= nearby_density * 0.0375 * player.vel.xyz * length(player.vel.xyz) * sim_data.dt
                    + 0.2 * normalize(player.vel.xyz) * sim_data.dt;
        }
        vec3 prev_collision_vec = player.collision_vec.xyz;
        player.collision_vec = ivec3(0);
        collide_player(
            player,
            prev_collision_vec
        );
    } else {
        player.pos += player.vel;
        player.vel *= 0.5;
    }

    players[entity_idx] = player;
}
