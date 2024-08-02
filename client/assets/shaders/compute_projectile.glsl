#version 450
#include <common.glsl>

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, r32ui) uniform uimage3D chunks;
layout(set = 0, binding = 1) buffer VoxelBuffer {
    uint voxels[];
};
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

uint get_index(uvec3 global_pos) {
    uvec4 indicies = get_indicies(global_pos, sim_data.render_size);
    return imageLoad(chunks, ivec3(indicies.xyz)).x * CHUNK_VOLUME + indicies.w;
}

uint get_data_unchecked(uvec3 global_pos) {
    return voxels[get_index(global_pos)];
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
    uint index = get_index(global_pos);
    voxels[index] = data;
}

vec3 ray_box_dist(vec3 pos, vec3 ray, vec3 vmin, vec3 vmax) {
    vec3 normMinDiff = (vmin - pos) / ray;
    vec3 normMaxDiff = (vmax - pos) / ray;
    return max(normMinDiff, normMaxDiff);
}

bool los_check(vec3 start, vec3 end) {
    vec3 ray = normalize(end - start);
    float max_dist = length(end - start);

    uint offset = 0;
    if (ray.x < 0) offset += 1;
    if (ray.y < 0) offset += 2;
    if (ray.z < 0) offset += 4;
    float depth = 0;

    vec3 pos = start;

    for (uint i = 0; i < 20; i++) {
        vec3 floor_pos = floor(pos);
        uint voxel_data = get_data(uvec3(floor_pos));
        vec3 v_min;
        vec3 v_max;
        uint voxel_material = voxel_data >> 24;
        if (voxel_material == MAT_AIR_OOB) {
            v_min = floor(pos / CHUNK_SIZE) * CHUNK_SIZE;
            v_max = v_min + vec3(CHUNK_SIZE);
        } else if (physics_properties[voxel_material].is_fluid) {
            uint dist = 0;
            if (physics_properties[voxel_material].is_data_standard_distance) {
                dist = get_dist(voxel_data, offset);
            }
            v_min = floor_pos - vec3(dist);
            v_max = floor_pos + vec3(dist + 1);
        } else {
            return false;
        }
        vec3 delta = ray_box_dist(pos, ray, v_min, v_max);
        float dist_diff = min(delta.x, min(delta.y, delta.z));
        depth += dist_diff;
        if (depth > max_dist) {
            return true;
        }
        pos += ray * dist_diff;
        if (delta.x < delta.y && delta.x < delta.z) {
            if (ray.x > 0 && pos.x < v_max.x) {
                pos.x = v_max.x;
            } else if (ray.x < 0 && pos.x >= v_min.x) {
                pos.x = v_min.x - 0.001;
            }
        } else if (delta.y < delta.z) {
            if (ray.y > 0 && pos.y < v_max.y) {
                pos.y = v_max.y;
            } else if (ray.y < 0 && pos.y >= v_min.y) {
                pos.y = v_min.y - 0.001;
            }
        } else {
            if (ray.z > 0 && pos.z < v_max.z) {
                pos.z = v_max.z;
            } else if (ray.z < 0 && pos.z >= v_min.z) {
                pos.z = v_min.z - 0.001;
            }
        }
    }

    return false;
}

shared uint collision_count;

void main() {
    uint projectile_idx = gl_WorkGroupSize.x * gl_WorkGroupID.x + gl_LocalInvocationID.x;
    if (projectile_idx >= sim_data.projectile_count) return;
    Projectile projectile = projectiles[projectile_idx];

    vec3 dir = quat_transform(projectile.dir, vec3(0.0, 0.0, 1.0));
    vec3 right = quat_transform(projectile.dir, vec3(1.0, 0.0, 0.0));
    vec3 up = quat_transform(projectile.dir, vec3(0.0, 1.0, 0.0));

    vec3 projectile_vectors[] = { right, up, dir };

    for (int i = 0; i < sim_data.player_count; i++) {
        Player player = players[i];
        uint collision = 0;
        vec3 prev_collision = vec3(0.0);
        for (int hitsphere_idx = 0; hitsphere_idx < HITSPHERES.length(); hitsphere_idx++) {
            Hitsphere transformed_hit_sphere = Hitsphere(player.pos.xyz + HITSPHERES[hitsphere_idx].offset * player.size.x, HITSPHERES[hitsphere_idx].radius * player.size.x, HITSPHERES[hitsphere_idx].headshot);
            if (collision > 0) {
                if (length(projectile.pos.xyz - transformed_hit_sphere.offset) > length(projectile.pos.xyz - prev_collision)) {
                    continue;
                }
            }
            bool collide = true;
            for (int d = 0; d < 3; d++) {
                if (abs(dot(transformed_hit_sphere.offset, projectile_vectors[d]) - dot(projectile.pos.xyz, projectile_vectors[d])) > projectile.size.xyz[d] + transformed_hit_sphere.radius) {
                    collide = false;
                    break;
                }
            }
            if (!collide) {
                continue;
            }
            if (transformed_hit_sphere.headshot) {
                collision = 2;
            } else {
                collision = 1;
            }
            prev_collision = transformed_hit_sphere.offset;
        }
        if (collision > 0 && los_check(projectile.pos.xyz, prev_collision)) {
            uint collision_idx = atomicAdd(collision_count, 1);
            collisions[collision_idx] = Collision(projectile_idx, i, collision);
        }
    }

    if (projectile.should_collide_with_terrain != 1) {
        return;
    }

    ivec3 grid_iteration_count = ivec3(ceil(2.0 * projectile.size * sqrt(2.0)));
    grid_iteration_count.z = int(ceil((2.0 * projectile.size.z + sim_data.dt * projectile.vel) * sqrt(2.0)));
    vec3 grid_dist = 2.0 * projectile.size.xyz / grid_iteration_count;
    grid_dist.z = (2.0 * projectile.size.z + sim_data.dt * projectile.vel) / grid_iteration_count.z;
    vec3 start = projectile.pos.xyz - dir * projectile.size.z - right * projectile.size.x - up * projectile.size.y;
    for (int i = 0; i <= grid_iteration_count.x; i++) {
        for (int j = 0; j <= grid_iteration_count.y; j++) {
            for (int k = 0; k <= grid_iteration_count.z; k++) {
                int mapped_i = grid_iteration_count.x / 2 + (2 * (i % 2) - 1) * ((i + 1) / 2);
                int mapped_j = grid_iteration_count.y / 2 + (2 * (j % 2) - 1) * ((j + 1) / 2);
                int mapped_k = grid_iteration_count.z / 2 + (2 * (k % 2) - 1) * ((k + 1) / 2);
                vec3 pos = start + dir * grid_dist.z * mapped_k + right * grid_dist.x * mapped_i + up * grid_dist.y * mapped_j;
                ivec3 voxel_pos = ivec3(pos);
                uint data = get_data(voxel_pos);
                uint voxel_mat = data >> 24;
                if (physics_properties[voxel_mat].is_fluid) {
                    continue;
                } else if (voxel_mat == MAT_OOB) {
                    projectile.health = 0.0;
                    projectile.chunk_update_pos.w = 0;
                    projectiles[projectile_idx] = projectile;
                    return;
                }

                float dist_past_bb = grid_dist.z * mapped_k - 2.0 * projectile.size.z;
                vec3 delta = ray_box_dist(pos, -dir, vec3(voxel_pos), vec3(voxel_pos + ivec3(1)));
                float dist_diff = min(delta.x, min(delta.y, delta.z));
                if (dist_past_bb > 0.0) {
                    projectile.pos += vec4(dir * (dist_past_bb - dist_diff), 0.0);
                }

                if (projectile.wall_bounce == 1) {
                    vec3 new_dir = dir;
                    if (delta.x == dist_diff) {
                        new_dir.x *= -1.0;
                    } else if (delta.y == dist_diff) {
                        new_dir.y *= -1.0;
                    } else if (delta.z == dist_diff) {
                        new_dir.z *= -1.0;
                    }
                    projectile.dir = quaternion_from_arc(vec3(0.0, 0.0, 1.0), new_dir);
                    projectiles[projectile_idx] = projectile;
                    return;
                }

                if (projectile.damage > 0) {
                    uint vox_index = get_index(voxel_pos);
                    atomicAdd(voxels[vox_index], int(projectile.damage));
                }

                projectile.chunk_update_pos = ivec4(voxel_pos, 1);

                projectile.health = 0.0;
                projectiles[projectile_idx] = projectile;
                return;
            }
        }
    }
}
