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

layout(push_constant) uniform SimData {
    uvec3 render_size;
    uvec3 start_pos;
    uvec3 voxel_update_offset;
    float dt;
    uint projectile_count;
    uint worldgen_count;
    int unload_index;
    uint unload_component;
    uint voxel_write_count;
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

vec3 RayBoxDist(vec3 pos, vec3 ray, vec3 vmin, vec3 vmax) {
    vec3 normMinDiff = (vmin - pos) / ray;
    vec3 normMaxDiff = (vmax - pos) / ray;
    return max(normMinDiff, normMaxDiff);
}

void main() {
    uint projectile_idx = gl_WorkGroupSize.x * gl_WorkGroupID.x + gl_LocalInvocationID.x;
    if (projectile_idx >= sim_data.projectile_count) return;
    Projectile projectile = projectiles[projectile_idx];

    ivec3 grid_iteration_count = ivec3(ceil(2.0 * projectile.size * sqrt(2.0)));
    grid_iteration_count.z = int(ceil((2.0 * projectile.size.z + sim_data.dt * projectile.vel) * sqrt(2.0)));
    vec3 grid_dist = 2.0 * projectile.size.xyz / grid_iteration_count;
    grid_dist.z = (2.0 * projectile.size.z + sim_data.dt * projectile.vel) / grid_iteration_count.z;
    vec3 dir = quat_transform(projectile.dir, vec3(0.0, 0.0, 1.0));
    vec3 right = quat_transform(projectile.dir, vec3(1.0, 0.0, 0.0));
    vec3 up = quat_transform(projectile.dir, vec3(0.0, 1.0, 0.0));
    vec3 start = projectile.pos.xyz - dir * projectile.size.z - right * projectile.size.x - up * projectile.size.y;
    for (int i = 0; i <= grid_iteration_count.x; i++) {
        for (int j = 0; j <= grid_iteration_count.y; j++) {
            for (int k = 0; k <= grid_iteration_count.z; k++) {
                vec3 pos = start + dir * grid_dist.z * k + right * grid_dist.x * i + up * grid_dist.y * j;
                ivec3 voxel_pos = ivec3(pos);
                uint data = get_data(voxel_pos);
                uint voxel_mat = data >> 24;
                if (voxel_mat == MAT_AIR || voxel_mat == MAT_AIR_OOB) {
                    continue;
                } else if (voxel_mat == MAT_OOB) {
                    projectile.health = 0.0;
                    projectile.chunk_update_pos.w = 0;
                    projectiles[projectile_idx] = projectile;
                    return;
                }

                float dist_past_bb = grid_dist.z * k - 2.0 * projectile.size.z;
                vec3 delta = RayBoxDist(pos, -dir, vec3(voxel_pos), vec3(voxel_pos + ivec3(1)));
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
