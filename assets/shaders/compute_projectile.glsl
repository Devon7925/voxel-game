#version 450
#include <common.glsl>

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer VoxelBuffer { uvec2 voxels[]; };

layout(set = 0, binding = 1) buffer Projectiles { Projectile projectiles[]; };

layout(set = 0, binding = 2) uniform SimData {
    uint max_dist;
    uvec3 render_size;
    ivec3 start_pos;
    float dt;
} sim_data;

uvec2 get_data_unchecked(ivec3 global_pos) {
    uint index = get_index(global_pos, sim_data.render_size);
    return voxels[index];
}

uvec2 get_data(ivec3 global_pos) {
    ivec3 rel_pos = global_pos - int(CHUNK_SIZE) * sim_data.start_pos;
    if (
        any(lessThan(rel_pos, ivec3(0)))
        ||
        any(greaterThanEqual(rel_pos, ivec3(CHUNK_SIZE*sim_data.render_size)))
    ) return uvec2(2, 0);
    return get_data_unchecked(global_pos);
}

void set_data(ivec3 global_pos, uvec2 data) {
    ivec3 rel_pos = global_pos - int(CHUNK_SIZE) * sim_data.start_pos;
    if (
        any(lessThan(rel_pos, ivec3(0)))
        ||
        any(greaterThanEqual(rel_pos, ivec3(CHUNK_SIZE*sim_data.render_size)))
    ) return;
    uint index = get_index(global_pos, sim_data.render_size);
    voxels[index] = data;
}

void main() {
    uint projectile_idx = gl_WorkGroupSize.x * gl_WorkGroupID.x + gl_LocalInvocationID.x;
    Projectile projectile = projectiles[projectile_idx];

    ivec3 grid_iteration_count =  ivec3(ceil(2.0*projectile.size * sqrt(2.0)));
    grid_iteration_count.z = int(ceil((2.0*projectile.size.z + sim_data.dt * projectile.vel) * sqrt(2.0)));
    vec3 grid_dist = 2.0 * projectile.size.xyz / grid_iteration_count;
    grid_dist.z = (2.0 * projectile.size.z + sim_data.dt * projectile.vel) / grid_iteration_count.z;
    vec4 inverse_rot = quat_inverse(projectile.dir);
    vec3 dir = quat_transform(inverse_rot, vec3(0.0, 0.0, 1.0));
    vec3 right = quat_transform(inverse_rot, vec3(1.0, 0.0, 0.0));
    vec3 up = quat_transform(inverse_rot, vec3(0.0, 1.0, 0.0));
    vec3 start = projectile.pos.xyz - dir*projectile.size.z - right*projectile.size.x - up*projectile.size.y;
    for (int i = 0; i <= grid_iteration_count.x; i++) {
        for (int j = 0; j <= grid_iteration_count.y; j++) {
            for (int k = 0; k <= grid_iteration_count.z; k++) {
                vec3 pos = start + dir*grid_dist.z*k + right*grid_dist.x*i + up*grid_dist.y*j;
                ivec3 voxel_pos = ivec3(pos);
                uvec2 data = get_data(voxel_pos);
                if (data.x == 0 || data.x == 2) {
                    continue;
                }
                uint vox_index = get_index(voxel_pos, sim_data.render_size);
                atomicAdd(voxels[vox_index].y, 1);

                projectile.chunk_update_pos = ivec4(voxel_pos, 0);

                projectile.health = 0.0;
                projectiles[projectile_idx] = projectile;
                return;
            }
        }
    }
    
}