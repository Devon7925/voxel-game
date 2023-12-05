#version 450
#include <common.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(set = 0, binding = 0) buffer VoxelBuffer { uint voxels[]; };
layout(set = 0, binding = 1) buffer ChunkUpdates { ivec4 chunk_updates[]; };

layout(set = 0, binding = 2) uniform SimData {
    uvec3 render_size;
    uvec3 start_pos;
} sim_data;

layout(set = 0, binding = 3) buffer Projectiles { Projectile projectiles[]; };

uint get_data_unchecked(uvec3 global_pos) {
    uint index = get_index(global_pos, sim_data.render_size);
    return voxels[index];
}

uint get_data(uvec3 global_pos) {
    uvec3 start_offset = CHUNK_SIZE * sim_data.start_pos;
    if (any(lessThan(global_pos, start_offset))) return MAT_OOB << 24;
    uvec3 rel_pos = global_pos - start_offset;
    if (any(greaterThanEqual(rel_pos, CHUNK_SIZE*sim_data.render_size))) return MAT_OOB << 24;
    return get_data_unchecked(global_pos);
}

void set_data(uvec3 global_pos, uint data) {
    uint index = get_index(global_pos, sim_data.render_size);
    if(voxels[index] != data) {
        chunk_updates[gl_WorkGroupID.x].w = 1;
    }
    voxels[index] = data;
}

void main() {
    ivec3 pos = ivec3(gl_WorkGroupSize)*chunk_updates[gl_WorkGroupID.x].xyz + ivec3(gl_LocalInvocationID); 
    uint pos_data = get_data(pos);
    uint voxel_mat = pos_data >> 24;
    uint voxel_data = pos_data & 0xFFFFFF;

    if (voxel_mat != MAT_AIR) {
        if (voxel_data >= material_damage_threshhold[voxel_mat]) {
            set_data(pos, MAT_AIR << 24);
            return;
        }
        set_data(pos, pos_data);
        return;
    }

    uint voxel_dist = 0;
    for (uint i = 0; i < 8; i++) {
        voxel_dist = voxel_dist << 3;
        ivec3 d = ivec3(i%2, (i / 2) % 2, i / 4) * ivec3(2) - ivec3(1);
        uint direction_dist = 7;
        for (uint j = 1; j < 8; j++) {
            ivec3 dir = d * ivec3(j % 2, (j / 2) % 2, j / 4);
            uint dir_data = get_data(pos + dir);
            if (dir_data >> 24 == MAT_OOB) {
                direction_dist = min(direction_dist, ((voxel_data >> (3*(7 - i))) & 0x7));
                continue;
            } else if (dir_data >> 24 != MAT_AIR) {
                direction_dist = 0;
                break;
            }
            direction_dist = min(direction_dist, ((dir_data >> (3*(7 - i))) & 0x7) + 1);
        }
        voxel_dist |= direction_dist & 0x7;
    }
    voxel_data = voxel_dist;
    set_data(pos, voxel_data);
}