#version 450
#include <common.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(set = 0, binding = 0) buffer VoxelBufferIn { uvec2 voxel_in[]; };
layout(set = 0, binding = 1) buffer VoxelBufferout { uvec2 voxel_out[]; };
layout(set = 0, binding = 2) buffer ChunkUpdates { ivec4 chunk_updates[]; };

layout(set = 0, binding = 3) uniform SimData {
    uint max_dist;
    uvec3 render_size;
    ivec3 start_pos;
} sim_data;

uvec2 get_data_unchecked(ivec3 global_pos) {
    uint index = get_index(global_pos, sim_data.render_size);
    return voxel_in[index];
}

uvec2 get_data(ivec3 global_pos) {
    ivec3 rel_pos = global_pos - int(CHUNK_SIZE) * sim_data.start_pos;
    if (
        any(lessThan(rel_pos, ivec3(0)))
        ||
        any(greaterThanEqual(rel_pos, int(CHUNK_SIZE)*ivec3(sim_data.render_size)))
    ) return uvec2(2, 0);
    return get_data_unchecked(global_pos);
}

void set_data(ivec3 global_pos, uvec2 data) {
    uint index = get_index(global_pos, sim_data.render_size);
    voxel_out[index] = data;
}

void main() {
    ivec3 pos = ivec3(gl_WorkGroupSize)*chunk_updates[gl_WorkGroupID.x].xyz + ivec3(gl_LocalInvocationID); 
    uvec2 pos_data = get_data(pos);

    if (pos_data.x != 0) {
        set_data(pos, pos_data);
        return;
    }

    uint voxel_dist = 0;
    for (uint i = 0; i < 8; i++) {
        voxel_dist = voxel_dist << 4;
        ivec3 d = ivec3(i%2, (i / 2) % 2, i / 4) * ivec3(2) - ivec3(1);
        uint direction_dist = sim_data.max_dist - 1;
        for (uint j = 1; j < 8; j++) {
            ivec3 dir = d * ivec3(j % 2, (j / 2) % 2, j / 4);
            uvec2 dir_data = get_data(pos + dir);
            if (dir_data.x == 1) {
                direction_dist = 0;
                break;
            } else if (dir_data.x == 2) {
                direction_dist = min(direction_dist, (pos_data.y >> (4*(7 - i))) & 0xF - 1);
                continue;
            }
            direction_dist = min(direction_dist, (dir_data.y >> (4*(7 - i))) & 0xF);
        }
        voxel_dist |= direction_dist + 1;
    }
    pos_data.y = voxel_dist;
    set_data(pos, pos_data);
}