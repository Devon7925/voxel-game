#version 450
#include <common.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(set = 0, binding = 0) buffer VoxelBufferA { uvec2 voxel_a[]; };
layout(set = 0, binding = 1) buffer VoxelBufferB { uvec2 voxel_b[]; };
layout(set = 0, binding = 2) buffer ChunkUpdates { ivec3 chunk_updates[]; };

layout(set = 0, binding = 3) uniform SimData {
    uint max_dist;
    uvec3 grid_size;
    ivec3 start_pos;
    bool is_a_in_buffer;
} sim_data;

uvec2 get_data(ivec3 global_pos) {
    vec3 rel_pos = global_pos - sim_data.start_pos;
    if (any(lessThan(rel_pos, vec3(0))) || any(greaterThanEqual(rel_pos, vec3(sim_data.grid_size)))) return uvec2(1, 0);
    uint index = get_index(global_pos, sim_data.grid_size);
    if (sim_data.is_a_in_buffer) {
        return voxel_a[index];
    } else {
        return voxel_b[index];
    }
}

void set_data(ivec3 global_pos, uvec2 data) {
    uint index = get_index(global_pos, sim_data.grid_size);
    if (sim_data.is_a_in_buffer) {
        voxel_b[index] = data;
    } else {
        voxel_a[index] = data;
    }
}

void main() {
    ivec3 pos = ivec3(gl_WorkGroupSize)*chunk_updates[gl_WorkGroupID.x] + ivec3(gl_LocalInvocationID); 
    uvec2 pos_data = get_data(pos);

    if (pos_data.x != 0) {
        return;
    }

    uint voxel_dist = 0;
    for (int i = 0; i < 8; i++) {
        voxel_dist = voxel_dist << 4;
        ivec3 d = ivec3(i%2, (i / 2) % 2, i / 4) * ivec3(2) - ivec3(1);
        uint direction_dist = sim_data.max_dist - 1;
        for (uint j = 1; j < 8; j++) {
            ivec3 dir = d * ivec3(j % 2, (j / 2) % 2, j / 4);
            uvec2 dir_data = get_data(pos + dir);
            if (dir_data.x != 0) {
                direction_dist = 0;
                break;
            }
            direction_dist = min(direction_dist, (dir_data.y >> (4*(7 - i))) & 0xF);
        }
        voxel_dist |= direction_dist + 1;
    }
    pos_data.y = voxel_dist;
    set_data(pos, pos_data);
}