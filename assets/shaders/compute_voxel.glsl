#version 450
#include <common.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(set = 0, binding = 0) buffer VoxelBuffer { uint voxels[]; };
layout(set = 0, binding = 1) buffer ChunkUpdates { ivec4 chunk_updates[]; };

layout(set = 0, binding = 2) uniform SimData {
    uvec3 render_size;
    uvec3 start_pos;
    uvec3 update_offset;
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
    ivec3 pos = 2*ivec3(gl_WorkGroupSize)*chunk_updates[gl_WorkGroupID.x].xyz + 2*ivec3(gl_LocalInvocationID) + ivec3(sim_data.update_offset); 
    uvec2 pos_data[2][2][2];
    for (uint i = 0; i < 2; i++) {
        for (uint j = 0; j < 2; j++) {
            for (uint k = 0; k < 2; k++) {
                uint raw_voxel = get_data(pos + ivec3(i, j, k));
                pos_data[i][j][k] = uvec2(raw_voxel >> 24, raw_voxel & 0xFFFFFF);
                if (pos_data[i][j][k].x == MAT_OOB) {
                    return;
                }
            }
        }
    }

    for (uint i = 0; i < 2; i++) {
        for (uint j = 0; j < 2; j++) {
            for (uint k = 0; k < 2; k++) {
                if (pos_data[i][j][k].x != MAT_AIR) {
                    if (pos_data[i][j][k].y >= material_damage_threshhold[pos_data[i][j][k].x]) {
                        pos_data[i][j][k] = uvec2(MAT_AIR, 0);
                    }
                }
            }
        }
    }

    for (uint i = 0; i < 2; i++) {
        for (uint j = 0; j < 2; j++) {
            for (uint k = 0; k < 2; k++) {
                if (pos_data[i][j][k].x == MAT_AIR) {
                    int offset = 3*int(((k << 2) | (j << 1) | i));
                    ivec3 d = ivec3(1) - ivec3(i, j, k) * ivec3(2);
                    uint direction_dist = 7;
                    for (uint m = 1; m < 8; m++) {
                        ivec3 dir = d * ivec3(m % 2, (m / 2) % 2, m / 4) + ivec3(i, j, k);
                        if (pos_data[dir.x][dir.y][dir.z].x == MAT_OOB) {
                            direction_dist = min(direction_dist, bitfieldExtract(pos_data[i][j][k].y, offset, 3));
                            continue;
                        } else if (pos_data[dir.x][dir.y][dir.z].x != MAT_AIR) {
                            direction_dist = 0;
                            break;
                        }
                        direction_dist = min(direction_dist, bitfieldExtract(pos_data[dir.x][dir.y][dir.z].y, offset, 3) + 1);
                    }
                    pos_data[i][j][k].y = bitfieldInsert(pos_data[i][j][k].y, direction_dist, offset, 3);
                }
            }
        }
    }

    for (uint i = 0; i < 2; i++) {
        for (uint j = 0; j < 2; j++) {
            for (uint k = 0; k < 2; k++) {
                set_data(pos + ivec3(i, j, k), (pos_data[i][j][k].x << 24) | pos_data[i][j][k].y);
            }
        }
    }
}