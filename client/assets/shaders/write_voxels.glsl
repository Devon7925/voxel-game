#version 450
#include <common.glsl>

layout(local_size_x = 256) in;

layout(set = 0, binding = 0, r32ui) uniform uimage3D chunks;
layout(set = 0, binding = 1) buffer VoxelBuffer {
    uint voxels[];
};
layout(set = 0, binding = 6) buffer VoxelWrites {
    uvec4 voxel_writes[];
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

void set_data(uvec3 global_pos, uint data) {
    uint index = get_index(global_pos);
    voxels[index] = data;
}

void main() {
    if (gl_GlobalInvocationID.x >= sim_data.voxel_write_count) {
        return;
    }
    set_data(voxel_writes[gl_GlobalInvocationID.x].xyz, voxel_writes[gl_GlobalInvocationID.x].w);
}
