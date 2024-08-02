#version 450
#include <common.glsl>

layout(local_size_x = 256) in;

layout(set = 0, binding = 0, r32ui) uniform uimage3D chunks;
layout(set = 0, binding = 3) buffer ChunkLoads {
    ivec4 chunk_loads[];
};

layout(set = 0, binding = 4) buffer ChunkLoadResults {
    uint load_results[];
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

void main() {
    if (gl_GlobalInvocationID.x >= sim_data.worldgen_count) {
        return;
    }
    uvec3 pos = 8 * chunk_loads[8 * gl_GlobalInvocationID.x].xyz;
    uvec4 indicies = get_indicies(pos, sim_data.render_size);
    imageStore(chunks, ivec3(indicies.xyz), uvec4(load_results[gl_GlobalInvocationID.x]));
}
