#version 450
#include <common.glsl>

layout(local_size_x = 256) in;

layout(set = 0, binding = 0, r32ui) uniform uimage3D chunks;
layout(set = 0, binding = 1) buffer ChunkLoads {
    ivec4 chunk_loads[];
};

layout(set = 0, binding = 2) uniform SimData {
    uvec3 render_size;
    uvec3 start_pos;
    uint count;
} sim_data;

layout(set = 0, binding = 3) buffer ChunkLoadResults {
    uint load_results[];
};

void main() {
    if (gl_GlobalInvocationID.x >= sim_data.count) {
        return;
    }
    uvec3 pos = 8 * chunk_loads[8 * gl_GlobalInvocationID.x].xyz;
    uvec4 indicies = get_indicies(pos, sim_data.render_size);
    imageStore(chunks, ivec3(indicies.xyz), uvec4(load_results[gl_GlobalInvocationID.x]));
}
