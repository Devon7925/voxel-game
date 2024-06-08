#version 450
#include <common.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(set = 0, binding = 1) buffer VoxelBuffer {
    uint voxels[];
};
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
    uint worldgen_count;
    int unload_index;
    uint unload_component;
    uint voxel_write_count;
} sim_data;

void set_data_in_chunk(uvec3 global_pos, uint chunk_idx, uint data) {
    uvec3 pos_in_chunk = global_pos & POS_IN_CHUNK_MASK;
    uint idx_in_chunk = pos_in_chunk.x * CHUNK_SIZE * CHUNK_SIZE + pos_in_chunk.y * CHUNK_SIZE + pos_in_chunk.z;
    voxels[chunk_idx * CHUNK_VOLUME + idx_in_chunk] = data;
}

uint get_worldgen(uvec3 global_pos) {
    vec3 random = vec3(pcg3d(global_pos) & 0xFF) / 128.0;
    vec3 true_pos = vec3(global_pos);
    float macro_noise = voronoise(0.01 * true_pos, 1.0, 1.0).w;
    float density = voronoise(0.04 * true_pos, 1.0, 1.0).w;
    float temperature = voronoise(vec3(0.015, 0.005, 0.015) * true_pos, 1.0, 1.0).w - clamp((true_pos.y - 1796.0) / 50.0, 0.0, 2.0);
    float cave_density = clamp(voronoise(vec3(0.06, 0.035, 0.06) * true_pos, 1.0, 1.0).w, -1.0, 0.5);

    float terrain_density =
        1.5 * cave_density
            + density
            + 0.7 * macro_noise
            - clamp((true_pos.y - 1800.0) / 17.0, -0.6, 10.0)
            + max(1.0 - 4.0 * abs(temperature - 0.1), 0.0) * clamp(0.3 * (1800.0 - true_pos.y), 0.0, 1.0);
    if (terrain_density <= 0.0) {
        if (true_pos[1] > 1796.0) {
            return MAT_AIR << 24;
        } else if (temperature + 0.075 * random.x > 0.1) {
            return MAT_WATER << 24;
        } else if (terrain_density + clamp((true_pos.y - 1796.0) / 30.0, -2.0, 0.0) < -0.27) {
            return MAT_AIR << 24;
        } else {
            return MAT_ICE << 24;
        }
    } else if (terrain_density + clamp((1796.0 - true_pos.y) / 30.0, 0.0, 0.7) > 0.35) {
        return MAT_STONE << 24;
    } else if (terrain_density > 0.2) {
        return MAT_DIRT << 24;
    } else {
        if (temperature + 0.05 * random.x > terrain_density && true_pos[1] >= 1796.0) {
            return MAT_GRASS << 24;
        } else {
            return MAT_DIRT << 24;
        }
    }
    return MAT_AIR << 24;
}

void main() {
    uvec3 pos = gl_WorkGroupSize * chunk_loads[gl_WorkGroupID.x].xyz + gl_LocalInvocationID;
    uvec4 indicies = get_indicies(pos, sim_data.render_size);
    uint data = get_worldgen(pos);
    int chunk_idx = chunk_loads[gl_WorkGroupID.x].w;
    set_data_in_chunk(pos, abs(chunk_idx), data);
    if (data >> 24 != MAT_AIR || chunk_idx < 0) {
        load_results[gl_WorkGroupID.x / 8] = abs(chunk_idx);
    }
}
