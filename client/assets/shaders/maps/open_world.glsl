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
    uint player_count;
    uint worldgen_count;
    int unload_index;
    uint unload_component;
    uint voxel_write_count;
    uint worldgen_seed;
} sim_data;

void set_data_in_chunk(uvec3 global_pos, uint chunk_idx, uint data) {
    uvec3 pos_in_chunk = global_pos & POS_IN_CHUNK_MASK;
    uint idx_in_chunk = pos_in_chunk.x * CHUNK_SIZE * CHUNK_SIZE + pos_in_chunk.y * CHUNK_SIZE + pos_in_chunk.z;
    voxels[chunk_idx * CHUNK_VOLUME + idx_in_chunk] = data;
}

const float SEA_LEVEL = 1796.0;
const vec3 MACRO_SCALE = vec3(0.01, 0.00001, 0.01);
const vec3 DENSITY_SCALE = vec3(0.04, 0.0, 0.04);
const vec3 TEMP_SCALE = vec3(0.005, 0.0005, 0.005);
const vec3 CAVE_SCALE = vec3(0.06, 0.035, 0.06);

vec4 y_gradient(float y, float scale, float start_y, float end_y) {
    return vec4(0.0, scale * (step(start_y, scale * y) - step(end_y, scale * y)), 0.0, clamp(scale * y, start_y, end_y));
}

vec4 clamp_gradient(vec4 gradient, float min_val, float max_val) {
    return vec4(gradient.xyz * (step(min_val, gradient.w) - step(max_val, gradient.w)), clamp(gradient.w, min_val, max_val));
}

vec4 abs_gradient(vec4 gradient) {
    return vec4(gradient.xyz * sign(gradient.w), abs(gradient.w));
}

vec4 max_gradient(vec4 a, vec4 b) {
    if (a.w > b.w) {
        return a;
    } else {
        return b;
    }
}

vec4 mul_gradient(vec4 a, vec4 b) {
    return vec4(a.xyz * b.w + a.w * b.xyz, a.w * b.w);
}

uint get_worldgen(uvec3 global_pos) {
    uvec2 chunk_pos = global_pos.xz / 32;
    vec3 tree_pos = 32.0 * vec3(chunk_pos, 0.0) + 3.0 + 26.0 * vec3(pcg3d(chunk_pos.xxy).xyz & 0xFF) / 256.0;
    tree_pos.z = tree_pos.z / 100.0 + 0.2;
    vec3 random = vec3(pcg3d(global_pos) & 0xFF) / 128.0;
    vec3 true_pos = vec3(global_pos);
    vec4 macro_noise = vec4(MACRO_SCALE, 1.0) * voronoise(MACRO_SCALE * true_pos, 1.0, 1.0);
    vec4 density = vec4(DENSITY_SCALE, 1.0) * voronoise(DENSITY_SCALE * true_pos, 1.0, 1.0);
    vec4 temperature = vec4(vec3(0.0), 0.2) + vec4(TEMP_SCALE, 1.0) * voronoise(TEMP_SCALE * true_pos, 1.0, 1.0)
            - y_gradient(true_pos.y - SEA_LEVEL, 1.0 / 80.0, -0.5, 2.0);
    vec4 cave_density = mul_gradient(
            clamp_gradient(vec4(CAVE_SCALE, 1.0) * voronoise(CAVE_SCALE * true_pos, 1.0, 1.0), -1.0, 0.5),
            -1.0 * y_gradient(true_pos.y - SEA_LEVEL, 1.0 / 80.0, -1.0, 0.3));

    vec4 terrain_density =
        1.5 * cave_density
            + density
            + 0.7 * macro_noise
            + vec4(vec3(0.0), 0.1)
            - y_gradient(true_pos.y - SEA_LEVEL, 1.0 / 62.0, -0.6, 10.0)
            + mul_gradient(max_gradient(vec4(vec3(0.0), 1.0) - 20.0 * abs_gradient(temperature - vec4(vec3(0.0), 0.1)), vec4(0.0)), -y_gradient(true_pos.y - SEA_LEVEL - 5.0, 0.3, -2.0, 0.0));
    if (terrain_density.w <= 0.0) {
        if (temperature.w + tree_pos.z > 0.2 && length(true_pos.xz - tree_pos.xy) < 0.7 + 0.2 / (temperature.w + tree_pos.z - 0.1) - 0.25 / (terrain_density.w - 0.2)) {
            return MAT_WOOD << 24;
        } else if (length(vec3(true_pos.xz - tree_pos.xy, 100.0 * (temperature.w + tree_pos.z - 0.2))) < 12.0 - 5.0 * random.x) {
            return MAT_LEAF << 24;
        } else if (true_pos[1] > SEA_LEVEL) {
            if (temperature.w - terrain_density.w < -0.9) {
                return MAT_ICE << 24;
            }
            return MAT_AIR << 24;
        } else if (temperature.w + 0.02 * random.x > 0.1) {
            return MAT_WATER << 24;
        } else if (terrain_density.w + clamp((true_pos.y - SEA_LEVEL) / 30.0, -2.0, 0.0) < -0.27 && temperature.w < 0.1) {
            return MAT_AIR << 24;
        } else {
            return MAT_ICE << 24;
        }
    } else if (terrain_density.w + clamp((SEA_LEVEL - true_pos.y) / 30.0, 0.0, 0.7) > 0.35) {
        return MAT_STONE << 24;
    } else if (terrain_density.w > 0.2) {
        return MAT_DIRT << 24;
    } else {
        if (temperature.w + 0.05 * random.x - 1.5 * terrain_density.y > terrain_density.w && true_pos[1] >= SEA_LEVEL && terrain_density.y < 0.01) {
            return MAT_GRASS << 24;
        } else {
            return MAT_DIRT << 24;
        }
    }
    return MAT_AIR << 24;
}

void main() {
    uvec3 pos = gl_WorkGroupSize * chunk_loads[gl_WorkGroupID.x].xyz + gl_LocalInvocationID;
    uint data = get_worldgen(pos);
    int chunk_idx = chunk_loads[gl_WorkGroupID.x].w;
    set_data_in_chunk(pos, abs(chunk_idx), data);
    if (data >> 24 != MAT_AIR || chunk_idx < 0) {
        load_results[gl_WorkGroupID.x / 8] = abs(chunk_idx);
    }
}
