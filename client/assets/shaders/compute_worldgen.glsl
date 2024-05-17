#version 450
#include <common.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(set = 0, binding = 0) buffer ChunkBuffer { uint chunks[]; };
layout(set = 0, binding = 1) buffer VoxelBuffer { uint voxels[]; };
layout(set = 0, binding = 2) buffer ChunkLoads { ivec4 chunk_loads[]; };

layout(set = 0, binding = 3) uniform SimData {
    uvec3 render_size;
    uvec3 start_pos;
} sim_data;

layout(set = 0, binding = 4) buffer Projectiles { Projectile projectiles[]; };

uint get_index(uvec3 global_pos) {
    uvec2 indicies = get_indicies(global_pos, sim_data.render_size);
    return chunks[indicies.x] * CHUNK_VOLUME + indicies.y;
}

uint get_data_unchecked(uvec3 global_pos) {
    return voxels[get_index(global_pos)];
}

uint get_data(uvec3 global_pos) {
    uvec3 start_offset = CHUNK_SIZE * sim_data.start_pos;
    if (any(lessThan(global_pos, start_offset))) return MAT_OOB << 24;
    uvec3 rel_pos = global_pos - start_offset;
    if (any(greaterThanEqual(rel_pos, CHUNK_SIZE*sim_data.render_size))) return MAT_OOB << 24;
    return get_data_unchecked(global_pos);
}

void set_data(uvec3 global_pos, uint data) {
    voxels[get_index(global_pos)] = data;
}

uint get_worldgen(uvec3 global_pos) {
    vec3 true_pos = vec3(global_pos);
    float macro_noise = voronoise(0.005 * true_pos, 1.0, 1.0).w;
    float density = voronoise(0.04 * true_pos, 1.0, 1.0).w;
    float terrain_density = density + macro_noise - (true_pos.y - 1800.0) / 15.0;

    float cave_density = voronoise(vec3(0.06, 0.035, 0.06) * true_pos, 1.0, 1.0).w;
    float pillar_density = cave_density - (true_pos.y - 1800.0) / 80.0;
    if(pillar_density > 0.2) {
        return MAT_STONE << 24;
    } else if(cave_density+density+0.5*macro_noise < -0.4) {
        return MAT_AIR << 24;
    } else if(terrain_density > 0.3) {
        return MAT_STONE << 24;
    } else if(terrain_density > 0.1) {
        return MAT_DIRT << 24;
    } else if(terrain_density > 0.0) {
        return MAT_GRASS << 24;
    } else if(true_pos[1] <= 1796.0) {
        return MAT_ICE << 24;
    }
    return MAT_AIR << 24;
}

void main() {
    uvec3 pos = gl_WorkGroupSize*chunk_loads[gl_WorkGroupID.x].xyz + gl_LocalInvocationID;
    uvec2 indicies = get_indicies(pos, sim_data.render_size);
    chunks[indicies.x] = chunk_loads[gl_WorkGroupID.x].w;
    set_data(pos, get_worldgen(pos));
}