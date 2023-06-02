#version 450
#include <common.glsl>
layout(location = 0) in vec2 v_tex_coords;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) buffer VoxelBuffer { uvec2 voxels[]; };
layout(set = 0, binding = 1) uniform CamData {
    vec3 pos;
    vec3 dir;
    vec3 up;
    vec3 right;
} cam_data;

layout(set = 0, binding = 2) uniform SimData {
    uint max_dist;
    uvec3 grid_size;
    ivec3 start_pos;
    bool is_a_in_buffer;
} sim_data;

uvec2 get_data(ivec3 global_pos) {
    uint index = get_index(global_pos, sim_data.grid_size);
    return voxels[index];
}

uint get_dist(vec3 pos, uint offset) {
    vec3 rel_pos = pos - sim_data.start_pos;
    if (any(lessThan(rel_pos, vec3(0))) || any(greaterThanEqual(rel_pos, vec3(sim_data.grid_size)))) return 1;
    uvec2 voxel_data = get_data(ivec3(pos));
    if (voxel_data.x != 0) return 0;
    return (voxel_data.y >> (offset * 4)) & 0xF;
}

float RayBoxDist(vec3 pos, vec3 ray, vec3 vmin, vec3 vmax) {
    vec3 normMinDiff = (vmin - pos) / ray;
    vec3 normMaxDiff = (vmax - pos) / ray;
    vec3 maxDiff = max(normMinDiff, normMaxDiff);
    return min(maxDiff.x, min(maxDiff.y, maxDiff.z));
}

void main() {
    vec3 ray = normalize(cam_data.dir + v_tex_coords.x * cam_data.right + v_tex_coords.y * cam_data.up);
    
    uint offset = 0;
    if(ray.x < 0) offset += 1;
    if(ray.y < 0) offset += 2;
    if(ray.z < 0) offset += 4;

    vec3 pos = cam_data.pos;
    float depth = 0;
    for(int i = 0; i < 100; i++) {
        uint dist = get_dist(pos, offset);
        if(dist == 0) {
            break;
        }
        if(i == 99) {
            f_color = vec4(1.0, 0.0, 0.0, 1.0);
            return;
        }
        vec3 min = vec3(floor(pos.x), floor(pos.y), floor(pos.z)) - vec3(dist-1);
        vec3 max = vec3(ceil(pos.x), ceil(pos.y), ceil(pos.z)) + vec3(dist-1);
        float delta = RayBoxDist(pos, ray, min, max)+0.01;
        depth += delta;
        pos += ray * delta;
    }
    f_color = vec4(vec3(depth/sim_data.grid_size.x), 1.0);
}