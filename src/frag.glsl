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
    uvec3 render_size;
    ivec3 start_pos;
} sim_data;

const vec3 light_dir = normalize(vec3(0.5, -1, 0.25));

uvec2 get_data(ivec3 global_pos) {
    ivec3 rel_pos = global_pos - ivec3(CHUNK_SIZE * sim_data.start_pos);
    if (any(lessThan(rel_pos, ivec3(0))) || any(greaterThanEqual(rel_pos, ivec3(CHUNK_SIZE * sim_data.render_size)))) return uvec2(2, 0);
    uint index = get_index(global_pos, sim_data.render_size);
    return voxels[index];
}

uint get_dist(uvec2 voxel_data, uint offset) {
    if (voxel_data.x == 1) return 0;
    return (voxel_data.y >> (offset * 4)) & 0xF;
}

vec3 RayBoxDist(vec3 pos, vec3 ray, vec3 vmin, vec3 vmax) {
    vec3 normMinDiff = (vmin - pos) / ray;
    vec3 normMaxDiff = (vmax - pos) / ray;
    return max(normMinDiff, normMaxDiff);
}

struct RaycastResult {
    vec3 pos;
    vec3 normal;
    uvec2 voxel_data;
    float dist;
    bool hit;
};

RaycastResult raycast(vec3 pos, vec3 ray) {
    uint offset = 0;
    if(ray.x < 0) offset += 1;
    if(ray.y < 0) offset += 2;
    if(ray.z < 0) offset += 4;

    float depth = 0;
    uvec2 voxel_data = uvec2(20, 0);
    vec3 normal = vec3(0);
    for(int i = 0; i < 100; i++) {
        voxel_data = get_data(ivec3(floor(pos)));
        if(voxel_data.x != 0) {
            break;
        }
        uint dist = get_dist(voxel_data, offset);
        if(dist == 0) {
            break;
        }
        if(i == 99) {
            return RaycastResult(pos, vec3(0), voxel_data, depth, false);
        }
        vec3 v_min = floor(pos) - vec3(dist-1);
        vec3 v_max = floor(pos) + vec3(dist);
        vec3 delta = RayBoxDist(pos, ray, v_min, v_max);
        float dist_diff = min(delta.x, min(delta.y, delta.z));
        depth += dist_diff;
        pos += ray * dist_diff;
        if (delta.x < delta.y && delta.x < delta.z) {
            normal = vec3(-sign(ray.x), 0, 0);
            if (ray.x > 0 && pos.x < v_max.x) {
                pos.x = v_max.x;
            } else if (ray.x < 0 && pos.x >= v_min.x) {
                pos.x = v_min.x-0.01;
            }
        } else if (delta.y < delta.z) {
            normal = vec3(0, -sign(ray.y), 0);
            if (ray.y > 0 && pos.y < v_max.y) {
                pos.y = v_max.y;
            } else if (ray.y < 0 && pos.y >= v_min.y) {
                pos.y = v_min.y-0.001;
            }
        } else {
            normal = vec3(0, 0, -sign(ray.z));
            if (ray.z > 0 && pos.z < v_max.z) {
                pos.z = v_max.z;
            } else if (ray.z < 0 && pos.z >= v_min.z) {
                pos.z = v_min.z-0.001;
            }
        }
    }
    return RaycastResult(pos, normal, voxel_data, depth, true);
}

void main() {
    vec3 ray = normalize(cam_data.dir + v_tex_coords.x * cam_data.right + v_tex_coords.y * cam_data.up);

    vec3 pos = cam_data.pos;
    RaycastResult result = raycast(pos, ray);
    if (!result.hit) {
        f_color = vec4(1.0, 0.0, 0.0, 1.0);
        return;
    } else if (result.voxel_data.x == 2) {
        f_color = vec4(0.0, 0.0, 1.0, 1.0);
        return;
    }
    RaycastResult shade_check = raycast(result.pos + 0.015*result.normal, -light_dir);
    vec3 ambient_light = vec3(0.1, 0.1, 0.1);
    if (shade_check.hit && shade_check.voxel_data.x != 2) {
        f_color = vec4(ambient_light, 1.0);
        return;
    }
    float diffuse = 0.9*max(dot(result.normal, -light_dir), 0.0);
    f_color = vec4(vec3(diffuse) + ambient_light, 1.0);
}