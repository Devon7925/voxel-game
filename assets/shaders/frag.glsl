#version 450
#include <common.glsl>

// The `color_input` parameter of the `draw` method.
layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_diffuse;
// The `normals_input` parameter of the `draw` method.
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;
// The `depth_input` parameter of the `draw` method.
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput u_depth;

layout(set = 1, binding = 0) buffer VoxelBuffer { uvec2 voxels[]; };

layout(set = 1, binding = 1) uniform SimData {
    uint max_dist;
    uint projectile_count;
    uvec3 render_size;
    ivec3 start_pos;
} sim_data;

layout(set = 1, binding = 2) buffer Players { Player players[]; };
layout(set = 1, binding = 3) buffer Projectiles { Projectile projectiles[]; };

layout(push_constant) uniform PushConstants {
    // The `screen_to_world` parameter of the `draw` method.
    mat4 screen_to_world;
    // The `color` parameter of the `draw` method.
    vec4 color;
    // The `position` parameter of the `draw` method.
    vec4 position;
} push_constants;

layout(location = 0) in vec2 v_screen_coords;
layout(location = 0) out vec4 f_color;

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

RaycastResult raycast(vec3 pos, vec3 ray, uint max_iterations, bool check_projectiles, float max_depth) {
    uint offset = 0;
    if(ray.x < 0) offset += 1;
    if(ray.y < 0) offset += 2;
    if(ray.z < 0) offset += 4;

    vec3 ray_pos = pos;
    vec3 normal = vec3(0);
    float depth = 0;
    uvec2 voxel_data = uvec2(20, 0);
    bool did_hit = false;
    for(uint i = 0; i < max_iterations; i++) {
        voxel_data = get_data(ivec3(floor(ray_pos)));
        if(voxel_data.x != 0) {
            did_hit = true;
            break;
        }
        uint dist = get_dist(voxel_data, offset);
        vec3 v_min = floor(ray_pos) - vec3(dist-1);
        vec3 v_max = floor(ray_pos) + vec3(dist);
        vec3 delta = RayBoxDist(ray_pos, ray, v_min, v_max);
        float dist_diff = min(delta.x, min(delta.y, delta.z));
        if (depth + dist_diff > max_depth && max_depth > 0) {
            break;
        }
        depth += dist_diff;
        ray_pos += ray * dist_diff;
        if (delta.x < delta.y && delta.x < delta.z) {
            normal = vec3(-sign(ray.x), 0, 0);
            if (ray.x > 0 && ray_pos.x < v_max.x) {
                ray_pos.x = v_max.x;
            } else if (ray.x < 0 && ray_pos.x >= v_min.x) {
                ray_pos.x = v_min.x-0.01;
            }
        } else if (delta.y < delta.z) {
            normal = vec3(0, -sign(ray.y), 0);
            if (ray.y > 0 && ray_pos.y < v_max.y) {
                ray_pos.y = v_max.y;
            } else if (ray.y < 0 && ray_pos.y >= v_min.y) {
                ray_pos.y = v_min.y-0.001;
            }
        } else {
            normal = vec3(0, 0, -sign(ray.z));
            if (ray.z > 0 && ray_pos.z < v_max.z) {
                ray_pos.z = v_max.z;
            } else if (ray.z < 0 && ray_pos.z >= v_min.z) {
                ray_pos.z = v_min.z-0.001;
            }
        }
    }

    if (check_projectiles) {
        //check if primary ray hit projectile
        float min_dist = 1000000.0;
        vec3 min_normal = vec3(0);
        for (int i = 0; i < sim_data.projectile_count; i++) {
            vec3 proj_pos = projectiles[i].pos.xyz;
            vec4 proj_rot_quaternion = quat_inverse(projectiles[i].dir);
            vec3 proj_size = projectiles[i].size.xyz;
            float does_exist = projectiles[i].pos.w;
            vec3 transformed_pos = quat_transform(projectiles[i].dir, (pos - proj_pos) / proj_size);
            vec3 ray = quat_transform(projectiles[i].dir, ray / proj_size);
            vec2 t_x = vec2((-1 - transformed_pos.x) / ray.x, (1 - transformed_pos.x) / ray.x);
            t_x = vec2(max(min(t_x.x, t_x.y), 0.0), min(max(t_x.x, t_x.y), depth));
            if (t_x.y < 0 || t_x.x > depth) continue;
            vec2 t_y = vec2((-1 - transformed_pos.y) / ray.y, (1 - transformed_pos.y) / ray.y);
            t_y = vec2(max(min(t_y.x, t_y.y), 0.0), min(max(t_y.x, t_y.y), depth));
            if (t_y.y < 0 || t_y.x > depth) continue;
            vec2 t_z = vec2((-1 - transformed_pos.z) / ray.z, (1 - transformed_pos.z) / ray.z);
            t_z = vec2(max(min(t_z.x, t_z.y), 0.0), min(max(t_z.x, t_z.y), depth));
            if (t_z.y < 0 || t_z.x > depth) continue;
            float t_min = max(max(t_x.x, t_y.x), t_z.x);
            float t_max = min(min(t_x.y, t_y.y), t_z.y);
            if (t_min > t_max) continue;
            if (t_min < min_dist) {
                min_dist = t_min;
                if (t_x.x == t_min) {
                    min_normal = vec3(-sign(ray.x), 0, 0);
                } else if (t_y.x == t_min) {
                    min_normal = vec3(0, -sign(ray.y), 0);
                } else {
                    min_normal = vec3(0, 0, -sign(ray.z));
                }
                min_normal = quat_transform(proj_rot_quaternion, min_normal);
            }
        }
        if (length(min_normal) > 0) {
            depth = min_dist;
            did_hit = true;
            ray_pos = pos + min_dist*ray;
            normal = min_normal;
            voxel_data = uvec2(1, 0);
        }
    }

    return RaycastResult(ray_pos, normal, voxel_data, depth, did_hit);
}

void main() {
    Player cam_data = players[0];

    float in_depth = subpassLoad(u_depth).x;

    float n = 0.01;
    float f = 100.0;

    float z_ndc = 2.0 * in_depth - 1.0;
    float z_eye = 2.0 * n * f / (f + n - z_ndc * (f - n));

    float tanFov = tan(radians(90) / 2.0);
    float aspect = 1.0;

    vec3 viewPos = vec3(
        z_eye * v_screen_coords.x * aspect * tanFov,
        z_eye * v_screen_coords.y * tanFov,
        -z_eye
    );

    float max_depth = 0.0;
    if (in_depth < 1.0) {
        max_depth = length(viewPos);
    }
    vec3 ray = normalize(cam_data.dir.xyz + v_screen_coords.x * cam_data.right.xyz - v_screen_coords.y * cam_data.up.xyz);

    vec3 pos = cam_data.pos.xyz;
    RaycastResult primary_ray = raycast(pos, ray, 100, false, max_depth);
    
    if (!primary_ray.hit) {
        if (in_depth >= 1.0) {
            f_color = vec4(1.0, 0.0, 0.0, 1.0);
            return;
        }
        vec3 in_normal = normalize(subpassLoad(u_normals).rgb);

        vec3 in_diffuse = subpassLoad(u_diffuse).rgb;
        // f_color.rgb = push_constants.color.rgb * light_percent * in_diffuse;
        primary_ray.hit = true;
        primary_ray.normal = in_normal;
        primary_ray.pos = pos + max_depth*ray;
        primary_ray.dist = max_depth;
        primary_ray.voxel_data = uvec2(1, 0);
    } else if (primary_ray.voxel_data.x == 2) {
        f_color = vec4(0.529, 0.808, 0.922, 1.0);
        return;
    }
    RaycastResult shade_check = raycast(primary_ray.pos + 0.015*primary_ray.normal, -light_dir, 100, true, 0.0);
    vec3 color = vec3(0.1, 0.1, 0.1);
    if (!shade_check.hit || shade_check.voxel_data.x == 2) {
        vec3 mat_color = vec3(1.0, 1.0, primary_ray.voxel_data.y / 10.0);
        float diffuse = 0.7*max(dot(primary_ray.normal, -light_dir), 0.0);
        color += diffuse*mat_color;
    }

    f_color = vec4(color, 1.0);
}