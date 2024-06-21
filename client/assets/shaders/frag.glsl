#version 450
#include <common.glsl>

// The `color_input` parameter of the `draw` method.
layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_diffuse;
// The `normals_input` parameter of the `draw` method.
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;
// The `depth_input` parameter of the `draw` method.
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput u_depth;

layout(set = 1, binding = 0, r32ui) uniform uimage3D chunks;
layout(set = 1, binding = 1) buffer VoxelBuffer {
    uint voxels[];
};

layout(set = 1, binding = 2) uniform SimData {
    uint projectile_count;
    uvec3 render_size;
    uvec3 start_pos;
} sim_data;

layout(set = 1, binding = 3) buffer Players {
    Player players[];
};
layout(set = 1, binding = 4) buffer Projectiles {
    Projectile projectiles[];
};

layout(push_constant) uniform PushConstants {
    // The `screen_to_world` parameter of the `draw` method.
    mat4 screen_to_world;
    float aspect_ratio;
    float time;
    uint primary_ray_dist;
    uint shadow_ray_dist;
    uint reflection_ray_dist;
    uint transparent_shadow_ray_dist;
    uint ao_ray_dist;
    uint vertical_resolution;
} push_constants;

layout(location = 0) in vec2 v_screen_coords;
layout(location = 0) out vec4 f_color;

const vec3 light_dir = normalize(vec3(0.5, -1, 0.25));

uint get_data_unchecked(uvec3 global_pos) {
    uvec4 indicies = get_indicies(global_pos, sim_data.render_size);
    return voxels[imageLoad(chunks, ivec3(indicies.xyz)).x * CHUNK_VOLUME + indicies.w];
}

uint get_data(uvec3 global_pos) {
    uvec3 start_offset = CHUNK_SIZE * sim_data.start_pos;
    if (any(lessThan(global_pos, start_offset))) return MAT_OOB << 24;
    uvec3 rel_pos = global_pos - start_offset;
    if (any(greaterThanEqual(rel_pos, CHUNK_SIZE * sim_data.render_size))) return MAT_OOB << 24;
    return get_data_unchecked(global_pos);
}

uint get_dist(uint voxel_data, uint offset) {
    return (voxel_data >> (offset * 3)) & 0x7;
}

vec3 ray_box_dist(vec3 pos, vec3 ray, vec3 vmin, vec3 vmax) {
    vec3 normMinDiff = (vmin - pos) / ray;
    vec3 normMaxDiff = (vmax - pos) / ray;
    return max(normMinDiff, normMaxDiff);
}

struct RaycastResultLayer {
    vec3 pos;
    vec3 normal;
    uint voxel_data;
    float dist;
    bool is_leaving_medium;
};

struct RaycastResult {
    RaycastResultLayer layers[5];
    uint layer_count;
};

RaycastResultLayer simple_raycast(vec3 pos, vec3 ray, uint max_iterations, bool check_projectiles) {
    uint offset = 0;
    if (ray.x < 0) offset += 1;
    if (ray.y < 0) offset += 2;
    if (ray.z < 0) offset += 4;

    vec3 ray_pos = pos;
    vec3 normal = vec3(0);
    float depth = 0;
    uint voxel_data = MAT_OOB << 24;
    uint medium = get_data(uvec3(floor(ray_pos))) >> 24;
    bool did_hit = false;

    float max_dist = 1000000.0;
    vec3 end_ray_pos = pos;
    vec3 end_normal = normal;
    uint end_voxel_data = voxel_data;
    float end_depth = depth;

    if (check_projectiles) {
        //check if primary ray hit projectile
        vec3 min_normal = vec3(0);
        for (int i = 0; i < sim_data.projectile_count; i++) {
            vec4 inv_proj_rot_quaternion = quat_inverse(projectiles[i].dir);
            vec3 proj_size = projectiles[i].size.xyz;
            vec3 transformed_pos = quat_transform(inv_proj_rot_quaternion, (pos - projectiles[i].pos.xyz)) / proj_size;
            vec3 ray = quat_transform(inv_proj_rot_quaternion, ray) / proj_size;
            vec2 t_x = vec2((-1 - transformed_pos.x) / ray.x, (1 - transformed_pos.x) / ray.x);
            t_x = vec2(max(min(t_x.x, t_x.y), 0.0), min(max(t_x.x, t_x.y), max_dist));
            if (t_x.y < 0 || t_x.x > max_dist) continue;
            vec2 t_y = vec2((-1 - transformed_pos.y) / ray.y, (1 - transformed_pos.y) / ray.y);
            t_y = vec2(max(min(t_y.x, t_y.y), 0.0), min(max(t_y.x, t_y.y), max_dist));
            if (t_y.y < 0 || t_y.x > max_dist) continue;
            vec2 t_z = vec2((-1 - transformed_pos.z) / ray.z, (1 - transformed_pos.z) / ray.z);
            t_z = vec2(max(min(t_z.x, t_z.y), 0.0), min(max(t_z.x, t_z.y), max_dist));
            if (t_z.y < 0 || t_z.x > max_dist) continue;
            float t_min = max(max(t_x.x, t_y.x), t_z.x);
            float t_max = min(min(t_x.y, t_y.y), t_z.y);
            if (t_min > t_max) continue;
            if (t_min < 0.01) continue;
            if (t_min < max_dist) {
                max_dist = t_min;
                if (t_x.x == t_min) {
                    min_normal = vec3(-sign(ray.x), 0, 0);
                } else if (t_y.x == t_min) {
                    min_normal = vec3(0, -sign(ray.y), 0);
                } else {
                    min_normal = vec3(0, 0, -sign(ray.z));
                }
                min_normal = quat_transform(projectiles[i].dir, min_normal);
            }
        }
        if (length(min_normal) > 0) {
            end_depth = max_dist;
            did_hit = true;
            end_ray_pos = pos + max_dist * ray;
            end_normal = min_normal;
            end_voxel_data = MAT_PROJECTILE << 24;
        }
    }

    for (uint i = 0; i < max_iterations; i++) {
        vec3 floor_pos = floor(ray_pos);
        voxel_data = get_data(uvec3(floor_pos));
        vec3 v_min;
        vec3 v_max;
        uint voxel_material = voxel_data >> 24;
        if (voxel_material == MAT_AIR_OOB) {
            v_min = floor(ray_pos / CHUNK_SIZE) * CHUNK_SIZE;
            v_max = v_min + vec3(CHUNK_SIZE);
        } else if (voxel_material == medium) {
            uint dist = 0;
            if (voxel_material == MAT_AIR || voxel_material == MAT_WATER) {
                dist = get_dist(voxel_data, offset);
            }
            v_min = floor_pos - vec3(dist);
            v_max = floor_pos + vec3(dist + 1);
        } else {
            did_hit = true;
            end_ray_pos = ray_pos;
            end_depth = depth;
            end_normal = normal;
            if (medium != MAT_AIR) {
                end_voxel_data = medium << 24;
            } else {
                end_voxel_data = voxel_data;
            }
            end_voxel_data = voxel_data;
            end_depth = depth;
            break;
        }
        vec3 delta = ray_box_dist(ray_pos, ray, v_min, v_max);
        float dist_diff = min(delta.x, min(delta.y, delta.z));
        depth += dist_diff;
        if (depth > max_dist) {
            break;
        }
        ray_pos += ray * dist_diff;
        if (delta.x < delta.y && delta.x < delta.z) {
            normal = vec3(-sign(ray.x), 0, 0);
            if (ray.x > 0 && ray_pos.x < v_max.x) {
                ray_pos.x = v_max.x;
            } else if (ray.x < 0 && ray_pos.x >= v_min.x) {
                ray_pos.x = v_min.x - 0.001;
            }
        } else if (delta.y < delta.z) {
            normal = vec3(0, -sign(ray.y), 0);
            if (ray.y > 0 && ray_pos.y < v_max.y) {
                ray_pos.y = v_max.y;
            } else if (ray.y < 0 && ray_pos.y >= v_min.y) {
                ray_pos.y = v_min.y - 0.001;
            }
        } else {
            normal = vec3(0, 0, -sign(ray.z));
            if (ray.z > 0 && ray_pos.z < v_max.z) {
                ray_pos.z = v_max.z;
            } else if (ray.z < 0 && ray_pos.z >= v_min.z) {
                ray_pos.z = v_min.z - 0.001;
            }
        }
    }

    if (!did_hit) {
        return RaycastResultLayer(pos, vec3(0.0), MAT_OOB << 24, 0.0, false);
    }

    return RaycastResultLayer(end_ray_pos, end_normal, end_voxel_data, end_depth, false);
}

const uint LAYER_COUNT = 5;
RaycastResult raycast(vec3 pos, vec3 ray, uint max_iterations, bool check_projectiles, float raster_depth) {
    RaycastResultLayer[5] layers;
    uint offset = 0;
    if (ray.x < 0) offset += 1;
    if (ray.y < 0) offset += 2;
    if (ray.z < 0) offset += 4;

    vec3 ray_pos = pos;
    vec3 normal = -ray;
    float depth = 0;
    uint voxel_data = MAT_OOB << 24;
    uint medium = MAT_AIR;
    uint layer_idx = 0;
    for (uint i = 0; i < max_iterations; i++) {
        vec3 floor_pos = floor(ray_pos);
        voxel_data = get_data(uvec3(floor_pos));
        uint voxel_material = voxel_data >> 24;
        vec3 v_min;
        vec3 v_max;
        if (voxel_material == MAT_AIR || voxel_material == MAT_AIR_OOB) {
            if (medium != MAT_AIR) {
                layers[layer_idx] = RaycastResultLayer(ray_pos, normal, medium << 24, depth, true);
                layer_idx++;
                if (layer_idx >= LAYER_COUNT) break;
            }
        }
        if (voxel_material == MAT_AIR) {
            uint dist = get_dist(voxel_data, offset);
            v_min = floor_pos - vec3(dist);
            v_max = floor_pos + vec3(dist + 1);
            medium = MAT_AIR;
        } else if (voxel_material == MAT_AIR_OOB) {
            v_min = floor(ray_pos / CHUNK_SIZE) * CHUNK_SIZE;
            v_max = v_min + vec3(CHUNK_SIZE);
            medium = MAT_AIR;
        } else if (is_transparent[voxel_material]) {
            uint dist = 0;
            if (voxel_material == MAT_WATER) {
                dist = get_dist(voxel_data, offset);
            }
            v_min = floor_pos;
            v_max = floor_pos + vec3(1);
            if (medium != voxel_material) {
                layers[layer_idx] = RaycastResultLayer(ray_pos, normal, voxel_data, depth, false);
                layer_idx++;
                if (layer_idx >= LAYER_COUNT) break;
            }
            medium = voxel_material;
        } else {
            layers[layer_idx] = RaycastResultLayer(ray_pos, normal, voxel_data, depth, false);
            layer_idx++;
            break;
        }
        vec3 delta = ray_box_dist(ray_pos, ray, v_min, v_max);
        float dist_diff = min(delta.x, min(delta.y, delta.z));
        if (depth + dist_diff > raster_depth && raster_depth > 0) {
            depth = raster_depth;
            ray_pos = pos + depth * ray;
            break;
        }
        depth += dist_diff;
        ray_pos += ray * dist_diff;
        if (delta.x < delta.y && delta.x < delta.z) {
            normal = vec3(-sign(ray.x), 0, 0);
            if (ray.x < 0 && ray_pos.x >= v_min.x) {
                ray_pos.x = v_min.x - 0.001;
            }
        } else if (delta.y < delta.z) {
            normal = vec3(0, -sign(ray.y), 0);
            if (ray.y < 0 && ray_pos.y >= v_min.y) {
                ray_pos.y = v_min.y - 0.001;
            }
        } else {
            normal = vec3(0, 0, -sign(ray.z));
            if (ray.z < 0 && ray_pos.z >= v_min.z) {
                ray_pos.z = v_min.z - 0.001;
            }
        }
    }

    if (raster_depth > 0.0 && layer_idx < LAYER_COUNT) {
        vec3 in_normal = normalize(subpassLoad(u_normals).rgb);
        vec4 in_diffuse = subpassLoad(u_diffuse);
        uint raster_material = MAT_PLAYER;
        if (in_diffuse.x == 0.0) {
            raster_material = MAT_PROJECTILE;
        }
        layers[layer_idx] = RaycastResultLayer(ray_pos, in_normal, raster_material << 24, raster_depth, false);
        layer_idx++;
    }

    if (check_projectiles) {
        //check if primary ray hit projectile
        vec3 normal = vec3(0);
        for (int i = 0; i < sim_data.projectile_count; i++) {
            vec4 inv_proj_rot_quaternion = quat_inverse(projectiles[i].dir);
            vec3 proj_size = projectiles[i].size.xyz;
            vec3 transformed_pos = quat_transform(inv_proj_rot_quaternion, (pos - projectiles[i].pos.xyz)) / proj_size;
            vec3 ray = quat_transform(inv_proj_rot_quaternion, ray) / proj_size;
            vec2 t_x = vec2((-1 - transformed_pos.x) / ray.x, (1 - transformed_pos.x) / ray.x);
            t_x = vec2(max(min(t_x.x, t_x.y), 0.0), min(max(t_x.x, t_x.y), depth));
            vec2 t_y = vec2((-1 - transformed_pos.y) / ray.y, (1 - transformed_pos.y) / ray.y);
            t_y = vec2(max(min(t_y.x, t_y.y), 0.0), min(max(t_y.x, t_y.y), depth));
            vec2 t_z = vec2((-1 - transformed_pos.z) / ray.z, (1 - transformed_pos.z) / ray.z);
            t_z = vec2(max(min(t_z.x, t_z.y), 0.0), min(max(t_z.x, t_z.y), depth));
            float dist = max(max(t_x.x, t_y.x), t_z.x);
            float t_max = min(min(t_x.y, t_y.y), t_z.y);
            if (t_max < 0 || dist > depth) continue;
            if (dist > t_max) continue;
            if (dist < 0.01) continue;
            if (t_x.x == dist) {
                normal = vec3(-sign(ray.x), 0, 0);
            } else if (t_y.x == dist) {
                normal = vec3(0, -sign(ray.y), 0);
            } else {
                normal = vec3(0, 0, -sign(ray.z));
            }
            normal = quat_transform(projectiles[i].dir, normal);

            RaycastResultLayer proj_layer = RaycastResultLayer(pos + dist * ray, normal, MAT_PROJECTILE << 24, dist, false);
            // insert layer
            if (layer_idx < LAYER_COUNT) {
                layer_idx++;
            }
            for (uint j = layer_idx - 1; j > 0; j--) {
                if (proj_layer.dist < layers[j - 1].dist) {
                    layers[j] = layers[j - 1];
                    if (j == 1) {
                        layers[0] = proj_layer;
                    }
                } else {
                    layers[j] = proj_layer;
                    break;
                }
            }
        }
    }
    return RaycastResult(layers, layer_idx);
}

struct MaterialProperties {
    vec3 albedo;
    vec3 normal;
    float ior;
    float roughness;
    float metallic;
    float emmision;
    float transparency;
    float depth_transparency;
};

float max3(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

MaterialProperties material_props(RaycastResultLayer resultLayer, vec3 ray_dir) {
    uint material = resultLayer.voxel_data >> 24;
    uint data = resultLayer.voxel_data & 0xFFFFFF;
    MaterialRenderProps mat_render_props = material_render_props[material];
    if (material == MAT_AIR || material == MAT_AIR_OOB) {
        // air: invalid state
        return MaterialProperties(vec3(1.0, 0.0, 0.0), resultLayer.normal, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    } else if (material == MAT_OOB) {
        // out of bounds: invalid state
        return MaterialProperties(vec3(0.0, 0.0, 1.0), resultLayer.normal, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    } else if (material == MAT_PROJECTILE || material == MAT_PLAYER) {
        return MaterialProperties(mat_render_props.color, resultLayer.normal, mat_render_props.ior, mat_render_props.roughness, 0.0, 0.1, mat_render_props.transparency, 0.0);
    }
    vec3 normal = resultLayer.normal;
    vec3 color = mat_render_props.color;
    float roughness = mat_render_props.roughness;
    float transparency = mat_render_props.transparency;
    for (int layer_idx = 0; layer_idx < 3; layer_idx++) {
        MaterialNoiseLayer layer = mat_render_props.layers[layer_idx];
        vec4 noise = grad_noise(layer.scale * resultLayer.pos + layer.movement * push_constants.time);
        float distance_noise_factor = clamp(-0.1 * float(push_constants.vertical_resolution) * dot(ray_dir, resultLayer.normal) / (max(resultLayer.dist, 0.1) * max3(layer.scale)), 0.0, 1.0);
        normal += distance_noise_factor * layer.normal_impact * noise.xyz * (vec3(1) - abs(resultLayer.normal));
        color += layer.layer_color * mix(0.0, noise.w, distance_noise_factor);
        roughness += distance_noise_factor * layer.roughness_impact * noise.w;
        transparency += distance_noise_factor * layer.transparency_impact * noise.w;
    }
    normal = normalize(normal);
    if (physics_properties[material].is_data_damage) {
        color *= (1.0 - float(data) / material_damage_threshhold[material]);
    }
    return MaterialProperties(
        color,
        normal,
        mat_render_props.ior,
        roughness,
        0.0,
        0.1,
        transparency,
        mat_render_props.depth_transparency
    );
}

MaterialProperties position_material(RaycastResultLayer resultLayer, vec3 ray_dir) {
    if (resultLayer.voxel_data >> 24 == MAT_PLAYER || resultLayer.voxel_data >> 24 == MAT_PROJECTILE) {
        return material_props(resultLayer, ray_dir);
    }
    vec3 relative_pos = resultLayer.pos - floor(resultLayer.pos) - 0.5;
    vec3 weights = abs(relative_pos);
    uint result_vox = resultLayer.voxel_data;
    HeightData voxel_height_data = height_data[result_vox >> 24];
    float result_height = (voxel_height_data.offset + voxel_height_data.impact * grad_noise(voxel_height_data.scale * resultLayer.pos + voxel_height_data.movement * push_constants.time).w) * (1.0 - weights.x) * (1.0 - weights.y) * (1.0 - weights.z);
    uint face_voxel_material = get_data(uvec3(floor(resultLayer.pos + resultLayer.normal))) >> 24;
    for (int i = 1; i < 8; i++) {
        vec3 voxel_direction = vec3(float(i & 1), float((i & 2) >> 1), float((i & 4) >> 2)) * sign(relative_pos);
        if (dot(voxel_direction, resultLayer.normal) > 0.0) continue;
        uint voxel = get_data(uvec3(floor(resultLayer.pos) + voxel_direction));
        uint voxel_material = voxel >> 24;
        if (voxel_material == face_voxel_material) continue;
        if (
            voxel_material == MAT_OOB
                || voxel_material == MAT_AIR
                || voxel_material == MAT_AIR_OOB
        ) continue;
        float weight = 1.0;
        if ((i & 1) == 0) weight *= 1.0 - weights.x;
        else weight *= weights.x;
        if ((i & 2) == 0) weight *= 1.0 - weights.y;
        else weight *= weights.y;
        if ((i & 4) == 0) weight *= 1.0 - weights.z;
        else weight *= weights.z;
        voxel_height_data = height_data[voxel_material];
        float distance_noise_factor = clamp(-0.1 * float(push_constants.vertical_resolution) * dot(ray_dir, resultLayer.normal) / (max(resultLayer.dist, 0.1) * max3(voxel_height_data.scale)), 0.0, 1.0);
        float height = (voxel_height_data.offset - voxel_height_data.impact * distance_noise_factor * grad_noise(voxel_height_data.scale * resultLayer.pos).w) * weight;
        if (height > result_height) {
            result_height = height;
            result_vox = voxel;
        }
    }
    resultLayer.voxel_data = result_vox;
    return material_props(resultLayer, ray_dir);
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 get_light(vec3 L, vec3 V, vec3 lightColor, float attenuation, MaterialProperties mat_props) {
    vec3 F0 = vec3(mat_props.ior);
    F0 = mix(F0, mat_props.albedo, mat_props.metallic);
    // calculate per-light radiance
    vec3 H = normalize(V + L);
    vec3 radiance = lightColor * attenuation;

    // cook-torrance brdf
    float NDF = DistributionGGX(mat_props.normal, H, mat_props.roughness);
    float G = GeometrySmith(mat_props.normal, V, L, mat_props.roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - mat_props.metallic;

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(mat_props.normal, V), 0.0) * max(dot(mat_props.normal, L), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;

    // add to outgoing radiance Lo
    float NdotL = max(dot(mat_props.normal, L), 0.0);
    return (kD * mat_props.albedo / PI + specular) * radiance * NdotL;
}
const float epsilon = 0.001;
vec3 get_color(vec3 pos, vec3 ray, RaycastResult primary_ray) {
    vec3 color = vec3(0.0);
    float multiplier = 1.0;
    int i = 0;

    while (multiplier > 0.05 && i < primary_ray.layer_count) {
        if (primary_ray.layers[i].voxel_data >> 24 == MAT_OOB) {
            float sky_brightness = max(dot(ray, -light_dir), 0.0);
            sky_brightness += pow(sky_brightness, 10.0);
            color += multiplier * (sky_brightness * vec3(0.429, 0.608, 0.622) + vec3(0.1, 0.1, 0.4));
            break;
        }
        MaterialProperties mat_props = position_material(primary_ray.layers[i], ray);
        color += (1 - mat_props.transparency) * multiplier * mat_props.albedo * mat_props.emmision;

        RaycastResultLayer shade_check = simple_raycast(primary_ray.layers[i].pos + epsilon * primary_ray.layers[i].normal, -light_dir, push_constants.shadow_ray_dist, true);
        float shade_transparency = material_render_props[shade_check.voxel_data >> 24].transparency;
        if (shade_transparency > 0.0) {
            vec3 v_min = floor(shade_check.pos);
            vec3 v_max = floor(shade_check.pos) + vec3(1);
            vec3 delta = ray_box_dist(shade_check.pos, -light_dir, v_min, v_max);
            float dist_diff = min(delta.x, min(delta.y, delta.z)) + 0.01;
            shade_check = simple_raycast(shade_check.pos - dist_diff * light_dir, -light_dir, push_constants.transparent_shadow_ray_dist, false);
        }
        if (shade_check.voxel_data >> 24 == MAT_OOB) {
            color += shade_transparency * (1 - mat_props.transparency) * multiplier * get_light(-light_dir, -ray, vec3(1.0), 1.0, mat_props);
        }

        vec3 reflection = reflect(ray, mat_props.normal);
        RaycastResultLayer reflection_check = simple_raycast(primary_ray.layers[i].pos + epsilon * primary_ray.layers[i].normal, reflection, push_constants.reflection_ray_dist, false);
        MaterialRenderProps reflection_props = material_render_props[reflection_check.voxel_data >> 24];
        RaycastResultLayer reflection_light_check = simple_raycast(reflection_check.pos + epsilon * reflection_check.normal, -light_dir, push_constants.ao_ray_dist, false);
        vec3 light = vec3(0);
        if (reflection_light_check.voxel_data >> 24 == MAT_OOB) {
            light += (1.0 - reflection_props.transparency) * reflection_props.color * 0.15;
        }
        if (reflection_check.voxel_data >> 24 == MAT_OOB || reflection_props.transparency == 1.0) {
            light += vec3(pow(dot(reflection, -light_dir), 3.0));
        }
        color += (1 - mat_props.transparency) * multiplier * get_light(reflection, -ray, light, 1.0, mat_props);

        vec3 ao_dir = mat_props.normal;
        RaycastResultLayer ao_check = simple_raycast(primary_ray.layers[i].pos + epsilon * primary_ray.layers[i].normal, ao_dir, push_constants.ao_ray_dist, false);
        if (ao_check.voxel_data >> 24 == MAT_OOB) {
            float light_power = pow(dot(ao_dir, -light_dir), 3.0);
            color += (1 - mat_props.transparency) * multiplier * get_light(ao_dir, -ray, vec3(1.0), light_power, mat_props);
        }

        ao_dir = normalize(primary_ray.layers[i].normal + mat_props.normal);
        ao_check = simple_raycast(primary_ray.layers[i].pos + epsilon * primary_ray.layers[i].normal, ao_dir, push_constants.ao_ray_dist, false);
        if (ao_check.voxel_data >> 24 == MAT_OOB) {
            float light_power = pow(dot(ao_dir, -light_dir), 3.0);
            color += (1 - mat_props.transparency) * multiplier * get_light(ao_dir, -ray, vec3(1.0), light_power, mat_props);
        }

        if (i + 1 < primary_ray.layer_count && mat_props.depth_transparency > 0.0 && !primary_ray.layers[i].is_leaving_medium) {
            float dist = primary_ray.layers[i + 1].dist - primary_ray.layers[i].dist;
            float depth_transparency = pow(mat_props.depth_transparency, dist);
            color += (1.0 - depth_transparency) * mat_props.transparency * multiplier * mat_props.albedo;
            multiplier *= depth_transparency;
        }

        multiplier *= mat_props.transparency;
        i++;
    }

    return color;
}

void main() {
    Player cam_data = players[0];

    float in_depth = subpassLoad(u_depth).x;

    // Find the world coordinates of the current pixel.
    vec2 scaled_screen_coords = v_screen_coords * vec2(push_constants.aspect_ratio, 1.0);
    vec4 world = push_constants.screen_to_world * vec4(scaled_screen_coords, in_depth, 1.0);
    world /= world.w;

    float max_depth = 0.0;
    if (in_depth < 1.0) {
        max_depth = length(world.xyz - cam_data.pos.xyz);
    }
    vec3 ray = normalize(cam_data.dir.xyz + scaled_screen_coords.x * cam_data.right.xyz - scaled_screen_coords.y * cam_data.up.xyz);

    vec3 pos = cam_data.pos.xyz; // + ray * 0.1;

    RaycastResult primary_ray = raycast(pos, ray, push_constants.primary_ray_dist, true, max_depth);

    if (primary_ray.layer_count == 0) {
        f_color = vec4(1.0, 0.0, 0.0, 1.0);
        return;
    }

    f_color = vec4(get_color(pos, ray, primary_ray), 1.0);
}
