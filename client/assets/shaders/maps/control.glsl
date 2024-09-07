#version 450
#include <common.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(set = 0, binding = 1, r32ui) uniform uimage3D voxels;
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
    uint z = chunk_idx;
    uint y = z/1024;
    z = z % 1024;
    uint x = y/1024;
    y = y % 1024;
    imageStore(voxels, ivec3(x * CHUNK_SIZE, y * CHUNK_SIZE, z * CHUNK_SIZE) + ivec3(global_pos % CHUNK_SIZE), uvec4(data, 0, 0, 0));
}

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

const int SPAWN_ROOM_OFFSET = 150;
const float WALL_SCALE = 0.03;
const float PATH_SCALE = 0.03;
const float GAP_SCALE = 0.06;

vec2 hash2(vec2 p, uint seed)
{
    // procedural white noise
    return vec2(pcg3d(ivec3(p, seed)).xy & 0xFF) / 256.0;
}

vec3 hash23(vec2 p, uint seed)
{
    // procedural white noise
    return vec3(pcg3d(ivec3(p, seed)) & 0xFF) / 256.0;
}

struct VoronoiResult {
    vec2 edge_dir;
    float height;
    float nearest_height;
    float edge_distance;
};

VoronoiResult voronoi_edge_dist(in vec2 x, uint seed)
{
    vec2 ip = floor(x);
    vec2 fp = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
    vec2 mr;
    float h;

    float md = 8.0;
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++)
        {
            vec2 g = vec2(float(i), float(j));
            vec2 coords = ip + g;
            coords.x = abs(coords.x);
            vec3 o = hash23(coords, seed);
            vec2 r = g + o.xy - fp;
            float d = dot(r, r);

            if (d < md)
            {
                md = d;
                mr = r;
                h = o.z;
            }
        }
    // Set center of search based on which half of the cell we are in,
    // since 4x4 is not centered around "n".
    vec2 mg = step(.5, fp) - 1.;

    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 8.0;
    float mh = h;
    vec2 max_edge_dir = vec2(0.0);
    for (int j = -1; j <= 2; j++)
        for (int i = -1; i <= 2; i++)
        {
            vec2 g = mg + vec2(float(i), float(j));
            vec2 coords = ip + g;
            coords.x = abs(coords.x);
            vec3 o = hash23(coords, seed);
            vec2 r = g + o.xy - fp;

            if (dot(mr - r, mr - r) > 0.00001) {
                vec2 edge_dir = normalize(r - mr);
                float d = dot(0.5 * (mr + r), edge_dir);
                if (d < md) {
                    md = d;
                    mh = o.z;
                    max_edge_dir = edge_dir;
                }
            }
        }

    return VoronoiResult(max_edge_dir, h, mh, md);
}

vec2 blurred_voronoi(in vec2 x, float w, uint seed)
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    vec2 m = vec2(8.0, 0.0);
    for (int j = -2; j <= 2; j++)
        for (int i = -2; i <= 2; i++)
        {
            vec2 g = vec2(float(i), float(j));
            vec2 coords = n + g;
            coords.x = abs(coords.x);
            vec3 o = hash23(coords, seed);

            // distance to cell
            float d = length(g - f + o.xy);
            float height = 30.0 * o.z - 15.0;
            if (height < -13) continue;

            // do the smooth min for heights and distances
            float h = smoothstep(-1.0, 1.0, (m.x - d) / w);
            m.x = mix(m.x, d, h) - h * (1.0 - h) * w / (1.0 + 3.0 * w); // distance
            m.y = mix(m.y, height, h) - h * (1.0 - h) * w / (1.0 + 3.0 * w); // height
        }

    return m;
}

float sdBox(in vec2 p, in vec2 b)
{
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

const int SPAWN_HEIGHT = 15;
const int SPAWN_DEPTH = 15;
const int SPAWN_WIDTH = 15;

uint get_worldgen(uvec3 global_pos) {
    uint WALL_SEED = sim_data.worldgen_seed;
    uint PATH_SEED = sim_data.worldgen_seed + 1;

    ivec3 signed_pos = ivec3(global_pos);
    signed_pos.x = abs(signed_pos.x - 10000);
    signed_pos.y = signed_pos.y - 1800;
    signed_pos.z = signed_pos.z - 10000;
    vec3 true_pos = vec3(signed_pos);
    if (signed_pos.x < 12 && abs(signed_pos.z) < 12) {
        if (signed_pos.y > 0) {
            return MAT_AIR << 24;
        }
        return MAT_UNBREAKABLE << 24;
    }
    if (signed_pos.y < -13) {
        return MAT_AIR << 24;
    }
    if (signed_pos.x < SPAWN_ROOM_OFFSET + SPAWN_DEPTH - 1 && signed_pos.x > SPAWN_ROOM_OFFSET && signed_pos.y < SPAWN_HEIGHT - 1 && signed_pos.y > 0 && abs(signed_pos.z) < SPAWN_WIDTH - 1) {
        return MAT_AIR << 24;
    }
    if (signed_pos.x < SPAWN_ROOM_OFFSET + SPAWN_DEPTH && signed_pos.x > SPAWN_ROOM_OFFSET && signed_pos.y < SPAWN_HEIGHT && abs(signed_pos.z) < SPAWN_WIDTH) {
        return MAT_UNBREAKABLE << 24;
    }
    if (signed_pos.x > SPAWN_ROOM_OFFSET) {
        return MAT_AIR << 24;
    }
    if (abs(true_pos.z) > 100) {
        return MAT_AIR << 24;
    }
    VoronoiResult wall_noise = voronoi_edge_dist(WALL_SCALE * true_pos.xz, WALL_SEED);
    wall_noise.height = 30.0 * wall_noise.height - 15.0;
    wall_noise.nearest_height = 30.0 * wall_noise.nearest_height - 15.0;
    VoronoiResult path_noise = voronoi_edge_dist(PATH_SCALE * true_pos.xz, PATH_SEED);
    vec2 path_center = path_noise.edge_distance / PATH_SCALE * path_noise.edge_dir + true_pos.xz;
    vec2 wall_noise_at_path_center = blurred_voronoi(WALL_SCALE * path_center, 0.5, WALL_SEED);
    vec4 gap_noise = grad_noise(vec3(GAP_SCALE * true_pos.x, 0.0, GAP_SCALE * true_pos.z));
    float flattening_factor = min(0.025 * length(true_pos.xz - vec2(SPAWN_ROOM_OFFSET, 0.0)), 1.0) * min(0.07 * sdBox(true_pos.xz, vec2(12.0)), 1.0);
    float adj_terrain_height = wall_noise.height * flattening_factor;
    float adj_path_height = wall_noise_at_path_center.y * flattening_factor;
    if (path_noise.edge_distance < 0.1 && adj_path_height >= -13) {
        if (abs(signed_pos.y - adj_path_height) < 0.5) {
            return MAT_UNBREAKABLE << 24;
        } else if (true_pos.y > adj_path_height && true_pos.y < adj_path_height + 15) {
            return MAT_AIR << 24;
        } else if (true_pos.y < adj_path_height && true_pos.y > adj_terrain_height && adj_terrain_height > -13) {
            return MAT_STONE << 24;
        }
    }
    if (true_pos.y < adj_terrain_height) {
        return MAT_STONE << 24;
    }
    if (wall_noise.edge_distance < 0.05 && true_pos.y < 12.0 + adj_terrain_height && wall_noise.height > wall_noise.nearest_height && gap_noise.w < 0.0 && flattening_factor > 0.2) {
        return MAT_STONE << 24;
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
