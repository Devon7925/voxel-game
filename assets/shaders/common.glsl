const uint CHUNK_SIZE = 16;

struct Projectile {
    vec4 pos;
    ivec4 chunk_update_pos;
    vec4 dir;
    vec4 size;
    float vel;
    float health;
    float lifetime;
    uint owner;
    float damage;
    uint proj_card_idx;
    uint wall_bounce;
    float _filler3;
};

struct Player {
    vec4 pos;
    vec4 rot;
    vec4 size;
    vec4 vel;
    vec4 dir;
    vec4 up;
    vec4 right;
};

const uint MAT_AIR = 0;
const uint MAT_STONE = 1;
const uint MAT_OOB = 2;
const uint MAT_DIRT = 3;
const uint MAT_GRASS = 4;
const uint MAT_PROJECTILE = 5;
const uint MAT_ICE = 6;
const uint MAT_GLASS = 7;
const uint MAT_PLAYER = 8;
const uint[] material_damage_threshhold = {0, 10, 0, 5, 5, 0, 5, 10};

uint get_index(ivec3 global_pos, uvec3 render_size) {
    uvec3 chunk_pos = (global_pos / CHUNK_SIZE) % render_size;
    uvec3 pos_in_chunk = global_pos % CHUNK_SIZE;
    uint chunk_idx = chunk_pos.x * render_size.y * render_size.z + chunk_pos.y * render_size.z + chunk_pos.z;
    uint idx_in_chunk = pos_in_chunk.x * CHUNK_SIZE * CHUNK_SIZE + pos_in_chunk.y * CHUNK_SIZE + pos_in_chunk.z;
    return chunk_idx * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE + idx_in_chunk;
}

uint get_chunk_index(ivec3 global_pos, uvec3 render_size) {
    uvec3 chunk_pos = (global_pos / CHUNK_SIZE) % render_size;
    uint chunk_idx = chunk_pos.x * render_size.y * render_size.z + chunk_pos.y * render_size.z + chunk_pos.z;
    return chunk_idx;
}

vec3 quat_transform(vec4 q, vec3 v) {
    return v + 2.*cross( q.xyz, cross( q.xyz, v ) + q.w*v ); 
}

vec4 quat_inverse(vec4 q) {
    return vec4(-q.xyz, q.w) / dot(q, q);
}

vec4 quaternion_from_arc(vec3 src, vec3 dst) {
    float mag_avg = sqrt(dot(src, src) * dot(dst, dst));
    float dotprod = dot(src, dst);
    if (dotprod == mag_avg) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    } else if (dotprod == -mag_avg) {
        vec3 v = cross(vec3(1.0, 0.0, 0.0), src);
        if(v == vec3(0.0)) {
            v = v = cross(vec3(0.0, 1.0, 0.0), src);
        }
        v = normalize(v);
        return vec4(v, radians(180) / 2.0);
    } else {
        return normalize(vec4(cross(src, dst), mag_avg + dotprod));
    }
}