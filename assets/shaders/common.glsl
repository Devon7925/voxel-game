const uint CHUNK_SIZE = 16;

struct Projectile {
    vec4 pos;
    ivec4 chunk_update_pos;
    vec4 dir;
    vec4 size;
    float vel;
    float health;
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