const uint CHUNK_MAGNITUDE = 4;
const uint CHUNK_SIZE = 1 << CHUNK_MAGNITUDE;
const uint CHUNK_VOLUME = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
const uint POS_IN_CHUNK_MASK = CHUNK_SIZE - 1;

struct Projectile {
    vec4 pos;
    uvec4 chunk_update_pos;
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
const uint[] material_damage_threshhold = {0, 10, 0, 5, 5, 1, 5, 10};

uvec2 get_indicies(uvec3 global_pos, uvec3 render_size) {
    uvec3 chunk_pos = (global_pos >> CHUNK_MAGNITUDE) % render_size;
    uvec3 pos_in_chunk = global_pos & POS_IN_CHUNK_MASK;
    uint chunk_idx = chunk_pos.x * render_size.y * render_size.z + chunk_pos.y * render_size.z + chunk_pos.z;
    uint idx_in_chunk = pos_in_chunk.x * CHUNK_SIZE * CHUNK_SIZE + pos_in_chunk.y * CHUNK_SIZE + pos_in_chunk.z;
    return uvec2(chunk_idx, idx_in_chunk);
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

ivec4 pcg4d(ivec4 v)
{
    v = v * 1664525 + 1013904223;
    
    v.x += v.y*v.w;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    v.w += v.y*v.z;
    
    v ^= v >> 16;
    
    v.x += v.y*v.w;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    v.w += v.y*v.z;
    
    return v;
}

vec4 voronoise( in vec3 p, float u, float v )
{
	float k = 1.0+63.0*pow(1.0-v,6.0);

    ivec4 i = ivec4(p, 0);
    vec3 f = fract(p);
    
	vec2 a = vec2(0.0,0.0);
    vec3 dir = vec3(0.0);
    for( int z=-2; z<=2; z++ )
    for( int y=-2; y<=2; y++ )
    for( int x=-2; x<=2; x++ )
    {
        vec3 g = vec3( x, y, z );
        vec4 hash = vec4(pcg4d(i + ivec4(x, y, z, 0)) & 0xFF) / 128.0 - 1.0;
		vec4 o = hash*vec4(vec3(u), 1.0);
		vec3 d = g - f + o.xyz;
		float w = pow( 1.0-smoothstep(0.0,1.414,length(d)), k );
		a += vec2(o.w*w,w);
        dir += d * (2.0*o.w - 1.0) * w;
    }
	
    return vec4(dir, a.x)/a.y;
}