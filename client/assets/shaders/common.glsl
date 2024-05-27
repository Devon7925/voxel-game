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
const uint MAT_AIR_OOB = 9;
const uint[] material_damage_threshhold = { 0, 10, 0, 5, 5, 1, 5, 10, 0 };

uvec4 get_indicies(uvec3 global_pos, uvec3 render_size) {
    uvec3 chunk_pos = (global_pos >> CHUNK_MAGNITUDE) % render_size;
    uvec3 pos_in_chunk = global_pos & POS_IN_CHUNK_MASK;
    uint idx_in_chunk = pos_in_chunk.x * CHUNK_SIZE * CHUNK_SIZE + pos_in_chunk.y * CHUNK_SIZE + pos_in_chunk.z;
    return uvec4(chunk_pos, idx_in_chunk);
}

vec3 quat_transform(vec4 q, vec3 v) {
    return v + 2. * cross(q.xyz, cross(q.xyz, v) + q.w * v);
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
        if (v == vec3(0.0)) {
            v = v = cross(vec3(0.0, 1.0, 0.0), src);
        }
        v = normalize(v);
        return vec4(v, radians(180) / 2.0);
    } else {
        return normalize(vec4(cross(src, dst), mag_avg + dotprod));
    }
}

uvec3 pcg3d(uvec3 v) {
    v = v * 1664525u + 1013904223u;

    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;

    v ^= v >> 16u;

    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;

    return v;
}

ivec4 pcg4d(ivec4 v)
{
    v = v * 1664525 + 1013904223;

    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;

    v ^= v >> 16;

    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;

    return v;
}

vec4 voronoise(in vec3 p, float u, float v)
{
    float k = 1.0 + 63.0 * pow(1.0 - v, 6.0);

    ivec4 i = ivec4(p, 0);
    vec3 f = fract(p);

    vec2 a = vec2(0.0, 0.0);
    vec3 dir = vec3(0.0);
    for (int z = -2; z <= 2; z++)
        for (int y = -2; y <= 2; y++)
            for (int x = -2; x <= 2; x++)
            {
                vec3 g = vec3(x, y, z);
                vec4 hash = vec4(pcg4d(i + ivec4(x, y, z, 0)) & 0xFF) / 128.0 - 1.0;
                vec4 o = hash * vec4(vec3(u), 1.0);
                vec3 d = g - f + o.xyz;
                float w = pow(1.0 - smoothstep(0.0, 1.414, length(d)), k);
                a += vec2(o.w * w, w);
                dir += d * (2.0 * o.w - 1.0) * w;
            }

    return vec4(dir, a.x) / a.y;
}

vec4 grad_noise( in vec3 x )
{
    // grid
    uvec3 p = uvec3(floor(x));
    vec3 w = fract(x);
    
    // quintic interpolant
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec3 du = 30.0*w*w*(w*(w-2.0)+1.0);
    
    // gradients
    vec3 ga = vec3(pcg3d( p+uvec3(0,0,0) ) & 0xFF) / 128.0 - 1.0;
    vec3 gb = vec3(pcg3d( p+uvec3(1,0,0) ) & 0xFF) / 128.0 - 1.0;
    vec3 gc = vec3(pcg3d( p+uvec3(0,1,0) ) & 0xFF) / 128.0 - 1.0;
    vec3 gd = vec3(pcg3d( p+uvec3(1,1,0) ) & 0xFF) / 128.0 - 1.0;
    vec3 ge = vec3(pcg3d( p+uvec3(0,0,1) ) & 0xFF) / 128.0 - 1.0;
    vec3 gf = vec3(pcg3d( p+uvec3(1,0,1) ) & 0xFF) / 128.0 - 1.0;
    vec3 gg = vec3(pcg3d( p+uvec3(0,1,1) ) & 0xFF) / 128.0 - 1.0;
    vec3 gh = vec3(pcg3d( p+uvec3(1,1,1) ) & 0xFF) / 128.0 - 1.0;
    
    // projections
    float va = dot( ga, w-vec3(0.0,0.0,0.0) );
    float vb = dot( gb, w-vec3(1.0,0.0,0.0) );
    float vc = dot( gc, w-vec3(0.0,1.0,0.0) );
    float vd = dot( gd, w-vec3(1.0,1.0,0.0) );
    float ve = dot( ge, w-vec3(0.0,0.0,1.0) );
    float vf = dot( gf, w-vec3(1.0,0.0,1.0) );
    float vg = dot( gg, w-vec3(0.0,1.0,1.0) );
    float vh = dot( gh, w-vec3(1.0,1.0,1.0) );
	
    // interpolation
    float v = va + 
              u.x*(vb-va) + 
              u.y*(vc-va) + 
              u.z*(ve-va) + 
              u.x*u.y*(va-vb-vc+vd) + 
              u.y*u.z*(va-vc-ve+vg) + 
              u.z*u.x*(va-vb-ve+vf) + 
              u.x*u.y*u.z*(-va+vb+vc-vd+ve-vf-vg+vh);
              
    vec3 d = ga + 
             u.x*(gb-ga) + 
             u.y*(gc-ga) + 
             u.z*(ge-ga) + 
             u.x*u.y*(ga-gb-gc+gd) + 
             u.y*u.z*(ga-gc-ge+gg) + 
             u.z*u.x*(ga-gb-ge+gf) + 
             u.x*u.y*u.z*(-ga+gb+gc-gd+ge-gf-gg+gh) +   
             
            du * (vec3(vb-va,vc-va,ve-va) + 
                   u.yzx*vec3(va-vb-vc+vd,va-vc-ve+vg,va-vb-ve+vf) + 
                   u.zxy*vec3(va-vb-ve+vf,va-vb-vc+vd,va-vc-ve+vg) + 
                   u.yzx*u.zxy*(-va+vb+vc-vd+ve-vf-vg+vh) );
                   
    return vec4( d, v );                   
}
