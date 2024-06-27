const uint CHUNK_MAGNITUDE = 4;
const uint CHUNK_SIZE = 1 << CHUNK_MAGNITUDE;
const uint CHUNK_VOLUME = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
const uint POS_IN_CHUNK_MASK = CHUNK_SIZE - 1;

const float PI = 3.14159265359;

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
    uint is_from_head;
    uint should_collide_with_terrain;
    uvec3 _filler;
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
const uint MAT_WATER = 7;
const uint MAT_PLAYER = 8;
const uint MAT_AIR_OOB = 9;
const uint MAT_WOOD = 10;
const uint MAT_LEAF = 11;
const uint MAT_UNBREAKABLE = 12;
const uint[] material_damage_threshhold = { 0, 10, 0, 5, 5, 1, 5, 1, 0, 8, 5, 5 };

struct PhysicsProperties {
    bool is_fluid;
    bool is_data_damage;
    bool is_data_standard_distance;
};

const PhysicsProperties physics_properties[] = {
    PhysicsProperties(true, false, false), //MAT_AIR
    PhysicsProperties(false, true, false), //MAT_STONE
    PhysicsProperties(false, false, false), //MAT_OOB
    PhysicsProperties(false, true, false), //MAT_DIRT
    PhysicsProperties(false, true, false), //MAT_GRASS
    PhysicsProperties(false, false, false), //MAT_PROJECTILE
    PhysicsProperties(false, true, false), //MAT_ICE
    PhysicsProperties(true, false, true), //MAT_WATER
    PhysicsProperties(false, false, false), //MAT_PLAYER
    PhysicsProperties(false, false, false), //MAT_AIR_OOB
    PhysicsProperties(false, true, false), //MAT_WOOD
    PhysicsProperties(false, false, true), //MAT_LEAF
    PhysicsProperties(false, false, false), //MAT_UNBREAKABLE
    };

const bool is_transparent[] = {
    true,  //MAT_AIR
    false,  //MAT_STONE
    false,  //MAT_OOB
    false,  //MAT_DIRT
    false,  //MAT_GRASS
    false,  //MAT_PROJECTILE
    true,  //MAT_ICE
    true,  //MAT_WATER
    false,  //MAT_PLAYER
    true,  //MAT_AIR_OOB
    false,  //MAT_WOOD
    true,  //MAT_LEAF
    false,  //MAT_UNBREAKABLE
    };

struct HeightData {
    float offset;
    vec3 scale;
    float impact;
    vec3 movement;
};

const HeightData height_data[] = {
    HeightData(0.0, vec3(0.0), 0.0, vec3(0.0)), //MAT_AIR
    HeightData(0.2, vec3(2.0), 0.75, vec3(0.0)), //MAT_STONE
    HeightData(0.0, vec3(0.0), 0.0, vec3(0.0)), //MAT_OOB
    HeightData(0.4, vec3(7.0), 0.3, vec3(0.0)), //MAT_DIRT
    HeightData(0.6, vec3(35.0, 10.0, 35.0), 0.75, vec3(0.0)), //MAT_GRASS
    HeightData(0.0, vec3(0.0), 0.0, vec3(0.0)), //MAT_PROJECTILE
    HeightData(0.7, vec3(1.7), 0.45, vec3(0.0)), //MAT_ICE
    HeightData(0.8, vec3(1.0), 0.3, vec3(0.25, -0.7, 0.2)), //MAT_WATER
    HeightData(0.0, vec3(0.0), 0.0, vec3(0.0)), //MAT_PLAYER
    HeightData(0.0, vec3(0.0), 0.0, vec3(0.0)), //MAT_AIR_OOB
    HeightData(0.2, vec3(10.0, 2.5, 10.0), 0.75, vec3(0.0)), //MAT_WOOD
    HeightData(0.7, vec3(1.7), 0.45, vec3(0.0)), //MAT_LEAF
    HeightData(0.2, vec3(2.0), 0.25, vec3(0.0)), //MAT_UNBREAKABLE
    };

struct MaterialNoiseLayer {
    vec3 scale;
    float normal_impact;
    float roughness_impact;
    float transparency_impact;
    vec3 movement;
    vec3 layer_color;
};

struct MaterialRenderProps {
    MaterialNoiseLayer layers[3];
    float ior;
    float roughness;
    float transparency;
    float depth_transparency;
    vec3 color;
};

const MaterialRenderProps material_render_props[] = {
    // AIR
    MaterialRenderProps(MaterialNoiseLayer[3](
            MaterialNoiseLayer(vec3(0.0), 0.0, 0.0, 0.0, vec3(0.0), vec3(0.0)),
            MaterialNoiseLayer(vec3(0.0), 0.0, 0.0, 0.0, vec3(0.0), vec3(0.0)),
            MaterialNoiseLayer(vec3(0.0), 0.0, 0.0, 0.0, vec3(0.0), vec3(0.0))
        ), 0.0, 0.0, 1.0, 1.0, vec3(0.0)),
    // STONE
    MaterialRenderProps(MaterialNoiseLayer[3](
            MaterialNoiseLayer(vec3(2.0), 0.35, 0.1, 0.0, vec3(0.0), vec3(0.25)),
            MaterialNoiseLayer(vec3(20.0), 0.2, 0.2, 0.0, vec3(0.0), vec3(0.05)),
            MaterialNoiseLayer(vec3(0.5), 0.05, 0.0, 0.0, vec3(0.0), vec3(0.05))
        ), 0.04, 0.35, 0.0, 0.0, vec3(0.55)),
    // OOB
    MaterialRenderProps(MaterialNoiseLayer[3](
            MaterialNoiseLayer(vec3(0.0), 0.0, 0.0, 0.0, vec3(0.0), vec3(0.0)),
            MaterialNoiseLayer(vec3(0.0), 0.0, 0.0, 0.0, vec3(0.0), vec3(0.0)),
            MaterialNoiseLayer(vec3(0.0), 0.0, 0.0, 0.0, vec3(0.0), vec3(0.0))
        ), 0.0, 0.0, 1.0, 1.0, vec3(0.0)),
    // DIRT
    MaterialRenderProps(MaterialNoiseLayer[3](
            MaterialNoiseLayer(vec3(7.0), 0.2, 0.0, -0.1, vec3(0.0), vec3(0.15, 0.025, -0.1)),
            MaterialNoiseLayer(vec3(20.0), 0.2, 0.2, 0.0, vec3(0.0), vec3(0.05)),
            MaterialNoiseLayer(vec3(0.5), 0.05, 0.0, 0.0, vec3(0.0), vec3(0.05))
        ), 0.02, 0.75, 0.0, 0.0, vec3(0.35, 0.225, 0.1)),
    // GRASS
    MaterialRenderProps(MaterialNoiseLayer[3](
            MaterialNoiseLayer(vec3(7.0), 0.2, -0.1, 0.0, vec3(0.0), vec3(0.07, 0.1, 0.07)),
            MaterialNoiseLayer(vec3(35.0, 10.0, 35.0), 0.6, -0.2, 0.0, vec3(0.0), vec3(0.15, 0.2, 0.15)),
            MaterialNoiseLayer(vec3(0.07), 0.0, 0.0, -0.2, vec3(0.0), vec3(0.1, 0.2, 0.025))
        ), 0.02, 0.8, 0.0, 0.0, vec3(0.17, 0.6, 0.2)),
    // PROJECTILE
    MaterialRenderProps(MaterialNoiseLayer[3](
            MaterialNoiseLayer(vec3(0.0), 0.0, 0.0, 0.0, vec3(0.0), vec3(0.0)),
            MaterialNoiseLayer(vec3(0.0), 0.0, 0.0, 0.0, vec3(0.0), vec3(0.0)),
            MaterialNoiseLayer(vec3(0.0), 0.0, 0.0, 0.0, vec3(0.0), vec3(0.0))
        ), 0.0, 1.0, 0.5, 0.5, vec3(1.0, 0.3, 0.3)),
    // ICE
    MaterialRenderProps(MaterialNoiseLayer[3](
            MaterialNoiseLayer(vec3(1.7), 0.2, 0.1, 0.1, vec3(0.0), vec3(0.05, 0.05, 0.175)),
            MaterialNoiseLayer(vec3(21.0), 0.1, 0.1, 0.05, vec3(0.0), vec3(0.05)),
            MaterialNoiseLayer(vec3(0.5), 0.05, 0.0, 0.05, vec3(0.0), vec3(0.05))
        ), 0.05, 0.35, 0.3, 0.3, vec3(0.7, 0.7, 0.925)),
    // WATER
    MaterialRenderProps(MaterialNoiseLayer[3](
            MaterialNoiseLayer(vec3(1.0), 0.20, 0.0, 0.0, vec3(0.25, -0.7, 0.2), vec3(0.0)),
            MaterialNoiseLayer(vec3(2.0), 0.10, 0.0, 0.0, vec3(0.375, -0.5, 0.475), vec3(0.0)),
            MaterialNoiseLayer(vec3(4.0), 0.05, 0.0, 0.0, vec3(0.5, -0.6, 0.5), vec3(0.0))
        ), 0.05, 0.25, 0.8, 0.85, vec3(0.25, 0.3, 0.6)),
    // PLAYER
    MaterialRenderProps(MaterialNoiseLayer[3](
            MaterialNoiseLayer(vec3(0.0), 0.0, 0.0, 0.0, vec3(0.0), vec3(0.0)),
            MaterialNoiseLayer(vec3(0.0), 0.0, 0.0, 0.0, vec3(0.0), vec3(0.0)),
            MaterialNoiseLayer(vec3(0.0), 0.0, 0.0, 0.0, vec3(0.0), vec3(0.0))
        ), 0.0, 0.2, 0.0, 0.0, vec3(0.8)),
    // AIR OOB
    MaterialRenderProps(MaterialNoiseLayer[3](
            MaterialNoiseLayer(vec3(0.0), 0.0, 0.0, 0.0, vec3(0.0), vec3(0.0)),
            MaterialNoiseLayer(vec3(0.0), 0.0, 0.0, 0.0, vec3(0.0), vec3(0.0)),
            MaterialNoiseLayer(vec3(0.0), 0.0, 0.0, 0.0, vec3(0.0), vec3(0.0))
        ), 0.0, 0.0, 1.0, 1.0, vec3(0.0)),
    // WOOD
    MaterialRenderProps(MaterialNoiseLayer[3](
            MaterialNoiseLayer(vec3(10.0, 2.5, 10.0), 0.17, 0.0, 0.0, vec3(0.0), vec3(0.25)),
            MaterialNoiseLayer(vec3(20.0, 5.0, 20.0), 0.1, 0.0, 0.0, vec3(0.0), vec3(0.05)),
            MaterialNoiseLayer(vec3(3.0), 0.12, 0.0, 0.0, vec3(0.0), vec3(0.05))
        ), 0.05, 0.8, 0.0, 0.0, vec3(0.37, 0.225, 0.1)),
    // LEAF
    MaterialRenderProps(MaterialNoiseLayer[3](
            MaterialNoiseLayer(vec3(1.7), 0.1, 0.0, 0.0, vec3(0.0), vec3(0.05, 0.175, 0.05)),
            MaterialNoiseLayer(vec3(9.5), 0.3, 0.2, -0.5, vec3(0.1, -0.2, 0.1), vec3(0.05, 0.1, 0.05)),
            MaterialNoiseLayer(vec3(3.5), 0.05, 0.0, 0.00, vec3(0.0), vec3(0.05))
        ), 0.05, 0.6, 0.4, 0.4, vec3(0.1, 0.6, 0.1)),
    
    // UNBREAKABLE
    MaterialRenderProps(MaterialNoiseLayer[3](
            MaterialNoiseLayer(vec3(2.0), 0.175, 0.1, 0.0, vec3(0.0), vec3(0.15)),
            MaterialNoiseLayer(vec3(20.0), 0.1, 0.2, 0.0, vec3(0.0), vec3(0.05)),
            MaterialNoiseLayer(vec3(0.5), 0.025, 0.0, 0.0, vec3(0.0), vec3(0.05))
        ), 0.04, 0.8, 0.0, 0.0, vec3(0.15)),
    };

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

uvec2 pcg2d(uvec2 v)
{
    v = v * 1664525u + 1013904223u;

    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;

    v = v ^ (v>>16u);

    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;

    v = v ^ (v>>16u);

    return v;
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

vec4 grad_noise(in vec3 x)
{
    // grid
    uvec3 p = uvec3(floor(x));
    vec3 w = fract(x);

    // quintic interpolant
    vec3 u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);
    vec3 du = 30.0 * w * w * (w * (w - 2.0) + 1.0);

    // gradients
    vec3 ga = vec3(pcg3d(p + uvec3(0, 0, 0)) & 0xFF) / 128.0 - 1.0;
    vec3 gb = vec3(pcg3d(p + uvec3(1, 0, 0)) & 0xFF) / 128.0 - 1.0;
    vec3 gc = vec3(pcg3d(p + uvec3(0, 1, 0)) & 0xFF) / 128.0 - 1.0;
    vec3 gd = vec3(pcg3d(p + uvec3(1, 1, 0)) & 0xFF) / 128.0 - 1.0;
    vec3 ge = vec3(pcg3d(p + uvec3(0, 0, 1)) & 0xFF) / 128.0 - 1.0;
    vec3 gf = vec3(pcg3d(p + uvec3(1, 0, 1)) & 0xFF) / 128.0 - 1.0;
    vec3 gg = vec3(pcg3d(p + uvec3(0, 1, 1)) & 0xFF) / 128.0 - 1.0;
    vec3 gh = vec3(pcg3d(p + uvec3(1, 1, 1)) & 0xFF) / 128.0 - 1.0;

    // projections
    float va = dot(ga, w - vec3(0.0, 0.0, 0.0));
    float vb = dot(gb, w - vec3(1.0, 0.0, 0.0));
    float vc = dot(gc, w - vec3(0.0, 1.0, 0.0));
    float vd = dot(gd, w - vec3(1.0, 1.0, 0.0));
    float ve = dot(ge, w - vec3(0.0, 0.0, 1.0));
    float vf = dot(gf, w - vec3(1.0, 0.0, 1.0));
    float vg = dot(gg, w - vec3(0.0, 1.0, 1.0));
    float vh = dot(gh, w - vec3(1.0, 1.0, 1.0));

    // interpolation
    float v = va +
            u.x * (vb - va) +
            u.y * (vc - va) +
            u.z * (ve - va) +
            u.x * u.y * (va - vb - vc + vd) +
            u.y * u.z * (va - vc - ve + vg) +
            u.z * u.x * (va - vb - ve + vf) +
            u.x * u.y * u.z * (-va + vb + vc - vd + ve - vf - vg + vh);

    vec3 d = ga +
            u.x * (gb - ga) +
            u.y * (gc - ga) +
            u.z * (ge - ga) +
            u.x * u.y * (ga - gb - gc + gd) +
            u.y * u.z * (ga - gc - ge + gg) +
            u.z * u.x * (ga - gb - ge + gf) +
            u.x * u.y * u.z * (-ga + gb + gc - gd + ge - gf - gg + gh) +
            du * (vec3(vb - va, vc - va, ve - va) +
                    u.yzx * vec3(va - vb - vc + vd, va - vc - ve + vg, va - vb - ve + vf) +
                    u.zxy * vec3(va - vb - ve + vf, va - vb - vc + vd, va - vc - ve + vg) +
                    u.yzx * u.zxy * (-va + vb + vc - vd + ve - vf - vg + vh));

    return vec4(d, v);
}
