#version 450
#include <common.glsl>

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Projectiles { Projectile projectiles[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    projectiles[idx] = Projectile(
        vec4(projectiles[idx].pos.xyz + projectiles[idx].vel.xyz, 1.0),
        projectiles[idx].dir,
        projectiles[idx].size,
        projectiles[idx].vel
    );
}