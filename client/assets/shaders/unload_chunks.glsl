#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, r32ui) uniform uimage3D chunks_image;

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
} sim_data;

void main() {
    ivec3 coords = ivec3(0);
    coords[sim_data.unload_component] = sim_data.unload_index % int(sim_data.render_size[sim_data.unload_component]);
    coords[(sim_data.unload_component + 1) % 3] = int(gl_GlobalInvocationID.x);
    coords[(sim_data.unload_component + 2) % 3] = int(gl_GlobalInvocationID.y);
    imageStore(chunks_image, coords, uvec4(0));
}
