#version 450
layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, r32ui) uniform uimage3D chunks_image;
layout(set = 0, binding = 1) uniform SliceData {
    int index;
    uint component;
} slice_data;

layout(set = 0, binding = 2) uniform SimData {
    uvec3 render_size;
    uvec3 start_pos;
} sim_data;

void main() {
    ivec3 coords = ivec3(0);
    coords[slice_data.component] = slice_data.index % int(sim_data.render_size[slice_data.component]);
    coords[(slice_data.component + 1) % 3] = int(gl_GlobalInvocationID.x);
    coords[(slice_data.component + 2) % 3] = int(gl_GlobalInvocationID.y);
    imageStore(chunks_image, coords, uvec4(0));
}