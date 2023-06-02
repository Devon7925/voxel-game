uint get_index(ivec3 global_pos, uvec3 grid_size) {
    uvec3 pos = global_pos % grid_size;
    return uint(pos.x + pos.y * grid_size.x + pos.z * grid_size.x * grid_size.y);
}