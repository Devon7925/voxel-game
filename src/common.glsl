const uint CHUNK_SIZE = 16;

uint get_index(ivec3 global_pos, uvec3 render_size) {
    uvec3 chunk_pos = (global_pos / CHUNK_SIZE) % render_size;
    uvec3 pos_in_chunk = global_pos % CHUNK_SIZE;
    uint chunk_idx = chunk_pos.x * render_size.y * render_size.z + chunk_pos.y * render_size.z + chunk_pos.z;
    uint idx_in_chunk = pos_in_chunk.x * CHUNK_SIZE * CHUNK_SIZE + pos_in_chunk.y * CHUNK_SIZE + pos_in_chunk.z;
    return chunk_idx * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE + idx_in_chunk;
}