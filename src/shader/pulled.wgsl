@group(0) @binding(0) var<uniform> window_size: vec2u;
@group(0) @binding(1) var<uniform> tile_size: vec2u;
@group(0) @binding(2) var<uniform> pulled_size: vec2u;

struct VertexOutput {
	@builtin(position) position: vec4f,
	@location(0) region_uv: vec2f,
};

@vertex
fn vs_main(@location(0) vert: vec2u) -> VertexOutput {
	var region_uv = vert * tile_size * pulled_size;
	var pos = vec2f(region_uv * 2) / vec2f(window_size);
	return VertexOutput(vec4f(pos.x - 1, 1 - pos.y, 0, 1), vec2f(region_uv));
}

//same as tiles.wgsl
@group(0) @binding(3) var<uniform> tiles: array<vec4<u32>, 128>;
@group(0) @binding(4) var tile_array: texture_2d_array<f32>;

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4f {
	var region_uv_u = vec2u(vertex.region_uv);
	var tile_grid_pos = region_uv_u / tile_size;
	var tile_offset = tile_grid_pos.y * pulled_size.x + tile_grid_pos.x;
	var background = vec3f(f32(((region_uv_u.x + region_uv_u.y) % 34) / 17) * 0.02 + 0.02);
	var tile = (tiles[tile_offset / 8][(tile_offset / 2) % 4] >> ((tile_offset % 2) * 16)) & 0xFFFF;
	var tile_index = tile & 0x1FFF;
	if tile_index == 0x1FFF {
		return vec4f(background, 1);
	} else {
		var tile_uv = region_uv_u % tile_size;
		tile_uv = select(tile_uv, tile_size - tile_uv - 1, vec2((tile & 0x4000) != 0, (tile & 0x8000) != 0));
		tile_uv = select(tile_uv, tile_uv.yx, tile_size.x == tile_size.y && (tile & 0x2000) != 0);
		var tile_index = tile & 0x1FFF;
		var color = textureLoad(tile_array, tile_uv, tile_index, 0);
		return vec4f(color.a * color.rgb + (1 - color.a) * background, 1);
	}
}
