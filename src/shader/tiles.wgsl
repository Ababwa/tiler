@group(0) @binding(0) var<uniform> window_size: vec2u;
@group(0) @binding(1) var<uniform> offset: vec2i;
@group(0) @binding(2) var<uniform> tile_size: vec2u;
@group(0) @binding(3) var<uniform> columns: u32;
@group(0) @binding(4) var<uniform> num_tiles: u32;

struct VertexOutput {
	@builtin(position) position: vec4f,
	@location(0) region_uv: vec2f,
};

@vertex
fn vs_main(@location(0) vert: vec2u) -> VertexOutput {
	var rows = (num_tiles + columns - 1) / columns;
	var region_size = vec2u(columns, rows) * tile_size;
	var region_uv = vert * region_size;
	var checked_offset = select(offset, vec2i(0), all(offset == vec2i(0x7FFFFFFF)));
	var pos = vec2f((vec2i(region_uv) - checked_offset) * 2) / vec2f(window_size);
	return VertexOutput(vec4f(pos.x - 1, 1 - pos.y, 0, 1), vec2f(region_uv));
}

//vec4 to align 16
//each u32 is two 16-bit tiles
//1024 tiles total
//tile_offset bits:
//AAAAAAAVVB
//A: offset into tiles array
//V: element in vec4
//B: use upper 16 bits
//tile bits:
//SVUTTTTTTTTTTTTT
//S: swap u and v coords
//V: reverse v coord
//U: reverse u coord
//T: texture array index
@group(0) @binding(5) var<uniform> tiles: array<vec4u, 128>;
@group(0) @binding(6) var tile_array: texture_2d_array<f32>;

fn get_color(vertex: VertexOutput, region_uv_u: vec2u) -> vec4f {
	var tile_grid_pos = region_uv_u / tile_size;
	var tile_offset = tile_grid_pos.y * columns + tile_grid_pos.x;
	if tile_offset < num_tiles {
		var tile = (tiles[tile_offset / 8][(tile_offset / 2) % 4] >> ((tile_offset % 2) * 16)) & 0xFFFF;
		var tile_index = tile & 0x1FFF;
		if tile_index == 0x1FFF {
			return vec4f(0);
		} else {
			var tile_uv = region_uv_u % tile_size;
			tile_uv = select(tile_uv, tile_size - tile_uv - 1, vec2((tile & 0x4000) != 0, (tile & 0x8000) != 0));
			tile_uv = select(tile_uv, tile_uv.yx, tile_size.x == tile_size.y && (tile & 0x2000) != 0);
			return textureLoad(tile_array, tile_uv, tile_index, 0);
		}
	} else {
		return vec4f(0);
	}
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4f {
	var region_uv_u = vec2u(vertex.region_uv);
	var color = get_color(vertex, region_uv_u);
	if all(offset == vec2i(0x7FFFFFFF)) {
		return color;
	} else {
		var background = vec3f(f32(((region_uv_u.x + region_uv_u.y) % 34) / 17) * 0.02 + 0.02);
		return vec4f(color.a * color.rgb + (1 - color.a) * background, 1);
	}
}
