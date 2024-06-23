@group(0) @binding(0) var<uniform> image_size: vec2u;
@group(0) @binding(1) var<uniform> window_size: vec2u;
@group(0) @binding(2) var<uniform> image_offset: vec2i;

struct VertexOutput {
	@builtin(position) position: vec4f,
	@location(0) uv: vec2f,
};

@vertex
fn vs_main(@location(0) vert: vec2u) -> VertexOutput {
	var uv = vert * image_size;
	var pos = vec2f((vec2i(uv) + image_offset) * 2) / vec2f(window_size);
	return VertexOutput(vec4f(pos.x - 1, 1 - pos.y, 0, 1), vec2f(uv));
}

@group(0) @binding(3) var<uniform> tile_size: vec2u;
@group(0) @binding(4) var<uniform> grid_size: vec2u;
@group(0) @binding(5) var<uniform> grid_offset: vec2u;
@group(0) @binding(6) var<uniform> gap: vec2u;
@group(0) @binding(7) var image: texture_2d<f32>;

fn grid(uv: vec2f) -> bool {
	var grid_region_max = vec2f(grid_size * tile_size + (grid_size - 1) * gap);
	var grid_region_uv = uv - vec2f(grid_offset);
	if all(grid_region_uv >= vec2f(0)) && all(grid_region_uv < grid_region_max) {
		var cell_uv = grid_region_uv % vec2f(tile_size + gap);
		var tile_size_f = vec2f(tile_size);
		var on_line = cell_uv < vec2f(1) || (cell_uv >= (tile_size_f - 1) && cell_uv < tile_size_f);
		var in_tile = cell_uv < tile_size_f;
		return (on_line.x && in_tile.y) || (on_line.y && in_tile.x);
	} else {
		return false;
	}
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4f {
	if grid(vertex.uv) {
		return vec4f(1);
	} else {
		var uv_u = vec2u(vertex.uv);
		var background = vec3f(f32(((uv_u.x + uv_u.y) % 34) / 17) * 0.02 + 0.02);
		var color = textureLoad(image, uv_u, 0);
		return vec4f(color.a * color.rgb + (1 - color.a) * background, 1);
	}
}
