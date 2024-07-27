@group(0) @binding(0) var<uniform> window_size: vec2u;
@group(0) @binding(1) var<uniform> offset: vec2i;
@group(0) @binding(2) var<uniform> tile_size: vec2u;
@group(0) @binding(3) var<uniform> select_pos1: vec2u;
@group(0) @binding(4) var<uniform> select_pos2: vec2u;

struct VertexOutput {
	@builtin(position) position: vec4f,
	@location(0) region_uv: vec2f,
	@location(1) region_size: vec2u,
};

@vertex
fn vs_main(@location(0) vert: vec2u) -> VertexOutput {
	var select_min = min(select_pos1, select_pos2);
	var region_size = (max(select_pos1, select_pos2) - select_min + 1) * tile_size;
	var region_uv = vert * region_size;
	var pos = vec2f((vec2i(region_uv + select_min * tile_size) - offset) * 2) / vec2f(window_size);
	return VertexOutput(vec4f(pos.x - 1, 1 - pos.y, 0, 1), vec2f(region_uv), region_size);
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4f {
	if any(vertex.region_uv < vec2f(1)) || any(vertex.region_uv >= vec2f(vertex.region_size - 1)) {
		return vec4f(1);
	} else {
		discard;
	}
}
