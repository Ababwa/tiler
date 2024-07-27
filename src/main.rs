mod ext;
mod gui;
mod key;
mod make;
mod rect;

use std::{ops::RangeInclusive, path::PathBuf, sync::mpsc::channel, time::Duration};
use egui_file_dialog::{DialogMode, FileDialog};
use glam::{ivec2, uvec2, vec2, DVec2, IVec2, UVec2};
use image::{ColorType, GenericImageView, ImageResult, Rgba, RgbaImage, SubImage};
use make::{BufferVal, PipelineAndLayout};
use rect::RectIter;
use wgpu::{
	BindGroup, BindGroupLayout, BindingResource, Buffer, BufferDescriptor, BufferUsages, Color, CommandEncoder,
	CommandEncoderDescriptor, Device, ImageCopyBuffer, ImageCopyTexture, ImageDataLayout, Maintain, MapMode,
	Origin3d, Queue, RenderPipeline, ShaderStages, TextureAspect, TextureFormat, TextureUsages, TextureView,
	TextureViewDescriptor, TextureViewDimension,
};
use winit::{
	event::{ElementState, MouseButton, MouseScrollDelta}, event_loop::EventLoopWindowTarget, keyboard::{KeyCode, ModifiersState}, window::Icon
};
use gui::Gui;
use ext::{AsBytes, IterPixels, Size, ToVec};

const PIXEL_BYTES: u32 = 4;
const MAX_TILES: usize = 1024;
const NULL_TILE: u16 = u16::MAX;
const DEFAULT_TILE_SIZE: u32 = 64;
const SQUARE: [[u16; 2]; 4] = [[0, 0], [1, 0], [0, 1], [1, 1]];
const SQUARE_VERTS: u32 = SQUARE.len() as u32;
const WRITABLE_UNIFORM: BufferUsages = BufferUsages::UNIFORM.union(BufferUsages::COPY_DST);

struct ColumnsDialog {
	new_columns: u32,
	pad: bool,
}

struct Split {
	image: RgbaImage,
	grid_size: BufferVal<UVec2>,
	grid_offset: BufferVal<UVec2>,
	gap: BufferVal<UVec2>,
	split_bind_group: BindGroup,
}

struct Tiles {
	pixels: Vec<Rgba<u8>>,
	tiles_bind_group: BindGroup,
	select_bind_group: BindGroup,
	pulled_bind_group: BindGroup,
	columns: BufferVal<u32>,
	select_pos1: BufferVal<UVec2>,
	select_pos2: BufferVal<UVec2>,
	tile_refs: BufferVal<Box<[u16; MAX_TILES]>>,
	num_tile_refs: BufferVal<u32>,
	pulled_tiles: BufferVal<Box<[u16; MAX_TILES]>>,
	pulled_size: BufferVal<UVec2>,
	columns_dialog: Option<ColumnsDialog>,
}

#[derive(Clone, Copy)]
enum Rotation {
	Cw,
	Ccw,
}

impl Tiles {
	fn index(&self, pos: UVec2) -> usize {
		(pos.y * self.columns.val + pos.x) as usize
	}
	
	fn tile_unchecked(&mut self, pos: UVec2) -> &mut u16 {
		&mut self.tile_refs.val[self.index(pos)]
	}
	
	fn tile_checked(&mut self, pos: UVec2) -> Option<&mut u16> {
		let index = self.index(pos);
		if index < self.num_tile_refs.val as usize {
			match &mut self.tile_refs.val[index] {
				&mut NULL_TILE => None,
				tile => Some(tile),
			}
		} else {
			None
		}
	}
	
	fn select_min(&self) -> UVec2 {
		self.select_pos1.val.min(self.select_pos2.val)
	}
	
	fn select_max(&self) -> UVec2 {
		self.select_pos1.val.max(self.select_pos2.val)
	}
	
	fn move_selection<F: Fn(&mut UVec2) + Copy>(&mut self, queue: &Queue, mod_fn: F) {
		self.select_pos1.modify_write(queue, mod_fn);
		self.select_pos2.modify_write(queue, mod_fn);
	}
	
	fn selection(&self) -> (UVec2, UVec2, usize, usize) {
		let min = self.select_min();
		let max = self.select_max();
		let size = max - min + 1;
		let start = self.index(min);
		let end = self.index(max) + 1;
		(min, size, start, end)
	}
	
	fn rotate(&mut self, offset: UVec2, size: UVec2, pos: UVec2, rot: Rotation) {
		let indices = [
			self.index(offset + pos),
			self.index(offset + uvec2(size.x - pos.y - 1, pos.x)),
			self.index(offset + uvec2(size.x - pos.x - 1, size.y - pos.y - 1)),
			self.index(offset + uvec2(pos.y, size.y - pos.x - 1)),
		];
		self.tile_refs.val.swap(indices[0], indices[1]);
		self.tile_refs.val.swap(indices[2], indices[3]);
		self.tile_refs.val.swap(indices[rot as usize], indices[rot as usize + 2]);
	}
	
	fn update_num_tile_refs(&mut self, queue: &Queue, mut end: usize) {
		if end as u32 >= self.num_tile_refs.val {
			while end != 0 && self.tile_refs.val[end - 1] == NULL_TILE {
				end -= 1;
			}
			self.num_tile_refs.set_write(queue, end as u32);
		}
	}
	
	fn has_pulled(&self) -> bool {
		self.pulled_size.val.element_product() != 0
	}
}

enum State {
	Split(Split),
	Tiles(Tiles, Option<Split>),
}

struct LoadedImage {
	window_size_buffer: Buffer,
	offset: BufferVal<IVec2>,
	tile_size: BufferVal<UVec2>,
	mouse_pos: DVec2,
	pan: Option<DVec2>,
	state: State,
}

struct Tiler {
	modifers: ModifiersState,
	file_dialog: FileDialog,
	error: Option<String>,
	square: Buffer,
	split_pal: PipelineAndLayout,
	tiles_pal: PipelineAndLayout,
	select_pal: PipelineAndLayout,
	pulled_pal: PipelineAndLayout,
	save_pipeline: RenderPipeline,
	loaded_image: Option<LoadedImage>,
}

fn new_split(
	device: &Device,
	queue: &Queue,
	split_bind_group_layout: &BindGroupLayout,
	window_size_buffer: &Buffer,
	offset: &BufferVal<IVec2>,
	tile_size: &BufferVal<UVec2>,
	image: RgbaImage,
) -> Split {
	let image_size = image.size();
	let grid_size = BufferVal::new(device, image_size / tile_size.val);
	let grid_offset = BufferVal::new(device, UVec2::ZERO);
	let gap = BufferVal::new(device, UVec2::ZERO);
	let split_bind_group = make::bind_group(device, split_bind_group_layout, vec![
		make::buffer(device, image_size.as_bytes(), BufferUsages::UNIFORM).as_entire_binding(),
		window_size_buffer.as_entire_binding(),
		offset.entry(),
		tile_size.entry(),
		grid_size.entry(),
		grid_offset.entry(),
		gap.entry(),
		BindingResource::TextureView(&make::bindable_texture_view(device, queue, image_size, 1, &image)),
	]);
	Split { image, grid_size, grid_offset, gap, split_bind_group }
}

impl Tiler {
	fn load_image(&mut self, device: &Device, queue: &Queue, window_size: UVec2, image: RgbaImage) {
		let image_size = image.size();
		if let Some(LoadedImage { window_size_buffer, offset, tile_size, state, .. }) = &mut self.loaded_image {
			if image_size.cmplt(tile_size.val).any() {
				self.error = Some("Image smaller than tile size".to_string());
			} else {
				let new_split = new_split(
					device, queue, &self.split_pal.layout, window_size_buffer, offset, tile_size, image,
				);
				match state {
					State::Split(split) => *split = new_split,
					State::Tiles(_, split) => *split = Some(new_split),
				}
			}
		} else {
			let window_size_buffer = make::buffer(device, window_size.as_bytes(), WRITABLE_UNIFORM);
			let offset = BufferVal::new(device, ivec2((image_size.x as i32 - window_size.x as i32) / 2, 0));
			let tile_size = BufferVal::new(device, UVec2::splat(DEFAULT_TILE_SIZE).min(image_size));
			let split = new_split(
				device, queue, &self.split_pal.layout, &window_size_buffer, &offset, &tile_size, image,
			);
			self.loaded_image = Some(LoadedImage {
				window_size_buffer, offset, tile_size, mouse_pos: DVec2::ZERO, pan: None, state: State::Split(split),
			});
		}
	}
}

fn window<R>(ctx: &egui::Context, title: &str, contents: impl FnOnce(&mut egui::Ui) -> R) -> R {
	egui::Window::new(title).collapsible(false).resizable(false).show(ctx, contents).unwrap().inner.unwrap()
}

fn drag_value(ui: &mut egui::Ui, val: &mut u32, bounds: RangeInclusive<u32>, label: &str) -> bool {
	ui.add(egui::DragValue::new(val).clamp_range(bounds).prefix(label)).changed()
}

fn drag_value_xy(ui: &mut egui::Ui, val: &mut UVec2, lower: u32, upper: UVec2) -> bool {
	drag_value(ui, &mut val.x, lower..=upper.x, "X: ") | drag_value(ui, &mut val.y, lower..=upper.y, "Y: ")
}

fn unique_tile<'a, I: Iterator<Item = &'a [Rgba<u8>]>>(mut tiles: I, new_tile: &SubImage<&RgbaImage>) -> bool {
	tiles.all(|tile| tile.iter().copied().zip(new_tile.iter_pixels()).any(|(a, b)| a != b))
}

fn interesting_tile(tile: &SubImage<&RgbaImage>) -> bool {
	let first_pixel = tile.get_pixel(0, 0);
	tile.iter_pixels().any(|pixel| pixel[3] != 0) && (
		tile.dimensions() == (1, 1) || tile.iter_pixels().any(|pixel| pixel != first_pixel)
	)
}

fn split_param(
	queue: &Queue,
	ui: &mut egui::Ui,
	label: &str,
	buffer_val: &mut BufferVal<UVec2>,
	lower: u32,
	upper: UVec2,
) {
	ui.horizontal(|ui| {
		ui.label(label);
		if drag_value_xy(ui, &mut buffer_val.val, lower, upper) {
			buffer_val.write(queue);
		}
	});
}

//after the first image is split, the tile size cannot be changed
enum TileSizeParam<'a> {
	Mutable(&'a mut BufferVal<UVec2>),
	Const(UVec2),
}

fn split_window(
	queue: &Queue,
	ctx: &egui::Context,
	image_size: UVec2,
	tile_size_param: TileSizeParam,
	grid_size: &mut BufferVal<UVec2>,
	grid_offset: &mut BufferVal<UVec2>,
	gap: &mut BufferVal<UVec2>,
) -> bool {
	window(ctx, "Split Tiles", |ui| {
		let tile_size = match tile_size_param {
			TileSizeParam::Mutable(tile_size) => {
				split_param(
					queue, ui, "Tile Size", tile_size, 1,
					(image_size - grid_offset.val - (grid_size.val - 1) * gap.val) / grid_size.val,
				);
				tile_size.val
			},
			TileSizeParam::Const(tile_size) => tile_size,
		};
		split_param(
			queue, ui, "Grid Size", grid_size, 1,
			(image_size - grid_offset.val + gap.val) / (tile_size + gap.val),
		);
		split_param(
			queue, ui, "Offset", grid_offset, 0,
			image_size - (grid_size.val - 1) * gap.val - grid_size.val * tile_size,
		);
		split_param(
			queue, ui, "Gap", gap, 0,
			(image_size - grid_offset.val - grid_size.val * tile_size) / UVec2::ONE.max(grid_size.val - 1),
		);
		ui.button("Ok").clicked()
	})
}

fn columns_window(ctx: &egui::Context, min: u32, columns: &mut u32, pad: &mut bool) -> Option<bool> {
	window(ctx, "Columns", |ui| {
		drag_value(ui, columns, min..=1000, "");
		ui.checkbox(pad, "Maintain tile positions");
		match ui.horizontal(|ui| (ui.button("Ok").clicked(), ui.button("Cancel").clicked())).inner {
			(true, _) => Some(true),
			(_, true) => Some(false),
			_ => None,
		}
	})
}

fn pad_columns(
	new_columns: u32, old_columns: u32, tile_refs: &[u16; MAX_TILES], num_tile_refs: u32,
) -> (Box<[u16; MAX_TILES]>, u32) {
	let mut new_tiles = vec![NULL_TILE; MAX_TILES];
	let mut new_index = 0;
	let mut old_index = 0;
	if new_columns > old_columns {
		loop {
			if (new_index % new_columns) < old_columns {
				new_tiles[new_index as usize] = tile_refs[old_index as usize];
				old_index += 1;
				new_index += 1;
				if old_index == num_tile_refs {
					break;
				}
			} else {
				new_index += 1;
			}
		}
	} else {
		let mut outside = vec![];
		loop {
			if (old_index % old_columns) < new_columns {
				new_tiles[new_index as usize] = tile_refs[old_index as usize];
				new_index += 1;
			} else {
				match tile_refs[old_index as usize] {
					NULL_TILE => {},
					tile => outside.push(tile),
				}
			}
			old_index += 1;
			if old_index == num_tile_refs {
				break;
			}
		}
		for tile in outside {
			new_tiles[new_index as usize] = tile;
			new_index += 1;
		}
	}
	(new_tiles.into_boxed_slice().try_into().unwrap(), new_index)
}

fn split_image(
	image: &RgbaImage,
	tile_size: UVec2,
	grid_size: UVec2,
	grid_offset: UVec2,
	gap: UVec2,
	pixels: &mut Vec<Rgba<u8>>,
) -> u32 {
	let num_pixels = tile_size.element_product() as usize;
	let mut num_new_tiles = 0;
	for grid_pos in RectIter::new(grid_size) {
		let tile_pos = grid_offset + grid_pos * (tile_size + gap);
		let new_tile = image.view(tile_pos.x, tile_pos.y, tile_size.x, tile_size.y);
		if interesting_tile(&new_tile) && unique_tile(pixels.chunks_exact(num_pixels), &new_tile) {
			pixels.extend(new_tile.iter_pixels());
			num_new_tiles += 1;
		}
	}
	num_new_tiles
}

fn first_split(
	device: &Device,
	queue: &Queue,
	tiles_layout: &BindGroupLayout,
	select_layout: &BindGroupLayout,
	pulled_layout: &BindGroupLayout,
	window_size_buffer: &Buffer,
	offset: &BufferVal<IVec2>,
	tile_size: &BufferVal<UVec2>,
	image: &RgbaImage,
	grid_size: UVec2,
	grid_offset: UVec2,
	gap: UVec2,
) -> Tiles {
	let mut pixels = Vec::with_capacity(grid_size.element_product() as usize);
	let num_tiles = split_image(image, tile_size.val, grid_size, grid_offset, gap, &mut pixels);
	let mut tile_refs = vec![NULL_TILE; MAX_TILES];
	for i in 0..num_tiles as usize {
		tile_refs[i] = i as u16;
	}
	let pulled_tiles = vec![NULL_TILE; MAX_TILES];
	let tiles = make::bindable_texture_view(device, queue, tile_size.val, num_tiles, pixels.as_bytes());
	let tile_refs = BufferVal::new(device, tile_refs.into_boxed_slice().try_into().unwrap());
	let pulled_tiles = BufferVal::new(device, pulled_tiles.into_boxed_slice().try_into().unwrap());
	let pulled_size = BufferVal::new(device, UVec2::ZERO);
	let columns = BufferVal::new(device, grid_size.x);
	let num_tile_refs = BufferVal::new(device, num_tiles);
	let select_pos1 = BufferVal::new(device, UVec2::ZERO);
	let select_pos2 = BufferVal::new(device, UVec2::ZERO);
	let tiles_bind_group = make::bind_group(
		device,
		tiles_layout,
		vec![
			window_size_buffer.as_entire_binding(),
			offset.entry(),
			tile_size.entry(),
			columns.entry(),
			num_tile_refs.entry(),
			tile_refs.entry(),
			BindingResource::TextureView(&tiles),
		],
	);
	let select_bind_group = make::bind_group(
		device,
		select_layout,
		vec![
			window_size_buffer.as_entire_binding(),
			offset.entry(),
			tile_size.entry(),
			select_pos1.entry(),
			select_pos2.entry(),
		],
	);
	let pulled_bind_group = make::bind_group(
		device,
		pulled_layout,
		vec![
			window_size_buffer.as_entire_binding(),
			tile_size.entry(),
			pulled_size.entry(),
			pulled_tiles.entry(),
			BindingResource::TextureView(&tiles),
		],
	);
	Tiles {
		pixels,
		tiles_bind_group,
		select_bind_group,
		pulled_bind_group,
		columns,
		select_pos1,
		select_pos2,
		num_tile_refs,
		tile_refs,
		pulled_tiles,
		pulled_size,
		columns_dialog: None,
	}
}

fn next_split(
	device: &Device,
	queue: &Queue,
	tiles_layout: &BindGroupLayout,
	pulled_layout: &BindGroupLayout,
	window_size_buffer: &Buffer,
	offset: &BufferVal<IVec2>,
	tile_size: &BufferVal<UVec2>,
	pixels: &mut Vec<Rgba<u8>>,
	tiles_bind_group: &mut BindGroup,
	pulled_bind_group: &mut BindGroup,
	num_tile_refs: &mut BufferVal<u32>,
	tile_refs: &mut BufferVal<Box<[u16; 1024]>>,
	columns: &BufferVal<u32>,
	pulled_size: &BufferVal<UVec2>,
	pulled_tiles: &BufferVal<Box<[u16; 1024]>>,
	image: &RgbaImage,
	grid_size: UVec2,
	grid_offset: UVec2,
	gap: UVec2,
) {
	let num_new_tiles = split_image(image, tile_size.val, grid_size, grid_offset, gap, pixels);
	let num_tiles = pixels.len() as u32 / tile_size.val.element_product();
	let old_num_tile_refs = num_tile_refs.val as usize;
	for i in 0..num_new_tiles as usize {
		tile_refs.val[old_num_tile_refs + i] = (num_tiles - num_new_tiles + i as u32) as u16;
	}
	let new_num_tile_refs = num_tile_refs.val + num_new_tiles;
	tile_refs.write_range(queue, old_num_tile_refs, new_num_tile_refs as usize);
	num_tile_refs.set_write(queue, new_num_tile_refs);
	let tiles = make::bindable_texture_view(device, queue, tile_size.val, num_tiles, pixels.as_bytes());
	*tiles_bind_group = make::bind_group(
		device,
		tiles_layout,
		vec![
			window_size_buffer.as_entire_binding(),
			offset.entry(),
			tile_size.entry(),
			columns.entry(),
			num_tile_refs.entry(),
			tile_refs.entry(),
			BindingResource::TextureView(&tiles),
		],
	);
	*pulled_bind_group = make::bind_group(
		device,
		pulled_layout,
		vec![
			window_size_buffer.as_entire_binding(),
			tile_size.entry(),
			pulled_size.entry(),
			pulled_tiles.entry(),
			BindingResource::TextureView(&tiles),
		],
	);
}

fn tiles_gui(queue: &Queue, ctx: &egui::Context, tiles: &mut Tiles) {
	if let Some(ColumnsDialog { mut new_columns, mut pad }) = tiles.columns_dialog.take() {
		let columns_min = if tiles.has_pulled() {
			tiles.pulled_size.val.x
		} else {
			1
		};
		if let Some(change_columns) = columns_window(ctx, columns_min, &mut new_columns, &mut pad) {
			if change_columns && new_columns != tiles.columns.val {
				if pad {
					let (new_tile_refs, new_num_tile_refs) = pad_columns(
						new_columns, tiles.columns.val, &tiles.tile_refs.val, tiles.num_tile_refs.val,
					);
					tiles.tile_refs.set_write(queue, new_tile_refs);
					tiles.num_tile_refs.set_write(queue, new_num_tile_refs);
				}
				tiles.columns.set_write(queue, new_columns);
				let select_max_x = tiles.select_max().x;
				if select_max_x >= tiles.columns.val {
					let delta = select_max_x - tiles.columns.val + 1;
					tiles.move_selection(queue, |pos| pos.x = pos.x.saturating_sub(delta));
				}
			}
		} else {
			tiles.columns_dialog = Some(ColumnsDialog { new_columns, pad });
		}
	}
}

fn save_image(
	window_size: UVec2, device: &Device, queue: &Queue, square: &Buffer, save_pipeline: &RenderPipeline,
	tiles_bind_group: &BindGroup, window_size_buffer: &Buffer, offset: &BufferVal<IVec2>, tile_size: UVec2,
	columns: u32, num_tile_refs: u32, path: &PathBuf,
) -> ImageResult<()> {
	let image_size = uvec2(columns, (num_tile_refs + columns - 1) / columns) * tile_size;
	let desc = make::texture_desc(image_size, 1, TextureUsages::COPY_SRC | TextureUsages::RENDER_ATTACHMENT);
	let texture = device.create_texture(&desc);
	let view = texture.create_view(&TextureViewDescriptor::default());
	let buffer = device.create_buffer(
		&BufferDescriptor {
			label: None,
			size: (image_size.element_product() * PIXEL_BYTES) as u64,
			usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
			mapped_at_creation: false,
		},
	);
	queue.write_buffer(window_size_buffer, 0, image_size.as_bytes());
	queue.write_buffer(&offset.buffer, 0, IVec2::MAX.as_bytes());
	let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
	let mut rpass = make::render_pass(&mut encoder, &view, Color::TRANSPARENT);
	rpass.set_vertex_buffer(0, square.slice(..));
	rpass.set_pipeline(save_pipeline);
	rpass.set_bind_group(0, tiles_bind_group, &[]);
	rpass.draw(0..SQUARE_VERTS, 0..1);
	drop(rpass);
	encoder.copy_texture_to_buffer(
		ImageCopyTexture {
			texture: &texture,
			mip_level: 0,
			origin: Origin3d::ZERO,
			aspect: TextureAspect::All,
		},
		ImageCopyBuffer {
			buffer: &buffer,
			layout: ImageDataLayout {
				offset: 0,
				bytes_per_row: Some(image_size.x * PIXEL_BYTES),
				rows_per_image: None,
			},
		},
		desc.size,
	);
	queue.submit([encoder.finish()]);
	queue.write_buffer(window_size_buffer, 0, window_size.as_bytes());
	offset.write(queue);
	let buffer_slice = buffer.slice(..);
	let (tx, rx) = channel();
	buffer_slice.map_async(MapMode::Read, move |result| tx.send(result).expect("map buffer send"));
	device.poll(Maintain::Wait);
	rx.recv_timeout(Duration::from_secs(5)).expect("map buffer receive").expect("map buffer");
	let data = &buffer_slice.get_mapped_range();
	image::save_buffer(path, data, image_size.x, image_size.y, ColorType::Rgba8)
}

impl Gui for Tiler {
	fn resize(&mut self, window_size: UVec2, queue: &Queue) {
		if let Some(LoadedImage { window_size_buffer, .. }) = &self.loaded_image {
			queue.write_buffer(window_size_buffer, 0, window_size.as_bytes());
		}
	}
	
	fn modifiers(&mut self, modifers: ModifiersState) {
		self.modifers = modifers;
	}
	
	fn key(
		&mut self, window_size: UVec2, device: &Device, queue: &Queue, target: &EventLoopWindowTarget<()>,
		keycode: KeyCode, state: ElementState, repeat: bool,
	) {
		key::key(self, window_size, device, queue, target, keycode, state, repeat);
	}
	
	fn mouse_button(&mut self, button: MouseButton, state: ElementState) {
		match (&mut self.loaded_image, state, button) {
			(Some(LoadedImage { pan, mouse_pos, .. }), ElementState::Pressed, MouseButton::Middle) => {
				*pan = Some(*mouse_pos);
			},
			(Some(LoadedImage { pan, .. }), ElementState::Released, MouseButton::Middle) => {
				*pan = None;
			},
			_ => {},
		}
	}
	
	fn mouse_moved(&mut self, queue: &Queue, position: DVec2) {
		if let Some(LoadedImage { offset, mouse_pos, pan, .. }) = &mut self.loaded_image {
			*mouse_pos = position;
			if let Some(pan) = pan {
				offset.val -= (position - *pan).as_ivec2();
				offset.write(queue);
				*pan = position;
			}
		}
	}
	
	fn mouse_wheel(&mut self, queue: &Queue, delta: MouseScrollDelta) {
		if let Some(LoadedImage { offset, tile_size, .. }) = &mut self.loaded_image {
			offset.val -= match delta {
				MouseScrollDelta::LineDelta(x, y) => (
					if self.modifers.shift_key() {
						vec2(y, x)
					} else {
						vec2(x, y)
					} * tile_size.val.as_vec2()
				).as_ivec2(),
				MouseScrollDelta::PixelDelta(delta) => delta.to_vec().as_ivec2(),
			};
			offset.write(queue);
		}
	}
	
	fn render(&mut self, encoder: &mut CommandEncoder, view: &TextureView) {
		if let Some(LoadedImage { state, .. }) = &self.loaded_image {
			let mut rpass = make::render_pass(encoder, view, Color::BLACK);
			rpass.set_vertex_buffer(0, self.square.slice(..));
			match state {
				State::Split(split) | State::Tiles(_, Some(split)) => {
					rpass.set_pipeline(&self.split_pal.pipeline);
					rpass.set_bind_group(0, &split.split_bind_group, &[]);
					rpass.draw(0..SQUARE_VERTS, 0..1);
				},
				State::Tiles(tiles, None) => {
					rpass.set_pipeline(&self.tiles_pal.pipeline);
					rpass.set_bind_group(0, &tiles.tiles_bind_group, &[]);
					rpass.draw(0..SQUARE_VERTS, 0..1);
					rpass.set_pipeline(&self.select_pal.pipeline);
					rpass.set_bind_group(0, &tiles.select_bind_group, &[]);
					rpass.draw(0..SQUARE_VERTS, 0..1);
					if tiles.has_pulled() {
						rpass.set_pipeline(&self.pulled_pal.pipeline);
						rpass.set_bind_group(0, &tiles.pulled_bind_group, &[]);
						rpass.draw(0..SQUARE_VERTS, 0..1);
					}
				},
			}
		}
	}
	
	fn gui(&mut self, window_size: UVec2, device: &Device, queue: &Queue, ctx: &egui::Context) {
		self.file_dialog.update(ctx);
		if let DialogMode::SelectFile = self.file_dialog.mode() {
			if let Some(path) = self.file_dialog.take_selected() {
				match image::open(path) {
					Ok(image) => self.load_image(device, queue, window_size, image.to_rgba8()),
					Err(e) => self.error = Some(e.to_string()),
				}
			}
		}
		if let Some(LoadedImage { window_size_buffer, offset, tile_size, state, .. }) = &mut self.loaded_image {
			match state {
				State::Split(Split { image, grid_size, grid_offset, gap, .. }) => {
					if split_window(
						queue, ctx, image.size(), TileSizeParam::Mutable(tile_size), grid_size, grid_offset, gap,
					) {
						let tiles = first_split(
							device, queue, &self.tiles_pal.layout, &self.select_pal.layout, &self.pulled_pal.layout,
							window_size_buffer, offset, tile_size, image, grid_size.val, grid_offset.val, gap.val,
						);
						*state = State::Tiles(tiles, None);
					}
				},
				State::Tiles(tiles, split) => {
					match split {
						Some(Split { image, grid_size, grid_offset, gap, .. }) => {
							if split_window(
								queue, ctx, image.size(), TileSizeParam::Const(tile_size.val), grid_size, grid_offset,
								gap,
							) {
								let Tiles {
									pixels, tiles_bind_group, pulled_bind_group, columns, tile_refs, num_tile_refs,
									pulled_tiles, pulled_size, ..
								} = tiles;
								next_split(
									device, queue, &self.tiles_pal.layout, &self.pulled_pal.layout, window_size_buffer,
									offset, tile_size, pixels, tiles_bind_group, pulled_bind_group, num_tile_refs,
									tile_refs, columns, pulled_size, pulled_tiles, image, grid_size.val,
									grid_offset.val, gap.val,
								);
								*split = None;
							}
						},
						None => {
							tiles_gui(queue, ctx, tiles);
							if let DialogMode::SaveFile = self.file_dialog.mode() {
								if let Some(path) = self.file_dialog.take_selected() {
									match save_image(
										window_size, device, queue, &self.square, &self.save_pipeline,
										&tiles.tiles_bind_group, window_size_buffer, offset, tile_size.val,
										tiles.columns.val, tiles.num_tile_refs.val, &path,
									) {
										Ok(()) => println!("image saved"),
										Err(e) => self.error = Some(e.to_string()),
									}
								}
							}
						},
					}
				},
			}
		} else {
			egui::panel::CentralPanel::default().show(ctx, |ui| {
				ui.centered_and_justified(|ui| {
					if ui.label("Ctrl+O or click to open file").clicked() {
						self.file_dialog.select_file();
					}
				});
			});
		}
		if let Some(error) = &self.error {
			if window(ctx, "Error", |ui| {
				ui.label(error);
				ui.button("OK").clicked()
			}) {
				self.error = None;
			}
		}
	}
}

macro_rules! shader {
	($device:expr, $path:literal) => {
		make::shader($device, include_str!($path))
	};
}

fn make_tiler(device: &Device) -> Tiler {
	let tiles_shader = shader!(device, "shader/tiles.wgsl");
	let split_pal = PipelineAndLayout::new(
		device,
		&shader!(device, "shader/split.wgsl"),
		vec![
			make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX),//image size
			make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX),//window size
			make::uniform_layout_entry::<IVec2>(ShaderStages::VERTEX),//offset
			make::uniform_layout_entry::<UVec2>(ShaderStages::FRAGMENT),//tile size
			make::uniform_layout_entry::<UVec2>(ShaderStages::FRAGMENT),//grid size
			make::uniform_layout_entry::<UVec2>(ShaderStages::FRAGMENT),//grid offset
			make::uniform_layout_entry::<UVec2>(ShaderStages::FRAGMENT),//gap
			make::texture_layout_entry(TextureViewDimension::D2),//image
		],
	);
	let tiles_pal = PipelineAndLayout::new(
		device,
		&tiles_shader,
		vec![
			make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX),//window size
			make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX_FRAGMENT),//offset
			make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX_FRAGMENT),//tile size
			make::uniform_layout_entry::<u32>(ShaderStages::VERTEX_FRAGMENT),//columns
			make::uniform_layout_entry::<u32>(ShaderStages::VERTEX_FRAGMENT),//num tiles
			make::uniform_layout_entry::<[u16; MAX_TILES]>(ShaderStages::FRAGMENT),//tiles
			make::texture_layout_entry(TextureViewDimension::D2Array),//tile images
		],
	);
	let select_pal = PipelineAndLayout::new(
		device,
		&shader!(device, "shader/select.wgsl"),
		vec![
			make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX),//window size
			make::uniform_layout_entry::<IVec2>(ShaderStages::VERTEX),//offset
			make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX),//tile size
			make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX),//select pos
			make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX),//select size
		],
	);
	let pulled_pal = PipelineAndLayout::new(
		device,
		&shader!(device, "shader/pulled.wgsl"),
		vec![
			make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX),//window size
			make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX_FRAGMENT),//tile size
			make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX_FRAGMENT),//pulled size
			make::uniform_layout_entry::<[u16; MAX_TILES]>(ShaderStages::VERTEX_FRAGMENT),//tiles
			make::texture_layout_entry(TextureViewDimension::D2Array),//tile images
		],
	);
	let save_pipeline = make::pipeline(device, &tiles_pal.layout, &tiles_shader, TextureFormat::Rgba8UnormSrgb);
	Tiler {
		modifers: ModifiersState::empty(),
		file_dialog: FileDialog::new(),
		error: None,
		square: make::buffer(device, SQUARE.as_bytes(), BufferUsages::VERTEX),
		split_pal,
		tiles_pal,
		select_pal,
		pulled_pal,
		save_pipeline,
		loaded_image: None,
	}
}

fn main() {
	let icon = Icon::from_rgba(include_bytes!("icon24.data").to_vec(), 24, 24).expect("icon");
	gui::run("Tiler", icon, make_tiler);
}
