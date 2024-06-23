mod ext;
mod gui;
mod make;
mod rect;

use std::ops::RangeInclusive;
use egui_file_dialog::FileDialog;
use glam::{ivec2, uvec2, vec2, DVec2, IVec2, UVec2};
use image::{GenericImageView, Rgba, RgbaImage, SubImage};
use make::{BufferVal, PipelineAndLayout};
use rect::RectIter;
use wgpu::{
	BindGroup, BindGroupLayout, BindingResource, Buffer, BufferUsages, CommandEncoder, Device,
	Queue, ShaderStages, TextureView, TextureViewDimension,
};
use winit::{
	event::{ElementState, MouseButton, MouseScrollDelta},
	keyboard::{KeyCode, ModifiersState},
	event_loop::EventLoopWindowTarget,
};
use gui::Gui;
use ext::{AsBytes, PixelsOnly, Size, ToVec};

const MAX_TILES: usize = 1024;
const NULL_TILE: u16 = u16::MAX;
const DEFAULT_TILE_SIZE: u32 = 64;
const SQUARE: [[u16; 2]; 4] = [[0, 0], [1, 0], [0, 1], [1, 1]];
const SQUARE_VERTS: u32 = SQUARE.len() as u32;
const WRITABLE_UNIFORM: BufferUsages = BufferUsages::UNIFORM.union(BufferUsages::COPY_DST);

fn window<R>(ctx: &egui::Context, title: &str, contents: impl FnOnce(&mut egui::Ui) -> R) -> R {
	egui::Window::new(title)
		.collapsible(false)
		.resizable(false)
		.show(ctx, contents)
		.unwrap()
		.inner
		.unwrap()
}

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
	columns: BufferVal<u32>,
	select_pos1: BufferVal<UVec2>,
	select_pos2: BufferVal<UVec2>,
	num_tiles: BufferVal<u32>,
	tiles: BufferVal<Box<[u16; MAX_TILES]>>,
	pulled_tiles: BufferVal<Box<[u16; MAX_TILES]>>,
	pulled_size: BufferVal<UVec2>,
	tiles_bind_group: BindGroup,
	select_bind_group: BindGroup,
	pulled_bind_group: BindGroup,
	columns_dialog: Option<ColumnsDialog>,
}

const ROTATE_CW: usize = 0;
const ROTATE_CCW: usize = 1;

impl Tiles {
	fn index(&self, pos: UVec2) -> usize {
		(pos.y * self.columns.val + pos.x) as usize
	}
	
	fn tile_unchecked(&mut self, pos: UVec2) -> &mut u16 {
		&mut self.tiles.val[self.index(pos)]
	}
	
	fn tile_checked(&mut self, pos: UVec2) -> Option<&mut u16> {
		let index = self.index(pos);
		if index < self.num_tiles.val as usize {
			match &mut self.tiles.val[index] {
				&mut NULL_TILE => None,
				tile => Some(tile),
			}
		} else {
			None
		}
	}
	
	fn selection(&self) -> (UVec2, UVec2, usize, usize) {
		let min = self.select_pos1.val.min(self.select_pos2.val);
		let max = self.select_pos1.val.max(self.select_pos2.val);
		let size = max - min + 1;
		let start = self.index(min);
		let end = self.index(max) + 1;
		(min, size, start, end)
	}
	
	fn rotate(&mut self, offset: UVec2, size: UVec2, pos: UVec2, dir: usize) {
		let indices = [
			self.index(offset + pos),
			self.index(offset + uvec2(size.x - pos.y - 1, pos.x)),
			self.index(offset + uvec2(size.x - pos.x - 1, size.y - pos.y - 1)),
			self.index(offset + uvec2(pos.y, size.y - pos.x - 1)),
		];
		self.tiles.val.swap(indices[0], indices[1]);
		self.tiles.val.swap(indices[2], indices[3]);
		self.tiles.val.swap(indices[dir], indices[dir + 2]);
	}
	
	fn update_num_tiles(&mut self, queue: &Queue, mut end: usize) {
		if end as u32 >= self.num_tiles.val {
			while end != 0 && self.tiles.val[end - 1] == NULL_TILE {
				end -= 1;
			}
			self.num_tiles.set_write(queue, end as u32);
		}
	}
}

enum State {
	Split(Split),
	Tiles(Tiles),
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
	loaded_image: Option<LoadedImage>,
}

fn load_image(
	window_size: UVec2,
	device: &Device,
	queue: &Queue,
	split_bind_group_layout: &BindGroupLayout,
	image: RgbaImage,
) -> LoadedImage {
	let image_size = image.size();
	let window_size_buffer = make::buffer(device, window_size.as_bytes(), WRITABLE_UNIFORM);
	let offset = BufferVal::new(device, ivec2((window_size.x as i32 - image_size.x as i32) / 2, 0));
	let tile_size = BufferVal::new(device, UVec2::splat(DEFAULT_TILE_SIZE).min(image_size));
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
		BindingResource::TextureView(&make::texture_view(device, queue, image_size, 1, &image)),
	]);
	LoadedImage {
		window_size_buffer,
		offset,
		tile_size,
		mouse_pos: DVec2::ZERO,
		pan: None,
		state: State::Split(Split { image, grid_size, grid_offset, gap, split_bind_group }),
	}
}

fn drag_value(ui: &mut egui::Ui, val: &mut u32, bounds: RangeInclusive<u32>, label: &str) -> bool {
	ui.add(egui::DragValue::new(val).clamp_range(bounds).prefix(label)).changed()
}

fn drag_value_xy(ui: &mut egui::Ui, val: &mut UVec2, lower: u32, upper: UVec2) -> bool {
	drag_value(ui, &mut val.x, lower..=upper.x, "X: ") |
	drag_value(ui, &mut val.y, lower..=upper.y, "Y: ")
}

fn unique_tile(tiles: &[SubImage<&RgbaImage>], tile: &SubImage<&RgbaImage>) -> bool {
	tiles
		.iter()
		.all(|existing_tile| existing_tile.pixels_only().zip(tile.pixels_only()).any(|(a, b)| a != b))
}

fn interesting_tile(tile: &SubImage<&RgbaImage>) -> bool {
	let first_pixel = tile.get_pixel(0, 0);
	tile.pixels_only().any(|pixel| pixel[3] != 0) && (
		tile.dimensions() == (1, 1) || tile.pixels_only().any(|pixel| pixel != first_pixel)
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

fn split_window(
	queue: &Queue,
	ctx: &egui::Context,
	tile_size: &mut BufferVal<UVec2>,
	image_size: UVec2,
	grid_size: &mut BufferVal<UVec2>,
	grid_offset: &mut BufferVal<UVec2>,
	gap: &mut BufferVal<UVec2>,
) -> bool {
	window(ctx, "Split Tiles", |ui| {
		split_param(queue, ui, "Tile Size", tile_size, 1,
			(image_size - grid_offset.val - (grid_size.val - 1) * gap.val) / grid_size.val,
		);
		split_param(queue, ui, "Grid Size", grid_size, 1,
			(image_size - grid_offset.val + gap.val) / (tile_size.val + gap.val),
		);
		split_param(queue, ui, "Offset", grid_offset, 0,
			image_size - (grid_size.val - 1) * gap.val - grid_size.val * tile_size.val,
		);
		split_param(queue, ui, "Gap", gap, 0,
			(image_size - grid_offset.val - grid_size.val * tile_size.val) /
			UVec2::ONE.max(grid_size.val - 1),
		);
		ui.button("Ok").clicked()
	})
}

fn columns_window(ctx: &egui::Context, columns: &mut u32, pad: &mut bool) -> Option<bool> {
	window(ctx, "Columns", |ui| {
		drag_value(ui, columns, 1..=1000, "");
		ui.checkbox(pad, "Maintain tile positions");
		match ui.horizontal(|ui| (ui.button("Ok").clicked(), ui.button("Cancel").clicked())).inner {
			(true, _) => Some(true),
			(_, true) => Some(false),
			_ => None,
		}
	})
}

fn pad_columns(
	new_columns: u32, old_columns: u32, tiles: &[u16; MAX_TILES], num_tiles: u32,
) -> (Box<[u16; MAX_TILES]>, u32) {
	let mut new_tiles = vec![NULL_TILE; MAX_TILES];
	let mut new_index = 0;
	let mut old_index = 0;
	if new_columns > old_columns {
		loop {
			if (new_index % new_columns) < old_columns {
				new_tiles[new_index as usize] = tiles[old_index as usize];
				old_index += 1;
				new_index += 1;
				if old_index == num_tiles {
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
				new_tiles[new_index as usize] = tiles[old_index as usize];
				new_index += 1;
			} else {
				match tiles[old_index as usize] {
					NULL_TILE => {},
					tile => outside.push(tile),
				}
			}
			old_index += 1;
			if old_index == num_tiles {
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
	device: &Device,
	queue: &Queue,
	tiles_bind_group_layout: &BindGroupLayout,
	select_bind_group_layout: &BindGroupLayout,
	pulled_bind_group_layout: &BindGroupLayout,
	window_size_buffer: &Buffer,
	offset: &BufferVal<IVec2>,
	tile_size: &BufferVal<UVec2>,
	image: &RgbaImage,
	grid_size: &BufferVal<UVec2>,
	grid_offset: &BufferVal<UVec2>,
	gap: &BufferVal<UVec2>,
) -> State {
	let mut tile_images = Vec::with_capacity(grid_size.val.element_product() as usize);
	for grid_pos in RectIter::new(grid_size.val) {
		let tile_pos = grid_offset.val + grid_pos * (tile_size.val + gap.val);
		let image = image.view(tile_pos.x, tile_pos.y, tile_size.val.x, tile_size.val.y);
		if interesting_tile(&image) && unique_tile(&tile_images, &image) {
			tile_images.push(image);
		}
	}
	let pixels = tile_images.iter().flat_map(|tile| tile.pixels_only()).collect::<Vec<_>>();
	let columns = BufferVal::new(device, grid_size.val.x);
	let select_pos1 = BufferVal::new(device, UVec2::ZERO);
	let select_pos2 = BufferVal::new(device, UVec2::ZERO);
	let num_tiles = BufferVal::new(device, tile_images.len() as u32);
	let mut tile_indices = vec![NULL_TILE; MAX_TILES];
	for i in 0..tile_images.len() {
		tile_indices[i] = i as u16;
	}
	let tiles = tile_indices.into_boxed_slice().try_into().unwrap();
	let tiles = BufferVal::new(device, tiles);
	let pulled_tiles = vec![NULL_TILE; MAX_TILES].into_boxed_slice().try_into().unwrap();
	let pulled_tiles = BufferVal::new(device, pulled_tiles);
	let pulled_size = BufferVal::new(device, UVec2::ZERO);
	let tile_images = make::texture_view(
		device, queue, tile_size.val, tile_images.len() as u32, pixels.as_bytes(),
	);
	let tiles_bind_group = make::bind_group(device, tiles_bind_group_layout, vec![
		window_size_buffer.as_entire_binding(),
		offset.entry(),
		tile_size.entry(),
		columns.entry(),
		num_tiles.entry(),
		tiles.entry(),
		BindingResource::TextureView(&tile_images),
	]);
	let select_bind_group = make::bind_group(device, select_bind_group_layout, vec![
		window_size_buffer.as_entire_binding(),
		offset.entry(),
		tile_size.entry(),
		select_pos1.entry(),
		select_pos2.entry(),
	]);
	let pulled_bind_group = make::bind_group(device, pulled_bind_group_layout, vec![
		window_size_buffer.as_entire_binding(),
		tile_size.entry(),
		pulled_size.entry(),
		pulled_tiles.entry(),
		BindingResource::TextureView(&tile_images),
	]);
	State::Tiles(Tiles {
		pixels,
		columns,
		select_pos1,
		select_pos2,
		num_tiles,
		tiles,
		pulled_tiles,
		pulled_size,
		tiles_bind_group,
		select_bind_group,
		pulled_bind_group,
		columns_dialog: None,
	})
}

macro_rules! image_fields {
	($key:pat, $($field:ident),*) => {
		(Some(LoadedImage { $($field,)* .. }), _, ElementState::Pressed, $key)
	};
}

macro_rules! tiles_fields {
	($key:pat, $($field:ident),* $(,)?) => {
		(
			Some(LoadedImage { state: State::Tiles(Tiles { $($field,)* .. }), .. }),
			_,
			ElementState::Pressed,
			$key,
		)
	};
}

macro_rules! tiles {
	($key:pat, $tiles:ident) => {
		(
			Some(LoadedImage { state: State::Tiles($tiles), .. }),
			_,
			ElementState::Pressed,
			$key,
		)
	};
}

fn up(v: &mut UVec2) {
	v.y -= 1;
}

fn left(v: &mut UVec2) {
	v.x -= 1;
}

fn down(v: &mut UVec2) {
	v.y += 1;
}

fn right(v: &mut UVec2) {
	v.x += 1;
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
		&mut self,
		window_size: UVec2,
		queue: &Queue,
		target: &EventLoopWindowTarget<()>,
		keycode: KeyCode,
		state: ElementState,
	) {
		match (&mut self.loaded_image, self.modifers, state, keycode) {
			(_, _, ElementState::Pressed, KeyCode::Escape) => target.exit(),
			(_, ModifiersState::CONTROL, ElementState::Pressed, KeyCode::KeyO) => {
				self.file_dialog.select_file();
			},
			image_fields!(KeyCode::KeyR, offset, tile_size, state) => {
				let content_width = match state {
					State::Split(Split { image, .. }) => image.width(),
					State::Tiles(Tiles { columns, .. }) => columns.val * tile_size.val.x,
				} as i32;
				offset.set_write(queue, ivec2((window_size.x as i32 - content_width) / 2, 0));
			},
			tiles_fields!(KeyCode::KeyC, columns, columns_dialog) => {
				*columns_dialog = Some(ColumnsDialog { new_columns: columns.val, pad: false });
			},
			tiles_fields!(
				KeyCode::ArrowUp | KeyCode::KeyW, select_pos1, select_pos2, pulled_size
			) => {
				if select_pos2.val.y > 0 {
					select_pos2.modify_write(queue, up);
					if pulled_size.val.element_product() != 0 {
						select_pos1.modify_write(queue, up);
					} else if !self.modifers.shift_key() {
						select_pos1.set_write(queue, select_pos2.val);
					}
				}
			},
			tiles_fields!(
				KeyCode::ArrowLeft | KeyCode::KeyA, select_pos1, select_pos2, pulled_size,
			) => {
				if select_pos2.val.x > 0 {
					select_pos2.modify_write(queue, left);
					if pulled_size.val.element_product() != 0 {
						select_pos1.modify_write(queue, left)
					} else if !self.modifers.shift_key() {
						select_pos1.set_write(queue, select_pos2.val);
					}
				}
			},
			tiles_fields!(
				KeyCode::ArrowDown | KeyCode::KeyS, select_pos1, select_pos2, pulled_size,
			) => {
				select_pos2.modify_write(queue, down);
				if pulled_size.val.element_product() != 0 {
					select_pos1.modify_write(queue, down);
				} else if !self.modifers.shift_key() {
					select_pos1.set_write(queue, select_pos2.val);
				}
			},
			tiles_fields!(
				KeyCode::ArrowRight | KeyCode::KeyD, select_pos1, select_pos2, columns, pulled_size,
			) => {
				if select_pos2.val.x < columns.val - 1 {
					select_pos2.modify_write(queue, right);
					if pulled_size.val.element_product() != 0 {
						select_pos1.modify_write(queue, right);
					} else if !self.modifers.shift_key() {
						select_pos1.set_write(queue, select_pos2.val);
					}
				}
			},
			tiles!(KeyCode::KeyQ, tiles) => {
				let (offset, size, start, end) = tiles.selection();
				for pos in RectIter::new(size) {
					if let Some(tile) = tiles.tile_checked(offset + pos) {
						*tile = ((*tile >> 14) * 3 + 2) % 5 << 14 | *tile & 0x3FFF ^ 0x2000;
					}
				}
				if size.x == size.y && self.modifers.control_key() {
					for pos in RectIter::new(uvec2(size.x / 2 + size.x % 2, size.y / 2)) {
						tiles.rotate(offset, size, pos, ROTATE_CCW);
					}
					tiles.update_num_tiles(queue, end);
				}
				tiles.tiles.write_range(queue, start, end);
			},
			tiles!(KeyCode::KeyE, tiles) => {
				let (offset, size, start, end) = tiles.selection();
				for pos in RectIter::new(size) {
					if let Some(tile) = tiles.tile_checked(offset + pos) {
						*tile = ((*tile >> 14) * 2 + 1) % 5 << 14 | *tile & 0x3FFF ^ 0x2000;
					}
				}
				if size.x == size.y && self.modifers.control_key() {
					for pos in RectIter::new(uvec2(size.x / 2 + size.x % 2, size.y / 2)) {
						tiles.rotate(offset, size, pos, ROTATE_CW);
					}
					tiles.update_num_tiles(queue, end);
				}
				tiles.tiles.write_range(queue, start, end);
			},
			tiles!(KeyCode::KeyF, tiles) => {
				let (offset, size, start, end) = tiles.selection();
				for pos in RectIter::new(size) {
					if let Some(tile) = tiles.tile_checked(offset + pos) {
						*tile ^= 0x4000 << self.modifers.shift_key() as u8 as u16;
					}
				}
				if self.modifers.control_key() {
					let axis = UVec2::AXES[self.modifers.shift_key() as u8 as usize];
					for pos in RectIter::new(size * (2 - axis) / 2) {
						let a = tiles.index(offset + pos);
						let b = tiles.index(offset + (size - pos - 1) * axis + (1 - axis) * pos);
						tiles.tiles.val.swap(a, b);
					}
					tiles.update_num_tiles(queue, end);
				}
				tiles.tiles.write_range(queue, start, end);
			},
			tiles!(KeyCode::Space | KeyCode::Enter | KeyCode::NumpadEnter, tiles) => {
				let (offset, size, start, end) = tiles.selection();
				let pull = RectIter::new(size)
					.map(|pos| offset + pos)
					.filter(|&pos| tiles.tile_checked(pos).is_some())
					.fold(None, |min_max, pos| match min_max {
						None => Some((pos, pos)),
						Some((min, max)) => Some((min.min(pos), max.max(pos))),
					})
					.map(|(min, max)| (
						min,
						max,
						RectIter::new(max - min + 1)
							.map(|pos| *tiles.tile_unchecked(min + pos))
							.collect::<Vec<_>>(),
					));
				if tiles.pulled_size.val.element_product() == 0 {
					for pos in RectIter::new(size) {
						*tiles.tile_unchecked(offset + pos) = NULL_TILE;
					}
				} else if tiles.pulled_size.val == size {
					for (index, pos) in RectIter::new(size).enumerate() {
						*tiles.tile_unchecked(offset + pos) = tiles.pulled_tiles.val[index];
					}
				} else {
					panic!("pulled size not zero and not equal to selection size");
				}
				tiles.update_num_tiles(queue, end);
				tiles.tiles.write_range(queue, start, end);
				match pull {
					None => tiles.pulled_size.set_write(queue, UVec2::ZERO),
					Some((min, max, pulled_tiles)) => {
						tiles.pulled_tiles.val[..pulled_tiles.len()].copy_from_slice(&pulled_tiles);
						tiles.pulled_tiles.write_range(queue, 0, pulled_tiles.len());
						tiles.pulled_size.set_write(queue, max - min + 1);
						tiles.select_pos1.set_write(queue, min);
						tiles.select_pos2.set_write(queue, max);
					},
				}
			},
			_ => {},
		}
	}
	
	fn mouse_button(&mut self, button: MouseButton, state: ElementState) {
		match (&mut self.loaded_image, state, button) {
			(
				Some(LoadedImage { pan, mouse_pos, .. }),
				ElementState::Pressed,
				MouseButton::Middle,
			) => *pan = Some(*mouse_pos),
			(
				Some(LoadedImage { pan, .. }),
				ElementState::Released,
				MouseButton::Middle,
			) => *pan = None,
			_ => {},
		}
	}
	
	fn mouse_moved(&mut self, queue: &Queue, position: DVec2) {
		if let Some(LoadedImage { offset, mouse_pos, pan, .. }) = &mut self.loaded_image {
			*mouse_pos = position;
			if let Some(pan) = pan {
				offset.val += (position - *pan).as_ivec2();
				offset.write(queue);
				*pan = position;
			}
		}
	}
	
	fn mouse_wheel(&mut self, queue: &Queue, delta: MouseScrollDelta) {
		if let Some(LoadedImage { offset, tile_size, .. }) = &mut self.loaded_image {
			let delta = match delta {
				MouseScrollDelta::LineDelta(x, y) => (
					if self.modifers.shift_key() {
						vec2(y, x)
					} else {
						vec2(x, y)
					} * tile_size.val.as_vec2()
				).as_ivec2(),
				MouseScrollDelta::PixelDelta(delta) => delta.to_vec().as_ivec2(),
			};
			offset.val += delta;
			offset.write(queue);
		}
	}
	
	fn render(&mut self, encoder: &mut CommandEncoder, view: &TextureView) {
		match &self.loaded_image {
			Some(LoadedImage { state: State::Split(Split { split_bind_group, .. }), .. }) => {
				let mut rpass = make::render_pass(encoder, view);
				rpass.set_vertex_buffer(0, self.square.slice(..));
				rpass.set_pipeline(&self.split_pal.pipeline);
				rpass.set_bind_group(0, split_bind_group, &[]);
				rpass.draw(0..SQUARE_VERTS, 0..1);
			},
			Some(LoadedImage { state: State::Tiles(Tiles {
				tiles_bind_group, select_bind_group, pulled_bind_group, pulled_size, ..
			}), .. }) => {
				let mut rpass = make::render_pass(encoder, view);
				rpass.set_vertex_buffer(0, self.square.slice(..));
				rpass.set_pipeline(&self.tiles_pal.pipeline);
				rpass.set_bind_group(0, tiles_bind_group, &[]);
				rpass.draw(0..SQUARE_VERTS, 0..1);
				rpass.set_pipeline(&self.select_pal.pipeline);
				rpass.set_bind_group(0, select_bind_group, &[]);
				rpass.draw(0..SQUARE_VERTS, 0..1);
				if pulled_size.val.element_product() != 0 {
					rpass.set_pipeline(&self.pulled_pal.pipeline);
					rpass.set_bind_group(0, pulled_bind_group, &[]);
					rpass.draw(0..SQUARE_VERTS, 0..1);
				}
			},
			_ => {},
		}
	}
	
	fn egui(&mut self, window_size: UVec2, device: &Device, queue: &Queue, ctx: &egui::Context) {
		self.file_dialog.update(ctx);
		if let Some(path) = self.file_dialog.take_selected() {
			match image::open(path) {
				Ok(image) => self.loaded_image = Some(load_image(
					window_size, device, queue, &self.split_pal.layout, image.to_rgba8(),
				)),
				Err(e) => self.error = Some(e.to_string()),
			}
		}
		match &mut self.loaded_image {
			Some(LoadedImage { window_size_buffer, offset, tile_size, state, .. }) => {
				match state {
					State::Split(Split { image, grid_size, grid_offset, gap, .. }) => {
						if split_window(
							queue, ctx, tile_size, image.size(), grid_size, grid_offset, gap,
						) {
							*state = split_image(
								device,
								queue,
								&self.tiles_pal.layout,
								&self.select_pal.layout,
								&self.pulled_pal.layout,
								window_size_buffer,
								offset,
								tile_size, 
								image, 
								grid_size,
								grid_offset,
								gap,
							);
						}
					},
					State::Tiles(Tiles { columns, num_tiles, tiles, columns_dialog, .. }) => {
						if let Some(ColumnsDialog { new_columns, pad }) = columns_dialog {
							if let Some(change_columns) = columns_window(ctx, new_columns, pad) {
								if change_columns && *new_columns != columns.val {
									if *pad {
										let (new_tiles, new_num_tiles) = pad_columns(
											*new_columns, columns.val, &tiles.val, num_tiles.val,
										);
										tiles.set_write(queue, new_tiles);
										num_tiles.set_write(queue, new_num_tiles);
									}
									columns.set_write(queue, *new_columns);
								}
								*columns_dialog = None;
							}
						}
					},
				}
			},
			None => {
				egui::panel::CentralPanel::default().show(ctx, |ui| {
					ui.centered_and_justified(|ui| {
						ui.label("Ctrl+O to open file");
					});
				});
			},
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

fn make_tiler(device: &Device) -> Tiler {
	let split_pal = PipelineAndLayout::new(device, include_str!("shader/split.wgsl"), vec![
		make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX),//image size
		make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX),//window size
		make::uniform_layout_entry::<IVec2>(ShaderStages::VERTEX),//offset
		make::uniform_layout_entry::<UVec2>(ShaderStages::FRAGMENT),//tile size
		make::uniform_layout_entry::<UVec2>(ShaderStages::FRAGMENT),//grid size
		make::uniform_layout_entry::<UVec2>(ShaderStages::FRAGMENT),//grid offset
		make::uniform_layout_entry::<UVec2>(ShaderStages::FRAGMENT),//gap
		make::texture_layout_entry(TextureViewDimension::D2),//image
	]);
	let tiles_pal = PipelineAndLayout::new(device, include_str!("shader/tiles.wgsl"), vec![
		make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX),//window size
		make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX),//offset
		make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX_FRAGMENT),//tile size
		make::uniform_layout_entry::<u32>(ShaderStages::VERTEX_FRAGMENT),//columns
		make::uniform_layout_entry::<u32>(ShaderStages::VERTEX_FRAGMENT),//num tiles
		make::uniform_layout_entry::<[u16; MAX_TILES]>(ShaderStages::FRAGMENT),//tiles
		make::texture_layout_entry(TextureViewDimension::D2Array),//tile images
	]);
	let select_pal = PipelineAndLayout::new(device, include_str!("shader/select.wgsl"), vec![
		make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX),//window size
		make::uniform_layout_entry::<IVec2>(ShaderStages::VERTEX),//offset
		make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX),//tile size
		make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX),//select pos
		make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX),//select size
	]);
	let pulled_pal = PipelineAndLayout::new(device, include_str!("shader/pulled.wgsl"), vec![
		make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX),//window size
		make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX_FRAGMENT),//tile size
		make::uniform_layout_entry::<UVec2>(ShaderStages::VERTEX_FRAGMENT),//pulled size
		make::uniform_layout_entry::<[u16; MAX_TILES]>(ShaderStages::VERTEX_FRAGMENT),//tiles
		make::texture_layout_entry(TextureViewDimension::D2Array),//tile images
	]);
	Tiler {
		modifers: ModifiersState::empty(),
		file_dialog: FileDialog::new(),
		error: None,
		square: make::buffer(device, SQUARE.as_bytes(), BufferUsages::VERTEX),
		split_pal,
		tiles_pal,
		select_pal,
		pulled_pal,
		loaded_image: None,
	}
}

fn main() {
	gui::run("Tiler", make_tiler);
}
