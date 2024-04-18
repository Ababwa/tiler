mod tile;
mod frame_buffer;

use std::{env::args, fs, mem::{replace, swap}, path::{Path, PathBuf}};
use anyhow::anyhow;
use frame_buffer::FrameBuffer;
use glam::{dvec2, ivec2, uvec2, vec2, IVec2, UVec2};
use image::{GenericImage, GenericImageView, Rgba, RgbaImage, SubImage};
use rfd::{FileDialog, MessageDialog};
use softbuffer::{Context, Surface};
use tile::{Tile, TileData};
use winit::{
	dpi::{PhysicalPosition, PhysicalSize},
	event::{ElementState, Event, KeyEvent, MouseScrollDelta, WindowEvent},
	event_loop::EventLoop,
	keyboard::{KeyCode, ModifiersState, PhysicalKey},
	window::{Window, WindowBuilder},
};

const DEFAULT_TILE_SIZE: u32 = 64;
const DEFAULT_COLUMNS: u32 = 4;
const DEFAULT_OFFSET: IVec2 = ivec2(0, 2);
const DEFAULT_BOX: UVec2 = UVec2::splat(1);

fn to_vec(v: PhysicalSize<u32>) -> UVec2 {
	uvec2(v.width, v.height)
}

fn grid_pos(columns: u32, index: u32) -> UVec2 {
	uvec2(index % columns, index / columns)
}

fn resize(surface: &mut Surface<&Window, &Window>, window_size: PhysicalSize<u32>) {
	let width = window_size.width.try_into().unwrap();
	let height = window_size.height.try_into().unwrap();
	surface.resize(width, height).expect("resize surface");
}

fn unique(tiles: &[Tile], images: &[RgbaImage], tile_size: UVec2, new_tile: &SubImage<&RgbaImage>) -> bool {
	for tile in tiles {
		if let Some(TileData { image_index, tile_index }) = tile.get() {
			let image = &images[image_index];
			let tile_pos = grid_pos(image.width() / tile_size.x, tile_index) * tile_size;
			if image.view(tile_pos.x, tile_pos.y, tile_size.x, tile_size.y)
				.pixels()
				.zip(new_tile.pixels())
				.any(|((_, _, a), (_, _, b))| a != b) {
				return false;
			}
		}
	}
	true
}

fn get_tiles(tiles: &[Tile], images: &[RgbaImage], tile_size: UVec2, new_image: &RgbaImage) -> Vec<Tile> {
	let new_image_grid = UVec2::from(new_image.dimensions()) / tile_size;
	let mut new_tiles = vec![];
	for new_tile_index in 0..new_image_grid.element_product() {
		let src_pos = grid_pos(new_image_grid.x, new_tile_index) * tile_size;
		let new_tile = new_image.view(src_pos.x, src_pos.y, tile_size.x, tile_size.y);
		let first_pixel = new_tile.get_pixel(0, 0);
		if new_tile.pixels().any(|(_, _, pixel)| pixel != first_pixel)
		&& new_tile.pixels().any(|(_, _, Rgba([.., a]))| a != 0)
		&& unique(tiles, images, tile_size, &new_tile) {
			new_tiles.push(Tile::new(images.len(), new_tile_index));
		}
	}
	new_tiles
}

struct Keys([bool; 256]);

impl Keys {
	fn new() -> Self {
		Self([false; 256])
	}
	
	fn just_pressed(&mut self, key: KeyCode, state: ElementState) -> bool {
		let jp = state.is_pressed() && !self.0[key as u8 as usize];
		self.0[key as u8 as usize] = state.is_pressed();
		jp
	}
}

fn read_image<P: AsRef<Path>>(tile_size: UVec2, path: P) -> anyhow::Result<RgbaImage> {
	let image = image::io::Reader::open(path)?.decode()?.into_rgba8();
	if tile_size.cmple(image.dimensions().into()).all() {
		Ok(image)
	} else {
		Err(anyhow!("Image smaller than tile size"))
	}
}

fn open<P: AsRef<Path>>(window: &Window, tiles: &mut Vec<Tile>, images: &mut Vec<RgbaImage>, columns: &mut u32, tile_size: UVec2, path: P) {
	match read_image(tile_size, &path) {
		Ok(image) => {
			tiles.extend(get_tiles(tiles, images, tile_size, &image));
			if images.is_empty() {
				*columns = image.width() / tile_size.x;
			}
			images.push(image);
		},
		Err(e) => _ = MessageDialog::new()
			.set_parent(window)
			.set_title("Error")
			.set_description(format!("Failed to open file: {}", e))
			.show(),
	}
}

fn save(window: &Window, tiles: &[Tile], images: &[RgbaImage], tile_size: UVec2, columns: u32, path: PathBuf) {
	let image_size = uvec2(columns, (tiles.len() as u32 + columns - 1) / columns) * tile_size;
	let mut output_image = RgbaImage::new(image_size.x, image_size.y);
	for (index, tile) in tiles.iter().enumerate() {
		if let Some(TileData { image_index, tile_index }) = tile.get() {
			let image = &images[image_index];
			let src_pos = grid_pos(image.width() / tile_size.x, tile_index) * tile_size;
			let src = *image.view(src_pos.x, src_pos.y, tile_size.x, tile_size.y);
			let dest_pos = grid_pos(columns, index as u32) * tile_size;
			output_image.copy_from(&src, dest_pos.x, dest_pos.y).expect("copy tile");
		}
	}
	if let Err(e) = output_image.save(&path) {
		MessageDialog::new()
			.set_parent(window)
			.set_title("Error")
			.set_description(format!("Failed to save file: {}", e))
			.show();
	}
}

enum Direction { Up, Left, Down, Right }

fn direction(
	select_index: &mut u32,
	select_box: &mut UVec2,
	scroll_offset: &mut IVec2,
	window_size: UVec2,
	tile_size: UVec2,
	columns: u32,
	grabbed: bool,
	shift: bool,
	dir: Direction,
) {
	if !shift && !grabbed {
		*select_box = DEFAULT_BOX;
	}
	let bound = match dir {
		Direction::Up => *select_index >= columns,
		Direction::Left => *select_index % columns > 0,
		Direction::Down => true,
		Direction::Right => (*select_index + select_box.x - 1) % columns < columns - 1,
	};
	if bound {
		if shift {
			if !grabbed {
				match dir {
					Direction::Up => {
						*select_index -= columns;
						select_box.y += 1;
					},
					Direction::Left => {
						*select_index -= 1;
						select_box.x += 1;
					},
					Direction::Down => select_box.y += 1,
					Direction::Right => select_box.x += 1,
				}
			}
		} else {
			match dir {
				Direction::Up => *select_index -= columns,
				Direction::Left => *select_index -= 1,
				Direction::Down => *select_index += columns,
				Direction::Right => *select_index += 1,
			}
		}
	}
	let offset = *scroll_offset + ivec2((window_size.x - (columns * tile_size.x)) as i32 / 2, 0);
	let select_min = offset.wrapping_add_unsigned(grid_pos(columns, *select_index) * tile_size) - 2;
	let select_max = select_min.wrapping_add_unsigned(*select_box * tile_size) + 4;
	let max_out = select_max - window_size.as_ivec2();
	match dir {
		Direction::Up => if select_min.y < 0 {
			scroll_offset.y -= select_min.y;
		},
		Direction::Left => if select_min.x < 0 {
			scroll_offset.x -= select_min.x;
		},
		Direction::Down => if max_out.y > 0 {
			scroll_offset.y -= max_out.y;
		},
		Direction::Right => if max_out.x > 0 {
			scroll_offset.x -= max_out.x;
		},
	}
}

fn parse_args(window: &Window, tiles: &mut Vec<Tile>, images: &mut Vec<RgbaImage>, mut tile_size: UVec2) -> (UVec2, u32) {
	let mut args = args().skip(1).peekable();
	if let Some(arg) = args.peek() {
		if let Ok(ts) = arg.parse::<u32>() {
			tile_size = UVec2::splat(ts);
			args.next();
		}
	}
	if let Some(arg) = args.peek() {
		if let Ok(ts_y) = arg.parse::<u32>() {
			tile_size.y = ts_y;
			args.next();
		}
	}
	let tile_size = if tile_size.cmpeq(UVec2::ZERO).any() {
		println!("tile size cannot be 0, using default");
		UVec2::splat(DEFAULT_TILE_SIZE)
	} else {
		tile_size
	};
	let mut columns = DEFAULT_COLUMNS;
	for arg in args {
		open(&window, tiles, images, &mut columns, tile_size, arg);
	}
	(tile_size, columns)
}

fn read_next_u32(source: &str, pos: &mut usize) -> Option<u32> {
	let mut start = None;
	while *pos < source.len() {
		if source.as_bytes()[*pos].is_ascii_digit() {
			start = Some(*pos);
			break;
		}
		*pos += 1;
	}
	let start = start?;
	let mut end = None;
	while *pos < source.len() {
		if !source.as_bytes()[*pos].is_ascii_digit() {
			end = Some(*pos);
			break;
		}
		*pos += 1;
	}
	match end {
		Some(end) => &source[start..end],
		None => &source[start..],
	}.parse().ok()
}

fn read_tile_size() -> UVec2 {
	let mut tile_size = UVec2::splat(DEFAULT_TILE_SIZE);
	if let Ok(file) = fs::read_to_string("tile_size") {
		let mut pos = 0;
		if let Some(ts) = read_next_u32(&file, &mut pos) {
			tile_size = UVec2::splat(ts);
		}
		if let Some(ts_y) = read_next_u32(&file, &mut pos) {
			tile_size.y = ts_y;
		}
	}
	tile_size
}

fn get_tile_size(window: &Window, tiles: &mut Vec<Tile>, images: &mut Vec<RgbaImage>) -> (UVec2, u32) {
	parse_args(&window, tiles, images, read_tile_size())
}

fn main() {
	let event_loop = EventLoop::new().expect("build event loop");
	let window = WindowBuilder::new()
		.with_title("Tiler")
		.with_min_inner_size(PhysicalSize::new(1, 1))
		.build(&event_loop)
		.expect("build window");
	let context = Context::new(&window).expect("new context");
	let mut surface = Surface::new(&context, &window).expect("new surface");
	let window_size = window.inner_size();
	resize(&mut surface, window_size);
	let mut buffer = surface.buffer_mut().expect("init get buffer");
	buffer.fill(0);
	buffer.present().expect("init present");
	
	let mut window_size = to_vec(window_size);
	let mut scroll_offset = DEFAULT_OFFSET;
	let mut tiles = Vec::<Tile>::new();
	let mut images = Vec::<RgbaImage>::new();
	let mut key_modifiers = ModifiersState::empty();
	let mut keys = Keys::new();
	let mut select_index = 0;
	let mut select_box = DEFAULT_BOX;
	let mut grabbed_tiles = Option::<Vec::<Tile>>::None;
	
	let (tile_size, mut columns) = get_tile_size(&window, &mut tiles, &mut images);
	
	event_loop.run(|event, target| match event {
		Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => target.exit(),
		Event::WindowEvent { event: WindowEvent::ModifiersChanged(modifiers), .. } => key_modifiers = modifiers.state(),
		Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
			resize(&mut surface, new_size);
			window_size = to_vec(new_size);
		},
		Event::WindowEvent { event: WindowEvent::MouseWheel { delta: MouseScrollDelta::LineDelta(x, y), .. }, .. } => {
			let delta = if key_modifiers.shift_key() { vec2(y, x) } else { vec2(x, y) };
			scroll_offset += (delta * tile_size.as_vec2()).as_ivec2();
			window.request_redraw();
		},
		Event::WindowEvent { event: WindowEvent::MouseWheel { delta: MouseScrollDelta::PixelDelta(PhysicalPosition { x, y }), .. }, .. } => {
			scroll_offset -= dvec2(x, y).as_ivec2();
			window.request_redraw();
		},
		Event::WindowEvent { event: WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(keycode), state, .. }, .. }, .. } => if keys.just_pressed(keycode, state) {
			if key_modifiers.control_key() {
				match keycode {
					KeyCode::KeyO => if let Some(paths) = FileDialog::new().set_parent(&window).pick_files() {
						for path in paths {
							open(&window, &mut tiles, &mut images, &mut columns, tile_size, path);
						}
					},
					KeyCode::KeyS => if !tiles.is_empty() {
						if let Some(path) = FileDialog::new().set_parent(&window).save_file() {
							save(&window, &tiles, &images, tile_size, columns, path);
						}
					},
					_ => {},
				}
			} else {
				match keycode {
					KeyCode::KeyW | KeyCode::ArrowUp => direction(&mut select_index, &mut select_box, &mut scroll_offset, window_size, tile_size, columns, grabbed_tiles.is_some(), key_modifiers.shift_key(), Direction::Up),
					KeyCode::KeyA | KeyCode::ArrowLeft => direction(&mut select_index, &mut select_box, &mut scroll_offset, window_size, tile_size, columns, grabbed_tiles.is_some(), key_modifiers.shift_key(), Direction::Left),
					KeyCode::KeyS | KeyCode::ArrowDown => direction(&mut select_index, &mut select_box, &mut scroll_offset, window_size, tile_size, columns, grabbed_tiles.is_some(), key_modifiers.shift_key(), Direction::Down),
					KeyCode::KeyD | KeyCode::ArrowRight => direction(&mut select_index, &mut select_box, &mut scroll_offset, window_size, tile_size, columns, grabbed_tiles.is_some(), key_modifiers.shift_key(), Direction::Right),
					KeyCode::KeyZ => if columns > select_index % columns + select_box.x {
						columns -= 1;
					},
					KeyCode::KeyX => columns += 1,
					KeyCode::KeyR => scroll_offset = DEFAULT_OFFSET,
					KeyCode::Delete | KeyCode::Backspace => grabbed_tiles = None,
					KeyCode::Space | KeyCode::Enter | KeyCode::NumpadEnter => if !tiles.is_empty() {
						let max_index = select_index + (select_box.x - 1) + (select_box.y - 1) * columns;
						let extra = max_index as isize - tiles.len() as isize + 1;
						if extra > 0 {
							tiles.reserve(extra as usize);
							for _ in 0..extra {
								tiles.push(Tile::null());
							}
						}
						let grabbed_tile_vec = match &mut grabbed_tiles {
							Some(grabbed_tile_vec) => {
								for grab_index in 0..select_box.element_product() {
									let tile_index = select_index + (grab_index % select_box.x) + (grab_index / select_box.x) * columns;
									swap(&mut grabbed_tile_vec[grab_index as usize], &mut tiles[tile_index as usize]);
								}
								grabbed_tile_vec
							},
							None => {
								let grab_len = select_box.element_product();
								let mut grabbed_tile_vec = Vec::with_capacity(grab_len as usize);
								for grab_index in 0..grab_len {
									let tile_index = select_index + (grab_index % select_box.x) + (grab_index / select_box.x) * columns;
									grabbed_tile_vec.push(replace(&mut tiles[tile_index as usize], Tile::null()));
								}
								grabbed_tiles.insert(grabbed_tile_vec)
							},
						};
						while !grabbed_tile_vec.is_empty() && grabbed_tile_vec[grabbed_tile_vec.len() - select_box.x as usize..].iter().all(|t| t.get().is_none()) {
							grabbed_tile_vec.truncate(grabbed_tile_vec.len() - select_box.x as usize);
							select_box.y -= 1;
						}
						while !grabbed_tile_vec.is_empty() && grabbed_tile_vec[..select_box.x as usize].iter().all(|t| t.get().is_none()) {
							grabbed_tile_vec.drain(..select_box.x as usize);
							select_box.y -= 1;
						}
						while !grabbed_tile_vec.is_empty() && (0..select_box.y).all(|y| grabbed_tile_vec[(y * select_box.x) as usize].get().is_none()) {
							for y in (0..select_box.y).rev() {
								grabbed_tile_vec.remove((y * select_box.x) as usize);
							}
							select_box.x -= 1;
						}
						while !grabbed_tile_vec.is_empty() && (0..select_box.y).all(|y| grabbed_tile_vec[((y + 1) * select_box.x) as usize - 1].get().is_none()) {
							for y in (0..select_box.y).rev() {
								grabbed_tile_vec.remove(((y + 1) * select_box.x) as usize - 1);
							}
							select_box.x -= 1;
						}
						if grabbed_tile_vec.is_empty() {
							grabbed_tiles = None;
							select_box = DEFAULT_BOX;
						}
						while let Some(true) = tiles.last().map(|t| t.get().is_none()) {
							tiles.pop();
						}
					},
					_ => {},
				}
			};
			window.request_redraw();
		},
		Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
			let mut buffer = surface.buffer_mut().expect("get buffer");
			let mut frame = FrameBuffer::new(&mut buffer, window_size);
			frame.fill();
			let offset = scroll_offset + ivec2((window_size.x - (columns * tile_size.x)) as i32 / 2, 0);
			for (index, tile) in tiles.iter().enumerate() {
				if let Some(TileData { image_index, tile_index }) = tile.get() {
					let image = &images[image_index];
					let src_pos = grid_pos(image.width() / tile_size.x, tile_index) * tile_size;
					let dest_pos = offset.wrapping_add_unsigned(grid_pos(columns, index as u32) * tile_size);
					frame.draw_image(dest_pos, image.view(src_pos.x, src_pos.y, tile_size.x, tile_size.y));
				}
			}
			if !tiles.is_empty() {
				let select_pos = offset.wrapping_add_unsigned(grid_pos(columns, select_index) * tile_size);
				let select_size = select_box * tile_size;
				frame.draw_rect(select_pos - 1, select_size + 2);
				frame.draw_rect(select_pos - 2, select_size + 4);
			}
			if let Some(grabbed_tile_vec) = &grabbed_tiles {
				for (grabbed_tile_index, tile) in grabbed_tile_vec.iter().enumerate() {
					if let Some(TileData { image_index, tile_index }) = tile.get() {
						let image = &images[image_index];
						let src_pos = grid_pos(image.width() / tile_size.x, tile_index) * tile_size;
						let dest_pos = grid_pos(select_box.x, grabbed_tile_index as u32) * tile_size + 2;
						frame.draw_image(dest_pos.as_ivec2(), image.view(src_pos.x, src_pos.y, tile_size.x, tile_size.y));
					}
				}
				let grab_box = select_box * tile_size;
				frame.draw_rect(IVec2::ZERO, grab_box + 4);
				frame.draw_rect(IVec2::splat(1), grab_box + 2);
			}
			buffer.present().expect("present");
		},
		_ => {},
	}).expect("run event loop");
}
