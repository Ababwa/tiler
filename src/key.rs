use glam::{ivec2, uvec2, IVec2, UVec2};
use wgpu::{Device, Queue};
use winit::{event::ElementState, event_loop::EventLoopWindowTarget, keyboard::{KeyCode, ModifiersState}};
use crate::{
	first_split, make::BufferVal, next_split, rect::RectIter, ColumnsDialog, LoadedImage, Rotation, Split, State,
	Tiler, Tiles, NULL_TILE,
};

macro_rules! tiles {
	($key:pat, $repeat:pat, $tiles:pat, $offset:pat, $tile_size:pat) => {
		(
			Some(LoadedImage { offset: $offset, tile_size: $tile_size, state: State::Tiles($tiles, None), .. }),
			_,
			ElementState::Pressed,
			$key,
			$repeat,
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

enum Dir { Up, Left, Down, Right }

fn update_offset(
	dir: Dir, window_size: UVec2, queue: &Queue, tile_size: UVec2, select_min: UVec2, select_max: UVec2,
	offset: &mut BufferVal<IVec2>,
) {
	let window_size = window_size.as_ivec2();
	let block_min = (select_min * tile_size).as_ivec2();
	let block_max = ((select_max + 1) * tile_size).as_ivec2();
	let (dir, dim) = match dir {
		Dir::Up => (0, 1),
		Dir::Left => (0, 0),
		Dir::Down => (1, 1),
		Dir::Right => (1, 0),
	};
	let (ahead, behind) = [
		(block_min[dim], block_min[dim] - window_size[dim]),
		(block_max[dim] - window_size[dim], block_max[dim]),
	][dir];
	let at = [ahead, offset.val[dim]];
	let bt = [behind, offset.val[dim]];
	if at[dir] < at[1 - dir] {
		offset.val[dim] = ahead;
	} else if bt[dir] > bt[1 - dir] {
		let snap = [block_min[dim], block_max[dim] - window_size[dim]];
		offset.val[dim] = if snap[0] < snap[1] {
			snap[dir]
		} else {
			snap[1 - dir]
		};
	}
	offset.write(queue);
}

pub fn key(
	tiler: &mut Tiler, window_size: UVec2, device: &Device, queue: &Queue, target: &EventLoopWindowTarget<()>,
	keycode: KeyCode, state: ElementState, repeat: bool,
) {
	match (&mut tiler.loaded_image, tiler.modifers, state, keycode, repeat) {
		(_, _, ElementState::Pressed, KeyCode::Escape, _) => target.exit(),
		(_, ModifiersState::CONTROL, ElementState::Pressed, KeyCode::KeyO, false) => tiler.file_dialog.select_file(),
		(
			Some(LoadedImage { state: State::Tiles(_, None), .. }),
			ModifiersState::CONTROL,
			ElementState::Pressed,
			KeyCode::KeyS,
			false,
		) => {
			tiler.file_dialog.save_file();
		},
		(Some(LoadedImage { offset, tile_size, state, .. }), _, ElementState::Pressed, KeyCode::KeyR, false) => {
			let content_width = match state {
				State::Split(split) | State::Tiles(_, Some(split)) => split.image.width(),
				State::Tiles(Tiles { columns, .. }, None) => columns.val * tile_size.val.x,
			} as i32;
			offset.set_write(queue, ivec2((content_width - window_size.x as i32) / 2, 0));
		},
		(
			Some(LoadedImage { window_size_buffer, offset, tile_size, state, .. }),
			_,
			ElementState::Pressed,
			KeyCode::Enter | KeyCode::NumpadEnter | KeyCode::Space,
			false,
		) => {
			match state {
				State::Split(Split { image, grid_size, grid_offset, gap, .. }) => {
					let tiles = first_split(
						device, queue, &tiler.tiles_pal.layout, &tiler.select_pal.layout, &tiler.pulled_pal.layout,
						window_size_buffer, offset, tile_size, image, grid_size.val, grid_offset.val, gap.val,
					);
					*state = State::Tiles(tiles, None);
				},
				State::Tiles(tiles, split) => {
					match split {
						Some(Split { image, grid_size, grid_offset, gap, .. }) => {
							let Tiles {
								pixels, tiles_bind_group, pulled_bind_group, columns, tile_refs, num_tile_refs,
								pulled_tiles, pulled_size, ..
							} = tiles;
							next_split(
								device, queue, &tiler.tiles_pal.layout, &tiler.pulled_pal.layout, window_size_buffer,
								offset, tile_size, pixels, tiles_bind_group, pulled_bind_group, num_tile_refs,
								tile_refs, columns, pulled_size, pulled_tiles, image, grid_size.val, grid_offset.val,
								gap.val,
							);
							*split = None;
						},
						None => {
							let (offset, size, start, end) = tiles.selection();
							let pull = RectIter::new(size)
								.map(|pos| offset + pos)
								.filter(|&pos| tiles.tile_checked(pos).is_some())
								.fold(None, |min_max, pos| {
									match min_max {
										None => Some((pos, pos)),
										Some((min, max)) => Some((min.min(pos), max.max(pos))),
									}
								})
								.map(|(min, max)| (
									min,
									max,
									RectIter::new(max - min + 1)
										.map(|pos| *tiles.tile_unchecked(min + pos))
										.collect::<Vec<_>>(),
								));
							if !tiles.has_pulled() {
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
							tiles.update_num_tile_refs(queue, end);
							tiles.tile_refs.write_range(queue, start, end);
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
					}
				},
			}
		},
		tiles!(KeyCode::KeyC, false, tiles, _, _) => {
			tiles.columns_dialog = Some(ColumnsDialog { new_columns: tiles.columns.val, pad: false });
		},
		tiles!(KeyCode::Delete | KeyCode::Backspace, false, tiles, _, _) => {
			tiles.pulled_size.set_write(queue, UVec2::ZERO);
		},
		tiles!(KeyCode::KeyQ, _, tiles, _, _) => {
			let (offset, size, start, end) = tiles.selection();
			for pos in RectIter::new(size) {
				if let Some(tile) = tiles.tile_checked(offset + pos) {
					*tile = ((*tile >> 14) * 3 + 2) % 5 << 14 | *tile & 0x3FFF ^ 0x2000;
				}
			}
			if size.x == size.y && tiler.modifers.control_key() {
				for pos in RectIter::new(uvec2(size.x / 2 + size.x % 2, size.y / 2)) {
					tiles.rotate(offset, size, pos, Rotation::Ccw);
				}
				tiles.update_num_tile_refs(queue, end);
			}
			tiles.tile_refs.write_range(queue, start, end);
		},
		tiles!(KeyCode::KeyE, _, tiles, _, _) => {
			let (offset, size, start, end) = tiles.selection();
			for pos in RectIter::new(size) {
				if let Some(tile) = tiles.tile_checked(offset + pos) {
					*tile = ((*tile >> 14) * 2 + 1) % 5 << 14 | *tile & 0x3FFF ^ 0x2000;
				}
			}
			if size.x == size.y && tiler.modifers.control_key() {
				for pos in RectIter::new(uvec2(size.x / 2 + size.x % 2, size.y / 2)) {
					tiles.rotate(offset, size, pos, Rotation::Cw);
				}
				tiles.update_num_tile_refs(queue, end);
			}
			tiles.tile_refs.write_range(queue, start, end);
		},
		tiles!(KeyCode::KeyF, _, tiles, _, _) => {
			let (offset, size, start, end) = tiles.selection();
			for pos in RectIter::new(size) {
				if let Some(tile) = tiles.tile_checked(offset + pos) {
					*tile ^= 0x4000 << tiler.modifers.shift_key() as u8 as u16;
				}
			}
			if tiler.modifers.control_key() {
				let axis = UVec2::AXES[tiler.modifers.shift_key() as u8 as usize];
				for pos in RectIter::new(size * (2 - axis) / 2) {
					let a = tiles.index(offset + pos);
					let b = tiles.index(offset + (size - pos - 1) * axis + (1 - axis) * pos);
					tiles.tile_refs.val.swap(a, b);
				}
				tiles.update_num_tile_refs(queue, end);
			}
			tiles.tile_refs.write_range(queue, start, end);
		},
		tiles!(KeyCode::ArrowUp | KeyCode::KeyW, _, tiles, offset, tile_size) => {
			if tiles.has_pulled() {
				if tiles.select_min().y > 0 {
					tiles.move_selection(queue, up);
				}
				update_offset(Dir::Up, window_size, queue, tile_size.val, tiles.select_min(), tiles.select_max(), offset);
			} else {
				if tiles.select_pos2.val.y > 0 {
					tiles.select_pos2.modify_write(queue, up);
				}
				if !tiler.modifers.shift_key() {
					tiles.select_pos1.set_write(queue, tiles.select_pos2.val);
				}
				update_offset(Dir::Up, window_size, queue, tile_size.val, tiles.select_pos2.val, tiles.select_pos2.val, offset);
			}
		},
		tiles!(KeyCode::ArrowLeft | KeyCode::KeyA, _, tiles, offset, tile_size) => {
			if tiles.has_pulled() {
				if tiles.select_min().x > 0 {
					tiles.move_selection(queue, left);
				}
				update_offset(Dir::Left, window_size, queue, tile_size.val, tiles.select_min(), tiles.select_max(), offset);
			} else {
				if tiles.select_pos2.val.x > 0 {
					tiles.select_pos2.modify_write(queue, left);
				}
				if !tiler.modifers.shift_key() {
					tiles.select_pos1.set_write(queue, tiles.select_pos2.val);
				}
				update_offset(Dir::Left, window_size, queue, tile_size.val, tiles.select_pos2.val, tiles.select_pos2.val, offset);
			}
		},
		tiles!(KeyCode::ArrowDown | KeyCode::KeyS, _, tiles, offset, tile_size) => {
			tiles.select_pos2.modify_write(queue, down);
			if tiles.has_pulled() {
				tiles.select_pos1.modify_write(queue, down);
				update_offset(Dir::Down, window_size, queue, tile_size.val, tiles.select_min(), tiles.select_max(), offset);
			} else {
				if !tiler.modifers.shift_key() {
					tiles.select_pos1.set_write(queue, tiles.select_pos2.val);
				}
				update_offset(Dir::Down, window_size, queue, tile_size.val, tiles.select_pos2.val, tiles.select_pos2.val, offset);
			}
		},
		tiles!(KeyCode::ArrowRight | KeyCode::KeyD, _, tiles, offset, tile_size) => {
			if tiles.has_pulled() {
				if tiles.select_max().x < tiles.columns.val - 1 {
					tiles.move_selection(queue, right);
				}
				update_offset(Dir::Right, window_size, queue, tile_size.val, tiles.select_min(), tiles.select_max(), offset);
			} else {
				if tiles.select_pos2.val.x < tiles.columns.val - 1 {
					tiles.select_pos2.modify_write(queue, right);
				}
				if !tiler.modifers.shift_key() {
					tiles.select_pos1.set_write(queue, tiles.select_pos2.val);
				}
				update_offset(Dir::Right, window_size, queue, tile_size.val, tiles.select_pos2.val, tiles.select_pos2.val, offset);
			}
		},
		_ => {},
	}
}
