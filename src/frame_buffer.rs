use glam::{IVec2, UVec2};
use image::{GenericImageView, RgbaImage, SubImage};

//pixels in 0rgb format

pub struct FrameBuffer<'a> {
	pixels: &'a mut [u32],
	size: UVec2,
}

struct DrawRect {
	size: UVec2,
	src_pos: UVec2,
	dest_pos: UVec2,
}

fn rgb(pixel: u32) -> [u32; 3] {
	[pixel >> 16 & 255, pixel >> 8 & 255, pixel & 255]
}

fn make_pixel(r: u32, g: u32, b: u32) -> u32 {
	r << 16 | g << 8 | b
}

impl<'a> FrameBuffer<'a> {
	pub fn new(pixels: &'a mut [u32], size: UVec2) -> Self {
		Self { pixels, size }
	}
	
	fn pixel_offset(&self, x: u32, y: u32) -> usize {
		(self.size.x * y + x) as usize
	}
	
 	fn pixel_mut(&mut self, x: u32, y: u32) -> &mut u32 {
		&mut self.pixels[self.pixel_offset(x, y)]
	}
	
	fn get_draw_rect(&self, pos: IVec2, size: UVec2) -> Option<DrawRect> {
		if pos.cmplt(self.size.as_ivec2()).all() {
			let src_pos = IVec2::ZERO.max(-pos).as_uvec2();
			if src_pos.cmplt(size).all() {
				let dest_pos = IVec2::ZERO.max(pos).as_uvec2();
				return Some(DrawRect {
					size: size.wrapping_add_signed(pos).min(self.size) - dest_pos,
					src_pos,
					dest_pos,
				});
			}
		}
		None
	}
	
	pub fn fill(&mut self) {
		for y in 0..self.size.y {
			for x in 0..self.size.x {
				*self.pixel_mut(x, y) = (((x + y) / 19) % 2) * 0x202020;
			}
		}
	}
	
	pub fn draw_image(&mut self, pos: IVec2, image: SubImage<&RgbaImage>) {
		if let Some(DrawRect { size, src_pos, dest_pos }) = self.get_draw_rect(pos, image.dimensions().into()) {
			for y in 0..size.y {
				for x in 0..size.x {
					let [sr, sg, sb, a] = image.get_pixel(src_pos.x + x, src_pos.y + y).0.map(|c| c as u32);
					let [sr, sg, sb] = [sr, sg, sb].map(|c| c * a / 255);
					let pixel = self.pixel_mut(dest_pos.x + x, dest_pos.y + y);
					let a = 255 - a;
					let [dr, dg, db] = rgb(*pixel).map(|c| c * a / 255);
					*pixel = make_pixel(sr + dr, sg + dg, sb + db);
				}
			}
		}
	}
	
	pub fn draw_rect(&mut self, pos: IVec2, rect: UVec2) {
		let pixel_op = |f: &mut FrameBuffer, x: u32, y: u32| {
			*f.pixel_mut(x, y) = u32::MAX;
		};
		if let Some(DrawRect { size, src_pos, dest_pos }) = self.get_draw_rect(pos, rect) {
			let src_end = src_pos + size;
			if src_pos.x == 0 {
				for y in 0..size.y {
					pixel_op(self, dest_pos.x, dest_pos.y + y);
				}
			}
			if src_pos.y == 0 {
				for x in 0..size.x {
					pixel_op(self, dest_pos.x + x, dest_pos.y);
				}
			}
			if src_end.x == rect.x {
				for y in 0..size.y {
					pixel_op(self, dest_pos.x + size.x - 1, dest_pos.y + y);
				}
			}
			if src_end.y == rect.y {
				for x in 0..size.x {
					pixel_op(self, dest_pos.x + x, dest_pos.y + size.y - 1);
				}
			}
		}
	}
}
