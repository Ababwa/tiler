use glam::{uvec2, UVec2};

pub struct RectIter {
	pos: UVec2,
	end: UVec2,
}

impl RectIter {
	pub fn new(end: UVec2) -> Self {
		Self { pos: uvec2(0, (end.x == 0) as u32 * end.y), end }
	}
	
	fn remaining(&self) -> usize {
		(self.end.element_product() - self.pos.y * self.end.x - self.pos.x) as usize
	}
}

impl Iterator for RectIter {
	type Item = UVec2;
	
	fn next(&mut self) -> Option<Self::Item> {
		if self.pos.y == self.end.y {
			None
		} else {
			let ret = self.pos;
			self.pos.x += 1;
			if self.pos.x == self.end.x {
				self.pos.x = 0;
				self.pos.y += 1;
			}
			Some(ret)
		}
	}
	
	fn size_hint(&self) -> (usize, Option<usize>) {
		let remaining = self.remaining();
		(remaining, Some(remaining))
	}
	
	fn count(self) -> usize {
		self.remaining()
	}
	
	fn last(self) -> Option<Self::Item> {
		if self.pos.y == self.end.y {
			None
		} else {
			Some(self.end - 1)
		}
	}
}
