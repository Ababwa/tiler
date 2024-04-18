const TILE_BITS: u8 = 10;
const TILE_MASK: u16 = (1 << TILE_BITS) - 1;

pub struct TileData {
	pub image_index: usize,
	pub tile_index: u32,
}

pub struct Tile(u16);

impl Tile {
	pub fn new(image_index: usize, tile_index: u32) -> Self {
		Self((image_index as u16) << TILE_BITS | tile_index as u16)
	}
	
	pub fn null() -> Self {
		Self(u16::MAX)
	}
	
	pub fn get(&self) -> Option<TileData> {
		match self.0 {
			u16::MAX => None,
			_ => Some(TileData {
				image_index: (self.0 >> TILE_BITS) as usize,
				tile_index: (self.0 & TILE_MASK) as u32,
			})
		}
	}
}
