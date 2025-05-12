use std::{future::Future, slice};
use glam::{DVec2, UVec2};
use image::{Rgba, RgbaImage, SubImage, GenericImageView};
use pollster::block_on;
use winit::dpi::{PhysicalPosition, PhysicalSize};

pub trait Wait: Future {
	fn wait(self) -> Self::Output;
}

impl<T: Future> Wait for T {
	fn wait(self) -> Self::Output {
		block_on(self)
	}
}

pub trait ToVec {
	type Output;
	fn to_vec(self) -> Self::Output;
}

macro_rules! impl_to_vec {
	($type:ty, $output:ty, $x:ident, $y:ident) => {
		impl ToVec for $type {
			type Output = $output;
			fn to_vec(self) -> Self::Output {
				Self::Output::new(self.$x, self.$y)
			}
		}
	};
}

impl_to_vec!(PhysicalSize<u32>, UVec2, width, height);
impl_to_vec!(PhysicalPosition<f64>, DVec2, x, y);

pub trait Size {
	fn size(&self) -> UVec2;
}

impl Size for RgbaImage {
	fn size(&self) -> UVec2 {
		UVec2::new(self.width(), self.height())
	}
}

pub trait IterPixels {
	fn iter_pixels(&self) -> impl Iterator<Item = Rgba<u8>>;
}

impl IterPixels for SubImage<&RgbaImage> {
	fn iter_pixels(&self) -> impl Iterator<Item = Rgba<u8>> {
		self.pixels().map(|(_, _, pixel)| pixel)
	}
}

pub trait AsBytes {
	fn as_bytes(&self) -> &[u8];
}

impl<T: ?Sized> AsBytes for T {
	fn as_bytes(&self) -> &[u8] {
		unsafe {
			slice::from_raw_parts(self as *const T as *const u8, size_of_val(self))
		}
	}
}
