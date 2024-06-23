use std::future::Future;
use glam::{DVec2, IVec2, UVec2};
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

pub trait PixelsOnly {
	fn pixels_only(&self) -> impl Iterator<Item = Rgba<u8>>;
}

impl PixelsOnly for SubImage<&RgbaImage> {
	fn pixels_only(&self) -> impl Iterator<Item = Rgba<u8>> {
		self.pixels().map(|(_, _, pixel)| pixel)
	}
}

pub trait AsBytes {
	fn as_bytes(&self) -> &[u8];
}

impl<T: AsBytes> AsBytes for [T] {
	fn as_bytes(&self) -> &[u8] {
		unsafe { reinterpret::slice(self) }
	}
}

impl<T: AsBytes, const N: usize> AsBytes for [T; N] {
	fn as_bytes(&self) -> &[u8] {
		unsafe { reinterpret::ref_to_slice(self) }
	}
}

impl<T: AsBytes> AsBytes for Box<T> {
	fn as_bytes(&self) -> &[u8] {
		(self as &T).as_bytes()
	}
}

macro_rules! impl_as_bytes {
	($type:ty) => {
		impl AsBytes for $type {
			fn as_bytes(&self) -> &[u8] {
				unsafe { reinterpret::ref_to_slice(self) }
			}
		}
	};
}

impl_as_bytes!(u16);
impl_as_bytes!(u32);
impl_as_bytes!(UVec2);
impl_as_bytes!(IVec2);
impl_as_bytes!(Rgba<u8>);
