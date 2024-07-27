use std::{mem::size_of, num::NonZeroU64};
use glam::UVec2;
use wgpu::{
	util::{BufferInitDescriptor, DeviceExt, TextureDataOrder},
	BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
	BindingResource, BindingType, Buffer, BufferBindingType, BufferUsages, Color, ColorTargetState, ColorWrites,
	CommandEncoder, Device, Extent3d, FragmentState, FrontFace, LoadOp, MultisampleState, Operations,
	PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology, Queue, RenderPass,
	RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, ShaderModule,
	ShaderModuleDescriptor, ShaderSource, ShaderStages, StoreOp, TextureDescriptor, TextureDimension, TextureFormat,
	TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension, VertexAttribute,
	VertexBufferLayout, VertexFormat, VertexState, VertexStepMode,
};
use crate::ext::AsBytes;

pub fn shader(device: &Device, shader_source: &str) -> ShaderModule {
	device.create_shader_module(
		ShaderModuleDescriptor {
			label: None,
			source: ShaderSource::Wgsl(shader_source.into()),
		},
	)
}

pub fn buffer(device: &Device, contents: &[u8], usage: BufferUsages) -> Buffer {
	device.create_buffer_init(
		&BufferInitDescriptor {
			label: None,
			contents,
			usage,
		},
	)
}

pub fn texture_layout_entry(view_dimension: TextureViewDimension) -> (BindingType, ShaderStages) {
	(
		BindingType::Texture {
			sample_type: TextureSampleType::Float { filterable: false },
			view_dimension,
			multisampled: false,
		},
		ShaderStages::FRAGMENT,
	)
}

pub fn uniform_layout_entry<T>(visibility: ShaderStages) -> (BindingType, ShaderStages) {
	(
		BindingType::Buffer {
			ty: BufferBindingType::Uniform,
			has_dynamic_offset: false,
			min_binding_size: NonZeroU64::new(size_of::<T>() as u64),
		},
		visibility,
	)
}

pub fn bind_group_layout(device: &Device, entries: Vec<(BindingType, ShaderStages)>) -> BindGroupLayout {
	device.create_bind_group_layout(
		&BindGroupLayoutDescriptor {
			label: None,
			entries: &entries
				.into_iter()
				.enumerate()
				.map(|(index, (ty, visibility))| {
					BindGroupLayoutEntry {
						binding: index as u32,
						visibility,
						ty,
						count: None,
					}
				})
				.collect::<Vec<_>>(),
		},
	)
}

pub fn texture_desc(size: UVec2, depth_or_array_layers: u32, usage: TextureUsages) -> TextureDescriptor<'static> {
	TextureDescriptor {
		label: None,
		size: Extent3d {
			width: size.x,
			height: size.y,
			depth_or_array_layers,
		},
		mip_level_count: 1,
		sample_count: 1,
		dimension: TextureDimension::D2,
		format: TextureFormat::Rgba8UnormSrgb,
		usage,
		view_formats: &[],
	}
}

pub fn bindable_texture_view(
	device: &Device, queue: &Queue, size: UVec2, depth_or_array_layers: u32, data: &[u8],
) -> TextureView {
	device
		.create_texture_with_data(
			queue,
			&texture_desc(size, depth_or_array_layers, TextureUsages::TEXTURE_BINDING),
			TextureDataOrder::default(),
			data,
		)
		.create_view(&TextureViewDescriptor::default())
}

pub fn bind_group(device: &Device, layout: &BindGroupLayout, entries: Vec<BindingResource>) -> BindGroup {
	device.create_bind_group(
		&BindGroupDescriptor {
			label: None,
			layout,
			entries: &entries
				.into_iter()
				.enumerate()
				.map(|(index, resource)| {
					BindGroupEntry {
						binding: index as u32,
						resource,
					}
				})
				.collect::<Vec<_>>(),
		},
	)
}

pub const PRIMITIVE_STATE: PrimitiveState = PrimitiveState {
	topology: PrimitiveTopology::TriangleStrip,
	strip_index_format: None,
	front_face: FrontFace::Ccw,
	cull_mode: None,
	unclipped_depth: false,
	polygon_mode: PolygonMode::Fill,
	conservative: false,
};

pub fn pipeline(
	device: &Device, bind_group_layout: & BindGroupLayout, module: &ShaderModule, format: TextureFormat,
) -> RenderPipeline {
	device.create_render_pipeline(
		&RenderPipelineDescriptor {
			label: None,
			layout: Some(&device.create_pipeline_layout(
				&PipelineLayoutDescriptor {
					label: None,
					bind_group_layouts: &[bind_group_layout],
					push_constant_ranges: &[],
				},
			)),
			vertex: VertexState {
				module,
				entry_point: "vs_main",
				buffers: &[
					VertexBufferLayout {
						array_stride: VertexFormat::Uint16x2.size(),
						step_mode: VertexStepMode::Vertex,
						attributes: &[
							VertexAttribute {
								format: VertexFormat::Uint16x2,
								offset: 0,
								shader_location: 0,
							},
						],
					}
				],
			},
			primitive: PRIMITIVE_STATE,
			depth_stencil: None,
			multisample: MultisampleState::default(),
			fragment: Some(FragmentState {
				entry_point: "fs_main",
				module,
				targets: &[
					Some(ColorTargetState {
						format,
						blend: None,
						write_mask: ColorWrites::all(),
					}),
				],
			}),
			multiview: None,
		},
	)
}

pub fn render_pass<'a>(encoder: &'a mut CommandEncoder, view: &'a TextureView, clear: Color) -> RenderPass<'a> {
	encoder.begin_render_pass(
		&RenderPassDescriptor {
			label: None,
			color_attachments: &[
				Some(RenderPassColorAttachment {
					view,
					resolve_target: None,
					ops: Operations {
						load: LoadOp::Clear(clear),
						store: StoreOp::Store,
					},
				}),
			],
			depth_stencil_attachment: None,
			timestamp_writes: None,
			occlusion_query_set: None,
		},
	)
}

pub struct BufferVal<T> {
	pub buffer: Buffer,
	pub val: T,
}

impl<T> BufferVal<T> {
	pub fn entry(&self) -> BindingResource {
		self.buffer.as_entire_binding()
	}
}

impl<T: AsBytes> BufferVal<T> {
	pub fn new(device: &Device, val: T) -> Self {
		Self {
			buffer: buffer(device, val.as_bytes(), BufferUsages::UNIFORM | BufferUsages::COPY_DST),
			val,
		}
	}
	
	pub fn write(&self, queue: &Queue) {
		queue.write_buffer(&self.buffer, 0, self.val.as_bytes());
	}
	
	pub fn set_write(&mut self, queue: &Queue, val: T) {
		self.val = val;
		self.write(queue);
	}
	
	pub fn modify_write<F: Fn(&mut T)>(&mut self, queue: &Queue, mod_fn: F) {
		mod_fn(&mut self.val);
		self.write(queue);
	}
}

impl<T: AsBytes, const N: usize> BufferVal<Box<[T; N]>> {
	pub fn write_range(&self, queue: &Queue, start: usize, end: usize) {
		const ALIGN_4_MASK: usize = !3;
		let byte_start = (start * size_of::<T>()) & ALIGN_4_MASK;
		let byte_end = (end * size_of::<T>() + 3) & ALIGN_4_MASK;
		let bytes = &self.val.as_bytes()[byte_start..byte_end];
		queue.write_buffer(&self.buffer, byte_start as u64, bytes);
	}
}

pub struct PipelineAndLayout {
	pub pipeline: RenderPipeline,
	pub layout: BindGroupLayout,
}

impl PipelineAndLayout {
	pub fn new(device: &Device, module: &ShaderModule, entries: Vec<(BindingType, ShaderStages)>) -> Self {
		let layout = bind_group_layout(device, entries);
		Self {
			pipeline: pipeline(device, &layout, module, TextureFormat::Bgra8UnormSrgb),
			layout,
		}
	}
}
