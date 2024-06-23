use std::{mem::size_of, num::NonZeroU64, slice};
use glam::UVec2;
use wgpu::{
	util::{BufferInitDescriptor, DeviceExt, TextureDataOrder},
	BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
	BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType, BufferUsages,
	Color, ColorTargetState, ColorWrites, CommandEncoder, Device, Extent3d, FragmentState,
	FrontFace, LoadOp, MultisampleState, Operations, PipelineLayout, PipelineLayoutDescriptor,
	PolygonMode, PrimitiveState, PrimitiveTopology, Queue, RenderPass, RenderPassColorAttachment,
	RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, ShaderModule,
	ShaderModuleDescriptor, ShaderSource, ShaderStages, StoreOp, TextureDescriptor,
	TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
	TextureViewDescriptor, TextureViewDimension, VertexAttribute, VertexBufferLayout, VertexFormat,
	VertexState, VertexStepMode,
};
use crate::ext::AsBytes;

pub fn buffer(device: &Device, contents: &[u8], usage: BufferUsages) -> Buffer {
	device.create_buffer_init(&BufferInitDescriptor { contents, label: None, usage })
}

pub fn texture_layout_entry(view_dimension: TextureViewDimension) -> (BindingType, ShaderStages) {
	(
		BindingType::Texture {
			multisampled: false,
			sample_type: TextureSampleType::Float { filterable: false },
			view_dimension,
		},
		ShaderStages::FRAGMENT,
	)
}

pub fn uniform_layout_entry<T>(visibility: ShaderStages) -> (BindingType, ShaderStages) {
	(
		BindingType::Buffer {
			has_dynamic_offset: false,
			min_binding_size: NonZeroU64::new(size_of::<T>() as u64),
			ty: BufferBindingType::Uniform,
		},
		visibility,
	)
}

pub fn bind_group_layout(
	device: &Device, entries: Vec<(BindingType, ShaderStages)>,
) -> BindGroupLayout {
	device.create_bind_group_layout(&BindGroupLayoutDescriptor {
		label: None,
		entries: &entries
			.into_iter()
			.enumerate()
			.map(|(index, (ty, visibility))| BindGroupLayoutEntry {
				binding: index as u32, count: None, ty, visibility,
			})
			.collect::<Vec<_>>(),
	})
}

pub fn texture_view(
	device: &Device, queue: &Queue, size: UVec2, depth_or_array_layers: u32, data: &[u8],
) -> TextureView {
	device.create_texture_with_data(
		queue,
		&TextureDescriptor {
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
			usage: TextureUsages::TEXTURE_BINDING,
			view_formats: &[],
		},
		TextureDataOrder::default(),
		data,
	).create_view(&TextureViewDescriptor::default())
}

pub fn bind_group(
	device: &Device, layout: &BindGroupLayout, entries: Vec<BindingResource>,
) -> BindGroup {
	device.create_bind_group(&BindGroupDescriptor {
		label: None,
		layout,
		entries: &entries.into_iter().enumerate().map(|(index, resource)| BindGroupEntry {
			binding: index as u32,
			resource,
		}).collect::<Vec<_>>(),
	})
}

pub fn shader(device: &Device, source: &str) -> ShaderModule {
	device.create_shader_module(ShaderModuleDescriptor {
		label: None,
		source: ShaderSource::Wgsl(source.into()),
	})
}

const COLOR_TARGET: Option<ColorTargetState> = Some(ColorTargetState {
	blend: None,
	format: TextureFormat::Bgra8UnormSrgb,
	write_mask: ColorWrites::all(),
});

pub fn fragment_state(module: &ShaderModule) -> Option<FragmentState> {
	Some(FragmentState { entry_point: "fs_main", module, targets: slice::from_ref(&COLOR_TARGET) })
}

const U16X2_LAYOUT: VertexBufferLayout = VertexBufferLayout {
	array_stride: VertexFormat::Uint16x2.size(),
	attributes: &[VertexAttribute {
		format: VertexFormat::Uint16x2,
		offset: 0,
		shader_location: 0,
	}],
	step_mode: VertexStepMode::Vertex,
};

pub fn vertex_state(module: &ShaderModule) -> VertexState {
	VertexState { buffers: slice::from_ref(&U16X2_LAYOUT), entry_point: "vs_main", module }
}

pub fn pipeline_layout(device: &Device, bind_group_layout: & BindGroupLayout) -> PipelineLayout {
	device.create_pipeline_layout(&PipelineLayoutDescriptor {
		bind_group_layouts: &[bind_group_layout],
		label: None,
		push_constant_ranges: &[],
	})
}

pub fn pipeline(
	device: &Device,
	primitive: PrimitiveState,
	bind_group_layout: & BindGroupLayout,
	shader_source: &str,
) -> RenderPipeline {
	let shader = shader(device, shader_source);
	device.create_render_pipeline(&RenderPipelineDescriptor {
		label: None,
		multiview: None,
		depth_stencil: None,
		primitive,
		multisample: MultisampleState::default(),
		layout: Some(&pipeline_layout(device, bind_group_layout)),
		vertex: vertex_state(&shader),
		fragment: fragment_state(&shader),
	})
}

pub fn render_pass<'a>(encoder: &'a mut CommandEncoder, view: &'a TextureView) -> RenderPass<'a> {
	encoder.begin_render_pass(&RenderPassDescriptor {
		label: None,
		color_attachments: &[Some(RenderPassColorAttachment {
			ops: Operations {
				load: LoadOp::Clear(Color::BLACK),
				store: StoreOp::Store,
			},
			resolve_target: None,
			view,
		})],
		depth_stencil_attachment: None,
		timestamp_writes: None,
		occlusion_query_set: None,
	})
}

pub struct BufferVal<T> {
	pub val: T,
	pub buffer: Buffer,
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
	
	pub fn modify_write(&mut self, queue: &Queue, mod_fn: fn(&mut T)) {
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

pub const PRIMITIVE_STATE: PrimitiveState = PrimitiveState {
	conservative: false,
	cull_mode: None,
	front_face: FrontFace::Ccw,
	polygon_mode: PolygonMode::Fill,
	strip_index_format: None,
	topology: PrimitiveTopology::TriangleStrip,
	unclipped_depth: false,
};

impl PipelineAndLayout {
	pub fn new(
		device: &Device, shader_source: &str, entries: Vec<(BindingType, ShaderStages)>,
	) -> Self {
		let layout = bind_group_layout(device, entries);
		Self { pipeline: pipeline(device, PRIMITIVE_STATE, &layout, shader_source), layout }
	}
}
