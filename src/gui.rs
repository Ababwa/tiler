use std::num::NonZeroU32;
use glam::{DVec2, UVec2};
use softbuffer::SoftBufferError;
use wgpu::{
	CommandEncoder, CommandEncoderDescriptor, Device, DeviceDescriptor, Features, Instance, Limits,
	LoadOp, Operations, PowerPreference, Queue, RenderPassColorAttachment, RenderPassDescriptor,
	RequestAdapterOptions, StoreOp, TextureFormat, TextureView, TextureViewDescriptor,
};
use winit::{
	dpi::PhysicalSize,
	event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
	event_loop::{EventLoop, EventLoopWindowTarget},
	keyboard::{KeyCode, ModifiersState, PhysicalKey},
	window::{Window, WindowBuilder},
};
use crate::ext::{ToVec, Wait};

fn fill_black(window: &Window) -> Result<(), SoftBufferError> {
	let size = window.inner_size();
	let mut surface = softbuffer::Surface::new(&softbuffer::Context::new(window)?, window)?;
	surface.resize(NonZeroU32::new(size.width).unwrap(), NonZeroU32::new(size.height).unwrap())?;
	let mut buffer = surface.buffer_mut()?;
	buffer.fill(0);
	Ok(buffer.present()?)
}

pub trait Gui {
	fn resize(&mut self, window_size: UVec2, queue: &Queue);
	fn modifiers(&mut self, modifers: ModifiersState);
	fn key(
		&mut self,
		window_size: UVec2,
		queue: &Queue,
		target: &EventLoopWindowTarget<()>,
		keycode: KeyCode,
		state: ElementState,
	);
	fn mouse_button(&mut self, button: MouseButton, state: ElementState);
	fn mouse_moved(&mut self, queue: &Queue, position: DVec2);
	fn mouse_wheel(&mut self, queue: &Queue, delta: MouseScrollDelta);
	fn render(&mut self, encoder: &mut CommandEncoder, view: &TextureView);
	fn egui(&mut self, window_size: UVec2, device: &Device, queue: &Queue, ctx: &egui::Context);
}

pub fn run<T: Into<String>, G: Gui, F: FnOnce(&Device) -> G>(title: T, make_gui: F) {
	env_logger::init();
	let event_loop = EventLoop::new().expect("new event loop");
	let window = WindowBuilder::new()
		.with_title(title)
		.with_min_inner_size(PhysicalSize::new(1, 1))
		.build(&event_loop)
		.expect("build window");
	fill_black(&window).expect("fill black");
	let mut window_size = window.inner_size();
	let instance = Instance::default();
	let surface = instance.create_surface(&window).expect("create surface");
	let adapter = instance.request_adapter(&RequestAdapterOptions {
		power_preference: PowerPreference::HighPerformance,
		force_fallback_adapter: false,
		compatible_surface: Some(&surface),
	}).wait().expect("request adapter");
	let (device, queue) = adapter.request_device(
		&DeviceDescriptor {
			label: None,
			required_features: Features::empty(),
			required_limits: Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits()),
		},
		None,
	).wait().expect("request device");
	let mut config = surface.get_default_config(&adapter, window_size.width, window_size.height).expect("get default config");
	surface.configure(&device, &config);
	let egui_ctx = egui::Context::default();
	let mut egui_input_state = egui_winit::State::new(egui_ctx.clone(), egui_ctx.viewport_id(), &window, None, None);
	let mut egui_renderer = egui_wgpu::Renderer::new(&device, TextureFormat::Bgra8UnormSrgb, None, 1);
	let mut gui = make_gui(&device);
	event_loop.run(|event, target| match event {
		Event::WindowEvent { event, .. } => if !egui_input_state.on_window_event(&window, &event).consumed {
			match event {
				WindowEvent::CloseRequested => target.exit(),
				WindowEvent::ModifiersChanged(modifiers) => gui.modifiers(modifiers.state()),
				WindowEvent::MouseInput { button, state, .. } => gui.mouse_button(button, state),
				WindowEvent::CursorMoved { position, .. } => gui.mouse_moved(&queue, position.to_vec()),
				WindowEvent::MouseWheel { delta, .. } => gui.mouse_wheel(&queue, delta),
				WindowEvent::KeyboardInput { event: KeyEvent { repeat, physical_key: PhysicalKey::Code(keycode), state, .. }, .. } => if !repeat {
					gui.key(window_size.to_vec(), &queue, target, keycode, state);
				},
				WindowEvent::Resized(new_size) => {
					window_size = new_size;
					config.width = window_size.width;
					config.height = window_size.height;
					surface.configure(&device, &config);
					gui.resize(window_size.to_vec(), &queue);
				},
				WindowEvent::RedrawRequested => {
					let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
					let frame = surface.get_current_texture().expect("get current texture");
					let view = &frame.texture.create_view(&TextureViewDescriptor::default());
					gui.render(&mut encoder, view);
					let egui_input = egui_input_state.take_egui_input(&window);
					let egui::FullOutput {
						platform_output,
						shapes,
						pixels_per_point,
						textures_delta: egui::TexturesDelta { set, free },
						..
					} = egui_ctx.run(egui_input, |ctx| gui.egui(window_size.to_vec(), &device, &queue, ctx));
					let screen_desc = egui_wgpu::ScreenDescriptor { pixels_per_point, size_in_pixels: window_size.into() };
					egui_input_state.handle_platform_output(&window, platform_output);
					for (id, delta) in &set {
						egui_renderer.update_texture(&device, &queue, *id, delta);
					}
					let egui_tris = egui_ctx.tessellate(shapes, pixels_per_point);
					egui_renderer.update_buffers(&device, &queue, &mut encoder, &egui_tris, &screen_desc);
					let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
						label: None,
						timestamp_writes: None,
						occlusion_query_set: None,
						depth_stencil_attachment: None,
						color_attachments: &[Some(RenderPassColorAttachment {
							view,
							resolve_target: None,
							ops: Operations { load: LoadOp::Load, store: StoreOp::Store },
						})],
					});
					egui_renderer.render(&mut rpass, &egui_tris, &screen_desc);
					drop(rpass);
					for id in &free {
						egui_renderer.free_texture(id);
					}
					queue.submit([encoder.finish()]);
					frame.present();
					window.request_redraw();
				},
				_ => {},
			}
		},
		_ => {},
	}).expect("run event loop");
}
