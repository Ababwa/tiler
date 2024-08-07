use std::{time::Duration, num::NonZeroU32, thread::{sleep, spawn}, sync::{Arc, mpsc::{channel, TryRecvError}}};
use glam::{DVec2, UVec2};
use wgpu::{
	CommandEncoder, CommandEncoderDescriptor, Device, DeviceDescriptor, Features, Instance, Limits, LoadOp, Operations,
	PowerPreference, Queue, RenderPassColorAttachment, RenderPassDescriptor, RequestAdapterOptions, StoreOp,
	TextureFormat, TextureView, TextureViewDescriptor,
};
use winit::{
	dpi::PhysicalSize, event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
	event_loop::{EventLoop, EventLoopWindowTarget}, keyboard::{KeyCode, ModifiersState, PhysicalKey},
	window::{Icon, Window, WindowBuilder},
};
use crate::ext::{ToVec, Wait};

fn sb_surface(window: &Window, size: PhysicalSize<u32>) -> softbuffer::Surface<&Window, &Window> {
	let mut surface = softbuffer::Surface::new(&softbuffer::Context::new(window).expect("sb context"), window)
		.expect("sb surface");
	surface
		.resize(
			NonZeroU32::new(size.width).expect("nonzero window width"),
			NonZeroU32::new(size.height).expect("nonzero window height"),
		)
		.expect("sb resize");
	surface
}

pub trait Gui {
	fn resize(&mut self, window_size: UVec2, queue: &Queue);
	fn modifiers(&mut self, modifers: ModifiersState);
	fn key(
		&mut self, window_size: UVec2, device: &Device, queue: &Queue, target: &EventLoopWindowTarget<()>,
		keycode: KeyCode, state: ElementState, repeat: bool,
	);
	fn mouse_button(&mut self, button: MouseButton, state: ElementState);
	fn mouse_moved(&mut self, queue: &Queue, position: DVec2);
	fn mouse_wheel(&mut self, queue: &Queue, delta: MouseScrollDelta);
	fn render(&mut self, encoder: &mut CommandEncoder, view: &TextureView);
	fn gui(&mut self, window_size: UVec2, device: &Device, queue: &Queue, ctx: &egui::Context);
}

pub fn run<T: Into<String>, G: Gui, F: FnOnce(&Device) -> G>(title: T, icon: Icon, make_gui: F) {
	env_logger::init();
	let event_loop = EventLoop::new().expect("new event loop");
	let window = WindowBuilder::new()
		.with_title(title)
		.with_min_inner_size(PhysicalSize::new(1, 1))
		.with_window_icon(Some(icon))
		.build(&event_loop)
		.expect("build window");
	let mut window_size = window.inner_size();
	let window = Arc::new(window);
	let painter_window = window.clone();
	let (tx, rx) = channel();
	let painter = spawn(move || {//something to look at during setup
		let mut surface = sb_surface(&painter_window, window_size);
		let width = window_size.width as usize;
		let mut t = 0;
		while let Err(TryRecvError::Empty) = rx.try_recv() {
			let mut buffer = surface.buffer_mut().expect("sb buffer_mut loop");
			for i in 0..buffer.len() {
				buffer[i] = ((((i % width) + (i / width) + 100000000 - t) % 46 / 23) * 0x111111 + 0x222222) as u32;
			}
			buffer.present().expect("sb present loop");
			t += 1;
			sleep(Duration::from_millis(10));
		}
	});
	let instance = Instance::default();
	let surface = instance.create_surface(&window).expect("create surface");//2000ms
	let adapter = instance
		.request_adapter(
			&RequestAdapterOptions {
				power_preference: PowerPreference::HighPerformance,
				force_fallback_adapter: false,
				compatible_surface: Some(&surface),
			},
		)
		.wait()
		.expect("request adapter");//430ms
	let (device, queue) = adapter
		.request_device(
			&DeviceDescriptor {
				label: None,
				required_features: Features::empty(),
				required_limits: Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits()),
			},
			None,
		)
		.wait()
		.expect("request device");//250ms
	let mut config = surface
		.get_default_config(&adapter, window_size.width, window_size.height)
		.expect("get default config");
	surface.configure(&device, &config);//250ms
	let egui_ctx = egui::Context::default();
	let mut egui_input_state = egui_winit::State::new(egui_ctx.clone(), egui_ctx.viewport_id(), &window, None, None);
	let mut egui_renderer = egui_wgpu::Renderer::new(&device, TextureFormat::Bgra8UnormSrgb, None, 1);
	let mut gui = make_gui(&device);
	tx.send(()).expect("signal painter");
	painter.join().expect("join painter");
	let result = event_loop.run(|event, target| match event {
		Event::WindowEvent { event, .. } => if !egui_input_state.on_window_event(&window, &event).consumed {
			match event {
				WindowEvent::CloseRequested => target.exit(),
				WindowEvent::ModifiersChanged(modifiers) => gui.modifiers(modifiers.state()),
				WindowEvent::MouseInput { button, state, .. } => gui.mouse_button(button, state),
				WindowEvent::CursorMoved { position, .. } => gui.mouse_moved(&queue, position.to_vec()),
				WindowEvent::MouseWheel { delta, .. } => gui.mouse_wheel(&queue, delta),
				WindowEvent::KeyboardInput {
					event: KeyEvent { repeat, physical_key: PhysicalKey::Code(keycode), state, .. }, ..
				} => gui.key(window_size.to_vec(), &device, &queue, target, keycode, state, repeat),
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
						textures_delta: egui::TexturesDelta { set, free },
						shapes,
						pixels_per_point,
						..
					} = egui_ctx.run(egui_input, |ctx| gui.gui(window_size.to_vec(), &device, &queue, ctx));
					let screen_desc = egui_wgpu::ScreenDescriptor {
						size_in_pixels: window_size.into(),
						pixels_per_point,
					};
					egui_input_state.handle_platform_output(&window, platform_output);
					for (id, delta) in &set {
						egui_renderer.update_texture(&device, &queue, *id, delta);
					}
					let egui_tris = egui_ctx.tessellate(shapes, pixels_per_point);
					egui_renderer.update_buffers(&device, &queue, &mut encoder, &egui_tris, &screen_desc);
					let mut rpass = encoder.begin_render_pass(
						&RenderPassDescriptor {
							label: None,
							color_attachments: &[
								Some(RenderPassColorAttachment {
									view,
									resolve_target: None,
									ops: Operations {
										load: LoadOp::Load,
										store: StoreOp::Store,
									},
								}),
							],
							depth_stencil_attachment: None,
							timestamp_writes: None,
							occlusion_query_set: None,
						},
					);
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
	});
	result.expect("run event loop");
}
