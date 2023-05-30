use anyhow::*;
use image::io::Reader;
use std::{borrow::Cow, fs::File, io::BufReader};
use wgpu::{Device, Queue};

const DATA_PER_PIXEL: u32 = 4;
const U8_SIZE: u32 = std::mem::size_of::<u8>() as u32;

fn align_up(num: u32, align: u32) -> u32 {
    (num + align - 1) & !(align - 1)
}

fn compute_and_get_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    buffer: &[u8],
) -> wgpu::Buffer {
    let align_width = align_up(
        width * DATA_PER_PIXEL * U8_SIZE,
        wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
    ) / U8_SIZE;

    let texture_size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader Module"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/compute.wgsl"))),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    view_dimension: wgpu::TextureViewDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    access: wgpu::StorageTextureAccess::WriteOnly,
                },
                count: None,
            },
        ],
    });

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Texture"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
    });

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        buffer,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(DATA_PER_PIXEL * width),
            rows_per_image: Some(height),
        },
        texture_size,
    );

    let output_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Output Texture"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
    });

    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let output_texture_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&output_texture_view),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "basic",
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(width, height, 1);
    }

    let image_texture = wgpu::ImageCopyTextureBase {
        texture: &output_texture,
        mip_level: 0,
        origin: wgpu::Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
    };

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Buffer"),
        size: (align_width * height) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let image_buffer = wgpu::ImageCopyBuffer {
        buffer: &output_buffer,
        layout: wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(align_width),
            rows_per_image: Some(height),
        },
    };

    encoder.copy_texture_to_buffer(image_texture, image_buffer, texture_size);

    queue.submit(Some(encoder.finish()));

    output_buffer
}

async fn get_device_and_queue() -> Result<(Device, Queue)> {
    let instance = wgpu::Instance::default();

    if let Some(adapter) = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
    {
        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .map_err(|err| anyhow!("Request device failed: {}", err))
    } else {
        bail!("No adapters are found that suffice all the 'hard' options.")
    }
}

async fn manipulate_buffer(width: u32, height: u32, buffer: &[u8]) -> Result<Vec<u8>> {
    let (device, queue) = get_device_and_queue().await?;

    let buffer = compute_and_get_texture(&device, &queue, width, height, buffer);

    view_into_buffer(&device, width, height, &buffer).await
}

fn run() -> Result<()> {
    let file = File::open("data/test.png")?;
    let reader = BufReader::new(file);

    let reader = Reader::new(reader).with_guessed_format()?;

    let image = reader.decode()?;

    let width = image.width();
    let height = image.height();
    let mut buffer = image.into_bytes();

    let buffer = futures::executor::block_on(manipulate_buffer(width, height, &mut buffer))?;

    image::save_buffer(
        "data/out.png",
        &buffer,
        width,
        height,
        image::ColorType::Rgba8,
    )?;

    Ok(())
}

fn trim_image_buffer(width: u32, height: u32, buffer: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity((width * height) as usize);

    let align_width = buffer.len() / height as usize;

    for i in 0..height as usize {
        for j in 0..width as usize {
            output.push(buffer[i * align_width + j]);
        }
    }

    output
}

async fn view_into_buffer(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    raw_buffer: &wgpu::Buffer,
) -> Result<Vec<u8>> {
    let slice = raw_buffer.slice(..);

    let (sender, receiver) = futures::channel::oneshot::channel();

    slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    device.poll(wgpu::Maintain::Wait);

    if let std::result::Result::Ok(_) = receiver.await {
        let buffer_view = slice.get_mapped_range();

        let buffer = trim_image_buffer(DATA_PER_PIXEL * width, height, &buffer_view);

        drop(buffer_view);
        raw_buffer.unmap();

        Ok(buffer)
    } else {
        bail!("Couldn't run compute on the GPU.")
    }
}

fn main() {
    run().unwrap();
}
