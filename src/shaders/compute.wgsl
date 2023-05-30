@group(0) @binding(0)
var textureInput: texture_2d<f32>;
@group(0) @binding(1)
var textureOutput: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2)
var<storage, read> color: vec4<f32>;

@compute @workgroup_size(1)
fn basic(@builtin(global_invocation_id) global_id: vec3<u32>) {
  textureStore(textureOutput, vec2<i32>(i32(global_id.x), i32(global_id.y)), color);
}