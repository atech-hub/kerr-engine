// Element-wise vector addition: y[i] = a[i] + b[i]
// One thread per element.

struct Params {
    len: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn vec_add(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.len) {
        return;
    }
    y[i] = a[i] + b[i];
}
