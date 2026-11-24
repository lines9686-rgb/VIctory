//! Depth planes kept in an unordered access twin of fewer bits. The
//! console's 16-bit depth with stencil is a D24S8 target on the host
//! (`resource::pc_format`), and the title's compute shaders write its depth
//! plane through 16-bit views (R16_UNORM): its twin is R16_TYPELESS, the
//! console's bits, which no copy moves into a 24-bit plane. Passes of ours
//! do: a full-screen draw writes the twin's values as depth (stencil kept),
//! and a compute pass writes the depth plane into the twin.
//!
//! Draws that test the depth equal to what the title wrote (tiled lighting
//! passes material ids so) pass where they did on the console as long as
//! they put their depth at the float the twin's value reads as (k/65535):
//! the 24-bit plane stores the depth a pixel shader outputs and the one a
//! draw rasterizes alike.
//!
//! The draw pass changes the list's targets, viewport, scissor and
//! topology and sets them back as the title had them
//! (`HostList::restore_fixed`); both passes leave pipeline, root signature
//! and descriptor heaps of their own, which the title's next draw or
//! dispatch binds again (`HostList::clobber`).
//!
//! shortcut: one subresource (no mips, no array slices); other such
//! targets get no 16-bit twin and their depth-plane writes are dropped.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{PCSTR, s};

use crate::host::{HostList, host};

/// The twin's values as depth: a triangle over the target whose pixels are
/// at the depth the twin holds there.
const TO_DEPTH: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility=SHADER_VISIBILITY_PIXEL)"
Texture2D<float> twin : register(t0);
[RootSignature(RS)]
float4 vs(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
[RootSignature(RS)]
float ps(float4 at : SV_Position) : SV_Depth {
    return twin.Load(int3(at.xy, 0));
}
"#;

/// The depth plane into the twin (rounded to its 16 bits).
const TO_TWIN: &str = r#"
#define RS "DescriptorTable(SRV(t0)), DescriptorTable(UAV(u0))"
Texture2D<float> depth : register(t0);
RWTexture2D<unorm float> twin : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint width, height;
    twin.GetDimensions(width, height);
    if (id.x < width && id.y < height) twin[id.xy] = depth.Load(int3(id.xy, 0));
}
"#;

/// Depth families whose depth plane a 16-bit twin holds: the typeless
/// format, the depth view format and the depth plane's shader view format.
const PLANES: [(u32, DXGI_FORMAT, DXGI_FORMAT); 1] = [(
    44,
    DXGI_FORMAT_D24_UNORM_S8_UINT,
    DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
)];

/// The twin's format: the console's 16 bits.
const R16_TYPELESS: i32 = 53;

/// The views' slots in [`Views::heap`].
const TWIN_SRV: u32 = 0;
const DEPTH_SRV: u32 = 1;
const TWIN_UAV: u32 = 2;

/// What the passes between a depth target and its 16-bit twin read and
/// write through: a shader-visible heap with the twin's shader view
/// (R16_UNORM), the depth plane's shader view and the twin's unordered
/// access view (R16_UNORM); and the target's size and family.
pub(crate) struct Views {
    heap: ID3D12DescriptorHeap,
    increment: u32,
    width: u32,
    height: u32,
    /// Index in [`PLANES`].
    plane: usize,
    /// Whether shaders may read the depth target (not denied shader
    /// resources): else nothing goes from it to the twin.
    readable: bool,
}

impl Views {
    /// The views for host depth target `depth` and its twin `twin`, if the
    /// twin has fewer bits than the depth plane (R16 for a D24S8 target).
    pub(crate) fn new(depth: &ID3D12Resource, twin: &ID3D12Resource) -> Option<Views> {
        let device = &host()?.device;
        // SAFETY: live resources.
        let (d, t) = unsafe { (depth.GetDesc(), twin.GetDesc()) };
        let family = crate::resource::family(d.Format.0 as u32);
        let plane = PLANES.iter().position(|&(f, ..)| f == family)?;
        if t.Format.0 != R16_TYPELESS || d.MipLevels != 1 || d.DepthOrArraySize != 1 {
            return None;
        }
        let readable = !d.Flags.contains(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE);
        // SAFETY: plain creation; the views name live resources (or none).
        unsafe {
            let heap: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 3,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .inspect_err(|e| tracing::warn!(error = %e, "no views for a 16-bit depth twin"))
                .ok()?;
            let increment =
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            let at = |slot: u32| {
                let mut handle = heap.GetCPUDescriptorHandleForHeapStart();
                handle.ptr += (slot * increment) as usize;
                handle
            };
            let srv = |format| D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MipLevels: 1,
                        ..Default::default()
                    },
                },
            };
            let twin_srv = srv(DXGI_FORMAT_R16_UNORM);
            device.CreateShaderResourceView(twin, Some(&raw const twin_srv), at(TWIN_SRV));
            // A depth target denied shader resources: a null view, never read.
            let read = readable.then_some(depth);
            let depth_srv = srv(PLANES[plane].2);
            device.CreateShaderResourceView(read, Some(&raw const depth_srv), at(DEPTH_SRV));
            let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R16_UNORM,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                ..Default::default()
            };
            let counter = None::<&ID3D12Resource>;
            device.CreateUnorderedAccessView(twin, counter, Some(&raw const uav), at(TWIN_UAV));
            crate::host::name(&heap, || "views of a 16-bit depth twin".into());
            Some(Views {
                heap,
                increment,
                width: d.Width as u32,
                height: d.Height,
                plane,
                readable,
            })
        }
    }

    fn gpu(&self, slot: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a shader-visible heap of ours.
        let mut handle = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        handle.ptr += u64::from(slot * self.increment);
        handle
    }
}

/// One of our passes: its root signature (inside the shader) and pipeline.
struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn compile(source: &str, entry: PCSTR, model: PCSTR) -> windows::core::Result<ID3DBlob> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("victory"),
            None,
            None,
            entry,
            model,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("16-bit depth twin pass: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    code.ok_or_else(windows::core::Error::empty)
}

/// A blob's bytes as shader bytecode.
fn bytecode(blob: &ID3DBlob) -> D3D12_SHADER_BYTECODE {
    // SAFETY: a live blob.
    unsafe {
        D3D12_SHADER_BYTECODE {
            pShaderBytecode: blob.GetBufferPointer(),
            BytecodeLength: blob.GetBufferSize(),
        }
    }
}

/// The draw that writes the twin's values as depth, for the depth view
/// format of `PLANES[plane]`: depth always passes and is written, stencil
/// is off and kept, no color targets.
fn to_depth_pass(plane: usize) -> Option<&'static Pass> {
    static PASSES: [std::sync::OnceLock<Option<Pass>>; PLANES.len()] =
        [const { std::sync::OnceLock::new() }; PLANES.len()];
    let make = || -> windows::core::Result<Pass> {
        let device = &host().ok_or_else(windows::core::Error::empty)?.device;
        let vs = compile(TO_DEPTH, s!("vs"), s!("vs_5_1"))?;
        let ps = compile(TO_DEPTH, s!("ps"), s!("ps_5_1"))?;
        // SAFETY: the blobs' own bytes; the root signature is inside them.
        unsafe {
            let code =
                std::slice::from_raw_parts(vs.GetBufferPointer().cast::<u8>(), vs.GetBufferSize());
            let root: ID3D12RootSignature = device.CreateRootSignature(0, code)?;
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: bytecode(&vs),
                PS: bytecode(&ps),
                BlendState: D3D12_BLEND_DESC::default(),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                    DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                    StencilEnable: false.into(),
                    StencilReadMask: 0,
                    StencilWriteMask: 0,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 0,
                DSVFormat: PLANES[plane].1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let pso = device.CreateGraphicsPipelineState(&desc)?;
            Ok(Pass { root, pso })
        }
    };
    PASSES[plane]
        .get_or_init(|| {
            make()
                .inspect_err(|e| {
                    tracing::warn!(error = %e, "no pass to write 16-bit depth twins back: their writes are dropped")
                })
                .ok()
        })
        .as_ref()
}

/// The compute pass that writes a depth plane into its 16-bit twin.
fn to_twin_pass() -> Option<&'static Pass> {
    static PASS: std::sync::OnceLock<Option<Pass>> = std::sync::OnceLock::new();
    let make = || -> windows::core::Result<Pass> {
        let device = &host().ok_or_else(windows::core::Error::empty)?.device;
        let cs = compile(TO_TWIN, s!("main"), s!("cs_5_1"))?;
        // SAFETY: the blob's own bytes; the root signature is inside it.
        unsafe {
            let code =
                std::slice::from_raw_parts(cs.GetBufferPointer().cast::<u8>(), cs.GetBufferSize());
            let root: ID3D12RootSignature = device.CreateRootSignature(0, code)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: bytecode(&cs),
                ..Default::default()
            })?;
            Ok(Pass { root, pso })
        }
    };
    PASS.get_or_init(|| {
        make()
            .inspect_err(|e| {
                tracing::warn!(error = %e, "no pass to fill 16-bit depth twins: they start stale")
            })
            .ok()
    })
    .as_ref()
}

/// Whether the pass to the depth (`to_depth`) or to the twin can run: it
/// was made, and for the twin the depth target is readable.
pub(crate) fn ready(views: &Views, to_depth: bool) -> bool {
    if to_depth {
        return to_depth_pass(views.plane).is_some();
    }
    if !views.readable {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!("a depth target denied shader resources: its 16-bit twin starts stale")
        });
        return false;
    }
    to_twin_pass().is_some()
}

/// Records, in `list`, the twin's values written as the depth of `depth`
/// (in the depth write state; the twin readable by pixel shaders). The
/// title's targets, viewports, scissors and topology are set again after.
pub(crate) fn to_depth(list: &HostList, views: &Views, depth: &ID3D12Resource) {
    let (Some(pass), Some(host)) = (to_depth_pass(views.plane), host()) else {
        return;
    };
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| {
        tracing::info!(
            width = views.width,
            height = views.height,
            "a depth plane written from its 16-bit unordered access twin (first time)"
        )
    });
    let dsv = host.dsv(
        depth,
        Some(&D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: PLANES[views.plane].1,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        }),
    );
    // SAFETY: an open list; our heap, root signature and pipeline, a fresh
    // view of a live texture in the depth write state.
    unsafe {
        let l = &list.list;
        l.OMSetRenderTargets(0, None, false, Some(&dsv));
        l.RSSetViewports(&[D3D12_VIEWPORT {
            Width: views.width as f32,
            Height: views.height as f32,
            MaxDepth: 1.0,
            ..Default::default()
        }]);
        l.RSSetScissorRects(&[RECT {
            right: views.width as i32,
            bottom: views.height as i32,
            ..Default::default()
        }]);
        l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        l.SetDescriptorHeaps(&[Some(views.heap.clone())]);
        l.SetGraphicsRootSignature(&pass.root);
        l.SetPipelineState(&pass.pso);
        l.SetGraphicsRootDescriptorTable(0, views.gpu(TWIN_SRV));
        l.DrawInstanced(3, 1, 0, 0);
    }
    list.restore_fixed();
    list.clobber();
}

/// Records, in `list`, the depth plane written into the twin `twin` (the
/// depth readable by non-pixel shaders, the twin in unordered access),
/// then an unordered access barrier on the twin.
pub(crate) fn to_twin(list: &HostList, views: &Views, twin: &ID3D12Resource) {
    let Some(pass) = to_twin_pass() else {
        return;
    };
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| {
        tracing::info!(
            width = views.width,
            height = views.height,
            "a depth plane copied into its 16-bit unordered access twin (first time)"
        )
    });
    // SAFETY: an open list; our heap, root signature and pipeline, live
    // textures in the states the views need.
    unsafe {
        let l = &list.list;
        l.SetDescriptorHeaps(&[Some(views.heap.clone())]);
        l.SetComputeRootSignature(&pass.root);
        l.SetPipelineState(&pass.pso);
        l.SetComputeRootDescriptorTable(0, views.gpu(DEPTH_SRV));
        l.SetComputeRootDescriptorTable(1, views.gpu(TWIN_UAV));
        l.Dispatch(views.width.div_ceil(8), views.height.div_ceil(8), 1);
        l.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    // A borrowed pointer; the barrier never releases it.
                    pResource: std::mem::transmute_copy(twin),
                }),
            },
        }]);
    }
    list.clobber();
}
