//! Heaps, committed and placed resources, and resource sizes.
//!
//! Heaps and resources are host memory and a resource's GPU address is its
//! CPU address, as on the console; textures also get a host copy, and
//! buffers a host copy when a draw reads them (`shadow`).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ffi::c_void;
use std::ptr::NonNull;

use victory_formats::texture;

use crate::object::{
    self, E_INVALIDARG, E_OUTOFMEMORY, HEAP, HRESULT, Iid, Object, RESOURCE, S_OK, Table, This,
    guid,
};

/// The interfaces every device child also answers for.
pub(crate) const PAGEABLE_IIDS: [Iid; 3] = [
    guid(
        0x63ee58fb,
        0x1268,
        0x4835,
        [0x86, 0xda, 0xf0, 0x08, 0xce, 0x62, 0xf0, 0xd6],
    ),
    guid(
        0x905db94b,
        0xa00c,
        0x4140,
        [0x9d, 0xf5, 0x2b, 0x64, 0xca, 0x9e, 0xa3, 0x57],
    ),
    guid(
        0xc4fec28f,
        0x7966,
        0x4e95,
        [0x9f, 0x94, 0xf4, 0x31, 0xcb, 0x56, 0xc3, 0xb8],
    ),
];
const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x6b3b2502,
        0x6e51,
        0x45b3,
        [0x90, 0xee, 0x98, 0x84, 0x26, 0x5e, 0x8d, 0xf3],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];
const RESOURCE_IIDS: [Iid; 4] = [
    guid(
        0x696442be,
        0xa72e,
        0x4059,
        [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

const KIB_64: u64 = 64 << 10;
const MIB_4: u64 = 4 << 20;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapProperties {
    kind: u32,
    cpu_pages: u32,
    pool: u32,
    creation_nodes: u32,
    visible_nodes: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    size: u64,
    properties: HeapProperties,
    alignment: u64,
    flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResourceDesc {
    dimension: u32,
    alignment: u64,
    width: u64,
    height: u32,
    depth_or_array_size: u16,
    mip_levels: u16,
    format: u32,
    sample_count: u32,
    sample_quality: u32,
    layout: u32,
    flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AllocationInfo {
    size: u64,
    alignment: u64,
}

const _: () = assert!(size_of::<HeapProperties>() == 20);
const _: () = assert!(size_of::<HeapDesc>() == 48);
const _: () = assert!(size_of::<ResourceDesc>() == 56);

const BUFFER: u32 = 1;
const TEXTURE_3D: u32 = 4;

/// Bytes per element and the element's width in texels (4 for block
/// compression), by DXGI format number. `None` for 1-bit and video
/// formats. The console adds 116/117 (10-bit float color), 118-120 (16-bit
/// depth with 8-bit stencil, two planes: 3 bytes per texel), 189 (signed
/// 10-bit color), 190 (two 4-bit channels) and 191 (24/8 color).
pub(crate) fn element(format: u32) -> Option<(u64, u64)> {
    Some(match format {
        1..=4 => (16, 1),
        5..=8 => (12, 1),
        9..=22 => (8, 1),
        23..=47 | 67 | 87..=93 | 116 | 117 | 189 | 191 => (4, 1),
        118..=120 => (3, 1),
        48..=59 | 68 | 69 | 85 | 86 | 115 => (2, 1),
        60..=65 | 190 => (1, 1),
        70..=72 | 79..=81 => (8, 4),
        73..=78 | 82..=84 | 94..=99 => (16, 4),
        _ => return None,
    })
}

/// A texture's layout in console memory, as the console's layout library
/// gives it (`victory-hle`'s `xg` serves the same): the description's
/// layout field is `0x100 | swizzle mode`, 1 linear, 0 the library's pick.
pub(crate) fn texture_layout(desc: &ResourceDesc) -> Option<texture::Texture> {
    use texture::{Dimension, Swizzle, TextureDesc};
    const TEXTURE_1D: u32 = 2;
    const TEXTURE_2D: u32 = 3;
    const RENDER_TARGET: u32 = 1;
    const DEPTH: u32 = 2;
    let dimension = match desc.dimension {
        TEXTURE_1D => Dimension::Tex1d,
        TEXTURE_2D => Dimension::Tex2d,
        TEXTURE_3D => Dimension::Tex3d,
        _ => return None,
    };
    let swizzle = match desc.layout {
        0 if dimension == Dimension::Tex1d => Swizzle::LINEAR,
        0 if desc.flags & DEPTH != 0 => Swizzle::SW_64KB_Z_X,
        0 if desc.flags & RENDER_TARGET != 0 => Swizzle::SW_64KB_R_X,
        0 => Swizzle::SW_4KB_S,
        1 => Swizzle::LINEAR,
        l if l & !0xff == 0x100 => Swizzle(l & 0xff),
        _ => return None,
    };
    texture::Texture::new(&TextureDesc {
        dimension,
        width: u32::try_from(desc.width).ok()?,
        height: desc.height,
        depth_or_array: u32::from(desc.depth_or_array_size),
        mips: u32::from(desc.mip_levels),
        format: desc.format,
        samples: desc.sample_count.max(1),
        swizzle,
    })
    .ok()
}

/// Bytes and alignment a resource takes: buffers their width (64 KiB
/// aligned), textures their console layout (formats it does not know:
/// every mip of every array slice and sample, packed, 64 KiB aligned),
/// rounded up to the alignment.
fn footprint(desc: &ResourceDesc) -> Option<AllocationInfo> {
    if desc.dimension != BUFFER
        && let Some(t) = texture_layout(desc)
    {
        // As the console: the layout's own alignment unless more is asked
        // (4 KiB for 4 KiB modes, where PC says 64 KiB). Grand Theft Auto
        // VI keeps these sizes in its texture tables, and its streaming
        // asserts when they disagree with the streamed layout.
        let alignment = desc.alignment.max(t.alignment);
        if !alignment.is_power_of_two() {
            return None;
        }
        return Some(AllocationInfo {
            size: t.size.checked_next_multiple_of(alignment)?,
            alignment,
        });
    }
    let bytes = if desc.dimension == BUFFER {
        desc.width
    } else {
        let (element, block) = element(desc.format)?;
        let (depth, slices) = if desc.dimension == TEXTURE_3D {
            (u64::from(desc.depth_or_array_size), 1)
        } else {
            (1, u64::from(desc.depth_or_array_size))
        };
        let height = u64::from(desc.height);
        let mips = mip_count(desc);
        let mut chain = 0u64;
        for mip in 0..mips.min(64) {
            let w = (desc.width >> mip).max(1).div_ceil(block);
            let h = (height >> mip).max(1).div_ceil(block);
            let d = (depth >> mip).max(1);
            chain = chain.checked_add(w.checked_mul(h)?.checked_mul(d)?.checked_mul(element)?)?;
        }
        chain
            .checked_mul(slices)?
            .checked_mul(u64::from(desc.sample_count.max(1)))?
    };
    if bytes == 0 {
        return None;
    }
    let minimum = if desc.sample_count > 1 { MIB_4 } else { KIB_64 };
    let alignment = desc.alignment.max(minimum);
    if !alignment.is_power_of_two() {
        return None;
    }
    Some(AllocationInfo {
        size: bytes.checked_next_multiple_of(KIB_64)?,
        alignment,
    })
}

/// A format as the host has it. The console's 16-bit depth with stencil
/// (and its typeless views) has no PC format: the 24-bit ones stand in,
/// or with [`d16_depth`] 16-bit depth (D16_UNORM, R16_UNORM) and, for the
/// stencil, R8_UINT (a texture of its own).
pub(crate) fn pc_format(format: u32) -> u32 {
    if !(118..=120).contains(&format) {
        return format;
    }
    let (d24, d16) = match format {
        118 => (45, 55),
        119 => (46, 56),
        _ => (47, 62),
    };
    if d16_depth() { d16 } else { d24 }
}

/// Whether the console's 16-bit depth with stencil (title formats 118-120)
/// is 16-bit depth on the host too (`VICTORY_LOCAL_D16_DEPTH=1`; in tests,
/// per thread: [`D16_FOR_TEST`]): D16_UNORM, which stores the depth a draw
/// rasterizes and the title's 16-bit writes at the same 16-bit values, as
/// the console does (draws testing depth equal pass where they did there),
/// with the stencil in an R8 texture of its own
/// (`HostTexture::stencil_alone`) that draws cannot test. Else D24S8, its
/// depth plane written from the title's 16-bit views by passes of ours
/// (`crate::depth_twin`).
/// LOCAL: a switch to compare the two in the title; drop it (or the D24S8
/// way) once one is proven.
pub(crate) fn d16_depth() -> bool {
    #[cfg(test)]
    {
        D16_FOR_TEST.with(std::cell::Cell::get)
    }
    #[cfg(not(test))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            let on = std::env::var("VICTORY_LOCAL_D16_DEPTH").is_ok_and(|v| v == "1");
            tracing::info!(
                on,
                "the console's 16-bit depth with stencil as host D16, the stencil apart (VICTORY_LOCAL_D16_DEPTH)"
            );
            on
        })
    }
}

#[cfg(test)]
thread_local! {
    /// [`d16_depth`] for the tests on this thread (their resources, views
    /// and pipelines are made on it).
    pub(crate) static D16_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The typeless format of a format's family (itself if it has none).
pub(crate) fn family(format: u32) -> u32 {
    const FAMILIES: [(u32, u32); 22] = [
        (1, 4),
        (5, 8),
        (9, 14),
        (15, 18),
        (19, 22),
        (23, 25),
        (27, 32),
        (33, 38),
        (39, 43),
        (44, 47),
        (48, 52),
        (53, 59),
        (60, 65),
        (70, 72),
        (73, 75),
        (76, 78),
        (79, 81),
        (82, 84),
        (90, 91),
        (92, 93),
        (94, 96),
        (97, 99),
    ];
    match format {
        87 => 90,
        88 => 92,
        f => FAMILIES
            .iter()
            .find(|&&(first, last)| (first..=last).contains(&f))
            .map_or(f, |&(first, _)| first),
    }
}

/// sRGB formats and their linear twins.
pub(crate) const SRGB: [(u32, u32); 3] = [(29, 28), (91, 87), (93, 88)];

/// The format a view of a title texture must name when its host copy is
/// typeless (see [`DEPTH_FORMATS`], [`SRGB`]): the title's own; `None` if
/// the host copy is typed.
pub(crate) fn view_format(this: This) -> Option<u32> {
    let own = pc_format(resource(this).desc.format);
    let host = host_resource(this)?;
    // SAFETY: a live resource.
    let format = unsafe { host.GetDesc() }.Format.0 as u32;
    (format != own).then_some(own)
}

/// Depth formats with their typeless family: depth textures that shaders
/// read are made typeless on the host, which only allows views of
/// typeless depth resources.
pub(crate) const DEPTH_FORMATS: [(u32, u32); 4] = [(20, 19), (40, 39), (45, 44), (55, 53)];

/// Mip levels, the full chain for 0.
fn mip_count(desc: &ResourceDesc) -> u32 {
    let depth = if desc.dimension == TEXTURE_3D {
        u64::from(desc.depth_or_array_size)
    } else {
        1
    };
    match desc.mip_levels {
        0 => {
            64 - desc
                .width
                .max(u64::from(desc.height))
                .max(depth)
                .max(1)
                .leading_zeros()
        }
        n => u32::from(n),
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlacedFootprint {
    pub(crate) offset: u64,
    format: u32,
    width: u32,
    height: u32,
    depth: u32,
    row_pitch: u32,
}

impl PlacedFootprint {
    /// The bytes it spans in a buffer: every row pitched but the last.
    pub(crate) fn bytes(&self) -> Option<u64> {
        let (element, block) = element(self.format)?;
        let lines = u64::from(self.height).div_ceil(block) * u64::from(self.depth);
        let last = u64::from(self.width).div_ceil(block) * element;
        Some(u64::from(self.row_pitch) * lines.saturating_sub(1) + last)
    }
}
const _: () = assert!(size_of::<PlacedFootprint>() == 32);

const NO_FOOTPRINT: PlacedFootprint = PlacedFootprint {
    offset: u64::MAX,
    format: 0,
    width: 0,
    height: 0,
    depth: 0,
    row_pitch: 0,
};

/// One subresource's size in an upload buffer: footprint (offset 0), rows
/// and bytes per row. Planes of depth-stencil formats are not told apart.
fn subresource(desc: &ResourceDesc, index: u32) -> Option<(PlacedFootprint, u32, u64)> {
    const PITCH: u64 = 256;
    if desc.dimension == BUFFER {
        let footprint = PlacedFootprint {
            offset: 0,
            format: 0,
            width: u32::try_from(desc.width).ok()?,
            height: 1,
            depth: 1,
            row_pitch: u32::try_from(desc.width.next_multiple_of(PITCH)).ok()?,
        };
        return (index == 0).then_some((footprint, 1, desc.width));
    }
    let mips = mip_count(desc);
    let slices = if desc.dimension == TEXTURE_3D {
        1
    } else {
        u32::from(desc.depth_or_array_size).max(1)
    };
    let (mip, slice) = (index % mips, index / mips);
    if slice >= slices {
        return None;
    }
    let (element, block) = element(desc.format)?;
    let width = (desc.width >> mip).max(1).next_multiple_of(block);
    let height = (u64::from(desc.height) >> mip)
        .max(1)
        .next_multiple_of(block);
    let depth = if desc.dimension == TEXTURE_3D {
        (u32::from(desc.depth_or_array_size) >> mip).max(1)
    } else {
        1
    };
    let row_size = width / block * element;
    let footprint = PlacedFootprint {
        offset: 0,
        format: desc.format,
        width: u32::try_from(width).ok()?,
        height: u32::try_from(height).ok()?,
        depth,
        row_pitch: u32::try_from(row_size.next_multiple_of(PITCH)).ok()?,
    };
    Some((footprint, u32::try_from(height / block).ok()?, row_size))
}

/// Device method: where each subresource goes in an upload buffer, rows
/// 256-byte and subresources 512-byte aligned, as on PC. Unknown sizes give
/// the documented all-ones values.
pub(crate) extern "system" fn copyable_footprints(
    _: This,
    desc: *const ResourceDesc,
    first: u32,
    count: u32,
    base: u64,
    layouts: *mut PlacedFootprint,
    rows: *mut u32,
    row_sizes: *mut u64,
    total: *mut u64,
) {
    const PLACEMENT: u64 = 512;
    // SAFETY: the title passes a description (null is tolerated).
    let desc = unsafe { desc.as_ref() };
    let mut end = base;
    let mut known = desc.is_some();
    for i in 0..count {
        let one = desc.and_then(|d| subresource(d, first + i));
        let (layout, row_count, row_size) = match one {
            Some((mut footprint, row_count, row_size)) => {
                footprint.offset = end.next_multiple_of(PLACEMENT);
                let pitch = u64::from(footprint.row_pitch);
                let lines = u64::from(row_count) * u64::from(footprint.depth);
                end = footprint.offset + pitch * lines.saturating_sub(1) + row_size;
                (footprint, row_count, row_size)
            }
            None => {
                known = false;
                (NO_FOOTPRINT, u32::MAX, u64::MAX)
            }
        };
        let i = i as usize;
        // SAFETY: each output is null or holds `count` entries.
        unsafe {
            if !layouts.is_null() {
                layouts.add(i).write(layout);
            }
            if !rows.is_null() {
                rows.add(i).write(row_count);
            }
            if !row_sizes.is_null() {
                row_sizes.add(i).write(row_size);
            }
        }
    }
    if !known {
        tracing::warn!(?desc, "no footprint for this resource");
    }
    if !total.is_null() {
        // SAFETY: checked non-null.
        unsafe { total.write(if known { end - base } else { u64::MAX }) };
    }
}

/// Device method: the footprint of resources placed one after another.
/// Returns a 16-byte value, through a hidden first argument. Invalid input
/// gives the documented size `u64::MAX`.
pub(crate) extern "system" fn allocation_info(
    out: *mut AllocationInfo,
    _: This,
    _visible: u32,
    count: u32,
    descs: *const ResourceDesc,
) -> *mut AllocationInfo {
    let descs = if descs.is_null() {
        &[][..]
    } else {
        // SAFETY: the title passes `count` descriptions.
        unsafe { std::slice::from_raw_parts(descs, count as usize) }
    };
    // SAFETY: the caller passes room for the value.
    unsafe { out.write(place(descs.iter(), |_, _| ())) };
    out
}

/// The newer description: the plain one, then a sampler feedback mip
/// region (width, height, depth).
#[repr(C)]
pub(crate) struct ResourceDesc1 {
    desc: ResourceDesc,
    mip_region: [u32; 3],
}
const _: () = assert!(size_of::<ResourceDesc1>() == 72);

/// Where one resource goes in [`allocation_info2`]'s placement.
#[repr(C)]
pub(crate) struct AllocationInfo1 {
    offset: u64,
    alignment: u64,
    size: u64,
}

/// Device method 0x360 (`ID3D12Device8`): [`allocation_info`] for newer
/// descriptions, also telling where each resource goes.
pub(crate) extern "system" fn allocation_info2(
    out: *mut AllocationInfo,
    _: This,
    _visible: u32,
    count: u32,
    descs: *const ResourceDesc1,
    each: *mut AllocationInfo1,
) -> *mut AllocationInfo {
    let descs = if descs.is_null() {
        &[][..]
    } else {
        // SAFETY: the title passes `count` descriptions.
        unsafe { std::slice::from_raw_parts(descs, count as usize) }
    };
    let total = place(descs.iter().map(|d| &d.desc), |i, info| {
        if !each.is_null() {
            // SAFETY: the title passes room for `count` entries.
            unsafe { each.add(i).write(info) };
        }
    });
    // SAFETY: the caller passes room for the value.
    unsafe { out.write(total) };
    out
}

/// Resources placed one after another: the whole footprint, telling `each`
/// where every resource goes. Unknown sizes give the documented `u64::MAX`.
fn place<'a>(
    descs: impl Iterator<Item = &'a ResourceDesc>,
    mut each: impl FnMut(usize, AllocationInfo1),
) -> AllocationInfo {
    let mut total = AllocationInfo {
        size: 0,
        alignment: KIB_64,
    };
    for (i, desc) in descs.enumerate() {
        let Some(one) = footprint(desc) else {
            tracing::warn!(?desc, "no size for this resource");
            each(
                i,
                AllocationInfo1 {
                    offset: u64::MAX,
                    alignment: 0,
                    size: u64::MAX,
                },
            );
            return AllocationInfo {
                size: u64::MAX,
                alignment: 0,
            };
        };
        tracing::debug!(?desc, ?one, "allocation info");
        let offset = total.size.next_multiple_of(one.alignment);
        each(
            i,
            AllocationInfo1 {
                offset,
                alignment: one.alignment,
                size: one.size,
            },
        );
        total.size = offset + one.size;
        total.alignment = total.alignment.max(one.alignment);
    }
    total
}

/// Zeroed host memory standing in for GPU memory.
pub(crate) struct Memory {
    ptr: NonNull<u8>,
    layout: Layout,
    /// Set when the memory came from the title's allocator, with these
    /// allocation attributes (it goes back there).
    attributes: Option<u64>,
}

pub type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
pub type FreeRoutine = extern "system" fn(*mut c_void, u64);
static ALLOCATOR: std::sync::OnceLock<(AllocRoutine, FreeRoutine)> = std::sync::OnceLock::new();

/// Committed resources get their memory from the title's general
/// allocator, as on the console, where titles hook it to place graphics
/// memory themselves (Grand Theft Auto VI places its back buffers in a
/// block of its own and checks they fill it). The runtime names it before
/// the device exists.
pub fn use_allocator(alloc: AllocRoutine, free: FreeRoutine) {
    let _ = ALLOCATOR.set((alloc, free));
}

/// Maps the physical pages behind one range of title memory at another,
/// `(to, from, len)` (`from` 0 unmaps); false on failure.
pub type PageMapper = fn(u64, u64, u64) -> bool;
static PAGE_MAPPER: std::sync::OnceLock<PageMapper> = std::sync::OnceLock::new();

/// The runtime's page mapper, for the queues' page-mapping copies (memory
/// is the runtime's, as on the console).
pub fn use_page_mapper(mapper: PageMapper) {
    let _ = PAGE_MAPPER.set(mapper);
}

pub(crate) fn page_mapper() -> Option<PageMapper> {
    PAGE_MAPPER.get().copied()
}

/// Allocation attributes for resource memory: allocator id, page size
/// (bits 14-15: 2 MiB pages from 2 MiB up, else 64 KiB), alignment as log2
/// (bits 24-28) and memory type (bits 29-32: write-combined for GPU-only
/// heaps, cacheable for the CPU-visible ones).
/// shortcut: the console runtime's own choice of page size and memory type
/// is not known; this is the plausible one.
fn resource_attributes(info: AllocationInfo, heap_kind: u32) -> u64 {
    const GRAPHICS_ALLOCATOR: u64 = 128 << 16;
    const DEFAULT_HEAP: u32 = 1;
    let pages: u64 = if info.size >= 2 << 20 { 2 } else { 1 };
    let memory: u64 = if heap_kind == DEFAULT_HEAP { 1 } else { 3 };
    let alignment = u64::from(info.alignment.max(1).trailing_zeros());
    GRAPHICS_ALLOCATOR | pages << 14 | alignment << 24 | memory << 29
}

impl Memory {
    /// A committed resource's memory: from the title's allocator if the
    /// runtime named one, else our own; zeroed either way.
    fn for_resource(info: AllocationInfo, heap_kind: u32) -> Result<Memory, HRESULT> {
        let Some(&(alloc, _)) = ALLOCATOR.get() else {
            return Memory::new(info);
        };
        let layout = Layout::from_size_align(info.size as usize, info.alignment as usize)
            .map_err(|_| E_INVALIDARG)?;
        let attributes = resource_attributes(info, heap_kind);
        let ptr =
            NonNull::new(alloc(layout.size(), attributes).cast::<u8>()).ok_or(E_OUTOFMEMORY)?;
        // SAFETY: the allocator handed out `size` bytes.
        unsafe { ptr.as_ptr().write_bytes(0, layout.size()) };
        Ok(Memory {
            ptr,
            layout,
            attributes: Some(attributes),
        })
    }

    /// `size` zeroed bytes aligned to `alignment` (a power of two).
    pub(crate) fn zeroed(size: u64, alignment: u64) -> Result<Memory, HRESULT> {
        Memory::new(AllocationInfo { size, alignment })
    }

    pub(crate) fn address(&self) -> u64 {
        self.ptr.as_ptr() as u64
    }

    fn new(info: AllocationInfo) -> Result<Memory, HRESULT> {
        let layout = usize::try_from(info.size)
            .ok()
            .zip(usize::try_from(info.alignment).ok())
            .filter(|&(size, _)| size > 0)
            .and_then(|(size, align)| Layout::from_size_align(size, align).ok())
            .ok_or(E_INVALIDARG)?;
        // SAFETY: the size is non-zero.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or(E_OUTOFMEMORY)?;
        Ok(Memory {
            ptr,
            layout,
            attributes: None,
        })
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        match (self.attributes, ALLOCATOR.get()) {
            (Some(attributes), Some(&(_, free))) => free(self.ptr.as_ptr().cast(), attributes),
            // SAFETY: allocated in `new` with this layout.
            _ => unsafe { dealloc(self.ptr.as_ptr(), self.layout) },
        }
    }
}

#[repr(C)]
struct Heap {
    object: Object,
    desc: HeapDesc,
    memory: Memory,
}

pub(crate) extern "system" fn create_heap(
    _: This,
    desc: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(
        size = desc.size,
        kind = desc.properties.kind,
        flags = format_args!("{:#x}", desc.flags),
        "heap"
    );
    if desc.alignment != 0 && !desc.alignment.is_power_of_two() {
        return E_INVALIDARG;
    }
    let info = AllocationInfo {
        size: desc.size,
        alignment: desc.alignment.max(KIB_64),
    };
    let memory = match Memory::new(info) {
        Ok(memory) => memory,
        Err(hr) => return hr,
    };
    let heap = Box::new(Heap {
        object: Object::new(&HEAP_FUNCTIONS),
        desc,
        memory,
    });
    object::hand_out(HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn heap_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(HEAP, iid, object::is(iid, &HEAP_IIDS).then_some(this), out)
}
extern "system" fn heap_release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn heap_desc(this: This, out: *mut HeapDesc) {
    if !out.is_null() {
        // SAFETY: `this` is a heap; the title passes room for a description.
        unsafe { out.write((*this.cast::<Heap>()).desc) };
    }
}

/// A resource's memory: its own, or a range of a heap it holds a reference
/// to.
enum Backing {
    Committed(Memory),
    Placed {
        heap: This,
        offset: u64,
    },
    /// Memory the title mapped itself, at this address.
    Address(*mut u8),
}

impl Drop for Backing {
    fn drop(&mut self) {
        if let Backing::Placed { heap, .. } = *self {
            heap_release(heap);
        }
    }
}

#[repr(C)]
struct Resource {
    object: Object,
    desc: ResourceDesc,
    heap: HeapProperties,
    heap_flags: u32,
    backing: Backing,
    /// Textures also live on the host GPU (decision 0007); buffers do not
    /// yet. Made on first use (a view, a target, a copy, a barrier), not
    /// at creation: a texture the title makes and never has the GPU use
    /// costs no host memory (the world makes thousands; their host copies
    /// took about 9 GB, past an 8 GB card's budget, 2026-11-24).
    #[cfg(windows)]
    host: std::sync::OnceLock<Option<std::sync::Arc<HostTexture>>>,
    /// The creation state and clear value the host copy is made with.
    #[cfg(windows)]
    made_with: (u32, Option<[u32; 5]>),
}

#[cfg(windows)]
impl Resource {
    /// The host copy, made now if this is its first use.
    fn host(&self) -> Option<&std::sync::Arc<HostTexture>> {
        self.host
            .get_or_init(|| {
                let (state, clear) = self.made_with;
                let clear = clear
                    .as_ref()
                    .map_or(std::ptr::null(), |c| std::ptr::from_ref(c).cast::<c_void>());
                let texture = host_texture(&self.desc, state, clear)?;
                let state =
                    crate::host::legal_state(&texture, crate::host::texture_state(state));
                // Depth and unordered access (see `host_texture`).
                const BOTH: u32 = 2 | 4;
                let twinned = self.desc.flags & BOTH == BOTH && twins_on();
                let twin = twinned
                    .then(|| host_twin(&texture, self.desc.format))
                    .flatten()
                    .map(HostTexture::plain);
                // The console's 16-bit depth as host D16 keeps its stencil
                // apart, whether or not the title writes it as unordered
                // access.
                const DEPTH: u32 = 2;
                let alone = self.desc.flags & DEPTH != 0
                    && SIXTEEN_BIT_WITH_STENCIL.contains(&self.desc.format)
                    && d16_depth();
                let stencil = if alone {
                    host_stencil_twin(&texture, true).map(|stencil| {
                        let clears = crate::depth_twin::Clears::new(&stencil);
                        std::sync::Arc::new(HostTexture {
                            clears,
                            ..HostTexture::new(stencil)
                        })
                    })
                } else {
                    twinned
                        .then(|| host_stencil_twin(&texture, false))
                        .flatten()
                        .map(HostTexture::plain)
                };
                let planes = twin
                    .as_ref()
                    .and_then(|t| crate::depth_twin::Views::new(&texture, &t.texture));
                Some(std::sync::Arc::new(HostTexture {
                    state: state.into(),
                    twin,
                    stencil_alone: alone && stencil.is_some(),
                    stencil,
                    planes,
                    ..HostTexture::new(texture)
                }))
            })
            .as_ref()
    }
}

/// A title texture's host copy and the state it is in. The title's barriers
/// name console states (looser than the host's), so we transition from this
/// one. Lists that use the texture hold it until they are reset: the title
/// may release a resource whose commands it recorded but not yet submitted
/// (console memory outlives the object; a host texture must outlive the
/// lists that name it).
#[cfg(windows)]
pub(crate) struct HostTexture {
    pub(crate) texture: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    pub(crate) state: std::sync::atomic::AtomicU32,
    /// A depth target's twin for unordered access ([`host_twin`]), which
    /// its unordered access views name; lists keep the two coherent
    /// (`HostList::transition`).
    pub(crate) twin: Option<std::sync::Arc<HostTexture>>,
    /// The twin of a depth target's stencil plane ([`host_stencil_twin`]),
    /// which unordered access views of plane 1 name.
    pub(crate) stencil: Option<std::sync::Arc<HostTexture>>,
    /// Whether `stencil` is the only stencil there is, not a twin: the
    /// console's 16-bit depth with stencil as host D16 ([`d16_depth`]).
    /// Shader views of the stencil, unordered access to it, its clears and
    /// copies go to it; nothing copies it to or from the depth target.
    pub(crate) stencil_alone: bool,
    /// With a `twin` of fewer bits than the depth plane (16 for the
    /// console's depth with stencil, 24 on the host): the views the passes
    /// that move the depth between the two use (`crate::depth_twin`), as
    /// copies cannot.
    pub(crate) planes: Option<crate::depth_twin::Views>,
    /// For a stencil kept apart: the views its clears go through.
    pub(crate) clears: Option<crate::depth_twin::Clears>,
}

#[cfg(windows)]
impl HostTexture {
    /// `texture` in the common state, without twins or views of its own.
    fn new(texture: windows::Win32::Graphics::Direct3D12::ID3D12Resource) -> HostTexture {
        HostTexture {
            texture,
            state: 0.into(),
            twin: None,
            stencil: None,
            stencil_alone: false,
            planes: None,
            clears: None,
        }
    }

    /// A twin: [`HostTexture::new`], shared.
    fn plain(
        texture: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    ) -> std::sync::Arc<HostTexture> {
        std::sync::Arc::new(HostTexture::new(texture))
    }
}

/// Moves the host texture to `wanted` (a host texture state); the host
/// resource and the state it leaves, if that is a change.
#[cfg(windows)]
pub(crate) fn host_transition(this: This, wanted: u32) -> Option<(*mut c_void, u32)> {
    use windows::core::Interface;
    let host = resource(this).host()?;
    let before = host
        .state
        .swap(wanted, std::sync::atomic::Ordering::Relaxed);
    (before != wanted).then(|| (host.texture.as_raw(), before))
}

/// The host GPU's copy of a title resource, if it has one.
#[cfg(windows)]
pub(crate) fn host_resource<'a>(
    this: This,
) -> Option<&'a windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    if this.is_null() {
        return None;
    }
    resource(this).host().map(|h| &h.texture)
}

/// The host texture unordered access views of `plane` of a title texture
/// name: the twin of that plane of a depth target that has one, with the
/// format such a view takes when the title's names none or a depth format;
/// else the host copy. `None` for a plane of a depth target that has no
/// twin for it.
#[cfg(windows)]
pub(crate) fn host_unordered<'a>(
    this: This,
    plane: u32,
) -> Option<(
    &'a windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    Option<u32>,
)> {
    if this.is_null() {
        return None;
    }
    let host = resource(this).host()?;
    if plane == 1 {
        // R8_UINT: the stencil's bits.
        return host.stencil.as_ref().map(|s| (&s.texture, Some(62)));
    }
    let Some(twin) = &host.twin else {
        return Some((&host.texture, None));
    };
    // SAFETY: a live resource.
    let typeless = unsafe { twin.texture.GetDesc() }.Format.0 as u32;
    let format = TWIN_FORMATS
        .iter()
        .find(|&&(f, _)| f == typeless)
        .map(|&(_, f)| f);
    Some((&twin.texture, format))
}

/// The stencil of a title texture whose host copy keeps it apart (the
/// console's 16-bit depth with stencil as host D16, [`d16_depth`]): an
/// R8_TYPELESS texture of the depth's size, mips and slices, which shader
/// views of the stencil name.
#[cfg(windows)]
pub(crate) fn host_stencil<'a>(
    this: This,
) -> Option<&'a windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    if this.is_null() {
        return None;
    }
    let host = resource(this).host()?;
    host.stencil
        .as_ref()
        .filter(|_| host.stencil_alone)
        .map(|s| &s.texture)
}

/// A title texture's host copy, shared, for a list to hold.
#[cfg(windows)]
pub(crate) fn host_copy(this: This) -> Option<std::sync::Arc<HostTexture>> {
    if this.is_null() {
        return None;
    }
    resource(this).host().cloned()
}

/// A host texture like the title's: same description, in GPU memory,
/// without the console's layout and flags.
#[cfg(windows)]
fn host_texture(
    desc: &ResourceDesc,
    state: u32,
    clear: *const c_void,
) -> Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    use windows::Win32::Graphics::Direct3D12::*;
    const PC_FLAGS: u32 = 0xf; // render target, depth, UAV, no shader access
    if desc.dimension == BUFFER {
        return None;
    }
    let host = crate::host::host()?;
    let mut d = *desc;
    d.layout = 0;
    d.alignment = 0;
    d.flags &= PC_FLAGS;
    // PC refuses depth targets that are also unordered-access (the console
    // allows both; Grand Theft Auto VI's shadow maps ask for it): the host
    // copy loses UAV use, its unordered access views name a twin
    // (`host_twin`).
    const DEPTH: u32 = 2;
    const UNORDERED: u32 = 4;
    if d.flags & DEPTH != 0 {
        d.flags &= !UNORDERED;
    }
    d.format = pc_format(d.format);
    const DENY_SHADER_RESOURCE: u32 = 8;
    if d.flags & DENY_SHADER_RESOURCE == 0
        && let Some(&(_, typeless)) = DEPTH_FORMATS.iter().find(|&&(f, _)| f == d.format)
    {
        d.format = typeless;
    }
    // The host has no sRGB unordered access: such textures are typeless,
    // their unordered access views plain UNORM.
    if d.flags & UNORDERED != 0 && SRGB.iter().any(|&(srgb, _)| srgb == d.format) {
        d.format = family(d.format);
    }
    // The clear value names the format too: format, then color or depth
    // and stencil (5 words).
    let clear_value = (!clear.is_null()).then(|| {
        // SAFETY: the title passes a whole clear value.
        let mut value = unsafe { clear.cast::<[u32; 5]>().read_unaligned() };
        value[0] = pc_format(value[0]);
        value
    });
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: the description has the host layout (checked size, same
    // fields); the out pointer is ours.
    let made = unsafe {
        host.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            (&raw const d).cast(),
            D3D12_RESOURCE_STATES(
                crate::host::legal_for(d.flags, crate::host::texture_state(state)) as i32,
            ),
            // The same layout: format, then color or depth and stencil.
            clear_value.as_ref().map(|v| std::ptr::from_ref(v).cast()),
            &mut texture,
        )
    };
    if let Err(e) = made {
        tracing::warn!(error = %e, desc = ?d, "no host texture");
    }
    if let Some(t) = &texture {
        crate::host::name(t, || {
            format!("texture {}x{} f{} flags {:#x}", d.width, d.height, d.format, d.flags)
        });
    }
    texture
}

/// Whether depth targets the title also writes as unordered access get
/// their twins (`VICTORY_LOCAL_DEPTH_UAV=1`; always in tests). Without
/// them such views are null and what compute shaders write through them
/// is dropped, as before the twins.
/// LOCAL: a switch to compare runs with and without the twins in one
/// build; drop it once the twins are proven in the title.
#[cfg(windows)]
fn twins_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    cfg!(test)
        || *ON.get_or_init(|| {
            let on = std::env::var("VICTORY_LOCAL_DEPTH_UAV").is_ok_and(|v| v == "1");
            tracing::info!(on, "unordered access twins of depth targets (VICTORY_LOCAL_DEPTH_UAV)");
            on
        })
}

/// Depth families that have unordered access formats (16-bit and 32-bit
/// depth; the stencil ones have none), and the format with the depth
/// format's bits: R16_UNORM for D16_UNORM, R32_FLOAT for D32_FLOAT.
#[cfg(windows)]
const TWIN_FORMATS: [(u32, u32); 2] = [(53, 56), (39, 41)];

/// Depth families with a stencil plane whose depth plane has a twin of its
/// own, single-plane family: D32_FLOAT_S8X24_UINT's depth plane is
/// R32_TYPELESS. Only the depth plane (plane 0) is twinned and copied; the
/// stencil plane stays in the depth target.
#[cfg(windows)]
const PLANAR_TWINS: [(u32, u32); 1] = [(19, 39)];

/// Depth families with a stencil plane: the plane's twin is R8_TYPELESS.
#[cfg(windows)]
const STENCIL_FAMILIES: [u32; 2] = [19, 44];

/// The console's 16-bit depth with stencil (D16_UNORM_S8_UINT and its
/// views' formats), a D24S8 target on the host (`pc_format`): its depth
/// plane's twin is R16_TYPELESS, the console's bits, which the title's
/// 16-bit views name; passes of ours move them to the 24-bit plane and
/// back (`crate::depth_twin`). With [`d16_depth`] it is a D16 target with
/// its stencil apart instead, and its twin is a plain 16-bit one.
#[cfg(windows)]
const SIXTEEN_BIT_WITH_STENCIL: std::ops::RangeInclusive<u32> = 118..=120;

/// Whether a host depth target in `family` made for a title texture in
/// format `title` keeps its depth plane in a 16-bit twin
/// ([`SIXTEEN_BIT_WITH_STENCIL`]).
#[cfg(windows)]
fn sixteen_bit_plane(family: u32, title: u32) -> bool {
    SIXTEEN_BIT_WITH_STENCIL.contains(&title) && STENCIL_FAMILIES.contains(&family)
}

/// Whether a title texture's host copy has a stencil plane twin (barriers
/// on it then cover every unordered access resource).
#[cfg(windows)]
pub(crate) fn has_stencil_twin(this: This) -> bool {
    !this.is_null()
        && resource(this)
            .host()
            .is_some_and(|h| h.stencil.is_some())
}

/// A twin for unordered access to the stencil plane of host depth target
/// `depth`: R8_TYPELESS, its size, mips and slices. Copies between them
/// move plane 1 (`HostList::copy_texture`). With `alone`, the stencil of a
/// depth target that has none on the host (`HostTexture::stencil_alone`).
#[cfg(windows)]
fn host_stencil_twin(
    depth: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    alone: bool,
) -> Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    use windows::Win32::Graphics::Direct3D12::*;
    let host = crate::host::host()?;
    // SAFETY: a live resource.
    let mut d = unsafe { depth.GetDesc() };
    if d.SampleDesc.Count > 1
        || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || !alone && !STENCIL_FAMILIES.contains(&family(d.Format.0 as u32))
    {
        if alone {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    samples = d.SampleDesc.Count,
                    "the console's 16-bit depth as host D16 without a stencil: its stencil is lost"
                )
            });
        }
        return None;
    }
    d.Format.0 = 60;
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut twin: Option<ID3D12Resource> = None;
    // SAFETY: the host's own description, changed to a legal one; the out
    // pointer is ours.
    let made = unsafe {
        host.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &d,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut twin,
        )
    };
    if let Err(e) = made {
        tracing::warn!(error = %e, desc = ?d, "no stencil unordered access twin");
    }
    twin
}

/// The typeless format of the twin of a host depth target in `family`
/// made for a title texture in format `title`, if it can have one (see
/// [`TWIN_FORMATS`], [`PLANAR_TWINS`], [`SIXTEEN_BIT_WITH_STENCIL`]).
#[cfg(windows)]
fn twin_family(family: u32, title: u32) -> Option<u32> {
    const R16_TYPELESS: u32 = 53;
    if sixteen_bit_plane(family, title) {
        return Some(R16_TYPELESS);
    }
    PLANAR_TWINS
        .iter()
        .find(|&&(f, _)| f == family)
        .map(|&(_, twin)| twin)
        .or_else(|| {
            TWIN_FORMATS
                .iter()
                .any(|&(f, _)| f == family)
                .then_some(family)
        })
}

/// A twin for the unordered access of host depth target `depth`: its
/// description with unordered access instead of depth use, in the
/// typeless format of its family, so that views of the twin take any
/// format of the family (the title's own). Copies between the two keep
/// the bits (CopyResource takes depth resources and formats of one type
/// group), except for the console's 16-bit depth with stencil, whose twin
/// is 16-bit ([`SIXTEEN_BIT_WITH_STENCIL`]; one subresource only). `None`
/// without such a family (stencil), multisampled, or not 2D.
#[cfg(windows)]
fn host_twin(
    depth: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    title: u32,
) -> Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    use windows::Win32::Graphics::Direct3D12::*;
    let host = crate::host::host()?;
    // SAFETY: a live resource.
    let mut d = unsafe { depth.GetDesc() };
    let own = family(d.Format.0 as u32);
    let twin_format = twin_family(own, title);
    if d.SampleDesc.Count > 1
        || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || twin_format.is_none()
        || sixteen_bit_plane(own, title) && (d.MipLevels != 1 || d.DepthOrArraySize != 1)
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                format = d.Format.0,
                samples = d.SampleDesc.Count,
                "no unordered access twin for this depth target: its UAV writes are dropped"
            )
        });
        return None;
    }
    d.Format.0 = twin_format.unwrap_or_default() as i32;
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut twin: Option<ID3D12Resource> = None;
    // SAFETY: the host's own description, changed to a legal one; the out
    // pointer is ours.
    let made = unsafe {
        host.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &d,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut twin,
        )
    };
    if let Err(e) = made {
        tracing::warn!(error = %e, desc = ?d, "no unordered access twin");
    }
    twin
}

/// A texture's console memory and description; `None` for buffers.
pub(crate) fn texture_memory(this: This) -> Option<(*const u8, ResourceDesc)> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    (r.desc.dimension != BUFFER).then(|| (r.data().cast_const(), r.desc))
}

/// A texture's own format (the title's numbering); `None` for buffers.
pub(crate) fn title_format(this: This) -> Option<u32> {
    texture_memory(this).map(|(_, d)| d.format)
}

/// A buffer's host memory and size; `None` for textures.
pub(crate) fn buffer_bytes(this: This) -> Option<(*mut u8, u64)> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    (r.desc.dimension == BUFFER).then(|| (r.data(), r.desc.width))
}

impl Resource {
    fn data(&self) -> *mut u8 {
        match self.backing {
            Backing::Committed(ref memory) => memory.ptr.as_ptr(),
            // SAFETY: placement checked the range lies inside the heap.
            Backing::Placed { heap, offset } => unsafe {
                (*heap.cast::<Heap>())
                    .memory
                    .ptr
                    .as_ptr()
                    .add(offset as usize)
            },
            Backing::Address(at) => at,
        }
    }
}

fn resource<'a>(this: This) -> &'a Resource {
    // SAFETY: called from resource methods only.
    unsafe { &*this.cast::<Resource>() }
}

// The creation methods' own arguments, passed on.
#[allow(clippy::too_many_arguments)]
fn hand_out_resource(
    desc: ResourceDesc,
    heap: HeapProperties,
    heap_flags: u32,
    backing: Backing,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    #[cfg(not(windows))]
    let _ = (state, clear);
    // The clear value is the caller's only during the call.
    #[cfg(windows)]
    let made_with = (
        state,
        // SAFETY: the title passes a whole clear value (5 words) or null.
        (!clear.is_null()).then(|| unsafe { clear.cast::<[u32; 5]>().read_unaligned() }),
    );
    let resource = Box::new(Resource {
        object: Object::new(&RESOURCE_FUNCTIONS),
        desc,
        heap,
        heap_flags,
        backing,
        #[cfg(windows)]
        host: std::sync::OnceLock::new(),
        #[cfg(windows)]
        made_with,
    });
    #[cfg(windows)]
    if desc.dimension == BUFFER {
        let at = std::ptr::from_ref::<Resource>(&resource) as usize;
        crate::shadow::register(at, resource.data() as u64, desc.width);
    }
    object::hand_out(RESOURCE, resource, iid, &RESOURCE_IIDS, out)
}

#[cfg(windows)]
impl Drop for Resource {
    fn drop(&mut self) {
        #[cfg(feature = "diagnostics")]
        if self
            .host
            .get()
            .and_then(Option::as_ref)
            .is_some_and(|h| std::sync::Arc::strong_count(h) > 1)
        {
            // Before lists held their textures, this freed a texture that
            // lists not yet run still named (a GPU page fault).
            static OUTLIVED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = OUTLIVED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                tracing::info!(
                    n,
                    desc = ?self.desc,
                    "a title texture was released while a list not yet reset names its host copy"
                );
            }
        }
        // A running submission may still read the host copy.
        if let Some(host) = self.host.take().flatten() {
            crate::host::retire(host);
        }
        crate::texture::forget(std::ptr::from_ref::<Resource>(self) as usize);
        crate::descriptor::forget(std::ptr::from_ref::<Resource>(self) as usize);
        if self.desc.dimension == BUFFER {
            crate::shadow::unregister(
                std::ptr::from_ref::<Resource>(self) as usize,
                self.data() as u64,
            );
        }
    }
}

pub(crate) extern "system" fn create_committed(
    _: This,
    heap: *const HeapProperties,
    heap_flags: u32,
    desc: *const ResourceDesc,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes both descriptions (null is tolerated).
    let (Some(&heap), Some(&desc)) = (unsafe { (heap.as_ref(), desc.as_ref()) }) else {
        return E_INVALIDARG;
    };
    let Some(info) = footprint(&desc) else {
        tracing::warn!(?desc, "no size for this resource");
        return E_INVALIDARG;
    };
    tracing::debug!(
        dimension = desc.dimension,
        size = info.size,
        heap = heap.kind,
        "committed resource"
    );
    let memory = match Memory::for_resource(info, heap.kind) {
        Ok(memory) => memory,
        Err(hr) => return hr,
    };
    hand_out_resource(
        desc,
        heap,
        heap_flags,
        Backing::Committed(memory),
        state,
        clear,
        iid,
        out,
    )
}

/// Device method 0x3f8, inferred from Grand Theft Auto VI's calls
/// (title+0x193e1b4: its back buffers, heap flag "allow display"): the
/// committed resource creation of the enhanced-barrier interfaces,
/// `(heap, heap flags, desc1, initial layout, clear value, protected
/// session, castable format count, castable formats, iid, out)`. The host
/// copy starts in the common state, as layouts are not tracked.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn create_committed3(
    this: This,
    heap: *const HeapProperties,
    heap_flags: u32,
    desc: *const ResourceDesc,
    _layout: u32,
    clear: *const c_void,
    _protected: This,
    _castable_count: u32,
    _castable: *const u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    const COMMON: u32 = 0;
    create_committed(this, heap, heap_flags, desc, COMMON, clear, iid, out)
}

/// Registered page pools (64 KiB pages the title maps into tiled
/// resources by page index): first address and page count; a handle is an
/// index plus one, a removed pool leaves `None`.
static PAGE_POOLS: std::sync::Mutex<Vec<Option<(u64, u32)>>> = std::sync::Mutex::new(Vec::new());

fn page_pools() -> std::sync::MutexGuard<'static, Vec<Option<(u64, u32)>>> {
    PAGE_POOLS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Device method 0x250: `(address, page count, handle out)`.
pub(crate) extern "system" fn register_page_pool(
    _: This,
    address: u64,
    pages: u32,
    out: *mut usize,
) -> HRESULT {
    if out.is_null() || pages == 0 || !address.is_multiple_of(KIB_64) {
        return E_INVALIDARG;
    }
    tracing::debug!(address = format_args!("{address:#x}"), pages, "page pool");
    let mut pools = page_pools();
    pools.push(Some((address, pages)));
    // SAFETY: checked non-null; the title passes room for a handle.
    unsafe { out.write(pools.len()) };
    S_OK
}

/// Device method 0x258.
pub(crate) extern "system" fn unregister_page_pool(_: This, handle: usize) -> HRESULT {
    match page_pools().get_mut(handle.wrapping_sub(1)) {
        Some(pool @ Some(_)) => {
            *pool = None;
            S_OK
        }
        _ => E_INVALIDARG,
    }
}

/// Device method 0x428, one of the `ID3D12Device9` to `ID3D12Device11`
/// additions. From Grand Theft Auto VI's call: a resource at a virtual
/// address the title mapped itself, `(address, desc1, initial barrier
/// layout, clear value, iid, out)`, as `CreatePlacedResourceX` with
/// enhanced barriers. GPU addresses are CPU addresses, so the resource is
/// a view of that memory. `D3D12_RESOURCE_DESC1` starts like the plain
/// description; its sampler feedback region is not needed.
pub(crate) extern "system" fn create_at_address(
    _: This,
    address: u64,
    desc: *const ResourceDesc,
    layout: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if address == 0 {
        return E_INVALIDARG;
    }
    tracing::debug!(
        address = format_args!("{address:#x}"),
        dimension = desc.dimension,
        width = desc.width,
        layout,
        "resource at an address"
    );
    const COMMON: u32 = 0;
    const DEFAULT: HeapProperties = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 1,
        visible_nodes: 1,
    };
    hand_out_resource(
        desc,
        DEFAULT,
        0,
        Backing::Address(address as *mut u8),
        COMMON,
        clear,
        iid,
        out,
    )
}

/// Textures the title placed in its own memory and views without a
/// resource object (device 0x230, Grand Theft Auto VI's streamed and
/// interface textures), by address and description: a resource at that
/// address, made the first time a draw reads one.
/// shortcut: never released (one host texture per distinct address and
/// description viewed); release them when the title unmaps or reuses the
/// memory once streaming grows VRAM use.
#[cfg(windows)]
static PLACED_TEXTURES: std::sync::Mutex<std::collections::BTreeMap<(u64, [u32; 14]), usize>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// The resource for a texture placed at `address` with the resource
/// description `desc` (as words), made on first use.
#[cfg(windows)]
pub(crate) fn texture_at(address: u64, desc: &[u32; 14]) -> Option<This> {
    let mut map = PLACED_TEXTURES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&r) = map.get(&(address, *desc)) {
        return Some(r as This);
    }
    // SAFETY: 56 bytes in the description's layout.
    let d = unsafe { desc.as_ptr().cast::<ResourceDesc>().read_unaligned() };
    if d.dimension == BUFFER {
        return None;
    }
    let mut out = std::ptr::null();
    let made = create_at_address(
        std::ptr::null(),
        address,
        &d,
        0,
        std::ptr::null(),
        &RESOURCE_IIDS[0],
        &mut out,
    );
    if made != S_OK {
        return None;
    }
    map.insert((address, *desc), out as usize);
    resource(out).host().is_some().then_some(out)
}

pub(crate) extern "system" fn create_placed(
    _: This,
    heap: This,
    offset: u64,
    desc: *const ResourceDesc,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if heap.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes one of our heaps.
    let h = unsafe { &*heap.cast::<Heap>() };
    let fits = footprint(&desc).is_some_and(|info| {
        offset.is_multiple_of(info.alignment)
            && offset
                .checked_add(info.size)
                .is_some_and(|end| end <= h.desc.size)
    });
    if !fits {
        tracing::warn!(
            ?desc,
            offset,
            heap = h.desc.size,
            "placed resource does not fit"
        );
        return E_INVALIDARG;
    }
    object::add_ref(heap);
    let backing = Backing::Placed { heap, offset };
    hand_out_resource(
        desc,
        h.desc.properties,
        h.desc.flags,
        backing,
        state,
        clear,
        iid,
        out,
    )
}

extern "system" fn resource_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        RESOURCE,
        iid,
        object::is(iid, &RESOURCE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn resource_release(this: This) -> u32 {
    object::release_boxed::<Resource>(this)
}
/// Memory is always mapped (subresources are not told apart yet); a null
/// `out` only asks for residency.
extern "system" fn map(this: This, _sub: u32, _read: *const c_void, out: *mut *mut u8) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a pointer.
        unsafe { out.write(resource(this).data()) };
    }
    S_OK
}
extern "system" fn unmap(_: This, _sub: u32, _written: *const c_void) {}
extern "system" fn resource_desc(this: This, out: *mut ResourceDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(resource(this).desc) };
    }
}
/// Device method `GetCustomHeapProperties`, returning its struct through a
/// hidden first argument: what a standard heap type is as a custom heap.
/// Memory is unified and cache-coherent (as the architecture query says),
/// so every type is in pool L0, default heaps are not CPU-visible and the
/// others are write-back.
pub(crate) extern "system" fn custom_heap_properties(
    out: *mut HeapProperties,
    _: This,
    nodes: u32,
    kind: u32,
) -> *mut HeapProperties {
    const CUSTOM: u32 = 4;
    const NOT_AVAILABLE: u32 = 1;
    const WRITE_BACK: u32 = 3;
    const L0: u32 = 1;
    let nodes = nodes.max(1);
    let cpu_pages = match kind {
        1 => NOT_AVAILABLE,
        2 | 3 => WRITE_BACK,
        _ => 0,
    };
    let properties = HeapProperties {
        kind: CUSTOM,
        cpu_pages,
        pool: if cpu_pages == 0 { 0 } else { L0 },
        creation_nodes: nodes,
        visible_nodes: nodes,
    };
    // SAFETY: the title passes room for the result (null is tolerated).
    if let Some(out) = unsafe { out.as_mut() } {
        *out = properties;
    }
    out
}

extern "system" fn gpu_address(this: This) -> u64 {
    let r = resource(this);
    let at = r.data() as u64;
    // The console's GPU addresses fit in 40 bits; its ray-tracing builder
    // refuses others (diagnostics: which memory breaks that).
    #[cfg(feature = "diagnostics")]
    if at >> 40 != 0 {
        static ONCE: [std::sync::Once; 2] = [std::sync::Once::new(), std::sync::Once::new()];
        let backing = match r.backing {
            Backing::Committed(ref m) => format!("committed, attributes {:?}", m.attributes),
            Backing::Placed { offset, .. } => format!("placed in a heap at {offset:#x}"),
            Backing::Address(_) => "at a title address".into(),
        };
        ONCE[usize::from(r.desc.dimension == BUFFER)].call_once(|| {
            tracing::warn!(address = format_args!("{at:#x}"), backing, desc = ?r.desc, "a GPU address above 40 bits")
        });
    }
    at
}
extern "system" fn heap_properties(
    this: This,
    properties: *mut HeapProperties,
    flags: *mut u32,
) -> HRESULT {
    let r = resource(this);
    // SAFETY: either pointer may be null; otherwise the title passes room.
    unsafe {
        if let Some(p) = properties.as_mut() {
            *p = r.heap;
        }
        if let Some(f) = flags.as_mut() {
            *f = r.heap_flags;
        }
    }
    S_OK
}

/// IUnknown 3, object 4, device child 1, heap 1.
static HEAP_FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<HEAP, 24>());
    f[0] = heap_query as *const ();
    f[2] = heap_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = heap_desc as *const (); // 0x40
    Table(f)
};

/// IUnknown 3, object 4, device child 1, resource 10 (7 as on PC, then 3
/// console ones).
static RESOURCE_FUNCTIONS: Table<32> = {
    let mut f = object::d3d12_object(object::missing_table::<RESOURCE, 32>());
    f[0] = resource_query as *const ();
    f[2] = resource_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = map as *const (); // 0x40
    f[9] = unmap as *const (); // 0x48
    f[10] = resource_desc as *const (); // 0x50
    f[11] = gpu_address as *const (); // 0x58
    f[14] = heap_properties as *const (); // 0x70
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    type Release = extern "system" fn(This) -> u32;
    type Map = extern "system" fn(This, u32, *const c_void, *mut *mut u8) -> HRESULT;
    type Address = extern "system" fn(This) -> u64;

    const PROPERTIES: HeapProperties = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 0,
        visible_nodes: 0,
    };

    fn buffer(width: u64) -> ResourceDesc {
        ResourceDesc {
            dimension: BUFFER,
            alignment: 0,
            width,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 0,
            sample_count: 1,
            sample_quality: 0,
            layout: 1,
            flags: 0,
        }
    }

    fn texture(width: u64, height: u32, format: u32, mips: u16) -> ResourceDesc {
        ResourceDesc {
            dimension: 3,
            width,
            height,
            format,
            mip_levels: mips,
            layout: 0,
            ..buffer(0)
        }
    }

    fn heap(size: u64) -> This {
        let desc = HeapDesc {
            size,
            properties: PROPERTIES,
            alignment: 0,
            flags: 0,
        };
        let mut heap = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &desc, &HEAP_IIDS[0], &mut heap),
            S_OK
        );
        heap
    }

    #[test]
    fn heaps_are_zeroed_aligned_memory() {
        let heap = heap(1 << 20);
        // SAFETY: a heap we just made.
        let memory = unsafe { &(*heap.cast::<Heap>()).memory };
        assert_eq!(memory.ptr.as_ptr() as u64 % KIB_64, 0);
        let mut back = std::mem::MaybeUninit::<HeapDesc>::uninit();
        slot::<extern "system" fn(This, *mut HeapDesc)>(heap, 0x40)(heap, back.as_mut_ptr());
        // SAFETY: written above.
        assert_eq!(unsafe { back.assume_init() }.size, 1 << 20);
        assert_eq!(slot::<Release>(heap, 0x10)(heap), 0);

        let bad = HeapDesc {
            size: 1 << 20,
            properties: PROPERTIES,
            alignment: 3,
            flags: 0,
        };
        let mut out = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &bad, &HEAP_IIDS[0], &mut out),
            E_INVALIDARG
        );
    }

    #[test]
    fn device_setup_heap_properties_and_resources_at_addresses() {
        let device = crate::device::device();
        // GetCustomHeapProperties (0xd0): the struct comes back through the
        // hidden first argument, which is also returned.
        type Custom =
            extern "system" fn(*mut HeapProperties, This, u32, u32) -> *mut HeapProperties;
        let mut p = PROPERTIES;
        let back = slot::<Custom>(device, 0xd0)(&mut p, device, 1, 2);
        assert_eq!(back, &raw mut p);
        assert_eq!(
            (
                p.kind,
                p.cpu_pages,
                p.pool,
                p.creation_nodes,
                p.visible_nodes
            ),
            (4, 3, 1, 1, 1),
            "upload: custom, write-back, L0"
        );
        slot::<Custom>(device, 0xd0)(&mut p, device, 1, 1);
        assert_eq!(p.cpu_pages, 1, "default: not CPU-visible");

        // 0x428: a buffer over memory the title owns, as GTA VI asks.
        type At = extern "system" fn(
            This,
            u64,
            *const ResourceDesc,
            u32,
            *const c_void,
            *const Iid,
            *mut This,
        ) -> HRESULT;
        let memory = vec![0u8; 0x1_0000];
        let address = memory.as_ptr() as u64;
        let mut r = std::ptr::null();
        let create = slot::<At>(device, 0x428);
        let desc = buffer(0x1_0000);
        let null = std::ptr::null();
        assert_eq!(
            create(
                device,
                address,
                &desc,
                u32::MAX,
                null,
                &RESOURCE_IIDS[0],
                &mut r
            ),
            0
        );
        assert_eq!(slot::<Address>(r, 0x58)(r), address);
        assert_eq!(slot::<Release>(r, 0x10)(r), 0);
        // The title's memory is still there after the release.
        assert_eq!(memory[0xffff], 0);
        assert_ne!(
            create(device, 0, &desc, u32::MAX, null, &RESOURCE_IIDS[0], &mut r),
            0
        );
    }

    #[test]
    fn sizes_cover_every_mip() {
        let size = |d: &ResourceDesc| footprint(d).map(|i| i.size);
        assert_eq!(size(&buffer(256)), Some(KIB_64));
        // Textures take their console layout and its alignment (4 KiB
        // standard swizzle when the description leaves it to the library):
        // 512x512 BC7 with 10 mips is 0x56000 bytes, 4 KiB aligned, as on
        // the console.
        let layout = |d: &ResourceDesc| texture_layout(d).unwrap().size;
        let rgba = texture(256, 256, 28, 0);
        assert_eq!(size(&rgba), Some(layout(&rgba).next_multiple_of(4096)));
        let bc7 = footprint(&texture(512, 512, 98, 10));
        let console = AllocationInfo {
            size: 0x56000,
            alignment: 0x1000,
        };
        assert_eq!(bc7, Some(console));
        assert!(layout(&rgba) >= 4 * (256 * 256 * 4) / 3, "every mip");
        // BC1 is 8 bytes per 4x4 block: 1024x1024, one mip = 512 KiB.
        assert_eq!(size(&texture(1024, 1024, 71, 1)), Some(512 << 10));
        assert_eq!(size(&texture(16, 16, 66, 1)), None, "1-bit format");
        // Linear layout (1) and an explicit mode (64 KiB display, 0x10a).
        let linear = ResourceDesc {
            layout: 1,
            ..texture(100, 60, 28, 1)
        };
        assert_eq!(
            texture_layout(&linear).unwrap().planes[0]
                .layout
                .desc
                .swizzle
                .0,
            0
        );
        let display = ResourceDesc {
            layout: 0x10a,
            ..texture(100, 60, 28, 1)
        };
        assert_eq!(
            texture_layout(&display).unwrap().planes[0]
                .layout
                .desc
                .swizzle
                .0,
            10
        );

        let mut info = AllocationInfo {
            size: 0,
            alignment: 0,
        };
        let descs = [buffer(100), texture(256, 256, 28, 0)];
        allocation_info(&mut info, std::ptr::null(), 0, 2, descs.as_ptr());
        assert_eq!(
            info,
            AllocationInfo {
                size: KIB_64 + size(&descs[1]).unwrap(),
                alignment: KIB_64
            }
        );
        allocation_info(&mut info, std::ptr::null(), 0, 1, &texture(1, 1, 999, 1));
        assert_eq!(info.size, u64::MAX);
    }

    #[test]
    fn upload_footprints_follow_the_pc_rules() {
        let null = std::ptr::null();
        let mut layouts = [NO_FOOTPRINT; 2];
        let (mut rows, mut sizes, mut total) = ([0; 2], [0; 2], 0);
        let mut footprints = |desc: &ResourceDesc, first, count| {
            let (l, r, s) = (layouts.as_mut_ptr(), rows.as_mut_ptr(), sizes.as_mut_ptr());
            copyable_footprints(null, desc, first, count, 0, l, r, s, &mut total);
            (layouts, rows, sizes, total)
        };
        // 100x60 RGBA8, 2 mips: 400-byte rows pitched to 512.
        let (l, r, s, t) = footprints(&texture(100, 60, 28, 2), 0, 2);
        assert_eq!((l[0].row_pitch, r[0], s[0]), (512, 60, 400));
        // Mip 1 (50x30, 200-byte rows) starts after 59 pitched rows and one
        // packed row, rounded up to 512.
        assert_eq!(l[1].offset, (512 * 59 + 400u64).next_multiple_of(512));
        assert_eq!((l[1].width, l[1].height, r[1]), (50, 30, 30));
        assert_eq!(t, l[1].offset + 256 * 29 + 200);
        // BC1: rows of 4x4 blocks, 8 bytes each.
        let (_, r, s, _) = footprints(&texture(64, 64, 71, 1), 0, 1);
        assert_eq!((r[0], s[0]), (16, 128));
        let (_, _, _, t) = footprints(&texture(64, 64, 71, 1), 1, 1);
        assert_eq!(t, u64::MAX, "no second subresource");
    }

    /// A 4x4 texture filled from a title buffer through our list and queue,
    /// then read back from the host GPU.
    #[cfg(windows)]
    #[test]
    fn buffer_data_reaches_the_host_texture() {
        use windows::Win32::Graphics::Direct3D12::*;
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        let iid = &RESOURCE_IIDS[0];
        const COPY_DEST: u32 = 0x400;
        let (mut image, mut source) = (null, null);
        let rgba = texture(4, 4, 28, 1);
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &rgba,
                COPY_DEST,
                null.cast(),
                iid,
                &mut image
            ),
            S_OK
        );
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &buffer(1024),
                0,
                null.cast(),
                iid,
                &mut source
            ),
            S_OK
        );
        let (data, _) = buffer_bytes(source).unwrap();
        // SAFETY: the buffer holds 1024 bytes.
        let bytes = unsafe { std::slice::from_raw_parts_mut(data, 1024) };
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }

        let queue_iid = guid(
            0x0ec870a6,
            0x5d7e,
            0x4c22,
            [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
        );
        let allocator_iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let (mut queue, mut allocator, mut list) = (null, null, null);
        let queue_desc = [0u32; 4];
        assert_eq!(
            crate::queue::create(null, queue_desc.as_ptr().cast(), &queue_iid, &mut queue),
            S_OK
        );
        assert_eq!(
            crate::list::create_allocator(null, 0, &allocator_iid, &mut allocator),
            S_OK
        );
        assert_eq!(
            crate::list::create_list(null, 0, 0, allocator, null, &list_iid, &mut list),
            S_OK
        );

        #[repr(C)]
        struct Location {
            resource: This,
            kind: u32,
            footprint: PlacedFootprint,
        }
        let rows = PlacedFootprint {
            offset: 0,
            format: 28,
            width: 4,
            height: 4,
            depth: 1,
            row_pitch: 256,
        };
        let to = Location {
            resource: image,
            kind: 0,
            footprint: PlacedFootprint { offset: 0, ..rows },
        };
        let from = Location {
            resource: source,
            kind: 1,
            footprint: rows,
        };
        type Copy =
            extern "system" fn(This, *const Location, u32, u32, u32, *const Location, *const u8);
        slot::<Copy>(list, 0x80)(list, &to, 0, 0, 0, &from, null.cast());
        slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list);
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);

        // Read the host texture back into a readback buffer.
        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 1024,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();
        let host_image = host_resource(image).unwrap();
        let back = host.new_list().unwrap();
        let (_, before) = host_transition(image, 0x800).unwrap();
        // The upload list holds the texture until it is reset.
        assert!(std::sync::Arc::strong_count(&host_copy(image).unwrap()) > 2);
        use windows::core::Interface;
        let out_to = Location {
            resource: readback.as_raw().cast_const().cast(),
            kind: 1,
            footprint: rows,
        };
        let out_from = Location {
            resource: host_image.as_raw().cast_const().cast(),
            kind: 0,
            footprint: to.footprint,
        };
        // SAFETY: an open list, live resources, locations in the host layout.
        unsafe {
            back.list.ResourceBarrier(&[crate::host::transition(
                host_image,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            back.list.CopyTextureRegion(
                (&raw const out_to).cast(),
                0,
                0,
                0,
                (&raw const out_from).cast(),
                None,
            );
        }
        back.close();
        host.execute(&[&back]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy finished (execute waits); the buffer is 1024 bytes.
        let read = unsafe {
            readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: 1024,
                    }),
                    Some(&mut mapped),
                )
                .unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 1024)
        };
        for row in 0..4 {
            let at = row * 256;
            assert_eq!(read[at..at + 16], bytes[at..at + 16], "row {row}");
        }
    }

    /// A depth target the title also writes as unordered access (shadow
    /// maps): what a compute shader writes through its view reaches the
    /// depth, in the list and in the next one, and the depth keeps its
    /// values where the shader does not write.
    #[cfg(windows)]
    #[test]
    fn unordered_writes_reach_a_depth_target() {
        use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::core::{Interface, s};
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null: This = std::ptr::null();
        let mut depth = null;
        // 8x8 D16_UNORM, depth and unordered access.
        let desc = ResourceDesc {
            flags: 2 | 4,
            ..texture(8, 8, 55, 1)
        };
        const DEPTH_WRITE: u32 = 0x10;
        let (iid, clear) = (&RESOURCE_IIDS[0], null.cast());
        let made = create_committed(
            null,
            &PROPERTIES,
            0,
            &desc,
            DEPTH_WRITE,
            clear,
            iid,
            &mut depth,
        );
        assert_eq!(made, S_OK);
        // The title's view, R16_UNORM 2D.
        let heap = crate::descriptor::host::HostHeap::new(false, 1).unwrap();
        let view = Some([56, 4, 0, 0, 0, 0, 0, 0, 0, 0]);
        heap.make_at(
            0,
            &crate::descriptor::View::Unordered(depth as usize, 0, view),
        );
        const FILL: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), DescriptorTable(UAV(u0))"
cbuffer Fill : register(b0) { float value; uint columns; };
RWTexture2D<unorm float> depth : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x < columns) depth[id.xy] = value;
}
"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; out pointer
        // ours. Then the blob's own bytes, the root signature inside.
        let (root, pso) = unsafe {
            let (source, len) = (FILL.as_ptr().cast(), FILL.len());
            let (name, main, model) = (s!("fill"), s!("main"), s!("cs_5_1"));
            D3DCompile(
                source, len, name, None, None, main, model, 0, 0, &mut code, None,
            )
            .unwrap();
            let code = code.unwrap();
            let (start, size) = (code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let bytes = std::slice::from_raw_parts(start, size);
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            };
            let pso: ID3D12PipelineState = host.device.CreateComputePipelineState(&desc).unwrap();
            (root, pso)
        };
        // Writes `value` to the columns before `columns`.
        let fill = |list: &crate::host::HostList, value: f32, columns: u32| {
            // As draws ready the views their shaders write.
            const UNORDERED_ACCESS: u32 = 0x8;
            crate::list::ensure_state(list, depth, UNORDERED_ACCESS);
            let constants = [value.to_bits(), columns];
            // SAFETY: an open list; our heap, root signature and pipeline.
            unsafe {
                let l = &list.list;
                l.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
                l.SetComputeRootSignature(&root);
                l.SetPipelineState(&pso);
                l.SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
                l.SetComputeRootDescriptorTable(1, heap.heap.GetGPUDescriptorHandleForHeapStart());
                l.Dispatch(1, 1, 1);
            }
        };

        // One list clears the depth, then writes four columns as it ends.
        let first = host.new_list().unwrap();
        crate::list::ensure_state(&first, depth, DEPTH_WRITE);
        let dsv = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: DXGI_FORMAT_D16_UNORM,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        let dsv = host.dsv(host_resource(depth).unwrap(), Some(&dsv));
        // SAFETY: an open list and a fresh view.
        unsafe {
            first
                .list
                .ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.25, 0, None)
        };
        fill(&first, 0.75, 4);
        first.close();
        host.execute(&[&first]);

        // The next writes two columns, then copies the depth out.
        let second = host.new_list().unwrap();
        fill(&second, 0.5, 2);
        crate::list::ensure_state(&second, depth, 0x800); // copy source
        let memory = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let buffer = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 8 * 256,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        let state = D3D12_RESOURCE_STATE_COPY_DEST;
        let flags = D3D12_HEAP_FLAG_NONE;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            let device = &host.device;
            device
                .CreateCommittedResource(&memory, flags, &buffer, state, None, &mut out)
                .unwrap();
        }
        let readback = out.unwrap();
        #[repr(C)]
        struct Location(This, u32, PlacedFootprint);
        // As the host has the depth: R16_TYPELESS.
        let rows = PlacedFootprint {
            offset: 0,
            format: 53,
            width: 8,
            height: 8,
            depth: 1,
            row_pitch: 256,
        };
        let to = Location(readback.as_raw().cast_const().cast(), 1, rows);
        let texture = host_resource(depth).unwrap().as_raw().cast_const().cast();
        let from = Location(
            texture,
            0,
            PlacedFootprint {
                offset: 0,
                ..NO_FOOTPRINT
            },
        );
        // SAFETY: an open list, live resources, locations in the host layout.
        unsafe {
            let (to, from) = ((&raw const to).cast(), (&raw const from).cast());
            second.list.CopyTextureRegion(to, 0, 0, 0, from, None)
        };
        second.close();
        host.execute(&[&second]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy finished (execute waits); 8 rows of 256 bytes.
        let read = unsafe {
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            std::slice::from_raw_parts(mapped.cast::<u16>(), 8 * 128)
        };
        for y in 0..8 {
            for x in 0..8 {
                let value: f32 = match x {
                    0..2 => 0.5,
                    2..4 => 0.75,
                    _ => 0.25,
                };
                let wanted = (value * 65535.0).round() as i32;
                let got = i32::from(read[y * 128 + x]);
                assert!(
                    (got - wanted).abs() <= 1,
                    "texel ({x}, {y}): {got:#x}, wanted {value} ({wanted:#x})"
                );
            }
        }
    }

    /// A 32-bit depth target with a stencil plane that the title also
    /// writes as unordered access, through an R32_FLOAT view of the depth
    /// plane: the writes reach the depth plane as in
    /// [`unordered_writes_reach_a_depth_target`], and the stencil plane
    /// keeps its values through the copies to the twin and back.
    #[cfg(windows)]
    #[test]
    fn unordered_writes_reach_a_depth_stencil_targets_depth_plane() {
        use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::core::{Interface, s};
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null: This = std::ptr::null();
        let mut depth = null;
        // 8x8 D32_FLOAT_S8X24_UINT, depth and unordered access.
        let desc = ResourceDesc {
            flags: 2 | 4,
            ..texture(8, 8, 20, 1)
        };
        const DEPTH_WRITE: u32 = 0x10;
        let (iid, clear) = (&RESOURCE_IIDS[0], null.cast());
        let made = create_committed(
            null,
            &PROPERTIES,
            0,
            &desc,
            DEPTH_WRITE,
            clear,
            iid,
            &mut depth,
        );
        assert_eq!(made, S_OK);
        // The title's view, R32_FLOAT 2D (plane 0).
        let heap = crate::descriptor::host::HostHeap::new(false, 1).unwrap();
        let view = Some([41, 4, 0, 0, 0, 0, 0, 0, 0, 0]);
        heap.make_at(
            0,
            &crate::descriptor::View::Unordered(depth as usize, 0, view),
        );
        const FILL: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), DescriptorTable(UAV(u0))"
cbuffer Fill : register(b0) { float value; uint columns; };
RWTexture2D<float> depth : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x < columns) depth[id.xy] = value;
}
"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; out pointer
        // ours. Then the blob's own bytes, the root signature inside.
        let (root, pso) = unsafe {
            let (source, len) = (FILL.as_ptr().cast(), FILL.len());
            let (name, main, model) = (s!("fill"), s!("main"), s!("cs_5_1"));
            D3DCompile(
                source, len, name, None, None, main, model, 0, 0, &mut code, None,
            )
            .unwrap();
            let code = code.unwrap();
            let (start, size) = (code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let bytes = std::slice::from_raw_parts(start, size);
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            };
            let pso: ID3D12PipelineState = host.device.CreateComputePipelineState(&desc).unwrap();
            (root, pso)
        };
        // Writes `value` to the columns before `columns`.
        let fill = |list: &crate::host::HostList, value: f32, columns: u32| {
            // As draws ready the views their shaders write.
            const UNORDERED_ACCESS: u32 = 0x8;
            crate::list::ensure_state(list, depth, UNORDERED_ACCESS);
            let constants = [value.to_bits(), columns];
            // SAFETY: an open list; our heap, root signature and pipeline.
            unsafe {
                let l = &list.list;
                l.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
                l.SetComputeRootSignature(&root);
                l.SetPipelineState(&pso);
                l.SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
                l.SetComputeRootDescriptorTable(1, heap.heap.GetGPUDescriptorHandleForHeapStart());
                l.Dispatch(1, 1, 1);
            }
        };

        const STENCIL: u8 = 0x5a;
        const CLEAR_BOTH: D3D12_CLEAR_FLAGS =
            D3D12_CLEAR_FLAGS(D3D12_CLEAR_FLAG_DEPTH.0 | D3D12_CLEAR_FLAG_STENCIL.0);
        // One list clears the depth and stencil, then writes four columns
        // as it ends.
        let first = host.new_list().unwrap();
        crate::list::ensure_state(&first, depth, DEPTH_WRITE);
        let dsv = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        let dsv = host.dsv(host_resource(depth).unwrap(), Some(&dsv));
        // SAFETY: an open list and a fresh view.
        unsafe {
            first
                .list
                .ClearDepthStencilView(dsv, CLEAR_BOTH, 0.25, STENCIL, None)
        };
        fill(&first, 0.75, 4);
        first.close();
        // With the debug layer (`VICTORY_D3D12_DEBUG=1`, diagnostics): what
        // the list recorded, the plane copies among it, is legal. (Executing
        // drains the messages, so they are read before.)
        assert_eq!(host.log_messages(), None, "debug layer");
        host.execute(&[&first]);

        // The next writes two columns, then copies the depth out.
        let second = host.new_list().unwrap();
        fill(&second, 0.5, 2);
        crate::list::ensure_state(&second, depth, 0x800); // copy source
        let memory = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let buffer = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 16 * 256,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        let state = D3D12_RESOURCE_STATE_COPY_DEST;
        let flags = D3D12_HEAP_FLAG_NONE;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            let device = &host.device;
            device
                .CreateCommittedResource(&memory, flags, &buffer, state, None, &mut out)
                .unwrap();
        }
        let readback = out.unwrap();
        #[repr(C)]
        struct Location(This, u32, PlacedFootprint);
        // The depth plane: R32_TYPELESS.
        let rows = PlacedFootprint {
            offset: 0,
            format: 39,
            width: 8,
            height: 8,
            depth: 1,
            row_pitch: 256,
        };
        let to = Location(readback.as_raw().cast_const().cast(), 1, rows);
        let texture = host_resource(depth).unwrap().as_raw().cast_const().cast();
        let from = Location(
            texture,
            0,
            PlacedFootprint {
                offset: 0,
                ..NO_FOOTPRINT
            },
        );
        // The stencil plane (subresource 1): R8_TYPELESS, after the depth.
        let stencil_to = Location(
            readback.as_raw().cast_const().cast(),
            1,
            PlacedFootprint {
                offset: 8 * 256,
                format: 60,
                ..rows
            },
        );
        let stencil_from = Location(
            texture,
            0,
            PlacedFootprint {
                offset: 1,
                ..NO_FOOTPRINT
            },
        );
        // SAFETY: an open list, live resources, locations in the host layout.
        unsafe {
            let (to, from) = ((&raw const to).cast(), (&raw const from).cast());
            second.list.CopyTextureRegion(to, 0, 0, 0, from, None);
            let (to, from) = (
                (&raw const stencil_to).cast(),
                (&raw const stencil_from).cast(),
            );
            second.list.CopyTextureRegion(to, 0, 0, 0, from, None);
        };
        second.close();
        // With the debug layer (`VICTORY_D3D12_DEBUG=1`, diagnostics): what
        // the list recorded, the plane copies among it, is legal. (Executing
        // drains the messages, so they are read before.)
        assert_eq!(host.log_messages(), None, "debug layer");
        host.execute(&[&second]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copies finished (execute waits); 16 rows of 256 bytes.
        let (read, stencil) = unsafe {
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            (
                std::slice::from_raw_parts(mapped.cast::<f32>(), 8 * 64),
                std::slice::from_raw_parts(mapped.cast::<u8>().add(8 * 256), 8 * 256),
            )
        };
        for y in 0..8 {
            for x in 0..8 {
                let value: f32 = match x {
                    0..2 => 0.5,
                    2..4 => 0.75,
                    _ => 0.25,
                };
                let got = read[y * 64 + x];
                assert_eq!(got, value, "depth texel ({x}, {y})");
                let got = stencil[y * 256 + x];
                assert_eq!(got, STENCIL, "stencil texel ({x}, {y})");
            }
        }
    }

    /// A depth target with a stencil plane whose stencil the title writes
    /// as unordered access, through an R8_UINT view of plane 1 (8x8, title
    /// format `format`, host depth-stencil view format `dsv_format`): the
    /// writes reach the stencil plane, in the list and in the next one, the
    /// stencil keeps its clear value where the shader does not write, and
    /// the depth plane (read back as `depth_rows`, `depth_bytes` per texel,
    /// its bits by `depth_of`) keeps the cleared `depth`.
    #[cfg(windows)]
    fn stencil_writes_reach_the_stencil_plane(
        format: u32,
        dsv_format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
        depth_rows: u32,
        depth_bytes: usize,
        depth_of: fn(&[u8]) -> u32,
        depth: u32,
    ) {
        use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::core::{Interface, s};
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null: This = std::ptr::null();
        let mut target = null;
        let desc = ResourceDesc {
            flags: 2 | 4,
            ..texture(8, 8, format, 1)
        };
        const DEPTH_WRITE: u32 = 0x10;
        let (iid, clear) = (&RESOURCE_IIDS[0], null.cast());
        let made = create_committed(
            null,
            &PROPERTIES,
            0,
            &desc,
            DEPTH_WRITE,
            clear,
            iid,
            &mut target,
        );
        assert_eq!(made, S_OK);
        // The title's view, R8_UINT 2D, plane 1.
        let heap = crate::descriptor::host::HostHeap::new(false, 1).unwrap();
        let view = Some([62, 4, 0, 1, 0, 0, 0, 0, 0, 0]);
        heap.make_at(
            0,
            &crate::descriptor::View::Unordered(target as usize, 0, view),
        );
        const FILL: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), DescriptorTable(UAV(u0))"
cbuffer Fill : register(b0) { uint value; uint columns; };
RWTexture2D<uint> stencil : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x < columns) stencil[id.xy] = value;
}
"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; out pointer
        // ours. Then the blob's own bytes, the root signature inside.
        let (root, pso) = unsafe {
            let (source, len) = (FILL.as_ptr().cast(), FILL.len());
            let (name, main, model) = (s!("fill"), s!("main"), s!("cs_5_1"));
            D3DCompile(
                source, len, name, None, None, main, model, 0, 0, &mut code, None,
            )
            .unwrap();
            let code = code.unwrap();
            let (start, size) = (code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let bytes = std::slice::from_raw_parts(start, size);
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            };
            let pso: ID3D12PipelineState = host.device.CreateComputePipelineState(&desc).unwrap();
            (root, pso)
        };
        // Writes `value` to the columns before `columns`.
        let fill = |list: &crate::host::HostList, value: u32, columns: u32| {
            const UNORDERED_ACCESS: u32 = 0x8;
            crate::list::ensure_state(list, target, UNORDERED_ACCESS);
            let constants = [value, columns];
            // SAFETY: an open list; our heap, root signature and pipeline.
            unsafe {
                let l = &list.list;
                l.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
                l.SetComputeRootSignature(&root);
                l.SetPipelineState(&pso);
                l.SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
                l.SetComputeRootDescriptorTable(1, heap.heap.GetGPUDescriptorHandleForHeapStart());
                l.Dispatch(1, 1, 1);
            }
        };

        const STENCIL: u8 = 0x5a;
        const CLEAR_BOTH: D3D12_CLEAR_FLAGS =
            D3D12_CLEAR_FLAGS(D3D12_CLEAR_FLAG_DEPTH.0 | D3D12_CLEAR_FLAG_STENCIL.0);
        // One list clears depth and stencil, then writes four columns of
        // stencil as it ends.
        let first = host.new_list().unwrap();
        crate::list::ensure_state(&first, target, DEPTH_WRITE);
        let dsv = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: dsv_format,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        let dsv = host.dsv(host_resource(target).unwrap(), Some(&dsv));
        // SAFETY: an open list and a fresh view.
        unsafe {
            first
                .list
                .ClearDepthStencilView(dsv, CLEAR_BOTH, 0.25, STENCIL, None)
        };
        fill(&first, 0x11, 4);
        first.close();
        // With the debug layer (`VICTORY_D3D12_DEBUG=1`, diagnostics): what
        // the list recorded, the plane copies among it, is legal. (Executing
        // drains the messages, so they are read before.)
        assert_eq!(host.log_messages(), None, "debug layer");
        host.execute(&[&first]);

        // The next writes two columns, then copies both planes out.
        let second = host.new_list().unwrap();
        fill(&second, 0x22, 2);
        crate::list::ensure_state(&second, target, 0x800); // copy source
        let memory = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let buffer = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 16 * 256,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        let state = D3D12_RESOURCE_STATE_COPY_DEST;
        let flags = D3D12_HEAP_FLAG_NONE;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            let device = &host.device;
            device
                .CreateCommittedResource(&memory, flags, &buffer, state, None, &mut out)
                .unwrap();
        }
        let readback = out.unwrap();
        #[repr(C)]
        struct Location(This, u32, PlacedFootprint);
        let rows = PlacedFootprint {
            offset: 0,
            format: depth_rows,
            width: 8,
            height: 8,
            depth: 1,
            row_pitch: 256,
        };
        let texture = host_resource(target).unwrap().as_raw().cast_const().cast();
        let buffer = readback.as_raw().cast_const().cast();
        // Plane 0 (subresource 0) at 0; plane 1 (subresource 1), R8_TYPELESS,
        // after it.
        let copies = [
            (
                Location(buffer, 1, rows),
                Location(
                    texture,
                    0,
                    PlacedFootprint {
                        offset: 0,
                        ..NO_FOOTPRINT
                    },
                ),
            ),
            (
                Location(
                    buffer,
                    1,
                    PlacedFootprint {
                        offset: 8 * 256,
                        format: 60,
                        ..rows
                    },
                ),
                Location(
                    texture,
                    0,
                    PlacedFootprint {
                        offset: 1,
                        ..NO_FOOTPRINT
                    },
                ),
            ),
        ];
        for (to, from) in &copies {
            // SAFETY: an open list, live resources, locations in the host
            // layout.
            unsafe {
                let (to, from) = ((&raw const *to).cast(), (&raw const *from).cast());
                second.list.CopyTextureRegion(to, 0, 0, 0, from, None);
            }
        }
        second.close();
        assert_eq!(host.log_messages(), None, "debug layer");
        host.execute(&[&second]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copies finished (execute waits); 16 rows of 256 bytes.
        let read = unsafe {
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 16 * 256)
        };
        for y in 0..8 {
            for x in 0..8 {
                let wanted = match x {
                    0..2 => 0x22,
                    2..4 => 0x11,
                    _ => STENCIL,
                };
                let got = read[8 * 256 + y * 256 + x];
                assert_eq!(got, wanted, "stencil texel ({x}, {y})");
                let at = y * 256 + x * depth_bytes;
                let got = depth_of(&read[at..at + depth_bytes]);
                assert_eq!(got, depth, "depth texel ({x}, {y})");
            }
        }
    }

    /// [`stencil_writes_reach_the_stencil_plane`] on D24_UNORM_S8_UINT (the
    /// host's depth for the console's 16-bit depth with stencil); its depth
    /// plane reads back as R32_TYPELESS, the depth in the low 24 bits.
    #[cfg(windows)]
    #[test]
    fn stencil_writes_reach_a_d24s8_targets_stencil_plane() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D24_UNORM_S8_UINT;
        let depth_of = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & 0xFF_FFFF;
        let depth = (0.25 * f64::from(0xFF_FFFF_u32)).round() as u32;
        stencil_writes_reach_the_stencil_plane(
            45,
            DXGI_FORMAT_D24_UNORM_S8_UINT,
            39,
            4,
            depth_of,
            depth,
        );
    }

    /// [`stencil_writes_reach_the_stencil_plane`] on D32_FLOAT_S8X24_UINT.
    #[cfg(windows)]
    #[test]
    fn stencil_writes_reach_a_d32s8_targets_stencil_plane() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
        let depth_of = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        stencil_writes_reach_the_stencil_plane(
            20,
            DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
            39,
            4,
            depth_of,
            0.25f32.to_bits(),
        );
    }

    /// What a test draw tests against a depth target the title writes from
    /// a compute shader: its stencil, equal to `reference`, or its depth,
    /// less than the depth the draw is at (0.5).
    #[cfg(windows)]
    #[derive(Clone, Copy)]
    enum Tested {
        Stencil { reference: u32 },
        Depth,
    }

    /// A depth target the title writes from a compute shader through the
    /// unordered access view `view` (8x8, title format `format`; values
    /// `written`, as the shader's `kind` stores them), then binds as the
    /// depth target of a draw that tests it (`tested`, host view format
    /// `dsv_format`): the draw sees the writes. The first list clears depth
    /// 0.25 and stencil 0x5a and writes columns 0-3, the writes reaching the
    /// target as that list ends; the next writes columns 4-5 and draws in
    /// the same list, the writes reaching the target before the draw, as
    /// draws make their depth target current (`ensure_state`, as
    /// `State::prepare_for` does). The draw colors columns 0-5 only.
    #[cfg(windows)]
    fn tested_draws_see_unordered_writes(
        format: u32,
        view: [u32; 10],
        (kind, written): (&str, &str),
        dsv_format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
        tested: Tested,
    ) {
        use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::core::{PCSTR, s};
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let device = &host.device;
        let compile = |source: &str, model: PCSTR| -> Vec<u8> {
            let mut code: Option<ID3DBlob> = None;
            let mut errors: Option<ID3DBlob> = None;
            // SAFETY: the source and names live through the call; out
            // pointers ours; then the blob's own bytes.
            unsafe {
                let made = D3DCompile(
                    source.as_ptr().cast(),
                    source.len(),
                    s!("test"),
                    None,
                    None,
                    s!("main"),
                    model,
                    0,
                    0,
                    &mut code,
                    Some(&mut errors),
                );
                if made.is_err() {
                    let e = errors.unwrap();
                    let text = std::slice::from_raw_parts(
                        e.GetBufferPointer().cast::<u8>(),
                        e.GetBufferSize(),
                    );
                    panic!("{}", String::from_utf8_lossy(text));
                }
                let code = code.unwrap();
                std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
                    .to_vec()
            }
        };
        let null: This = std::ptr::null();
        let mut target = null;
        let desc = ResourceDesc {
            flags: 2 | 4,
            ..texture(8, 8, format, 1)
        };
        const DEPTH_WRITE: u32 = 0x10;
        let made = create_committed(
            null,
            &PROPERTIES,
            0,
            &desc,
            DEPTH_WRITE,
            null.cast(),
            &RESOURCE_IIDS[0],
            &mut target,
        );
        assert_eq!(made, S_OK);
        let heap = crate::descriptor::host::HostHeap::new(false, 1).unwrap();
        heap.make_at(
            0,
            &crate::descriptor::View::Unordered(target as usize, 0, Some(view)),
        );

        // The title's compute shader: `written` to columns [from, to).
        let fill = format!(
            r#"
#define RS "RootConstants(num32BitConstants=2, b0), DescriptorTable(UAV(u0))"
cbuffer Fill : register(b0) {{ uint from; uint to; }};
RWTexture2D<{kind}> target : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
    if (id.x >= from && id.x < to) target[id.xy] = {written};
}}
"#
        );
        let code = compile(&fill, s!("cs_5_1"));
        // SAFETY: the shader's bytes, the root signature inside.
        let (fill_root, fill_pso) = unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &code).unwrap();
            let pso: ID3D12PipelineState = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        };
        let fill = |list: &crate::host::HostList, from: u32, to: u32| {
            const UNORDERED_ACCESS: u32 = 0x8;
            // As `readable` does for a table's unordered access view.
            crate::list::ensure_state(list, target, UNORDERED_ACCESS);
            let constants = [from, to];
            // SAFETY: an open list; our heap, root signature and pipeline.
            unsafe {
                let l = &list.list;
                l.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
                l.SetComputeRootSignature(&fill_root);
                l.SetPipelineState(&fill_pso);
                l.SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
                l.SetComputeRootDescriptorTable(1, heap.heap.GetGPUDescriptorHandleForHeapStart());
                l.Dispatch(1, 1, 1);
            }
        };

        // The draw: one triangle over the whole target at depth 0.5, white
        // where the test passes.
        const DRAW: &str = r#"
#define RS "RootFlags(0)"
[RootSignature(RS)]
float4 main(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0.5, 1);
}
"#;
        const WHITE: &str = r#"
float4 main() : SV_Target { return 1; }
"#;
        let (vs, ps) = (compile(DRAW, s!("vs_5_1")), compile(WHITE, s!("ps_5_1")));
        let equal = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_EQUAL,
        };
        let depth_test = match tested {
            Tested::Stencil { .. } => D3D12_DEPTH_STENCIL_DESC {
                StencilEnable: true.into(),
                StencilReadMask: 0xff,
                FrontFace: equal,
                BackFace: equal,
                ..Default::default()
            },
            Tested::Depth => D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                ..Default::default()
            },
        };
        // SAFETY: the shaders' bytes, the root signature inside the vertex
        // shader; the description points at live data.
        let (draw_root, draw_pso) = unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &vs).unwrap();
            let mut blend = D3D12_BLEND_DESC::default();
            blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
            formats[0] = DXGI_FORMAT_R8_UNORM;
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: vs.as_ptr().cast(),
                    BytecodeLength: vs.len(),
                },
                PS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: ps.as_ptr().cast(),
                    BytecodeLength: ps.len(),
                },
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                DepthStencilState: depth_test,
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: formats,
                DSVFormat: dsv_format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let pso: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc).unwrap();
            (root, pso)
        };

        // The color target (ours, not the title's) and its read-back
        // buffer: 8 rows of 256 bytes.
        let gpu_memory = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let color_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 8,
            Height: 8,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let read_memory = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let read_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 8 * 256,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let (mut color, mut readback) = (None::<ID3D12Resource>, None::<ID3D12Resource>);
        // SAFETY: plain creation; out pointers ours.
        unsafe {
            let none = D3D12_HEAP_FLAG_NONE;
            device
                .CreateCommittedResource(
                    &gpu_memory,
                    none,
                    &color_desc,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut color,
                )
                .unwrap();
            device
                .CreateCommittedResource(
                    &read_memory,
                    none,
                    &read_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let (color, readback) = (color.unwrap(), readback.unwrap());
        let dsv_desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: dsv_format,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };

        const CLEAR_BOTH: D3D12_CLEAR_FLAGS =
            D3D12_CLEAR_FLAGS(D3D12_CLEAR_FLAG_DEPTH.0 | D3D12_CLEAR_FLAG_STENCIL.0);
        let first = host.new_list().unwrap();
        crate::list::ensure_state(&first, target, DEPTH_WRITE);
        let dsv = host.dsv(host_resource(target).unwrap(), Some(&dsv_desc));
        // SAFETY: an open list and a fresh view.
        unsafe { first.list.ClearDepthStencilView(dsv, CLEAR_BOTH, 0.25, 0x5a, None) };
        fill(&first, 0, 4);
        first.close();
        assert_eq!(host.log_messages(), None, "debug layer");
        host.execute(&[&first]);

        let second = host.new_list().unwrap();
        fill(&second, 4, 6);
        // As a draw readies its depth target (`State::prepare_for`).
        crate::list::ensure_state(&second, target, DEPTH_WRITE);
        let dsv = host.dsv(host_resource(target).unwrap(), Some(&dsv_desc));
        let rtv = host.rtv(&color, None);
        // SAFETY: an open list; our pipeline, fresh views of live textures,
        // the target in the depth state, the color target in its own.
        unsafe {
            let l = &second.list;
            l.ClearRenderTargetView(rtv, &[0.0; 4], None);
            l.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
            l.RSSetViewports(&[D3D12_VIEWPORT {
                Width: 8.0,
                Height: 8.0,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            l.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                right: 8,
                bottom: 8,
                ..Default::default()
            }]);
            l.SetGraphicsRootSignature(&draw_root);
            l.SetPipelineState(&draw_pso);
            l.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            if let Tested::Stencil { reference } = tested {
                l.OMSetStencilRef(reference);
            }
            l.DrawInstanced(3, 1, 0, 0);
            l.ResourceBarrier(&[crate::host::transition(
                &color,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R8_UNORM,
                            Width: 8,
                            Height: 8,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&color),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        second.close();
        assert_eq!(host.log_messages(), None, "debug layer");
        host.execute(&[&second]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy finished (execute waits); 8 rows of 256 bytes.
        let read = unsafe {
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 8 * 256)
        };
        for y in 0..8 {
            for x in 0..8 {
                let wanted = if x < 6 { 255 } else { 0 };
                assert_eq!(read[y * 256 + x], wanted, "color texel ({x}, {y})");
            }
        }
    }

    /// [`tested_draws_see_unordered_writes`]: stencil written through an
    /// R8_UINT view of plane 1 of the console's 16-bit depth with stencil
    /// (a D24S8 target on the host), then tested equal by a draw.
    #[cfg(windows)]
    #[test]
    fn stencil_tested_draws_see_unordered_stencil_writes() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D24_UNORM_S8_UINT;
        tested_draws_see_unordered_writes(
            118,
            [62, 4, 0, 1, 0, 0, 0, 0, 0, 0],
            ("uint", "0x22"),
            DXGI_FORMAT_D24_UNORM_S8_UINT,
            Tested::Stencil { reference: 0x22 },
        );
    }

    /// The same through a view of plane 1 that names the console's stencil
    /// format (X16_TYPELESS_G8_UINT): the stencil's twin takes it as
    /// R8_UINT.
    #[cfg(windows)]
    #[test]
    fn stencil_views_in_a_stencil_format_write_the_stencil() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D24_UNORM_S8_UINT;
        tested_draws_see_unordered_writes(
            118,
            [120, 4, 0, 1, 0, 0, 0, 0, 0, 0],
            ("uint", "0x22"),
            DXGI_FORMAT_D24_UNORM_S8_UINT,
            Tested::Stencil { reference: 0x22 },
        );
    }

    /// [`tested_draws_see_unordered_writes`]: depth 0.75 written through an
    /// R16_UNORM view of a D16 target, then tested by a draw at 0.5 (less
    /// passes: where the shader wrote, not where the clear left 0.25).
    #[cfg(windows)]
    #[test]
    fn depth_tested_draws_see_unordered_depth_writes() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D16_UNORM;
        tested_draws_see_unordered_writes(
            55,
            [56, 4, 0, 0, 0, 0, 0, 0, 0, 0],
            ("unorm float", "0.75"),
            DXGI_FORMAT_D16_UNORM,
            Tested::Depth,
        );
    }

    /// HLSL `source` compiled for `model` (entry point `main`); panics with
    /// the compiler's errors.
    #[cfg(windows)]
    fn compiled(source: &str, model: windows::core::PCSTR) -> Vec<u8> {
        use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
        use windows::core::s;
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; out pointers
        // ours; then the blobs' own bytes.
        unsafe {
            let made = D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("test"),
                None,
                None,
                s!("main"),
                model,
                0,
                0,
                &mut code,
                Some(&mut errors),
            );
            let bytes = |b: &ID3DBlob| {
                std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize())
                    .to_vec()
            };
            if made.is_err() {
                panic!("{}", String::from_utf8_lossy(&bytes(&errors.unwrap())));
            }
            bytes(&code.unwrap())
        }
    }

    /// The console's 16-bit depth with stencil (a D24S8 target on the host)
    /// whose depth plane the title writes from a compute shader through an
    /// R16_UNORM view of plane 0, then tests with draws at those values'
    /// depths, depth equal (as tiled lighting tests material ids): each draw
    /// passes exactly where the value written is its own. The values are 0,
    /// 65535 and ones whose 24-bit depth lies next to a rounding tie (k/257
    /// near half an integer); pixel (x, y) gets the `(x + y) % 8`-th.
    ///
    /// The first list clears depth (to one of the values, `CLEARED`) and
    /// stencil, writes columns 0-3 and ends; the next writes columns 4-5, so
    /// the depth plane goes to the twin and back, columns 6-7 keeping the
    /// cleared value. Before it writes, the second list binds the draws'
    /// targets, viewport and a scissor that leaves out row 7, as the title
    /// does before its draws, and the draws find them set back after the
    /// pass that writes the twin to the depth. The draws also test the
    /// stencil equal to the cleared one: the passes keep it. Each draw adds 1
    /// to red and its number to green, so a pixel shows how many passed and
    /// which.
    #[cfg(windows)]
    #[test]
    fn equal_tested_draws_see_16_bit_depth_writes() {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::core::s;
        const VALUES: [u32; 8] = [0, 1, 128, 129, 25828, 51529, 65534, 65535];
        const CLEARED: usize = 5;
        const STENCIL: u32 = 0x5a;
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let device = &host.device;
        let null: This = std::ptr::null();
        let mut target = null;
        let desc = ResourceDesc {
            flags: 2 | 4,
            ..texture(8, 8, 118, 1)
        };
        const DEPTH_WRITE: u32 = 0x10;
        let made = create_committed(
            null,
            &PROPERTIES,
            0,
            &desc,
            DEPTH_WRITE,
            null.cast(),
            &RESOURCE_IIDS[0],
            &mut target,
        );
        assert_eq!(made, S_OK);
        // The title's view: R16_UNORM 2D, plane 0.
        let heap = crate::descriptor::host::HostHeap::new(false, 1).unwrap();
        let view = [56, 4, 0, 0, 0, 0, 0, 0, 0, 0];
        heap.make_at(
            0,
            &crate::descriptor::View::Unordered(target as usize, 0, Some(view)),
        );

        // The title's compute shader: the values to columns [from, to).
        let values = VALUES.map(|v| v.to_string()).join(", ");
        let fill = format!(
            r#"
#define RS "RootConstants(num32BitConstants=2, b0), DescriptorTable(UAV(u0))"
cbuffer Fill : register(b0) {{ uint from; uint to; }};
RWTexture2D<unorm float> target : register(u0);
static const uint VALUES[8] = {{ {values} }};
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
    if (id.x >= from && id.x < to) target[id.xy] = VALUES[(id.x + id.y) % 8] / 65535.0;
}}
"#
        );
        let code = compiled(&fill, s!("cs_5_1"));
        // SAFETY: the shader's bytes, the root signature inside.
        let (fill_root, fill_pso) = unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &code).unwrap();
            let pso: ID3D12PipelineState = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        };
        let fill = |list: &crate::host::HostList, from: u32, to: u32| {
            const UNORDERED_ACCESS: u32 = 0x8;
            // As `readable` does for a table's unordered access view, before
            // the dispatch binds its state (`State::bind`).
            crate::list::ensure_state(list, target, UNORDERED_ACCESS);
            let constants = [from, to];
            // SAFETY: an open list; our heap, root signature and pipeline.
            unsafe {
                let l = &list.list;
                l.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
                l.SetComputeRootSignature(&fill_root);
                l.SetPipelineState(&fill_pso);
                l.SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
                l.SetComputeRootDescriptorTable(1, heap.heap.GetGPUDescriptorHandleForHeapStart());
                l.Dispatch(1, 1, 1);
            }
        };

        // The draws: one triangle over the target at `depth`, adding 1 to
        // red and `id` to green where the tests pass.
        const DRAW: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0)"
cbuffer Draw : register(b0) { float depth; uint id; };
[RootSignature(RS)]
float4 main(uint vertex : SV_VertexID) : SV_Position {
    float2 uv = float2((vertex << 1) & 2, vertex & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), depth, 1);
}
"#;
        const COUNT: &str = r#"
cbuffer Draw : register(b0) { float depth; uint id; };
float4 main() : SV_Target { return float4(1, id, 0, 0) / 255; }
"#;
        let (vs, ps) = (compiled(DRAW, s!("vs_5_1")), compiled(COUNT, s!("ps_5_1")));
        let equal = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_EQUAL,
        };
        // SAFETY: the shaders' bytes, the root signature inside the vertex
        // shader; the description points at live data.
        let (draw_root, draw_pso) = unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &vs).unwrap();
            let mut blend = D3D12_BLEND_DESC::default();
            blend.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
                BlendEnable: true.into(),
                SrcBlend: D3D12_BLEND_ONE,
                DestBlend: D3D12_BLEND_ONE,
                BlendOp: D3D12_BLEND_OP_ADD,
                SrcBlendAlpha: D3D12_BLEND_ONE,
                DestBlendAlpha: D3D12_BLEND_ONE,
                BlendOpAlpha: D3D12_BLEND_OP_ADD,
                RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                ..Default::default()
            };
            let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
            formats[0] = DXGI_FORMAT_R8G8_UNORM;
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: vs.as_ptr().cast(),
                    BytecodeLength: vs.len(),
                },
                PS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: ps.as_ptr().cast(),
                    BytecodeLength: ps.len(),
                },
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D12_COMPARISON_FUNC_EQUAL,
                    StencilEnable: true.into(),
                    StencilReadMask: 0xff,
                    StencilWriteMask: 0,
                    FrontFace: equal,
                    BackFace: equal,
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: formats,
                DSVFormat: DXGI_FORMAT_D24_UNORM_S8_UINT,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let pso: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc).unwrap();
            (root, pso)
        };

        // The color target (ours, not the title's) and its read-back
        // buffer: 8 rows of 256 bytes.
        let color_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 8,
            Height: 8,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let read_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 8 * 256,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let (mut color, mut readback) = (None::<ID3D12Resource>, None::<ID3D12Resource>);
        // SAFETY: plain creation; out pointers ours.
        unsafe {
            let memory = |kind| D3D12_HEAP_PROPERTIES {
                Type: kind,
                ..Default::default()
            };
            let none = D3D12_HEAP_FLAG_NONE;
            device
                .CreateCommittedResource(
                    &memory(D3D12_HEAP_TYPE_DEFAULT),
                    none,
                    &color_desc,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut color,
                )
                .unwrap();
            device
                .CreateCommittedResource(
                    &memory(D3D12_HEAP_TYPE_READBACK),
                    none,
                    &read_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let (color, readback) = (color.unwrap(), readback.unwrap());
        let dsv_desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: DXGI_FORMAT_D24_UNORM_S8_UINT,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        // A value's depth as the twin reads it, and the draws put it.
        let depth_of = |value: u32| value as f32 / 65535.0;

        const CLEAR_BOTH: D3D12_CLEAR_FLAGS =
            D3D12_CLEAR_FLAGS(D3D12_CLEAR_FLAG_DEPTH.0 | D3D12_CLEAR_FLAG_STENCIL.0);
        let first = host.new_list().unwrap();
        crate::list::ensure_state(&first, target, DEPTH_WRITE);
        let dsv = host.dsv(host_resource(target).unwrap(), Some(&dsv_desc));
        let cleared = depth_of(VALUES[CLEARED]);
        // SAFETY: an open list and a fresh view.
        unsafe {
            first
                .list
                .ClearDepthStencilView(dsv, CLEAR_BOTH, cleared, STENCIL as u8, None)
        };
        fill(&first, 0, 4);
        first.close();
        assert_eq!(host.log_messages(), None, "debug layer");
        host.execute(&[&first]);

        let second = host.new_list().unwrap();
        // The title's state for its draws, bound while the twin is current.
        let depth = host_resource(target).unwrap().clone();
        second.set_targets(vec![(color.clone(), None)], Some((depth, Some(dsv_desc))));
        second.set_viewports(&[D3D12_VIEWPORT {
            Width: 8.0,
            Height: 8.0,
            MaxDepth: 1.0,
            ..Default::default()
        }]);
        second.set_scissors(&[RECT {
            right: 8,
            bottom: 7,
            ..Default::default()
        }]);
        second.set_topology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        fill(&second, 4, 6);
        // As a draw readies its depth target (`State::prepare_for`).
        crate::list::ensure_state(&second, target, DEPTH_WRITE);
        let rtv = host.rtv(&color, None);
        // SAFETY: an open list; our pipeline, fresh views of live textures,
        // the target in the depth state, the color target in its own.
        unsafe {
            let l = &second.list;
            l.ClearRenderTargetView(rtv, &[0.0; 4], None);
            // The draws' own pipeline, as `State::bind` binds it.
            l.SetGraphicsRootSignature(&draw_root);
            l.SetPipelineState(&draw_pso);
            l.OMSetStencilRef(STENCIL);
            for (id, &value) in (1u32..).zip(&VALUES) {
                let constants = [depth_of(value).to_bits(), id];
                l.SetGraphicsRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
                l.DrawInstanced(3, 1, 0, 0);
            }
            l.ResourceBarrier(&[crate::host::transition(
                &color,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R8G8_UNORM,
                            Width: 8,
                            Height: 8,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&color),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        second.close();
        assert_eq!(host.log_messages(), None, "debug layer");
        host.execute(&[&second]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy finished (execute waits); 8 rows of 256 bytes.
        let read = unsafe {
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 8 * 256)
        };
        for y in 0..8 {
            for x in 0..8 {
                // (Draws that passed, the sum of their numbers.)
                let wanted: (usize, usize) = match (x, y) {
                    (_, 7) => (0, 0), // outside the title's scissor
                    (0..6, _) => (1, (x + y) % 8 + 1),
                    _ => (1, CLEARED + 1),
                };
                let got = (read[y * 256 + 2 * x], read[y * 256 + 2 * x + 1]);
                assert_eq!(
                    (usize::from(got.0), usize::from(got.1)),
                    wanted,
                    "texel ({x}, {y}): draws passed, sum of their numbers"
                );
            }
        }
    }

    /// What [`console_equivalent_depths`] found: per form of the draws'
    /// depth, the texels its draws did not pass as on the console, and the
    /// texels the probe read other than written (each described).
    #[cfg(windows)]
    struct Found {
        misses: [Vec<String>; 2],
        probe: Vec<String>,
    }

    /// The console's 16-bit depth with stencil (title format 118) written by
    /// the title's compute shader through an R16_UNORM view of plane 0 and
    /// tested by draws whose vertex shaders put each value's depth where the
    /// console's 16 bits hold it, but not at the float the twin's value
    /// reads as (as the title's tiled lighting need not): form 0 at
    /// `float(k) * (1.0 / 65535.0)`, form 1 at `(k + 0.25) / 65535` (`k -
    /// 0.25` for 65535, inside the depth range). Depth equal, no depth
    /// writes, no stencil, as the title's lighting draws. On the console
    /// each value's draws pass exactly where it was written; on the host
    /// that takes 16-bit depth (`d16`, [`d16_depth`]): D24 stores form 1's
    /// depth 64 steps from the written one's.
    ///
    /// The first list clears depth (to one of the values, `CLEARED`) and
    /// stencil (0x5a) through the title's depth view, then writes columns
    /// 0-6 of the depth and the stencil of column 6 (0x33, through an
    /// R8_UINT view of plane 1); the second binds the draws' targets (the
    /// title's depth view keeps the stencil read-only) and draws each value
    /// in both forms, each adding 1 and its number to its form's channels.
    /// Unless the target is denied shader resources (`readable` false), a
    /// probe then reads the depth plane and the stencil through the title's
    /// shader views (R16_UNORM_X8_TYPELESS, X16_TYPELESS_G8_UINT of plane
    /// 1). `None` without a host GPU.
    #[cfg(windows)]
    fn console_equivalent_depths(d16: bool, readable: bool) -> Option<Found> {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::core::s;
        const VALUES: [u32; 8] = [0, 1, 128, 129, 25828, 51529, 65534, 65535];
        const CLEARED: usize = 5;
        const STENCIL: u32 = 0x5a;
        const WRITTEN: u32 = 0x33;
        const DEPTH_WRITE: u32 = 0x10;
        const UNORDERED_ACCESS: u32 = 0x8;
        const SHADER_RESOURCE: u32 = 0x40 | 0x80;
        const COPY_SOURCE: u32 = 0x800;
        let host = crate::host::host()?;
        /// Sets [`D16_FOR_TEST`] back as it drops.
        struct Back;
        impl Drop for Back {
            fn drop(&mut self) {
                D16_FOR_TEST.with(|on| on.set(false));
            }
        }
        D16_FOR_TEST.with(|on| on.set(d16));
        let _back = Back;
        let device = &host.device;
        let null: This = std::ptr::null();
        let made = |desc: &ResourceDesc, state| {
            let mut resource = null;
            let iid = &RESOURCE_IIDS[0];
            let made = create_committed(
                null,
                &PROPERTIES,
                0,
                desc,
                state,
                null.cast(),
                iid,
                &mut resource,
            );
            assert_eq!(made, S_OK);
            resource
        };
        const DENY_SHADER_RESOURCE: u32 = 8;
        let deny = if readable { 0 } else { DENY_SHADER_RESOURCE };
        let target = made(
            &ResourceDesc {
                flags: 2 | 4 | deny,
                ..texture(8, 8, 118, 1)
            },
            DEPTH_WRITE,
        );
        // What the probe writes: R32_UINT.
        let probed = made(
            &ResourceDesc {
                flags: 4,
                ..texture(8, 8, 42, 1)
            },
            UNORDERED_ACCESS,
        );
        // SAFETY: a live resource.
        let host_format = unsafe { host_resource(target).unwrap().GetDesc() }.Format;
        let wanted_format = match (d16, readable) {
            (true, true) => DXGI_FORMAT_R16_TYPELESS,
            (true, false) => DXGI_FORMAT_D16_UNORM,
            (false, true) => DXGI_FORMAT_R24G8_TYPELESS,
            (false, false) => DXGI_FORMAT_D24_UNORM_S8_UINT,
        };
        assert_eq!(host_format, wanted_format, "the host's depth format");
        assert_eq!(host_stencil(target).is_some(), d16, "a stencil apart");

        // The title's depth views: to clear, and the draws' (stencil
        // read-only), recorded as `CreateDepthStencilView` records them.
        let mut slots = [[0u64; 16]; 2];
        for (slot, flags) in slots.iter_mut().zip([0, 2]) {
            let words = [118, 3, flags, 0, 0, 0];
            crate::descriptor::create_dsv(null, target, &words, slot.as_mut_ptr() as u64);
        }
        let [clear_view, draw_view] =
            slots.map(|slot| crate::descriptor::dsv_record(slot.as_ptr() as u64).unwrap());

        // The title's compute shader: the values to columns 0-6, the
        // stencil of column 6.
        let fill_views = crate::descriptor::host::HostHeap::new(false, 2).unwrap();
        let depth_uav = [56, 4, 0, 0, 0, 0, 0, 0, 0, 0];
        let stencil_uav = [62, 4, 0, 1, 0, 0, 0, 0, 0, 0];
        for (slot, words) in (0..).zip([depth_uav, stencil_uav]) {
            let view = crate::descriptor::View::Unordered(target as usize, 0, Some(words));
            fill_views.make_at(slot, &view);
        }
        let values = VALUES.map(|v| v.to_string()).join(", ");
        let fill = format!(
            r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=2))"
RWTexture2D<unorm float> depth : register(u0);
RWTexture2D<uint> stencil : register(u1);
static const uint VALUES[8] = {{ {values} }};
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
    if (id.x < 7) depth[id.xy] = VALUES[(id.x + id.y) % 8] / 65535.0;
    if (id.x == 6) stencil[id.xy] = {WRITTEN};
}}
"#
        );
        // The probe: the depth plane as 16 bits, the stencil above them.
        const PROBE: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=2), UAV(u0))"
Texture2D<float> depth : register(t0);
Texture2D<uint2> stencil : register(t1);
RWTexture2D<uint> probed : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int3 at = int3(id.xy, 0);
    probed[id.xy] = uint(depth.Load(at) * 65535.0 + 0.5) | stencil.Load(at).g << 16;
}
"#;
        let compute = |source: &str| {
            let code = compiled(source, s!("cs_5_1"));
            // SAFETY: the shader's bytes, the root signature inside.
            unsafe {
                let root: ID3D12RootSignature = device.CreateRootSignature(0, &code).unwrap();
                let pso: ID3D12PipelineState = device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: code.as_ptr().cast(),
                            BytecodeLength: code.len(),
                        },
                        ..Default::default()
                    })
                    .unwrap();
                (root, pso)
            }
        };
        let (fill_root, fill_pso) = compute(&fill);
        let (probe_root, probe_pso) = compute(PROBE);
        let probe_views = crate::descriptor::host::HostHeap::new(false, 3).unwrap();
        if readable {
            // The title's shader views, in the default mapping.
            const MAPPING: u32 = 0x1688;
            let depth_srv = [119, 4, MAPPING, 0, 0, 1, 0, 0, 0, 0];
            let stencil_srv = [120, 4, MAPPING, 0, 0, 1, 1, 0, 0, 0];
            for (slot, words) in (0..).zip([depth_srv, stencil_srv]) {
                let view = crate::descriptor::View::Shader(target as usize, Some(words));
                probe_views.make_at(slot, &view);
            }
            let view = crate::descriptor::View::Unordered(probed as usize, 0, None);
            probe_views.make_at(2, &view);
        }

        // The draws: one triangle over the target at value `k`'s depth in
        // form `form`, adding 1 and `id` to the form's two channels where
        // the depth test passes.
        const DRAW: &str = r#"
#define RS "RootConstants(num32BitConstants=3, b0)"
cbuffer Draw : register(b0) { uint k; uint form; uint id; };
[RootSignature(RS)]
float4 main(uint vertex : SV_VertexID) : SV_Position {
    float2 uv = float2((vertex << 1) & 2, vertex & 2);
    float off = k == 65535 ? -0.25 : 0.25;
    float depth = form == 0 ? float(k) * (1.0 / 65535.0) : (float(k) + off) / 65535.0;
    return float4(uv * float2(2, -2) + float2(-1, 1), depth, 1);
}
"#;
        const COUNT: &str = r#"
cbuffer Draw : register(b0) { uint k; uint form; uint id; };
float4 main() : SV_Target {
    return (form == 0 ? float4(1, id, 0, 0) : float4(0, 0, 1, id)) / 255;
}
"#;
        let (vs, ps) = (compiled(DRAW, s!("vs_5_1")), compiled(COUNT, s!("ps_5_1")));
        // SAFETY: the shaders' bytes, the root signature inside the vertex
        // shader; the description points at live data.
        let (draw_root, draw_pso) = unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &vs).unwrap();
            let mut blend = D3D12_BLEND_DESC::default();
            blend.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
                BlendEnable: true.into(),
                SrcBlend: D3D12_BLEND_ONE,
                DestBlend: D3D12_BLEND_ONE,
                BlendOp: D3D12_BLEND_OP_ADD,
                SrcBlendAlpha: D3D12_BLEND_ONE,
                DestBlendAlpha: D3D12_BLEND_ONE,
                BlendOpAlpha: D3D12_BLEND_OP_ADD,
                RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                ..Default::default()
            };
            let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
            formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: vs.as_ptr().cast(),
                    BytecodeLength: vs.len(),
                },
                PS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: ps.as_ptr().cast(),
                    BytecodeLength: ps.len(),
                },
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D12_COMPARISON_FUNC_EQUAL,
                    StencilEnable: false.into(),
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: formats,
                // The title's pipeline's depth format, as the host takes it.
                DSVFormat: DXGI_FORMAT(pc_format(118) as i32),
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let pso: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc).unwrap();
            (root, pso)
        };

        // The color target (ours) and the read-back buffer: the color's 8
        // rows of 256 bytes, then the probe's.
        let color_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 8,
            Height: 8,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        const PROBED_AT: usize = 8 * 256;
        let read_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 2 * 8 * 256,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let (mut color, mut readback) = (None::<ID3D12Resource>, None::<ID3D12Resource>);
        // SAFETY: plain creation; out pointers ours.
        unsafe {
            let memory = |kind| D3D12_HEAP_PROPERTIES {
                Type: kind,
                ..Default::default()
            };
            let none = D3D12_HEAP_FLAG_NONE;
            device
                .CreateCommittedResource(
                    &memory(D3D12_HEAP_TYPE_DEFAULT),
                    none,
                    &color_desc,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut color,
                )
                .unwrap();
            device
                .CreateCommittedResource(
                    &memory(D3D12_HEAP_TYPE_READBACK),
                    none,
                    &read_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let (color, readback) = (color.unwrap(), readback.unwrap());
        // `texture`'s 8x8 texels in `format` into the read-back buffer at
        // `offset`, 256 bytes a row.
        let read_back = |list: &crate::host::HostList,
                         texture: &ID3D12Resource,
                         format: DXGI_FORMAT,
                         offset: usize| {
            // SAFETY: an open list; live resources, the texture a copy source.
            unsafe {
                let to = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&readback),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: offset as u64,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: format,
                                Width: 8,
                                Height: 8,
                                Depth: 1,
                                RowPitch: 256,
                            },
                        },
                    },
                };
                let from = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(texture),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: 0,
                    },
                };
                list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            }
        };
        let depth_of = |value: u32| value as f32 / 65535.0;

        let first = host.new_list().unwrap();
        crate::list::clear_depth_stencil(
            &first,
            &clear_view,
            1 | 2,
            depth_of(VALUES[CLEARED]),
            STENCIL as u8,
            None,
        );
        // As `readable` does for a table's unordered access views, before
        // the dispatch binds its state (`State::bind`).
        crate::list::ensure_state(&first, target, UNORDERED_ACCESS);
        // SAFETY: an open list; our heap, root signature and pipeline.
        unsafe {
            let l = &first.list;
            let table = fill_views.heap.GetGPUDescriptorHandleForHeapStart();
            l.SetDescriptorHeaps(&[Some(fill_views.heap.clone())]);
            l.SetComputeRootSignature(&fill_root);
            l.SetPipelineState(&fill_pso);
            l.SetComputeRootDescriptorTable(0, table);
            l.Dispatch(1, 1, 1);
        }
        first.close();
        assert_eq!(host.log_messages(), None, "debug layer");
        host.execute(&[&first]);

        let second = host.new_list().unwrap();
        // The title's state for its draws, as `State::set_targets` and the
        // fixed-function calls set it.
        let depth = host_resource(target).unwrap().clone();
        second.set_targets(
            vec![(color.clone(), None)],
            Some((depth, draw_view.depth_desc())),
        );
        second.set_viewports(&[D3D12_VIEWPORT {
            Width: 8.0,
            Height: 8.0,
            MaxDepth: 1.0,
            ..Default::default()
        }]);
        second.set_scissors(&[RECT {
            right: 8,
            bottom: 8,
            ..Default::default()
        }]);
        second.set_topology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        // As a draw readies its depth target (`State::prepare_for`).
        crate::list::ensure_state(&second, target, DEPTH_WRITE);
        let rtv = host.rtv(&color, None);
        // SAFETY: an open list; our pipeline, fresh views of live textures
        // in the states they need.
        unsafe {
            let l = &second.list;
            l.ClearRenderTargetView(rtv, &[0.0; 4], None);
            l.SetGraphicsRootSignature(&draw_root);
            l.SetPipelineState(&draw_pso);
            for form in 0..2 {
                for (id, &k) in (1u32..).zip(&VALUES) {
                    let constants = [k, form, id];
                    l.SetGraphicsRoot32BitConstants(0, 3, constants.as_ptr().cast(), 0);
                    l.DrawInstanced(3, 1, 0, 0);
                }
            }
            l.ResourceBarrier(&[crate::host::transition(
                &color,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
        }
        read_back(&second, &color, DXGI_FORMAT_R8G8B8A8_UNORM, 0);
        if readable {
            // As a dispatch readies what it reads and writes.
            crate::list::ensure_state(&second, target, SHADER_RESOURCE);
            crate::list::ensure_state(&second, probed, UNORDERED_ACCESS);
            // SAFETY: an open list; our heap, root signature and pipeline.
            unsafe {
                let l = &second.list;
                let table = probe_views.heap.GetGPUDescriptorHandleForHeapStart();
                l.SetDescriptorHeaps(&[Some(probe_views.heap.clone())]);
                l.SetComputeRootSignature(&probe_root);
                l.SetPipelineState(&probe_pso);
                l.SetComputeRootDescriptorTable(0, table);
                l.Dispatch(1, 1, 1);
            }
            crate::list::ensure_state(&second, probed, COPY_SOURCE);
            let probed = host_resource(probed).unwrap();
            read_back(&second, probed, DXGI_FORMAT_R32_UINT, PROBED_AT);
        }
        second.close();
        assert_eq!(host.log_messages(), None, "debug layer");
        host.execute(&[&second]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copies finished (execute waits); the whole buffer.
        let read = unsafe {
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 2 * 8 * 256)
        };
        let mut found = Found {
            misses: [Vec::new(), Vec::new()],
            probe: Vec::new(),
        };
        for y in 0..8 {
            for x in 0..8 {
                let i = if x < 7 { (x + y) % 8 } else { CLEARED };
                let texel = &read[y * 256 + 4 * x..][..4];
                for (form, misses) in found.misses.iter_mut().enumerate() {
                    // Draws that passed, the sum of their numbers.
                    let got = (texel[2 * form], texel[2 * form + 1]);
                    if got != (1, i as u8 + 1) {
                        misses.push(format!("({x}, {y}) holds {}: {got:?}", VALUES[i]));
                    }
                }
                if readable {
                    let at = PROBED_AT + y * 256 + 4 * x;
                    let got = u32::from_le_bytes(read[at..at + 4].try_into().unwrap());
                    let stencil = if x == 6 { WRITTEN } else { STENCIL };
                    let wanted = VALUES[i] | stencil << 16;
                    if got != wanted {
                        found.probe.push(format!("({x}, {y}): {got:#x}, not {wanted:#x}"));
                    }
                }
            }
        }
        Some(found)
    }

    /// The console's 16-bit depth as host D16 ([`d16_depth`]): draws at its
    /// values' depths worked out otherwise than the title's 16 bits read
    /// pass depth equal exactly where each value was written, as on the
    /// console, whether the twin is copied to a typeless or a typed depth
    /// target; the stencil apart is cleared, written and read through the
    /// title's views; the debug layer reports nothing.
    #[cfg(windows)]
    #[test]
    fn equal_tested_draws_at_console_equivalent_depths_pass_on_16_bit_depth() {
        for readable in [true, false] {
            let Some(found) = console_equivalent_depths(true, readable) else {
                return; // no host GPU here
            };
            for (form, misses) in found.misses.iter().enumerate() {
                assert_eq!(misses, &[] as &[String], "form {form}, readable {readable}");
            }
            assert_eq!(found.probe, &[] as &[String], "probe, readable {readable}");
        }
    }

    /// The same as D24S8 with passes of ours (the console's 16-bit depth
    /// without [`d16_depth`]): what was written reads back, but draws at
    /// `(k + 0.25) / 65535` (16-bit `k` on the console) miss every texel,
    /// D24 holding their depth 64 steps from the written one's: why the
    /// title's lighting found none of its material ids.
    #[cfg(windows)]
    #[test]
    fn equal_tested_draws_off_the_written_depth_miss_on_24_bit_depth() {
        let Some(found) = console_equivalent_depths(false, true) else {
            return; // no host GPU here
        };
        assert_eq!(found.probe, &[] as &[String], "probe");
        assert_eq!(found.misses[1].len(), 64, "form 1 passes nowhere");
        // Form 0 (at k * (1 / 65535)) may land on the written depth's 24
        // bits or not, by GPU; shown for the record.
        eprintln!("form 0 misses on D24S8: {:?}", found.misses[0]);
    }

    #[test]
    fn committed_and_placed_resources_map_at_their_gpu_address() {
        let null = std::ptr::null();
        let mut committed = null;
        let hr = create_committed(
            null,
            &PROPERTIES,
            0,
            &buffer(256),
            0,
            null.cast(),
            &RESOURCE_IIDS[0],
            &mut committed,
        );
        assert_eq!(hr, S_OK);
        let mut data = std::ptr::null_mut();
        assert_eq!(
            slot::<Map>(committed, 0x40)(committed, 0, null.cast(), &mut data),
            S_OK
        );
        assert_eq!(slot::<Address>(committed, 0x58)(committed), data as u64);
        // SAFETY: the buffer is at least 256 zeroed bytes.
        assert_eq!(unsafe { *data.add(255) }, 0);
        assert_eq!(slot::<Release>(committed, 0x10)(committed), 0);

        let heap = heap(4 * KIB_64);
        let place = |offset, desc: &ResourceDesc, out: &mut This| {
            create_placed(
                null,
                heap,
                offset,
                desc,
                0,
                null.cast(),
                &RESOURCE_IIDS[0],
                out,
            )
        };
        let mut placed = null;
        assert_eq!(place(KIB_64, &buffer(KIB_64), &mut placed), S_OK);
        // SAFETY: a heap we made.
        let base = unsafe { (*heap.cast::<Heap>()).memory.ptr.as_ptr() } as u64;
        assert_eq!(slot::<Address>(placed, 0x58)(placed), base + KIB_64);
        let mut other = null;
        assert_eq!(
            place(4 * KIB_64, &buffer(1), &mut other),
            E_INVALIDARG,
            "past the end"
        );
        assert_eq!(
            place(100, &buffer(1), &mut other),
            E_INVALIDARG,
            "misaligned"
        );
        // The resource keeps the heap alive.
        assert_eq!(slot::<Release>(heap, 0x10)(heap), 1);
        assert_eq!(slot::<Release>(placed, 0x10)(placed), 0);
    }
}
