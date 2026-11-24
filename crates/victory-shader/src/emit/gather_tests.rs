//! Gathers at an explicit level (`image_gather4_l`, `image_gather4_l_o`),
//! which DXIL's gather (level 0 only) cannot express: hand-assembled GFX10
//! compute programs lowered to DXIL, validated by `dxil.dll` when present,
//! and run on the host GPU and WARP over a texture with a chain of levels,
//! each lane's four values checked against the console's footprint.

use windows::Win32::Graphics::Direct3D12::*;

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Level 0's side; the chain halves it down to 1x1.
const SIDE: u32 = 8;
const LEVELS: u32 = 4;

/// The program's start: the views' table in s0 (the texture at entry 0,
/// a raw UAV at entry 1), the samplers' in s1; lane l takes u = (l & 7) /
/// 8 + 0.03, v = 1 - u and level (l >> 3) * 1.2 (v2, v3, v4), and its
/// result's byte offset 16 l (v16).
const PROLOGUE: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF40C_0302,
    0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
    0xF408_0203,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
    0xF408_0A02,
    0xFA00_0020, // s_load_dwordx4 s[40:43], s[4:5], 0x20
    0x3602_0087, // v_and_b32 v1, 7, v0
    0x7E04_0D01, // v_cvt_f32_u32 v2, v1
    0x1004_04FF,
    0x3E00_0000, // v_mul_f32 v2, 0.125, v2
    0x0604_04FF,
    0x3CF5_C28F, // v_add_f32 v2, 0.03, v2
    0x0806_04F2, // v_sub_f32 v3, 1.0, v2
    0x2C02_0083, // v_lshrrev_b32 v1, 3, v0
    0x7E08_0D01, // v_cvt_f32_u32 v4, v1
    0x1008_08FF,
    0x3F99_999A, // v_mul_f32 v4, 1.2, v4
    0x3420_0084, // v_lshlrev_b32 v16, 4, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
];

/// The program's end: the four gathered values to the UAV.
const EPILOGUE: &[u32] = &[
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE078_1000,
    0x800A_0C10, // buffer_store_dwordx4 v[12:15], v16, s[40:43], 0 offen
    0xBF81_0000, // s_endpgm
];

/// `image_gather4_l v[12:15], v[2:4], s[12:19], s[8:11] dmask:0x1 dim:2D`.
const GATHER4_L: &[u32] = &[0xF110_0108, 0x0043_0C02];

/// The texel offset word (x +1, y -1) in v1, then
/// `image_gather4_l_o v[12:15], v[1:4], s[12:19], s[8:11] dmask:0x1 dim:2D`.
const GATHER4_L_O: &[u32] = &[
    0x7E02_02FF,
    0x0000_3F01, // v_mov_b32 v1, 0x3f01
    0xF150_0108,
    0x0043_0C01,
];

/// Lowers a wave32 compute program of 32 threads with `body` between the
/// prologue and the epilogue.
fn build(body: &[u32]) -> HostShader {
    let table = |user_data: u8| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    };
    let range = |kind, table_offset_dwords, dwords, user_data| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot: user_data,
    };
    let words: Vec<u32> = [PROLOGUE, body, EPILOGUE].concat();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![table(0), table(1)],
        used: vec![
            range(ResourceType::Srv, 0, 8, 0),
            range(ResourceType::Uav, 8, 4, 0),
            range(ResourceType::Sampler, 0, 4, 1),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        target: super::Target::ANY,
        console_cf: false,
        per_lane_only: false,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        stride_table: lowered.stride_table,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// Texel (x, y) of level `level`: its red channel names all three.
fn texel(level: u32, x: u32, y: u32) -> f32 {
    (level * 100 + y * 10 + x) as f32
}

/// The levels' texels in turn, level 0 first.
fn chain() -> Vec<[f32; 4]> {
    (0..LEVELS)
        .flat_map(|level| {
            let side = (SIDE >> level).max(1);
            (0..side * side).map(move |at| [texel(level, at % side, at / side), 0.0, 0.0, 1.0])
        })
        .collect()
}

/// What the console's gather returns to lane `lane` with texel offset
/// `offset`: the 2x2 texels around the lane's coordinates at the nearest
/// level (within the chain), clamped to the edges, in gather order.
fn expected(lane: u32, offset: (i32, i32)) -> [f32; 4] {
    let u = (lane & 7) as f32 * 0.125 + 0.03;
    let v = 1.0 - u;
    let lod = (lane >> 3) as f32 * 1.2;
    let level = ((lod + 0.5).floor() as u32).min(LEVELS - 1);
    let side = (SIDE >> level).max(1) as i32;
    let first = |c: f32, o: i32| (c * side as f32 - 0.5).floor() as i32 + o;
    let (i0, j0) = (first(u, offset.0), first(v, offset.1));
    let at = |i: i32, j: i32| {
        texel(
            level,
            i.clamp(0, side - 1) as u32,
            j.clamp(0, side - 1) as u32,
        )
    };
    [
        at(i0, j0 + 1),
        at(i0 + 1, j0 + 1),
        at(i0 + 1, j0),
        at(i0, j0),
    ]
}

/// Runs `shader` over the chain with a point, clamping sampler on the
/// hardware GPU and on WARP (each skipped without a device): per one that
/// ran, each lane's four values.
fn run(shader: &HostShader) -> Vec<(&'static str, Vec<[f32; 4]>)> {
    let samplers = [D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    }];
    let texels = chain();
    let views = [
        View::Mips(SIDE, SIDE, LEVELS, &texels),
        View::RawUav(32 * 4),
    ];
    let mut out = Vec::new();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        // Both tables start at heap index 0.
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &samplers,
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let lanes = words[0]
            .chunks(4)
            .map(|c| [0, 1, 2, 3].map(|k| f32::from_bits(c[k])))
            .collect();
        out.push((label, lanes));
    }
    out
}

#[test]
fn a_gather_at_an_explicit_level_reads_that_level() {
    let shader = build(GATHER4_L);
    let want: Vec<[f32; 4]> = (0..32).map(|l| expected(l, (0, 0))).collect();
    // The cases cover levels 0 to 2, a level past the chain (clamped to its
    // last) and footprints across the left and bottom edges.
    assert!(want.iter().any(|w| w[0] >= 300.0));
    for (label, got) in run(&shader) {
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn a_gather_at_an_explicit_level_moves_by_its_texel_offset() {
    let shader = build(GATHER4_L_O);
    let want: Vec<[f32; 4]> = (0..32).map(|l| expected(l, (1, -1))).collect();
    assert_ne!(
        want,
        (0..32).map(|l| expected(l, (0, 0))).collect::<Vec<_>>()
    );
    for (label, got) in run(&shader) {
        assert_eq!(got, want, "{label}");
    }
}
