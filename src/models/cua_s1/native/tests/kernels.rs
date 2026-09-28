//! GPU checks of the attention and Gated DeltaNet kernels on random inputs, and of the
//! GEMM plan import. They need
//! a GPU and CUA_S1_CUDA_LIB pointing at libqwen3_5_cuda.so, so they only run when
//! asked for:
//!
//!     CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
//!       cargo test --release -p omni-cua-s1-native --test kernels -- --ignored

use std::path::PathBuf;

use half::bf16;
use omni_cua_s1_native::cuda::{self, DeviceBuffer, GemmPlan, Stream, api, check};

fn setup() -> Stream {
    let lib = std::env::var_os("CUA_S1_CUDA_LIB")
        .map(PathBuf::from)
        .expect("CUA_S1_CUDA_LIB must point at libqwen3_5_cuda.so");
    cuda::load(&lib).unwrap();
    cuda::set_device(0).unwrap();
    cuda::new_stream().unwrap()
}

/// Uniform values in [-amp, amp), rounded to bfloat16, from a fixed seed.
fn random(n: usize, seed: u64, amp: f32) -> Vec<bf16> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            bf16::from_f32(((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp)
        })
        .collect()
}

fn to_device(v: &[bf16], st: Stream) -> DeviceBuffer {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let buf = DeviceBuffer::new(bytes.len()).unwrap();
    // SAFETY: the buffer was allocated for these bytes.
    unsafe { cuda::upload(buf.at(0), &bytes, st).unwrap() };
    buf
}

fn f32_to_device(v: &[f32], st: Stream) -> DeviceBuffer {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let buf = DeviceBuffer::new(bytes.len()).unwrap();
    // SAFETY: the buffer was allocated for these bytes.
    unsafe { cuda::upload(buf.at(0), &bytes, st).unwrap() };
    buf
}

fn from_device(buf: &DeviceBuffer, n: usize, st: Stream) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 2];
    // SAFETY: the buffer holds n bfloat16 values.
    unsafe { cuda::download(&mut bytes, buf.at(0), st).unwrap() };
    bytes
        .chunks_exact(2)
        .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB"]
fn flash_attention_matches_float32_kernel() {
    let st = setup();
    let (hq, hk, dh) = (16usize, 4usize, 256usize);
    for (t, amp) in [
        (1, 8.0),
        (63, 8.0),
        (65, 0.5),
        (139, 2.0),
        (700, 8.0),
        (2048, 0.5),
    ] {
        let q = to_device(&random(t * hq * dh, 1, amp), st);
        let k = to_device(&random(t * hk * dh, 2, amp), st);
        // v is read in place from the q|k|v projection output, rows of 10240 as in the model
        let (ldv, v_at) = (10240usize, (hq * 2 + hk) * dh);
        let qkv = to_device(&random(t * ldv, 3, 1.0), st);
        let v = qkv.at(v_at * 2);
        let flash = DeviceBuffer::new(t * hq * dh * 2).unwrap();
        let simple = DeviceBuffer::new(t * hq * dh * 2).unwrap();
        let (ti, hqi, hki, dhi, ldv) = (t as i32, hq as i32, hk as i32, dh as i32, ldv as i32);
        // SAFETY: every buffer holds t rows of the given widths.
        unsafe {
            check(
                (api().cs1_attention)(
                    q.at(0),
                    k.at(0),
                    v,
                    ldv,
                    flash.at(0),
                    ti,
                    hqi,
                    hki,
                    dhi,
                    0.0625,
                    st,
                ),
                "flash",
            )
            .unwrap();
            check(
                (api().cs1_attention_simple)(
                    q.at(0),
                    k.at(0),
                    v,
                    ldv,
                    simple.at(0),
                    ti,
                    hqi,
                    hki,
                    dhi,
                    0.0625,
                    st,
                ),
                "simple",
            )
            .unwrap();
        }
        let a = from_device(&flash, t * hq * dh, st);
        let b = from_device(&simple, t * hq * dh, st);
        // per (token, head): the largest difference over the largest magnitude
        let mut worst = 0f32;
        for (ra, rb) in a.chunks_exact(dh).zip(b.chunks_exact(dh)) {
            let d = ra
                .iter()
                .zip(rb)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            let m = rb.iter().map(|y| y.abs()).fold(1e-3f32, f32::max);
            assert!(
                ra.iter().all(|x| x.is_finite()),
                "non-finite output at t = {t}"
            );
            worst = worst.max(d / m);
        }
        eprintln!("attention t = {t}, amplitude {amp}: largest relative difference {worst:.2e}");
        assert!(worst < 1.6e-2, "t = {t}: {worst}");
    }
}

/// Transformers' torch_recurrent_gated_delta_rule in float64, one token at a time,
/// with the L2 norms of q and k and q scaled by K^-1/2.
#[allow(clippy::too_many_arguments)]
fn gated_delta_reference(
    q: &[bf16],
    k: &[bf16],
    v: &[bf16],
    g: &[f32],
    beta: &[bf16],
    t: usize,
    h: usize,
    hk: usize,
    d: usize,
) -> Vec<f64> {
    let mut out = vec![0f64; t * h * d];
    for head in 0..h {
        let kh = head / (h / hk);
        let mut s = vec![0f64; d * d]; // [K][V]
        for tok in 0..t {
            let norm = |x: &[bf16]| {
                let x: Vec<f64> = x.iter().map(|v| v.to_f64()).collect();
                let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() + 1e-6).sqrt();
                x.into_iter().map(|v| v * inv).collect::<Vec<f64>>()
            };
            let qv: Vec<f64> = norm(&q[(tok * hk + kh) * d..][..d])
                .into_iter()
                .map(|x| x / (d as f64).sqrt())
                .collect();
            let kv = norm(&k[(tok * hk + kh) * d..][..d]);
            let vv: Vec<f64> = v[(tok * h + head) * d..][..d]
                .iter()
                .map(|x| x.to_f64())
                .collect();
            let decay = (g[tok * h + head] as f64).exp();
            let b = beta[tok * h + head].to_f64();
            s.iter_mut().for_each(|x| *x *= decay);
            for j in 0..d {
                let mem: f64 = (0..d).map(|i| kv[i] * s[i * d + j]).sum();
                let delta = (vv[j] - mem) * b;
                for i in 0..d {
                    s[i * d + j] += kv[i] * delta;
                }
            }
            for j in 0..d {
                out[(tok * h + head) * d + j] = (0..d).map(|i| qv[i] * s[i * d + j]).sum();
            }
        }
    }
    out
}

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB"]
fn gated_delta_rule_matches_recurrent_reference() {
    let st = setup();
    let (h, hk, d) = (4usize, 2usize, 128usize);
    for t in [1usize, 64, 150] {
        // q close to k, so that q.k and the outputs are of order one as in the model
        let k = random(t * hk * d, 12, 1.0);
        let q: Vec<bf16> = k
            .iter()
            .zip(random(t * hk * d, 11, 1.0))
            .map(|(k, n)| bf16::from_f32(0.8 * k.to_f32() + 0.2 * n.to_f32()))
            .collect();
        let v = random(t * h * d, 13, 1.0);
        // log decays in (-2, 0) and learning rates in (0, 1), as sigmoid and -exp * softplus give
        let g: Vec<f32> = random(t * h, 14, 1.0)
            .iter()
            .map(|x| x.to_f32() - 1.0)
            .collect();
        let beta: Vec<bf16> = random(t * h, 15, 0.5)
            .iter()
            .map(|x| bf16::from_f32(x.to_f32() + 0.5))
            .collect();
        let want = gated_delta_reference(&q, &k, &v, &g, &beta, t, h, hk, d);
        let (qd, kd, vd, gd, bd) = (
            to_device(&q, st),
            to_device(&k, st),
            to_device(&v, st),
            f32_to_device(&g, st),
            to_device(&beta, st),
        );
        let o = DeviceBuffer::new(t * h * d * 2).unwrap();
        // SAFETY: pure function of its arguments.
        let floats = unsafe { (api().cs1_gdn_workspace_floats)(t as i32, h as i32) };
        let ws = DeviceBuffer::new(floats * 4).unwrap();
        // SAFETY: every buffer holds t rows of the given widths, the workspace its size.
        unsafe {
            check(
                (api().cs1_gdn_prefill)(
                    qd.at(0),
                    kd.at(0),
                    vd.at(0),
                    gd.at(0).cast::<f32>(),
                    bd.at(0),
                    o.at(0),
                    ws.at(0).cast::<f32>(),
                    t as i32,
                    h as i32,
                    hk as i32,
                    (d as f32).powf(-0.5),
                    st,
                ),
                "gdn prefill",
            )
            .unwrap();
        }
        let got = from_device(&o, t * h * d, st);
        let scale = want.iter().fold(0f64, |m, x| m.max(x.abs()));
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as f64 - b).abs())
            .fold(0f64, f64::max);
        eprintln!(
            "gated delta t = {t}: largest difference {worst:.2e}, largest |reference| {scale:.2}"
        );
        assert!(worst <= 2e-2 * scale, "t = {t}: {worst} vs scale {scale}");
    }
}

/// A tuned GEMM plan moves to another cuBLASLt handle through export and import, and a
/// plan that says it was tuned with another cuBLASLt version is refused without
/// changing anything.
#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB"]
fn gemm_plans_import_only_for_their_cublaslt_version() {
    let st = setup();
    let (m, n, k) = (64i32, 256i32, 512i32);
    let x = to_device(&random((m * k) as usize, 21, 1.0), st);
    let w = to_device(&random((n * k) as usize, 22, 1.0), st);
    let y = DeviceBuffer::new((m * n * 2) as usize).unwrap();
    // SAFETY: the buffers hold m x k, n x k and m x n values; the handles are destroyed
    // at the end and not used after.
    unsafe {
        let api = api();
        let (a, b) = (
            (api.cs1_gemm_create)(32 << 20),
            (api.cs1_gemm_create)(32 << 20),
        );
        assert!(!a.is_null() && !b.is_null());
        check(
            (api.cs1_gemm_tune)(a, x.at(0), w.at(0), y.at(0), m, n, k, n, 0, st),
            "tune",
        )
        .unwrap();
        (api.cs1_gemm_tune_done)(a);
        let count = (api.cs1_gemm_export)(a, std::ptr::null_mut(), 0);
        assert_eq!(count, 1);
        let mut plans = vec![GemmPlan::default(); count];
        (api.cs1_gemm_export)(a, plans.as_mut_ptr(), count);
        assert_eq!(plans[0].cublaslt_version, (api.cs1_gemm_version)() as u64);

        let mut other = plans.clone();
        other[0].cublaslt_version += 1;
        assert_ne!((api.cs1_gemm_import)(b, other.as_ptr(), 1), 0);
        assert_eq!((api.cs1_gemm_export)(b, std::ptr::null_mut(), 0), 0);
        check((api.cs1_gemm_import)(b, plans.as_ptr(), 1), "import").unwrap();
        assert_eq!((api.cs1_gemm_export)(b, std::ptr::null_mut(), 0), 1);
        (api.cs1_gemm_destroy)(a);
        (api.cs1_gemm_destroy)(b);
    }
}
