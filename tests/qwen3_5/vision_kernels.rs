//! Opt-in GPU checks for additive vision kernels; CPU workspace tests skip these.
use half::bf16;
use omni_qwen3_5_native::cuda::{self, DeviceBuffer, Stream, api, check};
use std::path::PathBuf;
fn setup() -> Stream {
    let path = PathBuf::from(std::env::var_os("CUA_S1_CUDA_LIB").expect("CUA_S1_CUDA_LIB"));
    cuda::load(&path).unwrap();
    cuda::set_device(0).unwrap();
    cuda::new_stream().unwrap()
}
fn random(n: usize, seed: u64) -> Vec<bf16> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            bf16::from_f32((x >> 40) as f32 / (1u64 << 24) as f32 * 2. - 1.)
        })
        .collect()
}
fn upload(bytes: &[u8], stream: Stream) -> DeviceBuffer {
    let buffer = DeviceBuffer::new(bytes.len()).unwrap();
    // SAFETY: allocation holds the supplied byte slice.
    unsafe { cuda::upload(buffer.at(0), bytes, stream).unwrap() };
    buffer
}
fn bf(v: &[bf16], s: Stream) -> DeviceBuffer {
    upload(
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
        s,
    )
}
fn floats(v: &[f32], s: Stream) -> DeviceBuffer {
    upload(
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
        s,
    )
}
fn read(x: &DeviceBuffer, n: usize, s: Stream) -> Vec<bf16> {
    let mut bytes = vec![0; n * 2];
    // SAFETY: callers supply an allocation for at least n BF16 elements.
    unsafe { cuda::download(&mut bytes, x.at(0), s).unwrap() };
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|x| bf16::from_le_bytes(*x))
        .collect()
}
#[test]
#[ignore = "needs GPU and rebuilt CUA_S1_CUDA_LIB"]
fn v2_64_preserves_legacy_position_rope_attention_exactly() {
    let s = setup();
    let v2 = api().vision_v2().unwrap();
    for n in [1usize, 31, 65, 129] {
        let table = bf(&random(2304 * 1024, 3), s);
        let input = random(n * 1024, 5);
        let x = bf(&input, s);
        let y = bf(&input, s);
        let indices: Vec<i32> = (0..n * 4).map(|i| (i % 2304) as i32).collect();
        let ids = upload(
            &indices
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<_>>(),
            s,
        );
        let weights = floats(&vec![0.25; n * 4], s);
        let qkv = bf(&random(n * 3072, 7), s);
        let co = floats(&vec![0.6; n * 32], s);
        let si = floats(&vec![0.8; n * 32], s);
        let q1 = DeviceBuffer::new(n * 1024 * 2).unwrap();
        let k1 = DeviceBuffer::new(n * 1024 * 2).unwrap();
        let q2 = DeviceBuffer::new(n * 1024 * 2).unwrap();
        let k2 = DeviceBuffer::new(n * 1024 * 2).unwrap();
        let o1 = DeviceBuffer::new(n * 1024 * 2).unwrap();
        let o2 = DeviceBuffer::new(n * 1024 * 2).unwrap();
        // SAFETY: every allocation matches the legacy and v2 16*64 shapes.
        unsafe {
            check(
                (api().cs1_vision_position)(
                    x.at(0),
                    table.at(0),
                    ids.at(0).cast(),
                    weights.at(0).cast(),
                    n as i32,
                    s,
                ),
                "legacy positions",
            )
            .unwrap();
            check(
                (v2.position)(
                    y.at(0),
                    table.at(0),
                    ids.at(0).cast(),
                    weights.at(0).cast(),
                    n as i32,
                    1024,
                    s,
                ),
                "v2 positions",
            )
            .unwrap();
            check(
                (api().cs1_vision_rope)(
                    qkv.at(0),
                    co.at(0).cast(),
                    si.at(0).cast(),
                    q1.at(0),
                    k1.at(0),
                    n as i32,
                    s,
                ),
                "legacy rope",
            )
            .unwrap();
            check(
                (v2.rope)(
                    qkv.at(0),
                    co.at(0).cast(),
                    si.at(0).cast(),
                    q2.at(0),
                    k2.at(0),
                    n as i32,
                    1024,
                    64,
                    s,
                ),
                "v2 rope",
            )
            .unwrap();
            check(
                (api().cs1_vision_attention)(
                    q1.at(0),
                    k1.at(0),
                    qkv.at(2048 * 2),
                    o1.at(0),
                    n as i32,
                    s,
                ),
                "legacy attention",
            )
            .unwrap();
            check(
                (v2.attention)(
                    q2.at(0),
                    k2.at(0),
                    qkv.at(2048 * 2),
                    o2.at(0),
                    n as i32,
                    16,
                    64,
                    std::ptr::null_mut(),
                    s,
                ),
                "v2 attention",
            )
            .unwrap();
        }
        assert_eq!(
            read(&x, n * 1024, s),
            read(&y, n * 1024, s),
            "positions n={n}"
        );
        assert_eq!(read(&q1, n * 1024, s), read(&q2, n * 1024, s), "q n={n}");
        assert_eq!(read(&k1, n * 1024, s), read(&k2, n * 1024, s), "k n={n}");
        assert_eq!(
            read(&o1, n * 1024, s),
            read(&o2, n * 1024, s),
            "attention n={n}"
        );
    }
    cuda::synchronize(s).unwrap();
    unsafe { (api().cs1_stream_destroy)(s) };
}
#[test]
#[ignore = "needs GPU and rebuilt CUA_S1_CUDA_LIB"]
fn v2_72_full_attention_matches_float64_reference_and_is_image_local() {
    let s = setup();
    let v2 = api().vision_v2().unwrap();
    for (image, n) in [1usize, 31, 65, 129].into_iter().enumerate() {
        let h = 16usize;
        let d = 72usize;
        let hidden = h * d;
        let q = random(n * hidden, 11 + image as u64);
        let k = random(n * hidden, 23 + image as u64);
        let qkv = random(n * hidden * 3, 37 + image as u64);
        let qd = bf(&q, s);
        let kd = bf(&k, s);
        let vd = bf(&qkv, s);
        let out = DeviceBuffer::new(n * hidden * 2).unwrap();
        // Poison scratch so missing zero padding cannot pass from an empty fresh allocation.
        let scratch = bf(&vec![bf16::from_f32(9.); n * h * 80 * 4], s);
        // SAFETY: q/k compact [n,16,72], v a slice of [n,3*hidden], padded workspace sized per ABI.
        unsafe {
            check(
                (v2.attention)(
                    qd.at(0),
                    kd.at(0),
                    vd.at(hidden * 2 * 2),
                    out.at(0),
                    n as i32,
                    h as i32,
                    d as i32,
                    scratch.at(0),
                    s,
                ),
                "v2 attention72",
            )
            .unwrap()
        };
        let actual = read(&out, n * hidden, s);
        let mut worst = 0f64;
        for t in 0..n {
            for head in 0..h {
                let mut scores: Vec<f64> = (0..n)
                    .map(|j| {
                        (0..d)
                            .map(|i| {
                                q[(t * h + head) * d + i].to_f64()
                                    * k[(j * h + head) * d + i].to_f64()
                            })
                            .sum::<f64>()
                            / (d as f64).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for score in &mut scores {
                    *score = (*score - max).exp()
                }
                let total = scores.iter().sum::<f64>();
                let mut row_error = 0f64;
                let mut row_scale = 1e-3f64;
                for i in 0..d {
                    let expected = scores
                        .iter()
                        .enumerate()
                        .map(|(j, p)| {
                            p / total * qkv[j * hidden * 3 + hidden * 2 + head * d + i].to_f64()
                        })
                        .sum::<f64>();
                    let found = actual[(t * h + head) * d + i].to_f64();
                    assert!(found.is_finite());
                    row_error = row_error.max((found - expected).abs());
                    row_scale = row_scale.max(expected.abs());
                    if n == 1 {
                        assert_eq!(
                            actual[head * d + i],
                            qkv[hidden * 2 + head * d + i],
                            "one image's sole patch cannot attend to another image"
                        );
                    }
                }
                worst = worst.max(row_error / row_scale);
            }
        }
        // Same per-row relative bound as the existing Qwen attention tests.
        assert!(worst < 1.6e-2, "image={image} n={n} worst={worst}");
    }
    cuda::synchronize(s).unwrap();
    unsafe { (api().cs1_stream_destroy)(s) };
}
#[test]
#[ignore = "needs GPU and rebuilt CUA_S1_CUDA_LIB"]
fn v2_72_rotary_matches_float32_product_order() {
    let s = setup();
    let v2 = api().vision_v2().unwrap();
    let n = 65usize;
    let h = 1152usize;
    let d = 72usize;
    let half = d / 2;
    let input = random(n * h * 3, 41);
    let qkv = bf(&input, s);
    let angles: Vec<f32> = (0..n * half).map(|i| (i % half) as f32 * 0.07).collect();
    let co: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
    let si: Vec<f32> = angles.iter().map(|a| a.sin()).collect();
    let cod = floats(&co, s);
    let sid = floats(&si, s);
    let q = DeviceBuffer::new(n * h * 2).unwrap();
    let k = DeviceBuffer::new(n * h * 2).unwrap();
    // SAFETY: full 72-dim Q/K and 36-value cos/sin rows.
    unsafe {
        check(
            (v2.rope)(
                qkv.at(0),
                cod.at(0).cast(),
                sid.at(0).cast(),
                q.at(0),
                k.at(0),
                n as i32,
                h as i32,
                d as i32,
                s,
            ),
            "rope72",
        )
        .unwrap()
    };
    for (part, actual) in [read(&q, n * h, s), read(&k, n * h, s)]
        .into_iter()
        .enumerate()
    {
        for (i, found) in actual.iter().enumerate() {
            let token = i / h;
            let channel = i % h;
            let dim = channel % d;
            let partner = if dim < half {
                channel + half
            } else {
                channel - half
            };
            let a =
                input[token * h * 3 + part * h + channel].to_f32() * co[token * half + dim % half];
            let b = if dim < half { -1. } else { 1. }
                * input[token * h * 3 + part * h + partner].to_f32()
                * si[token * half + dim % half];
            assert_eq!(*found, bf16::from_f32(a + b), "part={part}i={i}");
        }
    }
    cuda::synchronize(s).unwrap();
    unsafe { (api().cs1_stream_destroy)(s) };
}

#[test]
#[ignore = "needs GPU and rebuilt CUA_S1_CUDA_LIB"]
fn v2_1152_positions_match_separate_float32_interpolation_and_rounding() {
    let s = setup();
    let v2 = api().vision_v2().unwrap();
    let n = 65usize;
    let h = 1152usize;
    let table = random(2304 * h, 17);
    let input = random(n * h, 19);
    let td = bf(&table, s);
    let x = bf(&input, s);
    let indices: Vec<i32> = (0..n * 4).map(|i| ((i * 113 + 29) % 2304) as i32).collect();
    let weights: Vec<f32> = (0..n * 4)
        .map(|i| [0.14, 0.21, 0.26, 0.39][i % 4])
        .collect();
    let ids = upload(
        &indices
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect::<Vec<_>>(),
        s,
    );
    let wd = floats(&weights, s);
    // SAFETY: four legal table indices per token and complete 1152-dimensional rows.
    unsafe {
        check(
            (v2.position)(
                x.at(0),
                td.at(0),
                ids.at(0).cast(),
                wd.at(0).cast(),
                n as i32,
                h as i32,
                s,
            ),
            "positions1152",
        )
        .unwrap()
    };
    let actual = read(&x, n * h, s);
    for (i, found) in actual.iter().enumerate() {
        let t = i / h;
        let d = i % h;
        let mut sum = 0f32;
        for j in 0..4 {
            sum += table[indices[t * 4 + j] as usize * h + d].to_f32() * weights[t * 4 + j];
        }
        let rounded = bf16::from_f32(sum).to_f32();
        assert_eq!(*found, bf16::from_f32(input[i].to_f32() + rounded), "i={i}");
    }
    cuda::synchronize(s).unwrap();
    unsafe { (api().cs1_stream_destroy)(s) };
}
