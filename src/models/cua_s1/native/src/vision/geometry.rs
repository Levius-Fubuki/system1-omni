//! Geometry in the processor's 2×2 block-major patch order.
use anyhow::{Result, ensure};
pub(super) struct Geometry {
    pub indices: Vec<i32>,
    pub weights: Vec<f32>,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}
impl Geometry {
    pub fn new([t, h, w]: [usize; 3]) -> Result<Self> {
        ensure!(
            t == 1 && h > 0 && w > 0 && h % 2 == 0 && w % 2 == 0,
            "expected one image with an even, nonzero patch grid"
        );
        let n = h
            .checked_mul(w)
            .ok_or_else(|| anyhow::anyhow!("vision grid overflow"))?;
        // Processor bounds allow rounding up a 1,048,576-pixel source and narrow upscaled images.
        ensure!(
            n <= 4608 && h <= 512 && w <= 512,
            "vision grid exceeds processor bounds"
        );
        let mut g = Self {
            indices: Vec::with_capacity(n * 4),
            weights: Vec::with_capacity(n * 4),
            cos: Vec::with_capacity(n * 32),
            sin: Vec::with_capacity(n * 32),
        };
        for br in 0..h / 2 {
            for bc in 0..w / 2 {
                for ir in 0..2 {
                    for ic in 0..2 {
                        let row = br * 2 + ir;
                        let col = bc * 2 + ic;
                        let y = (row as f32 * 47.) / (h - 1) as f32;
                        let x = (col as f32 * 47.) / (w - 1) as f32;
                        let yl = y.floor() as usize;
                        let xl = x.floor() as usize;
                        let fy = y - yl as f32;
                        let fx = x - xl as f32;
                        for (yy, wy) in [(yl, 1. - fy), ((yl + 1).min(47), fy)] {
                            for (xx, wx) in [(xl, 1. - fx), ((xl + 1).min(47), fx)] {
                                g.indices.push((yy * 48 + xx) as i32);
                                g.weights.push(wy * wx);
                            }
                        }
                        for pos in [row, col] {
                            for i in 0..16 {
                                let angle = pos as f32 / 10000f32.powf(i as f32 / 16.);
                                g.cos.push(angle.cos());
                                g.sin.push(angle.sin());
                            }
                        }
                    }
                }
            }
        }
        Ok(g)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interpolation_corners_and_merge_order() {
        let g = Geometry::new([1, 2, 4]).unwrap();
        // Sequence: (0,0), (0,1), (1,0), (1,1), (0,2), (0,3), (1,2), (1,3).
        assert_eq!(&g.indices[..4], &[0, 1, 48, 49]);
        assert_eq!(&g.weights[..4], &[1., 0., 0., 0.]);
        assert_eq!(g.indices[2 * 4], 47 * 48);
        assert_eq!(g.indices[7 * 4], 2303);
        assert!((g.weights[4] - 1. / 3.).abs() < 2e-6);
        assert_eq!(g.cos[0], 1.);
        assert_eq!(g.sin[0], 0.);
        assert!((g.sin[32 + 16] - 1f32.sin()).abs() < 1e-6);
        assert!((g.sin[2 * 32] - 1f32.sin()).abs() < 1e-6);
        for w in g.weights.as_chunks::<4>().0 {
            assert!((w.iter().sum::<f32>() - 1.).abs() < 1e-6);
        }
    }
    #[test]
    fn geometry_rejects_video_odd_empty_and_oversize_grids() {
        for grid in [
            [2, 16, 16],
            [1, 0, 16],
            [1, 15, 16],
            [1, 128, 128],
            [1, usize::MAX, 2],
        ] {
            assert!(Geometry::new(grid).is_err(), "accepted {grid:?}");
        }
        assert!(Geometry::new([1, 64, 64]).is_ok());
        assert!(Geometry::new([1, 2, 224]).is_ok());
    }
}
