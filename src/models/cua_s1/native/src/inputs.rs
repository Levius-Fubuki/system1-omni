//! Batch-one, unpadded inputs at the adapted-vision / language-model boundary.

use anyhow::{Result, ensure};
use half::bf16;

/// Image rows are already adapted to the language hidden size, in placeholder
/// order. Positions are the T/H/W slices of an int64 `[3, 1, sequence]` tensor.
/// The caller owns preprocessing, vision execution and the position calculation.
pub struct MultimodalInput<'a> {
    pub token_ids: &'a [u32],
    pub image_token_indices: &'a [usize],
    pub image_embeddings: &'a [bf16],
    pub position_ids: [&'a [i64]; 3],
}

impl MultimodalInput<'_> {
    /// Check the entire boundary before allocating buffers or launching CUDA.
    pub fn validate(
        &self,
        hidden: usize,
        vocab: usize,
        image_token: u32,
        max_position: usize,
    ) -> Result<()> {
        let t = self.token_ids.len();
        ensure!(t > 0 && t <= max_position, "empty or oversized prompt");
        ensure!(
            self.token_ids.iter().all(|&id| (id as usize) < vocab),
            "token id outside the vocabulary"
        );
        let expected: Vec<usize> = self
            .token_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == image_token).then_some(i))
            .collect();
        ensure!(
            self.image_token_indices == expected,
            "image indices must exactly match the ordered placeholders"
        );
        ensure!(
            Some(self.image_embeddings.len()) == expected.len().checked_mul(hidden),
            "image embedding shape mismatch"
        );
        ensure!(
            self.image_embeddings.iter().all(|x| x.is_finite()),
            "non-finite image embedding"
        );
        ensure!(
            self.position_ids.iter().all(|axis| axis.len() == t),
            "position_ids must have shape [3, 1, sequence]"
        );
        ensure!(
            self.position_ids
                .iter()
                .flat_map(|axis| axis.iter())
                .all(|&p| p >= 0 && (p as u64) < max_position as u64),
            "position outside the configured range"
        );
        Ok(())
    }
}

/// Qwen3.5's interleaved recomposition: overwrite H at 1::3 and W at 2::3 up
/// to section[axis] * 3, retaining T elsewhere. The second rotary half repeats
/// these frequencies, which the existing attention-prep kernel handles.
/// Float32 inverse frequencies/products and host float64 trig preserve the
/// original native text table rounding when all three axes are equal.
pub(crate) fn rotary_tables(
    positions: [&[i64]; 3],
    half: usize,
    theta: f64,
    sections: [usize; 3],
) -> (Vec<u8>, Vec<u8>) {
    let inv: Vec<f32> = (0..half)
        .map(|i| 1f32 / (theta as f32).powf((2 * i) as f32 / (2 * half) as f32))
        .collect();
    let mut cos = Vec::with_capacity(positions[0].len() * half * 2);
    let mut sin = Vec::with_capacity(cos.capacity());
    for (t, _) in positions[0].iter().enumerate() {
        for (i, &f) in inv.iter().enumerate() {
            let axis = if i % 3 == 1 && i < sections[1] * 3 {
                1
            } else if i % 3 == 2 && i < sections[2] * 3 {
                2
            } else {
                0
            };
            let angle = (f * positions[axis][t] as f32) as f64;
            cos.extend(bf16::from_f32(angle.cos() as f32).to_le_bytes());
            sin.extend(bf16::from_f32(angle.sin() as f32).to_le_bytes());
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    #[test]
    fn valid_input_and_three_distinct_axes() {
        let input = MultimodalInput {
            token_ids: &[1, 99, 99, 2],
            image_token_indices: &[1, 2],
            image_embeddings: &[bf16::ONE; 8],
            position_ids: [&[0, 1, 1, 3], &[0, 1, 2, 3], &[0, 2, 1, 3]],
        };
        input.validate(4, 100, 99, 100).unwrap();
    }

    #[test]
    fn rejects_bad_placeholder_inventory_and_feature_rows() {
        for indices in [vec![2, 1], vec![1, 1], vec![1], vec![0, 1], vec![1, 4]] {
            let input = MultimodalInput {
                token_ids: &[1, 99, 99, 2],
                image_token_indices: &indices,
                image_embeddings: &[bf16::ONE; 8],
                position_ids: [&[0, 1, 1, 3]; 3],
            };
            assert!(input.validate(4, 100, 99, 100).is_err(), "{indices:?}");
        }
        for features in [vec![bf16::ONE; 7], vec![bf16::ONE; 9], vec![bf16::NAN; 8]] {
            let input = MultimodalInput {
                token_ids: &[1, 99, 99, 2],
                image_token_indices: &[1, 2],
                image_embeddings: &features,
                position_ids: [&[0, 1, 1, 3]; 3],
            };
            assert!(input.validate(4, 100, 99, 100).is_err());
        }
    }

    #[test]
    fn rejects_bad_tokens_positions_and_empty_sequence() {
        let features = [bf16::ONE; 4];
        for (ids, positions) in [
            (vec![100, 99], vec![0, 1]),
            (vec![1, 99], vec![0]),
            (vec![1, 99], vec![0, -1]),
            (vec![1, 99], vec![0, 100]),
            (vec![], vec![]),
        ] {
            let input = MultimodalInput {
                token_ids: &ids,
                image_token_indices: &[1],
                image_embeddings: &features,
                position_ids: [&positions; 3],
            };
            assert!(input.validate(4, 100, 99, 100).is_err());
        }
    }

    #[test]
    fn image_free_explicit_positions_are_valid() {
        MultimodalInput {
            token_ids: &[1, 2],
            image_token_indices: &[],
            image_embeddings: &[],
            position_ids: [&[7, 8]; 3],
        }
        .validate(4, 100, 99, 100)
        .unwrap();
    }

    #[test]
    fn rotary_interleaves_height_width_and_leaves_temporal_tail() {
        // theta=1 makes every inverse frequency 1. Axis values differ so a plain
        // text table or a contiguous-section implementation fails this check.
        for (sections, tail) in [([12, 10, 10], [0, 0]), ([11, 11, 10], [0, 1])] {
            let (cos, sin) = rotary_tables([&[0], &[1], &[2]], 32, 1.0, sections);
            // Ten T/H/W triples, then T/T for the synthetic layout or T/H for
            // the real checkpoint. In particular, frequency 31 must use H.
            let axes = [
                0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
                1, 2, tail[0], tail[1],
            ];
            for (i, axis) in axes.into_iter().enumerate() {
                let angle = axis as f32;
                assert_eq!(
                    &cos[i * 2..i * 2 + 2],
                    &bf16::from_f32(angle.cos()).to_le_bytes()
                );
                assert_eq!(
                    &sin[i * 2..i * 2 + 2],
                    &bf16::from_f32(angle.sin()).to_le_bytes()
                );
            }
        }
    }

    #[test]
    fn equal_axes_reproduce_the_existing_text_tables() {
        let pos: Vec<i64> = (0..257).collect();
        let (cos, sin) = rotary_tables([&pos; 3], 32, 10_000_000.0, [11, 11, 10]);
        for (t, &p) in pos.iter().enumerate() {
            for i in 0..32 {
                let inv = 1f32 / 10_000_000f32.powf((2 * i) as f32 / 64.0);
                let angle = (inv * p as f32) as f64;
                let offset = (t * 32 + i) * 2;
                assert_eq!(
                    &cos[offset..offset + 2],
                    &bf16::from_f32(angle.cos() as f32).to_le_bytes()
                );
                assert_eq!(
                    &sin[offset..offset + 2],
                    &bf16::from_f32(angle.sin() as f32).to_le_bytes()
                );
            }
        }
    }
}
