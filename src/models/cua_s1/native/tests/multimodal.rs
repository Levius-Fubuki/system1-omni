//! Real CUDA regression: explicit positions must not contaminate later text calls.
use std::path::PathBuf;

use half::bf16;
use omni_cua_s1_native::{inputs::MultimodalInput, model::Model};

#[test]
#[ignore = "needs CUA_S1_MODEL and CUA_S1_CUDA_LIB on a GPU"]
fn text_multimodal_text_keeps_text_positions_and_overwrites_image_rows() {
    let dir = PathBuf::from(std::env::var_os("CUA_S1_MODEL").expect("CUA_S1_MODEL"));
    let lib = PathBuf::from(std::env::var_os("CUA_S1_CUDA_LIB").expect("CUA_S1_CUDA_LIB"));
    let mut model = Model::load(&dir, &lib).unwrap();
    let ids = [32, 33, 34, 35];
    let positions = [0, 1, 2, 3];
    let baseline = model.forward(&ids).unwrap();
    let explicit = model
        .forward_multimodal(&MultimodalInput {
            token_ids: &ids,
            image_token_indices: &[],
            image_embeddings: &[],
            position_ids: [&positions; 3],
        })
        .unwrap();
    assert_eq!(baseline, explicit);

    let image_token = model.cfg.image_token_id.unwrap();
    let image_ids = [32, image_token, image_token, 35];
    let features = vec![bf16::ONE; 2 * model.cfg.hidden];
    let different = model
        .forward_multimodal(&MultimodalInput {
            token_ids: &image_ids,
            image_token_indices: &[1, 2],
            image_embeddings: &features,
            position_ids: [&[0, 1, 1, 2], &[0, 1, 2, 2], &[0, 2, 1, 2]],
        })
        .unwrap();
    assert!(different.iter().all(|x| x.is_finite()));
    assert_ne!(baseline, different);
    assert_eq!(baseline, model.forward(&ids).unwrap());

    // Reusing the same layout with changed features must not retain old rows.
    let features = vec![bf16::from_f32(-1.0); 2 * model.cfg.hidden];
    let changed = model
        .forward_multimodal(&MultimodalInput {
            token_ids: &image_ids,
            image_token_indices: &[1, 2],
            image_embeddings: &features,
            position_ids: [&[0, 1, 1, 2], &[0, 1, 2, 2], &[0, 2, 1, 2]],
        })
        .unwrap();
    assert_ne!(different, changed);
    assert_eq!(baseline, model.forward(&ids).unwrap());

    // Non-adjacent image spans exercise separate uploads and untouched text rows.
    let disjoint_ids = [image_token, 33, image_token, 35];
    let disjoint = MultimodalInput {
        token_ids: &disjoint_ids,
        image_token_indices: &[0, 2],
        image_embeddings: &features,
        position_ids: [&positions; 3],
    };
    let last = model.forward_multimodal(&disjoint).unwrap();
    assert_eq!(last, model.forward_multimodal(&disjoint).unwrap());

    // A rejected boundary must not interfere with restoring the next text call.
    assert!(
        model
            .forward_multimodal(&MultimodalInput {
                image_token_indices: &[2, 0],
                ..disjoint
            })
            .is_err()
    );
    assert_eq!(baseline, model.forward(&ids).unwrap());

    // Cross the 1024-row allocation boundary, then use a shorter custom layout
    // before restoring all text rows in the larger scratch allocation.
    let long_ids = vec![32; 1025];
    let long_text = model.forward(&long_ids).unwrap();
    model
        .forward_multimodal(&MultimodalInput {
            token_ids: &image_ids,
            image_token_indices: &[1, 2],
            image_embeddings: &features,
            position_ids: [&[0, 1, 1, 2], &[0, 1, 2, 2], &[0, 2, 1, 2]],
        })
        .unwrap();
    assert_eq!(long_text, model.forward(&long_ids).unwrap());
    assert_eq!(baseline, model.forward(&ids).unwrap());
}
