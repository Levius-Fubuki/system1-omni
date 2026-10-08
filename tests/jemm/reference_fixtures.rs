use omni_jemm_native::contract;
use serde_json::{Value, json};
fn chat(prompt: &str) -> String {
    format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        contract::SYSTEM
    )
}
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/text-contract.json")).unwrap()
}
#[test]
fn pinned_reference_prompts_match_exactly_in_fixture_order() {
    let f = fixture();
    assert_eq!(
        f["provenance"]["source_revision"],
        contract::SOURCE_REVISION
    );
    assert_eq!(f["provenance"]["base_revision"], contract::BASE_REVISION);
    for case in f["cases"].as_array().unwrap() {
        let questions = contract::compile(&serde_json::to_vec(&case["request"]).unwrap()).unwrap();
        let expected = case["questions"].as_array().unwrap();
        assert_eq!(questions.len(), expected.len());
        for (q, e) in questions.iter().zip(expected) {
            assert_eq!(q.id, e["question_id"].as_str().unwrap(), "{}", case["id"]);
            assert_eq!(q.kind, e["type"].as_str().unwrap());
            assert_eq!(
                q.keys,
                serde_json::from_value::<Vec<String>>(e["candidate_ids"].clone()).unwrap()
            );
            assert_eq!(
                chat(&q.prompt),
                e["prompt"].as_str().unwrap(),
                "{}",
                case["id"]
            );
        }
    }
}
#[test]
#[ignore = "requires JEMM_TOKENIZER_METADATA pointing to pinned local tokenizer metadata; no CUDA"]
fn pinned_tokenizer_ids_match_reference_for_every_fixture_question() {
    let dir = std::env::var_os("JEMM_TOKENIZER_METADATA").expect("set JEMM_TOKENIZER_METADATA");
    let tokenizer =
        tokenizers::Tokenizer::from_file(std::path::Path::new(&dir).join("tokenizer.json"))
            .unwrap();
    let f = fixture();
    let ids = contract::LABELS
        .chars()
        .map(|c| {
            let encoded = tokenizer.encode(c.to_string(), false).unwrap();
            assert_eq!(encoded.len(), 1);
            encoded.get_ids()[0]
        })
        .collect::<Vec<_>>();
    assert_eq!(json!(ids), f["label_token_ids"]);
    let prefix = format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n",
        contract::SYSTEM
    );
    let suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    for case in f["cases"].as_array().unwrap() {
        let questions = contract::compile(&serde_json::to_vec(&case["request"]).unwrap()).unwrap();
        for (q, e) in questions.iter().zip(case["questions"].as_array().unwrap()) {
            let encoded = tokenizer
                .encode(format!("{prefix}{}{suffix}", q.prompt), false)
                .unwrap();
            assert_eq!(
                json!(encoded.get_ids()),
                e["input_ids"],
                "{} {}",
                case["id"],
                q.id
            );
            assert_eq!(json!(encoded.len()), e["input_tokens"]);
        }
    }
}
