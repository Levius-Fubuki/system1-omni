//! The model side: prompt tokenization, and one prefill-only forward pass per
//! question through the native Qwen3.5 model, scored with the 26 letter rows of
//! the output projection.

use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value as Json;
use sha2::Digest;
use tokenizers::Tokenizer;

use crate::contract::{self, LETTERS, Question};
use crate::model::Model;
pub use crate::model::{Mode, Options};

/// What `cua_s1_export.json` records about a merged checkpoint.
#[derive(Debug, Clone)]
pub struct Provenance {
    pub base_revision: String,
    pub adapter_revision: String,
}

/// Read and check the export record written next to the merged weights.
pub fn provenance(dir: &Path) -> Result<Provenance> {
    let path = dir.join("cua_s1_export.json");
    let info: Json = serde_json::from_str(&std::fs::read_to_string(&path).with_context(|| {
        format!(
            "{} is missing; export the merged checkpoint first",
            path.display()
        )
    })?)?;
    ensure!(
        info["format"] == "cua-s1-text-merged/1",
        "{}: unknown format {}",
        path.display(),
        info["format"]
    );
    ensure!(
        info["base"]["repo"] == contract::BASE_REPO,
        "base is not {}",
        contract::BASE_REPO
    );
    ensure!(
        info["adapter"]["repo"] == contract::ADAPTER_REPO && info["adapter"]["subfolder"] == "text",
        "adapter is not the `text` adapter of {}",
        contract::ADAPTER_REPO
    );
    // Transformers 5.17 tokenizes Qwen3.5 with the rule saved in this file, not the
    // one in the base repo's tokenizer.json, so the file must be the exported one.
    let want = info["tokenizer"]["sha256"]
        .as_str()
        .context("cua_s1_export.json does not record the tokenizer's sha256")?;
    let got = format!(
        "{:x}",
        sha2::Sha256::digest(std::fs::read(dir.join("tokenizer.json"))?)
    );
    ensure!(
        got == want,
        "tokenizer.json (sha256 {got}) is not the one exported with the weights ({want})"
    );
    let rev = |v: &Json| v.as_str().map(str::to_string).context("revision missing");
    Ok(Provenance {
        base_revision: rev(&info["base"]["revision"])?,
        adapter_revision: rev(&info["adapter"]["revision"])?,
    })
}

/// Chat text and token ids for a question; needs only `tokenizer.json`.
pub struct Prompter {
    tokenizer: Tokenizer,
    pub letter_ids: Vec<u32>,
}

impl Prompter {
    /// Checks `tokenizer.json` against `cua_s1_export.json` first (see `provenance`).
    pub fn load(dir: &Path) -> Result<Self> {
        provenance(dir)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer.json: {e}"))?;
        let mut letter_ids = Vec::with_capacity(LETTERS.len());
        for letter in LETTERS.chars() {
            let enc = tokenizer
                .encode(letter.to_string(), false)
                .map_err(|e| anyhow::anyhow!(e))?;
            ensure!(
                enc.get_ids().len() == 1,
                "letter {letter} is not a single token"
            );
            letter_ids.push(enc.get_ids()[0]);
        }
        Ok(Self {
            tokenizer,
            letter_ids,
        })
    }

    pub fn encode(&self, state: &str, question: &Question) -> Result<Vec<u32>> {
        let text = contract::chat_text(state, question);
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(enc.get_ids().to_vec())
    }
}

/// Where the output projection can live in a Qwen3.5 text checkpoint; with tied
/// weights (as in Qwen3.5-4B) only the embedding is stored.
const HEAD_NAMES: &[&str] = &[
    "lm_head.weight",
    "language_model.lm_head.weight",
    "model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
];

/// The letter rows of the output projection, as float32, read straight from the
/// safetensors files.
fn letter_rows(dir: &Path, letter_ids: &[u32]) -> Result<(Vec<f32>, usize)> {
    let index_path = dir.join("model.safetensors.index.json");
    let (file, name) = if index_path.exists() {
        let index: Json = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
        let map = &index["weight_map"];
        let name = HEAD_NAMES
            .iter()
            .find(|n| map.get(**n).is_some())
            .with_context(|| {
                format!(
                    "no output projection or embedding in {}",
                    index_path.display()
                )
            })?;
        let file = map[*name]
            .as_str()
            .context("weight_map entry is not a file name")?;
        (dir.join(file), Some(name.to_string()))
    } else {
        (dir.join("model.safetensors"), None)
    };
    let file = std::fs::File::open(&file)?;
    // SAFETY: the checkpoint is not modified while the worker runs.
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = safetensors::SafeTensors::deserialize(&mmap)?;
    let name = match name {
        Some(n) => n,
        None => {
            let names = st.names();
            HEAD_NAMES
                .iter()
                .find(|n| names.iter().any(|m| m == *n))
                .context("no output projection or embedding in model.safetensors")?
                .to_string()
        }
    };
    let view = st.tensor(&name)?;
    ensure!(
        view.dtype() == safetensors::Dtype::BF16 && view.shape().len() == 2,
        "{name}: expected a 2-D bfloat16 tensor, got {:?} {:?}",
        view.dtype(),
        view.shape()
    );
    let (vocab, hidden) = (view.shape()[0], view.shape()[1]);
    let data = view.data();
    let mut rows = Vec::with_capacity(letter_ids.len() * hidden);
    for &id in letter_ids {
        let id = id as usize;
        ensure!(id < vocab, "letter id {id} outside the vocabulary");
        let row = &data[id * hidden * 2..(id + 1) * hidden * 2];
        rows.extend(
            row.chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32()),
        );
    }
    Ok((rows, hidden))
}

pub struct Engine {
    pub prompter: Prompter,
    model: Arc<Mutex<Model>>,
    letters: Vec<f32>,
    hidden: usize,
    pub load_seconds: f64,
    pub device: String,
}

impl Engine {
    /// Load the CUDA library and the model, and prepare CUDA graphs (see `Options`).
    pub async fn load(dir: &Path, opts: &Options) -> Result<Self> {
        let started = Instant::now();
        let prompter = Prompter::load(dir)?;
        let (letters, hidden) = letter_rows(dir, &prompter.letter_ids)?;
        let dir = dir.to_path_buf();
        let opts = opts.clone();
        let model = tokio::task::spawn_blocking(move || Model::load(&dir, &opts)).await??;
        ensure!(
            model.cfg.hidden == hidden,
            "hidden size {} does not match the head ({hidden})",
            model.cfg.hidden
        );
        Ok(Self {
            prompter,
            model: Arc::new(Mutex::new(model)),
            letters,
            hidden,
            load_seconds: started.elapsed().as_secs_f64(),
            device: "cuda".to_string(),
        })
    }

    fn lock(&self) -> Result<MutexGuard<'_, Model>> {
        self.model
            .lock()
            .map_err(|_| anyhow::anyhow!("model lock poisoned"))
    }

    /// The longest prompt that runs as a CUDA graph (0: none).
    pub fn graph_max_tokens(&self) -> usize {
        self.lock().map(|m| m.graph_max_tokens()).unwrap_or(0)
    }

    /// The final-norm hidden state at the last position.
    async fn last_hidden(&self, ids: Vec<u32>, mode: Mode) -> Result<Vec<f32>> {
        let model = self.model.clone();
        tokio::task::spawn_blocking(move || {
            let mut model = model
                .lock()
                .map_err(|_| anyhow::anyhow!("model lock poisoned"))?;
            model.forward(&ids, mode)
        })
        .await?
    }

    /// One forward pass on the calling thread, for timing.
    pub fn last_hidden_blocking(&self, ids: &[u32]) -> Result<Vec<f32>> {
        self.lock()?.forward(ids, Mode::Auto)
    }

    /// Option probabilities for one prompt: the final-norm hidden state at the last
    /// position times the letter rows, in float32 with float64 accumulation, then a
    /// softmax over the first `n_options` letters.
    pub async fn score(&self, ids: Vec<u32>, n_options: usize) -> Result<Vec<f32>> {
        self.score_mode(ids, n_options, Mode::Auto).await
    }

    /// As `score`, run eagerly or from a graph.
    pub async fn score_mode(
        &self,
        ids: Vec<u32>,
        n_options: usize,
        mode: Mode,
    ) -> Result<Vec<f32>> {
        let last = self.last_hidden(ids, mode).await?;
        if last.len() != self.hidden {
            bail!(
                "hidden size {} does not match the head ({})",
                last.len(),
                self.hidden
            );
        }
        let logits: Vec<f32> = self
            .letters
            .chunks_exact(self.hidden)
            .take(n_options)
            .map(|w| {
                w.iter()
                    .zip(&last)
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum::<f64>() as f32
            })
            .collect();
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exps: Vec<f64> = logits.iter().map(|&l| (l as f64 - max).exp()).collect();
        let total: f64 = exps.iter().sum();
        Ok(exps.iter().map(|e| (e / total) as f32).collect())
    }
}
