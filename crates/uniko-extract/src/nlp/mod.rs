//! Multi-task NLP pipeline backed by an ONNX model.
//!
//! [`NlpPipeline`] orchestrates tokenization, ONNX inference, and
//! post-processing to produce structured NLP annotations (NER, POS,
//! dependency parse, sentence classification) from raw text.
//!
//! The pipeline loads its model from HuggingFace via uni-xervo's
//! [`LocalOnnxProvider`](uni_db::LocalOnnxProvider) and embeds the
//! tokenizer and label maps at compile time.  If the ONNX runtime is
//! unavailable, [`NlpPipeline::try_new`] returns `None` and callers
//! fall back to rule-based extraction.

// Rust guideline compliant

pub mod assets;
pub mod decode;
pub mod types;

use std::sync::Arc;

use ndarray::ArrayD;
use uni_db::{OnnxRunner, TensorBatch, TensorValue};

use uniko_store::KnowledgeBase;
use uniko_store::UnikoError;
use uniko_store::schema::NLP_ALIAS;

use types::NlpResult;

/// ONNX-backed multi-task NLP pipeline.
///
/// Performs tokenization → ONNX inference → post-processing in a single
/// `analyze()` call.  The embedded tokenizer and label maps are parsed
/// once (global `OnceLock`); only the ONNX runner handle is per-instance.
#[derive(Clone)]
pub struct NlpPipeline {
    runner: Arc<dyn OnnxRunner>,
}

impl std::fmt::Debug for NlpPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NlpPipeline")
            .field("runner", &"<OnnxRunner>")
            .finish()
    }
}

impl NlpPipeline {
    /// Try to create a pipeline from the KB's xervo runtime.
    ///
    /// Returns `None` if the ONNX provider is unavailable (no runtime,
    /// alias not registered, or model download fails).
    pub async fn try_new(kb: &KnowledgeBase) -> Option<Self> {
        let runner = match kb.db().xervo().onnx_runner(NLP_ALIAS).await {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(error = %e, alias = NLP_ALIAS, "ONNX runner unavailable");
                return None;
            }
        };
        Some(Self { runner })
    }

    /// Run full NLP analysis on a text string.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Pipeline`] if tokenization or ONNX
    /// inference fails.
    pub async fn analyze(&self, text: &str) -> Result<NlpResult, UnikoError> {
        let tokenizer = assets::tokenizer();
        let labels = assets::label_maps();

        // 1. Tokenize.
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| UnikoError::Pipeline(format!("tokenize: {e}")))?;

        let ids = encoding.get_ids();
        let attention = encoding.get_attention_mask();
        let seq_len = ids.len();

        // Build word-level alignment.
        // token_to_word returns Option<(sequence_id, word_id)>.
        // We extract just the word_id, discarding the sequence_id.
        let word_ids: Vec<Option<u32>> = (0..seq_len)
            .map(|i| encoding.token_to_word(i).map(|(_, word_id)| word_id))
            .collect();

        let tokens: Vec<String> = encoding.get_tokens().to_vec();

        // 2. Build input TensorBatch.
        let input_ids: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = attention.iter().map(|&x| x as i64).collect();

        let mut inputs = TensorBatch::default();
        inputs.insert(
            "input_ids",
            TensorValue::I64(
                ArrayD::from_shape_vec(vec![1, seq_len], input_ids)
                    .map_err(|e| UnikoError::Pipeline(format!("shape: {e}")))?,
            ),
        );
        inputs.insert(
            "attention_mask",
            TensorValue::I64(
                ArrayD::from_shape_vec(vec![1, seq_len], attention_mask)
                    .map_err(|e| UnikoError::Pipeline(format!("shape: {e}")))?,
            ),
        );

        // 3. Run ONNX inference.
        let outputs = self
            .runner
            .run(&inputs)
            .await
            .map_err(|e| UnikoError::Pipeline(format!("onnx: {e}")))?;

        // 4. Extract output tensors (squeeze batch dimension).
        let ner_logits = extract_f32_2d(&outputs, "ner_logits")?;
        let pos_logits = extract_f32_2d(&outputs, "pos_logits")?;
        let dep_logits = extract_f32_2d(&outputs, "dep_logits")?;
        let cls_logits = extract_f32_1d(&outputs, "cls_logits")?;

        // 5. Argmax per subword token.
        let ner_subword = decode::argmax_rows(&ner_logits.view());
        let pos_subword = decode::argmax_rows(&pos_logits.view());
        let dep_subword = decode::argmax_rows(&dep_logits.view());
        let (cls_index, cls_confidence) = decode::argmax_with_confidence(&cls_logits.view());

        // 6. Align to words (first-subword-wins).
        let ner_indices = decode::align_to_words(&ner_subword, &word_ids);
        let pos_indices = decode::align_to_words(&pos_subword, &word_ids);
        let dep_indices = decode::align_to_words(&dep_subword, &word_ids);
        let words = decode::extract_words(&tokens, &word_ids);

        // 7. Decode structured outputs.
        let entities = decode::merge_bio_spans(&words, &ner_indices, &labels.ner_labels);
        let dep_arcs = decode::decode_dep_tree(
            &dep_indices,
            &pos_indices,
            &labels.dep_labels,
            &labels.pos_labels,
        );
        let sentence_class = decode::decode_cls(cls_index, &labels.cls_labels);

        Ok(NlpResult {
            words,
            ner_indices,
            pos_indices,
            dep_indices,
            cls_index,
            cls_confidence,
            entities,
            dep_arcs,
            sentence_class,
        })
    }
}

/// Extract a 2D f32 tensor from outputs, squeezing the batch dimension.
///
/// Expects shape `[1, seq_len, num_classes]` → returns `[seq_len, num_classes]`.
fn extract_f32_2d(outputs: &TensorBatch, name: &str) -> Result<ndarray::Array2<f32>, UnikoError> {
    let tensor = outputs
        .get(name)
        .ok_or_else(|| UnikoError::Pipeline(format!("missing output tensor: {name}")))?;

    match tensor {
        TensorValue::F32(arr) => {
            let shape = arr.shape();
            if shape.len() == 3 && shape[0] == 1 {
                // Squeeze batch dim: [1, seq, classes] → [seq, classes]
                let squeezed = arr
                    .clone()
                    .into_shape_with_order(ndarray::IxDyn(&[shape[1], shape[2]]))
                    .map_err(|e| UnikoError::Pipeline(format!("reshape {name}: {e}")))?;
                squeezed
                    .into_dimensionality()
                    .map_err(|e| UnikoError::Pipeline(format!("dim {name}: {e}")))
            } else if shape.len() == 2 {
                arr.clone()
                    .into_dimensionality()
                    .map_err(|e| UnikoError::Pipeline(format!("dim {name}: {e}")))
            } else {
                Err(UnikoError::Pipeline(format!(
                    "{name}: unexpected shape {shape:?}"
                )))
            }
        }
        _ => Err(UnikoError::Pipeline(format!("{name}: expected F32 tensor"))),
    }
}

/// Extract a 1D f32 tensor from outputs, squeezing the batch dimension.
///
/// Expects shape `[1, num_classes]` → returns `[num_classes]`.
fn extract_f32_1d(outputs: &TensorBatch, name: &str) -> Result<ndarray::Array1<f32>, UnikoError> {
    let tensor = outputs
        .get(name)
        .ok_or_else(|| UnikoError::Pipeline(format!("missing output tensor: {name}")))?;

    match tensor {
        TensorValue::F32(arr) => {
            let shape = arr.shape();
            if shape.len() == 2 && shape[0] == 1 {
                // Squeeze: [1, classes] → [classes]
                let squeezed = arr
                    .clone()
                    .into_shape_with_order(ndarray::IxDyn(&[shape[1]]))
                    .map_err(|e| UnikoError::Pipeline(format!("reshape {name}: {e}")))?;
                squeezed
                    .into_dimensionality()
                    .map_err(|e| UnikoError::Pipeline(format!("dim {name}: {e}")))
            } else if shape.len() == 1 {
                arr.clone()
                    .into_dimensionality()
                    .map_err(|e| UnikoError::Pipeline(format!("dim {name}: {e}")))
            } else {
                Err(UnikoError::Pipeline(format!(
                    "{name}: unexpected shape {shape:?}"
                )))
            }
        }
        _ => Err(UnikoError::Pipeline(format!("{name}: expected F32 tensor"))),
    }
}
