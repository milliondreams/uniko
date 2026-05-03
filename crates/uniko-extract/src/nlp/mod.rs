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
use uni_db::{RawTensorModel, TensorBatch, TensorValue};

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
    runner: Arc<dyn RawTensorModel>,
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
        let runner = match kb.db().xervo().raw_tensor_model(NLP_ALIAS).await {
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
        // SRL head is unused; predicate_idx=0 is a no-op because
        // pred_embedding is zero-initialized in the trained model.
        inputs.insert(
            "predicate_idx",
            TensorValue::I64(
                ArrayD::from_shape_vec(vec![1], vec![0i64])
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
        let arc_scores = extract_f32_2d_squeeze_batch(&outputs, "arc_scores")?;
        let label_scores = extract_f32_3d_squeeze_batch(&outputs, "label_scores")?;
        let cls_logits = extract_f32_1d(&outputs, "cls_logits")?;

        // 5. Argmax per subword token (NER/POS/CLS).
        let ner_subword = decode::argmax_rows(&ner_logits.view());
        let pos_subword = decode::argmax_rows(&pos_logits.view());
        let (cls_index, cls_confidence) = decode::argmax_with_confidence(&cls_logits.view());
        let cls_probs = decode::softmax_1d(&cls_logits.view());

        // 6. Align to words (first-subword-wins).
        let ner_indices = decode::align_to_words(&ner_subword, &word_ids);
        let pos_indices = decode::align_to_words(&pos_subword, &word_ids);
        let words = decode::extract_words(&tokens, &word_ids);

        // 7. Decode structured outputs.
        let entities = decode::merge_bio_spans(&words, &ner_indices, &labels.ner_labels);
        let dep_arcs = decode::decode_dep_arcs_biaffine(
            &arc_scores.view(),
            &label_scores.view(),
            &word_ids,
            &labels.dep_rel_labels,
        );
        let sentence_class = decode::decode_cls(cls_index, &labels.cls_labels);

        Ok(NlpResult {
            words,
            ner_indices,
            pos_indices,
            cls_index,
            cls_confidence,
            cls_probs,
            entities,
            dep_arcs,
            sentence_class,
        })
    }

    /// Analyze text by splitting into sentences first.
    ///
    /// Returns one [`NlpResult`] per sentence in a single batched ONNX
    /// forward pass.  Each sentence still gets its own CLS, DEP tree and
    /// NER spans — avoiding the problem where a greeting prefix causes
    /// the whole message to be classified as non-informative.
    ///
    /// Sentences shorter than 4 words are filtered out by
    /// [`split_sentences`] to skip noise like "Wow!" or "Yeah!".
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Pipeline`] if any sentence fails
    /// tokenization or the ONNX inference fails.
    pub async fn analyze_sentences(&self, text: &str) -> Result<Vec<NlpResult>, UnikoError> {
        let sentences = split_sentences(text);
        if sentences.is_empty() {
            return Ok(Vec::new());
        }
        // Single sentence — skip the padding overhead.
        if sentences.len() == 1 {
            let result = self.analyze(&sentences[0]).await?;
            return Ok(vec![result]);
        }
        self.analyze_batch(&sentences).await
    }

    /// Analyze multiple pre-split sentences in a single ONNX forward
    /// pass.  All sentences share one encoder invocation; per-row
    /// decoding splits the output back into individual `NlpResult`s.
    async fn analyze_batch(&self, sentences: &[String]) -> Result<Vec<NlpResult>, UnikoError> {
        let tokenizer = assets::tokenizer();
        let labels = assets::label_maps();

        // 1. Tokenize each sentence and collect per-row metadata.
        struct RowMeta {
            ids: Vec<i64>,
            attention: Vec<i64>,
            word_ids: Vec<Option<u32>>,
            tokens: Vec<String>,
        }
        let mut rows = Vec::with_capacity(sentences.len());
        let mut max_seq_len: usize = 0;
        for sentence in sentences {
            let encoding = tokenizer
                .encode(sentence.as_str(), true)
                .map_err(|e| UnikoError::Pipeline(format!("tokenize: {e}")))?;
            let n = encoding.get_ids().len();
            max_seq_len = max_seq_len.max(n);
            rows.push(RowMeta {
                ids: encoding.get_ids().iter().map(|&x| x as i64).collect(),
                attention: encoding
                    .get_attention_mask()
                    .iter()
                    .map(|&x| x as i64)
                    .collect(),
                word_ids: (0..n)
                    .map(|i| encoding.token_to_word(i).map(|(_, word_id)| word_id))
                    .collect(),
                tokens: encoding.get_tokens().to_vec(),
            });
        }

        // 2. Build padded [B, max_seq_len] tensors.
        //    pad_id from the tokenizer if available, else 0.
        let pad_id: i64 = tokenizer
            .token_to_id("[PAD]")
            .map(|v| v as i64)
            .unwrap_or(0);
        let batch = rows.len();
        let mut input_ids = Vec::with_capacity(batch * max_seq_len);
        let mut attention = Vec::with_capacity(batch * max_seq_len);
        for r in &rows {
            input_ids.extend_from_slice(&r.ids);
            input_ids.resize(input_ids.len() + (max_seq_len - r.ids.len()), pad_id);
            attention.extend_from_slice(&r.attention);
            attention.resize(attention.len() + (max_seq_len - r.attention.len()), 0);
        }

        let mut inputs = TensorBatch::default();
        inputs.insert(
            "input_ids",
            TensorValue::I64(
                ArrayD::from_shape_vec(vec![batch, max_seq_len], input_ids)
                    .map_err(|e| UnikoError::Pipeline(format!("shape: {e}")))?,
            ),
        );
        inputs.insert(
            "attention_mask",
            TensorValue::I64(
                ArrayD::from_shape_vec(vec![batch, max_seq_len], attention)
                    .map_err(|e| UnikoError::Pipeline(format!("shape: {e}")))?,
            ),
        );
        // SRL head unused; pass zero predicate per row.
        inputs.insert(
            "predicate_idx",
            TensorValue::I64(
                ArrayD::from_shape_vec(vec![batch], vec![0i64; batch])
                    .map_err(|e| UnikoError::Pipeline(format!("shape: {e}")))?,
            ),
        );

        // 3. Single ONNX forward pass over the whole batch.
        let outputs = self
            .runner
            .run(&inputs)
            .await
            .map_err(|e| UnikoError::Pipeline(format!("onnx: {e}")))?;

        // 4. Outputs: ner/pos = [B, S, C]; arc_scores = [B, S, S];
        //    label_scores = [B, S, S, num_rels]; cls = [B, C].
        let ner = extract_f32_3d(&outputs, "ner_logits", batch, max_seq_len)?;
        let pos = extract_f32_3d(&outputs, "pos_logits", batch, max_seq_len)?;
        let arc = extract_f32_3d_square(&outputs, "arc_scores", batch, max_seq_len)?;
        let label = extract_f32_4d(&outputs, "label_scores", batch, max_seq_len)?;
        let cls = extract_f32_2d_batch(&outputs, "cls_logits", batch)?;

        // 5. Per-row decode using each sentence's true seq_len.
        let mut results = Vec::with_capacity(batch);
        for (i, r) in rows.into_iter().enumerate() {
            let seq = r.ids.len();
            let ner_slice = ner.slice(ndarray::s![i, ..seq, ..]);
            let pos_slice = pos.slice(ndarray::s![i, ..seq, ..]);
            let arc_slice = arc.slice(ndarray::s![i, ..seq, ..seq]);
            let label_slice = label.slice(ndarray::s![i, ..seq, ..seq, ..]);
            let cls_slice = cls.slice(ndarray::s![i, ..]);

            let ner_subword = decode::argmax_rows(&ner_slice);
            let pos_subword = decode::argmax_rows(&pos_slice);
            let (cls_index, cls_confidence) = decode::argmax_with_confidence(&cls_slice);
            let cls_probs = decode::softmax_1d(&cls_slice);

            let ner_indices = decode::align_to_words(&ner_subword, &r.word_ids);
            let pos_indices = decode::align_to_words(&pos_subword, &r.word_ids);
            let words = decode::extract_words(&r.tokens, &r.word_ids);

            let entities = decode::merge_bio_spans(&words, &ner_indices, &labels.ner_labels);
            let dep_arcs = decode::decode_dep_arcs_biaffine(
                &arc_slice,
                &label_slice,
                &r.word_ids,
                &labels.dep_rel_labels,
            );
            let sentence_class = decode::decode_cls(cls_index, &labels.cls_labels);

            results.push(NlpResult {
                words,
                ner_indices,
                pos_indices,
                cls_index,
                cls_confidence,
                cls_probs,
                entities,
                dep_arcs,
                sentence_class,
            });
        }

        Ok(results)
    }
}

/// Split text into sentences for per-sentence NLP analysis.
///
/// Splits on sentence-ending punctuation (`.` `!` `?`) followed by
/// whitespace or end-of-string. Filters fragments under 4 words.
/// If no sentences survive filtering, returns the original text
/// (provided it has >= 4 words).
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0;

    for (byte_pos, ch) in text.char_indices() {
        if ch != '.' && ch != '!' && ch != '?' {
            continue;
        }
        let end = byte_pos + ch.len_utf8();
        // Must be followed by whitespace or end-of-string.
        let at_boundary = end >= text.len()
            || text[end..]
                .chars()
                .next()
                .is_some_and(|c| c.is_whitespace());
        if !at_boundary {
            continue;
        }

        let sentence = text[start..end].trim();
        if sentence.split_whitespace().count() >= 4 {
            sentences.push(sentence.to_string());
        }
        // Advance past punctuation + whitespace.
        start = end;
        while start < text.len() {
            match text[start..].chars().next() {
                Some(ws) if ws.is_whitespace() => start += ws.len_utf8(),
                _ => break,
            }
        }
    }

    // Trailing text without terminal punctuation.
    if start < text.len() {
        let remainder = text[start..].trim();
        if remainder.split_whitespace().count() >= 4 {
            sentences.push(remainder.to_string());
        }
    }

    // Fallback: if nothing survived, use the original text.
    if sentences.is_empty() && text.split_whitespace().count() >= 4 {
        sentences.push(text.trim().to_string());
    }

    sentences
}

/// Extract a 3D f32 tensor from outputs as `[batch, max_seq, num_classes]`.
///
/// Used by [`NlpPipeline::analyze_batch`] for batched per-row decoding.
/// The caller slices the returned array per-row using each row's true
/// (unpadded) sequence length.
fn extract_f32_3d(
    outputs: &TensorBatch,
    name: &str,
    expected_batch: usize,
    expected_seq: usize,
) -> Result<ndarray::Array3<f32>, UnikoError> {
    let tensor = outputs
        .get(name)
        .ok_or_else(|| UnikoError::Pipeline(format!("missing output tensor: {name}")))?;

    match tensor {
        TensorValue::F32(arr) => {
            let shape = arr.shape();
            if shape.len() != 3 || shape[0] != expected_batch || shape[1] != expected_seq {
                return Err(UnikoError::Pipeline(format!(
                    "{name}: expected [{expected_batch}, {expected_seq}, *], got {shape:?}"
                )));
            }
            arr.clone()
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| UnikoError::Pipeline(format!("dim {name}: {e}")))
        }
        _ => Err(UnikoError::Pipeline(format!("{name}: expected F32 tensor"))),
    }
}

/// Extract a 2D f32 tensor from outputs as `[batch, num_classes]`.
///
/// Used for the `cls_logits` head when running a multi-row batch.
fn extract_f32_2d_batch(
    outputs: &TensorBatch,
    name: &str,
    expected_batch: usize,
) -> Result<ndarray::Array2<f32>, UnikoError> {
    let tensor = outputs
        .get(name)
        .ok_or_else(|| UnikoError::Pipeline(format!("missing output tensor: {name}")))?;

    match tensor {
        TensorValue::F32(arr) => {
            let shape = arr.shape();
            if shape.len() != 2 || shape[0] != expected_batch {
                return Err(UnikoError::Pipeline(format!(
                    "{name}: expected [{expected_batch}, *], got {shape:?}"
                )));
            }
            arr.clone()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| UnikoError::Pipeline(format!("dim {name}: {e}")))
        }
        _ => Err(UnikoError::Pipeline(format!("{name}: expected F32 tensor"))),
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

/// Extract `arc_scores[1, seq, seq]` and squeeze to `[seq, seq]`.
fn extract_f32_2d_squeeze_batch(
    outputs: &TensorBatch,
    name: &str,
) -> Result<ndarray::Array2<f32>, UnikoError> {
    let tensor = outputs
        .get(name)
        .ok_or_else(|| UnikoError::Pipeline(format!("missing output tensor: {name}")))?;
    match tensor {
        TensorValue::F32(arr) => {
            let shape = arr.shape();
            if shape.len() != 3 || shape[0] != 1 {
                return Err(UnikoError::Pipeline(format!(
                    "{name}: expected [1, seq, seq], got {shape:?}"
                )));
            }
            let squeezed = arr
                .clone()
                .into_shape_with_order(ndarray::IxDyn(&[shape[1], shape[2]]))
                .map_err(|e| UnikoError::Pipeline(format!("reshape {name}: {e}")))?;
            squeezed
                .into_dimensionality()
                .map_err(|e| UnikoError::Pipeline(format!("dim {name}: {e}")))
        }
        _ => Err(UnikoError::Pipeline(format!("{name}: expected F32 tensor"))),
    }
}

/// Extract `label_scores[1, seq, seq, num_rels]` and squeeze to
/// `[seq, seq, num_rels]`.
fn extract_f32_3d_squeeze_batch(
    outputs: &TensorBatch,
    name: &str,
) -> Result<ndarray::Array3<f32>, UnikoError> {
    let tensor = outputs
        .get(name)
        .ok_or_else(|| UnikoError::Pipeline(format!("missing output tensor: {name}")))?;
    match tensor {
        TensorValue::F32(arr) => {
            let shape = arr.shape();
            if shape.len() != 4 || shape[0] != 1 {
                return Err(UnikoError::Pipeline(format!(
                    "{name}: expected [1, seq, seq, num_rels], got {shape:?}"
                )));
            }
            let squeezed = arr
                .clone()
                .into_shape_with_order(ndarray::IxDyn(&[shape[1], shape[2], shape[3]]))
                .map_err(|e| UnikoError::Pipeline(format!("reshape {name}: {e}")))?;
            squeezed
                .into_dimensionality()
                .map_err(|e| UnikoError::Pipeline(format!("dim {name}: {e}")))
        }
        _ => Err(UnikoError::Pipeline(format!("{name}: expected F32 tensor"))),
    }
}

/// Extract `arc_scores[B, S, S]` from a batched forward pass.
fn extract_f32_3d_square(
    outputs: &TensorBatch,
    name: &str,
    expected_batch: usize,
    expected_seq: usize,
) -> Result<ndarray::Array3<f32>, UnikoError> {
    let tensor = outputs
        .get(name)
        .ok_or_else(|| UnikoError::Pipeline(format!("missing output tensor: {name}")))?;
    match tensor {
        TensorValue::F32(arr) => {
            let shape = arr.shape();
            if shape.len() != 3
                || shape[0] != expected_batch
                || shape[1] != expected_seq
                || shape[2] != expected_seq
            {
                return Err(UnikoError::Pipeline(format!(
                    "{name}: expected [{expected_batch}, {expected_seq}, {expected_seq}], got {shape:?}"
                )));
            }
            arr.clone()
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| UnikoError::Pipeline(format!("dim {name}: {e}")))
        }
        _ => Err(UnikoError::Pipeline(format!("{name}: expected F32 tensor"))),
    }
}

/// Extract `label_scores[B, S, S, num_rels]` from a batched forward pass.
fn extract_f32_4d(
    outputs: &TensorBatch,
    name: &str,
    expected_batch: usize,
    expected_seq: usize,
) -> Result<ndarray::Array4<f32>, UnikoError> {
    let tensor = outputs
        .get(name)
        .ok_or_else(|| UnikoError::Pipeline(format!("missing output tensor: {name}")))?;
    match tensor {
        TensorValue::F32(arr) => {
            let shape = arr.shape();
            if shape.len() != 4
                || shape[0] != expected_batch
                || shape[1] != expected_seq
                || shape[2] != expected_seq
            {
                return Err(UnikoError::Pipeline(format!(
                    "{name}: expected [{expected_batch}, {expected_seq}, {expected_seq}, *], got {shape:?}"
                )));
            }
            arr.clone()
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(|e| UnikoError::Pipeline(format!("dim {name}: {e}")))
        }
        _ => Err(UnikoError::Pipeline(format!("{name}: expected F32 tensor"))),
    }
}
