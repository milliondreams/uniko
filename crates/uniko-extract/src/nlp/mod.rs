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
    /// Mirror of `UnikoConfig.nlp_srl_enabled`. When `true`, after the
    /// main forward pass we issue a single batched inference covering
    /// every `(sentence, verb)` pair in the message and decode the
    /// resulting `srl_logits` into per-sentence `NlpResult.srl_frames`.
    srl_enabled: bool,
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
        let srl_enabled = kb.config().nlp_srl_enabled;
        Some(Self {
            runner,
            srl_enabled,
        })
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

        // 8. SRL frames — one batched forward covering every VERB token in
        // this sentence. The cascade takes `predicate_idx: i64[V]` and
        // emits `srl_logits: f32[V, S, C]`; we identify VERBs from the POS
        // argmax we just decoded, pack them all into a single inference
        // call, then decode per row. Skipped when the kill-switch
        // `nlp_srl_enabled = false` is set on UnikoConfig.
        let srl_frames = if self.srl_enabled {
            let srl_input_ids: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
            let srl_attention: Vec<i64> = attention.iter().map(|&x| x as i64).collect();
            let pad_id: i64 = tokenizer
                .token_to_id("[PAD]")
                .map(|v| v as i64)
                .unwrap_or(0);
            let mut frames_per_row = self
                .compute_srl_frames_batched(
                    std::slice::from_ref(&srl_input_ids),
                    std::slice::from_ref(&srl_attention),
                    std::slice::from_ref(&word_ids),
                    std::slice::from_ref(&words),
                    std::slice::from_ref(&pos_indices),
                    pad_id,
                    &labels.pos_labels,
                    &labels.srl_labels,
                )
                .await;
            frames_per_row.pop().unwrap_or_default()
        } else {
            Vec::new()
        };

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
            srl_frames,
        })
    }

    /// Run SRL inference for every VERB across a message in a single
    /// batched forward pass.
    ///
    /// The cascade accepts `predicate_idx: i64[V]` and emits
    /// `srl_logits: f32[V, S, C]`. We flatten `(row_i, verb_word_idx)`
    /// pairs across every input row (one row per sentence in a message),
    /// pack the source row's `input_ids` / `attention_mask` for each
    /// pair into a `[V_total, max_seq_len]` tensor, and issue **one**
    /// `runner.run` regardless of how many sentences or verbs the
    /// message contains.
    ///
    /// Output is per-row: `frames_per_row[i]` is the `SrlFrame` list for
    /// the `i`-th input row, with frame word indices remaining local to
    /// that row's `words` vec. Rows with zero VERB tokens get an empty
    /// `Vec`. If the whole message has zero VERBs, we return early
    /// without invoking the model. On any runner or tensor-shape
    /// failure, every row's frames fall back to empty (consistent with
    /// the pre-batching per-verb skip behavior, scaled to the new unit
    /// of failure).
    #[expect(clippy::too_many_arguments, reason = "decode requires per-row state")]
    async fn compute_srl_frames_batched(
        &self,
        row_ids: &[Vec<i64>],
        row_attention: &[Vec<i64>],
        row_word_ids: &[Vec<Option<u32>>],
        row_words: &[Vec<String>],
        row_pos_indices: &[Vec<usize>],
        pad_id: i64,
        pos_labels: &[String],
        srl_labels: &[String],
    ) -> Vec<Vec<types::SrlFrame>> {
        let n_rows = row_ids.len();
        let mut frames_per_row: Vec<Vec<types::SrlFrame>> = vec![Vec::new(); n_rows];

        // 1. Collect (row_i, word_idx, subword_idx) for every VERB in
        //    every row. POS indices are word-aligned; predicate_idx must
        //    be subword-aligned, so map via first_subword_per_word.
        let mut pairs: Vec<(usize, usize, usize)> = Vec::new();
        for (row_i, pos_indices) in row_pos_indices.iter().enumerate() {
            let word_to_subword = first_subword_per_word(&row_word_ids[row_i]);
            for (word_idx, &pos_idx) in pos_indices.iter().enumerate() {
                let pos = pos_labels.get(pos_idx).map(String::as_str).unwrap_or("");
                if pos != "VERB" {
                    continue;
                }
                let Some(&subword_idx) = word_to_subword.get(&word_idx) else {
                    continue;
                };
                pairs.push((row_i, word_idx, subword_idx));
            }
        }

        if pairs.is_empty() {
            return frames_per_row;
        }

        // 2. Build [V_total, max_seq_len] padded tensors. All pairs share
        //    one steady tensor shape, so the runner sees a single
        //    well-formed batched call.
        let max_seq_len = row_ids.iter().map(Vec::len).max().unwrap_or(0);
        if max_seq_len == 0 {
            return frames_per_row;
        }
        let v_total = pairs.len();
        let mut flat_ids: Vec<i64> = Vec::with_capacity(v_total * max_seq_len);
        let mut flat_attn: Vec<i64> = Vec::with_capacity(v_total * max_seq_len);
        let mut pidx: Vec<i64> = Vec::with_capacity(v_total);
        for &(row_i, _word_idx, subword_idx) in &pairs {
            let ids = &row_ids[row_i];
            let attn = &row_attention[row_i];
            flat_ids.extend_from_slice(ids);
            flat_ids.resize(flat_ids.len() + (max_seq_len - ids.len()), pad_id);
            flat_attn.extend_from_slice(attn);
            flat_attn.resize(flat_attn.len() + (max_seq_len - attn.len()), 0);
            pidx.push(subword_idx as i64);
        }

        let mut inputs = TensorBatch::default();
        let Ok(ids_arr) = ArrayD::from_shape_vec(vec![v_total, max_seq_len], flat_ids) else {
            tracing::debug!(v_total, max_seq_len, "SRL batched: input_ids shape failed");
            return frames_per_row;
        };
        let Ok(attn_arr) = ArrayD::from_shape_vec(vec![v_total, max_seq_len], flat_attn) else {
            tracing::debug!(v_total, max_seq_len, "SRL batched: attention shape failed");
            return frames_per_row;
        };
        let Ok(pred_arr) = ArrayD::from_shape_vec(vec![v_total], pidx) else {
            tracing::debug!(v_total, "SRL batched: predicate_idx shape failed");
            return frames_per_row;
        };
        inputs.insert("input_ids", TensorValue::I64(ids_arr));
        inputs.insert("attention_mask", TensorValue::I64(attn_arr));
        inputs.insert("predicate_idx", TensorValue::I64(pred_arr));

        // 3. One ONNX forward for every (sentence, verb) pair.
        let outputs = match self.runner.run(&inputs).await {
            Ok(o) => o,
            Err(e) => {
                tracing::debug!(error = %e, v_total, "SRL batched forward failed");
                return frames_per_row;
            }
        };
        let srl_logits = match extract_f32_3d(&outputs, "srl_logits", v_total, max_seq_len) {
            Ok(t) => t,
            Err(e) => {
                tracing::debug!(error = %e, "SRL batched: extract srl_logits failed");
                return frames_per_row;
            }
        };

        // 4. Scatter back per pair; slice to the source row's true
        //    seq_len so padded positions do not leak into decode.
        for (v, &(row_i, word_idx, _subword_idx)) in pairs.iter().enumerate() {
            let seq_len_i = row_ids[row_i].len();
            let slice = srl_logits.slice(ndarray::s![v, ..seq_len_i, ..]);
            let srl_subword = decode::argmax_rows(&slice);
            let srl_word_indices = decode::align_to_words(&srl_subword, &row_word_ids[row_i]);
            frames_per_row[row_i].push(decode::decode_srl_frame(
                &srl_word_indices,
                &row_words[row_i],
                srl_labels,
                word_idx,
            ));
        }

        frames_per_row
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

        // 5. Per-row decode using each sentence's true seq_len. SRL is
        //    deferred — we collect per-row state and run a single
        //    message-wide SRL batch after the loop.
        let mut results = Vec::with_capacity(batch);
        let mut row_ids_vec: Vec<Vec<i64>> = Vec::with_capacity(batch);
        let mut row_attn_vec: Vec<Vec<i64>> = Vec::with_capacity(batch);
        let mut row_word_ids_vec: Vec<Vec<Option<u32>>> = Vec::with_capacity(batch);
        let mut row_words_vec: Vec<Vec<String>> = Vec::with_capacity(batch);
        let mut row_pos_indices_vec: Vec<Vec<usize>> = Vec::with_capacity(batch);

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

            // Stash per-row state for the post-loop batched SRL call.
            row_ids_vec.push(r.ids);
            row_attn_vec.push(r.attention);
            row_word_ids_vec.push(r.word_ids);
            row_words_vec.push(words.clone());
            row_pos_indices_vec.push(pos_indices.clone());

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
                srl_frames: Vec::new(),
            });
        }

        // 6. Message-scope SRL: one ONNX forward for every (sentence,
        //    verb) pair across the entire message. Empty messages or
        //    messages with zero VERBs skip the call entirely.
        if self.srl_enabled {
            let frames_per_row = self
                .compute_srl_frames_batched(
                    &row_ids_vec,
                    &row_attn_vec,
                    &row_word_ids_vec,
                    &row_words_vec,
                    &row_pos_indices_vec,
                    pad_id,
                    &labels.pos_labels,
                    &labels.srl_labels,
                )
                .await;
            for (r, frames) in results.iter_mut().zip(frames_per_row) {
                r.srl_frames = frames;
            }
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

// --- Tensor extraction helpers ---
//
// All `extract_f32_*` functions share three operations: fetch the named
// tensor, assert it's an F32 array, and coerce its dynamic dimensions to
// a rank-typed `Array<f32, D>`. These two helpers factor those out so
// each public extractor reduces to its shape contract.

/// Fetch a named output tensor and clone its underlying `ArrayD<f32>`.
fn fetch_f32(outputs: &TensorBatch, name: &str) -> Result<ndarray::ArrayD<f32>, UnikoError> {
    let tensor = outputs
        .get(name)
        .ok_or_else(|| UnikoError::Pipeline(format!("missing output tensor: {name}")))?;
    match tensor {
        TensorValue::F32(arr) => Ok(arr.clone()),
        _ => Err(UnikoError::Pipeline(format!("{name}: expected F32 tensor"))),
    }
}

/// Coerce dynamic dims → rank-typed dims, mapping the error consistently.
fn into_dim<D: ndarray::Dimension>(
    arr: ndarray::ArrayD<f32>,
    name: &str,
) -> Result<ndarray::Array<f32, D>, UnikoError> {
    arr.into_dimensionality::<D>()
        .map_err(|e| UnikoError::Pipeline(format!("dim {name}: {e}")))
}

/// Reshape to `new_shape` then coerce to rank-typed dims.
fn reshape_into<D: ndarray::Dimension>(
    arr: ndarray::ArrayD<f32>,
    name: &str,
    new_shape: &[usize],
) -> Result<ndarray::Array<f32, D>, UnikoError> {
    let reshaped = arr
        .into_shape_with_order(ndarray::IxDyn(new_shape))
        .map_err(|e| UnikoError::Pipeline(format!("reshape {name}: {e}")))?;
    into_dim(reshaped, name)
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
    let arr = fetch_f32(outputs, name)?;
    let shape = arr.shape();
    if shape.len() != 3 || shape[0] != expected_batch || shape[1] != expected_seq {
        return Err(UnikoError::Pipeline(format!(
            "{name}: expected [{expected_batch}, {expected_seq}, *], got {shape:?}"
        )));
    }
    into_dim(arr, name)
}

/// Extract a 2D f32 tensor from outputs as `[batch, num_classes]`.
///
/// Used for the `cls_logits` head when running a multi-row batch.
fn extract_f32_2d_batch(
    outputs: &TensorBatch,
    name: &str,
    expected_batch: usize,
) -> Result<ndarray::Array2<f32>, UnikoError> {
    let arr = fetch_f32(outputs, name)?;
    let shape = arr.shape();
    if shape.len() != 2 || shape[0] != expected_batch {
        return Err(UnikoError::Pipeline(format!(
            "{name}: expected [{expected_batch}, *], got {shape:?}"
        )));
    }
    into_dim(arr, name)
}

/// Extract a 2D f32 tensor from outputs, squeezing the batch dimension.
///
/// Expects shape `[1, seq_len, num_classes]` → returns `[seq_len, num_classes]`.
fn extract_f32_2d(outputs: &TensorBatch, name: &str) -> Result<ndarray::Array2<f32>, UnikoError> {
    let arr = fetch_f32(outputs, name)?;
    let shape = arr.shape().to_vec();
    match shape.len() {
        3 if shape[0] == 1 => reshape_into(arr, name, &[shape[1], shape[2]]),
        2 => into_dim(arr, name),
        _ => Err(UnikoError::Pipeline(format!(
            "{name}: unexpected shape {shape:?}"
        ))),
    }
}

/// Extract a 1D f32 tensor from outputs, squeezing the batch dimension.
///
/// Expects shape `[1, num_classes]` → returns `[num_classes]`.
fn extract_f32_1d(outputs: &TensorBatch, name: &str) -> Result<ndarray::Array1<f32>, UnikoError> {
    let arr = fetch_f32(outputs, name)?;
    let shape = arr.shape().to_vec();
    match shape.len() {
        2 if shape[0] == 1 => reshape_into(arr, name, &[shape[1]]),
        1 => into_dim(arr, name),
        _ => Err(UnikoError::Pipeline(format!(
            "{name}: unexpected shape {shape:?}"
        ))),
    }
}

/// Extract `arc_scores[1, seq, seq]` and squeeze to `[seq, seq]`.
fn extract_f32_2d_squeeze_batch(
    outputs: &TensorBatch,
    name: &str,
) -> Result<ndarray::Array2<f32>, UnikoError> {
    let arr = fetch_f32(outputs, name)?;
    let shape = arr.shape().to_vec();
    if shape.len() != 3 || shape[0] != 1 {
        return Err(UnikoError::Pipeline(format!(
            "{name}: expected [1, seq, seq], got {shape:?}"
        )));
    }
    reshape_into(arr, name, &[shape[1], shape[2]])
}

/// Extract `label_scores[1, seq, seq, num_rels]` and squeeze to
/// `[seq, seq, num_rels]`.
fn extract_f32_3d_squeeze_batch(
    outputs: &TensorBatch,
    name: &str,
) -> Result<ndarray::Array3<f32>, UnikoError> {
    let arr = fetch_f32(outputs, name)?;
    let shape = arr.shape().to_vec();
    if shape.len() != 4 || shape[0] != 1 {
        return Err(UnikoError::Pipeline(format!(
            "{name}: expected [1, seq, seq, num_rels], got {shape:?}"
        )));
    }
    reshape_into(arr, name, &[shape[1], shape[2], shape[3]])
}

/// Extract `arc_scores[B, S, S]` from a batched forward pass.
fn extract_f32_3d_square(
    outputs: &TensorBatch,
    name: &str,
    expected_batch: usize,
    expected_seq: usize,
) -> Result<ndarray::Array3<f32>, UnikoError> {
    let arr = fetch_f32(outputs, name)?;
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
    into_dim(arr, name)
}

/// Extract `label_scores[B, S, S, num_rels]` from a batched forward pass.
fn extract_f32_4d(
    outputs: &TensorBatch,
    name: &str,
    expected_batch: usize,
    expected_seq: usize,
) -> Result<ndarray::Array4<f32>, UnikoError> {
    let arr = fetch_f32(outputs, name)?;
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
    into_dim(arr, name)
}

/// Build a `word_idx → first_subword_idx` map from the tokenizer's
/// per-token word_id alignment. The model takes `predicate_idx` as a
/// subword index (it indexes encoder hidden states), so we project our
/// word-level VERB indices back into subword space before each SRL
/// re-forward.
fn first_subword_per_word(word_ids: &[Option<u32>]) -> std::collections::HashMap<usize, usize> {
    let mut map = std::collections::HashMap::new();
    for (subword_idx, &wid) in word_ids.iter().enumerate() {
        if let Some(w) = wid {
            map.entry(w as usize).or_insert(subword_idx);
        }
    }
    map
}
