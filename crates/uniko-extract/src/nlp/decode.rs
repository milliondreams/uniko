//! Post-processing decoders for ONNX model output tensors.
//!
//! Converts raw logit tensors into structured NLP annotations:
//! BIO tags → entity spans, dep2label tags → dependency tree,
//! and dependency tree → SVO triples.

// Rust guideline compliant

use ndarray::{ArrayView1, ArrayView2};

use super::types::{DepArc, NerEntityType, NerSpan, SentenceClass};

// ── Argmax helpers ─────────────────────────────────────────────────

/// Argmax along the last axis of a 2D logits array.
///
/// Returns the index of the maximum value for each row.
pub fn argmax_rows(logits: &ArrayView2<f32>) -> Vec<usize> {
    logits
        .rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect()
}

/// Argmax and softmax confidence of a 1D logits array.
///
/// Returns `(predicted_index, confidence)` where confidence is the
/// softmax probability of the predicted class.
pub fn argmax_with_confidence(logits: &ArrayView1<f32>) -> (usize, f32) {
    let (idx, &max_val) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));

    // Softmax for the winning class: exp(max) / sum(exp(all))
    // Subtract max for numerical stability.
    let sum_exp: f32 = logits
        .iter()
        .map(|&v| (v - max_val).exp())
        .collect::<Vec<_>>()
        .iter()
        .sum();
    let confidence = 1.0 / sum_exp; // exp(0) / sum = 1 / sum

    (idx, confidence)
}

// ── Subword → word alignment ──────────────────────────────────────

/// Align subword-level predictions to word-level using first-subword-wins.
///
/// For each distinct word ID, takes the prediction at the word's first
/// subword token position.  Special tokens (word_id = None) are skipped.
pub fn align_to_words(subword_preds: &[usize], word_ids: &[Option<u32>]) -> Vec<usize> {
    let mut result = Vec::new();
    let mut last_word_id: Option<u32> = None;

    for (&pred, &wid) in subword_preds.iter().zip(word_ids.iter()) {
        if let Some(w) = wid
            && last_word_id != Some(w)
        {
            result.push(pred);
            last_word_id = Some(w);
        }
        // None = special token ([CLS], [SEP], padding) — skip.
    }

    result
}

/// Extract aligned word strings from the tokenizer encoding.
///
/// Groups subword tokens by word_id and joins them, stripping the
/// leading space markers used by different tokenizers:
/// - RoBERTa: `\u{0120}` (Ġ)
/// - DeBERTa/SentencePiece: `\u{2581}` (▁)
pub fn extract_words(tokens: &[String], word_ids: &[Option<u32>]) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    let mut last_word_id: Option<u32> = None;

    for (token, &wid) in tokens.iter().zip(word_ids.iter()) {
        if let Some(w) = wid {
            // Strip space markers from both RoBERTa (Ġ) and DeBERTa (▁).
            let clean = token
                .trim_start_matches('\u{0120}')
                .trim_start_matches('\u{2581}');
            if last_word_id == Some(w) {
                // Continuation subword — append to current word.
                if let Some(word) = words.last_mut() {
                    word.push_str(clean);
                }
            } else {
                // New word.
                words.push(clean.to_string());
                last_word_id = Some(w);
            }
        }
    }

    words
}

// ── BIO → entity spans ───────────────────────────────────────────

/// Merge BIO-tagged words into contiguous NER spans.
///
/// Handles orphan I- tags (I- without matching B-) by promoting them
/// to B- spans.
pub fn merge_bio_spans(
    words: &[String],
    tag_indices: &[usize],
    ner_labels: &[String],
) -> Vec<NerSpan> {
    let mut spans = Vec::new();
    let mut current_start: Option<usize> = None;
    let mut current_type: Option<NerEntityType> = None;

    for (i, &idx) in tag_indices.iter().enumerate() {
        let label = ner_labels.get(idx).map(String::as_str).unwrap_or("O");

        if let Some(stripped) = label.strip_prefix("B-") {
            // Flush any active span.
            if let (Some(start), Some(etype)) = (current_start, current_type) {
                spans.push(build_span(words, start, i, etype));
            }
            current_start = Some(i);
            current_type = Some(parse_ner_type(stripped));
        } else if let Some(stripped) = label.strip_prefix("I-") {
            let etype = parse_ner_type(stripped);
            if current_type == Some(etype) {
                // Continue current span.
            } else {
                // Orphan I- tag — promote to B-.
                if let (Some(start), Some(prev_type)) = (current_start, current_type) {
                    spans.push(build_span(words, start, i, prev_type));
                }
                current_start = Some(i);
                current_type = Some(etype);
            }
        } else {
            // O tag — flush active span.
            if let (Some(start), Some(etype)) = (current_start, current_type) {
                spans.push(build_span(words, start, i, etype));
            }
            current_start = None;
            current_type = None;
        }
    }

    // Flush trailing span.
    if let (Some(start), Some(etype)) = (current_start, current_type) {
        spans.push(build_span(words, start, words.len(), etype));
    }

    spans
}

fn build_span(words: &[String], start: usize, end: usize, entity_type: NerEntityType) -> NerSpan {
    let text = words[start..end].join(" ");
    NerSpan {
        text,
        entity_type,
        start_word: start,
        end_word: end,
        // Confidence is set to a fixed value here; the pipeline can
        // refine this using per-token softmax probabilities.
        confidence: 0.9,
    }
}

fn parse_ner_type(suffix: &str) -> NerEntityType {
    match suffix {
        // CoNLL scheme (distilroberta model)
        "PER" => NerEntityType::Person,
        // OntoNotes scheme (deberta model)
        "PERSON" => NerEntityType::Person,
        "ORG" => NerEntityType::Organization,
        "GPE" => NerEntityType::Location, // geopolitical entity → location
        "LOC" => NerEntityType::Location,
        "FAC" => NerEntityType::Location, // facility → location
        "DATE" | "TIME" => NerEntityType::Date,
        "MONEY" | "PERCENT" | "QUANTITY" | "CARDINAL" | "ORDINAL" => NerEntityType::Numeric,
        "EVENT" => NerEntityType::Event,
        "PRODUCT" => NerEntityType::Product,
        "WORK_OF_ART" => NerEntityType::WorkOfArt,
        "NORP" => NerEntityType::Group, // nationality/religious/political group
        "LAW" | "LANGUAGE" => NerEntityType::Misc,
        _ => NerEntityType::Misc,
    }
}

// ── dep2label → dependency tree ──────────────────────────────────

/// Decode dep2label predictions into dependency arcs.
///
/// Each dep2label string has the format `"{offset}@{relation}@{head_POS}"`.
/// The offset is a signed integer specifying how many tokens of the
/// given POS to skip in the indicated direction to find the head.
/// An offset of 0 with relation "root" marks the sentence root.
pub fn decode_dep_tree(
    dep_indices: &[usize],
    pos_indices: &[usize],
    dep_labels: &[String],
    pos_labels: &[String],
) -> Vec<DepArc> {
    let n = dep_indices.len();
    let mut arcs = Vec::with_capacity(n);

    for (i, &dep_idx) in dep_indices.iter().enumerate() {
        let label = dep_labels
            .get(dep_idx)
            .map(String::as_str)
            .unwrap_or("0@root@ROOT");

        let parts: Vec<&str> = label.splitn(3, '@').collect();
        if parts.len() < 3 {
            arcs.push(DepArc {
                dependent: i,
                head: usize::MAX,
                relation: "dep".to_string(),
            });
            continue;
        }

        let offset_str = parts[0];
        let relation = parts[1].to_string();
        let head_pos = parts[2];

        let offset: i32 = offset_str.parse().unwrap_or(0);

        if offset == 0 {
            // Root token.
            arcs.push(DepArc {
                dependent: i,
                head: usize::MAX,
                relation,
            });
            continue;
        }

        let head_idx = resolve_offset(i, offset, pos_indices, head_pos, pos_labels, n);
        arcs.push(DepArc {
            dependent: i,
            head: head_idx,
            relation,
        });
    }

    arcs
}

/// Resolve a signed POS-counted offset to an absolute token index.
///
/// Scans left (negative offset) or right (positive offset) from
/// `token_idx`, counting tokens whose POS label matches `target_pos`.
/// Returns the absolute index when the count reaches `|offset|`, or
/// `usize::MAX` if the boundary is reached first.
#[expect(
    clippy::needless_range_loop,
    reason = "positional scanning with early return is clearer than enumerate+skip"
)]
fn resolve_offset(
    token_idx: usize,
    offset: i32,
    pos_indices: &[usize],
    target_pos: &str,
    pos_labels: &[String],
    n: usize,
) -> usize {
    let abs_offset = offset.unsigned_abs() as usize;
    let mut count = 0usize;

    if offset > 0 {
        // Scan right.
        for j in (token_idx + 1)..n {
            let pos = pos_labels
                .get(pos_indices[j])
                .map(String::as_str)
                .unwrap_or("");
            if pos == target_pos {
                count += 1;
                if count == abs_offset {
                    return j;
                }
            }
        }
    } else {
        // Scan left.
        for j in (0..token_idx).rev() {
            let pos = pos_labels
                .get(pos_indices[j])
                .map(String::as_str)
                .unwrap_or("");
            if pos == target_pos {
                count += 1;
                if count == abs_offset {
                    return j;
                }
            }
        }
    }

    // Boundary reached without finding the head — mark as unresolved.
    usize::MAX
}

// ── SVO extraction from dependency tree ──────────────────────────

/// Subject-verb-object triple extracted from the dependency tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvoTriple {
    /// Subject text.
    pub subject: String,
    /// Verb text.
    pub verb: String,
    /// Object text (may be empty if intransitive).
    pub object: String,
}

/// Extract SVO triples from word tokens and dependency arcs.
///
/// Finds verb tokens, then looks for `nsubj`/`nsubj:pass` and
/// `obj`/`dobj`/`iobj` dependents to form structured triples.
pub fn extract_svo_triples(
    words: &[String],
    pos_indices: &[usize],
    dep_arcs: &[DepArc],
    pos_labels: &[String],
) -> Vec<SvoTriple> {
    let n = words.len();
    let mut triples = Vec::new();

    // Find verb tokens (POS == VERB or AUX).
    for verb_idx in 0..n {
        let pos = pos_labels
            .get(*pos_indices.get(verb_idx).unwrap_or(&0))
            .map(String::as_str)
            .unwrap_or("");
        if pos != "VERB" {
            continue;
        }

        let mut subject: Option<String> = None;
        let mut object: Option<String> = None;

        // Find dependents of this verb.
        for arc in dep_arcs {
            if arc.head != verb_idx {
                continue;
            }

            match arc.relation.as_str() {
                "nsubj" | "nsubj:pass" | "csubj" => {
                    subject = Some(collect_subtree(arc.dependent, words, dep_arcs));
                }
                "obj" | "dobj" | "iobj" => {
                    object = Some(collect_subtree(arc.dependent, words, dep_arcs));
                }
                _ => {}
            }
        }

        if let Some(subj) = subject {
            triples.push(SvoTriple {
                subject: subj,
                verb: words[verb_idx].clone(),
                object: object.unwrap_or_default(),
            });
        }
    }

    triples
}

/// Collect a subtree rooted at `root_idx` as a space-joined string.
///
/// Recursively gathers all tokens whose head chain leads to `root_idx`,
/// sorted by position for natural reading order.
fn collect_subtree(root_idx: usize, words: &[String], dep_arcs: &[DepArc]) -> String {
    let mut indices = vec![root_idx];
    // BFS to find all descendants.
    let mut queue = vec![root_idx];
    while let Some(head) = queue.pop() {
        for arc in dep_arcs {
            if arc.head == head && !indices.contains(&arc.dependent) {
                indices.push(arc.dependent);
                queue.push(arc.dependent);
            }
        }
    }
    indices.sort_unstable();
    indices
        .iter()
        .filter_map(|&i| words.get(i))
        .cloned()
        .collect::<Vec<_>>()
        .join(" ")
}

/// A clean declarative observation reconstructed from the DEP tree.
#[derive(Debug, Clone)]
pub struct DepObservation {
    /// Clean declarative sentence (e.g., "Jon got a temp job").
    pub content: String,
    /// Subject of the observation (speaker name for first-person).
    pub subject: String,
    /// Confidence score.
    pub confidence: f64,
}

/// Extract clean declarative observations from DEP tree + speaker.
///
/// For each predicate (VERB or copular ADJ/NOUN with `nsubj`),
/// reconstruct `subject predicate objects` from the dependency
/// structure. Pronouns are resolved using the session-level
/// [`SentenceContext`]: first-person → speaker, second-person →
/// other speaker, third-person → last noun from context.
///
/// After extraction, the context is updated with any new noun
/// phrases found as subjects or objects.
pub fn extract_dep_observations(
    words: &[String],
    pos_indices: &[usize],
    dep_arcs: &[DepArc],
    pos_labels: &[String],
    speaker: &str,
    ctx: &mut crate::ingest::context::SentenceContext,
) -> Vec<DepObservation> {
    let n = words.len();
    let mut observations = Vec::new();

    for pred_idx in 0..n {
        let pos = pos_labels
            .get(*pos_indices.get(pred_idx).unwrap_or(&0))
            .map(String::as_str)
            .unwrap_or("");

        // Accept VERB tokens, and ADJ/NOUN tokens that act as copular
        // predicates (e.g., "stories were inspiring", "mom has been supportive").
        let is_verb = pos == "VERB";
        let is_copular_pred = (pos == "ADJ" || pos == "NOUN")
            && dep_arcs
                .iter()
                .any(|a| a.head == pred_idx && a.relation == "cop");

        if !is_verb && !is_copular_pred {
            continue;
        }

        let mut subject: Option<String> = None;
        let mut objects: Vec<String> = Vec::new();
        let mut modifiers: Vec<String> = Vec::new();
        let mut has_cop = false;

        for arc in dep_arcs {
            if arc.head != pred_idx {
                continue;
            }

            match arc.relation.as_str() {
                "nsubj" | "nsubj:pass" | "csubj" => {
                    let subj_pos = pos_labels
                        .get(*pos_indices.get(arc.dependent).unwrap_or(&0))
                        .map(String::as_str)
                        .unwrap_or("");
                    let raw_text = collect_noun_phrase(arc.dependent, words, dep_arcs);
                    subject = Some(resolve_subject(&raw_text, subj_pos, ctx, speaker));
                }
                "obj" | "dobj" | "iobj" => {
                    objects.push(collect_noun_phrase(arc.dependent, words, dep_arcs));
                }
                "obl" | "advmod" | "xcomp" | "advcl" => {
                    modifiers.push(collect_subtree(arc.dependent, words, dep_arcs));
                }
                "cop" => has_cop = true,
                _ => {}
            }
        }

        // Need at least a subject or object to form an observation.
        if subject.is_none() && objects.is_empty() {
            continue;
        }

        let subj = subject.unwrap_or_else(|| speaker.to_string());
        let pred_word = strip_trailing_punct(&words[pred_idx]);

        // Build content: for copular predicates include "is/was" + adjective.
        let mut content = if is_copular_pred && has_cop {
            format!("{subj} is {pred_word}")
        } else {
            format!("{subj} {pred_word}")
        };
        for obj in &objects {
            content.push(' ');
            content.push_str(&strip_trailing_punct(obj));
        }
        for m in &modifiers {
            content.push(' ');
            content.push_str(&strip_trailing_punct(m));
        }

        // Quality gate: skip observations shorter than 3 content words.
        let content_words = content.split_whitespace().count();
        if content_words < 3 {
            continue;
        }

        observations.push(DepObservation {
            content,
            subject: subj.clone(),
            confidence: 0.85,
        });
    }

    // Update context with noun phrases from this sentence.
    update_sentence_context(ctx, words, pos_indices, dep_arcs, pos_labels);

    observations
}

/// Resolve a subject token to a concrete name using the context window.
///
/// Priority: first-person → speaker, second-person → other speaker,
/// third-person pronoun → last noun from context, not a pronoun → as-is.
fn resolve_subject(
    text: &str,
    pos: &str,
    ctx: &crate::ingest::context::SentenceContext,
    speaker: &str,
) -> String {
    let lower = text.to_lowercase();

    // First person → speaker.
    if lower == "i" || lower == "we" || lower == "me" || lower == "my" || lower == "myself" {
        return speaker.to_string();
    }
    if lower.starts_with("i'") || lower.starts_with("i\u{2019}") {
        return speaker.to_string();
    }
    if lower.starts_with("we'") || lower.starts_with("we\u{2019}") {
        return speaker.to_string();
    }

    // Second person → other speaker.
    if lower == "you"
        || lower.starts_with("you'")
        || lower.starts_with("you\u{2019}")
        || lower == "your"
        || lower == "yours"
        || lower == "yourself"
    {
        if let Some(other) = ctx.other_speakers.first() {
            return other.clone();
        }
    }

    // Third person pronoun → last noun from context.
    if pos == "PRON"
        && matches!(
            lower.as_str(),
            "it" | "it's" | "it'd" | "it'll" | "this" | "that" | "they" | "there"
        )
    {
        if let Some(ref obj) = ctx.last_noun_object {
            return obj.clone();
        }
        if let Some(ref subj) = ctx.last_noun_subject {
            return subj.clone();
        }
        // No context available — skip this observation by returning
        // the pronoun (will likely be filtered by quality gate or
        // won't match any entity).
    }

    text.to_string()
}

/// Collect a clean noun phrase from a dependency subtree.
///
/// Only includes content-bearing modifiers: `compound`, `amod`, `flat`,
/// `nmod:poss`, `det`. Excludes function words and clausal dependents
/// (`cc`, `punct`, `mark`, `acl`, `advcl`, `conj`) that cause garbled output.
fn collect_noun_phrase(root_idx: usize, words: &[String], dep_arcs: &[DepArc]) -> String {
    let mut indices = vec![root_idx];
    let mut queue = vec![root_idx];
    while let Some(head) = queue.pop() {
        for arc in dep_arcs {
            if arc.head == head && !indices.contains(&arc.dependent) {
                // Only include content-bearing noun modifiers.
                match arc.relation.as_str() {
                    "compound" | "amod" | "flat" | "flat:name" | "nmod:poss" | "det" | "nummod" => {
                        indices.push(arc.dependent);
                        queue.push(arc.dependent);
                    }
                    _ => {}
                }
            }
        }
    }
    indices.sort_unstable();
    let phrase = indices
        .iter()
        .filter_map(|&i| words.get(i))
        .map(|w| strip_trailing_punct(w))
        .collect::<Vec<_>>()
        .join(" ");
    phrase
}

/// Update the sentence context with noun phrases from a processed sentence.
///
/// Scans DEP arcs for NOUN/PROPN tokens in subject and object positions,
/// recording them as the latest antecedents for pronoun resolution.
pub fn update_sentence_context(
    ctx: &mut crate::ingest::context::SentenceContext,
    words: &[String],
    pos_indices: &[usize],
    dep_arcs: &[DepArc],
    pos_labels: &[String],
) {
    for arc in dep_arcs {
        let dep_pos = pos_labels
            .get(*pos_indices.get(arc.dependent).unwrap_or(&0))
            .map(String::as_str)
            .unwrap_or("");

        // Only update context from content words, not pronouns.
        if dep_pos != "NOUN" && dep_pos != "PROPN" {
            continue;
        }

        let phrase = collect_noun_phrase(arc.dependent, words, dep_arcs);
        if phrase.is_empty() {
            continue;
        }

        match arc.relation.as_str() {
            "nsubj" | "nsubj:pass" | "csubj" => {
                ctx.last_noun_subject = Some(phrase);
            }
            "obj" | "dobj" | "iobj" => {
                ctx.last_noun_object = Some(phrase);
            }
            _ => {}
        }
    }
}

/// Strip trailing punctuation from a word or phrase.
fn strip_trailing_punct(s: &str) -> String {
    s.trim_end_matches(|c: char| c.is_ascii_punctuation() && c != '\'')
        .to_string()
}

/// Map CLS index to sentence class.
pub fn decode_cls(cls_index: usize, cls_labels: &[String]) -> SentenceClass {
    cls_labels
        .get(cls_index)
        .map(|l| SentenceClass::from_label(l))
        .unwrap_or(SentenceClass::Statement)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn argmax_rows_picks_correct_indices() {
        let logits = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.1, 0.9, 0.2, 0.3, // row 0 → idx 1
                0.5, 0.1, 0.8, 0.2, // row 1 → idx 2
                0.3, 0.3, 0.3, 0.4, // row 2 → idx 3
            ],
        )
        .unwrap();
        assert_eq!(argmax_rows(&logits.view()), vec![1, 2, 3]);
    }

    #[test]
    fn align_to_words_first_subword_wins() {
        // word_ids: [None, 0, 0, 1, 2, 2, 2, None]
        // preds:    [  99, 3, 5, 7, 1, 2, 4,   99]
        // Expected: [3, 7, 1] (first subword of each word)
        let preds = vec![99, 3, 5, 7, 1, 2, 4, 99];
        let word_ids = vec![
            None,
            Some(0),
            Some(0),
            Some(1),
            Some(2),
            Some(2),
            Some(2),
            None,
        ];
        assert_eq!(align_to_words(&preds, &word_ids), vec![3, 7, 1]);
    }

    #[test]
    fn merge_bio_spans_basic() {
        let words = vec![
            "Caroline".into(),
            "went".into(),
            "to".into(),
            "Paris".into(),
        ];
        let labels = vec![
            "O".into(),
            "B-PER".into(),
            "I-PER".into(),
            "B-ORG".into(),
            "I-ORG".into(),
            "B-LOC".into(),
            "I-LOC".into(),
            "B-MISC".into(),
            "I-MISC".into(),
        ];
        // Caroline=B-PER(1), went=O(0), to=O(0), Paris=B-LOC(5)
        let tags = vec![1, 0, 0, 5];
        let spans = merge_bio_spans(&words, &tags, &labels);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].text, "Caroline");
        assert_eq!(spans[0].entity_type, NerEntityType::Person);
        assert_eq!(spans[1].text, "Paris");
        assert_eq!(spans[1].entity_type, NerEntityType::Location);
    }

    #[test]
    fn merge_bio_spans_multi_token() {
        let words = vec!["New".into(), "York".into(), "City".into()];
        let labels: Vec<String> = vec![
            "O".into(),
            "B-PER".into(),
            "I-PER".into(),
            "B-ORG".into(),
            "I-ORG".into(),
            "B-LOC".into(),
            "I-LOC".into(),
            "B-MISC".into(),
            "I-MISC".into(),
        ];
        // B-LOC(5), I-LOC(6), I-LOC(6)
        let tags = vec![5, 6, 6];
        let spans = merge_bio_spans(&words, &tags, &labels);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "New York City");
        assert_eq!(spans[0].entity_type, NerEntityType::Location);
    }

    #[test]
    fn decode_dep_tree_root_and_nsubj() {
        let dep_labels: Vec<String> = vec![
            "0@root@ROOT".into(),   // 0
            "+1@nsubj@VERB".into(), // 1
            "-1@obj@VERB".into(),   // 2
        ];
        let pos_labels: Vec<String> = vec!["NOUN".into(), "VERB".into()];
        // Words: ["Caroline", "works", "hard"]
        // POS:   [  NOUN(0),  VERB(1), NOUN(0)]
        // Dep:   [  +1@nsubj@VERB(1), 0@root@ROOT(0), -1@obj@VERB(2)]
        let dep_indices = vec![1, 0, 2];
        let pos_indices = vec![0, 1, 0];

        let arcs = decode_dep_tree(&dep_indices, &pos_indices, &dep_labels, &pos_labels);

        assert_eq!(arcs.len(), 3);

        // "Caroline" → head is "works" (first VERB to the right = idx 1)
        assert_eq!(arcs[0].dependent, 0);
        assert_eq!(arcs[0].head, 1);
        assert_eq!(arcs[0].relation, "nsubj");

        // "works" → root
        assert_eq!(arcs[1].dependent, 1);
        assert_eq!(arcs[1].head, usize::MAX);
        assert_eq!(arcs[1].relation, "root");

        // "hard" → head is "works" (first VERB to the left = idx 1)
        assert_eq!(arcs[2].dependent, 2);
        assert_eq!(arcs[2].head, 1);
        assert_eq!(arcs[2].relation, "obj");
    }

    #[test]
    fn extract_svo_simple() {
        let words: Vec<String> = vec![
            "Caroline".into(),
            "works".into(),
            "at".into(),
            "the".into(),
            "hospital".into(),
        ];
        let pos_labels: Vec<String> = vec![
            "NOUN".into(),  // 0
            "VERB".into(),  // 1
            "ADP".into(),   // 2
            "DET".into(),   // 3
            "PROPN".into(), // 4
        ];
        let pos_indices = vec![0, 1, 2, 3, 0]; // NOUN, VERB, ADP, DET, NOUN
        let dep_arcs = vec![
            DepArc {
                dependent: 0,
                head: 1,
                relation: "nsubj".into(),
            },
            DepArc {
                dependent: 1,
                head: usize::MAX,
                relation: "root".into(),
            },
            DepArc {
                dependent: 2,
                head: 4,
                relation: "case".into(),
            },
            DepArc {
                dependent: 3,
                head: 4,
                relation: "det".into(),
            },
            DepArc {
                dependent: 4,
                head: 1,
                relation: "obj".into(),
            },
        ];

        let triples = extract_svo_triples(&words, &pos_indices, &dep_arcs, &pos_labels);
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].subject, "Caroline");
        assert_eq!(triples[0].verb, "works");
        assert!(triples[0].object.contains("hospital"));
    }

    #[test]
    fn decode_cls_maps_labels() {
        let labels: Vec<String> = vec![
            "statement".into(),
            "question".into(),
            "question_fact".into(),
            "command".into(),
            "greeting".into(),
            "filler".into(),
            "acknowledgment".into(),
        ];
        assert_eq!(decode_cls(0, &labels), SentenceClass::Statement);
        assert_eq!(decode_cls(1, &labels), SentenceClass::Question);
        assert_eq!(decode_cls(4, &labels), SentenceClass::Greeting);
    }
}
