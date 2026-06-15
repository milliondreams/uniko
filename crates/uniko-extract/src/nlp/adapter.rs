//! Adapter: xervo `NlpModel` output → uniko's [`NlpResult`].
//!
//! xervo decodes POS/NER/DEP/SRL/CLS internally and returns a
//! token-oriented [`uni_xervo::traits::NlpResult`]. uniko's downstream
//! (`ner`, `observations`) consumes a word-oriented
//! [`crate::nlp::types::NlpResult`]. This module reconciles the three
//! representation differences the `nlp-parity` bench proved to be the
//! *only* divergences between the two decoders on identical weights:
//!
//! 1. **Tokenization** — xervo surfaces the SentencePiece metaspace: a
//!    token starting with a space (or `▁`) opens a new word, otherwise it
//!    continues the previous one. Grouping on that boundary recovers
//!    uniko's `token_to_word` word units (and folds trailing punctuation
//!    such as `.` back onto the number it follows).
//! 2. **Dependency heads** — xervo's `DepLink.head` (0.14+) is an
//!    `Option<usize>` 0-based *global* token index, `None` = sentence
//!    root; we map the head token to its word, treating `None`/self as
//!    root (the word-level [`DepArc::head`] sentinel is `usize::MAX`).
//! 3. **NER** — xervo returns raw per-token BIO tags (`B-PERSON`, …), so
//!    we BIO-merge them into spans via [`super::decode::build_span`] /
//!    [`super::decode::parse_ner_type`], exactly as the in-crate decoder
//!    did, preserving the same [`NerEntityType`](super::types::NerEntityType)
//!    mapping.

use uni_xervo::traits::NlpResult as XervoNlp;

use super::assets::LabelMaps;
use super::decode::{build_span, parse_ner_type};
use super::types::{DepArc, NerSpan, NlpResult, SentenceClass, SrlArg, SrlFrame};

/// Convert one xervo sentence result into uniko's [`NlpResult`].
///
/// `srl_enabled` mirrors `UnikoConfig.nlp_srl_enabled`; when `false`,
/// `srl_frames` is left empty regardless of what xervo returned.
pub fn xervo_to_uniko(x: &XervoNlp, labels: &LabelMaps, srl_enabled: bool) -> NlpResult {
    let (words, tok2word) = reconstruct_words(x);
    let n = words.len();

    // First token index of each word (first-subword-wins, matching uniko).
    let mut first_tok: Vec<Option<usize>> = vec![None; n];
    for (i, &w) in tok2word.iter().enumerate() {
        first_tok[w].get_or_insert(i);
    }

    // POS: map each word's first-token tag string to its label index.
    let pos_unknown = labels.pos_labels.iter().position(|l| l == "X").unwrap_or(0);
    let pos_indices: Vec<usize> = (0..n)
        .map(|w| {
            first_tok[w]
                .and_then(|i| x.tokens[i].pos.as_deref())
                .and_then(|p| {
                    labels
                        .pos_labels
                        .iter()
                        .position(|l| l.eq_ignore_ascii_case(p))
                })
                .unwrap_or(pos_unknown)
        })
        .collect();

    // NER: per-word tag (first token), then label index + BIO-merge.
    let word_ner: Vec<&str> = (0..n)
        .map(|w| {
            first_tok[w]
                .and_then(|i| x.tokens[i].ner.as_deref())
                .unwrap_or("O")
        })
        .collect();
    let ner_indices: Vec<usize> = word_ner
        .iter()
        .map(|tag| {
            labels
                .ner_labels
                .iter()
                .position(|l| l.eq_ignore_ascii_case(tag))
                .unwrap_or(0)
        })
        .collect();
    let entities = merge_word_bio(&words, &word_ner);

    // DEP: one word-level arc per word, from its first token's link.
    let mut dep_arcs = Vec::new();
    for (w, slot) in first_tok.iter().enumerate() {
        let Some(i) = *slot else { continue };
        let Some(d) = x.tokens[i].dep.as_ref() else {
            continue;
        };
        // xervo head: 0-based global token index, `None` = sentence root.
        // Map the head token back to a word; a head landing in the same
        // word (or `None`) is the sentence root.
        let head = d
            .head
            .and_then(|ht| tok2word.get(ht).copied())
            .filter(|&hw| hw != w)
            .unwrap_or(usize::MAX);
        dep_arcs.push(DepArc {
            dependent: w,
            head,
            relation: d.relation.clone(),
        });
    }

    // SRL: token spans → word spans (xervo span is inclusive `[first,
    // last]`; uniko `end_word` is exclusive).
    let srl_frames = if srl_enabled {
        x.frames
            .iter()
            .map(|f| {
                let predicate_idx = tok2word.get(f.predicate_token).copied().unwrap_or(0);
                let args = f
                    .roles
                    .iter()
                    .filter_map(|r| {
                        let start = tok2word.get(r.span.0).copied()?;
                        let last = tok2word.get(r.span.1).copied()?;
                        let end = (last + 1).clamp(start + 1, n);
                        Some(SrlArg {
                            role: r.label.clone(),
                            text: words.get(start..end)?.join(" "),
                            start_word: start,
                            end_word: end,
                        })
                    })
                    .collect();
                SrlFrame {
                    predicate_idx,
                    predicate_word: words.get(predicate_idx).cloned().unwrap_or_default(),
                    args,
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    // CLS: top speech act → class + index. xervo exposes only the top
    // label + confidence (no full softmax), so `cls_probs` stays empty
    // and the observation gate falls back to `sentence_class`.
    let (cls_index, cls_confidence, sentence_class) = match x.speech_acts.first() {
        Some(act) => (
            labels
                .cls_labels
                .iter()
                .position(|l| l.eq_ignore_ascii_case(&act.label))
                .unwrap_or(0),
            act.confidence,
            SentenceClass::from_label(&act.label),
        ),
        None => (0, 0.0, SentenceClass::Statement),
    };

    NlpResult {
        words,
        ner_indices,
        pos_indices,
        cls_index,
        cls_confidence,
        cls_probs: Vec::new(),
        entities,
        dep_arcs,
        sentence_class,
        srl_frames,
    }
}

/// Reconstruct uniko word units from xervo tokens by SentencePiece
/// metaspace boundaries. Returns the word surfaces and a `token → word`
/// index map.
fn reconstruct_words(x: &XervoNlp) -> (Vec<String>, Vec<usize>) {
    let mut words: Vec<String> = Vec::new();
    let mut map: Vec<usize> = Vec::with_capacity(x.tokens.len());
    for tok in &x.tokens {
        let raw = tok.text.as_str();
        let starts_word = words.is_empty() || raw.starts_with([' ', '\u{2581}']);
        let piece = raw.trim_start_matches([' ', '\u{2581}']);
        if starts_word {
            words.push(piece.to_string());
        } else if let Some(last) = words.last_mut() {
            last.push_str(piece);
        } else {
            words.push(piece.to_string());
        }
        map.push(words.len() - 1);
    }
    (words, map)
}

/// BIO-merge per-word NER tags into [`NerSpan`]s (`end_word` exclusive).
///
/// `B-X` (or an orphan `I-X`) opens a span; a matching `I-X` extends it;
/// `O`/empty closes it. Mirrors `decode::merge_bio_spans`.
fn merge_word_bio(words: &[String], word_ner: &[&str]) -> Vec<NerSpan> {
    let mut spans = Vec::new();
    let mut run: Option<(String, usize)> = None; // (type suffix, start word)
    for (w, tag) in word_ner.iter().enumerate() {
        if *tag == "O" || tag.is_empty() {
            if let Some((suf, start)) = run.take() {
                spans.push(build_span(words, start, w, parse_ner_type(&suf)));
            }
            continue;
        }
        let (prefix, suffix) = tag.split_once('-').unwrap_or(("B", tag));
        let continues = matches!(&run, Some((cur, _)) if prefix == "I" && cur == suffix);
        if !continues {
            if let Some((suf, start)) = run.take() {
                spans.push(build_span(words, start, w, parse_ner_type(&suf)));
            }
            run = Some((suffix.to_string(), w));
        }
    }
    if let Some((suf, start)) = run.take() {
        spans.push(build_span(words, start, words.len(), parse_ner_type(&suf)));
    }
    spans
}
