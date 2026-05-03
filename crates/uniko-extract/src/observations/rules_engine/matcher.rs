//! Generic pattern matcher.
//!
//! Walks a DEP tree once per anchor candidate; for each pattern whose
//! anchor POS and required children all match, fills captures, renders
//! the template, applies the global filters, and emits a
//! [`DepObservation`].

use std::collections::{BTreeSet, HashMap};

use crate::ingest::context::SentenceContext;
use crate::nlp::decode::DepObservation;
use crate::nlp::types::DepArc;

use super::resolver::{
    ResolvedSubject, collect_subtree, collect_with_relations, resolve_subject, strip_trailing_punct,
    update_sentence_context,
};
use super::rules::{ChildSpec, Pattern, PhraseCollector, Rules};
use super::template::render;

const SUBJECT_CAPTURE: &str = "subject";
const ANCHOR_CAPTURE: &str = "anchor";

/// Apply every pattern in `rules` to the sentence and return the
/// resulting observations. Updates `ctx` with antecedents at the end
/// (matching the legacy pipeline's behaviour).
pub fn extract_with_rules(
    rules: &Rules,
    words: &[String],
    pos_indices: &[usize],
    dep_arcs: &[DepArc],
    pos_labels: &[String],
    speaker: &str,
    ctx: &mut SentenceContext,
) -> Vec<DepObservation> {
    let mut out = Vec::new();
    let mut seen_text: BTreeSet<String> = BTreeSet::new();

    for anchor_idx in 0..words.len() {
        let anchor_pos = pos_at(anchor_idx, pos_indices, pos_labels);
        for pattern in &rules.patterns {
            if !pattern.match_.anchor.pos.contains(anchor_pos) {
                continue;
            }
            if let Some(obs) = try_match(
                pattern, anchor_idx, anchor_pos, words, pos_indices, dep_arcs, pos_labels, speaker,
                ctx, rules,
            ) && seen_text.insert(obs.content.clone()) {
                out.push(obs);
            }
        }
    }

    let np_relations = rules
        .phrase_collectors
        .get("noun_phrase")
        .map(|c| c.include_relations.clone())
        .unwrap_or_default();
    update_sentence_context(ctx, words, pos_indices, dep_arcs, pos_labels, &np_relations);

    out
}

#[allow(clippy::too_many_arguments)]
fn try_match(
    pattern: &Pattern,
    anchor_idx: usize,
    anchor_pos: &str,
    words: &[String],
    pos_indices: &[usize],
    dep_arcs: &[DepArc],
    pos_labels: &[String],
    speaker: &str,
    ctx: &SentenceContext,
    rules: &Rules,
) -> Option<DepObservation> {
    let mut captures: HashMap<String, String> = HashMap::new();
    captures.insert(
        ANCHOR_CAPTURE.into(),
        strip_trailing_punct(words.get(anchor_idx)?),
    );
    // Default `{subject}` to the speaker so templates can always include
    // it without requiring an explicit subject child rule. Any pattern
    // that does capture a subject will override below.
    captures.insert(SUBJECT_CAPTURE.into(), speaker.to_string());

    let mut subject_confidence: Option<f64> = None;

    for child in &pattern.match_.anchor.children {
        let matches: Vec<&DepArc> = dep_arcs
            .iter()
            .filter(|a| a.head == anchor_idx && child.dep.contains(&a.relation))
            .collect();

        if matches.is_empty() {
            if child.required {
                return None;
            }
            continue;
        }

        let Some(capture_name) = &child.capture else {
            continue;
        };

        let collected =
            collect_children(child, &matches, words, pos_indices, dep_arcs, pos_labels, rules);

        if collected.is_empty() {
            if child.required {
                return None;
            }
            continue;
        }

        if capture_name == SUBJECT_CAPTURE {
            // Subject POS gate: only NOUN/PROPN/PRON.
            let first_match = matches[0];
            let subj_pos = pos_at(first_match.dependent, pos_indices, pos_labels);
            if !rules
                .filters
                .subject_pos_required
                .iter()
                .any(|p| p == subj_pos)
            {
                return None;
            }
            // Resolve pronouns / drop unresolvable.
            let raw_subject =
                collect_with_relations(first_match.dependent, words, dep_arcs, &noun_phrase_relations(rules));
            let resolved = resolve_subject(
                &raw_subject,
                subj_pos,
                ctx,
                speaker,
                &rules.subject_resolution,
            );
            match resolved {
                ResolvedSubject::Resolved(name) => {
                    captures.insert(SUBJECT_CAPTURE.into(), name);
                    subject_confidence = Some(0.85);
                }
                ResolvedSubject::Unresolvable => {
                    if rules.filters.drop_when_subject_unresolvable {
                        return None;
                    }
                    captures.insert(SUBJECT_CAPTURE.into(), raw_subject);
                }
            }
        } else {
            captures.insert(capture_name.clone(), collected);
        }
    }

    // Pattern-level quality knobs.
    if let Some(min_mod_words) = pattern.quality.min_modifier_words {
        let mods = captures.get("modifier").or_else(|| captures.get("modifiers"));
        let words_in_mod = mods.map(|s| s.split_whitespace().count()).unwrap_or(0);
        if words_in_mod < min_mod_words {
            return None;
        }
    }

    // Render and apply global filters.
    let content = render(&pattern.template, &captures);
    if content.split_whitespace().count() < rules.filters.min_content_words {
        return None;
    }

    let subject = captures
        .get(SUBJECT_CAPTURE)
        .cloned()
        .unwrap_or_else(|| speaker.to_string());

    let _ = anchor_pos; // kept for future per-POS routing; presently unused
    Some(DepObservation {
        content,
        subject,
        confidence: subject_confidence.unwrap_or(0.85),
    })
}

#[allow(clippy::too_many_arguments)]
fn collect_children(
    child: &ChildSpec,
    matches: &[&DepArc],
    words: &[String],
    _pos_indices: &[usize],
    dep_arcs: &[DepArc],
    _pos_labels: &[String],
    rules: &Rules,
) -> String {
    let collector = lookup_collector(&child.collect, rules);
    let strategy: Box<dyn Fn(usize) -> String> = match collector {
        Some(c) if c.subtree => {
            let words = words.to_vec();
            let arcs = dep_arcs.to_vec();
            Box::new(move |idx| collect_subtree(idx, &words, &arcs))
        }
        Some(c) => {
            let words = words.to_vec();
            let arcs = dep_arcs.to_vec();
            let rels = c.include_relations.clone();
            Box::new(move |idx| collect_with_relations(idx, &words, &arcs, &rels))
        }
        None => {
            // Fallback: literal word.
            let words = words.to_vec();
            Box::new(move |idx| {
                words
                    .get(idx)
                    .map(|w| strip_trailing_punct(w))
                    .unwrap_or_default()
            })
        }
    };

    if child.multi {
        let mut indexed: Vec<(usize, String)> = matches
            .iter()
            .map(|a| (a.dependent, strategy(a.dependent)))
            .filter(|(_, s)| !s.is_empty())
            .collect();
        indexed.sort_by_key(|(i, _)| *i);
        indexed
            .into_iter()
            .map(|(_, s)| s)
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        matches
            .first()
            .map(|a| strategy(a.dependent))
            .unwrap_or_default()
    }
}

fn lookup_collector<'a>(name: &str, rules: &'a Rules) -> Option<&'a PhraseCollector> {
    if let Some(c) = rules.phrase_collectors.get(name) {
        return Some(c);
    }
    // Built-in synonym for `subtree` if the YAML didn't define one.
    if name == "subtree" {
        static SUBTREE: std::sync::OnceLock<PhraseCollector> = std::sync::OnceLock::new();
        return Some(SUBTREE.get_or_init(|| PhraseCollector {
            include_relations: vec![],
            subtree: true,
        }));
    }
    None
}

fn noun_phrase_relations(rules: &Rules) -> Vec<String> {
    rules
        .phrase_collectors
        .get("noun_phrase")
        .map(|c| c.include_relations.clone())
        .unwrap_or_default()
}

fn pos_at<'a>(idx: usize, pos_indices: &[usize], pos_labels: &'a [String]) -> &'a str {
    pos_labels
        .get(*pos_indices.get(idx).unwrap_or(&0))
        .map(String::as_str)
        .unwrap_or("")
}
