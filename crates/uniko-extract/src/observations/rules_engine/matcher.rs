//! Generic pattern matcher.
//!
//! Walks a DEP tree once per anchor candidate; for each pattern whose
//! anchor POS and required children all match, fills captures, renders
//! the template, applies the global filters, and emits a
//! [`DepObservation`].

use std::collections::{BTreeSet, HashMap};

use crate::ingest::context::SentenceContext;
use crate::nlp::decode::DepObservation;
use crate::nlp::types::{DepArc, SrlFrame};

use super::resolver::{
    ResolvedSubject, collect_subtree, collect_with_relations, resolve_subject, strip_trailing_punct,
    update_sentence_context,
};
use super::rules::{ChildSpec, Pattern, PhraseCollector, Rules, SrlPattern};
use super::template::render;

const SUBJECT_CAPTURE: &str = "subject";
const ANCHOR_CAPTURE: &str = "anchor";

/// Apply every pattern in `rules` to the sentence and return the
/// resulting observations. Runs both DEP-anchored patterns
/// (`rules.patterns`) over the dependency tree and SRL-anchored
/// patterns (`rules.srl_patterns`) over `srl_frames`. Output is
/// deduplicated by rendered text, so DEP and SRL paths producing the
/// same observation collapse to one entry. Updates `ctx` with
/// antecedents at the end (matching the legacy pipeline's behaviour).
#[allow(clippy::too_many_arguments)]
pub fn extract_with_rules(
    rules: &Rules,
    words: &[String],
    pos_indices: &[usize],
    dep_arcs: &[DepArc],
    pos_labels: &[String],
    srl_frames: &[SrlFrame],
    speaker: &str,
    ctx: &mut SentenceContext,
) -> Vec<DepObservation> {
    let mut out = Vec::new();
    let mut seen_text: BTreeSet<String> = BTreeSet::new();

    // DEP-anchored patterns: walk every word, test every pattern whose
    // anchor POS matches.
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

    // SRL-anchored patterns: every SrlFrame from the cascade × every
    // SrlPattern. Frames with no recognised arguments still get tested
    // (a pattern with all-optional captures may still render via its
    // {predicate} variable).
    for frame in srl_frames {
        for pattern in &rules.srl_patterns {
            if let Some(obs) = try_match_srl(pattern, frame, speaker, &rules.filters)
                && seen_text.insert(obs.content.clone())
            {
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

/// Match a single SRL frame against a single SRL pattern.
///
/// `{predicate}` is auto-populated from `frame.predicate_word`.
/// `{subject}` defaults to `speaker` so templates can always include
/// it without an explicit ARG0 capture.
///
/// Template selection: the renderer picks the longest of
/// `[template, ...template_alternatives]` whose referenced variables
/// are *all* non-empty. This lets one pattern emit the right
/// observation shape based on which optional ARGM-* roles fired.
fn try_match_srl(
    pattern: &SrlPattern,
    frame: &SrlFrame,
    speaker: &str,
    filters: &super::rules::Filters,
) -> Option<DepObservation> {
    let mut captures: HashMap<String, String> = HashMap::new();
    captures.insert("predicate".into(), frame.predicate_word.clone());
    captures.insert(SUBJECT_CAPTURE.into(), speaker.to_string());

    for cap in &pattern.captures {
        let arg_text = frame
            .args
            .iter()
            .find(|a| a.role == cap.role)
            .map(|a| a.text.clone());
        match arg_text {
            Some(text) => {
                captures.insert(cap.as_name.clone(), text);
            }
            None => {
                if cap.required {
                    return None;
                }
            }
        }
    }

    // Pick the most specific (longest non-empty) template. Walk
    // templates in declared order so the base `template` is the
    // fallback.
    let mut best: Option<String> = None;
    for tmpl in std::iter::once(&pattern.template).chain(pattern.template_alternatives.iter()) {
        let rendered = render(tmpl, &captures);
        if rendered.is_empty() {
            continue;
        }
        // Reject this rendering if any referenced template variable
        // resolved to empty (gives the alternatives a chance to
        // produce a cleaner shape).
        if template_has_unresolved_vars(tmpl, &captures) {
            continue;
        }
        match &best {
            Some(prev) if prev.split_whitespace().count() >= rendered.split_whitespace().count() => {}
            _ => best = Some(rendered),
        }
    }
    let content = best?;

    if content.split_whitespace().count() < filters.min_content_words {
        return None;
    }

    let subject = captures
        .get(SUBJECT_CAPTURE)
        .cloned()
        .unwrap_or_else(|| speaker.to_string());
    let predicate_raw = captures
        .get("predicate")
        .cloned()
        .unwrap_or_else(|| frame.predicate_word.clone());
    let predicate = crate::nlp::decode::normalize_predicate(&predicate_raw);
    // Object: prefer the explicit "object" capture; fall back to the
    // ARG1 frame argument (the canonical patient role).
    let object = captures
        .get("object")
        .cloned()
        .or_else(|| {
            frame
                .args
                .iter()
                .find(|a| a.role == "ARG1")
                .map(|a| a.text.clone())
        })
        .filter(|s| !s.trim().is_empty());
    Some(DepObservation {
        content,
        subject,
        predicate: (!predicate.is_empty()).then_some(predicate),
        object,
        confidence: 0.85,
    })
}

/// Returns `true` if `tmpl` contains a `{var}` whose value in
/// `captures` is missing or empty. Used by [`try_match_srl`] to skip
/// alternatives whose slots aren't filled by the current frame.
fn template_has_unresolved_vars(tmpl: &str, captures: &HashMap<String, String>) -> bool {
    let mut iter = tmpl.chars().peekable();
    while let Some(c) = iter.next() {
        if c != '{' {
            continue;
        }
        let mut name = String::new();
        for next in iter.by_ref() {
            if next == '}' {
                break;
            }
            name.push(next);
        }
        let val = captures.get(&name).map(String::as_str).unwrap_or("");
        if val.is_empty() {
            return true;
        }
    }
    false
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

    // Surface the (predicate, object) triple alongside content so P4
    // Consolidation can group observations by `(subject, predicate)`
    // without re-parsing.  Predicate = the matched anchor verb token
    // (already normalized); object = first non-empty among the
    // canonical capture names.
    let predicate = crate::nlp::decode::normalize_predicate(
        captures.get(ANCHOR_CAPTURE).map(String::as_str).unwrap_or(""),
    );
    let object = ["object", "obj", "target", "complement"]
        .iter()
        .find_map(|key| captures.get(*key).cloned())
        .filter(|s| !s.trim().is_empty());
    Some(DepObservation {
        content,
        subject,
        predicate: (!predicate.is_empty()).then_some(predicate),
        object,
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
