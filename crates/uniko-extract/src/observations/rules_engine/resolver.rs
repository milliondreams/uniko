//! Subject resolution + sentence-context update — driven by the YAML
//! `subject_resolution` vocabulary, but the lookup logic against
//! [`SentenceContext`] stays in Rust because it isn't pattern-shaped.

use crate::ingest::context::SentenceContext;
use crate::nlp::types::DepArc;

use super::rules::SubjectResolution;

/// Outcome of resolving a captured subject phrase.
pub enum ResolvedSubject {
    /// Pronoun was resolved to a concrete name (or wasn't a pronoun).
    Resolved(String),
    /// Pronoun is in the unresolvable vocabulary, or third-person with
    /// no antecedent. Caller should drop the observation.
    Unresolvable,
}

/// Resolve a captured subject phrase against the current sentence context.
///
/// Mirrors the behaviour of the legacy `resolve_subject` /
/// `is_unresolvable_pronoun` pair — `Unresolvable` covers both the
/// hardcoded relative/indefinite list and the case where a third-person
/// demonstrative finds no antecedent in `ctx`.
pub fn resolve_subject(
    text: &str,
    pos: &str,
    ctx: &SentenceContext,
    speaker: &str,
    config: &SubjectResolution,
) -> ResolvedSubject {
    let lower = text.to_lowercase();

    if config.first_person.matches(&lower) {
        return ResolvedSubject::Resolved(speaker.to_string());
    }

    if config.second_person.matches(&lower) {
        if let Some(other) = ctx.other_speakers.first() {
            return ResolvedSubject::Resolved(other.clone());
        }
        // Second-person without an addressee → drop. The legacy code
        // would have left "you" in the output; that text fails the
        // unresolvable check below, so behaviour matches.
        return ResolvedSubject::Unresolvable;
    }

    if pos == "PRON" && config.third_person_demonstrative.matches(&lower) {
        if let Some(ref obj) = ctx.last_noun_object {
            return ResolvedSubject::Resolved(obj.clone());
        }
        if let Some(ref subj) = ctx.last_noun_subject {
            return ResolvedSubject::Resolved(subj.clone());
        }
        return ResolvedSubject::Unresolvable;
    }

    // Not a pronoun (or pronoun not in any vocabulary): keep the literal
    // text, but reject it if it falls in the explicit unresolvable list
    // (`who`, `which`, `others`, …).
    if config.unresolvable.matches(&lower) {
        return ResolvedSubject::Unresolvable;
    }
    ResolvedSubject::Resolved(text.to_string())
}

/// Update the sentence context with NOUN/PROPN antecedents from this
/// sentence. Ported verbatim from `decode::update_sentence_context`.
pub fn update_sentence_context(
    ctx: &mut SentenceContext,
    words: &[String],
    pos_indices: &[usize],
    dep_arcs: &[DepArc],
    pos_labels: &[String],
    noun_phrase_relations: &[String],
) {
    for arc in dep_arcs {
        let dep_pos = pos_labels
            .get(*pos_indices.get(arc.dependent).unwrap_or(&0))
            .map(String::as_str)
            .unwrap_or("");

        if dep_pos != "NOUN" && dep_pos != "PROPN" {
            continue;
        }

        let phrase = collect_with_relations(arc.dependent, words, dep_arcs, noun_phrase_relations);
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

/// Local copy of the relation-filtered BFS used by both the resolver
/// (for context updates) and the matcher (for noun-phrase captures).
pub fn collect_with_relations(
    root: usize,
    words: &[String],
    dep_arcs: &[DepArc],
    relations: &[String],
) -> String {
    let mut indices = vec![root];
    let mut queue = vec![root];
    while let Some(head) = queue.pop() {
        for arc in dep_arcs {
            if arc.head == head
                && !indices.contains(&arc.dependent)
                && relations.iter().any(|r| r == &arc.relation)
            {
                indices.push(arc.dependent);
                queue.push(arc.dependent);
            }
        }
    }
    indices.sort_unstable();
    indices
        .iter()
        .filter_map(|&i| words.get(i))
        .map(|w| strip_trailing_punct(w))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn collect_subtree(root: usize, words: &[String], dep_arcs: &[DepArc]) -> String {
    let mut indices = vec![root];
    let mut queue = vec![root];
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

pub fn strip_trailing_punct(s: &str) -> String {
    s.trim_end_matches(|c: char| c.is_ascii_punctuation() && c != '\'')
        .to_string()
}
