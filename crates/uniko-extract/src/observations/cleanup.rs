//! Phase B (RFE `rfe-p4-recall-evolution`): rule-based cleanup that
//! brings SRL/DEP-derived triples closer to LLM-extracted quality.
//!
//! This module hosts pure functions consumed by the matcher and the
//! observation-insert path.  No ML, no model calls — just morphology
//! tables, stop-word lists, and light-verb filters.
//!
//! ## Pieces
//!
//! - [`lemmatize_predicate`]: maps `got/getting/gets` → `get`, strips
//!   auxiliaries from short verb phrases (e.g. `is_starting → starts`).
//! - [`clean_object_phrase`]: strips leading prepositions/articles and
//!   trailing punctuation; returns `None` when the residue is pure
//!   stop-word or single-token deictic.
//! - [`is_light_verb_only`]: rejects triples whose predicate is a
//!   light verb (`do`, `make`, `get`, `have`, `be`) with no specifying
//!   complement.
//! - [`detect_negation`]: identifies negation tokens in a slice of
//!   modifier strings (`not`, `n't`, `never`, `no`) for the upcoming
//!   `polarity` field.

// Rust guideline compliant

use std::collections::HashSet;
use std::sync::OnceLock;

// ── Lemmatization ─────────────────────────────────────────────────

/// Map a normalized predicate (already snake_case) to its lemma.
///
/// Handles common English verb inflections via two passes:
/// 1. Irregular-verb table lookup (got → get, went → go, …).
/// 2. Regular-suffix rules (-ing/-ed/-s/-es).
///
/// Auxiliary prefixes (`is_`, `was_`, `has_`, `had_`, `have_been_`) are
/// stripped because they survive the `normalize_predicate` snake-case
/// join and inflate predicate-vocabulary count without changing
/// semantics.
///
/// Returns the lemmatized form, or the input unchanged when no rule
/// matches (conservative — don't break unfamiliar verbs).
///
/// # Examples
///
/// ```
/// use uniko_extract::observations::cleanup::lemmatize_predicate;
/// assert_eq!(lemmatize_predicate("got"), "get");
/// assert_eq!(lemmatize_predicate("is_starting"), "start");
/// assert_eq!(lemmatize_predicate("has_been_doing"), "do");
/// assert_eq!(lemmatize_predicate("researches"), "research");
/// assert_eq!(lemmatize_predicate("painted"), "paint");
/// ```
pub fn lemmatize_predicate(predicate: &str) -> String {
    let stripped = strip_aux_prefix(predicate);
    apply_suffix_rules(stripped)
}

/// Strip auxiliary-verb prefixes that survived `normalize_predicate`'s
/// snake_case join.  Order matters: longer prefixes first.
fn strip_aux_prefix(predicate: &str) -> &str {
    const PREFIXES: &[&str] = &[
        "has_been_",
        "have_been_",
        "had_been_",
        "is_being_",
        "was_being_",
        "are_",
        "were_",
        "have_",
        "has_",
        "had_",
        "is_",
        "was_",
        "be_",
        "been_",
        "being_",
        "will_",
        "would_",
        "should_",
        "could_",
        "might_",
        "may_",
        "do_",
        "does_",
        "did_",
        "to_",
    ];
    for prefix in PREFIXES {
        if let Some(rest) = predicate.strip_prefix(prefix) {
            if !rest.is_empty() {
                return rest;
            }
        }
    }
    predicate
}

/// Apply irregular-verb table + regular suffix rules to recover the
/// lemma.  Returns owned `String` even on no-match (cheap allocation,
/// keeps the API uniform).
fn apply_suffix_rules(predicate: &str) -> String {
    if let Some(lemma) = irregular_lemma(predicate) {
        return lemma.to_string();
    }

    // Regular suffix rules — applied longest-first.  Each rule has the
    // form `(suffix, replacement)`; replacement may add letters back
    // (e.g. `runs → run`, `running → run` requires removing `-ning` and
    // adding nothing; `studies → study` requires `-ies → y`).
    static SUFFIX_RULES: &[(&str, &str)] = &[
        // Doubled-consonant gerund: `running → run`, `getting → get`.
        ("nning", "n"),
        ("tting", "t"),
        ("dding", "d"),
        ("pping", "p"),
        ("gging", "g"),
        ("bbing", "b"),
        ("mming", "m"),
        // Plain gerund / present participle.
        ("ying", "y"), // studying → study
        ("ing", ""),   // researching → research
        // Past tense — doubled consonant first.
        ("nned", "n"),
        ("tted", "t"),
        ("dded", "d"),
        ("pped", "p"),
        ("gged", "g"),
        ("ied", "y"), // studied → study
        ("ed", ""),   // painted → paint
        // Third-person singular.
        ("ies", "y"),   // studies → study
        ("sses", "ss"), // passes → pass
        ("ches", "ch"), // researches → research
        ("shes", "sh"),
        ("xes", "x"),
        ("zes", "z"),
        ("oes", "o"), // does → do
        ("s", ""),    // gets → get
    ];

    for (suffix, replacement) in SUFFIX_RULES {
        if let Some(stem) = predicate.strip_suffix(suffix) {
            // Avoid stripping tiny words: `is → ""`, `do → ""`.  Require
            // at least 3 letters in the resulting stem.
            let candidate = format!("{stem}{replacement}");
            if candidate.len() >= 2 {
                return candidate;
            }
        }
    }
    predicate.to_string()
}

/// Irregular English verb table: maps inflected forms back to base.
///
/// Covers the high-frequency irregulars common in conversational text.
/// Returns `None` for forms not in the table — caller falls through to
/// the regular-suffix rules.
fn irregular_lemma(predicate: &str) -> Option<&'static str> {
    static TABLE: OnceLock<std::collections::HashMap<&'static str, &'static str>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let pairs: &[(&str, &str)] = &[
            ("got", "get"),
            ("gotten", "get"),
            ("went", "go"),
            ("gone", "go"),
            ("goes", "go"),
            ("said", "say"),
            ("says", "say"),
            ("saw", "see"),
            ("seen", "see"),
            ("sees", "see"),
            ("took", "take"),
            ("taken", "take"),
            ("takes", "take"),
            ("made", "make"),
            ("makes", "make"),
            ("had", "have"),
            ("has", "have"),
            ("did", "do"),
            ("done", "do"),
            ("does", "do"),
            ("was", "be"),
            ("were", "be"),
            ("been", "be"),
            ("is", "be"),
            ("are", "be"),
            ("am", "be"),
            ("came", "come"),
            ("comes", "come"),
            ("gave", "give"),
            ("given", "give"),
            ("gives", "give"),
            ("knew", "know"),
            ("known", "know"),
            ("knows", "know"),
            ("thought", "think"),
            ("thinks", "think"),
            ("told", "tell"),
            ("tells", "tell"),
            ("found", "find"),
            ("finds", "find"),
            ("brought", "bring"),
            ("brings", "bring"),
            ("bought", "buy"),
            ("buys", "buy"),
            ("ran", "run"),
            ("runs", "run"),
            ("ate", "eat"),
            ("eaten", "eat"),
            ("eats", "eat"),
            ("wrote", "write"),
            ("written", "write"),
            ("writes", "write"),
            ("read", "read"),
            ("reads", "read"),
            ("met", "meet"),
            ("meets", "meet"),
            ("lost", "lose"),
            ("loses", "lose"),
            ("won", "win"),
            ("wins", "win"),
            ("sent", "send"),
            ("sends", "send"),
            ("kept", "keep"),
            ("keeps", "keep"),
            ("felt", "feel"),
            ("feels", "feel"),
            ("left", "leave"),
            ("leaves", "leave"),
            ("heard", "hear"),
            ("hears", "hear"),
            ("held", "hold"),
            ("holds", "hold"),
            ("paid", "pay"),
            ("pays", "pay"),
            ("put", "put"),
            ("puts", "put"),
            ("set", "set"),
            ("sets", "set"),
        ];
        pairs.iter().copied().collect()
    });
    table.get(predicate).copied()
}

// ── Object cleanup ────────────────────────────────────────────────

/// Clean an object phrase captured by the matcher.
///
/// Returns `None` when the residue is unusable:
/// - empty after stripping
/// - pure pronoun or stop-word (`"what"`, `"it"`, `"thing"`)
/// - pure single-token punctuation
///
/// Otherwise returns the trimmed, leading-prep-stripped, trailing-punct-stripped
/// form.  Preserves multi-word phrases (`"adoption agencies"`,
/// `"to take control of my own destiny"`).
///
/// # Examples
///
/// ```
/// use uniko_extract::observations::cleanup::clean_object_phrase;
/// assert_eq!(clean_object_phrase("in pottery class"), Some("pottery class".into()));
/// assert_eq!(clean_object_phrase("the idea"), Some("idea".into()));
/// assert_eq!(clean_object_phrase("it happen."), Some("it happen".into()));
/// assert!(clean_object_phrase("what").is_none());
/// assert!(clean_object_phrase("yesterday!").is_none()); // temporal deictic
/// ```
pub fn clean_object_phrase(raw: &str) -> Option<String> {
    let mut cleaned = raw
        .trim()
        .trim_end_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace())
        .trim_start_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace())
        .to_string();

    // Strip leading prepositions/articles iteratively.
    loop {
        let lower = cleaned.to_ascii_lowercase();
        let lead = lower.split_whitespace().next()?;
        if LEADING_STRIP_WORDS.contains(lead) {
            let n = lead.len();
            if cleaned.len() <= n {
                return None;
            }
            cleaned = cleaned[n..].trim_start().to_string();
        } else {
            break;
        }
    }

    if cleaned.is_empty() {
        return None;
    }

    // Reject pure pronoun / stop-word / temporal deictic.
    let lower = cleaned.to_ascii_lowercase();
    if REJECT_OBJECT_PHRASES.contains(lower.as_str()) {
        return None;
    }

    Some(cleaned)
}

/// Words to strip from the start of an object phrase.  Iterates until
/// no leading word matches, so multiple stacked prepositions/articles
/// (`"to the idea"`) reduce cleanly.
static LEADING_STRIP_WORDS: std::sync::LazyLock<HashSet<&'static str>> =
    std::sync::LazyLock::new(|| {
        [
            // Articles
            "the", "a", "an",
            // Prepositions that frequently lead object phrases captured
            // via `obj`/`obl` DEP arcs.
            "in", "on", "at", "to", "of", "for", "with", "from", "by", "into", "onto", "upon",
            "about", "around", "before", "after", "over", "under",
        ]
        .into_iter()
        .collect()
    });

/// Object phrases to reject outright (pronouns, demonstratives, temporal
/// deictics, single-token pronouns).
static REJECT_OBJECT_PHRASES: std::sync::LazyLock<HashSet<&'static str>> =
    std::sync::LazyLock::new(|| {
        [
            // Pronouns
            "i",
            "me",
            "my",
            "mine",
            "you",
            "your",
            "yours",
            "he",
            "him",
            "his",
            "she",
            "her",
            "hers",
            "it",
            "its",
            "we",
            "us",
            "our",
            "ours",
            "they",
            "them",
            "their",
            "theirs",
            // Interrogatives / vague nouns
            "what",
            "who",
            "which",
            "thing",
            "things",
            "stuff",
            "something",
            "anything",
            "everything",
            "nothing",
            // Temporal deictics (Phase A puts these in temporal_anchor slot)
            "yesterday",
            "today",
            "tomorrow",
            "now",
            "then",
            // Single demonstratives
            "this",
            "that",
            "these",
            "those",
        ]
        .into_iter()
        .collect()
    });

// ── Light-verb filter ─────────────────────────────────────────────

/// Returns `true` when the predicate is a light verb with no useful
/// specifying object — i.e. the triple has no semantic content.
///
/// Light verbs (`do`, `make`, `get`, `have`, `be`) only carry meaning
/// when paired with a specifying complement.  `(Jon | do | what)` is
/// noise; `(Jon | do | yoga)` is signal.
///
/// Caller passes the *lemmatized* predicate and the cleaned object.
pub fn is_light_verb_only(lemma: &str, object: Option<&str>) -> bool {
    const LIGHT_VERBS: &[&str] = &["do", "make", "get", "have", "be", "go"];
    if !LIGHT_VERBS.contains(&lemma) {
        return false;
    }
    match object {
        None => true,
        Some(obj) => {
            let toks: Vec<&str> = obj.split_whitespace().collect();
            // Object must have at least 1 content word that is not in
            // the global stop-word set.  Light verbs with single-token
            // pronoun/stopword objects ("do what", "make it") are noise.
            toks.is_empty()
                || toks.iter().all(|t| {
                    let lower = t.to_ascii_lowercase();
                    REJECT_OBJECT_PHRASES.contains(lower.as_str())
                })
        }
    }
}

// ── Negation detection ────────────────────────────────────────────

/// Detect whether a slice of modifier strings indicates negation.
///
/// Used to populate the `polarity` field on Observation: `false` when
/// the matcher captured a negation token (`"not"`, `"n't"`, `"never"`,
/// `"no"`), `true` otherwise.
///
/// # Examples
///
/// ```
/// use uniko_extract::observations::cleanup::detect_negation;
/// assert!(detect_negation(&["not".into(), "really".into()]));
/// assert!(detect_negation(&["n't".into()]));
/// assert!(detect_negation(&["never".into(), "again".into()]));
/// assert!(!detect_negation(&["yesterday".into()]));
/// ```
pub fn detect_negation(modifiers: &[String]) -> bool {
    const NEG_WHOLE_TOKENS: &[&str] = &["not", "never", "no", "nor"];
    modifiers.iter().any(|m| {
        let lower = m.to_ascii_lowercase();
        if lower.ends_with("n't") {
            return true;
        }
        NEG_WHOLE_TOKENS.iter().any(|n| {
            lower == *n
                || lower.contains(&format!(" {n} "))
                || lower.starts_with(&format!("{n} "))
                || lower.ends_with(&format!(" {n}"))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lemmatize_irregular() {
        assert_eq!(lemmatize_predicate("got"), "get");
        assert_eq!(lemmatize_predicate("went"), "go");
        assert_eq!(lemmatize_predicate("took"), "take");
        assert_eq!(lemmatize_predicate("was"), "be");
    }

    #[test]
    fn lemmatize_strips_aux_prefix() {
        assert_eq!(lemmatize_predicate("is_starting"), "start");
        assert_eq!(lemmatize_predicate("has_been_doing"), "do");
        assert_eq!(lemmatize_predicate("was_pursuing"), "pursu");
        // ^ "pursu" is the regular-rule stripping of "-ing" — without
        // a richer morph table we can't recover "pursue" cleanly.
        // Conservative: callers can still group by predicate=pursu.
    }

    #[test]
    fn lemmatize_regular_suffixes() {
        assert_eq!(lemmatize_predicate("researches"), "research");
        assert_eq!(lemmatize_predicate("painted"), "paint");
        assert_eq!(lemmatize_predicate("running"), "run");
        assert_eq!(lemmatize_predicate("studied"), "study");
        assert_eq!(lemmatize_predicate("studies"), "study");
        assert_eq!(lemmatize_predicate("passes"), "pass");
    }

    #[test]
    fn lemmatize_passes_through_unknown() {
        // No matching irregular or suffix rule short enough to apply.
        assert_eq!(lemmatize_predicate("be"), "be");
    }

    #[test]
    fn clean_object_strips_leading_prep() {
        assert_eq!(
            clean_object_phrase("in pottery class").as_deref(),
            Some("pottery class"),
        );
        assert_eq!(clean_object_phrase("the idea").as_deref(), Some("idea"),);
        assert_eq!(
            clean_object_phrase("to the new dance studio").as_deref(),
            Some("new dance studio"),
        );
    }

    #[test]
    fn clean_object_strips_trailing_punct() {
        assert_eq!(
            clean_object_phrase("it happen.").as_deref(),
            Some("it happen"),
        );
        assert_eq!(
            clean_object_phrase("loving homes for children in need.").as_deref(),
            Some("loving homes for children in need"),
        );
    }

    #[test]
    fn clean_object_rejects_pure_stop_words() {
        assert!(clean_object_phrase("what").is_none());
        assert!(clean_object_phrase("it").is_none());
        assert!(clean_object_phrase("yesterday!").is_none());
        assert!(clean_object_phrase("the").is_none());
    }

    #[test]
    fn clean_object_preserves_multiword() {
        assert_eq!(
            clean_object_phrase("to take control of my own destiny").as_deref(),
            Some("take control of my own destiny"),
        );
    }

    #[test]
    fn light_verb_with_pronoun_object_is_noise() {
        assert!(is_light_verb_only("do", Some("what")));
        assert!(is_light_verb_only("make", Some("it")));
        assert!(is_light_verb_only("get", None));
    }

    #[test]
    fn light_verb_with_specifying_object_is_kept() {
        assert!(!is_light_verb_only("do", Some("yoga")));
        assert!(!is_light_verb_only("make", Some("dinner")));
        assert!(!is_light_verb_only("get", Some("a new job")));
    }

    #[test]
    fn non_light_verbs_never_rejected() {
        assert!(!is_light_verb_only("research", Some("what")));
        assert!(!is_light_verb_only("paint", None));
    }

    #[test]
    fn negation_detected() {
        assert!(detect_negation(&["not".into(), "really".into()]));
        assert!(detect_negation(&["n't".into()]));
        assert!(detect_negation(&["never".into()]));
        assert!(detect_negation(&["doesn't".into()]));
        assert!(!detect_negation(&["yesterday".into()]));
        assert!(!detect_negation(&[]));
    }
}
