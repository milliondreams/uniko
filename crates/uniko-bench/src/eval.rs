//! Evaluation metrics for LoCoMo benchmark.
//!
//! Implements two scoring methods:
//! 1. Token-level F1 (original LoCoMo metric) with Porter stemming
//! 2. LLM-as-judge binary scoring

use std::collections::HashMap;

use rust_stemmers::{Algorithm, Stemmer};

use crate::data::QuestionCategory;

// ── Token-Level F1 (Original LoCoMo Metric) ──────────────────────

/// Compute the token-level F1 score for a prediction against gold.
///
/// Handles category-specific evaluation:
/// - Adversarial: binary (1.0 if prediction says "no information")
/// - Multi-hop: per-sub-answer F1 averaged
/// - All others: standard bag-of-words F1 with stemming
pub fn token_f1(prediction: &str, gold: &str, category: QuestionCategory) -> f64 {
    match category {
        QuestionCategory::Adversarial => adversarial_score(prediction),
        QuestionCategory::MultiHop => multi_hop_f1(prediction, gold),
        _ => single_f1(prediction, gold),
    }
}

/// Standard token-level F1 between prediction and gold answer.
fn single_f1(prediction: &str, gold: &str) -> f64 {
    let stemmer = Stemmer::create(Algorithm::English);
    let pred_tokens = tokenize_and_stem(prediction, &stemmer);
    let gold_tokens = tokenize_and_stem(gold, &stemmer);

    if pred_tokens.is_empty() || gold_tokens.is_empty() {
        return 0.0;
    }

    let pred_counts = bag_of_words(&pred_tokens);
    let gold_counts = bag_of_words(&gold_tokens);

    let common: usize = pred_counts
        .iter()
        .map(|(tok, &count)| count.min(*gold_counts.get(tok).unwrap_or(&0)))
        .sum();

    if common == 0 {
        return 0.0;
    }

    let precision = common as f64 / pred_tokens.len() as f64;
    let recall = common as f64 / gold_tokens.len() as f64;
    2.0 * precision * recall / (precision + recall)
}

/// Multi-hop F1: split gold by commas, per-sub-answer F1, average.
fn multi_hop_f1(prediction: &str, gold: &str) -> f64 {
    let sub_answers: Vec<&str> = gold
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if sub_answers.is_empty() {
        return single_f1(prediction, gold);
    }
    let sum: f64 = sub_answers
        .iter()
        .map(|sub| single_f1(prediction, sub))
        .sum();
    sum / sub_answers.len() as f64
}

/// Adversarial scoring: 1.0 if prediction indicates "no information".
///
/// Matches both past-participle ("not mentioned") and active-voice
/// ("does not mention") forms — earlier versions of the whitelist
/// missed verb-form abstentions, scoring otherwise-correct answers
/// like "The context does not mention any gift from grandma to
/// Melanie; it only describes a necklace that Caroline received" as
/// F1=0 confabulations.  See RFE rfe-p4-recall-evolution open-question
/// follow-up.
fn adversarial_score(prediction: &str) -> f64 {
    let lower = prediction.to_lowercase();
    let negation_phrases = [
        // Past-participle abstentions (canonical wording).
        "no information",
        "not mentioned",
        "not available",
        "cannot be determined",
        "no evidence",
        "not in the conversation",
        "not discussed",
        "no record",
        "not specified",
        "not stated",
        "not provided",
        // Verb-form abstentions (the model often says these instead).
        "does not mention",
        "doesn't mention",
        "does not specify",
        "doesn't specify",
        "does not state",
        "doesn't state",
        "does not provide",
        "doesn't provide",
        "no mention of",
        "no specific mention",
        // Defensive variations.
        "context does not",
        "is not present in",
        "is not specified",
        "not appear in",
        "cannot find",
        "no such information",
    ];
    if negation_phrases.iter().any(|p| lower.contains(p)) {
        1.0
    } else {
        0.0
    }
}

/// Normalize text for token comparison.
///
/// Lowercases, removes punctuation, removes articles (a, an, the, and),
/// and collapses whitespace.
fn normalize(text: &str) -> String {
    let lower = text.to_lowercase();
    let no_punct: String = lower
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    no_punct
        .split_whitespace()
        .filter(|w| !matches!(*w, "a" | "an" | "the" | "and"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Normalize, tokenize, and stem text.
fn tokenize_and_stem(text: &str, stemmer: &Stemmer) -> Vec<String> {
    normalize(text)
        .split_whitespace()
        .map(|w| stemmer.stem(w).to_string())
        .collect()
}

/// Count token occurrences (bag of words).
fn bag_of_words(tokens: &[String]) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for tok in tokens {
        *counts.entry(tok.clone()).or_insert(0) += 1;
    }
    counts
}

// ── Evidence Hit Rate ────────────────────────────────────────────

/// Check how many evidence messages appear in the recalled items.
///
/// Two-stage match:
/// 1. Each evidence message is "found" if a significant substring of its
///    text appears in any recalled item's content. Uses multiple
///    fingerprints to handle speaker prefixes (e.g., `"Jon: message..."`).
/// 2. Fallback: if zero evidence items matched but the `gold_answer`
///    appears at a word boundary in some recalled chunk, count one hit.
///    Catches the case where recall surfaces a paraphrased mention of
///    the answer rather than the LoCoMo-tagged evidence message.
///
/// Returns `(found, total)`.
pub fn evidence_hit(
    recalled_contents: &[&str],
    evidence_texts: &[String],
    gold_answer: &str,
) -> (usize, usize) {
    let total = evidence_texts.len();
    if total == 0 {
        return (0, 0);
    }

    let mut found = 0;
    for evidence in evidence_texts {
        let ev_lower = evidence.to_lowercase();
        // Try multiple fingerprint positions to handle prefixed chunks.
        // Skip short words at the start that might be speaker names.
        // Use floor_char_boundary to avoid slicing inside multi-byte chars.
        let end0 = ev_lower.floor_char_boundary(50);
        let fingerprints: Vec<&str> = vec![
            &ev_lower[..end0],
            // Skip first word (might be a speaker prefix in the chunk)
            ev_lower
                .find(' ')
                .map(|i| {
                    let start = i + 1;
                    let end = ev_lower.floor_char_boundary(start + 50);
                    &ev_lower[start..end]
                })
                .unwrap_or(""),
            // Use a chunk from the middle for longer texts
            if ev_lower.len() > 30 {
                let start = ev_lower.ceil_char_boundary(10);
                let end = ev_lower.floor_char_boundary(60);
                &ev_lower[start..end]
            } else {
                ""
            },
        ];

        if recalled_contents.iter().any(|rc| {
            let rc_lower = rc.to_lowercase();
            fingerprints
                .iter()
                .any(|fp| !fp.is_empty() && rc_lower.contains(*fp))
        }) {
            found += 1;
        }
    }

    // Fallback: gold answer present in retrieved chunks counts as one hit.
    if found == 0 && gold_answer_in_retrieved(gold_answer, recalled_contents) {
        found = 1;
    }

    (found, total)
}

/// Whether `gold_answer` appears at a word boundary in any recalled chunk.
///
/// Word-boundary matching avoids spurious hits where the gold answer is
/// a substring of a longer unrelated word (e.g. `"ice"` inside `"rice"`).
/// Skips gold answers shorter than 4 characters to bound false positives.
fn gold_answer_in_retrieved(gold_answer: &str, recalled_contents: &[&str]) -> bool {
    let needle = gold_answer.trim().to_lowercase();
    if needle.len() < 4 {
        return false;
    }
    recalled_contents
        .iter()
        .any(|rc| contains_at_word_boundary(&rc.to_lowercase(), &needle))
}

/// Case-insensitive substring search with word boundaries on both sides.
///
/// Both inputs are expected to already be lowercased. A boundary is any
/// non-alphanumeric character (or string edge).
fn contains_at_word_boundary(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let bytes = haystack.as_bytes();
    let n = needle.len();
    let mut start = 0;
    while let Some(idx) = haystack[start..].find(needle) {
        let abs = start + idx;
        let before_ok = abs == 0
            || !(bytes[abs - 1] as char).is_ascii_alphanumeric();
        let after = abs + n;
        let after_ok = after == bytes.len()
            || !(bytes[after] as char).is_ascii_alphanumeric();
        if before_ok && after_ok {
            return true;
        }
        start = abs + 1;
    }
    false
}

// ── LLM-as-Judge ─────────────────────────────────────────────────

/// Run LLM-as-judge evaluation on a single question.
///
/// Returns 1.0 if the judge says "correct", 0.0 otherwise.
///
/// # Errors
///
/// Returns an error if LLM generation fails.
pub async fn llm_judge(
    kb: &uniko_store::KnowledgeBase,
    question: &str,
    gold_answer: &str,
    predicted_answer: &str,
    judge_alias: &str,
) -> anyhow::Result<f64> {
    use uni_db::xervo::{GenerationOptions, Message};

    let prompt = format!(
        "You are evaluating an answer to a question about a conversation.\n\n\
         Question: {question}\n\
         Correct Answer: {gold_answer}\n\
         Generated Answer: {predicted_answer}\n\n\
         Is the generated answer correct? It doesn't need to match word-for-word, \
         but it must convey the same core information as the correct answer.\n\
         Respond with ONLY the word 'correct' or 'wrong'."
    );

    let messages = vec![Message::user(&prompt)];
    // Use provider defaults: GPT-5 rejects `max_tokens` (requires
    // `max_completion_tokens`) and `temperature: 0` (only accepts the
    // default 1.0). The judge prompt constrains the response to a
    // single word, so neither parameter is load-bearing.
    let options = GenerationOptions::default();

    let result = kb
        .db()
        .xervo()
        .generate(judge_alias, &messages, options)
        .await?;
    let response = result.text.trim().to_lowercase();
    Ok(if response.contains("correct") {
        1.0
    } else {
        0.0
    })
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_f1_exact_match() {
        assert!((single_f1("mental health", "mental health") - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_single_f1_partial() {
        let f1 = single_f1("mental health awareness", "mental health");
        assert!(f1 > 0.5);
        assert!(f1 < 1.0);
    }

    #[test]
    fn test_single_f1_no_overlap() {
        assert!((single_f1("apple banana", "car truck")).abs() < 0.01);
    }

    #[test]
    fn test_adversarial_correct() {
        assert!(
            (adversarial_score("The information is not mentioned in the conversation.") - 1.0)
                .abs()
                < 0.01
        );
    }

    #[test]
    fn test_adversarial_wrong() {
        assert!((adversarial_score("She went to the hospital.")).abs() < 0.01);
    }

    #[test]
    fn test_multi_hop() {
        let f1 = multi_hop_f1("mental health and self-care", "mental health, self-care");
        assert!(f1 > 0.5);
    }

    #[test]
    fn test_normalize_removes_articles() {
        assert_eq!(normalize("The quick and a fox"), "quick fox");
    }

    // ── evidence_hit tests ─────────────────────────────────────

    #[test]
    fn test_evidence_hit_matches_evidence_text() {
        let chunks = vec!["Jon: I'm a huge fan of dance, especially contemporary"];
        let evidence = vec!["Jon: I'm a huge fan of dance, especially contemporary".to_string()];
        let (found, total) = evidence_hit(&chunks, &evidence, "Contemporary");
        assert_eq!((found, total), (1, 1));
    }

    #[test]
    fn test_evidence_hit_gold_answer_fallback() {
        // Evidence-text fingerprint won't match (different chunk),
        // but gold answer "Harry Potter" appears verbatim.
        let chunks = vec!["movie marathon with friends harry potter"];
        let evidence =
            vec!["John: I'm a huge Harry Potter fan, just rewatched the whole series".to_string()];
        let (found, total) = evidence_hit(&chunks, &evidence, "Harry Potter");
        assert_eq!((found, total), (1, 1), "gold answer fallback should hit");
    }

    #[test]
    fn test_evidence_hit_no_match() {
        let chunks = vec!["unrelated content about cooking"];
        let evidence = vec!["John: I'm a huge Harry Potter fan".to_string()];
        let (found, total) = evidence_hit(&chunks, &evidence, "Harry Potter");
        assert_eq!((found, total), (0, 1));
    }

    #[test]
    fn test_evidence_hit_word_boundary_substring_does_not_match() {
        // "ice" should NOT match inside "rice" or "price".
        let chunks = vec!["He bought rice at the price of $5"];
        let evidence = vec!["completely different evidence text not found in chunks".to_string()];
        let (found, total) = evidence_hit(&chunks, &evidence, "ice");
        assert_eq!((found, total), (0, 1), "short gold should not trigger");
    }

    #[test]
    fn test_evidence_hit_word_boundary_real_match() {
        let chunks = vec!["I love ice cream a lot"];
        let evidence = vec!["completely different evidence text not found in chunks".to_string()];
        let (found, total) = evidence_hit(&chunks, &evidence, "ice cream");
        assert_eq!((found, total), (1, 1), "multi-word gold matches with boundaries");
    }

    #[test]
    fn test_evidence_hit_short_gold_skipped() {
        // 3-char gold should not trigger fallback.
        let chunks = vec!["The cat sat on the mat"];
        let evidence = vec!["unrelated evidence string".to_string()];
        let (found, total) = evidence_hit(&chunks, &evidence, "cat");
        assert_eq!((found, total), (0, 1));
    }

    #[test]
    fn test_evidence_hit_partial_not_bumped() {
        // 1/2 evidence matched — gold-answer fallback should NOT bump.
        let chunks = vec!["First evidence text completely included Harry Potter"];
        let evidence = vec![
            "First evidence text completely included Harry Potter".to_string(),
            "Second evidence message that is missing entirely".to_string(),
        ];
        let (found, total) = evidence_hit(&chunks, &evidence, "Harry Potter");
        assert_eq!(
            (found, total),
            (1, 2),
            "fallback only applies when zero evidence matched"
        );
    }

    #[test]
    fn test_evidence_hit_gold_case_insensitive() {
        let chunks = vec!["Tim cheers for THE WOLVES every weekend"];
        let evidence = vec!["completely different evidence text not in chunks".to_string()];
        let (found, total) = evidence_hit(&chunks, &evidence, "the wolves");
        assert_eq!((found, total), (1, 1));
    }
}
