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
fn adversarial_score(prediction: &str) -> f64 {
    let lower = prediction.to_lowercase();
    let negation_phrases = [
        "no information",
        "not mentioned",
        "not available",
        "cannot be determined",
        "no evidence",
        "not in the conversation",
        "not discussed",
        "no record",
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
/// An evidence message is "found" if the first 50 characters of its text
/// appear as a substring in any recalled item's content.
///
/// Returns `(found, total)`.
pub fn evidence_hit(recalled_contents: &[&str], evidence_texts: &[String]) -> (usize, usize) {
    let total = evidence_texts.len();
    if total == 0 {
        return (0, 0);
    }

    let mut found = 0;
    for evidence in evidence_texts {
        let fingerprint = &evidence[..evidence.len().min(50)].to_lowercase();
        if recalled_contents
            .iter()
            .any(|rc| rc.to_lowercase().contains(fingerprint.as_str()))
        {
            found += 1;
        }
    }

    (found, total)
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
    let options = GenerationOptions {
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..Default::default()
    };

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
}
