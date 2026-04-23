//! Evaluation metrics for LongMemEval benchmark.
//!
//! Implements scoring methods specific to LongMemEval:
//! - context_contains_answer (Phase 1 retrieval metric)
//! - Recall@k and NDCG@k (retrieval quality)
//! - Category-specific GPT-4o judge prompts
//! - Abstention and knowledge-update scoring

use std::collections::HashSet;

use super::data::LmeQuestionType;

// ── Phase 1 Retrieval Metrics ───────────────────────────────────

/// Check whether the gold answer text appears in the top-k recalled items.
///
/// Performs case-insensitive substring matching after normalization.
/// This is the primary Phase 1 gate metric (target: ≥90% at k=5).
pub fn context_contains_answer(recalled_contents: &[&str], gold_answer: &str, k: usize) -> bool {
    let gold_normalized = normalize(gold_answer);
    if gold_normalized.is_empty() {
        return false;
    }

    recalled_contents.iter().take(k).any(|content| {
        let content_normalized = normalize(content);
        content_normalized.contains(&gold_normalized)
    })
}

/// Session-level Recall@k.
///
/// What fraction of `answer_session_ids` appear in the top-k
/// retrieved session IDs?
pub fn recall_at_k(
    retrieved_session_ids: &[String],
    answer_session_ids: &[String],
    k: usize,
) -> f64 {
    if answer_session_ids.is_empty() {
        return 0.0;
    }

    let answer_set: HashSet<&str> = answer_session_ids.iter().map(|s| s.as_str()).collect();
    let retrieved_top_k: HashSet<&str> = retrieved_session_ids
        .iter()
        .take(k)
        .map(|s| s.as_str())
        .collect();

    let found = answer_set.intersection(&retrieved_top_k).count();
    found as f64 / answer_set.len() as f64
}

/// NDCG@k for ranked retrieval quality.
///
/// Measures not just whether relevant sessions are retrieved, but
/// whether they appear at higher ranks (earlier positions).
pub fn ndcg_at_k(retrieved_session_ids: &[String], answer_session_ids: &[String], k: usize) -> f64 {
    if answer_session_ids.is_empty() {
        return 0.0;
    }

    let answer_set: HashSet<&str> = answer_session_ids.iter().map(|s| s.as_str()).collect();

    // DCG: sum of 1/log2(rank+1) for relevant items in top-k
    let dcg: f64 = retrieved_session_ids
        .iter()
        .take(k)
        .enumerate()
        .filter(|(_, sid)| answer_set.contains(sid.as_str()))
        .map(|(rank, _)| 1.0 / (rank as f64 + 2.0).log2())
        .sum();

    // Ideal DCG: best possible ranking (all relevant items at top)
    let ideal_dcg: f64 = (0..answer_set.len().min(k))
        .map(|rank| 1.0 / (rank as f64 + 2.0).log2())
        .sum();

    if ideal_dcg == 0.0 {
        0.0
    } else {
        dcg / ideal_dcg
    }
}

// ── Full Benchmark Scoring ──────────────────────────────────────

/// Abstention scoring: 1.0 if the prediction correctly refuses to answer.
pub fn abstention_score(prediction: &str) -> f64 {
    let lower = prediction.to_lowercase();
    let abstention_phrases = [
        "no information",
        "not mentioned",
        "not available",
        "cannot be determined",
        "no evidence",
        "not in the conversation",
        "not discussed",
        "no record",
        "don't have",
        "do not have",
        "cannot find",
        "unable to find",
        "unanswerable",
        "cannot answer",
    ];
    if abstention_phrases.iter().any(|p| lower.contains(p)) {
        1.0
    } else {
        0.0
    }
}

/// LLM-as-judge evaluation with category-specific prompts.
///
/// Returns 1.0 if the judge says the answer is correct, 0.0 otherwise.
/// Uses prompts matching the LongMemEval paper's evaluation methodology.
pub async fn lme_judge(
    kb: &uniko_store::KnowledgeBase,
    question: &str,
    gold_answer: &str,
    predicted_answer: &str,
    question_type: LmeQuestionType,
    is_abstention: bool,
    judge_alias: &str,
) -> anyhow::Result<f64> {
    use uni_db::xervo::{GenerationOptions, Message};

    let category_instruction = if is_abstention {
        "Does the model correctly identify the question as unanswerable? \
         The model should indicate that the information is not available \
         in the conversation history."
    } else {
        match question_type {
            LmeQuestionType::TemporalReasoning => {
                "Does the response contain the correct answer? \
                 Do not penalize off-by-one errors for number of days."
            }
            LmeQuestionType::KnowledgeUpdate => {
                "Is the response correct? The answer is correct as long as \
                 the updated answer is present, even if old information is \
                 also mentioned."
            }
            LmeQuestionType::SingleSessionPreference => {
                "Does the response correctly utilize the user's personal \
                 information? It does not need to hit all rubric points, \
                 as long as it recalls and uses the user's personal \
                 information correctly."
            }
            _ => {
                "Does the response contain the correct answer? \
                 If only a subset of required information is present, \
                 answer no."
            }
        }
    };

    let prompt = format!(
        "You are evaluating an answer to a question about a conversation history.\n\n\
         Question: {question}\n\
         Correct Answer: {gold_answer}\n\
         Generated Answer: {predicted_answer}\n\n\
         {category_instruction}\n\n\
         Respond with ONLY 'yes' or 'no'."
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
    Ok(if response.contains("yes") { 1.0 } else { 0.0 })
}

// ── Helpers ─────────────────────────────────────────────────────

/// Normalize text for comparison: lowercase, collapse whitespace.
fn normalize(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_contains_answer_basic() {
        let recalled = vec![
            "The user said their favorite color is blue.",
            "They also like green.",
        ];
        assert!(context_contains_answer(&recalled, "blue", 5));
        assert!(!context_contains_answer(&recalled, "red", 5));
    }

    #[test]
    fn test_context_contains_answer_case_insensitive() {
        let recalled = vec!["Alice lives in San Francisco."];
        assert!(context_contains_answer(&recalled, "san francisco", 5));
        assert!(context_contains_answer(&recalled, "San Francisco", 5));
    }

    #[test]
    fn test_context_contains_answer_top_k() {
        let recalled = vec!["first item", "second item", "the answer is blue"];
        assert!(!context_contains_answer(&recalled, "blue", 2));
        assert!(context_contains_answer(&recalled, "blue", 3));
    }

    #[test]
    fn test_recall_at_k() {
        let retrieved = vec!["s1".into(), "s2".into(), "s3".into(), "s4".into()];
        let answer = vec!["s2".into(), "s4".into()];

        assert!((recall_at_k(&retrieved, &answer, 5) - 1.0).abs() < 0.01);
        assert!((recall_at_k(&retrieved, &answer, 2) - 0.5).abs() < 0.01);
        assert!((recall_at_k(&retrieved, &answer, 1) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_ndcg_at_k() {
        // Perfect ranking: relevant item at rank 0
        let retrieved = vec!["s1".into(), "s2".into()];
        let answer = vec!["s1".into()];
        let score = ndcg_at_k(&retrieved, &answer, 5);
        assert!((score - 1.0).abs() < 0.01);

        // Imperfect: relevant item at rank 1
        let retrieved = vec!["s2".into(), "s1".into()];
        let score = ndcg_at_k(&retrieved, &answer, 5);
        assert!(score < 1.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_abstention_score() {
        assert!((abstention_score("I don't have that information.") - 1.0).abs() < 0.01);
        assert!((abstention_score("The answer is 42.") - 0.0).abs() < 0.01);
        assert!(
            (abstention_score("This information is not mentioned in the conversation.") - 1.0)
                .abs()
                < 0.01
        );
    }
}
