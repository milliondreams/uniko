//! Recall and answer generation for benchmark questions.

use std::time::Instant;

use anyhow::Result;
use uniko_memory::recall::{ContextBundle, RecallConfig, recall};
use uniko_store::KnowledgeBase;

use crate::data::{QaPair, QuestionCategory};

/// One retrieved item: node type, score, full content. Surfaced
/// per-question for offline diagnostics (which chunks/messages/obs
/// were actually pulled, in what order, with what scores).
#[derive(Debug, Clone, serde::Serialize)]
pub struct RecalledItem {
    pub node_id: i64,
    pub node_type: String,
    pub tier: String,
    pub score: f64,
    pub content: String,
}

/// Result of querying a single question.
#[derive(Debug)]
pub struct QueryResult {
    /// Original question text.
    pub question: String,
    /// Gold answer from the dataset.
    pub gold_answer: String,
    /// Model-generated or retrieval-based answer.
    pub predicted_answer: String,
    /// Question category.
    pub category: QuestionCategory,
    /// Number of recall items retrieved.
    pub recall_items: usize,
    /// Full bundle dump (every item the recall layer surfaced, in
    /// score-descending order). Diagnostics only; large.
    pub recall_bundle: Vec<RecalledItem>,
    /// Evidence messages found / total evidence messages.
    pub evidence_found: usize,
    /// Total evidence messages for this question.
    pub evidence_total: usize,
    /// Recall phase latency in milliseconds.
    pub recall_latency_ms: u64,
    /// Answer generation latency in milliseconds (0 if retrieval-only).
    pub generation_latency_ms: u64,
}

/// Run recall for a question and optionally generate an answer.
///
/// In retrieval-only mode, the "predicted answer" is the concatenated
/// content of the top recall items.
pub async fn run_query(
    kb: &KnowledgeBase,
    qa: &QaPair,
    evidence_texts: &[String],
    llm_alias: Option<&str>,
) -> Result<QueryResult> {
    // Pull recall settings from the KB's UnikoConfig so the bench
    // honors `--reranker` and other recall overrides set on the
    // ingest config (rather than always using `RecallConfig::default()`).
    let recall_config = RecallConfig::from_uniko_config(kb.config());

    // Run recall.
    let recall_start = Instant::now();
    let bundle = recall(kb, &qa.question, &recall_config).await?;
    let recall_latency_ms = recall_start.elapsed().as_millis() as u64;

    // Compute evidence hit rate.
    let recalled_contents: Vec<&str> = bundle.items.iter().map(|i| i.content.as_str()).collect();
    let (evidence_found, evidence_total) =
        crate::eval::evidence_hit(&recalled_contents, evidence_texts, qa.gold_answer());

    // Generate answer.
    let gen_start = Instant::now();
    let predicted_answer = if let Some(alias) = llm_alias {
        generate_answer(kb, &bundle, &qa.question, alias).await?
    } else {
        uniko_bench::retrieval_answer(&bundle, 5)
    };
    let generation_latency_ms = gen_start.elapsed().as_millis() as u64;

    let recall_bundle = bundle
        .items
        .iter()
        .map(|it| RecalledItem {
            node_id: it.node_id,
            node_type: it.node_type.clone(),
            tier: format!("{:?}", it.tier),
            score: it.score,
            content: it.content.clone(),
        })
        .collect();

    Ok(QueryResult {
        question: qa.question.clone(),
        gold_answer: qa.gold_answer().to_string(),
        predicted_answer,
        category: qa.question_category(),
        evidence_found,
        evidence_total,
        recall_items: bundle.items.len(),
        recall_bundle,
        recall_latency_ms,
        generation_latency_ms,
    })
}

/// Generate an answer using the LLM from the recall context.
async fn generate_answer(
    kb: &KnowledgeBase,
    bundle: &ContextBundle,
    question: &str,
    llm_alias: &str,
) -> Result<String> {
    use uni_db::xervo::{GenerationOptions, Message};

    let context = uniko_bench::format_context(bundle);

    let system = "You are a helpful assistant answering questions about conversations. \
        Use ONLY the provided context to answer. If the information is not available \
        in the context, say 'The information is not mentioned in the conversation.' \
        Answer concisely in one or two sentences.";

    let user = format!("Context:\n{context}\n\nQuestion: {question}\n\nAnswer:");

    let messages = vec![Message::system(system), Message::user(&user)];
    // Gemma 4 (and other reasoning models) emit chain-of-thought into a
    // separate `reasoning_content` field that consumes the token budget.
    // 2048 leaves headroom for ~1500 reasoning tokens + a 1-2 sentence
    // answer. For non-reasoning models this is just an upper bound;
    // they self-terminate at the EOS well before hitting it.
    let options = GenerationOptions {
        max_tokens: Some(2048),
        temperature: Some(0.1),
        ..Default::default()
    };

    let result = kb
        .db()
        .xervo()
        .generate(llm_alias, &messages, options)
        .await?;
    Ok(result.text.trim().to_string())
}
