//! Recall and answer generation for benchmark questions.

use std::fmt::Write;
use std::time::Instant;

use anyhow::Result;
use uniko_memory::recall::{ContextBundle, RecallConfig, recall};
use uniko_store::KnowledgeBase;

use crate::data::{QaPair, QuestionCategory};

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
    let recall_config = RecallConfig::default();

    // Run recall.
    let recall_start = Instant::now();
    let bundle = recall(kb, &qa.question, &recall_config).await?;
    let recall_latency_ms = recall_start.elapsed().as_millis() as u64;

    // Compute evidence hit rate.
    let recalled_contents: Vec<&str> = bundle.items.iter().map(|i| i.content.as_str()).collect();
    let (evidence_found, evidence_total) =
        crate::eval::evidence_hit(&recalled_contents, evidence_texts);

    // Generate answer.
    let gen_start = Instant::now();
    let predicted_answer = if let Some(alias) = llm_alias {
        generate_answer(kb, &bundle, &qa.question, alias).await?
    } else {
        retrieval_answer(&bundle)
    };
    let generation_latency_ms = gen_start.elapsed().as_millis() as u64;

    Ok(QueryResult {
        question: qa.question.clone(),
        gold_answer: qa.gold_answer().to_string(),
        predicted_answer,
        category: qa.question_category(),
        evidence_found,
        evidence_total,
        recall_items: bundle.items.len(),
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

    let context = format_context(bundle);

    let system = "You are a helpful assistant answering questions about conversations. \
        Use ONLY the provided context to answer. If the information is not available \
        in the context, say 'The information is not mentioned in the conversation.' \
        Answer concisely in one or two sentences.";

    let user = format!("Context:\n{context}\n\nQuestion: {question}\n\nAnswer:");

    let messages = vec![Message::system(system), Message::user(&user)];
    let options = GenerationOptions {
        max_tokens: Some(256),
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

/// In retrieval-only mode, concatenate top recall items as the answer.
fn retrieval_answer(bundle: &ContextBundle) -> String {
    bundle
        .items
        .iter()
        .take(5)
        .map(|item| item.content.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Format recall items into a context string for the LLM prompt.
fn format_context(bundle: &ContextBundle) -> String {
    let mut ctx = String::new();
    for (i, item) in bundle.items.iter().enumerate() {
        let _ = writeln!(
            &mut ctx,
            "[{}] ({}, score={:.3}): {}",
            i + 1,
            item.node_type,
            item.score,
            item.content,
        );
    }
    ctx
}
