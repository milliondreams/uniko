//! Recall and answer generation for LongMemEval questions.

use std::time::Instant;

use anyhow::Result;
use uniko_memory::recall::{ContextBundle, RecallConfig, recall};
use uniko_store::KnowledgeBase;

use super::data::LmeQuestionType;
use super::ingest::EvidenceMap;

/// Result of querying a single LongMemEval question.
#[derive(Debug)]
pub struct LmeQueryResult {
    /// Unique question identifier.
    pub question_id: String,
    /// Original question text.
    pub question: String,
    /// Question type category.
    pub question_type: LmeQuestionType,
    /// Whether this is an abstention question.
    pub is_abstention: bool,
    /// Gold answer from the dataset.
    pub gold_answer: String,
    /// Model-generated or retrieval-based answer.
    pub predicted_answer: String,
    /// Session IDs of recalled chunks (for R@k computation).
    pub recalled_session_ids: Vec<String>,
    /// Number of recall items retrieved.
    pub recall_items: usize,
    /// Whether the gold answer appears in top-5 recalled items.
    pub context_contains_answer: bool,
    /// Session-level Recall@5.
    pub recall_at_5: f64,
    /// NDCG@5.
    pub ndcg_at_5: f64,
    /// Recall phase latency in milliseconds.
    pub recall_latency_ms: u64,
    /// Answer generation latency in milliseconds (0 if retrieval-only).
    pub generation_latency_ms: u64,
}

/// Run recall for a LongMemEval question and optionally generate an answer.
#[allow(clippy::too_many_arguments)]
pub async fn run_lme_query(
    kb: &KnowledgeBase,
    question_id: &str,
    question: &str,
    question_type: LmeQuestionType,
    gold_answer: &str,
    evidence_map: &EvidenceMap,
    token_budget: usize,
    llm_alias: Option<&str>,
) -> Result<LmeQueryResult> {
    let recall_config = RecallConfig {
        token_budget,
        ..Default::default()
    };

    // Run recall.
    let recall_start = Instant::now();
    let bundle = recall(kb, question, &recall_config).await?;
    let recall_latency_ms = recall_start.elapsed().as_millis() as u64;

    // Extract session IDs from recalled items.
    let recalled_session_ids = extract_session_ids(kb, &bundle).await;

    // Compute retrieval metrics.
    let recalled_contents: Vec<&str> = bundle.items.iter().map(|i| i.content.as_str()).collect();
    let answer_session_ids: Vec<String> = evidence_map.answer_session_ids.iter().cloned().collect();

    let context_contains = super::eval::context_contains_answer(&recalled_contents, gold_answer, 5);
    let recall_5 = super::eval::recall_at_k(&recalled_session_ids, &answer_session_ids, 5);
    let ndcg_5 = super::eval::ndcg_at_k(&recalled_session_ids, &answer_session_ids, 5);

    // Generate answer.
    let gen_start = Instant::now();
    let predicted_answer = if let Some(alias) = llm_alias {
        generate_answer(kb, &bundle, question, alias).await?
    } else {
        crate::retrieval_answer(&bundle, 5)
    };
    let generation_latency_ms = gen_start.elapsed().as_millis() as u64;

    let is_abstention = super::data::LmeQuestionType::is_abstention(question_id);

    Ok(LmeQueryResult {
        question_id: question_id.to_string(),
        question: question.to_string(),
        question_type,
        is_abstention,
        gold_answer: gold_answer.to_string(),
        predicted_answer,
        recalled_session_ids,
        recall_items: bundle.items.len(),
        context_contains_answer: context_contains,
        recall_at_5: recall_5,
        ndcg_at_5: ndcg_5,
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

    let context = crate::format_context(bundle);

    let system = "You are a helpful assistant answering questions about past conversations. \
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

/// Extract unique session IDs from recalled items by querying the graph.
///
/// For each recalled node, traverses BELONGS_TO/PART_OF edges to find
/// the parent Session node and its session_id property.
async fn extract_session_ids(kb: &KnowledgeBase, bundle: &ContextBundle) -> Vec<String> {
    let mut session_ids = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let session = kb.db().session();
    for item in &bundle.items {
        let query = format!(
            "MATCH (n)-[:BELONGS_TO|PART_OF*1..2]->(s:Session) \
             WHERE id(n) = {} RETURN s.session_id AS sid LIMIT 1",
            item.node_id
        );
        if let Ok(result) = session.query_with(&query).fetch_all().await {
            for row in result.rows() {
                if let Ok(sid) = row.get::<String>("sid")
                    && seen.insert(sid.clone())
                {
                    session_ids.push(sid);
                }
            }
        }
    }

    session_ids
}
