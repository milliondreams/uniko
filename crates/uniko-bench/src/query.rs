//! Recall and answer generation for benchmark questions.

use std::collections::HashMap;
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
    /// Conversation sample this question belongs to (LoCoMo `sample_id`).
    pub sample_id: String,
    /// Position of this question within `sample.qa` (post-category filter).
    /// Reproducible across runs because question order is stable.
    pub question_index: usize,
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
    /// Whether the recall cascade exited early from Phase 1 (Compact).
    pub phase1_only: bool,
    /// Coverage score reported by the recall cascade (0.0–1.0).
    pub coverage: f64,
    /// Estimated total tokens in the recall bundle.
    pub total_tokens: usize,
}

/// Run recall for a question and optionally generate an answer.
///
/// In retrieval-only mode, the "predicted answer" is the concatenated
/// content of the top recall items.
pub async fn run_query(
    kb: &KnowledgeBase,
    sample_id: &str,
    question_index: usize,
    qa: &QaPair,
    evidence_texts: &[String],
    llm_alias: Option<&str>,
    llm_use_default_options: bool,
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
        generate_answer(kb, &bundle, &qa.question, alias, llm_use_default_options).await?
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
        sample_id: sample_id.to_string(),
        question_index,
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
        phase1_only: bundle.phase1_only,
        coverage: bundle.coverage,
        total_tokens: bundle.total_tokens,
    })
}

/// Generate an answer using the LLM from the recall context.
async fn generate_answer(
    kb: &KnowledgeBase,
    bundle: &ContextBundle,
    question: &str,
    llm_alias: &str,
    use_default_options: bool,
) -> Result<String> {
    use uni_db::xervo::{GenerationOptions, Message};

    let node_ids: Vec<i64> = bundle.items.iter().map(|i| i.node_id).collect();
    let session_dates = fetch_session_dates(kb, &node_ids).await;
    let temporal_anchors = fetch_temporal_anchors(kb, &node_ids).await;
    let context = uniko_bench::format_context(bundle, &session_dates, &temporal_anchors);

    let system = "You are a helpful assistant answering questions about conversations. \
        Answer using the provided context. You may paraphrase or make direct \
        inferences from what the context says, including using the `session_date` \
        field to resolve relative dates like 'yesterday' or 'next month'. \
        When an item carries a `temporal` field, treat that date as the \
        resolved anchor for the claim — prefer it over relative phrases like \
        'last Friday' that appear inside the content. \
        For speculative questions phrased as 'Would X likely…?' or 'What would X \
        think about…?', reason from the speaker's stated preferences, beliefs, \
        and behaviors in the context to give a reasoned inference rather than \
        abstaining. Only say 'The information is not mentioned in the \
        conversation' when no relevant preferences or behaviors appear in the \
        context. \
        Answer concisely in one or two sentences.";

    let user = format!("Context:\n{context}\n\nQuestion: {question}\n\nAnswer:");

    let messages = vec![Message::system(system), Message::user(&user)];
    // Gemma 4 (and other reasoning models) emit chain-of-thought into a
    // separate `reasoning_content` field that consumes the token budget.
    // 2048 leaves headroom for ~1500 reasoning tokens + a 1-2 sentence
    // answer. For non-reasoning models this is just an upper bound;
    // they self-terminate at the EOS well before hitting it.
    // Reasoning models (gemma4 / phi4 via mistralrs) need a token budget
    // for chain-of-thought; non-reasoning local models tolerate the
    // custom temperature. OpenAI's gpt-5/o-series rejects both — it
    // requires `max_completion_tokens` and only accepts the default
    // temperature.  Caller passes `use_default_options=true` for those.
    let options = if use_default_options {
        GenerationOptions::default()
    } else {
        GenerationOptions {
            max_tokens: Some(2048),
            temperature: Some(0.1),
            ..Default::default()
        }
    };

    let result = kb
        .db()
        .xervo()
        .generate(llm_alias, &messages, options)
        .await?;
    Ok(result.text.trim().to_string())
}

/// Resolve the originating Session.started_at for each recalled node.
///
/// Walks the same edge set as `longmemeval::query::extract_session_ids`
/// (Message →IN_SESSION, Session →HAS_CHUNK, Observation →OBSERVED_IN
/// →Message →IN_SESSION) but projects `started_at` instead of
/// `session_id`. Nodes with no Session ancestor (e.g. label types not
/// anchored in a session) are simply absent from the returned map.
///
/// One query per item — matches the per-item pattern already used for
/// session_id extraction. Cheap relative to the LLM call that follows.
async fn fetch_session_dates(kb: &KnowledgeBase, node_ids: &[i64]) -> HashMap<i64, String> {
    let mut out = HashMap::new();
    let session = kb.db().session();
    for &nid in node_ids {
        let q = format!(
            "MATCH (n) WHERE id(n) = {nid} \
             OPTIONAL MATCH (n)-[:IN_SESSION]->(s1:Session) \
             OPTIONAL MATCH (s2:Session)-[:HAS_CHUNK]->(n) \
             OPTIONAL MATCH (n)-[:OBSERVED_IN]->(:Message)-[:IN_SESSION]->(s3:Session) \
             RETURN coalesce(s1.started_at, s2.started_at, s3.started_at) AS ts \
             LIMIT 1"
        );
        if let Ok(res) = session.query_with(&q).fetch_all().await {
            for row in res.rows() {
                if let Ok(ts) = row.get::<String>("ts") {
                    // uni-db formats DateTime as "YYYY-MM-DDTHH:MM:SS±HHMM"; the
                    // model only needs the date for relative-date resolution.
                    let date = ts.split('T').next().unwrap_or(&ts).to_string();
                    out.insert(nid, date);
                    break;
                }
            }
        }
    }
    out
}

/// Resolve the per-Observation `temporal_anchor` for each recalled node.
///
/// Phase A surfaces ARGM-TMP captures from SRL as a resolved absolute
/// date at ingest.  This helper pulls that field for any Observation in
/// the bundle so the LLM sees the resolved anchor rather than the raw
/// `"Last Fri"` phrase buried in chunk text.
///
/// Returns an empty entry for non-Observation node types or for
/// Observations whose `temporal_anchor` is null.
async fn fetch_temporal_anchors(kb: &KnowledgeBase, node_ids: &[i64]) -> HashMap<i64, String> {
    let mut out = HashMap::new();
    let session = kb.db().session();
    for &nid in node_ids {
        let q = format!(
            "MATCH (n:Observation) WHERE id(n) = {nid} \
             AND n.temporal_anchor IS NOT NULL \
             RETURN n.temporal_anchor AS ts \
             LIMIT 1"
        );
        if let Ok(res) = session.query_with(&q).fetch_all().await {
            for row in res.rows() {
                if let Ok(ts) = row.get::<String>("ts") {
                    let date = ts.split('T').next().unwrap_or(&ts).to_string();
                    out.insert(nid, date);
                    break;
                }
            }
        }
    }
    out
}
