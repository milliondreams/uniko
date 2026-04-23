//! LongMemEval item ingestion into a fresh KnowledgeBase.
//!
//! Each LongMemEval item is self-contained with its own haystack
//! of chat sessions. We create one KB per item and ingest all
//! sessions with entity and observation extraction.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use tokio_util::sync::CancellationToken;
use uniko_pipes::Step;

use uni_db::ModelAliasSpec;
use uniko_extract::ingest::message::ingest_message;
use uniko_extract::ner::EntityExtractionStep;
use uniko_extract::observations::ObservationExtractionStep;
use uniko_pipes::circuit_breaker::CircuitBreaker;
use uniko_pipes::step::PipelineContext;
use uniko_pipes::types::IngestMessage;
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;

use super::data::LongMemEvalItem;

/// Evidence mapping built during ingestion.
///
/// Tracks which sessions and messages contain evidence for the answer,
/// enabling retrieval metric computation (Recall@k, NDCG@k).
#[derive(Debug)]
pub struct EvidenceMap {
    /// Session IDs that contain evidence (from `answer_session_ids`).
    pub answer_session_ids: HashSet<String>,
    /// Message IDs where `has_answer == true`.
    pub answer_message_ids: Vec<String>,
    /// Maps session_id → list of message_ids in that session.
    pub session_to_messages: HashMap<String, Vec<String>>,
}

/// Ingest a single LongMemEval item into a fresh KB.
///
/// Creates a new KnowledgeBase, ingests all haystack sessions,
/// runs entity/observation extraction, and returns the populated KB
/// along with an evidence map for retrieval evaluation.
pub async fn ingest_item(
    item: &LongMemEvalItem,
    ingest_dir: &Path,
    config: UnikoConfig,
    extra_catalog: &[ModelAliasSpec],
) -> Result<(Arc<KnowledgeBase>, EvidenceMap)> {
    let kb = KnowledgeBase::open_with_xervo(ingest_dir, config, extra_catalog.to_vec())
        .await
        .context("creating KB")?;
    let kb = Arc::new(kb);

    let breaker = Arc::new(CircuitBreaker::new(5, 60_000));
    let cancel = CancellationToken::new();
    let entity_step = EntityExtractionStep;
    let obs_step = ObservationExtractionStep;

    let mut evidence_map = EvidenceMap {
        answer_session_ids: item.answer_session_ids.iter().cloned().collect(),
        answer_message_ids: Vec::new(),
        session_to_messages: HashMap::new(),
    };

    let mut total_turns = 0u32;
    let mut total_entities = 0usize;
    let mut total_observations = 0usize;
    let mut total_session_chunks = 0usize;

    // Iterate over sessions: zip haystack_sessions, haystack_session_ids, haystack_dates.
    for (session_idx, ((session_turns, session_id), date_str)) in item
        .haystack_sessions
        .iter()
        .zip(item.haystack_session_ids.iter())
        .zip(item.haystack_dates.iter())
        .enumerate()
    {
        let base_ts = parse_lme_datetime(date_str);

        // Create session-level context for pronoun resolution.
        let mut session_ctx = uniko_extract::ingest::context::SessionContext::new(
            session_id.clone(),
            0, // resolved during first ingest
        );
        session_ctx.register_participant("user", 0);
        session_ctx.register_participant("assistant", 0);

        let mut session_message_ids = Vec::new();

        for (turn_idx, turn) in session_turns.iter().enumerate() {
            let timestamp = base_ts + Duration::seconds(turn_idx as i64 * 30);
            let message_id = format!("{}-s{}-t{}", item.question_id, session_idx, turn_idx);

            let other_role = if turn.role == "user" {
                "assistant"
            } else {
                "user"
            };

            session_ctx.set_current_speaker(&turn.role);

            let msg = IngestMessage {
                message_id: message_id.clone(),
                content: turn.content.clone(),
                content_type: "text".to_string(),
                sender_id: turn.role.clone(),
                session_id: session_id.clone(),
                addressed_to: Some(vec![other_role.to_string()]),
                timestamp,
                metadata: HashMap::new(),
            };

            // Ingest the message.
            let result = ingest_message(&kb, &msg, &mut session_ctx)
                .await
                .with_context(|| format!("ingesting {message_id}"))?;

            // Run entity extraction.
            let mut ctx = PipelineContext::new(
                result.message_node_id,
                turn.content.clone(),
                "text".to_string(),
                cancel.clone(),
                kb.clone(),
                breaker.clone(),
            );
            ctx.metadata.insert(
                "timestamp".into(),
                serde_json::Value::String(timestamp.to_rfc3339()),
            );
            if let Ok(val) = serde_json::to_value(&session_ctx) {
                ctx.metadata.insert("session_context".into(), val);
            }

            let _ = entity_step.execute(&mut ctx).await;
            total_entities += ctx.extracted_entities.len();

            // Run observation extraction.
            let _ = obs_step.execute(&mut ctx).await;
            total_observations += ctx.extracted_observations.len();

            // Read back updated sentence context.
            if let Some(updated) = ctx.metadata.get("sentence_ctx_updated")
                && let Ok(sent_ctx) = serde_json::from_value::<
                    uniko_extract::ingest::context::SentenceContext,
                >(updated.clone())
            {
                session_ctx.sentence_ctx = sent_ctx;
            }

            // Track evidence.
            if turn.has_answer {
                evidence_map.answer_message_ids.push(message_id.clone());
            }
            session_message_ids.push(message_id);

            total_turns += 1;

            if total_turns == 1 || total_turns.is_multiple_of(50) {
                tracing::info!(
                    turn = total_turns,
                    session = session_idx,
                    entities = ctx.extracted_entities.len(),
                    observations = ctx.extracted_observations.len(),
                    "turn processed",
                );
            }
        }

        evidence_map
            .session_to_messages
            .insert(session_id.clone(), session_message_ids);

        // Chunk the session for retrieval.
        let chunk_ids = uniko_extract::ingest::session_chunk::chunk_session(&kb, session_id)
            .await
            .with_context(|| format!("chunking session {session_id}"))?;
        total_session_chunks += chunk_ids.len();

        // Chunk session observations.
        let obs_chunk_ids =
            uniko_extract::ingest::session_chunk::chunk_session_observations(&kb, session_id)
                .await
                .with_context(|| format!("chunking observations {session_id}"))?;
        total_session_chunks += obs_chunk_ids.len();
    }

    tracing::info!(
        question_id = %item.question_id,
        sessions = item.haystack_sessions.len(),
        turns = total_turns,
        entities = total_entities,
        observations = total_observations,
        session_chunks = total_session_chunks,
        evidence_messages = evidence_map.answer_message_ids.len(),
        "item ingested",
    );

    Ok((kb, evidence_map))
}

/// Parse LongMemEval datetime format: "YYYY/MM/DD (Day) HH:MM".
fn parse_lme_datetime(dt: &str) -> DateTime<Utc> {
    // Try the LongMemEval format first: "2023/04/10 (Mon) 23:07"
    for fmt in &["%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M"] {
        if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(dt, fmt) {
            return naive.and_utc();
        }
    }

    // Try RFC 3339 fallback.
    if let Ok(parsed) = DateTime::parse_from_rfc3339(dt) {
        return parsed.with_timezone(&Utc);
    }

    // Try standard formats.
    for fmt in &["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"] {
        if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(dt, fmt) {
            return naive.and_utc();
        }
        if let Ok(naive) = chrono::NaiveDate::parse_from_str(dt, fmt) {
            return naive.and_hms_opt(0, 0, 0).unwrap().and_utc();
        }
    }

    tracing::warn!(dt, "could not parse LME datetime, using epoch");
    DateTime::UNIX_EPOCH
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Datelike;

    #[test]
    fn test_parse_lme_datetime() {
        let dt = parse_lme_datetime("2023/04/10 (Mon) 23:07");
        assert_eq!(dt.year(), 2023);
        assert_eq!(dt.month(), 4);
        assert_eq!(dt.day(), 10);
    }

    #[test]
    fn test_parse_lme_datetime_fallback() {
        let dt = parse_lme_datetime("2023-01-15T10:30:00+00:00");
        assert_eq!(dt.year(), 2023);

        let dt = parse_lme_datetime("garbage");
        assert_eq!(dt, DateTime::UNIX_EPOCH);
    }
}
