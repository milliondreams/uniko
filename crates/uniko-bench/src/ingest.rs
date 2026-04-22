//! Conversation ingestion into a fresh KnowledgeBase.
//!
//! For each conversation, creates an isolated in-memory KB, ingests all
//! turns with entity and observation extraction, then returns the KB
//! ready for querying.

use std::collections::HashMap;
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

use crate::data::{Conversation, LocomoSample, ParsedSession};

/// Ingest a full LoCoMo conversation into a fresh in-memory KB.
///
/// Returns the populated KB ready for recall queries.
///
/// # Errors
///
/// Returns an error if KB creation or any ingestion step fails.
/// Open an existing persistent KB without ingesting.
pub async fn open_kb(
    ingest_dir: &Path,
    config: UnikoConfig,
    extra_catalog: &[ModelAliasSpec],
) -> Result<Arc<KnowledgeBase>> {
    let kb = KnowledgeBase::open_with_xervo(ingest_dir, config, extra_catalog.to_vec())
        .await
        .context("opening KB")?;
    Ok(Arc::new(kb))
}

/// Ingest a full LoCoMo conversation into a persistent KB.
pub async fn ingest_conversation(
    sample: &LocomoSample,
    sessions: &[ParsedSession],
    ingest_dir: &Path,
    config: UnikoConfig,
    extra_catalog: &[ModelAliasSpec],
) -> Result<Arc<KnowledgeBase>> {
    let kb = KnowledgeBase::open_with_xervo(ingest_dir, config, extra_catalog.to_vec())
        .await
        .context("creating KB")?;
    let kb = Arc::new(kb);

    let breaker = Arc::new(CircuitBreaker::new(5, 60_000));
    let cancel = CancellationToken::new();
    let entity_step = EntityExtractionStep;
    let obs_step = ObservationExtractionStep;

    let mut total_turns = 0u32;
    let mut total_entities = 0usize;
    let mut total_observations = 0usize;
    let mut total_session_chunks = 0usize;

    for session in sessions {
        let base_ts = parse_session_datetime(&session.date_time);

        for (turn_idx, turn) in session.turns.iter().enumerate() {
            let timestamp = base_ts + Duration::seconds(turn_idx as i64 * 30);

            let other_speaker = other_speaker_name(&turn.speaker, &sample.conversation);

            let msg = IngestMessage {
                message_id: format!("{}-{}", sample.sample_id, turn.dia_id),
                content: turn.text.clone(),
                content_type: "text".to_string(),
                sender_id: turn.speaker.clone(),
                session_id: session.session_id.clone(),
                addressed_to: Some(vec![other_speaker]),
                timestamp,
                metadata: HashMap::new(),
            };

            let turn_start = std::time::Instant::now();

            // Ingest the message (creates node, edges, chunks).
            let result = ingest_message(&kb, &msg)
                .await
                .with_context(|| format!("ingesting {}", turn.dia_id))?;

            let ingest_ms = turn_start.elapsed().as_millis();

            // Run entity extraction on the message node.
            let mut ctx = PipelineContext::new(
                result.message_node_id,
                turn.text.clone(),
                "text".to_string(),
                cancel.clone(),
                kb.clone(),
                breaker.clone(),
            );
            ctx.metadata.insert(
                "timestamp".into(),
                serde_json::Value::String(timestamp.to_rfc3339()),
            );

            let ner_start = std::time::Instant::now();
            let _ = entity_step.execute(&mut ctx).await;
            let ner_ms = ner_start.elapsed().as_millis();
            total_entities += ctx.extracted_entities.len();

            // Run observation extraction on the same context.
            let obs_start = std::time::Instant::now();
            let _ = obs_step.execute(&mut ctx).await;
            let obs_ms = obs_start.elapsed().as_millis();
            total_observations += ctx.extracted_observations.len();

            total_turns += 1;

            // Progress every 20 turns or on first turn.
            if total_turns == 1 || total_turns % 20 == 0 {
                tracing::info!(
                    turn = total_turns,
                    dia_id = %turn.dia_id,
                    ingest_ms,
                    ner_ms,
                    obs_ms,
                    entities = ctx.extracted_entities.len(),
                    observations = ctx.extracted_observations.len(),
                    "turn processed",
                );
            }
        }

        // Chunk the session for retrieval (concatenates turns with speaker prefixes).
        let chunk_ids =
            uniko_extract::ingest::session_chunk::chunk_session(&kb, &session.session_id)
                .await
                .with_context(|| format!("chunking session {}", session.session_id))?;
        total_session_chunks += chunk_ids.len();

        // Aggregate per-message observations into searchable session-level chunks.
        let obs_chunk_ids =
            uniko_extract::ingest::session_chunk::chunk_session_observations(
                &kb,
                &session.session_id,
            )
            .await
            .with_context(|| {
                format!("chunking session observations {}", session.session_id)
            })?;
        total_session_chunks += obs_chunk_ids.len();
    }

    tracing::info!(
        sample_id = %sample.sample_id,
        turns = total_turns,
        entities = total_entities,
        observations = total_observations,
        session_chunks = total_session_chunks,
        "conversation ingested",
    );

    Ok(kb)
}

/// Parse session datetime string with flexible fallback.
///
/// LoCoMo uses the format `"4:04 pm on 20 January, 2023"`.
fn parse_session_datetime(dt: &str) -> DateTime<Utc> {
    // Try RFC 3339 first.
    if let Ok(parsed) = DateTime::parse_from_rfc3339(dt) {
        return parsed.with_timezone(&Utc);
    }

    // LoCoMo format: "4:04 pm on 20 January, 2023"
    // Strip "on " to get "4:04 pm 20 January, 2023"
    let normalized = dt.replace(" on ", " ");
    for fmt in &[
        "%l:%M %P %d %B, %Y", // "4:04 pm 20 January, 2023"
        "%l:%M %P %d %B %Y",  // "4:04 pm 20 January 2023"
        "%I:%M %P %d %B, %Y", // "04:04 pm 20 January, 2023"
    ] {
        if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(normalized.trim(), fmt) {
            return naive.and_utc();
        }
    }

    // Try standard formats.
    for fmt in &[
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d %B %Y",
        "%B %d, %Y",
        "%Y-%m-%d",
    ] {
        if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(dt, fmt) {
            return naive.and_utc();
        }
        if let Ok(naive) = chrono::NaiveDate::parse_from_str(dt, fmt) {
            return naive.and_hms_opt(0, 0, 0).unwrap().and_utc();
        }
    }
    tracing::warn!(dt, "could not parse session datetime, using epoch");
    DateTime::UNIX_EPOCH
}

/// Get the other speaker's name in a 2-person conversation.
fn other_speaker_name(current_speaker: &str, conv: &Conversation) -> String {
    if current_speaker == conv.speaker_a {
        conv.speaker_b.clone()
    } else {
        conv.speaker_a.clone()
    }
}
