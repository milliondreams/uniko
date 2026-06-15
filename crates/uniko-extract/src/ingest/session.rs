//! Session lifecycle management for the ingest pipeline.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use uniko_store::Value;

use uniko_store::schema::constants::edges;
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, NodeId};

/// Get or create a Session node by `session_id`.
///
/// If a session with the given ID already exists, returns its internal
/// node ID.  Otherwise creates a new Session node with `started_at`
/// set to the provided timestamp.
///
/// # Errors
///
/// Returns a storage error if the graph operation fails.
pub async fn get_or_create_session(
    kb: &KnowledgeBase,
    session_id: &str,
    timestamp: &DateTime<Utc>,
) -> uniko_store::Result<NodeId> {
    // Fast path: session already exists.
    if let Some((nid, _)) = kb
        .get_node_by_ext_id("Session", "session_id", session_id)
        .await?
    {
        return Ok(nid);
    }

    // Create a new session.
    let mut props = HashMap::new();
    props.insert("session_id".into(), Value::String(session_id.to_string()));
    props.insert("started_at".into(), datetime_value(*timestamp));
    let start = std::time::Instant::now();
    let nid = kb.create_node("Session", &props).await?;
    tracing::info!(
        target: "tx_perf",
        tx_phase = "session_create",
        total_ms = start.elapsed().as_millis() as u64,
        commit_ms = start.elapsed().as_millis() as u64,
        "tx phase",
    );
    Ok(nid)
}

/// Ensure a Participant node exists for the given ID.
///
/// Uses merge semantics: creates the participant if absent, updates
/// `last_seen` if present.
///
/// # Errors
///
/// Returns a storage error if the graph operation fails.
pub(crate) async fn ensure_participant(
    kb: &KnowledgeBase,
    participant_id: &str,
    timestamp: DateTime<Utc>,
) -> uniko_store::Result<NodeId> {
    let mut props = HashMap::new();
    props.insert("name".into(), Value::String(participant_id.to_string()));
    props.insert("kind".into(), Value::String("unknown".to_string()));
    props.insert("last_seen".into(), datetime_value(timestamp));
    let start = std::time::Instant::now();
    let nid = kb
        .merge_node("Participant", "participant_id", participant_id, &props)
        .await?;
    let ms = start.elapsed().as_millis() as u64;
    tracing::info!(
        target: "tx_perf",
        tx_phase = "participant_ensure",
        total_ms = ms,
        commit_ms = ms,
        "tx phase",
    );
    Ok(nid)
}

/// Create a `PARTICIPATED_IN` edge from participant to session.
///
/// Call exactly once per (participant, session) pair — typically at
/// session start. No idempotency check; the caller is responsible
/// for not calling this twice for the same pair.
///
/// # Errors
///
/// Returns a storage error if the edge creation fails.
pub(crate) async fn link_participant_to_session(
    kb: &KnowledgeBase,
    participant_nid: NodeId,
    session_nid: NodeId,
) -> uniko_store::Result<()> {
    let start = std::time::Instant::now();
    kb.create_edge(
        edges::PARTICIPATED_IN,
        participant_nid,
        session_nid,
        &HashMap::new(),
    )
    .await?;
    let ms = start.elapsed().as_millis() as u64;
    tracing::info!(
        target: "tx_perf",
        tx_phase = "participant_link",
        total_ms = ms,
        commit_ms = ms,
        "tx phase",
    );
    Ok(())
}
