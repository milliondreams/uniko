//! Session lifecycle management for the ingest pipeline.

// Rust guideline compliant

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use uni_db::Value;

use uniko_store::schema::constants::edges;
use uniko_store::{KnowledgeBase, NodeId};

/// Convert a `chrono::DateTime<Utc>` to `uni_db::Value::Temporal(DateTime)`.
fn datetime_value(dt: &DateTime<Utc>) -> Value {
    Value::Temporal(uni_db::common::TemporalValue::DateTime {
        nanos_since_epoch: dt.timestamp_nanos_opt().unwrap_or(0),
        offset_seconds: 0,
        timezone_name: None,
    })
}

/// Get or create a Session node by `session_id`.
///
/// If a session with the given ID already exists, returns its internal
/// node ID.  Otherwise creates a new Session node with `started_at`
/// set to the provided timestamp.
///
/// # Errors
///
/// Returns a storage error if the graph operation fails.
pub(crate) async fn get_or_create_session(
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
    props.insert("started_at".into(), datetime_value(timestamp));
    kb.create_node("Session", &props).await
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
    timestamp: &str,
) -> uniko_store::Result<NodeId> {
    let mut props = HashMap::new();
    props.insert("name".into(), Value::String(participant_id.to_string()));
    props.insert("kind".into(), Value::String("unknown".to_string()));
    props.insert("last_seen".into(), Value::String(timestamp.to_string()));
    kb.merge_node("Participant", "participant_id", participant_id, &props)
        .await
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
    kb.create_edge(
        edges::PARTICIPATED_IN,
        participant_nid,
        session_nid,
        &HashMap::new(),
    )
    .await?;
    Ok(())
}
