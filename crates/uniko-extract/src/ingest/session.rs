//! Session lifecycle management for the ingest pipeline.

// Rust guideline compliant

use std::collections::HashMap;

use uni_db::Value;

use uniko_store::schema::constants::edges;
use uniko_store::storage::edges::Direction;
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
pub(crate) async fn get_or_create_session(
    kb: &KnowledgeBase,
    session_id: &str,
    timestamp: &str,
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
    props.insert("started_at".into(), Value::String(timestamp.to_string()));
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
    props.insert("kind".into(), Value::String("unknown".to_string()));
    props.insert("last_seen".into(), Value::String(timestamp.to_string()));
    kb.merge_node("Participant", "participant_id", participant_id, &props)
        .await
}

/// Ensure a `PARTICIPATED_IN` edge exists from participant to session.
///
/// The first participant in a session gets role `"initiator"`;
/// subsequent participants get `"responder"`.  If the edge already
/// exists, this is a no-op.
///
/// # Errors
///
/// Returns a storage error if the graph operation fails.
pub(crate) async fn ensure_participated_in(
    kb: &KnowledgeBase,
    participant_nid: NodeId,
    session_nid: NodeId,
) -> uniko_store::Result<()> {
    // Check if edge already exists.
    let existing = kb
        .get_edges(participant_nid, edges::PARTICIPATED_IN, Direction::Outgoing)
        .await?;
    if existing.iter().any(|e| e.to == session_nid) {
        return Ok(()); // Already linked.
    }

    // Determine role: "initiator" if no one has joined yet, else "responder".
    let session_participants = kb
        .get_edges(session_nid, edges::PARTICIPATED_IN, Direction::Incoming)
        .await?;
    let role = if session_participants.is_empty() {
        "initiator"
    } else {
        "responder"
    };

    let mut props = HashMap::new();
    props.insert("role".into(), Value::String(role.to_string()));
    kb.create_edge(edges::PARTICIPATED_IN, participant_nid, session_nid, &props)
        .await?;
    Ok(())
}
