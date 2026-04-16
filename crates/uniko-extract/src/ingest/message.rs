//! Message ingest: create the Message node and all associated edges.

// Rust guideline compliant

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use uni_db::Value;

use uniko_pipes::types::IngestMessage;
use uniko_store::{KnowledgeBase, NodeId};

use super::chunking::{count_tokens, select_chunker, ChunkConfig};
use super::session::{ensure_participant, ensure_participated_in, get_or_create_session};

/// Convert a `chrono::DateTime<Utc>` to `uni_db::Value::Temporal(DateTime)`.
fn datetime_value(dt: DateTime<Utc>) -> Value {
    Value::Temporal(uni_db::common::TemporalValue::DateTime {
        nanos_since_epoch: dt.timestamp_nanos_opt().unwrap_or(0),
        offset_seconds: 0,
        timezone_name: None,
    })
}

/// Result of ingesting a single message.
#[derive(Debug)]
pub struct MessageIngestResult {
    /// Internal node ID of the created Message.
    pub message_node_id: NodeId,
    /// Node IDs of chunks (empty if the message was not chunked).
    pub chunk_node_ids: Vec<NodeId>,
    /// Internal node ID of the session.
    pub session_node_id: NodeId,
}

/// Ingest a message into the knowledge graph.
///
/// Creates the Message node, SENT_BY, IN_SESSION, and NEXT edges.
/// Chunks long messages (> `message_chunk_threshold` tokens) into Chunk
/// nodes with HAS_CHUNK edges.
///
/// # Errors
///
/// Returns a storage error if any graph operation fails.
pub async fn ingest_message(
    kb: &KnowledgeBase,
    msg: &IngestMessage,
) -> uniko_store::Result<MessageIngestResult> {
    let ts = msg.timestamp.to_rfc3339();
    let ts_value = datetime_value(msg.timestamp);

    // 1. Idempotency: skip if this message already exists.
    if let Some((existing_id, _)) = kb
        .get_node_by_ext_id("Message", "message_id", &msg.message_id)
        .await?
    {
        // Find the session for the existing message.
        let session_id = kb
            .get_node_by_ext_id("Session", "session_id", &msg.session_id)
            .await?
            .map(|(nid, _)| nid)
            .unwrap_or(0);
        return Ok(MessageIngestResult {
            message_node_id: existing_id,
            chunk_node_ids: Vec::new(),
            session_node_id: session_id,
        });
    }

    // 2. Ensure Participant exists.
    let participant_nid = ensure_participant(kb, &msg.sender_id, &ts).await?;

    // 3. Ensure Session exists.
    let session_nid = get_or_create_session(kb, &msg.session_id, &ts).await?;

    // 4. Create Message node.
    let mut props = HashMap::new();
    props.insert(
        "message_id".into(),
        Value::String(msg.message_id.clone()),
    );
    props.insert("content".into(), Value::String(msg.content.clone()));
    props.insert(
        "content_type".into(),
        Value::String(msg.content_type.clone()),
    );
    props.insert("timestamp".into(), ts_value.clone());
    let message_nid = kb.create_node("Message", &props).await?;

    // 5. Create SENT_BY edge (Message → Participant).
    let mut sent_by_props = HashMap::new();
    sent_by_props.insert("role".into(), Value::String("user".to_string()));
    kb.create_edge("SENT_BY", message_nid, participant_nid, &sent_by_props)
        .await?;

    // 6. Create IN_SESSION edge (Message → Session).
    kb.create_edge("IN_SESSION", message_nid, session_nid, &HashMap::new())
        .await?;

    // 6a. Create PARTICIPATED_IN edge (Participant → Session) if not exists.
    ensure_participated_in(kb, participant_nid, session_nid).await?;

    // 6b. Create ADDRESSED_TO edges (Message → Participant recipients).
    create_addressed_to_edges(kb, message_nid, msg, participant_nid, session_nid).await?;

    // 7. Find previous message in session and create NEXT edge.
    find_and_link_previous(kb, &msg.session_id, &ts_value, message_nid).await?;

    // 8. Chunk long messages.
    let chunk_threshold = kb.config().message_chunk_threshold;
    let chunk_node_ids = if count_tokens(&msg.content) > chunk_threshold {
        let chunk_cfg = ChunkConfig::from_uniko_config(kb.config());
        let chunker = select_chunker(&msg.content_type, None);
        let chunks = chunker.chunk(&msg.content, &chunk_cfg);
        create_chunks(kb, &msg.message_id, message_nid, &chunks, "Message").await?
    } else {
        Vec::new()
    };

    Ok(MessageIngestResult {
        message_node_id: message_nid,
        chunk_node_ids,
        session_node_id: session_nid,
    })
}

/// Find the previous message in the same session and create a NEXT edge.
async fn find_and_link_previous(
    kb: &KnowledgeBase,
    session_id: &str,
    timestamp: &Value,
    new_message_nid: NodeId,
) -> uniko_store::Result<()> {
    let cypher = "\
        MATCH (prev:Message)-[:IN_SESSION]->(s:Session {session_id: $sid}) \
        WHERE prev.timestamp < $ts \
        RETURN id(prev) AS vid \
        ORDER BY prev.timestamp DESC \
        LIMIT 1";

    let session = kb.db().session();
    let result = session
        .query_with(cypher)
        .param("sid", session_id)
        .param("ts", timestamp.clone())
        .fetch_all()
        .await
        .map_err(|e| uniko_store::UnikoError::Storage(e.to_string()))?;

    if let Some(row) = result.rows().first() {
        let prev_vid: i64 = row
            .get("vid")
            .map_err(|e| uniko_store::UnikoError::Storage(e.to_string()))?;

        // gap_ms is approximate — we don't have the previous timestamp
        // parsed here, so we use 0 and let the caller compute if needed.
        let mut edge_props = HashMap::new();
        edge_props.insert("gap_ms".into(), Value::Int(0));
        kb.create_edge("NEXT", prev_vid, new_message_nid, &edge_props)
            .await?;
    }

    Ok(())
}

/// Create ADDRESSED_TO edges from the message to recipient participants.
///
/// If `msg.addressed_to` is provided, uses those IDs.  Otherwise infers
/// recipients from the session's PARTICIPATED_IN edges (all participants
/// except the sender).
async fn create_addressed_to_edges(
    kb: &KnowledgeBase,
    message_nid: NodeId,
    msg: &IngestMessage,
    sender_nid: NodeId,
    session_nid: NodeId,
) -> uniko_store::Result<()> {
    use uniko_store::schema::constants::edges;
    use uniko_store::storage::edges::Direction;

    let recipient_nids: Vec<NodeId> = if let Some(ref ids) = msg.addressed_to {
        // Explicit recipients — ensure each exists and collect node IDs.
        let mut nids = Vec::with_capacity(ids.len());
        for pid in ids {
            let nid = super::session::ensure_participant(kb, pid, &msg.timestamp.to_rfc3339()).await?;
            nids.push(nid);
        }
        nids
    } else {
        // Infer from session: all PARTICIPATED_IN participants except sender.
        let participated = kb
            .get_edges(session_nid, edges::PARTICIPATED_IN, Direction::Incoming)
            .await?;
        participated
            .iter()
            .filter(|e| e.from != sender_nid)
            .map(|e| e.from)
            .collect()
    };

    for &recipient_nid in &recipient_nids {
        kb.create_edge("ADDRESSED_TO", message_nid, recipient_nid, &HashMap::new())
            .await?;
    }

    Ok(())
}

/// Create Chunk nodes + HAS_CHUNK edges for a parent node.
pub async fn create_chunks(
    kb: &KnowledgeBase,
    parent_ext_id: &str,
    parent_nid: NodeId,
    chunks: &[super::chunking::ChunkData],
    _parent_label: &str,
) -> uniko_store::Result<Vec<NodeId>> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    let chunk_props: Vec<HashMap<String, Value>> = chunks
        .iter()
        .map(|c| {
            let cid = uniko_store::id::chunk_id(parent_ext_id, c.index);
            let mut props = HashMap::new();
            props.insert("chunk_id".into(), Value::String(cid));
            props.insert("text".into(), Value::String(c.text.clone()));
            props.insert("index".into(), Value::Int(c.index as i64));
            props.insert("start".into(), Value::Int(c.start as i64));
            props.insert("end".into(), Value::Int(c.end as i64));
            props.insert("token_count".into(), Value::Int(c.token_count as i64));
            props.insert("chunk_type".into(), Value::String(c.chunk_type.clone()));
            if let Some(ref lang) = c.language {
                props.insert("language".into(), Value::String(lang.clone()));
            }
            if let Some(ref sym) = c.symbol_name {
                props.insert("symbol_name".into(), Value::String(sym.clone()));
            }
            if let Some(ref h) = c.heading {
                props.insert("heading".into(), Value::String(h.clone()));
            }
            props
        })
        .collect();

    let chunk_nids = kb.batch_create_nodes("Chunk", &chunk_props).await?;

    let edges: Vec<(NodeId, NodeId, HashMap<String, Value>)> = chunk_nids
        .iter()
        .enumerate()
        .map(|(i, &cid)| {
            let mut props = HashMap::new();
            props.insert("index".into(), Value::Int(i as i64));
            (parent_nid, cid, props)
        })
        .collect();

    kb.batch_create_edges("HAS_CHUNK", &edges).await?;

    Ok(chunk_nids)
}
