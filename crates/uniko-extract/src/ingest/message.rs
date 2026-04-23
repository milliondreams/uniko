//! Message ingest: create the Message node and all associated edges.

// Rust guideline compliant

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use uni_db::Value;

use uniko_pipes::types::IngestMessage;
use uniko_store::{KnowledgeBase, NodeId};

use super::chunking::{ChunkConfig, count_tokens, select_chunker};
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
    session_ctx: &mut super::context::SessionContext,
) -> uniko_store::Result<MessageIngestResult> {
    let ts = msg.timestamp.to_rfc3339();
    let ts_value = datetime_value(msg.timestamp);

    // 1. Idempotency: skip if this message already exists.
    if let Some((existing_id, _)) = kb
        .get_node_by_ext_id("Message", "message_id", &msg.message_id)
        .await?
    {
        return Ok(MessageIngestResult {
            message_node_id: existing_id,
            chunk_node_ids: Vec::new(),
            session_node_id: session_ctx.session_nid,
        });
    }

    let ingest_start = std::time::Instant::now();

    // 2. Ensure Participant exists (use cache when available).
    let participant_nid = if let Some(nid) = session_ctx.participant_nid(&msg.sender_id) {
        if nid != 0 {
            nid
        } else {
            let nid = ensure_participant(kb, &msg.sender_id, &ts).await?;
            session_ctx.register_participant(&msg.sender_id, nid);
            nid
        }
    } else {
        let nid = ensure_participant(kb, &msg.sender_id, &ts).await?;
        session_ctx.register_participant(&msg.sender_id, nid);
        nid
    };

    // 3. Ensure Session exists (use cache when available).
    let session_nid = if session_ctx.session_nid != 0 {
        session_ctx.session_nid
    } else {
        let nid = get_or_create_session(kb, &msg.session_id, &msg.timestamp).await?;
        session_ctx.session_nid = nid;
        nid
    };
    let setup_ms = ingest_start.elapsed().as_millis();

    // 4. Create Message node (triggers auto-embed on content).
    let create_start = std::time::Instant::now();
    let mut props = HashMap::new();
    props.insert("message_id".into(), Value::String(msg.message_id.clone()));
    props.insert("content".into(), Value::String(msg.content.clone()));
    props.insert(
        "content_type".into(),
        Value::String(msg.content_type.clone()),
    );
    props.insert("timestamp".into(), ts_value.clone());
    let message_nid = kb.create_node("Message", &props).await?;
    let create_ms = create_start.elapsed().as_millis();

    // 5-6. Create edges.
    let edges_start = std::time::Instant::now();
    let mut sent_by_props = HashMap::new();
    sent_by_props.insert("role".into(), Value::String("user".to_string()));
    kb.create_edge("SENT_BY", message_nid, participant_nid, &sent_by_props)
        .await?;

    kb.create_edge("IN_SESSION", message_nid, session_nid, &HashMap::new())
        .await?;

    ensure_participated_in(kb, participant_nid, session_nid).await?;

    create_addressed_to_edges(kb, message_nid, msg, participant_nid, session_nid).await?;

    // 7. Create NEXT edge to link to previous message.
    if let Some(prev_nid) = session_ctx.prev_message_nid {
        let mut edge_props = HashMap::new();
        edge_props.insert("gap_ms".into(), Value::Int(0));
        kb.create_edge("NEXT", prev_nid, message_nid, &edge_props)
            .await?;
    }
    session_ctx.prev_message_nid = Some(message_nid);
    let edges_ms = edges_start.elapsed().as_millis();

    // 8. Chunk long messages.
    let chunk_start = std::time::Instant::now();
    let chunk_threshold = kb.config().message_chunk_threshold;
    let chunk_node_ids = if count_tokens(&msg.content) > chunk_threshold {
        let chunk_cfg = ChunkConfig::from_uniko_config(kb.config());
        let chunker = select_chunker(&msg.content_type, None);
        let chunks = chunker.chunk(&msg.content, &chunk_cfg);
        create_chunks(kb, &msg.message_id, message_nid, &chunks, "Message").await?
    } else {
        Vec::new()
    };
    let chunk_ms = chunk_start.elapsed().as_millis();

    tracing::info!(
        message_id = %msg.message_id,
        setup_ms,
        create_ms,
        edges_ms,
        chunk_ms,
        total_ms = ingest_start.elapsed().as_millis(),
        "message ingest",
    );

    Ok(MessageIngestResult {
        message_node_id: message_nid,
        chunk_node_ids,
        session_node_id: session_nid,
    })
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
            let nid =
                super::session::ensure_participant(kb, pid, &msg.timestamp.to_rfc3339()).await?;
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
