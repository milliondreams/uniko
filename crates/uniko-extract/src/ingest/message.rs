//! Message ingest: create the Message node and all associated edges.

// Rust guideline compliant

use std::collections::HashMap;

use uni_db::Value;

use uniko_pipes::types::IngestMessage;
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, NodeId};

use super::chunking::{ChunkConfig, count_tokens, select_chunker};
use super::session::{ensure_participant, get_or_create_session, link_participant_to_session};

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

    // 2. Ensure Session exists (use cache when available).
    let session_nid = if session_ctx.session_nid != 0 {
        session_ctx.session_nid
    } else {
        let nid = get_or_create_session(kb, &msg.session_id, &msg.timestamp).await?;
        session_ctx.session_nid = nid;
        nid
    };

    // 3. Ensure Participant exists and is linked to session.
    //    link_participant_to_session is called once per participant per
    //    session (on first sight), not per message.
    let participant_nid = if let Some(nid) = session_ctx.participant_nid(&msg.sender_id) {
        if nid != 0 {
            nid
        } else {
            let nid = ensure_participant(kb, &msg.sender_id, msg.timestamp).await?;
            link_participant_to_session(kb, nid, session_nid).await?;
            session_ctx.register_participant(&msg.sender_id, nid);
            nid
        }
    } else {
        let nid = ensure_participant(kb, &msg.sender_id, msg.timestamp).await?;
        link_participant_to_session(kb, nid, session_nid).await?;
        session_ctx.register_participant(&msg.sender_id, nid);
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

    // 5-7. Collect and create all edges in a single transaction.
    let edges_start = std::time::Instant::now();

    let mut edge_specs: Vec<(&str, NodeId, NodeId, HashMap<String, Value>)> = Vec::with_capacity(4);

    // SENT_BY
    let mut sent_by_props = HashMap::new();
    sent_by_props.insert("role".into(), Value::String("user".to_string()));
    edge_specs.push(("SENT_BY", message_nid, participant_nid, sent_by_props));

    // IN_SESSION
    edge_specs.push(("IN_SESSION", message_nid, session_nid, HashMap::new()));

    // ADDRESSED_TO
    let recipient_nids =
        resolve_recipients(kb, msg, participant_nid, session_nid).await?;
    for &rnid in &recipient_nids {
        edge_specs.push(("ADDRESSED_TO", message_nid, rnid, HashMap::new()));
    }

    // NEXT
    if let Some(prev_nid) = session_ctx.prev_message_nid {
        let mut next_props = HashMap::new();
        next_props.insert("gap_ms".into(), Value::Int(0));
        edge_specs.push(("NEXT", prev_nid, message_nid, next_props));
    }

    kb.create_edges_tx(&edge_specs).await?;
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
        edge_count = edge_specs.len(),
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

/// Resolve ADDRESSED_TO recipient node IDs.
///
/// If `msg.addressed_to` is provided, ensures each participant exists
/// and returns their node IDs.  Otherwise infers recipients from the
/// session's PARTICIPATED_IN edges (all participants except sender).
async fn resolve_recipients(
    kb: &KnowledgeBase,
    msg: &IngestMessage,
    sender_nid: NodeId,
    session_nid: NodeId,
) -> uniko_store::Result<Vec<NodeId>> {
    use uniko_store::schema::constants::edges;
    use uniko_store::storage::edges::Direction;

    if let Some(ref ids) = msg.addressed_to {
        let mut nids = Vec::with_capacity(ids.len());
        for pid in ids {
            let nid =
                super::session::ensure_participant(kb, pid, msg.timestamp).await?;
            nids.push(nid);
        }
        Ok(nids)
    } else {
        let participated = kb
            .get_edges(session_nid, edges::PARTICIPATED_IN, Direction::Incoming)
            .await?;
        Ok(participated
            .iter()
            .filter(|e| e.from != sender_nid)
            .map(|e| e.from)
            .collect())
    }
}

/// Create Chunk nodes + HAS_CHUNK edges for a parent node.
pub async fn create_chunks(
    kb: &KnowledgeBase,
    parent_ext_id: &str,
    parent_nid: NodeId,
    chunks: &[super::chunking::ChunkData],
    parent_label: &str,
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

    // Source label is dynamic (Message, Session, or Artifact); target is :Chunk.
    kb.batch_create_edges_fast(
        "HAS_CHUNK",
        Some(parent_label),
        Some(uniko_store::schema::constants::labels::CHUNK),
        &edges,
    )
    .await?;

    Ok(chunk_nids)
}
