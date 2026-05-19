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
    /// `(participant_node_id, participant_name)` for the sender —
    /// `Some` on the create path so downstream pipeline steps can
    /// skip a `SENT_BY` lookup + Participant property fetch. `None`
    /// when the idempotency path returned an already-existing
    /// Message (callers fall back to a DB lookup in that case).
    pub sender: Option<(NodeId, String)>,
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
            // Skipped on idempotency: callers fall back to a SENT_BY
            // DB lookup if they need the sender for a re-ingest.
            sender: None,
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

    // 4-8. All per-message writes (Message node, its edges, optional
    // chunks) run inside ONE transaction. Replaces what used to be
    // 2-4 separate commits per message; uni-db's commit cost
    // (~12 ms WAL+index per tx) is the same regardless of how much
    // is bundled in, so folding cuts message-ingest tx overhead by
    // ~3×.
    let session = kb.db().session();
    let tx = session
        .tx()
        .await
        .map_err(|e| uniko_store::UnikoError::Storage(e.to_string()))?;

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
    let message_nid = kb.create_node_in_tx(&tx, "Message", &props).await?;
    let create_ms = create_start.elapsed().as_millis();

    // 5-7. Create all per-message edges in ONE Cypher statement
    // inside the same transaction. At sess=24 the per-message write
    // phase was bottlenecked by writer-lock contention across the
    // 4 separate UNWIND-by-type statements that the generic
    // `create_edges_in_tx` would issue; coalescing to a single
    // statement cuts the per-message writer-lock acquisitions 4 → 1.
    let edges_start = std::time::Instant::now();
    let recipient_nids = resolve_recipients(kb, msg, participant_nid, session_ctx).await?;
    let edge_count =
        2 + recipient_nids.len() + usize::from(session_ctx.prev_message_nid.is_some());
    kb.create_message_edges_in_tx(
        &tx,
        message_nid,
        participant_nid,
        "user",
        session_nid,
        &recipient_nids,
        session_ctx.prev_message_nid,
    )
    .await?;
    session_ctx.prev_message_nid = Some(message_nid);
    let edges_ms = edges_start.elapsed().as_millis();

    // 8. Chunk long messages — also in the shared tx.
    let chunk_start = std::time::Instant::now();
    let chunk_threshold = kb.config().message_chunk_threshold;
    let chunk_node_ids = if count_tokens(&msg.content) > chunk_threshold {
        let chunk_cfg = ChunkConfig::from_uniko_config(kb.config());
        let chunker = select_chunker(&msg.content_type, None);
        let chunks = chunker.chunk(&msg.content, &chunk_cfg);
        create_chunks_in_tx(kb, &tx, &msg.message_id, message_nid, &chunks, "Message").await?
    } else {
        Vec::new()
    };
    let chunk_ms = chunk_start.elapsed().as_millis();

    // Commit folds Message + every edge + any chunks under one
    // WAL fsync.
    let commit_start = std::time::Instant::now();
    tx.commit()
        .await
        .map_err(|e| uniko_store::UnikoError::Storage(e.to_string()))?;
    let commit_ms = commit_start.elapsed().as_millis();

    let total_ms = ingest_start.elapsed().as_millis() as u64;
    tracing::info!(
        message_id = %msg.message_id,
        setup_ms,
        create_ms,
        edges_ms,
        edge_count,
        chunk_ms,
        commit_ms,
        total_ms,
        "message ingest",
    );
    tracing::info!(
        target: "tx_perf",
        tx_phase = "message_ingest",
        total_ms,
        commit_ms = commit_ms as u64,
        "tx phase",
    );

    Ok(MessageIngestResult {
        message_node_id: message_nid,
        chunk_node_ids,
        session_node_id: session_nid,
        // `ensure_participant` stores `Participant.name = sender_id`,
        // so the in-Rust string matches what the observation step
        // would otherwise refetch via `get_node(participant_nid)`.
        sender: Some((participant_nid, msg.sender_id.clone())),
    })
}

/// Resolve ADDRESSED_TO recipient node IDs.
///
/// If `msg.addressed_to` is provided, ensures each participant exists
/// and returns their node IDs. Otherwise infers recipients from the
/// in-memory [`SessionContext::participants`] cache — every participant
/// who has previously sent a message in this session was registered
/// there during their first message's ingest.
///
/// The cache and the on-disk PARTICIPATED_IN-edge set are populated by
/// the same code path (only on sender first-sight in `ingest_message`),
/// so the cache is exactly equivalent to the previous
/// `get_edges(... PARTICIPATED_IN ...)` query — without the per-message
/// DB read that was costing ~30% of `edges_ms` at sess=24.
async fn resolve_recipients(
    kb: &KnowledgeBase,
    msg: &IngestMessage,
    sender_nid: NodeId,
    session_ctx: &super::context::SessionContext,
) -> uniko_store::Result<Vec<NodeId>> {
    if let Some(ref ids) = msg.addressed_to {
        let mut nids = Vec::with_capacity(ids.len());
        for pid in ids {
            // Hit the cache first; only `ensure_participant` (a
            // merge_node DB roundtrip) on miss.
            let nid = match session_ctx.participant_nid(pid) {
                Some(nid) if nid != 0 => nid,
                _ => super::session::ensure_participant(kb, pid, msg.timestamp).await?,
            };
            nids.push(nid);
        }
        Ok(nids)
    } else {
        Ok(session_ctx
            .participants
            .values()
            .copied()
            .filter(|&nid| nid != sender_nid && nid != 0)
            .collect())
    }
}

/// Create Chunk nodes + HAS_CHUNK edges for a parent node.
///
/// # Errors
///
/// Returns a storage error if either the node create or edge create
/// batch fails.
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
    let start = std::time::Instant::now();
    let session = kb.db().session();
    let tx = session
        .tx()
        .await
        .map_err(|e| uniko_store::UnikoError::Storage(e.to_string()))?;
    let nids =
        create_chunks_in_tx(kb, &tx, parent_ext_id, parent_nid, chunks, parent_label).await?;
    let commit_start = std::time::Instant::now();
    tx.commit()
        .await
        .map_err(|e| uniko_store::UnikoError::Storage(e.to_string()))?;
    let commit_ms = commit_start.elapsed().as_millis() as u64;
    let total_ms = start.elapsed().as_millis() as u64;
    tracing::info!(
        target: "tx_perf",
        tx_phase = match parent_label {
            "Session" => "session_chunks",
            "Message" => "message_chunks_standalone",
            _ => "chunks_standalone",
        },
        total_ms,
        commit_ms,
        chunk_count = chunks.len() as u64,
        "tx phase",
    );
    Ok(nids)
}

/// Same as [`create_chunks`] but defers commit to the caller's tx.
///
/// Used by `ingest_message` to fold Chunk creation under the same
/// transaction as the Message node and its edges.
///
/// # Errors
///
/// Returns a storage error if either batched write fails.
pub async fn create_chunks_in_tx(
    kb: &KnowledgeBase,
    tx: &uni_db::Transaction,
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

    let chunk_nids = kb
        .batch_create_nodes_in_tx(tx, "Chunk", &chunk_props)
        .await?;

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
    kb.batch_create_edges_fast_in_tx(
        tx,
        "HAS_CHUNK",
        Some(parent_label),
        Some(uniko_store::schema::constants::labels::CHUNK),
        &edges,
    )
    .await?;

    Ok(chunk_nids)
}
