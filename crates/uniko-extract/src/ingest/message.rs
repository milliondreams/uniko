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

    // 2 + 3. Ensure Session and sender Participant exist (own commits;
    // first-sight only).
    let (session_nid, participant_nid) = ensure_session_and_sender(kb, msg, session_ctx).await?;
    let setup_ms = ingest_start.elapsed().as_millis();

    // 4 + 5. Resolve recipients (uses the participant cache) and snap
    // the prev_message_nid BEFORE opening the tx so apply_message_writes_in_tx
    // sees the value as of this call (we update session_ctx.prev_message_nid
    // after the write succeeds).
    let recipient_nids = resolve_recipients(kb, msg, participant_nid, session_ctx).await?;
    let setup = MessageSetup {
        session_nid,
        participant_nid,
        recipient_nids,
        prev_msg_nid: session_ctx.prev_message_nid,
    };

    // 6-10. All per-message writes (Message node, its edges, optional
    // chunks) run inside ONE transaction.
    let session = kb.db().session();
    let tx = session
        .tx()
        .await
        .map_err(|e| uniko_store::UnikoError::Storage(e.to_string()))?;
    let writes = apply_message_writes_in_tx(kb, &tx, msg, &setup).await?;

    let commit_start = std::time::Instant::now();
    tx.commit()
        .await
        .map_err(|e| uniko_store::UnikoError::Storage(e.to_string()))?;
    let commit_ms = commit_start.elapsed().as_millis();
    session_ctx.prev_message_nid = Some(writes.message_node_id);

    let total_ms = ingest_start.elapsed().as_millis() as u64;
    tracing::info!(
        message_id = %msg.message_id,
        setup_ms,
        create_ms = writes.create_ms,
        edges_ms = writes.edges_ms,
        edge_count = writes.edge_count,
        chunk_ms = writes.chunk_ms,
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
        message_node_id: writes.message_node_id,
        chunk_node_ids: writes.chunk_node_ids,
        session_node_id: session_nid,
        // `ensure_participant` stores `Participant.name = sender_id`,
        // so the in-Rust string matches what the observation step
        // would otherwise refetch via `get_node(participant_nid)`.
        sender: Some((participant_nid, msg.sender_id.clone())),
    })
}

/// Inputs that the orchestrator (or `ingest_message`) computes in
/// pre-tx phase and hands to [`apply_message_writes_in_tx`].
#[derive(Debug)]
pub struct MessageSetup {
    pub session_nid: NodeId,
    pub participant_nid: NodeId,
    pub recipient_nids: Vec<NodeId>,
    /// Snapshot of `session_ctx.prev_message_nid` BEFORE this message.
    /// Caller is responsible for updating `session_ctx` after a
    /// successful commit.
    pub prev_msg_nid: Option<NodeId>,
}

/// Per-phase write metrics; useful for callers that want to log a
/// granular breakdown (the legacy `ingest_message` and the atomic
/// orchestrator both do).
#[derive(Debug)]
pub struct MessageWriteResult {
    pub message_node_id: NodeId,
    pub chunk_node_ids: Vec<NodeId>,
    pub create_ms: u128,
    pub edges_ms: u128,
    pub chunk_ms: u128,
    pub edge_count: usize,
}

/// Write Message + edges + chunks into the caller's tx.
///
/// All pre-tx setup (idempotency, session/participant ensure, recipient
/// resolution) is the caller's responsibility — supply
/// [`MessageSetup`] with the resolved ids. Does NOT commit.
///
/// # Errors
///
/// Returns [`uniko_store::UnikoError`] on any underlying write failure.
pub(crate) async fn apply_message_writes_in_tx(
    kb: &KnowledgeBase,
    tx: &uni_db::Transaction,
    msg: &IngestMessage,
    setup: &MessageSetup,
) -> uniko_store::Result<MessageWriteResult> {
    let ts_value = datetime_value(msg.timestamp);

    // Create Message node (triggers auto-embed on content via uni-db's
    // process_embeddings_for_batch, which acquires the BGE-small ORT
    // Session Mutex).
    //
    // Measurement-only knob: UNIKO_BENCH_NO_MSG_EMBED=1 pre-populates
    // a zero embedding to skip auto-embed for A/B testing. NOT a
    // production code path — recall results from such runs are invalid.
    let create_start = std::time::Instant::now();
    let mut props = HashMap::new();
    props.insert("message_id".into(), Value::String(msg.message_id.clone()));
    props.insert("content".into(), Value::String(msg.content.clone()));
    props.insert(
        "content_type".into(),
        Value::String(msg.content_type.clone()),
    );
    props.insert("timestamp".into(), ts_value);
    if std::env::var("UNIKO_BENCH_NO_MSG_EMBED").is_ok() {
        let zero_vec: Vec<Value> = (0..384).map(|_| Value::Float(0.0)).collect();
        props.insert("embedding".into(), Value::List(zero_vec));
    }
    let message_nid = kb.create_node_in_tx(tx, "Message", &props).await?;
    let create_ms = create_start.elapsed().as_millis();

    // Create all per-message edges in ONE Cypher statement.
    let edges_start = std::time::Instant::now();
    let edge_count = 2 + setup.recipient_nids.len() + usize::from(setup.prev_msg_nid.is_some());
    kb.create_message_edges_in_tx(
        tx,
        message_nid,
        setup.participant_nid,
        "user",
        setup.session_nid,
        &setup.recipient_nids,
        setup.prev_msg_nid,
    )
    .await?;
    let edges_ms = edges_start.elapsed().as_millis();

    // Chunk long messages — in the same tx.
    let chunk_start = std::time::Instant::now();
    let chunk_threshold = kb.config().message_chunk_threshold;
    let chunk_node_ids = if count_tokens(&msg.content) > chunk_threshold {
        let chunk_cfg = ChunkConfig::from_uniko_config(kb.config());
        let chunker = select_chunker(&msg.content_type, None);
        let chunks = chunker.chunk(&msg.content, &chunk_cfg);
        create_chunks_in_tx(kb, tx, &msg.message_id, message_nid, &chunks, "Message").await?
    } else {
        Vec::new()
    };
    let chunk_ms = chunk_start.elapsed().as_millis();

    Ok(MessageWriteResult {
        message_node_id: message_nid,
        chunk_node_ids,
        create_ms,
        edges_ms,
        chunk_ms,
        edge_count,
    })
}

/// First-sight setup for Session + sender Participant (own commits today).
/// Used by both `ingest_message` and the atomic orchestrator; uses
/// `SessionContext` caches to skip on repeat invocations.
pub(crate) async fn ensure_session_and_sender(
    kb: &KnowledgeBase,
    msg: &IngestMessage,
    session_ctx: &mut super::context::SessionContext,
) -> uniko_store::Result<(NodeId, NodeId)> {
    let session_nid = if session_ctx.session_nid != 0 {
        session_ctx.session_nid
    } else {
        let nid = get_or_create_session(kb, &msg.session_id, &msg.timestamp).await?;
        session_ctx.session_nid = nid;
        nid
    };
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
    Ok((session_nid, participant_nid))
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
pub(crate) async fn resolve_recipients(
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
            if let Some(ref meta) = c.metadata {
                props.insert("metadata".into(), json_to_uni_value(meta));
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

/// Convert a `serde_json::Value` into a `uni_db::Value` suitable for
/// storage in a `DataType::CypherValue` column.
///
/// JSON numbers split across `Value::Int` (when they fit in `i64` and
/// have no fractional part) or `Value::Float` otherwise. Nulls become
/// `Value::Null`; arrays / objects recurse.
fn json_to_uni_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                // u64 > i64::MAX — fall back to string to avoid lossy cast.
                Value::String(n.to_string())
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(arr) => Value::List(arr.iter().map(json_to_uni_value).collect()),
        serde_json::Value::Object(obj) => {
            let mut m: HashMap<String, Value> = HashMap::with_capacity(obj.len());
            for (k, v) in obj {
                m.insert(k.clone(), json_to_uni_value(v));
            }
            Value::Map(m)
        }
    }
}
