//! Session-level chunking for retrieval.
//!
//! Two chunking strategies:
//!
//! 1. **Transcript chunks** ([`chunk_session`]): concatenate all messages
//!    with speaker prefixes, then chunk into 400-512 token segments.
//!
//! 2. **Observation chunks** ([`chunk_session_observations`]): aggregate
//!    per-message Observation content into dense factual chunks. Individual
//!    Observation nodes are trace-only (no indexes); these chunks are the
//!    searchable representation.

// Rust guideline compliant

use std::collections::{HashMap, HashSet};

use uniko_store::{KnowledgeBase, NodeId, UnikoError};

use super::chunking::text::TextChunker;
use super::chunking::{ChunkConfig, ChunkData, Chunker};
use super::message::create_chunks;

/// Chunk all messages in a session into searchable Chunk nodes.
///
/// Queries messages in the session ordered by timestamp, concatenates
/// them with `"speaker: content\n"` prefixes, then runs the text
/// chunker. Creates Chunk nodes linked to the Session via HAS_CHUNK.
///
/// Idempotent: skips if chunks already exist for this session.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] on graph query or write failure.
pub async fn chunk_session(
    kb: &KnowledgeBase,
    session_id: &str,
) -> uniko_store::Result<Vec<NodeId>> {
    // Look up the session node.
    let session_nid = match kb
        .get_node_by_ext_id("Session", "session_id", session_id)
        .await?
    {
        Some((nid, _)) => nid,
        None => {
            tracing::debug!(session_id, "session not found, skipping chunking");
            return Ok(Vec::new());
        }
    };

    // Check if chunks already exist (idempotent).
    let existing = kb
        .get_edges(
            session_nid,
            "HAS_CHUNK",
            uniko_store::storage::edges::Direction::Outgoing,
        )
        .await?;
    if !existing.is_empty() {
        tracing::debug!(
            session_id,
            chunks = existing.len(),
            "session already chunked"
        );
        return Ok(existing.iter().map(|e| e.to).collect());
    }

    // Query all messages in this session with their sender names,
    // ordered by timestamp.
    let cypher = "\
        MATCH (m:Message)-[:IN_SESSION]->(s:Session {session_id: $sid}) \
        OPTIONAL MATCH (m)-[:SENT_BY]->(p:Participant) \
        RETURN m.content AS content, p.name AS speaker, m.timestamp AS ts \
        ORDER BY m.timestamp";

    let session = kb.db().session();
    let result = session
        .query_with(cypher)
        .param("sid", session_id)
        .fetch_all()
        .await
        .map_err(|e| UnikoError::Storage(e.to_string()))?;

    if result.is_empty() {
        tracing::debug!(session_id, "no messages in session");
        return Ok(Vec::new());
    }

    // Concatenate messages with speaker prefixes.
    let mut transcript = String::new();
    let mut speakers = HashSet::new();

    for row in result.rows() {
        let content: String = row.get("content").unwrap_or_default();
        let speaker: String = row.get("speaker").unwrap_or_else(|_| "unknown".to_string());

        if !content.is_empty() {
            speakers.insert(speaker.clone());
            transcript.push_str(&speaker);
            transcript.push_str(": ");
            transcript.push_str(&content);
            transcript.push('\n');
        }
    }

    if transcript.is_empty() {
        return Ok(Vec::new());
    }

    // Chunk the transcript.
    let chunk_cfg = ChunkConfig::from_uniko_config(kb.config());
    let chunker = TextChunker;
    let mut chunks: Vec<ChunkData> = Chunker::chunk(&chunker, &transcript, &chunk_cfg);

    // Set metadata on each chunk.
    let speaker_list = {
        let mut sorted: Vec<&str> = speakers.iter().map(String::as_str).collect();
        sorted.sort_unstable();
        sorted.join(", ")
    };

    // Get session topic if available.
    let topic = kb.get_node(session_nid).await?.and_then(|(_, props)| {
        props
            .get("topic")
            .and_then(|v| v.as_str())
            .map(String::from)
    });

    for chunk in &mut chunks {
        chunk.chunk_type = "session".to_string();
        chunk.heading = topic.clone();
        // Store speakers in symbol_name field (reusing existing schema field).
        chunk.symbol_name = Some(speaker_list.clone());
    }

    // Create chunk nodes and HAS_CHUNK edges.
    let chunk_nids = create_chunks(kb, session_id, session_nid, &chunks, "Session").await?;

    tracing::info!(
        session_id,
        messages = result.len(),
        chunks = chunk_nids.len(),
        transcript_len = transcript.len(),
        "session chunked",
    );

    Ok(chunk_nids)
}

/// Aggregate per-message Observations into searchable Chunk nodes.
///
/// Queries all Observation nodes linked to messages in this session,
/// deduplicates by normalized content, concatenates into a dense text
/// block, then creates Chunk node(s) with `chunk_type = "observation"`.
///
/// Also wires ABOUT edges from observation chunks to any entities
/// referenced by the underlying observations, preserving entity-scoped
/// search capability.
///
/// Idempotent: skips if observation chunks already exist for this session.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] on graph query or write failure.
pub async fn chunk_session_observations(
    kb: &KnowledgeBase,
    session_id: &str,
) -> uniko_store::Result<Vec<NodeId>> {
    // Look up the session node.
    let session_nid = match kb
        .get_node_by_ext_id("Session", "session_id", session_id)
        .await?
    {
        Some((nid, _)) => nid,
        None => {
            tracing::debug!(session_id, "session not found, skipping obs chunking");
            return Ok(Vec::new());
        }
    };

    // Check if observation chunks already exist (idempotent).
    let check_cypher = "\
        MATCH (s:Session {session_id: $sid})-[:HAS_CHUNK]->(c:Chunk {chunk_type: 'observation'}) \
        RETURN id(c) AS cid";
    let check = kb
        .db()
        .session()
        .query_with(check_cypher)
        .param("sid", session_id)
        .fetch_all()
        .await
        .map_err(|e| UnikoError::Storage(e.to_string()))?;
    if !check.is_empty() {
        tracing::debug!(session_id, chunks = check.len(), "observation chunks already exist");
        let ids: Vec<NodeId> = check
            .rows()
            .iter()
            .filter_map(|r| r.get::<NodeId>("cid").ok())
            .collect();
        return Ok(ids);
    }

    // Query all observations in this session.
    let cypher = "\
        MATCH (o:Observation)-[:OBSERVED_IN]->(m:Message)-[:IN_SESSION]->(s:Session {session_id: $sid}) \
        RETURN o.content AS content, o.subject AS subject \
        ORDER BY m.timestamp";

    let session = kb.db().session();
    let result = session
        .query_with(cypher)
        .param("sid", session_id)
        .fetch_all()
        .await
        .map_err(|e| UnikoError::Storage(e.to_string()))?;

    if result.is_empty() {
        tracing::debug!(session_id, "no observations in session");
        return Ok(Vec::new());
    }

    // Deduplicate and collect observations.
    let mut seen = HashSet::new();
    let mut text_block = String::new();
    let mut subjects = HashSet::new();

    for row in result.rows() {
        let content: String = row.get("content").unwrap_or_default();
        let subject: String = row.get("subject").unwrap_or_default();

        if content.is_empty() {
            continue;
        }

        // Deduplicate by normalized (lowercased, trimmed) content.
        let key = content.trim().to_lowercase();
        if !seen.insert(key) {
            continue;
        }

        if !subject.is_empty() {
            subjects.insert(subject);
        }
        text_block.push_str(content.trim());
        text_block.push('\n');
    }

    if text_block.is_empty() {
        return Ok(Vec::new());
    }

    // Chunk the observation text.
    let chunk_cfg = ChunkConfig::from_uniko_config(kb.config());
    let chunker = TextChunker;
    let mut chunks: Vec<ChunkData> = Chunker::chunk(&chunker, &text_block, &chunk_cfg);

    // Set metadata on each chunk.
    let subject_list = {
        let mut sorted: Vec<&str> = subjects.iter().map(String::as_str).collect();
        sorted.sort_unstable();
        sorted.join(", ")
    };

    let topic = kb.get_node(session_nid).await?.and_then(|(_, props)| {
        props
            .get("topic")
            .and_then(|v| v.as_str())
            .map(String::from)
    });

    for chunk in &mut chunks {
        chunk.chunk_type = "observation".to_string();
        chunk.heading = topic.clone();
        chunk.symbol_name = Some(subject_list.clone());
    }

    // Create chunk nodes and HAS_CHUNK edges from Session.
    let obs_ext_id = format!("{session_id}:obs");
    let chunk_nids =
        create_chunks(kb, &obs_ext_id, session_nid, &chunks, "Session").await?;

    // Wire ABOUT edges from observation chunks to entities.
    let entity_cypher = "\
        MATCH (o:Observation)-[:OBSERVED_IN]->(m:Message)-[:IN_SESSION]->(s:Session {session_id: $sid}) \
        MATCH (o)-[:ABOUT]->(e:Entity) \
        RETURN DISTINCT id(e) AS eid";

    let entity_result = session
        .query_with(entity_cypher)
        .param("sid", session_id)
        .fetch_all()
        .await
        .map_err(|e| UnikoError::Storage(e.to_string()))?;

    let entity_nids: Vec<NodeId> = entity_result
        .rows()
        .iter()
        .filter_map(|row| row.get::<NodeId>("eid").ok())
        .collect();

    if !entity_nids.is_empty() {
        let about_edges: Vec<(NodeId, NodeId, HashMap<String, uni_db::Value>)> = chunk_nids
            .iter()
            .flat_map(|&chunk_nid| {
                entity_nids
                    .iter()
                    .map(move |&entity_nid| (chunk_nid, entity_nid, HashMap::new()))
            })
            .collect();
        kb.batch_create_edges("ABOUT", &about_edges).await?;
    }

    tracing::info!(
        session_id,
        observations = seen.len(),
        chunks = chunk_nids.len(),
        entities = entity_nids.len(),
        "session observations chunked",
    );

    Ok(chunk_nids)
}