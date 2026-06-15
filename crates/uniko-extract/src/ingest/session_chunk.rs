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

use std::collections::{HashMap, HashSet};

use uniko_store::{KnowledgeBase, NodeId, Value};

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
    let rows = kb.session_transcript_rows(session_id).await?;

    if rows.is_empty() {
        tracing::debug!(session_id, "no messages in session");
        return Ok(Vec::new());
    }

    // Concatenate messages with speaker prefixes.
    let mut transcript = String::new();
    let mut speakers = HashSet::new();

    for row in &rows {
        if !row.content.is_empty() {
            speakers.insert(row.speaker.clone());
            transcript.push_str(&row.speaker);
            transcript.push_str(": ");
            transcript.push_str(&row.content);
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
        messages = rows.len(),
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
    let existing_chunk_ids = kb.session_observation_chunk_ids(session_id).await?;
    if !existing_chunk_ids.is_empty() {
        tracing::debug!(
            session_id,
            chunks = existing_chunk_ids.len(),
            "observation chunks already exist"
        );
        return Ok(existing_chunk_ids);
    }

    // Query all observations in this session.
    let rows = kb.session_observation_rows(session_id).await?;

    if rows.is_empty() {
        tracing::debug!(session_id, "no observations in session");
        return Ok(Vec::new());
    }

    // Deduplicate and collect observations.
    let mut seen = HashSet::new();
    let mut text_block = String::new();
    let mut subjects = HashSet::new();

    for row in &rows {
        if row.content.is_empty() {
            continue;
        }

        // Deduplicate by normalized (lowercased, trimmed) content.
        let key = row.content.trim().to_lowercase();
        if !seen.insert(key) {
            continue;
        }

        if !row.subject.is_empty() {
            subjects.insert(row.subject.clone());
        }
        text_block.push_str(row.content.trim());
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
    let chunk_nids = create_chunks(kb, &obs_ext_id, session_nid, &chunks, "Session").await?;

    // Wire ABOUT edges from observation chunks to entities.
    let entity_nids = kb.session_observation_entity_ids(session_id).await?;

    if !entity_nids.is_empty() {
        let about_edges: Vec<(NodeId, NodeId, HashMap<String, Value>)> = chunk_nids
            .iter()
            .flat_map(|&chunk_nid| {
                entity_nids
                    .iter()
                    .map(move |&entity_nid| (chunk_nid, entity_nid, HashMap::new()))
            })
            .collect();
        // Observation chunks → entities. Source is :Chunk, target is :Entity.
        let about_start = std::time::Instant::now();
        kb.batch_create_edges_fast(
            "ABOUT",
            Some(uniko_store::schema::constants::labels::CHUNK),
            Some(uniko_store::schema::constants::labels::ENTITY),
            &about_edges,
        )
        .await?;
        let ms = about_start.elapsed().as_millis() as u64;
        tracing::info!(
            target: "tx_perf",
            tx_phase = "obs_chunk_about_entity",
            total_ms = ms,
            commit_ms = ms,
            edge_count = about_edges.len() as u64,
            "tx phase",
        );
    }

    // Also propagate Participant ABOUT edges so an obs Chunk is
    // reachable via the participant (e.g. Caroline) — needed for
    // entity-anchored multi-hop recall.
    let participant_nids = kb.session_observation_participant_ids(session_id).await?;
    if !participant_nids.is_empty() {
        let p_edges: Vec<(NodeId, NodeId, HashMap<String, Value>)> = chunk_nids
            .iter()
            .flat_map(|&chunk_nid| {
                participant_nids
                    .iter()
                    .map(move |&p_nid| (chunk_nid, p_nid, HashMap::new()))
            })
            .collect();
        let p_start = std::time::Instant::now();
        kb.batch_create_edges_fast(
            "ABOUT",
            Some(uniko_store::schema::constants::labels::CHUNK),
            Some(uniko_store::schema::constants::labels::PARTICIPANT),
            &p_edges,
        )
        .await?;
        let ms = p_start.elapsed().as_millis() as u64;
        tracing::info!(
            target: "tx_perf",
            tx_phase = "obs_chunk_about_participant",
            total_ms = ms,
            commit_ms = ms,
            edge_count = p_edges.len() as u64,
            "tx phase",
        );
    }

    tracing::info!(
        session_id,
        observations = seen.len(),
        chunks = chunk_nids.len(),
        entities = entity_nids.len(),
        participants = participant_nids.len(),
        "session observations chunked",
    );

    Ok(chunk_nids)
}
