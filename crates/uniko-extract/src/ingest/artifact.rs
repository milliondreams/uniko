//! Artifact ingest: hash, dedup, create Artifact node, chunk, and link.

use std::collections::HashMap;

use sha2::{Digest, Sha256};
use uniko_store::Value;

use uniko_pipes::types::IngestArtifact;
use uniko_store::storage::blob::MergeContent;
use uniko_store::{KnowledgeBase, NodeId};

use super::chunking::{ChunkConfig, select_chunker};
use super::message::create_chunks;

/// Result of ingesting a single artifact.
#[derive(Debug)]
pub struct ArtifactIngestResult {
    /// Internal node ID of the Artifact.
    pub artifact_node_id: NodeId,
    /// Node IDs of the created chunks.
    pub chunk_node_ids: Vec<NodeId>,
    /// Whether this artifact was a duplicate (skipped chunking).
    pub was_deduplicated: bool,
}

/// Ingest an artifact into the knowledge graph.
///
/// Computes a SHA-256 content hash for deduplication.  If the hash
/// already exists, returns the existing artifact without re-chunking.
/// Otherwise creates the Artifact node, selects a chunker based on
/// content type, and creates Chunk nodes with HAS_CHUNK edges.
///
/// # Errors
///
/// Returns a storage error if any graph operation fails.
pub async fn ingest_artifact(
    kb: &KnowledgeBase,
    artifact: &IngestArtifact,
) -> uniko_store::Result<ArtifactIngestResult> {
    // 1. Compute content hash.
    let hash = hex::encode(Sha256::digest(artifact.content.as_bytes()));

    // 2. Dedup: check if an artifact with this hash already exists.
    if let Some((existing_id, _)) = kb.get_node_by_ext_id("Artifact", "hash", &hash).await? {
        return Ok(ArtifactIngestResult {
            artifact_node_id: existing_id,
            chunk_node_ids: Vec::new(),
            was_deduplicated: true,
        });
    }

    // Also check by artifact_id for idempotency.
    if let Some((existing_id, _)) = kb
        .get_node_by_ext_id("Artifact", "artifact_id", &artifact.artifact_id)
        .await?
    {
        return Ok(ArtifactIngestResult {
            artifact_node_id: existing_id,
            chunk_node_ids: Vec::new(),
            was_deduplicated: true,
        });
    }

    // 3. Detect language from file extension.
    let language = detect_language(artifact.path.as_deref());

    // 4. Backend PUT + MERGE :ArtifactContent. Per §5.1 step 4-5 of
    // multimodal-knowledge-store-design.md. For Lance backend the
    // bytes flow back through `PutOutcome::bytes_inline` so the
    // MERGE writes them inline; for Fs/S3 the bytes are already
    // persisted by `put_blob` and the MERGE only stores the `uri`.
    let bytes = artifact.content.as_bytes();
    let size = bytes.len() as i64;
    let put = kb.put_blob(&hash, bytes).await?;
    let content_nid = kb
        .merge_artifact_content(MergeContent {
            content_id: hash.clone(),
            bytes: put.bytes_inline,
            uri: put.uri,
            mime: mime_for_kind(&artifact.kind, language.as_deref()),
            size,
            perceptual_hash: None,
            audio_fingerprint: None,
        })
        .await?;

    // 5. Create :Artifact metadata node (no `content` — that's on
    // :ArtifactContent now). The `hash` field stays as a denorm cache
    // of `HAS_CONTENT.target.content_id` (§4.5).
    let mut props = HashMap::new();
    props.insert(
        "artifact_id".into(),
        Value::String(artifact.artifact_id.clone()),
    );
    props.insert("kind".into(), Value::String(artifact.kind.clone()));
    if let Some(ref path) = artifact.path {
        props.insert("path".into(), Value::String(path.clone()));
    }
    props.insert("hash".into(), Value::String(hash));
    props.insert("size".into(), Value::Int(size));
    if let Some(ref lang) = language {
        props.insert("language".into(), Value::String(lang.clone()));
    }
    let artifact_nid = kb.create_node("Artifact", &props).await?;

    // 6. HAS_CONTENT edge: Artifact → ArtifactContent, role="primary".
    let mut edge_props = HashMap::new();
    edge_props.insert("role".into(), Value::String("primary".into()));
    kb.create_edge("HAS_CONTENT", artifact_nid, content_nid, &edge_props)
        .await?;

    // 6b. Contextual provenance (F18/F22/F30). Best-effort: a missing
    // referenced node is skipped with a debug rather than failing
    // ingestion. Corpus ingestion (no context fields) wires nothing.
    link_artifact_context(kb, artifact_nid, artifact).await?;

    // 5. Select chunker and create chunks. A caller-supplied
    // `metadata["content_type"]` wins (e.g. a fetched URL whose bytes
    // are HTML), so URL / upload ingestion uses the right chunker even
    // without a file extension; otherwise infer from detected language.
    let hinted_content_type = artifact
        .metadata
        .get("content_type")
        .and_then(|v| v.as_str());
    let content_type =
        hinted_content_type.unwrap_or(if language.is_some() { "code" } else { "text" });
    let chunker = select_chunker(content_type, language.as_deref());
    let chunk_cfg = ChunkConfig::from_uniko_config(kb.config());
    let chunks = chunker.chunk(&artifact.content, &chunk_cfg);

    let chunk_nids =
        create_chunks(kb, &artifact.artifact_id, artifact_nid, &chunks, "Artifact").await?;

    // Populate Artifact.text_embedding by mean-pooling child Chunk
    // embeddings. No-op when chunks haven't auto-embedded yet (e.g.,
    // in unit tests that don't configure an embedder); the backfill
    // migration fills these in post-hoc.
    if !chunk_nids.is_empty()
        && let Err(e) = kb.mean_pool_artifact_text_embedding(artifact_nid).await
    {
        tracing::warn!(
            target: "uniko_extract::ingest",
            error = %e,
            "mean_pool_artifact_text_embedding failed; leaving NULL for backfill"
        );
    }

    Ok(ArtifactIngestResult {
        artifact_node_id: artifact_nid,
        chunk_node_ids: chunk_nids,
        was_deduplicated: false,
    })
}

/// Wire an artifact's optional conversational-context edges.
///
/// `PRODUCED` (Action → Artifact) when `produced_by_action_id` resolves;
/// `ATTACHED_TO` (Artifact → Session) when `session_id` resolves;
/// `ATTACHED_TO` (Artifact → Message) when `triggered_by_message_id`
/// resolves.  Each is best-effort: an unresolved reference is logged at
/// debug and skipped, so a stale id never aborts ingestion.
async fn link_artifact_context(
    kb: &KnowledgeBase,
    artifact_nid: NodeId,
    artifact: &IngestArtifact,
) -> uniko_store::Result<()> {
    use uniko_store::schema::{edges, labels};

    let attach_props = || {
        let mut p = HashMap::new();
        p.insert(
            "attached_at".into(),
            uniko_store::types::datetime_value(chrono::Utc::now()),
        );
        p
    };

    // PRODUCED: Action → Artifact (note the reversed direction — the
    // producing Action is the edge source, matching record_action).
    if let Some(action_id) = artifact.produced_by_action_id.as_deref() {
        match kb
            .get_node_by_ext_id(labels::ACTION, "action_id", action_id)
            .await?
        {
            Some((action_nid, _)) => {
                kb.create_edge(edges::PRODUCED, action_nid, artifact_nid, &HashMap::new())
                    .await?;
            }
            None => tracing::debug!(
                action_id,
                "ingest_artifact: producing Action not found — PRODUCED skipped"
            ),
        }
    }

    // ATTACHED_TO: Artifact → Session.
    if let Some(session_id) = artifact.session_id.as_deref() {
        match kb
            .get_node_by_ext_id(labels::SESSION, "session_id", session_id)
            .await?
        {
            Some((session_nid, _)) => {
                kb.create_edge(
                    edges::ATTACHED_TO,
                    artifact_nid,
                    session_nid,
                    &attach_props(),
                )
                .await?;
            }
            None => tracing::debug!(
                session_id,
                "ingest_artifact: Session not found — ATTACHED_TO skipped"
            ),
        }
    }

    // ATTACHED_TO: Artifact → Message.
    if let Some(msg_id) = artifact.triggered_by_message_id.as_deref() {
        match kb
            .get_node_by_ext_id(labels::MESSAGE, "message_id", msg_id)
            .await?
        {
            Some((msg_nid, _)) => {
                kb.create_edge(edges::ATTACHED_TO, artifact_nid, msg_nid, &attach_props())
                    .await?;
            }
            None => tracing::debug!(
                msg_id,
                "ingest_artifact: triggering Message not found — ATTACHED_TO skipped"
            ),
        }
    }

    Ok(())
}

/// Best-effort MIME from kind + detected language.
///
/// Text ingest only — caller passes the same `kind` the upstream API
/// used (`"text"`, `"code"`, `"html"`, `"markdown"`, …). Modality-
/// specific ingest paths (image / audio / pdf / video) supply the
/// MIME directly and never call this.
fn mime_for_kind(kind: &str, language: Option<&str>) -> String {
    match (kind, language) {
        (_, Some("python")) => "text/x-python".into(),
        (_, Some("rust")) => "text/x-rust".into(),
        (_, Some("javascript")) => "application/javascript".into(),
        (_, Some("typescript")) => "application/typescript".into(),
        (_, Some("tsx")) => "application/typescript".into(),
        (_, Some("html")) => "text/html".into(),
        (_, Some("css")) => "text/css".into(),
        (_, Some("json")) => "application/json".into(),
        (_, Some("csv")) => "text/csv".into(),
        (_, Some("markdown")) => "text/markdown".into(),
        ("markdown", _) => "text/markdown".into(),
        ("html", _) => "text/html".into(),
        ("code", _) => "text/plain".into(),
        _ => "text/plain".into(),
    }
}

/// Detect programming language from a file path extension.
fn detect_language(path: Option<&str>) -> Option<String> {
    let path = path?;
    let ext = path.rsplit('.').next()?;
    match ext {
        "py" => Some("python".into()),
        "rs" => Some("rust".into()),
        "js" | "mjs" | "cjs" => Some("javascript".into()),
        "ts" => Some("typescript".into()),
        "tsx" => Some("tsx".into()),
        "html" | "htm" => Some("html".into()),
        "css" => Some("css".into()),
        "json" => Some("json".into()),
        "csv" => Some("csv".into()),
        "md" | "markdown" => Some("markdown".into()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language(Some("main.py")), Some("python".into()));
        assert_eq!(detect_language(Some("lib.rs")), Some("rust".into()));
        assert_eq!(detect_language(Some("app.tsx")), Some("tsx".into()));
        assert_eq!(detect_language(Some("data.csv")), Some("csv".into()));
        assert_eq!(detect_language(Some("unknown.xyz")), None);
        assert_eq!(detect_language(None), None);
    }

    #[test]
    fn test_hash_consistency() {
        let hash1 = hex::encode(Sha256::digest(b"hello world"));
        let hash2 = hex::encode(Sha256::digest(b"hello world"));
        assert_eq!(hash1, hash2);
    }
}
