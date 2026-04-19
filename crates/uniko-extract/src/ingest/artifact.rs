//! Artifact ingest: hash, dedup, create Artifact node, chunk, and link.

// Rust guideline compliant

use std::collections::HashMap;

use sha2::{Digest, Sha256};
use uni_db::Value;

use uniko_pipes::types::IngestArtifact;
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

    // 4. Create Artifact node.
    let mut props = HashMap::new();
    props.insert(
        "artifact_id".into(),
        Value::String(artifact.artifact_id.clone()),
    );
    props.insert("kind".into(), Value::String(artifact.kind.clone()));
    if let Some(ref path) = artifact.path {
        props.insert("path".into(), Value::String(path.clone()));
    }
    props.insert("content".into(), Value::String(artifact.content.clone()));
    props.insert("hash".into(), Value::String(hash));
    props.insert("size".into(), Value::Int(artifact.content.len() as i64));
    if let Some(ref lang) = language {
        props.insert("language".into(), Value::String(lang.clone()));
    }
    let artifact_nid = kb.create_node("Artifact", &props).await?;

    // 5. Select chunker and create chunks.
    let content_type = if language.is_some() { "code" } else { "text" };
    let chunker = select_chunker(content_type, language.as_deref());
    let chunk_cfg = ChunkConfig::from_uniko_config(kb.config());
    let chunks = chunker.chunk(&artifact.content, &chunk_cfg);

    let chunk_nids =
        create_chunks(kb, &artifact.artifact_id, artifact_nid, &chunks, "Artifact").await?;

    Ok(ArtifactIngestResult {
        artifact_node_id: artifact_nid,
        chunk_node_ids: chunk_nids,
        was_deduplicated: false,
    })
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
