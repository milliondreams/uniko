//! One-shot, idempotent schema migrations for the multimodal store
//! work landed in Phase 1.
//!
//! All migrations are host-side (Rust) rather than pure Cypher so they
//! can branch on [`BlobStorage`](crate::blob_store::BlobStorage). Each
//! migration is safe to re-run — MERGE-on-`content_id` is naturally
//! idempotent and backend PUTs are idempotent on the hash key (CAS
//! dedup).
//
// Rust guideline compliant.

use std::collections::HashMap;

use chrono::Utc;
use uni_db::Value;

use crate::error::{Result, UnikoError};
use crate::storage::KnowledgeBase;
use crate::storage::blob::MergeContent;
use crate::types::{NodeId, datetime_value};

/// Summary of a migration pass.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MigrationReport {
    /// Rows scanned by the migration.
    pub scanned: usize,
    /// Rows migrated (newly written content / edge).
    pub migrated: usize,
    /// Rows skipped because they were already migrated.
    pub already_migrated: usize,
}

impl KnowledgeBase {
    /// Migrate `Artifact.content` → `:ArtifactContent` per §8.1.
    ///
    /// Reads every `:Artifact` row with `content IS NOT NULL`, writes
    /// the bytes to the configured blob backend, MERGEs the
    /// `:ArtifactContent` node, MERGEs a `HAS_CONTENT` edge, and
    /// removes `a.content`. Idempotent — re-running on an already-
    /// migrated KB returns `MigrationReport { migrated: 0, ... }`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database / backend failure.
    pub async fn migrate_artifact_content(&self) -> Result<MigrationReport> {
        let session = self.db.session();
        let result = session
            .query_with(
                "MATCH (a:Artifact) WHERE a.content IS NOT NULL \
                 RETURN id(a) AS vid, a.content AS content, a.hash AS hash, \
                        a.kind AS kind",
            )
            .fetch_all()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;

        let mut report = MigrationReport::default();
        for row in result.rows() {
            report.scanned += 1;

            let vid: NodeId = row
                .get("vid")
                .map_err(|e| UnikoError::Storage(e.to_string()))?;
            let content = match row.value("content") {
                Some(Value::String(s)) => s.clone(),
                _ => continue, // already-null content from a partial migration
            };
            let cached_hash = match row.value("hash") {
                Some(Value::String(s)) => Some(s.clone()),
                _ => None,
            };
            let kind = match row.value("kind") {
                Some(Value::String(s)) => s.clone(),
                _ => "text".into(),
            };

            let bytes = content.as_bytes();
            let hash = cached_hash.unwrap_or_else(|| Self::sha256_hex(bytes));

            // PUT bytes via the configured backend (no-op return for
            // Lance; precedes the MERGE for Fs/S3).
            let put = self.put_blob(&hash, bytes).await?;
            let content_nid = self
                .merge_artifact_content(MergeContent {
                    content_id: hash.clone(),
                    bytes: put.bytes_inline,
                    uri: put.uri,
                    mime: best_mime_guess(&kind),
                    size: bytes.len() as i64,
                    perceptual_hash: None,
                    audio_fingerprint: None,
                })
                .await?;

            // MERGE the HAS_CONTENT edge so re-runs don't create
            // duplicates. uni-db requires both endpoints addressed by
            // id() in a single MERGE pattern. We perform it manually:
            // OPTIONAL MATCH for existing edge, CREATE if absent.
            let tx = session
                .tx()
                .await
                .map_err(|e| UnikoError::Storage(e.to_string()))?;
            let existing_edge = tx
                .query_with(
                    "MATCH (a:Artifact)-[r:HAS_CONTENT]->(c:ArtifactContent) \
                     WHERE id(a) = $a AND id(c) = $c RETURN id(r) AS eid LIMIT 1",
                )
                .param("a", vid)
                .param("c", content_nid)
                .fetch_all()
                .await
                .map_err(|e| UnikoError::Storage(e.to_string()))?;

            if existing_edge.rows().is_empty() {
                tx.query_with(
                    "MATCH (a:Artifact), (c:ArtifactContent) \
                     WHERE id(a) = $a AND id(c) = $c \
                     CREATE (a)-[:HAS_CONTENT {role: 'primary'}]->(c)",
                )
                .param("a", vid)
                .param("c", content_nid)
                .fetch_all()
                .await
                .map_err(|e| UnikoError::Storage(e.to_string()))?;
            }

            // Strip the legacy content field.
            tx.query_with(
                "MATCH (a:Artifact) WHERE id(a) = $a \
                 SET a.content = NULL, a.hash = $hash, a.updated_at = $now",
            )
            .param("a", vid)
            .param("hash", Value::String(hash))
            .param("now", datetime_value(Utc::now()))
            .fetch_all()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;

            tx.commit()
                .await
                .map_err(|e| UnikoError::Storage(e.to_string()))?;

            if existing_edge.rows().is_empty() {
                report.migrated += 1;
            } else {
                report.already_migrated += 1;
            }
        }

        Ok(report)
    }

    /// Mean-pool a single Artifact's child Chunk embeddings into
    /// `Artifact.text_embedding`. Skips Chunks whose `embedding` is
    /// NULL. No-op when the artifact has zero pool-able chunks.
    ///
    /// Returns `true` when a pooled vector was written, `false` if
    /// nothing was poolable (no chunks, or all embeddings NULL).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn mean_pool_artifact_text_embedding(&self, artifact_nid: NodeId) -> Result<bool> {
        let session = self.db.session();
        let result = session
            .query_with(
                "MATCH (a:Artifact)-[:HAS_CHUNK]->(c:Chunk) \
                 WHERE id(a) = $a AND c.embedding IS NOT NULL \
                 RETURN c.embedding AS emb",
            )
            .param("a", artifact_nid)
            .fetch_all()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;

        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for row in result.rows() {
            if let Some(Value::Vector(v)) = row.value("emb") {
                vectors.push(v.clone());
            }
        }
        let Some(pooled) = mean_pool(&vectors) else {
            return Ok(false);
        };

        let mut props = HashMap::new();
        props.insert("text_embedding".into(), Value::Vector(pooled));
        self.update_node(artifact_nid, &props).await?;
        Ok(true)
    }

    /// Backfill `Artifact.text_embedding` for all Artifacts that have
    /// chunks but no pooled vector yet. Idempotent — skips rows where
    /// `text_embedding IS NOT NULL`.
    ///
    /// Returns the count of artifacts that received a pooled vector.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn backfill_artifact_text_embedding(&self) -> Result<usize> {
        let session = self.db.session();
        let result = session
            .query_with(
                "MATCH (a:Artifact) \
                 WHERE a.text_embedding IS NULL AND (a)-[:HAS_CHUNK]->(:Chunk) \
                 RETURN id(a) AS vid",
            )
            .fetch_all()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;

        let mut count = 0;
        for row in result.rows() {
            let vid: NodeId = row
                .get("vid")
                .map_err(|e| UnikoError::Storage(e.to_string()))?;
            if self.mean_pool_artifact_text_embedding(vid).await? {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Backfill `Chunk.modality = 'text'` for rows where it is NULL.
    /// Per §8.2. Idempotent.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn backfill_chunk_modality_text(&self) -> Result<usize> {
        let session = self.db.session();
        let tx = session
            .tx()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        let result = tx
            .query_with(
                "MATCH (c:Chunk) WHERE c.modality IS NULL \
                 SET c.modality = 'text' RETURN count(c) AS updated",
            )
            .fetch_all()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        let updated: i64 = result
            .rows()
            .first()
            .and_then(|r| r.get("updated").ok())
            .unwrap_or(0);
        tx.commit()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        Ok(updated as usize)
    }
}

/// Average a slice of equal-length f32 vectors. Returns `None` for
/// empty input or when widths disagree (an internal-invariant failure
/// we surface rather than silently mis-pool).
fn mean_pool(vectors: &[Vec<f32>]) -> Option<Vec<f32>> {
    let first = vectors.first()?;
    let dim = first.len();
    if dim == 0 || vectors.iter().any(|v| v.len() != dim) {
        return None;
    }
    let n = vectors.len() as f32;
    let mut acc = vec![0.0f32; dim];
    for v in vectors {
        for (a, x) in acc.iter_mut().zip(v.iter()) {
            *a += *x;
        }
    }
    for a in &mut acc {
        *a /= n;
    }
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_pool_basic() {
        let vecs = vec![vec![1.0_f32, 2.0, 3.0], vec![3.0, 2.0, 1.0]];
        let p = mean_pool(&vecs).expect("pool");
        assert_eq!(p, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_mean_pool_empty_input() {
        assert!(mean_pool(&[] as &[Vec<f32>]).is_none());
    }

    #[test]
    fn test_mean_pool_dim_mismatch() {
        let vecs = vec![vec![1.0_f32, 2.0], vec![1.0, 2.0, 3.0]];
        assert!(mean_pool(&vecs).is_none());
    }
}

/// Coarse MIME guess from the legacy `Artifact.kind` value. Text-only
/// path was the only producer of `Artifact.content`, so the guess is
/// always a `text/*` variant.
fn best_mime_guess(kind: &str) -> String {
    match kind {
        "html" => "text/html".into(),
        "markdown" | "md" => "text/markdown".into(),
        "code" => "text/plain".into(),
        "csv" => "text/csv".into(),
        "json" => "application/json".into(),
        _ => "text/plain".into(),
    }
}
