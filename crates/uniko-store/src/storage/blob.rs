//! Host helpers on [`KnowledgeBase`] for blob content storage.
//!
//! Encapsulates the §5.1 ingest skeleton's step-4-and-5 surface:
//!
//! - [`KnowledgeBase::put_blob`] — dispatches to the configured
//!   [`BlobStore`](crate::blob_store::BlobStore). For Lance the bytes
//!   come back inline (`PutOutcome::bytes_inline`); for Fs / S3 the
//!   bytes have been persisted by the backend and a `uri` is returned.
//! - [`KnowledgeBase::merge_artifact_content`] — graph-side MERGE on
//!   `content_id`. Atomic dedup point: concurrent ingests of the same
//!   hash collapse to one `:ArtifactContent` row.
//! - [`KnowledgeBase::fetch_blob`] — reads bytes back. For Lance reads
//!   the `:ArtifactContent.bytes` column; for Fs / S3 dispatches to
//!   the backend.
//
use chrono::Utc;
use sha2::{Digest, Sha256};
use uni_db::Value;

use crate::blob_store::{BlobStore, PutOutcome, build_backend};
use crate::error::{Result, UnikoError};
use crate::storage::KnowledgeBase;
use crate::types::datetime_value;

/// Inputs to [`KnowledgeBase::merge_artifact_content`].
///
/// `bytes` and `uri` are mutually exclusive — exactly one is populated
/// depending on backend. The struct shape mirrors the §5.1 skeleton
/// `MergeContent` type so the caller threads a single value through.
#[derive(Debug, Clone)]
pub struct MergeContent {
    /// SHA-256 hex digest of the bytes.
    pub content_id: String,
    /// Inline bytes (Lance backend only).
    pub bytes: Option<Vec<u8>>,
    /// Backend-resolvable URI (Fs / S3 backends).
    pub uri: Option<String>,
    /// Canonical MIME type, e.g., `"text/plain"` / `"image/jpeg"`.
    pub mime: String,
    /// Byte length.
    pub size: i64,
    /// Optional pHash (image-only).
    pub perceptual_hash: Option<i64>,
    /// Optional chromaprint fingerprint (audio-only).
    pub audio_fingerprint: Option<Vec<u8>>,
}

impl KnowledgeBase {
    /// Compute the SHA-256 hex digest of `bytes`.
    ///
    /// Exposed as a host helper because every ingest path needs it and
    /// the digest must match what the backend uses as its key.
    #[must_use]
    pub fn sha256_hex(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let digest = hasher.finalize();
        hex::encode(digest)
    }

    /// Run the configured blob backend's `put` for `bytes` keyed by
    /// `content_id` (SHA-256 hex).
    ///
    /// For Lance the result's `bytes_inline` is `Some(_)`; the caller
    /// passes that to [`merge_artifact_content`](Self::merge_artifact_content)
    /// so the bytes land in `:ArtifactContent.bytes` in the same
    /// transaction. For Fs / S3 the backend has already persisted the
    /// bytes; the result's `uri` is populated and `bytes_inline` is
    /// `None`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on backend failure or
    /// [`UnikoError::Config`] when the configured backend is not yet
    /// wired (e.g., `BlobStorage::S3` until that lands).
    pub async fn put_blob(&self, content_id: &str, bytes: &[u8]) -> Result<PutOutcome> {
        let backend = self.blob_backend()?;
        backend.put(content_id, bytes).await
    }

    /// MERGE the `:ArtifactContent` row for `spec.content_id`. Returns
    /// the `NodeId`. Idempotent — re-running on the same hash is a no-op
    /// on the graph side.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn merge_artifact_content(&self, spec: MergeContent) -> Result<crate::types::NodeId> {
        // Host-side MATCH-or-CREATE instead of `MERGE (c:ArtifactContent
        // {content_id: $cid}) ON CREATE SET ...`: cheap because content_id
        // is Hash-indexed, and idempotent because the second leg short-
        // circuits on an existing row. (uni-db <= 2.4 also evaluated NOT
        // NULL on the MERGE create before ON CREATE SET ran, blowing up on
        // required columns like mime/created_at; 2.5.0 folds ON CREATE SET
        // into the seed props, so that is no longer why we avoid MERGE.)
        //
        // The real reason to keep the host-side split: the check-and-create
        // must be atomic per content_id. Without a
        // guard, two concurrent ingests of the same content both read
        // "absent" and both CREATE a duplicate row (this method is called
        // per-item under the spawn-per-message ingest worker). Hold the
        // striped lock across the existence re-read AND the create, like
        // `merge_node`.
        let _rmw_guard = self
            .rmw_locks
            .lock(&crate::locks::content_lock_key(&spec.content_id))
            .await;

        let session = self.db.session();
        let existing = session
            .query_with("MATCH (c:ArtifactContent {content_id: $cid}) RETURN id(c) AS vid LIMIT 1")
            .param("cid", Value::String(spec.content_id.clone()))
            .fetch_all()
            .await?;
        if let Some(row) = existing.rows().first() {
            let vid: i64 = row.get("vid")?;
            return Ok(vid);
        }

        let now = Utc::now();
        self.transact_with_retry(uni_db::RetryOptions::default(), move |tx| {
            let spec = spec.clone();
            async move {
                let r = async {
                    let result = tx
                        .query_with(
                            "CREATE (c:ArtifactContent {
                                content_id: $cid, bytes: $bytes, uri: $uri,
                                mime: $mime, size: $size, perceptual_hash: $phash,
                                audio_fingerprint: $afp, created_at: $created_at
                            }) RETURN id(c) AS vid",
                        )
                        .param("cid", Value::String(spec.content_id.clone()))
                        .param("bytes", spec.bytes.map(Value::Bytes).unwrap_or(Value::Null))
                        .param("uri", spec.uri.map(Value::String).unwrap_or(Value::Null))
                        .param("mime", Value::String(spec.mime.clone()))
                        .param("size", Value::Int(spec.size))
                        .param(
                            "phash",
                            spec.perceptual_hash.map(Value::Int).unwrap_or(Value::Null),
                        )
                        .param(
                            "afp",
                            spec.audio_fingerprint
                                .map(Value::Bytes)
                                .unwrap_or(Value::Null),
                        )
                        .param("created_at", datetime_value(now))
                        .fetch_all()
                        .await?;
                    let row = result
                        .rows()
                        .first()
                        .ok_or_else(|| UnikoError::Storage("CREATE returned no rows".into()))?;
                    let vid: i64 = row.get("vid")?;
                    Ok(vid)
                }
                .await;
                (tx, r)
            }
        })
        .await
    }

    /// Fetch the bytes for a content node, dispatching on backend.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] when the content node is missing
    /// or the backend read fails.
    pub async fn fetch_blob(&self, content_id: &str) -> Result<Vec<u8>> {
        // Read the row to learn (bytes_inline, uri).
        let session = self.db.session();
        let result = session
            .query_with(
                "MATCH (c:ArtifactContent {content_id: $cid}) \
                 RETURN c.bytes AS bytes, c.uri AS uri",
            )
            .param("cid", Value::String(content_id.to_string()))
            .fetch_all()
            .await?;
        let row = result.rows().first().ok_or_else(|| {
            UnikoError::Storage(format!("no :ArtifactContent for content_id={content_id}"))
        })?;
        let bytes_inline = match row.value("bytes") {
            Some(Value::Bytes(b)) => Some(b.clone()),
            _ => None,
        };
        let uri = match row.value("uri") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        };

        if let Some(b) = bytes_inline {
            return Ok(b);
        }
        // Bytes live in the backend.
        let backend = self.blob_backend()?;
        backend.get(content_id, uri.as_deref()).await
    }

    /// Build a fresh backend handle from `self.config.blob_storage`.
    ///
    /// Re-builds on every call rather than caching; the cost is low
    /// (one allocation for the trait object) and avoids stashing a
    /// `Box<dyn BlobStore>` inside `KnowledgeBase` (which would force
    /// changes through the `Clone` derive).
    fn blob_backend(&self) -> Result<Box<dyn BlobStore>> {
        build_backend(&self.config.blob_storage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_hex_known_vector() {
        let h = KnowledgeBase::sha256_hex(b"abc");
        assert_eq!(
            h,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
