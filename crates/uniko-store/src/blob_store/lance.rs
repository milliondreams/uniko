//! Lance-inline blob backend.
//!
//! [`LanceBlobStore::put`] returns the bytes unchanged so the caller
//! threads them into the same Cypher transaction that creates the
//! `:ArtifactContent` row. There is no separate backend write — the
//! Lance column store holds the bytes directly.
//
// Rust guideline compliant.

use async_trait::async_trait;

use super::{BlobStore, PutOutcome};
use crate::error::{Result, UnikoError};

/// Lance-inline blob store. Bytes flow through `PutOutcome::bytes_inline`.
#[derive(Debug, Default, Clone, Copy)]
pub struct LanceBlobStore;

impl LanceBlobStore {
    /// Construct a fresh Lance backend handle. Zero state.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl BlobStore for LanceBlobStore {
    async fn put(&self, _content_id: &str, bytes: &[u8]) -> Result<PutOutcome> {
        Ok(PutOutcome {
            bytes_inline: Some(bytes.to_vec()),
            uri: None,
        })
    }

    async fn get(&self, _content_id: &str, _uri: Option<&str>) -> Result<Vec<u8>> {
        // Lance bytes live in the graph row, not in the backend.
        // Callers fetch via a Cypher query on `:ArtifactContent.bytes`.
        Err(UnikoError::Storage(
            "LanceBlobStore::get is not callable; fetch bytes from \
             :ArtifactContent.bytes via Cypher instead"
                .into(),
        ))
    }

    async fn exists(&self, _content_id: &str, _uri: Option<&str>) -> Result<bool> {
        // Existence is determined by the graph MERGE, not the backend.
        Ok(true)
    }

    async fn delete(&self, _content_id: &str, _uri: Option<&str>) -> Result<()> {
        // No backend state to delete; row-level cleanup happens via
        // DETACH DELETE on the `:ArtifactContent` node.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_put_returns_bytes_inline() {
        let store = LanceBlobStore::new();
        let out = store.put("dead", b"hello").await.unwrap();
        assert_eq!(out.bytes_inline.as_deref(), Some(&b"hello"[..]));
        assert!(out.uri.is_none());
    }

    #[tokio::test]
    async fn test_get_errors() {
        let store = LanceBlobStore::new();
        assert!(store.get("dead", None).await.is_err());
    }
}
