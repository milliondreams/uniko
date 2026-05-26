//! Filesystem content-addressed blob backend.
//!
//! Bytes live at `<root>/<aa>/<bb>/<content_id>`. The first two pairs
//! of hex chars form the directory fanout so a million blobs leaves
//! ~256 entries per second-level dir.
//
// Rust guideline compliant.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use tokio::fs as tfs;
use tokio::io::AsyncReadExt;

use super::{fs_relative_path, BlobStore, PutOutcome};
use crate::error::{Result, UnikoError};

/// Local filesystem CAS blob store rooted at [`Self::root`].
#[derive(Debug, Clone)]
pub struct FsBlobStore {
    root: PathBuf,
}

impl FsBlobStore {
    /// Construct a new backend. Creates `root` if it does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] if `root` cannot be created.
    pub fn new(root: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&root).map_err(|e| {
            UnikoError::Storage(format!("failed to create FsBlobStore root {root:?}: {e}"))
        })?;
        Ok(Self { root })
    }

    /// Resolve the absolute path of a content blob.
    fn path_for(&self, content_id: &str) -> PathBuf {
        self.root.join(fs_relative_path(content_id))
    }

    /// Format a `fs://` URI for the resolved absolute path.
    fn uri_for(&self, content_id: &str) -> String {
        format!("fs://{}", self.path_for(content_id).display())
    }
}

#[async_trait]
impl BlobStore for FsBlobStore {
    async fn put(&self, content_id: &str, bytes: &[u8]) -> Result<PutOutcome> {
        let path = self.path_for(content_id);

        // Idempotent: if the file already exists with the same length,
        // skip the write. We trust the hash — equal hash + equal length
        // is sufficient under SHA-256, no need to re-read and compare.
        if let Ok(meta) = tfs::metadata(&path).await
            && meta.len() as usize == bytes.len()
        {
            return Ok(PutOutcome {
                bytes_inline: None,
                uri: Some(self.uri_for(content_id)),
            });
        }

        if let Some(parent) = path.parent() {
            tfs::create_dir_all(parent).await.map_err(|e| {
                UnikoError::Storage(format!("failed to create dir {parent:?}: {e}"))
            })?;
        }

        // Write to a temp file in the same directory, then rename for
        // atomicity. Avoids torn writes if the process dies mid-PUT.
        let tmp = path.with_extension("partial");
        tfs::write(&tmp, bytes)
            .await
            .map_err(|e| UnikoError::Storage(format!("failed to write {tmp:?}: {e}")))?;
        tfs::rename(&tmp, &path).await.map_err(|e| {
            UnikoError::Storage(format!("failed to rename {tmp:?} -> {path:?}: {e}"))
        })?;

        Ok(PutOutcome {
            bytes_inline: None,
            uri: Some(self.uri_for(content_id)),
        })
    }

    async fn get(&self, content_id: &str, uri: Option<&str>) -> Result<Vec<u8>> {
        let path = resolve_path(self.path_for(content_id), uri);
        let mut f = tfs::File::open(&path)
            .await
            .map_err(|e| UnikoError::Storage(format!("failed to open {path:?}: {e}")))?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)
            .await
            .map_err(|e| UnikoError::Storage(format!("failed to read {path:?}: {e}")))?;
        Ok(buf)
    }

    async fn exists(&self, content_id: &str, uri: Option<&str>) -> Result<bool> {
        let path = resolve_path(self.path_for(content_id), uri);
        Ok(tfs::metadata(&path).await.is_ok())
    }

    async fn delete(&self, content_id: &str, uri: Option<&str>) -> Result<()> {
        let path = resolve_path(self.path_for(content_id), uri);
        match tfs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(UnikoError::Storage(format!(
                "failed to delete {path:?}: {e}"
            ))),
        }
    }
}

/// Prefer the persisted URI when present; fall back to the
/// content-id-derived path. Strips a `fs://` scheme if present.
fn resolve_path(default_path: PathBuf, uri: Option<&str>) -> PathBuf {
    match uri {
        Some(u) => Path::new(u.strip_prefix("fs://").unwrap_or(u)).to_path_buf(),
        None => default_path,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_roundtrip_and_dedup() {
        let tmp = tempdir_unique("uniko-fs-bs");
        let store = FsBlobStore::new(tmp.clone()).unwrap();

        let hash = "ab".repeat(32); // 64-char hex, SHA-256-shaped
        let out1 = store.put(&hash, b"hello world").await.unwrap();
        let uri1 = out1.uri.clone().unwrap();
        assert!(out1.bytes_inline.is_none());

        // Second put with the same bytes is a no-op (same uri returned)
        let out2 = store.put(&hash, b"hello world").await.unwrap();
        assert_eq!(out2.uri, Some(uri1.clone()));

        assert!(store.exists(&hash, Some(&uri1)).await.unwrap());
        let bytes = store.get(&hash, Some(&uri1)).await.unwrap();
        assert_eq!(bytes, b"hello world");

        store.delete(&hash, Some(&uri1)).await.unwrap();
        assert!(!store.exists(&hash, Some(&uri1)).await.unwrap());

        // Deleting a missing blob is Ok.
        store.delete(&hash, Some(&uri1)).await.unwrap();

        std::fs::remove_dir_all(&tmp).ok();
    }

    fn tempdir_unique(prefix: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("{prefix}-{nonce}"));
        p
    }
}
