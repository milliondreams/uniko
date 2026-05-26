//! S3-compatible blob backend (AWS S3, R2, GCS-via-S3, MinIO, in-memory for tests).
//!
//! Wraps `object_store::ObjectStore` so the same code services real S3
//! in production and `object_store::memory::InMemory` under test — no
//! MinIO container needed in CI.
//!
//! Key layout mirrors [`super::fs::FsBlobStore`]: `<prefix>/<aa>/<bb>/<content_id>`,
//! so an operator can copy a `Fs` tree to S3 without re-keying.
//
// Rust guideline compliant.

use std::sync::Arc;

use async_trait::async_trait;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjPath;
use object_store::{ObjectStore, PutPayload};

use crate::blob_store::{BlobStore, PutOutcome, fs_relative_path};
use crate::error::{Result, UnikoError};

/// S3-compatible content-addressed blob backend.
///
/// One backend instance is shared across all writes for a KB. The
/// inner [`ObjectStore`] is reference-counted so cheap to clone.
#[derive(Clone)]
pub struct S3BlobStore {
    inner: Arc<dyn ObjectStore>,
    bucket: String,
    prefix: String,
}

impl std::fmt::Debug for S3BlobStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3BlobStore")
            .field("bucket", &self.bucket)
            .field("prefix", &self.prefix)
            .finish()
    }
}

impl S3BlobStore {
    /// Construct from the persisted `BlobStorage::S3` config.
    ///
    /// Uses [`AmazonS3Builder`] which honors AWS credential chain
    /// (env, profile, IMDS); pass `endpoint` to override for R2 / MinIO.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] if the builder rejects the
    /// combination (e.g., bucket name invalid).
    pub fn from_config(
        bucket: &str,
        prefix: Option<&str>,
        endpoint: Option<&str>,
        region: Option<&str>,
    ) -> Result<Self> {
        let mut b = AmazonS3Builder::from_env().with_bucket_name(bucket);
        if let Some(e) = endpoint {
            b = b.with_endpoint(e).with_allow_http(e.starts_with("http://"));
        }
        if let Some(r) = region {
            b = b.with_region(r);
        }
        let inner = b
            .build()
            .map_err(|e| UnikoError::Config(format!("AmazonS3Builder: {e}")))?;
        let inner: Arc<dyn ObjectStore> = Arc::new(inner);
        Ok(Self {
            inner,
            bucket: bucket.into(),
            prefix: prefix.unwrap_or("").trim_matches('/').into(),
        })
    }

    /// Construct around an arbitrary [`ObjectStore`] — used by tests
    /// to inject `object_store::memory::InMemory` without touching the
    /// network.
    #[must_use]
    pub fn new_for_test(inner: Arc<dyn ObjectStore>, bucket: String, prefix: String) -> Self {
        Self {
            inner,
            bucket,
            prefix: prefix.trim_matches('/').into(),
        }
    }

    /// Compose the object-store key for `content_id` using the same
    /// two-level fanout as [`FsBlobStore`].
    fn key(&self, content_id: &str) -> ObjPath {
        let rel = fs_relative_path(content_id);
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        let full = if self.prefix.is_empty() {
            rel_str.to_string()
        } else {
            format!("{}/{rel_str}", self.prefix)
        };
        ObjPath::from(full)
    }

    fn uri(&self, key: &ObjPath) -> String {
        format!("s3://{}/{}", self.bucket, key)
    }
}

#[async_trait]
impl BlobStore for S3BlobStore {
    async fn put(&self, content_id: &str, bytes: &[u8]) -> Result<PutOutcome> {
        let key = self.key(content_id);
        // Idempotency: skip upload if HEAD shows matching size.
        if let Ok(meta) = self.inner.head(&key).await
            && meta.size == bytes.len()
        {
            return Ok(PutOutcome {
                bytes_inline: None,
                uri: Some(self.uri(&key)),
            });
        }
        let payload = PutPayload::from(bytes.to_vec());
        self.inner
            .put(&key, payload)
            .await
            .map_err(|e| UnikoError::Storage(format!("S3 put {content_id}: {e}")))?;
        Ok(PutOutcome {
            bytes_inline: None,
            uri: Some(self.uri(&key)),
        })
    }

    async fn get(&self, content_id: &str, _uri: Option<&str>) -> Result<Vec<u8>> {
        let key = self.key(content_id);
        let got = self
            .inner
            .get(&key)
            .await
            .map_err(|e| UnikoError::Storage(format!("S3 get {content_id}: {e}")))?;
        let bytes = got
            .bytes()
            .await
            .map_err(|e| UnikoError::Storage(format!("S3 read body {content_id}: {e}")))?;
        Ok(bytes.to_vec())
    }

    async fn exists(&self, content_id: &str, _uri: Option<&str>) -> Result<bool> {
        let key = self.key(content_id);
        match self.inner.head(&key).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { .. }) => Ok(false),
            Err(e) => Err(UnikoError::Storage(format!("S3 head {content_id}: {e}"))),
        }
    }

    async fn delete(&self, content_id: &str, _uri: Option<&str>) -> Result<()> {
        let key = self.key(content_id);
        match self.inner.delete(&key).await {
            Ok(()) => Ok(()),
            Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(e) => Err(UnikoError::Storage(format!(
                "S3 delete {content_id}: {e}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use object_store::memory::InMemory;

    use super::*;

    fn store() -> S3BlobStore {
        let inner: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        S3BlobStore::new_for_test(inner, "test-bucket".into(), "kb/x".into())
    }

    #[tokio::test]
    async fn roundtrip() {
        let s = store();
        let cid = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
        let payload = b"hello, multimodal world";

        let out = s.put(cid, payload).await.unwrap();
        assert!(out.bytes_inline.is_none());
        assert_eq!(out.uri.as_deref(), Some("s3://test-bucket/kb/x/ab/cd/abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"));

        assert!(s.exists(cid, None).await.unwrap());
        let got = s.get(cid, None).await.unwrap();
        assert_eq!(got, payload);

        // Idempotent re-put.
        let out2 = s.put(cid, payload).await.unwrap();
        assert_eq!(out, out2);

        s.delete(cid, None).await.unwrap();
        assert!(!s.exists(cid, None).await.unwrap());
        // Idempotent delete.
        s.delete(cid, None).await.unwrap();
    }

    #[tokio::test]
    async fn empty_prefix_omits_leading_slash() {
        let inner: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let s = S3BlobStore::new_for_test(inner, "b".into(), "".into());
        let cid = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
        let out = s.put(cid, b"x").await.unwrap();
        assert_eq!(
            out.uri.as_deref(),
            Some("s3://b/00/11/00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff")
        );
    }
}
