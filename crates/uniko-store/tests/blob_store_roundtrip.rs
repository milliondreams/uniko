//! End-to-end round-trip tests for every [`BlobStorage`] backend.
//!
//! Each backend goes through identical assertions:
//! 1. `put` random bytes → `exists` true → `get` returns identical bytes.
//! 2. Re-`put` of the same bytes is idempotent (same outcome shape, no
//!    error).
//! 3. `delete` → `exists` false → re-`delete` is idempotent.
//
// Rust guideline compliant.

use std::sync::Arc;

use object_store::ObjectStore;
use object_store::memory::InMemory;
use uniko_store::blob_store::s3::S3BlobStore;
use uniko_store::blob_store::{BlobStorage, build_backend};

const HEX: &str = "a3f2c1d4e5b67890a3f2c1d4e5b67890a3f2c1d4e5b67890a3f2c1d4e5b67890";

async fn roundtrip_backend(backend: Box<dyn uniko_store::blob_store::BlobStore>) {
    let payload = b"hello, multimodal world!";

    let outcome = backend.put(HEX, payload).await.expect("put");
    // At least one of bytes_inline / uri is populated.
    assert!(outcome.bytes_inline.is_some() || outcome.uri.is_some());

    assert!(backend.exists(HEX, outcome.uri.as_deref()).await.unwrap());

    // Persistent backends (Fs, S3) round-trip bytes through the backend.
    // Lance is a pass-through — bytes live in `:ArtifactContent.bytes`,
    // so `get` is intentionally an error there.
    let backend_persists = outcome.bytes_inline.is_none();
    if backend_persists {
        let got = backend.get(HEX, outcome.uri.as_deref()).await.expect("get");
        assert_eq!(got, payload);
    }

    // Idempotent re-put.
    let outcome2 = backend.put(HEX, payload).await.expect("put 2");
    assert_eq!(outcome.uri, outcome2.uri);

    backend
        .delete(HEX, outcome.uri.as_deref())
        .await
        .expect("delete");
    if backend_persists {
        // Backend now reports gone.
        assert!(!backend.exists(HEX, outcome.uri.as_deref()).await.unwrap());
    }

    // Idempotent re-delete.
    backend
        .delete(HEX, outcome.uri.as_deref())
        .await
        .expect("delete 2");
}

#[tokio::test]
async fn lance_backend_roundtrip() {
    let backend = build_backend(&BlobStorage::Lance).expect("lance");
    roundtrip_backend(backend).await;
}

#[tokio::test]
async fn fs_backend_roundtrip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let backend = build_backend(&BlobStorage::Fs {
        root: tmp.path().to_path_buf(),
    })
    .expect("fs");
    roundtrip_backend(backend).await;
}

#[tokio::test]
async fn s3_backend_roundtrip_in_memory() {
    // Inject an in-process ObjectStore so no MinIO container is needed
    // in CI. The S3BlobStore wraps it identically to a real AmazonS3.
    let inner: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let s3 = S3BlobStore::new_for_test(inner, "uniko-test".into(), "kb-1".into());
    let backend: Box<dyn uniko_store::blob_store::BlobStore> = Box::new(s3);
    roundtrip_backend(backend).await;
}
