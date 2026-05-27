//! Dedicated integration tests for the `:ArtifactContent` label,
//! `HAS_CONTENT` edge, and `:KnowledgeBaseStats` singleton shipped in
//! Phase 1.
//!
//! These tests live separately from `schema_completeness.rs` because:
//! - they exercise property-level shape and singleton enforcement, not
//!   just label-count canaries; and
//! - the singleton-mismatch test needs an on-disk KB (the in-memory
//!   variant can't be reopened).
//
// Rust guideline compliant.

use uniko_store::KnowledgeBase;
use uniko_store::blob_store::BlobStorage;
use uniko_store::config::UnikoConfig;

#[tokio::test]
async fn artifact_content_label_has_expected_properties() {
    let kb = KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB");
    let info = kb
        .db()
        .get_label_info("ArtifactContent")
        .await
        .unwrap()
        .expect(":ArtifactContent must be registered");
    let names: Vec<&str> = info.properties.iter().map(|p| p.name.as_str()).collect();

    for required in [
        "content_id",
        "bytes",
        "uri",
        "mime",
        "size",
        "perceptual_hash",
        "audio_fingerprint",
        "created_at",
    ] {
        assert!(
            names.contains(&required),
            ":ArtifactContent missing required property {required}; have {names:?}"
        );
    }
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn has_content_edge_registered() {
    let kb = KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB");
    assert!(
        kb.db().edge_type_exists("HAS_CONTENT").await.unwrap(),
        "HAS_CONTENT edge type must be registered"
    );
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn knowledge_base_stats_singleton_exists_after_init() {
    let kb = KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB");
    let info = kb
        .db()
        .get_label_info("KnowledgeBaseStats")
        .await
        .unwrap()
        .expect(":KnowledgeBaseStats must be registered");
    let names: Vec<&str> = info.properties.iter().map(|p| p.name.as_str()).collect();
    for required in [
        "stats_id",
        "blob_storage",
        "modality_presence",
        "updated_at",
    ] {
        assert!(
            names.contains(&required),
            ":KnowledgeBaseStats missing {required}; have {names:?}"
        );
    }

    // `init_kb_stats` writes the singleton on first open. Reading the
    // modality_presence should succeed (default false) and reading the
    // singleton row count via Cypher should be exactly one.
    let presence = kb.read_modality_presence().await.unwrap();
    assert!(!presence.has_image_content);
    assert!(!presence.has_audio_content);

    let session = kb.db().session();
    let r = session
        .query("MATCH (s:KnowledgeBaseStats) RETURN count(s) AS n")
        .await
        .unwrap();
    let count: i64 = r.rows().first().unwrap().get("n").unwrap();
    assert_eq!(count, 1, "exactly one :KnowledgeBaseStats row expected");

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn reopen_with_mismatched_blob_storage_is_hard_error() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().to_path_buf();

    // First open: default (Lance) backend, write singleton.
    {
        let kb = KnowledgeBase::open(&path, UnikoConfig::default())
            .await
            .expect("first open Lance");
        kb.shutdown().await.unwrap();
    }

    // Reopen with Fs backend: must error from init_kb_stats.
    let blob_root = tmp.path().join("blobs");
    let mut cfg = UnikoConfig::default();
    cfg.blob_storage = BlobStorage::Fs {
        root: blob_root.clone(),
    };
    let err = KnowledgeBase::open(&path, cfg).await;
    assert!(
        err.is_err(),
        "reopening with a different blob_storage kind must fail"
    );
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("blob_storage mismatch"),
        "expected mismatch error, got: {msg}"
    );
}
