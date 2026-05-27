//! End-to-end tests for `migrate_artifact_content` (§8.1).
//!
//! Seeds a KB with a legacy `:Artifact { content: "..." }` row, runs
//! the migration, and asserts the post-state: a matching
//! `:ArtifactContent` exists with the right SHA-256 hash, a
//! `HAS_CONTENT` edge connects the pair, and `a.content` is NULL.
//! Re-running the migration is a no-op.
//
use uniko_store::KnowledgeBase;
use uniko_store::blob_store::BlobStorage;
use uniko_store::config::UnikoConfig;

async fn seed_legacy_artifact(kb: &KnowledgeBase, body: &str) -> i64 {
    let session = kb.db().session();
    let tx = session.tx().await.unwrap();
    let r = tx
        .query_with(
            "CREATE (a:Artifact {artifact_id: $aid, kind: 'text', content: $body, \
             created_at: datetime()}) RETURN id(a) AS vid",
        )
        .param("aid", "art-legacy-1")
        .param("body", body)
        .fetch_all()
        .await
        .unwrap();
    tx.commit().await.unwrap();
    r.rows().first().unwrap().get("vid").unwrap()
}

async fn assert_post_migration(kb: &KnowledgeBase, vid: i64, body: &str) {
    let expected_hash = KnowledgeBase::sha256_hex(body.as_bytes());
    let session = kb.db().session();
    let r = session
        .query_with(
            "MATCH (a:Artifact)-[:HAS_CONTENT]->(c:ArtifactContent) \
             WHERE id(a) = $a RETURN c.content_id AS cid, c.size AS sz, a.content AS leg",
        )
        .param("a", vid)
        .fetch_all()
        .await
        .unwrap();
    let row = r
        .rows()
        .first()
        .expect("HAS_CONTENT row should exist post-migration");
    let cid: String = row.get("cid").unwrap();
    let sz: i64 = row.get("sz").unwrap();
    assert_eq!(
        cid, expected_hash,
        "ArtifactContent.content_id must match SHA-256"
    );
    assert_eq!(sz, body.len() as i64);

    // a.content stripped.
    match row.value("leg") {
        None | Some(uni_db::Value::Null) => {}
        other => panic!("expected a.content to be NULL post-migration, got {other:?}"),
    }
}

#[tokio::test]
async fn migration_lance_backend_basic_and_idempotent() {
    let kb = KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("KB");
    let body = "Phase 1 migration smoke body.";
    let vid = seed_legacy_artifact(&kb, body).await;

    let report = kb.migrate_artifact_content().await.expect("migrate");
    assert_eq!(report.scanned, 1);
    assert_eq!(report.migrated, 1);
    assert_eq!(report.already_migrated, 0);

    assert_post_migration(&kb, vid, body).await;

    // Idempotent re-run: scanned drops to 0 because a.content is NULL now.
    let report2 = kb.migrate_artifact_content().await.expect("migrate 2");
    assert_eq!(report2.scanned, 0, "second pass must scan zero rows");
    assert_eq!(report2.migrated, 0);

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn migration_fs_backend_writes_bytes_to_disk() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = UnikoConfig {
        blob_storage: BlobStorage::Fs {
            root: tmp.path().to_path_buf(),
        },
        ..Default::default()
    };
    let kb = KnowledgeBase::in_memory(cfg).await.expect("KB Fs");

    let body = "Hello from the Fs-backed migration test!";
    let vid = seed_legacy_artifact(&kb, body).await;
    let report = kb.migrate_artifact_content().await.expect("migrate");
    assert_eq!(report.migrated, 1);

    // For Fs backend, c.uri is populated and c.bytes is NULL.
    let session = kb.db().session();
    let r = session
        .query_with(
            "MATCH (a:Artifact)-[:HAS_CONTENT]->(c:ArtifactContent) \
             WHERE id(a) = $a RETURN c.uri AS uri",
        )
        .param("a", vid)
        .fetch_all()
        .await
        .unwrap();
    let uri: String = r.rows().first().unwrap().get("uri").unwrap();
    assert!(uri.starts_with("fs://"), "expected fs:// URI, got {uri}");

    // fetch_blob round-trips through the backend.
    let hash = KnowledgeBase::sha256_hex(body.as_bytes());
    let got = kb.fetch_blob(&hash).await.expect("fetch_blob");
    assert_eq!(got, body.as_bytes());

    kb.shutdown().await.unwrap();
}
