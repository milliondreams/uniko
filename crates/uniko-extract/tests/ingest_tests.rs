//! Integration tests for the P1 ingest pipeline.

use std::collections::HashMap;

use chrono::Utc;

use uniko_pipes::types::{IngestArtifact, IngestMessage};
use uniko_store::config::UnikoConfig;
use uniko_store::storage::KnowledgeBase;
use uniko_store::storage::edges::Direction;

async fn test_kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

fn test_message(id: &str, content: &str, session: &str, sender: &str) -> IngestMessage {
    IngestMessage {
        message_id: id.to_string(),
        content: content.to_string(),
        content_type: "text".to_string(),
        sender_id: sender.to_string(),
        session_id: session.to_string(),
        addressed_to: None,
        timestamp: Utc::now(),
        metadata: HashMap::new(),
    }
}

fn test_artifact(id: &str, content: &str, kind: &str, path: Option<&str>) -> IngestArtifact {
    IngestArtifact {
        artifact_id: id.to_string(),
        content: content.to_string(),
        kind: kind.to_string(),
        path: path.map(String::from),
        metadata: HashMap::new(),
    }
}

// ── Message ingest ──────────────────────────────────────────────────

#[tokio::test]
async fn test_ingest_message_creates_node_and_edges() {
    let kb = test_kb().await;

    let msg = test_message("m-1", "Hello world", "s-1", "p-1");
    let result = uniko_extract::ingest::message::ingest_message(
        &kb,
        &msg,
        &mut uniko_extract::ingest::context::SessionContext::new(msg.session_id.clone(), 0),
    )
    .await
    .unwrap();

    // Message node should exist.
    let (label, props) = kb
        .get_node(result.message_node_id)
        .await
        .unwrap()
        .expect("message node must exist");
    assert_eq!(label, "Message");
    assert_eq!(
        props.get("content").and_then(|v| v.as_str()),
        Some("Hello world")
    );

    // SENT_BY edge should exist.
    let sent_by = kb
        .get_edges(result.message_node_id, "SENT_BY", Direction::Outgoing)
        .await
        .unwrap();
    assert_eq!(sent_by.len(), 1, "should have one SENT_BY edge");

    // IN_SESSION edge should exist.
    let in_session = kb
        .get_edges(result.message_node_id, "IN_SESSION", Direction::Outgoing)
        .await
        .unwrap();
    assert_eq!(in_session.len(), 1, "should have one IN_SESSION edge");

    // Session node should have been created.
    let session = kb
        .get_node_by_ext_id("Session", "session_id", "s-1")
        .await
        .unwrap();
    assert!(session.is_some(), "session should have been auto-created");
}

#[tokio::test]
async fn test_ingest_message_idempotent() {
    let kb = test_kb().await;

    let msg = test_message("m-idem", "Same message", "s-1", "p-1");
    let r1 = uniko_extract::ingest::message::ingest_message(
        &kb,
        &msg,
        &mut uniko_extract::ingest::context::SessionContext::new(msg.session_id.clone(), 0),
    )
    .await
    .unwrap();
    let r2 = uniko_extract::ingest::message::ingest_message(
        &kb,
        &msg,
        &mut uniko_extract::ingest::context::SessionContext::new(msg.session_id.clone(), 0),
    )
    .await
    .unwrap();

    assert_eq!(r1.message_node_id, r2.message_node_id);
}

#[tokio::test]
async fn test_ingest_message_next_chain() {
    let kb = test_kb().await;

    // Ingest 5 messages with increasing timestamps via SessionContext.
    let mut node_ids = Vec::new();
    let mut session_ctx = uniko_extract::ingest::context::SessionContext::new("s-chain".into(), 0);
    for i in 0..5 {
        let mut msg = test_message(
            &format!("m-chain-{i}"),
            &format!("msg {i}"),
            "s-chain",
            "p-1",
        );
        msg.timestamp = Utc::now() + chrono::Duration::milliseconds(i * 100);
        let result = uniko_extract::ingest::message::ingest_message(&kb, &msg, &mut session_ctx)
            .await
            .unwrap();
        node_ids.push(result.message_node_id);
    }

    // Each message (except the first) should have a NEXT edge from the previous.
    for i in 1..5 {
        let edges = kb
            .get_edges(node_ids[i - 1], "NEXT", Direction::Outgoing)
            .await
            .unwrap();
        assert!(
            !edges.is_empty(),
            "message {i} should have a NEXT edge from message {}",
            i - 1
        );
        assert_eq!(edges[0].to, node_ids[i]);
    }
}

#[tokio::test]
async fn test_ingest_message_long_content_chunked() {
    let kb = test_kb().await;

    // Create content exceeding the 1024-token threshold.
    let long_content = "This is a test sentence for chunking purposes. ".repeat(200);
    let msg = test_message("m-long", &long_content, "s-long", "p-1");
    let result = uniko_extract::ingest::message::ingest_message(
        &kb,
        &msg,
        &mut uniko_extract::ingest::context::SessionContext::new(msg.session_id.clone(), 0),
    )
    .await
    .unwrap();

    assert!(
        !result.chunk_node_ids.is_empty(),
        "long message should produce chunks"
    );

    // Verify HAS_CHUNK edges exist.
    let edges = kb
        .get_edges(result.message_node_id, "HAS_CHUNK", Direction::Outgoing)
        .await
        .unwrap();
    assert_eq!(
        edges.len(),
        result.chunk_node_ids.len(),
        "one HAS_CHUNK edge per chunk"
    );
}

#[tokio::test]
async fn test_ingest_message_short_no_chunks() {
    let kb = test_kb().await;

    let msg = test_message("m-short", "Short", "s-short", "p-1");
    let result = uniko_extract::ingest::message::ingest_message(
        &kb,
        &msg,
        &mut uniko_extract::ingest::context::SessionContext::new(msg.session_id.clone(), 0),
    )
    .await
    .unwrap();

    assert!(
        result.chunk_node_ids.is_empty(),
        "short message should not be chunked"
    );
}

// ── Artifact ingest ─────────────────────────────────────────────────

#[tokio::test]
async fn test_ingest_artifact_creates_node_and_chunks() {
    let kb = test_kb().await;

    let content = "fn main() {\n    println!(\"hello\");\n}\n\nfn helper() {\n    // ...\n}";
    let art = test_artifact("art-1", content, "file", Some("main.rs"));
    let result = uniko_extract::ingest::artifact::ingest_artifact(&kb, &art)
        .await
        .unwrap();

    assert!(!result.was_deduplicated);

    // Artifact node should exist with hash.
    let (label, props) = kb
        .get_node(result.artifact_node_id)
        .await
        .unwrap()
        .expect("artifact node must exist");
    assert_eq!(label, "Artifact");
    assert!(props.contains_key("hash"));

    // Should have at least one chunk.
    assert!(
        !result.chunk_node_ids.is_empty(),
        "artifact should produce chunks"
    );
}

#[tokio::test]
async fn test_ingest_artifact_dedup_by_hash() {
    let kb = test_kb().await;

    let content = "deduplicated content here";
    let art1 = test_artifact("art-dup-1", content, "file", None);
    let r1 = uniko_extract::ingest::artifact::ingest_artifact(&kb, &art1)
        .await
        .unwrap();
    assert!(!r1.was_deduplicated);

    let art2 = test_artifact("art-dup-2", content, "file", None);
    let r2 = uniko_extract::ingest::artifact::ingest_artifact(&kb, &art2)
        .await
        .unwrap();
    assert!(r2.was_deduplicated);
    assert_eq!(r1.artifact_node_id, r2.artifact_node_id);
}

#[tokio::test]
async fn test_ingest_artifact_deterministic_chunk_ids() {
    let kb = test_kb().await;

    let content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
    let art = test_artifact("art-det", content, "document", None);
    let result = uniko_extract::ingest::artifact::ingest_artifact(&kb, &art)
        .await
        .unwrap();

    for (i, &chunk_nid) in result.chunk_node_ids.iter().enumerate() {
        let (_, props) = kb
            .get_node(chunk_nid)
            .await
            .unwrap()
            .expect("chunk must exist");
        let cid = props
            .get("chunk_id")
            .and_then(|v| v.as_str())
            .expect("chunk_id must be set");
        let expected = format!("art-det:{i}");
        assert_eq!(cid, expected);
    }
}
