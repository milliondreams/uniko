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
        ..Default::default()
    }
}

// ── Message ingest ──────────────────────────────────────────────────

#[tokio::test]
async fn test_ingest_message_creates_node_and_edges() {
    let kb = test_kb().await;

    let msg = test_message("m-1", "Hello world", "s-1", "p-1");
    let result = uniko_extract::ingest::atomic::ingest_message_atomic(
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
    let r1 = uniko_extract::ingest::atomic::ingest_message_atomic(
        &kb,
        &msg,
        &mut uniko_extract::ingest::context::SessionContext::new(msg.session_id.clone(), 0),
    )
    .await
    .unwrap();
    let r2 = uniko_extract::ingest::atomic::ingest_message_atomic(
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
        let result =
            uniko_extract::ingest::atomic::ingest_message_atomic(&kb, &msg, &mut session_ctx)
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
    let result = uniko_extract::ingest::atomic::ingest_message_atomic(
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
    let result = uniko_extract::ingest::atomic::ingest_message_atomic(
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

#[tokio::test]
async fn test_ingest_artifact_links_conversational_context() {
    use uni_db::Value;
    use uniko_store::types::datetime_value;

    let kb = test_kb().await;
    let now = Utc::now();

    // Seed the context an agent would reference when sharing a file
    // mid-conversation: a Session, the Message that introduced it, and
    // the Action that produced it.
    let mut sprops: HashMap<String, Value> = HashMap::new();
    sprops.insert("session_id".into(), Value::String("sess-ctx".into()));
    sprops.insert("started_at".into(), datetime_value(now));
    kb.merge_node("Session", "session_id", "sess-ctx", &sprops)
        .await
        .unwrap();

    let mut mprops: HashMap<String, Value> = HashMap::new();
    mprops.insert("message_id".into(), Value::String("msg-ctx".into()));
    mprops.insert("content".into(), Value::String("here is the file".into()));
    mprops.insert("timestamp".into(), datetime_value(now));
    kb.merge_node("Message", "message_id", "msg-ctx", &mprops)
        .await
        .unwrap();

    let mut aprops: HashMap<String, Value> = HashMap::new();
    aprops.insert("action_id".into(), Value::String("act-ctx".into()));
    aprops.insert("action_type".into(), Value::String("file_write".into()));
    kb.merge_node("Action", "action_id", "act-ctx", &aprops)
        .await
        .unwrap();

    let art = IngestArtifact {
        artifact_id: "art-ctx".into(),
        content: "contextual file body".into(),
        kind: "file".into(),
        path: Some("notes.txt".into()),
        metadata: HashMap::new(),
        session_id: Some("sess-ctx".into()),
        triggered_by_message_id: Some("msg-ctx".into()),
        produced_by_action_id: Some("act-ctx".into()),
    };
    let result = uniko_extract::ingest::artifact::ingest_artifact(&kb, &art)
        .await
        .unwrap();
    assert!(!result.was_deduplicated);

    // Assert each contextual edge by matching specific endpoint node
    // ids. We deliberately avoid relying on a target-label predicate in
    // the pattern (e.g. `(a)-[:ATTACHED_TO]->(m:Message)`): the Artifact
    // has two ATTACHED_TO edges to different labels (Session + Message),
    // and uni-db does not reliably prune the other-label target by the
    // pattern's label under that shape. Counting the edge between the two
    // concrete node ids is label-independent and fully deterministic.
    let aid = result.artifact_node_id;
    let (session_nid, _) = kb
        .get_node_by_ext_id("Session", "session_id", "sess-ctx")
        .await
        .unwrap()
        .expect("session node");
    let (message_nid, _) = kb
        .get_node_by_ext_id("Message", "message_id", "msg-ctx")
        .await
        .unwrap()
        .expect("message node");
    let (action_nid, _) = kb
        .get_node_by_ext_id("Action", "action_id", "act-ctx")
        .await
        .unwrap()
        .expect("action node");

    assert_eq!(
        edge_count(&kb, "ATTACHED_TO", aid, session_nid).await,
        1,
        "Artifact must ATTACHED_TO its Session"
    );
    assert_eq!(
        edge_count(&kb, "ATTACHED_TO", aid, message_nid).await,
        1,
        "Artifact must ATTACHED_TO its Message"
    );
    assert_eq!(
        edge_count(&kb, "PRODUCED", action_nid, aid).await,
        1,
        "producing Action must PRODUCED the Artifact"
    );
}

/// Count directed `edge_type` edges from `from` to `to` by node id.
#[cfg(test)]
async fn edge_count(
    kb: &uniko_store::storage::KnowledgeBase,
    edge_type: &str,
    from: i64,
    to: i64,
) -> i64 {
    let cypher = format!(
        "MATCH (a)-[r:{edge_type}]->(b) WHERE id(a) = $from AND id(b) = $to RETURN count(r) AS c"
    );
    kb.db()
        .session()
        .query_with(&cypher)
        .param("from", from)
        .param("to", to)
        .fetch_all()
        .await
        .unwrap()
        .rows()
        .first()
        .expect("count row")
        .get::<i64>("c")
        .expect("count")
}
