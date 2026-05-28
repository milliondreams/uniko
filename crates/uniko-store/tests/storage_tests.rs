//! Integration tests for KnowledgeBase node/edge CRUD and batch operations.

use std::collections::HashMap;
use uni_db::Value;
use uniko_store::config::UnikoConfig;
use uniko_store::storage::KnowledgeBase;
use uniko_store::storage::edges::Direction;
use uniko_store::storage::filter::Filter;

async fn test_kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

// ── Node CRUD ──

#[tokio::test]
async fn test_create_and_get_node() {
    let kb = test_kb().await;
    let mut props = HashMap::new();
    props.insert("participant_id".into(), Value::String("p-1".into()));
    props.insert("kind".into(), Value::String("agent".into()));
    props.insert("name".into(), Value::String("TestBot".into()));

    let nid = kb.create_node("Participant", &props).await.unwrap();
    let (label, got) = kb.get_node(nid).await.unwrap().expect("node must exist");
    assert_eq!(label, "Participant");
    assert_eq!(got.get("name").and_then(|v| v.as_str()), Some("TestBot"));

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_get_node_by_ext_id() {
    let kb = test_kb().await;
    let mut props = HashMap::new();
    props.insert("message_id".into(), Value::String("m-1".into()));
    props.insert("content".into(), Value::String("hello".into()));
    props.insert(
        "timestamp".into(),
        Value::String("2024-01-01T00:00:00Z".into()),
    );

    let nid = kb.create_node("Message", &props).await.unwrap();
    let (found_id, found_props) = kb
        .get_node_by_ext_id("Message", "message_id", "m-1")
        .await
        .unwrap()
        .expect("must find by ext_id");
    assert_eq!(found_id, nid);
    assert_eq!(
        found_props.get("content").and_then(|v| v.as_str()),
        Some("hello")
    );

    // Non-existent ext_id returns None.
    assert!(
        kb.get_node_by_ext_id("Message", "message_id", "m-999")
            .await
            .unwrap()
            .is_none()
    );

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_update_node() {
    let kb = test_kb().await;
    let mut props = HashMap::new();
    props.insert("participant_id".into(), Value::String("p-1".into()));
    props.insert("kind".into(), Value::String("agent".into()));
    props.insert("name".into(), Value::String("Bot".into()));
    let nid = kb.create_node("Participant", &props).await.unwrap();

    // Update only the name.
    let mut updates = HashMap::new();
    updates.insert("name".into(), Value::String("UpdatedBot".into()));
    kb.update_node(nid, &updates).await.unwrap();

    let (_, got) = kb.get_node(nid).await.unwrap().unwrap();
    assert_eq!(got.get("name").and_then(|v| v.as_str()), Some("UpdatedBot"));
    // kind is preserved.
    assert_eq!(got.get("kind").and_then(|v| v.as_str()), Some("agent"));

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_delete_node() {
    let kb = test_kb().await;
    let mut props = HashMap::new();
    props.insert("participant_id".into(), Value::String("p-del".into()));
    props.insert("kind".into(), Value::String("human".into()));
    let nid = kb.create_node("Participant", &props).await.unwrap();

    assert!(kb.delete_node(nid).await.unwrap());
    assert!(kb.get_node(nid).await.unwrap().is_none());

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_merge_node_creates() {
    let kb = test_kb().await;
    let mut props = HashMap::new();
    props.insert("name".into(), Value::String("MergeBot".into()));
    props.insert("kind".into(), Value::String("agent".into()));

    let nid = kb
        .merge_node("Participant", "participant_id", "p-merge-1", &props)
        .await
        .unwrap();
    let (_, got) = kb.get_node(nid).await.unwrap().unwrap();
    assert_eq!(got.get("name").and_then(|v| v.as_str()), Some("MergeBot"));

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_merge_node_updates() {
    let kb = test_kb().await;
    let mut props = HashMap::new();
    props.insert("name".into(), Value::String("V1".into()));
    props.insert("kind".into(), Value::String("agent".into()));
    let nid1 = kb
        .merge_node("Participant", "participant_id", "p-merge-2", &props)
        .await
        .unwrap();

    // Merge again with updated name.
    let mut props2 = HashMap::new();
    props2.insert("name".into(), Value::String("V2".into()));
    props2.insert("kind".into(), Value::String("agent".into()));
    let nid2 = kb
        .merge_node("Participant", "participant_id", "p-merge-2", &props2)
        .await
        .unwrap();

    assert_eq!(nid1, nid2, "MERGE should not create a duplicate");
    let (_, got) = kb.get_node(nid2).await.unwrap().unwrap();
    assert_eq!(got.get("name").and_then(|v| v.as_str()), Some("V2"));

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_merge_idempotent() {
    let kb = test_kb().await;
    let mut props = HashMap::new();
    props.insert("name".into(), Value::String("Same".into()));
    props.insert("kind".into(), Value::String("agent".into()));

    let nid1 = kb
        .merge_node("Participant", "participant_id", "p-idem", &props)
        .await
        .unwrap();
    let nid2 = kb
        .merge_node("Participant", "participant_id", "p-idem", &props)
        .await
        .unwrap();
    assert_eq!(nid1, nid2);

    // Only one node should exist.
    let all = kb.query_nodes("Participant", None, None).await.unwrap();
    let count = all
        .iter()
        .filter(|(_, p)| p.get("participant_id").and_then(|v| v.as_str()) == Some("p-idem"))
        .count();
    assert_eq!(count, 1);

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_query_nodes_no_filter() {
    let kb = test_kb().await;
    for i in 0..5 {
        let mut props = HashMap::new();
        props.insert("entity_id".into(), Value::String(format!("ent-q-{i}")));
        props.insert("name".into(), Value::String(format!("Entity{i}")));
        props.insert("entity_type".into(), Value::String("concept".into()));
        kb.create_node("Entity", &props).await.unwrap();
    }

    let all = kb.query_nodes("Entity", None, None).await.unwrap();
    assert_eq!(all.len(), 5);

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_query_nodes_with_eq_filter() {
    let kb = test_kb().await;
    for (i, kind) in ["agent", "human", "agent", "service", "agent"]
        .iter()
        .enumerate()
    {
        let mut props = HashMap::new();
        props.insert("participant_id".into(), Value::String(format!("pf-{i}")));
        props.insert("kind".into(), Value::String(kind.to_string()));
        kb.create_node("Participant", &props).await.unwrap();
    }

    let filter = Filter::Eq("kind".into(), Value::String("agent".into()));
    let agents = kb
        .query_nodes("Participant", Some(&filter), None)
        .await
        .unwrap();
    assert_eq!(agents.len(), 3);

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_invalid_label_rejected() {
    let kb = test_kb().await;
    let result = kb.create_node("BogusLabel", &HashMap::new()).await;
    assert!(result.is_err());
    kb.shutdown().await.unwrap();
}

// ── Edge CRUD ──

#[tokio::test]
async fn test_create_and_get_edge() {
    let kb = test_kb().await;

    let mut mp = HashMap::new();
    mp.insert("message_id".into(), Value::String("m-e1".into()));
    mp.insert("content".into(), Value::String("hi".into()));
    mp.insert(
        "timestamp".into(),
        Value::String("2024-01-01T00:00:00Z".into()),
    );
    let m_id = kb.create_node("Message", &mp).await.unwrap();

    let mut pp = HashMap::new();
    pp.insert("participant_id".into(), Value::String("p-e1".into()));
    pp.insert("kind".into(), Value::String("human".into()));
    let p_id = kb.create_node("Participant", &pp).await.unwrap();

    let mut ep = HashMap::new();
    ep.insert("role".into(), Value::String("user".into()));
    let eid = kb.create_edge("SENT_BY", m_id, p_id, &ep).await.unwrap();

    let edges = kb
        .get_edges(m_id, "SENT_BY", Direction::Outgoing)
        .await
        .unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].id, eid);
    assert_eq!(
        edges[0].properties.get("role").and_then(|v| v.as_str()),
        Some("user")
    );

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_delete_edge() {
    let kb = test_kb().await;

    let mut mp = HashMap::new();
    mp.insert("message_id".into(), Value::String("m-de".into()));
    mp.insert("content".into(), Value::String("x".into()));
    mp.insert(
        "timestamp".into(),
        Value::String("2024-01-01T00:00:00Z".into()),
    );
    let m_id = kb.create_node("Message", &mp).await.unwrap();

    let mut pp = HashMap::new();
    pp.insert("participant_id".into(), Value::String("p-de".into()));
    pp.insert("kind".into(), Value::String("human".into()));
    let p_id = kb.create_node("Participant", &pp).await.unwrap();

    let eid = kb
        .create_edge("SENT_BY", m_id, p_id, &HashMap::new())
        .await
        .unwrap();
    assert!(kb.delete_edge(eid).await.unwrap());

    let edges = kb
        .get_edges(m_id, "SENT_BY", Direction::Outgoing)
        .await
        .unwrap();
    assert!(edges.is_empty());

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_delete_edges_between() {
    let kb = test_kb().await;

    let mut m1 = HashMap::new();
    m1.insert("message_id".into(), Value::String("m-deb1".into()));
    m1.insert("content".into(), Value::String("a".into()));
    m1.insert(
        "timestamp".into(),
        Value::String("2024-01-01T00:00:00Z".into()),
    );
    let id1 = kb.create_node("Message", &m1).await.unwrap();

    let mut m2 = HashMap::new();
    m2.insert("message_id".into(), Value::String("m-deb2".into()));
    m2.insert("content".into(), Value::String("b".into()));
    m2.insert(
        "timestamp".into(),
        Value::String("2024-01-01T00:00:01Z".into()),
    );
    let id2 = kb.create_node("Message", &m2).await.unwrap();

    // Create two NEXT edges between them.
    kb.create_edge("NEXT", id1, id2, &HashMap::new())
        .await
        .unwrap();
    kb.create_edge("NEXT", id1, id2, &HashMap::new())
        .await
        .unwrap();

    let deleted = kb.delete_edges_between("NEXT", id1, id2).await.unwrap();
    assert_eq!(deleted, 2);

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_update_edge() {
    let kb = test_kb().await;

    let mut m1 = HashMap::new();
    m1.insert("message_id".into(), Value::String("m-ue1".into()));
    m1.insert("content".into(), Value::String("a".into()));
    m1.insert(
        "timestamp".into(),
        Value::String("2024-01-01T00:00:00Z".into()),
    );
    let id1 = kb.create_node("Message", &m1).await.unwrap();

    let mut m2 = HashMap::new();
    m2.insert("message_id".into(), Value::String("m-ue2".into()));
    m2.insert("content".into(), Value::String("b".into()));
    m2.insert(
        "timestamp".into(),
        Value::String("2024-01-01T00:00:01Z".into()),
    );
    let id2 = kb.create_node("Message", &m2).await.unwrap();

    let mut ep = HashMap::new();
    ep.insert("gap_ms".into(), Value::Int(100));
    let eid = kb.create_edge("NEXT", id1, id2, &ep).await.unwrap();

    // Update gap_ms.
    let mut updates = HashMap::new();
    updates.insert("gap_ms".into(), Value::Int(500));
    kb.update_edge(eid, &updates).await.unwrap();

    let edges = kb
        .get_edges(id1, "NEXT", Direction::Outgoing)
        .await
        .unwrap();
    assert_eq!(
        edges[0].properties.get("gap_ms").and_then(|v| v.as_i64()),
        Some(500)
    );

    kb.shutdown().await.unwrap();
}

// ── Batch ──

#[tokio::test]
async fn test_batch_create_nodes() {
    let kb = test_kb().await;
    let items: Vec<HashMap<String, Value>> = (0..50)
        .map(|i| {
            let mut props = HashMap::new();
            props.insert("chunk_id".into(), Value::String(format!("art::{i}")));
            props.insert("text".into(), Value::String(format!("chunk {i}")));
            props
        })
        .collect();

    let ids = kb.batch_create_nodes("Chunk", &items).await.unwrap();
    assert_eq!(ids.len(), 50);

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_batch_create_edges() {
    let kb = test_kb().await;

    let mut art = HashMap::new();
    art.insert("artifact_id".into(), Value::String("art-b".into()));
    art.insert("kind".into(), Value::String("file".into()));
    let art_id = kb.create_node("Artifact", &art).await.unwrap();

    let items: Vec<HashMap<String, Value>> = (0..10)
        .map(|i| {
            let mut props = HashMap::new();
            props.insert("chunk_id".into(), Value::String(format!("art-b:{i}")));
            props.insert("text".into(), Value::String(format!("chunk {i}")));
            props
        })
        .collect();
    let chunk_ids = kb.batch_create_nodes("Chunk", &items).await.unwrap();

    let edges: Vec<_> = chunk_ids
        .iter()
        .enumerate()
        .map(|(i, &cid)| {
            let mut props = HashMap::new();
            props.insert("index".into(), Value::Int(i as i64));
            (art_id, cid, props)
        })
        .collect();

    let eids = kb.batch_create_edges("HAS_CHUNK", &edges).await.unwrap();
    assert_eq!(eids.len(), 10);

    kb.shutdown().await.unwrap();
}

// ── Prepared statement & single-tx tests ──

#[tokio::test]
async fn test_prepare_basic() {
    let kb = test_kb().await;

    // Verify tx.prepare() works and doesn't hang.
    let session = kb.db().session();
    let tx = session.tx().await.unwrap();
    let prepared = tx
        .prepare("CREATE (n:Chunk {chunk_id: $cid, text: $txt}) RETURN id(n) AS vid")
        .await
        .unwrap();

    let result = prepared
        .execute(&[
            ("cid", Value::String("test-1".into())),
            ("txt", Value::String("hello".into())),
        ])
        .await
        .unwrap();
    let rows = result.rows();
    eprintln!("prepared execute returned {} rows", rows.len());
    for (i, row) in rows.iter().enumerate() {
        eprintln!("  row {i}: {:?}", row);
    }
    assert!(!rows.is_empty(), "prepared execute returned no rows");
    let vid: i64 = rows.first().unwrap().get("vid").unwrap();
    eprintln!("vid = {vid}");
    assert!(vid >= 0);

    let result2 = prepared
        .execute(&[
            ("cid", Value::String("test-2".into())),
            ("txt", Value::String("world".into())),
        ])
        .await
        .unwrap();
    let vid2: i64 = result2.rows().first().unwrap().get("vid").unwrap();
    assert!(vid2 > vid);

    tx.commit().await.unwrap();
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_batch_create_nodes_and_edges() {
    let kb = test_kb().await;

    // Create an Artifact to wire chunk edges to.
    let mut art_props = HashMap::new();
    art_props.insert("artifact_id".into(), Value::String("art-x".into()));
    art_props.insert("kind".into(), Value::String("file".into()));
    let art_id = kb.create_node("Artifact", &art_props).await.unwrap();

    // Create chunks + HAS_CHUNK edges in a single transaction.
    let chunk_props: Vec<HashMap<String, Value>> = (0..5)
        .map(|i| {
            let mut props = HashMap::new();
            props.insert("chunk_id".into(), Value::String(format!("cx-{i}")));
            props.insert("text".into(), Value::String(format!("chunk {i}")));
            props
        })
        .collect();

    let node_ids = kb.batch_create_nodes("Chunk", &chunk_props).await.unwrap();
    assert_eq!(node_ids.len(), 5);

    let edge_specs: Vec<_> = node_ids
        .iter()
        .enumerate()
        .map(|(i, &nid)| {
            let mut props = HashMap::new();
            props.insert("index".into(), Value::Int(i as i64));
            (art_id, nid, props)
        })
        .collect();
    kb.batch_create_edges_fast("HAS_CHUNK", Some("Artifact"), Some("Chunk"), &edge_specs)
        .await
        .unwrap();

    // Verify edges exist.
    let edges = kb
        .get_edges(art_id, "HAS_CHUNK", Direction::Outgoing)
        .await
        .unwrap();
    assert_eq!(edges.len(), 5);

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_bump_modality_presence_concurrent_no_lost_update() {
    // Regression: `bump_modality_presence` used to read the current
    // map via a fresh session outside its own transaction, so two
    // concurrent bumps could race and clobber each other. Both flips
    // must persist after the dust settles.
    let kb = test_kb().await;
    kb.init_kb_stats().await.unwrap();

    let kb_a = kb.clone();
    let kb_b = kb.clone();
    let a = tokio::spawn(async move { kb_a.bump_modality_presence("image").await });
    let b = tokio::spawn(async move { kb_b.bump_modality_presence("audio").await });
    a.await.unwrap().unwrap();
    b.await.unwrap().unwrap();

    let presence = kb.read_modality_presence().await.unwrap();
    assert!(
        presence.has_image_content,
        "image flag was lost: {presence:?}"
    );
    assert!(
        presence.has_audio_content,
        "audio flag was lost: {presence:?}"
    );

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_batch_create_edges_fast_rejects_invalid_property_name() {
    // Regression: the fast/bulk path used to silently discard the
    // result of `validate_property_name`, letting invalid keys reach
    // uni-db's `bulk_insert_edges`. It must now surface a Schema error.
    let kb = test_kb().await;

    let mut props_a = HashMap::new();
    props_a.insert("artifact_id".into(), Value::String("art-bad".into()));
    props_a.insert("kind".into(), Value::String("file".into()));
    let art_id = kb.create_node("Artifact", &props_a).await.unwrap();

    let mut props_c = HashMap::new();
    props_c.insert("chunk_id".into(), Value::String("cx-bad".into()));
    props_c.insert("text".into(), Value::String("c".into()));
    let chunk_id = kb.create_node("Chunk", &props_c).await.unwrap();

    let mut bad_edge_props = HashMap::new();
    bad_edge_props.insert("".into(), Value::Int(1));
    let edge_specs = vec![(art_id, chunk_id, bad_edge_props)];

    let err = kb
        .batch_create_edges_fast("HAS_CHUNK", Some("Artifact"), Some("Chunk"), &edge_specs)
        .await
        .expect_err("empty property name must be rejected");
    assert!(
        matches!(err, uniko_store::UnikoError::Schema(_)),
        "expected Schema error, got {err:?}"
    );

    kb.shutdown().await.unwrap();
}
