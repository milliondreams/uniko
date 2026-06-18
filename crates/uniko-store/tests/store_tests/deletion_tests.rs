//! Integration tests for cascade deletion and soft-forget.
//!
//! Build small graphs directly with `create_node` / `create_edge` (no
//! models needed) and assert post-delete graph state.

use std::collections::HashMap;

use uni_db::common::uni_btic::btic::POS_INF;
use uni_db::common::{TemporalValue, Value};
use uniko_store::NodeId;
use uniko_store::config::UnikoConfig;
use uniko_store::schema::btic::btic_active;
use uniko_store::storage::KnowledgeBase;

async fn test_kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

fn props(pairs: &[(&str, &str)]) -> HashMap<String, Value> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), Value::String((*v).to_string())))
        .collect()
}

async fn mk_message(kb: &KnowledgeBase, id: &str) -> NodeId {
    let mut p = props(&[("message_id", id), ("content", id)]);
    p.insert(
        "timestamp".into(),
        Value::String("2024-01-01T00:00:00Z".into()),
    );
    kb.create_node("Message", &p).await.unwrap()
}

async fn mk_chunk(kb: &KnowledgeBase, id: &str) -> NodeId {
    kb.create_node("Chunk", &props(&[("chunk_id", id), ("text", id)]))
        .await
        .unwrap()
}

async fn mk_observation(kb: &KnowledgeBase, id: &str) -> NodeId {
    kb.create_node(
        "Observation",
        &props(&[("observation_id", id), ("content", id)]),
    )
    .await
    .unwrap()
}

async fn mk_open_fact(kb: &KnowledgeBase, id: &str) -> NodeId {
    let mut p = props(&[("fact_id", id), ("subject", "s"), ("predicate", "p")]);
    let active = btic_active(chrono::Utc::now());
    p.insert(
        "valid_at".into(),
        Value::Temporal(TemporalValue::Btic {
            lo: active.lo(),
            hi: active.hi(),
            meta: active.meta(),
        }),
    );
    kb.create_node("Fact", &p).await.unwrap()
}

async fn mk_participant(kb: &KnowledgeBase, id: &str) -> NodeId {
    kb.create_node(
        "Participant",
        &props(&[("participant_id", id), ("kind", "human")]),
    )
    .await
    .unwrap()
}

async fn edge(kb: &KnowledgeBase, ty: &str, from: NodeId, to: NodeId) {
    kb.create_edge(ty, from, to, &HashMap::new()).await.unwrap();
}

async fn next_edge(kb: &KnowledgeBase, from: NodeId, to: NodeId, gap_ms: i64) {
    let mut p = HashMap::new();
    p.insert("gap_ms".into(), Value::Int(gap_ms));
    kb.create_edge("NEXT", from, to, &p).await.unwrap();
}

async fn exists(kb: &KnowledgeBase, nid: NodeId) -> bool {
    kb.get_node(nid).await.unwrap().is_some()
}

/// Read a node's string property, or `None`.
async fn str_prop(kb: &KnowledgeBase, nid: NodeId, key: &str) -> Option<String> {
    let (_, props) = kb.get_node(nid).await.unwrap()?;
    match props.get(key) {
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    }
}

/// Whether a node has `key` set to a non-null value (any type).
async fn has_prop(kb: &KnowledgeBase, nid: NodeId, key: &str) -> bool {
    let Some((_, props)) = kb.get_node(nid).await.unwrap() else {
        return false;
    };
    matches!(props.get(key), Some(v) if !matches!(v, Value::Null))
}

async fn mk_artifact(kb: &KnowledgeBase, id: &str) -> NodeId {
    kb.create_node(
        "Artifact",
        &props(&[("artifact_id", id), ("kind", "document")]),
    )
    .await
    .unwrap()
}

async fn mk_content(kb: &KnowledgeBase, id: &str) -> NodeId {
    let mut p = props(&[("content_id", id), ("mime", "text/plain")]);
    p.insert("size".into(), Value::Int(3));
    p.insert(
        "created_at".into(),
        Value::String("2024-01-01T00:00:00Z".into()),
    );
    kb.create_node("ArtifactContent", &p).await.unwrap()
}

/// Run a single-row read returning one i64 column, or `None`.
async fn query_i64(kb: &KnowledgeBase, cypher: &str) -> Option<i64> {
    let r = kb.db().session().query(cypher).await.unwrap();
    r.rows().first().and_then(|row| row.get::<i64>("v").ok())
}

#[tokio::test]
async fn delete_turn_cascades_and_splices() {
    let kb = test_kb().await;
    let m1 = mk_message(&kb, "m1").await;
    let m2 = mk_message(&kb, "m2").await;
    let m3 = mk_message(&kb, "m3").await;
    next_edge(&kb, m1, m2, 100).await;
    next_edge(&kb, m2, m3, 200).await;
    let c = mk_chunk(&kb, "c2").await;
    let o = mk_observation(&kb, "o2").await;
    edge(&kb, "HAS_CHUNK", m2, c).await;
    edge(&kb, "OBSERVED_IN", o, m2).await;

    let report = kb.delete_message("m2").await.unwrap();
    assert!(report.root_existed);
    assert_eq!(report.chains_repaired, 1);
    // m2 + its chunk + its observation gone; neighbors survive.
    assert!(!exists(&kb, m2).await);
    assert!(!exists(&kb, c).await);
    assert!(!exists(&kb, o).await);
    assert!(exists(&kb, m1).await);
    assert!(exists(&kb, m3).await);
    // NEXT spliced m1 -> m3 with summed gap.
    let gap = query_i64(
        &kb,
        "MATCH (:Message {message_id:'m1'})-[r:NEXT]->(:Message {message_id:'m3'}) \
         RETURN r.gap_ms AS v",
    )
    .await;
    assert_eq!(gap, Some(300));
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn multisupport_fact_survives_until_last_evidence() {
    let kb = test_kb().await;
    let m1 = mk_message(&kb, "m1").await;
    let m2 = mk_message(&kb, "m2").await;
    let o1 = mk_observation(&kb, "o1").await;
    let o2 = mk_observation(&kb, "o2").await;
    edge(&kb, "OBSERVED_IN", o1, m1).await;
    edge(&kb, "OBSERVED_IN", o2, m2).await;
    let f = mk_open_fact(&kb, "f1").await;
    edge(&kb, "SUPPORTED_BY", f, o1).await;
    edge(&kb, "SUPPORTED_BY", f, o2).await;

    // Delete m1: fact keeps support from o2 → still open.
    let r1 = kb.delete_message("m1").await.unwrap();
    assert_eq!(r1.facts_invalidated, 0);
    assert!(exists(&kb, f).await);
    assert_eq!(kb.fact_valid_at(f).await.unwrap().unwrap().hi(), POS_INF);

    // Delete m2: last support gone → fact soft-invalidated, node persists.
    let r2 = kb.delete_message("m2").await.unwrap();
    assert_eq!(r2.facts_invalidated, 1);
    assert!(exists(&kb, f).await);
    assert!(kb.fact_valid_at(f).await.unwrap().unwrap().hi() < POS_INF);
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn forget_turn_redacts_but_keeps_nodes() {
    let kb = test_kb().await;
    let m = mk_message(&kb, "m1").await;
    let c = mk_chunk(&kb, "c1").await;
    let o = mk_observation(&kb, "o1").await;
    edge(&kb, "HAS_CHUNK", m, c).await;
    edge(&kb, "OBSERVED_IN", o, m).await;

    let report = kb.forget_message("m1").await.unwrap();
    assert!(report.root_existed);
    assert_eq!(report.nodes_deleted, 0);
    assert!(report.nodes_redacted >= 3);
    // Nodes persist.
    assert!(exists(&kb, m).await);
    assert!(exists(&kb, c).await);
    assert!(exists(&kb, o).await);
    // Redaction flags / visibility set.
    let redacted = kb.fetch_redacted(&[m, c]).await.unwrap();
    assert!(redacted.contains(&m));
    assert!(redacted.contains(&c));
    let vis = kb.fetch_visibilities(&[o]).await.unwrap();
    assert_eq!(vis.get(&o).map(String::as_str), Some("redacted:forgotten"));
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn delete_session_cascades_messages() {
    let kb = test_kb().await;
    let mut sp = props(&[("session_id", "s1")]);
    sp.insert(
        "started_at".into(),
        Value::String("2024-01-01T00:00:00Z".into()),
    );
    let s = kb.create_node("Session", &sp).await.unwrap();
    let m1 = mk_message(&kb, "m1").await;
    let m2 = mk_message(&kb, "m2").await;
    let c = mk_chunk(&kb, "c1").await;
    edge(&kb, "IN_SESSION", m1, s).await;
    edge(&kb, "IN_SESSION", m2, s).await;
    edge(&kb, "HAS_CHUNK", m1, c).await;

    let report = kb.delete_session_cascade("s1").await.unwrap();
    assert!(report.root_existed);
    assert!(!exists(&kb, s).await);
    assert!(!exists(&kb, m1).await);
    assert!(!exists(&kb, m2).await);
    assert!(!exists(&kb, c).await);
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn forget_participant_erases_authored_data() {
    let kb = test_kb().await;
    let p = mk_participant(&kb, "p1").await;
    let m1 = mk_message(&kb, "m1").await;
    edge(&kb, "SENT_BY", m1, p).await;
    let o1 = mk_observation(&kb, "o1").await;
    edge(&kb, "OBSERVED_IN", o1, m1).await;
    edge(&kb, "ABOUT", o1, p).await;
    let f1 = mk_open_fact(&kb, "f1").await;
    edge(&kb, "SUPPORTED_BY", f1, o1).await;

    let report = kb.forget_participant("p1").await.unwrap();
    assert!(report.root_existed);
    // Participant, authored message, and observation about them are gone.
    assert!(!exists(&kb, p).await);
    assert!(!exists(&kb, m1).await);
    assert!(!exists(&kb, o1).await);
    // The fact grounded in that evidence is force-invalidated, not deleted.
    assert!(exists(&kb, f1).await);
    assert!(report.facts_invalidated >= 1);
    assert!(kb.fact_valid_at(f1).await.unwrap().unwrap().hi() < POS_INF);
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn invalidation_records_reason_on_node() {
    let kb = test_kb().await;
    let m = mk_message(&kb, "m1").await;
    let o = mk_observation(&kb, "o1").await;
    edge(&kb, "OBSERVED_IN", o, m).await;
    let f = mk_open_fact(&kb, "f1").await;
    edge(&kb, "SUPPORTED_BY", f, o).await;

    kb.delete_message("m1").await.unwrap();
    // Fact persists, soft-invalidated, with a node-level reason + timestamp.
    assert_eq!(
        str_prop(&kb, f, "invalidation_reason").await.as_deref(),
        Some("evidence_removed")
    );
    assert!(has_prop(&kb, f, "invalidated_at").await);
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn gdpr_invalidates_each_fact_once() {
    let kb = test_kb().await;
    let p = mk_participant(&kb, "p1").await;
    let m1 = mk_message(&kb, "m1").await;
    edge(&kb, "SENT_BY", m1, p).await;
    let o1 = mk_observation(&kb, "o1").await;
    edge(&kb, "OBSERVED_IN", o1, m1).await;
    edge(&kb, "ABOUT", o1, p).await;
    let f = mk_open_fact(&kb, "f1").await;
    edge(&kb, "SUPPORTED_BY", f, o1).await;

    // The fact is reachable by both the force-pass and the per-message
    // re-eval, but the open-guard counts it exactly once.
    let report = kb.forget_participant("p1").await.unwrap();
    assert_eq!(report.facts_invalidated, 1);
    assert_eq!(
        str_prop(&kb, f, "invalidation_reason").await.as_deref(),
        Some("subject_erased")
    );
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn delete_document_drops_unshared_content_keeps_shared() {
    let kb = test_kb().await;
    // a1 and a2 share content ac1; a1 alone owns ac2.
    let a1 = mk_artifact(&kb, "a1").await;
    let a2 = mk_artifact(&kb, "a2").await;
    let shared = mk_content(&kb, "shared").await;
    let owned = mk_content(&kb, "owned").await;
    edge(&kb, "HAS_CONTENT", a1, shared).await;
    edge(&kb, "HAS_CONTENT", a2, shared).await;
    edge(&kb, "HAS_CONTENT", a1, owned).await;

    let report = kb.delete_artifact("a1").await.unwrap();
    assert!(report.root_existed);
    assert!(!exists(&kb, a1).await);
    // Unshared content gone; shared content survives (a2 still references it).
    assert!(!exists(&kb, owned).await);
    assert!(exists(&kb, shared).await);
    assert!(exists(&kb, a2).await);
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn delete_missing_id_is_idempotent() {
    let kb = test_kb().await;
    let report = kb.delete_message("does-not-exist").await.unwrap();
    assert!(!report.root_existed);
    assert_eq!(report.nodes_deleted, 0);
    assert_eq!(report.facts_invalidated, 0);
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn purge_empties_the_graph() {
    let kb = test_kb().await;
    mk_message(&kb, "m1").await;
    mk_message(&kb, "m2").await;
    let report = kb.purge_all().await.unwrap();
    assert!(report.root_existed);
    assert!(report.nodes_deleted >= 2);
    let remaining = query_i64(&kb, "MATCH (n) RETURN count(n) AS v").await;
    assert_eq!(remaining, Some(0));
    kb.shutdown().await.unwrap();
}
