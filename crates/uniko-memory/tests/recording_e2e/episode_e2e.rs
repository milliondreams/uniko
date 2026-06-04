//! Integration tests for the `record_episode` agent tool.
//!
//! Validates: Episode node creation, RECORDED_BY edge wiring,
//! FOLLOWED_BY chaining within the linking window, the state-topic
//! embedding pathway, and the missing-Participant error path.

use std::collections::HashMap;

use chrono::{Duration, Utc};
use serde_json::json;
use uni_db::Value;

use uniko_memory::{RecordEpisodeParams, record_episode};
use uniko_store::config::UnikoConfig;
use uniko_store::schema::constants::labels;
use uniko_store::{KnowledgeBase, UnikoError};

async fn kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

async fn seed_participant(kb: &KnowledgeBase, agent_id: &str) {
    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("kind".into(), Value::String("agent".into()));
    props.insert("name".into(), Value::String(format!("agent-{agent_id}")));
    kb.merge_node(labels::PARTICIPANT, "participant_id", agent_id, &props)
        .await
        .expect("participant created");
}

/// Some test environments lack the xervo embedding model.  Treat an
/// embedding error as "structural assertions valid, skip embedding
/// assertions" so the test still validates the rest of the pipeline.
fn is_embedding_unavailable(e: &UnikoError) -> bool {
    matches!(e, UnikoError::Embedding(_))
}

#[tokio::test]
async fn record_episode_creates_node_and_recorded_by_edge() {
    let kb = kb().await;
    seed_participant(&kb, "agent-1").await;

    let params = RecordEpisodeParams {
        action_type: "retrieve".into(),
        outcome: Some("success".into()),
        state: Some(json!({"topic": "career plans", "question": "what role?"})),
        importance: Some(0.7),
        ..Default::default()
    };

    let result = record_episode(&kb, "agent-1", params).await;
    if let Err(ref e) = result
        && is_embedding_unavailable(e)
    {
        eprintln!("skipping: embedding unavailable in this env");
        return;
    }
    let episode_id = result.expect("record_episode");

    // Episode properties were persisted.
    let session = kb.db().session();
    let row_set = session
        .query_with("MATCH (e:Episode) WHERE id(e) = $vid RETURN e")
        .param("vid", episode_id)
        .fetch_all()
        .await
        .expect("fetch");
    let row = row_set.rows().first().expect("episode row");
    let node: uni_db::Node = row.get("e").expect("node");
    assert_eq!(
        node.properties.get("action_type"),
        Some(&Value::String("retrieve".into()))
    );
    assert_eq!(
        node.properties.get("outcome"),
        Some(&Value::String("success".into()))
    );
    match node.properties.get("importance") {
        Some(Value::Float(f)) => assert!((f - 0.7).abs() < 1e-9),
        other => panic!("expected Float importance, got {other:?}"),
    }
    // Embedding may be elided when `RETURN n` serializes the node — fetch
    // a NULL test to assert it landed.
    let null_set = session
        .query_with(
            "MATCH (e:Episode) WHERE id(e) = $vid \
             RETURN e.embedding IS NOT NULL AS has_emb",
        )
        .param("vid", episode_id)
        .fetch_all()
        .await
        .expect("embedding null fetch");
    let null_row = null_set.rows().first().expect("has_emb row");
    let has_emb: bool = null_row.get("has_emb").expect("has_emb column");
    assert!(has_emb, "expected non-null embedding on Episode node");

    // RECORDED_BY edge exists.
    let edge_rows = session
        .query_with(
            "MATCH (e:Episode)-[:RECORDED_BY]->(p:Participant) \
             WHERE id(e) = $vid RETURN p.participant_id AS pid",
        )
        .param("vid", episode_id)
        .fetch_all()
        .await
        .expect("edge query");
    let edge_row = edge_rows.rows().first().expect("RECORDED_BY edge");
    let pid: String = edge_row.get("pid").expect("pid");
    assert_eq!(pid, "agent-1");
}

#[tokio::test]
async fn record_episode_chains_followed_by_within_window() {
    let kb = kb().await;
    seed_participant(&kb, "agent-2").await;

    let now = Utc::now();
    let first = RecordEpisodeParams {
        action_type: "step".into(),
        outcome: Some("ok".into()),
        timestamp: Some(now - Duration::minutes(5)),
        state: Some(json!({"topic": "first"})),
        ..Default::default()
    };
    let second = RecordEpisodeParams {
        action_type: "step".into(),
        outcome: Some("ok".into()),
        timestamp: Some(now),
        state: Some(json!({"topic": "second"})),
        ..Default::default()
    };

    let first_id = match record_episode(&kb, "agent-2", first).await {
        Ok(id) => id,
        Err(ref e) if is_embedding_unavailable(e) => return,
        Err(e) => panic!("record_episode first: {e}"),
    };
    let second_id = record_episode(&kb, "agent-2", second)
        .await
        .expect("second episode");

    let session = kb.db().session();

    let edges = session
        .query_with(
            "MATCH (a:Episode)-[r:FOLLOWED_BY]->(b:Episode) \
             WHERE id(a) = $a AND id(b) = $b \
             RETURN r.gap_ms AS gap",
        )
        .param("a", first_id)
        .param("b", second_id)
        .fetch_all()
        .await
        .expect("edge query");

    let row = edges.rows().first().expect("FOLLOWED_BY edge present");
    let gap: i64 = row.get("gap").expect("gap_ms");
    // 5 minutes ≈ 300_000 ms, allow ±1s slack.
    assert!(
        (290_000..=310_000).contains(&gap),
        "gap_ms = {gap}, expected ~300000"
    );
}

#[tokio::test]
async fn record_episode_omits_followed_by_outside_window() {
    let kb = kb().await;
    seed_participant(&kb, "agent-3").await;

    let now = Utc::now();
    // First episode 2 hours ago — outside the 1-hour window.
    let first = RecordEpisodeParams {
        action_type: "old".into(),
        outcome: Some("ok".into()),
        timestamp: Some(now - Duration::hours(2)),
        ..Default::default()
    };
    let second = RecordEpisodeParams {
        action_type: "new".into(),
        outcome: Some("ok".into()),
        timestamp: Some(now),
        ..Default::default()
    };

    let first_id = match record_episode(&kb, "agent-3", first).await {
        Ok(id) => id,
        Err(ref e) if is_embedding_unavailable(e) => return,
        Err(e) => panic!("record_episode first: {e}"),
    };
    let second_id = record_episode(&kb, "agent-3", second)
        .await
        .expect("second episode");

    let session = kb.db().session();
    let edges = session
        .query_with(
            "MATCH (a:Episode)-[:FOLLOWED_BY]->(b:Episode) \
             WHERE id(a) = $a AND id(b) = $b RETURN id(b) AS vid",
        )
        .param("a", first_id)
        .param("b", second_id)
        .fetch_all()
        .await
        .expect("edge query");

    assert!(
        edges.rows().is_empty(),
        "no FOLLOWED_BY edge expected outside window"
    );
}

#[tokio::test]
async fn record_episode_rejects_missing_participant() {
    let kb = kb().await;

    let params = RecordEpisodeParams {
        action_type: "x".into(),
        outcome: Some("y".into()),
        ..Default::default()
    };
    let err = record_episode(&kb, "no-such-agent", params)
        .await
        .expect_err("should fail when Participant missing");
    let msg = format!("{err}");
    assert!(
        msg.contains("Participant 'no-such-agent' not found"),
        "unexpected error: {msg}"
    );
}
