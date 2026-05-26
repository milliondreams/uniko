//! Integration tests for procedure promotion (P5, F41/F42/F43).

use std::collections::HashMap;

use chrono::{Duration, Utc};
use serde_json::json;
use uni_db::Value;

use uniko_cortex::procedures::{
    LifecycleConfig, STATUS_ACTIVE, STATUS_CANDIDATE, STATUS_DEPRECATED, match_procedures,
    promote_procedures_once, record_procedure_use,
};
use uniko_memory::{RecordEpisodeParams, record_episode};
use uniko_store::config::UnikoConfig;
use uniko_store::schema::constants::labels;
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, UnikoError};

async fn kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

async fn seed_participant(kb: &KnowledgeBase, agent_id: &str) {
    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("kind".into(), Value::String("agent".into()));
    kb.merge_node(labels::PARTICIPANT, "participant_id", agent_id, &props)
        .await
        .expect("participant");
}

/// Silent-skip helper intentionally removed — see notes in the
/// memory-side action_e2e tests.  Embedding failures must surface
/// as test failures, not green-bar lies.
#[allow(dead_code)]
fn embedding_must_work() {}

async fn record_seq(
    kb: &KnowledgeBase,
    agent_id: &str,
    first: &str,
    second: &str,
    when: chrono::DateTime<Utc>,
) -> Result<(), UnikoError> {
    record_episode(
        kb,
        agent_id,
        RecordEpisodeParams {
            action_type: first.into(),
            outcome: Some("success".into()),
            timestamp: Some(when),
            state: Some(json!({"topic": first})),
            ..Default::default()
        },
    )
    .await?;
    record_episode(
        kb,
        agent_id,
        RecordEpisodeParams {
            action_type: second.into(),
            outcome: Some("success".into()),
            timestamp: Some(when + Duration::seconds(30)),
            state: Some(json!({"topic": second})),
            ..Default::default()
        },
    )
    .await?;
    Ok(())
}

#[tokio::test]
async fn promote_procedures_creates_candidate_below_threshold() {
    let kb = kb().await;
    seed_participant(&kb, "a1").await;

    let base = Utc::now() - Duration::hours(1);
    // 1 pair only — below default threshold (3).
    record_seq(&kb, "a1", "investigate", "implement", base)
        .await
        .expect("record_seq must succeed (local embedder required)");

    let report = promote_procedures_once(&kb, "a1", LifecycleConfig::default())
        .await
        .expect("promote");
    assert_eq!(report.created, 1);
    assert_eq!(report.promoted, 0);

    let session = kb.db().session();
    let rows = session
        .query_with(
            "MATCH (p:Procedure) WHERE p.name = $n RETURN p.status AS st",
        )
        .param("n", "investigate → implement")
        .fetch_all()
        .await
        .expect("query");
    let st: String = rows.rows().first().expect("row").get("st").expect("status");
    assert_eq!(st, STATUS_CANDIDATE);
}

#[tokio::test]
async fn promote_procedures_promotes_when_threshold_reached() {
    let kb = kb().await;
    seed_participant(&kb, "a2").await;

    let base = Utc::now() - Duration::hours(3);
    // 3 pairs of the same sequence → meets default threshold.
    for i in 0..3 {
        let when = base + Duration::minutes(i * 5);
        record_seq(&kb, "a2", "build", "test", when)
            .await
            .expect("record_seq");
    }

    let report = promote_procedures_once(&kb, "a2", LifecycleConfig::default())
        .await
        .expect("promote");
    // Created in active status counts as `created`, not `promoted` —
    // promoted increments only when a candidate flips to active across
    // two runs.
    assert!(report.created + report.promoted >= 1);

    let session = kb.db().session();
    let rows = session
        .query_with(
            "MATCH (p:Procedure) WHERE p.name = $n RETURN p.status AS st",
        )
        .param("n", "build → test")
        .fetch_all()
        .await
        .expect("query");
    let st: String = rows.rows().first().expect("row").get("st").expect("status");
    assert_eq!(st, STATUS_ACTIVE);
}

#[tokio::test]
async fn match_procedures_filters_by_precondition() {
    let kb = kb().await;
    seed_participant(&kb, "a3").await;

    let base = Utc::now() - Duration::hours(2);
    for i in 0..3 {
        let when = base + Duration::minutes(i * 5);
        record_seq(&kb, "a3", "scan", "report", when)
            .await
            .expect("record_seq");
    }
    promote_procedures_once(&kb, "a3", LifecycleConfig::default())
        .await
        .expect("promote");

    let mut state = HashMap::new();
    state.insert("last_action_type".to_string(), "scan".to_string());
    let matched = match_procedures(&kb, &state).await.expect("match");
    assert!(matched.iter().any(|m| m.name == "scan → report"));

    state.insert("last_action_type".to_string(), "other".to_string());
    let matched = match_procedures(&kb, &state).await.expect("match");
    assert!(!matched.iter().any(|m| m.name == "scan → report"));
}

#[tokio::test]
async fn record_procedure_use_demotes_and_repromotes() {
    let kb = kb().await;
    seed_participant(&kb, "a4").await;

    // Create one Procedure manually so we don't depend on episode plumbing.
    let now = datetime_value(Utc::now());
    let mut props = HashMap::new();
    props.insert("name".into(), Value::String("seed".into()));
    props.insert("status".into(), Value::String(STATUS_ACTIVE.into()));
    props.insert("use_count".into(), Value::Int(0));
    props.insert("success_count".into(), Value::Int(0));
    props.insert("failure_count".into(), Value::Int(0));
    props.insert("effectiveness".into(), Value::Float(1.0));
    props.insert("created_at".into(), now.clone());
    props.insert("last_used_at".into(), now);
    kb.merge_node(labels::PROCEDURE, "procedure_id", "p_seed", &props)
        .await
        .expect("seed procedure");

    let cfg = LifecycleConfig::default();
    // 5 failures in a row → effectiveness 0.0 → demote.
    for _ in 0..5 {
        record_procedure_use(&kb, "p_seed", false, cfg)
            .await
            .expect("record use");
    }
    let st = read_status(&kb, "p_seed").await;
    assert_eq!(st, STATUS_DEPRECATED, "should be deprecated after failures");

    // 10 successes → effectiveness > 0.6 → re-promote.
    for _ in 0..10 {
        record_procedure_use(&kb, "p_seed", true, cfg)
            .await
            .expect("record use");
    }
    let st = read_status(&kb, "p_seed").await;
    assert_eq!(st, STATUS_ACTIVE, "should be re-promoted after wins");
}

async fn read_status(kb: &KnowledgeBase, pid: &str) -> String {
    let session = kb.db().session();
    let rows = session
        .query_with("MATCH (p:Procedure) WHERE p.procedure_id = $p RETURN p.status AS st")
        .param("p", pid)
        .fetch_all()
        .await
        .expect("query");
    rows.rows()
        .first()
        .expect("row")
        .get("st")
        .expect("status")
}
