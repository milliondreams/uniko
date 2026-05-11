//! Integration tests for P4 Consolidation: Observations → Facts.
//!
//! Validates the cycle end-to-end against an in-memory KB.  Seeds
//! Observations directly with structured triples (bypassing the P3 NLP
//! pipeline) so the test stays focused on consolidation logic and runs
//! in milliseconds without needing an ONNX model.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, TimeZone, Utc};
use uni_db::Value;

use uniko_store::config::UnikoConfig;
use uniko_store::schema::constants::labels;
use uniko_store::{KnowledgeBase, NodeId};

use uniko_memory::consolidation::run_cycle;

async fn test_kb() -> Arc<KnowledgeBase> {
    Arc::new(
        KnowledgeBase::in_memory(UnikoConfig::default())
            .await
            .expect("in-memory KB"),
    )
}

fn ts(y: i32, m: u32, d: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
}

async fn seed_observation(
    kb: &KnowledgeBase,
    subject: &str,
    predicate: Option<&str>,
    object: Option<&str>,
    content: &str,
    observed_at: DateTime<Utc>,
) -> NodeId {
    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert(
        "observation_id".into(),
        Value::String(uniko_store::id::new_id()),
    );
    props.insert("content".into(), Value::String(content.into()));
    props.insert("subject".into(), Value::String(subject.into()));
    if let Some(p) = predicate {
        props.insert("predicate".into(), Value::String(p.into()));
    }
    if let Some(o) = object {
        props.insert("object".into(), Value::String(o.into()));
    }
    props.insert(
        "observed_at".into(),
        Value::String(observed_at.to_rfc3339()),
    );
    props.insert("confidence".into(), Value::Float(0.85));
    kb.create_node(labels::OBSERVATION, &props)
        .await
        .expect("seed observation")
}

async fn count_facts(kb: &KnowledgeBase) -> i64 {
    let session = kb.db().session();
    let result = session
        .query_with("MATCH (f:Fact) RETURN count(f) AS n")
        .fetch_all()
        .await
        .expect("count facts");
    result
        .rows()
        .first()
        .and_then(|row| row.get::<i64>("n").ok())
        .unwrap_or(0)
}

async fn fetch_fact(
    kb: &KnowledgeBase,
    subject: &str,
    predicate: &str,
) -> Option<HashMap<String, Value>> {
    let session = kb.db().session();
    let cypher = "MATCH (f:Fact {subject: $s, predicate: $p}) RETURN f";
    let result = session
        .query_with(cypher)
        .param("s", subject)
        .param("p", predicate)
        .fetch_all()
        .await
        .ok()?;
    let row = result.rows().first()?;
    let node: uni_db::Node = row.get("f").ok()?;
    Some(node.properties)
}

#[tokio::test]
async fn cycle_derives_one_fact_per_triple_cluster() {
    let kb = test_kb().await;

    // Four observations with the same (subject, predicate) and matching
    // object — should consolidate to a single Fact with count=4.
    for (i, obj) in [
        "adoption agencies",
        "adoption agencies",
        "adoption agencies",
        "adoption agencies",
    ]
    .iter()
    .enumerate()
    {
        seed_observation(
            &kb,
            "caroline",
            Some("researches"),
            Some(obj),
            &format!("Caroline researches adoption agencies (msg {i})"),
            ts(2024, 1, 1 + i as u32),
        )
        .await;
    }

    // Unrelated observation — should produce its own Fact.
    seed_observation(
        &kb,
        "jon",
        Some("got"),
        Some("temp job"),
        "Jon got a temp job",
        ts(2024, 1, 5),
    )
    .await;

    // Rule-based observation with no triple — should be skipped.
    seed_observation(
        &kb,
        "melanie",
        None,
        None,
        "Melanie loves rain",
        ts(2024, 1, 6),
    )
    .await;

    let stats = run_cycle(&kb, "agent-1", None).await.expect("cycle ok");

    assert_eq!(stats.observations_processed, 5, "5 triple-bearing obs");
    assert_eq!(stats.facts_created, 2, "two distinct (subject, predicate) clusters");
    assert_eq!(stats.facts_reinforced, 0);

    assert_eq!(count_facts(&kb).await, 2);

    let caroline_fact =
        fetch_fact(&kb, "caroline", "researches")
            .await
            .expect("caroline fact present");
    assert_eq!(
        caroline_fact
            .get("observation_count")
            .and_then(|v| v.as_i64()),
        Some(4),
        "all four contributing observations counted",
    );
    let conf = caroline_fact
        .get("confidence")
        .and_then(|v| v.as_f64())
        .expect("confidence stored");
    // Laplace: (4+1)/(4+2) = 0.833...
    assert!((conf - (5.0 / 6.0)).abs() < 1e-6, "Laplace confidence");
}

#[tokio::test]
async fn cycle_is_idempotent_within_run() {
    let kb = test_kb().await;
    seed_observation(
        &kb,
        "alex",
        Some("loves"),
        Some("rust"),
        "Alex loves rust",
        ts(2024, 2, 1),
    )
    .await;

    let first = run_cycle(&kb, "agent-1", None).await.expect("first ok");
    assert_eq!(first.observations_processed, 1);
    assert_eq!(first.facts_created, 1);

    // Second cycle: the PROCESSED edge from the first cycle excludes
    // the same Observation, so the body is empty.
    let second = run_cycle(&kb, "agent-1", None).await.expect("second ok");
    assert_eq!(
        second.observations_processed, 0,
        "no unprocessed observations remain"
    );
    assert_eq!(second.facts_created, 0);
    assert_eq!(count_facts(&kb).await, 1);
}

#[tokio::test]
async fn cycle_reinforces_existing_fact() {
    let kb = test_kb().await;

    // Cycle 1: two contributors → Fact created with count=2.
    for i in 0..2 {
        seed_observation(
            &kb,
            "sam",
            Some("plays"),
            Some("piano"),
            "Sam plays piano",
            ts(2024, 3, 1 + i),
        )
        .await;
    }
    let s1 = run_cycle(&kb, "agent-1", None).await.expect("cycle 1");
    assert_eq!(s1.facts_created, 1);

    // Cycle 2: three new contributors → reinforcement.
    for i in 0..3 {
        seed_observation(
            &kb,
            "sam",
            Some("plays"),
            Some("piano"),
            "Sam plays piano (again)",
            ts(2024, 4, 1 + i),
        )
        .await;
    }
    let s2 = run_cycle(&kb, "agent-1", None).await.expect("cycle 2");
    assert_eq!(s2.facts_created, 0);
    assert_eq!(s2.facts_reinforced, 1);
    assert_eq!(s2.observations_processed, 3);

    let fact = fetch_fact(&kb, "sam", "plays").await.expect("sam fact");
    assert_eq!(
        fact.get("observation_count").and_then(|v| v.as_i64()),
        Some(5),
        "2 + 3 contributors total",
    );
}
