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
    assert_eq!(
        stats.facts_created, 2,
        "two distinct (subject, predicate) clusters"
    );
    assert_eq!(stats.facts_reinforced, 0);

    assert_eq!(count_facts(&kb).await, 2);

    let caroline_fact = fetch_fact(&kb, "caroline", "researches")
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

// ── F38 contradiction detection ──────────────────────────────────────

async fn count_invalidates_edges_to(kb: &KnowledgeBase, stale_fact: NodeId) -> i64 {
    let session = kb.db().session();
    let result = session
        .query_with(
            "MATCH (new:Fact)-[:INVALIDATES]->(old:Fact) \
             WHERE id(old) = $vid \
             RETURN count(*) AS n",
        )
        .param("vid", stale_fact)
        .fetch_all()
        .await
        .expect("invalidates count");
    result
        .rows()
        .first()
        .map(|r| r.get::<i64>("n").unwrap_or(0))
        .unwrap_or(0)
}

async fn fact_node_id(kb: &KnowledgeBase, subject: &str, predicate: &str, object: &str) -> NodeId {
    let session = kb.db().session();
    let result = session
        .query_with(
            "MATCH (f:Fact) \
             WHERE f.subject = $s AND f.predicate = $p AND f.object = $o \
             RETURN id(f) AS nid",
        )
        .param("s", subject)
        .param("p", predicate)
        .param("o", object)
        .fetch_all()
        .await
        .expect("fact node lookup");
    result
        .rows()
        .first()
        .and_then(|r| r.get::<i64>("nid").ok())
        .expect("fact node present")
}

#[tokio::test]
async fn balanced_split_does_not_trigger_invalidation() {
    let kb = test_kb().await;

    // Seed an established Fact: 4 observations all saying "X is Y".
    for i in 0..4 {
        seed_observation(
            &kb,
            "alex",
            Some("uses"),
            Some("rust"),
            "Alex uses rust",
            ts(2024, 1, 1 + i),
        )
        .await;
    }
    run_cycle(&kb, "agent-1", None).await.expect("cycle 1");

    // Balanced 50/50 split — 3 say "rust", 3 say "go".  Below the 40%
    // contradiction threshold (it's exactly 50%, but the rule is
    // "contradicting > 40%" → 3/6 = 50% IS over; this should trigger).
    // Use 3 rust + 2 go to land at 40% contradiction — boundary case.
    for i in 0..3 {
        seed_observation(
            &kb,
            "alex",
            Some("uses"),
            Some("rust"),
            "still rust",
            ts(2024, 2, 1 + i),
        )
        .await;
    }
    for i in 0..2 {
        seed_observation(
            &kb,
            "alex",
            Some("uses"),
            Some("go"),
            "now go",
            ts(2024, 2, 4 + i),
        )
        .await;
    }
    let stats = run_cycle(&kb, "agent-1", None).await.expect("cycle 2");

    // 2/5 = 40%, not strictly > 40% → no invalidation.
    assert_eq!(
        stats.facts_invalidated, 0,
        "40% contradiction is boundary, not above threshold",
    );
    assert_eq!(stats.drift_alerts, 0);
}

#[tokio::test]
async fn majority_contradiction_invalidates_old_fact() {
    let kb = test_kb().await;

    // Cycle 1: 4 observations saying "Alex uses rust" → Fact created.
    for i in 0..4 {
        seed_observation(
            &kb,
            "alex",
            Some("uses"),
            Some("rust"),
            "Alex uses rust",
            ts(2024, 1, 1 + i),
        )
        .await;
    }
    let s1 = run_cycle(&kb, "agent-1", None).await.expect("cycle 1");
    assert_eq!(s1.facts_created, 1);
    let old_fact_id = fact_node_id(&kb, "alex", "uses", "rust").await;

    // Cycle 2: 5 fresh observations, 4 say "go" and 1 says "rust" →
    // 80% contradict "rust" relative to the new canonical "go" → trigger.
    for i in 0..4 {
        seed_observation(
            &kb,
            "alex",
            Some("uses"),
            Some("go"),
            "Alex switched to go",
            ts(2024, 2, 1 + i),
        )
        .await;
    }
    seed_observation(
        &kb,
        "alex",
        Some("uses"),
        Some("rust"),
        "one straggler still rust",
        ts(2024, 2, 6),
    )
    .await;

    let s2 = run_cycle(&kb, "agent-1", None).await.expect("cycle 2");

    // New Fact for (alex, uses, go) created; old (alex, uses, rust)
    // invalidated; the canonical-matching "rust" straggler reinforces
    // the old Fact via upsert_fact_by_triple (same fact_id key).
    assert_eq!(s2.facts_invalidated, 1, "old rust fact invalidated");

    let new_fact_id = fact_node_id(&kb, "alex", "uses", "go").await;
    assert_ne!(new_fact_id, old_fact_id);

    // INVALIDATES edge from new → old.
    assert_eq!(count_invalidates_edges_to(&kb, old_fact_id).await, 1);

    // Old Fact's BTIC closed.
    let old_fact = fetch_fact(&kb, "alex", "uses")
        .await
        .expect("rust fact still present");
    // The mode-by-recency tiebreak picks the most recently observed
    // value when counts tie; here counts differ (5 rust vs 4 go in
    // store) but fetch_fact returns *some* Fact for (alex, uses) — we
    // want to confirm one of them now has a closed interval.
    let _ = old_fact;
    let session = kb.db().session();
    let probe = session
        .query_with(
            "MATCH (f:Fact) WHERE id(f) = $vid \
             RETURN f.valid_at AS vat",
        )
        .param("vid", old_fact_id)
        .fetch_all()
        .await
        .expect("probe");
    let row = probe.rows().first().expect("old fact row");
    let node_row = session
        .query_with("MATCH (f:Fact) WHERE id(f) = $vid RETURN f")
        .param("vid", old_fact_id)
        .fetch_all()
        .await
        .expect("node probe");
    let node: uni_db::Node = node_row
        .rows()
        .first()
        .expect("node row")
        .get("f")
        .expect("node");
    let _ = row;

    // uni-db 2.1.0 returns a Node's Temporal columns as structured
    // `Value::Temporal(Btic{..})` (uni `4583ee870`), no longer
    // stringified.  A closed BTIC has a finite `hi`; an open one has
    // `hi == POS_INF`.  The majority contradiction must have closed it.
    match node.properties.get("valid_at") {
        Some(Value::Temporal(uni_db::common::TemporalValue::Btic { hi, .. })) => {
            assert!(
                *hi < uni_db::common::uni_btic::btic::POS_INF,
                "old fact BTIC was not closed (hi == POS_INF): hi={hi}",
            );
        }
        other => panic!("expected Temporal(Btic) valid_at, got {other:?}"),
    }
}

#[tokio::test]
async fn drift_threshold_flips_entity_unstable() {
    let kb = test_kb().await;

    // Seed initial Fact, then drive 5 successive contradictions on the
    // same subject "kim" with different objects each round.  Each round
    // should invalidate the prior open Fact.
    let objects = ["a", "b", "c", "d", "e", "f"];
    for (round, obj) in objects.iter().enumerate() {
        for i in 0..4 {
            seed_observation(
                &kb,
                "kim",
                Some("prefers"),
                Some(obj),
                "Kim prefers something",
                ts(2024, 1 + round as u32, 1 + i),
            )
            .await;
        }
        // Add a single contradicting observation so the next round is
        // pre-contradicted relative to the prior canonical.
        if round + 1 < objects.len() {
            seed_observation(
                &kb,
                "kim",
                Some("prefers"),
                Some(objects[round + 1]),
                "tiny hint of next",
                ts(2024, 1 + round as u32, 28),
            )
            .await;
        }
        let _ = run_cycle(&kb, "agent-1", None).await.expect("cycle");
    }

    // After 5+ invalidations, the kim Entity should be flagged unstable.
    let session = kb.db().session();
    let result = session
        .query_with(
            "MATCH (e:Entity {name: 'kim'}) RETURN e.unstable AS u, e.invalidation_count AS c",
        )
        .fetch_all()
        .await
        .expect("entity probe");
    let row = result.rows().first().expect("kim entity row");
    let unstable: bool = row.get("u").unwrap_or(false);
    let count: i64 = row.get("c").unwrap_or(0);
    assert!(
        unstable,
        "kim should be unstable after {count} invalidations",
    );
    assert!(count > 4, "expected >4 invalidations, got {count}");
}
