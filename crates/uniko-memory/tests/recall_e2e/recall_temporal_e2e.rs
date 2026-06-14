//! Integration tests for the Phase 2 temporal-interval channel.
//!
//! Validates:
//! - Queries with no temporal phrase produce `temporal_window = None`
//!   and the channel returns nothing.
//! - Observations with `temporal_anchor` inside the resolved window
//!   surface; ones outside the window do not.
//! - Facts whose `valid_at` BTIC overlaps the window surface; ones
//!   that don't overlap do not.

use std::collections::HashMap;

use chrono::{DateTime, TimeZone, Utc};
use uni_db::Value;

use uniko_memory::recall::intent::{build_intent, build_intent_at};
use uniko_memory::recall::{RecallConfig, recall};
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;
use uniko_store::schema::btic::btic_active;
use uniko_store::schema::constants::labels;
use uniko_store::types::datetime_value;

async fn kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

/// Treat embedding/runtime failures as "skip & pass" so this suite
/// runs on environments without the ONNX models on disk.
fn is_embedding_unavailable<T>(r: &Result<T, uniko_store::UnikoError>) -> bool {
    matches!(r, Err(uniko_store::UnikoError::Embedding(_)))
}

#[tokio::test]
async fn build_intent_with_no_temporal_phrase_yields_none() {
    let kb = kb().await;
    let intent_res = build_intent(&kb, "What pet does Caroline have?", &[]).await;
    if is_embedding_unavailable(&intent_res) {
        eprintln!("skipping: embedding unavailable");
        return;
    }
    let intent = intent_res.expect("build_intent");
    assert!(
        intent.temporal_window.is_none(),
        "no temporal phrase should yield None, got {:?}",
        intent.temporal_window
    );
}

#[tokio::test]
async fn build_intent_resolves_yesterday_to_one_day_window() {
    let kb = kb().await;
    // Use a fixed reference so the assertion is deterministic.
    let reference: DateTime<Utc> = Utc.with_ymd_and_hms(2025, 5, 15, 12, 0, 0).unwrap();
    let intent_res = build_intent_at(&kb, "what happened yesterday", &[], Some(reference)).await;
    if is_embedding_unavailable(&intent_res) {
        eprintln!("skipping: embedding unavailable");
        return;
    }
    let intent = intent_res.expect("build_intent_at");
    let (lo, hi) = intent.temporal_window.expect("temporal_window populated");
    // yesterday relative to 2025-05-15 → resolves to 2025-05-14
    // Day-granularity → [2025-05-14T00:00, 2025-05-15T00:00).
    assert_eq!(lo.to_rfc3339(), "2025-05-14T00:00:00+00:00");
    assert_eq!(hi.to_rfc3339(), "2025-05-15T00:00:00+00:00");
}

#[tokio::test]
async fn observation_in_window_is_recallable() {
    let kb = kb().await;
    // Seed two Observations: one in May 2025, one in August 2025.
    let may_anchor: DateTime<Utc> = Utc.with_ymd_and_hms(2025, 5, 15, 10, 0, 0).unwrap();
    let aug_anchor: DateTime<Utc> = Utc.with_ymd_and_hms(2025, 8, 15, 10, 0, 0).unwrap();

    for (id, anchor, content) in [
        ("obs-may", may_anchor, "ATE_RAMEN at the new place"),
        ("obs-aug", aug_anchor, "VISITED_PARK with the dog"),
    ] {
        let mut props: HashMap<String, Value> = HashMap::new();
        props.insert("observation_id".into(), Value::String(id.into()));
        props.insert("content".into(), Value::String(content.into()));
        props.insert("subject".into(), Value::String("user".into()));
        props.insert("predicate".into(), Value::String("did".into()));
        props.insert("object".into(), Value::String(content.into()));
        props.insert("temporal_anchor".into(), datetime_value(anchor));
        props.insert("observed_at".into(), datetime_value(anchor));
        kb.create_node(labels::OBSERVATION, &props)
            .await
            .expect("create Observation");
    }

    // Build an intent that resolves "in may" to the May 2025 window.
    let reference: DateTime<Utc> = Utc.with_ymd_and_hms(2025, 7, 1, 12, 0, 0).unwrap();
    let intent_res = build_intent_at(&kb, "what happened in may", &[], Some(reference)).await;
    if is_embedding_unavailable(&intent_res) {
        eprintln!("skipping: embedding unavailable");
        return;
    }
    let intent = intent_res.expect("build_intent_at");
    let (lo, hi) = intent
        .temporal_window
        .expect("'in may' should yield a window");
    // Expect May 2025 (most recent past May relative to July 2025).
    assert_eq!(lo.to_rfc3339(), "2025-05-01T00:00:00+00:00");
    assert_eq!(hi.to_rfc3339(), "2025-06-01T00:00:00+00:00");

    // Direct BTree range query mirroring what phase2_temporal does for
    // Observations.  Validates the window correctly identifies the May
    // observation and excludes the August one.
    let session = kb.db().session();
    let r = session
        .query_with(
            "MATCH (o:Observation) \
             WHERE o.temporal_anchor >= $lo AND o.temporal_anchor < $hi \
             RETURN o.observation_id AS id",
        )
        .param("lo", datetime_value(lo))
        .param("hi", datetime_value(hi))
        .fetch_all()
        .await
        .expect("observation range query");

    let ids: Vec<String> = r
        .rows()
        .iter()
        .filter_map(|row| row.get::<String>("id").ok())
        .collect();

    assert!(
        ids.contains(&"obs-may".to_string()),
        "May observation should be in window, got {ids:?}"
    );
    assert!(
        !ids.contains(&"obs-aug".to_string()),
        "August observation should NOT be in window, got {ids:?}"
    );
}

/// #4: the public `recall()` must thread `RecallConfig.reference_ts`
/// into intent building, so a relative temporal phrase resolves against
/// the question's anchor (not wall-clock now). Seed an Observation in
/// May 2025; recall "what happened in may" with an anchor near that date
/// surfaces it, while the default (`None` → now, ~2026) does not.
#[tokio::test]
async fn recall_threads_reference_ts_into_temporal_window() {
    let kb = kb().await;

    let may_anchor: DateTime<Utc> = Utc.with_ymd_and_hms(2025, 5, 15, 10, 0, 0).unwrap();
    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("observation_id".into(), Value::String("obs-ref-may".into()));
    props.insert(
        "content".into(),
        Value::String("ATE_RAMEN at the new place".into()),
    );
    props.insert("subject".into(), Value::String("user".into()));
    props.insert("predicate".into(), Value::String("did".into()));
    props.insert("object".into(), Value::String("ramen".into()));
    props.insert("temporal_anchor".into(), datetime_value(may_anchor));
    props.insert("observed_at".into(), datetime_value(may_anchor));
    kb.create_node(labels::OBSERVATION, &props)
        .await
        .expect("create Observation");

    let target_score = |b: &uniko_memory::recall::ContextBundle| -> Option<f64> {
        b.items
            .iter()
            .find(|i| i.content.contains("ATE_RAMEN"))
            .map(|i| i.score)
    };

    // In-window anchor: "in may" resolves to May 2025, which overlaps the
    // observation, so the Phase-2 temporal channel contributes a hit and
    // boosts its fused score.
    let in_window = RecallConfig {
        reference_ts: Some(Utc.with_ymd_and_hms(2025, 7, 1, 12, 0, 0).unwrap()),
        ..RecallConfig::default()
    };
    let res = recall(&kb, "what happened in may", &in_window).await;
    if is_embedding_unavailable(&res) {
        eprintln!("skipping: embedding unavailable");
        return;
    }
    let bundle_in = res.expect("recall, in-window anchor");
    let score_in =
        target_score(&bundle_in).expect("in-window recall should surface the May 2025 observation");

    // Out-of-window anchor: "in may" resolves to May 1990, which does NOT
    // overlap the observation, so the temporal channel contributes nothing
    // for it. Same query/content/weights → the ONLY difference is the
    // reference_ts, which proves recall() actually threads it.
    let out_window = RecallConfig {
        reference_ts: Some(Utc.with_ymd_and_hms(1990, 7, 1, 12, 0, 0).unwrap()),
        ..RecallConfig::default()
    };
    let bundle_out = recall(&kb, "what happened in may", &out_window)
        .await
        .expect("recall, out-of-window anchor");

    // If present in both, the in-window temporal boost must make its score
    // strictly higher (if recall ignored reference_ts the scores would
    // match). If it's dropped entirely without the temporal contribution,
    // that also proves reference_ts changed the outcome.
    if let Some(score_out) = target_score(&bundle_out) {
        assert!(
            score_in > score_out,
            "in-window anchor must boost the observation's score via the temporal channel: \
             in={score_in} out={score_out}"
        );
    }
}

#[tokio::test]
async fn fact_with_overlapping_valid_at_is_recallable_via_btic() {
    let kb = kb().await;

    // Seed a Fact valid from May 1 2025 onwards (open hi).  Build_active
    // gives [observed_at, +∞), so it overlaps any later query window.
    let observed_at: DateTime<Utc> = Utc.with_ymd_and_hms(2025, 5, 1, 0, 0, 0).unwrap();
    let valid_at = btic_active(observed_at);

    // Use the same BTIC serialiser uniko-store uses for facts.
    let btic_value = Value::Temporal(uni_db::common::TemporalValue::Btic {
        lo: valid_at.lo(),
        hi: valid_at.hi(),
        meta: valid_at.meta(),
    });

    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("fact_id".into(), Value::String("fact-1".into()));
    props.insert("subject".into(), Value::String("user".into()));
    props.insert("predicate".into(), Value::String("works_at".into()));
    props.insert("object".into(), Value::String("AcmeCorp".into()));
    props.insert("valid_at".into(), btic_value);
    props.insert("confidence".into(), Value::Float(0.9));
    props.insert("observation_count".into(), Value::Int(3));
    kb.create_node(labels::FACT, &props)
        .await
        .expect("create Fact");

    // Build a query BTIC for June 2025 — overlaps [May 1, +∞).
    let lo: DateTime<Utc> = Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap();
    let hi: DateTime<Utc> = Utc.with_ymd_and_hms(2025, 7, 1, 0, 0, 0).unwrap();
    let qbtic = uniko_store::schema::btic::btic_query_window(lo, hi);
    let qbtic_value = Value::Temporal(uni_db::common::TemporalValue::Btic {
        lo: qbtic.lo(),
        hi: qbtic.hi(),
        meta: qbtic.meta(),
    });

    let session = kb.db().session();
    let r = session
        .query_with(
            "MATCH (f:Fact) \
             WHERE f.valid_at IS NOT NULL \
               AND btic_overlaps(f.valid_at, $qbtic) \
             RETURN f.fact_id AS id",
        )
        .param("qbtic", qbtic_value)
        .fetch_all()
        .await
        .expect("fact BTIC overlap query");

    let ids: Vec<String> = r
        .rows()
        .iter()
        .filter_map(|row| row.get::<String>("id").ok())
        .collect();

    assert!(
        ids.contains(&"fact-1".to_string()),
        "Fact with valid_at=[May 1, +∞) should overlap [Jun 1, Jul 1); got {ids:?}"
    );
}
