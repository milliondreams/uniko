//! E2E test for the consolidation worker's cortex sweep gate.
//!
//! Verifies that the worker invokes P5 (procedure promotion) and P6
//! (topic detection) after a successful consolidation cycle when the
//! cadence gates allow, and that the per-agent counter gate works.

// Rust guideline compliant

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use uni_db::Value;

use uniko_memory::PipelineSystem;
use uniko_pipes::config::PipelineConfig;
use uniko_pipes::step::Step;
use uniko_pipes::types::ConsolidationTask;
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;
use uniko_store::schema::constants::{edges, labels};
use uniko_store::types::datetime_value;

async fn kb_arc() -> Arc<KnowledgeBase> {
    Arc::new(
        KnowledgeBase::in_memory(UnikoConfig::default())
            .await
            .expect("in-memory KB"),
    )
}

async fn mk_message(kb: &KnowledgeBase, mid: &str) -> i64 {
    let mut props = HashMap::new();
    props.insert("content".into(), Value::String("msg".into()));
    props.insert("timestamp".into(), datetime_value(Utc::now()));
    kb.merge_node(labels::MESSAGE, "message_id", mid, &props)
        .await
        .expect("message")
}

async fn mk_entity(kb: &KnowledgeBase, eid: &str, name: &str) -> i64 {
    let mut props = HashMap::new();
    props.insert("name".into(), Value::String(name.into()));
    props.insert("frequency".into(), Value::Int(1));
    kb.merge_node(labels::ENTITY, "entity_id", eid, &props)
        .await
        .expect("entity")
}

async fn mention(kb: &KnowledgeBase, src: i64, ent: i64) {
    let mut props = HashMap::new();
    props.insert("count".into(), Value::Int(1));
    kb.create_edge(edges::MENTIONS, src, ent, &props)
        .await
        .expect("MENTIONS");
}

async fn seed_cooccurrence(kb: &KnowledgeBase) {
    // One cluster of three co-mentioned entities — enough for LPA to
    // produce one community ≥ min_community_size = 2.
    let m1 = mk_message(kb, "m1").await;
    let m2 = mk_message(kb, "m2").await;
    let e1 = mk_entity(kb, "e1", "alpha").await;
    let e2 = mk_entity(kb, "e2", "beta").await;
    let e3 = mk_entity(kb, "e3", "gamma").await;
    for src in [m1, m2] {
        mention(kb, src, e1).await;
        mention(kb, src, e2).await;
        mention(kb, src, e3).await;
    }
}

async fn topic_count(kb: &KnowledgeBase) -> i64 {
    let result = kb
        .db()
        .session()
        .query_with("MATCH (t:Topic) RETURN count(t) AS c")
        .fetch_all()
        .await
        .expect("count");
    result
        .rows()
        .first()
        .and_then(|row| row.get("c").ok())
        .unwrap_or(0)
}

#[tokio::test]
async fn cortex_sweep_fires_after_consolidation_when_gate_open() {
    let kb = kb_arc().await;
    seed_cooccurrence(&kb).await;

    // Cadence: run cortex after every successful consolidation cycle,
    // no wall-clock throttle.
    let mut config = PipelineConfig::default();
    config.cortex_cycle_every_n_consolidations = 1;
    config.cortex_min_interval_secs = 0;

    let ps = PipelineSystem::new(config, kb.clone(), Vec::<Box<dyn Step>>::new());

    assert_eq!(topic_count(&kb).await, 0, "no topics before cycle");

    ps.submit_consolidation(ConsolidationTask::RunCycle {
        agent_id: "agent-1".into(),
    })
    .expect("submit");

    // The worker is async; poll briefly until Topics appear or we
    // exceed the budget.
    let mut count = 0;
    for _ in 0..30 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        count = topic_count(&kb).await;
        if count > 0 {
            break;
        }
    }
    assert!(
        count > 0,
        "expected Topic nodes after cortex sweep, found {count}",
    );

    ps.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn cortex_sweep_skipped_until_counter_threshold() {
    let kb = kb_arc().await;
    seed_cooccurrence(&kb).await;

    // Require two consolidation cycles before cortex fires.
    let mut config = PipelineConfig::default();
    config.cortex_cycle_every_n_consolidations = 2;
    config.cortex_min_interval_secs = 0;

    let ps = PipelineSystem::new(config, kb.clone(), Vec::<Box<dyn Step>>::new());

    // First cycle — counter goes to 1, cortex must NOT run.
    ps.submit_consolidation(ConsolidationTask::RunCycle {
        agent_id: "agent-1".into(),
    })
    .expect("submit");
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(
        topic_count(&kb).await,
        0,
        "cortex should not run after one cycle when threshold is 2",
    );

    // Second cycle — counter resets and cortex fires.
    ps.submit_consolidation(ConsolidationTask::RunCycle {
        agent_id: "agent-1".into(),
    })
    .expect("submit");
    let mut count = 0;
    for _ in 0..30 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        count = topic_count(&kb).await;
        if count > 0 {
            break;
        }
    }
    assert!(
        count > 0,
        "expected Topic nodes after second cycle, found {count}",
    );

    ps.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn cortex_sweep_disabled_when_every_n_is_zero() {
    let kb = kb_arc().await;
    seed_cooccurrence(&kb).await;

    let mut config = PipelineConfig::default();
    config.cortex_cycle_every_n_consolidations = 0;
    config.cortex_min_interval_secs = 0;

    let ps = PipelineSystem::new(config, kb.clone(), Vec::<Box<dyn Step>>::new());

    ps.submit_consolidation(ConsolidationTask::RunCycle {
        agent_id: "agent-1".into(),
    })
    .expect("submit");
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(
        topic_count(&kb).await,
        0,
        "cortex must stay off when cortex_cycle_every_n_consolidations = 0",
    );

    ps.shutdown().await.expect("shutdown");
}
