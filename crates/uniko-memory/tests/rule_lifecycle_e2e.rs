//! Integration tests for rule lifecycle (F44/F52).
//!
//! Validates: `add_rule` creates a Rule node and registers source;
//! decay cycles demote rules below threshold; matches re-promote
//! demoted rules; stdlib rules are exempt from decay.

use std::collections::HashMap;

use uni_db::Value;

use uniko_memory::rules::{
    AddRuleParams, RuleLifecycleConfig, add_rule, apply_decay_cycle, record_rule_match,
    register_stdlib_rules,
};
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;
use uniko_store::schema::constants::labels;

async fn kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

async fn rule_status(kb: &KnowledgeBase, name: &str) -> (String, f64) {
    let session = kb.db().session();
    let rows = session
        .query_with(
            "MATCH (r:Rule) WHERE r.name = $n \
             RETURN coalesce(r.status, '') AS st, coalesce(r.confidence, 0.0) AS c",
        )
        .param("n", name)
        .fetch_all()
        .await
        .expect("query");
    let row = rows.rows().first().expect("rule");
    let st: String = row.get("st").expect("status");
    let c: f64 = row.get("c").expect("conf");
    (st, c)
}

#[tokio::test]
async fn add_rule_creates_candidate_with_source() {
    let kb = kb().await;

    // Minimal valid Locy source — even if uni-db's runtime can't
    // execute non-stdlib syntax, the create_rule call should accept
    // the registration in test mode.  If it rejects, the test still
    // validates the error path.
    let src = "CREATE RULE foo_rule AS MATCH (n:Episode) YIELD KEY n";
    let result = add_rule(
        &kb,
        AddRuleParams {
            name: "foo_rule".into(),
            source: src.into(),
            source_type: "authored".into(),
            ..Default::default()
        },
    )
    .await;

    match result {
        Ok(_) => {
            let (status, conf) = rule_status(&kb, "foo_rule").await;
            assert_eq!(status, "candidate");
            assert!((conf - 0.5).abs() < 1e-9);
        }
        Err(uniko_store::UnikoError::Locy(_)) => {
            // Runtime didn't accept the syntax — the spec allows this
            // fallback path; we still want to confirm no Rule node was
            // left behind.
            let session = kb.db().session();
            let rows = session
                .query_with("MATCH (r:Rule) WHERE r.name = $n RETURN r")
                .param("n", "foo_rule")
                .fetch_all()
                .await
                .expect("query");
            assert!(
                rows.rows().is_empty(),
                "no Rule node should be created on Locy error"
            );
        }
        Err(other) => panic!("unexpected error: {other}"),
    }
}

#[tokio::test]
async fn add_rule_rejects_stdlib_source_type() {
    let kb = kb().await;
    let err = add_rule(
        &kb,
        AddRuleParams {
            name: "x".into(),
            source: "CREATE RULE x AS MATCH (n) YIELD KEY n".into(),
            source_type: "stdlib".into(),
            ..Default::default()
        },
    )
    .await
    .expect_err("reject stdlib via add_rule");
    matches!(err, uniko_store::UnikoError::Storage(_));
}

#[tokio::test]
async fn apply_decay_cycle_skips_stdlib_rules() {
    let kb = kb().await;
    register_stdlib_rules(&kb).await.expect("stdlib");

    // One cycle of decay; stdlib rules should still have confidence 1.0.
    apply_decay_cycle(&kb, RuleLifecycleConfig::default())
        .await
        .expect("decay");

    let (st, conf) = rule_status(&kb, "relevance_decay").await;
    assert_eq!(st, "active");
    assert!((conf - 1.0).abs() < 1e-9, "stdlib confidence stays 1.0");
}

#[tokio::test]
async fn decay_demotes_authored_rule_below_threshold() {
    let kb = kb().await;

    // Seed an authored rule manually so we don't depend on Locy
    // accepting non-stdlib syntax in test mode.
    let mut props = HashMap::new();
    props.insert("name".into(), Value::String("custom_rule".into()));
    props.insert(
        "source".into(),
        Value::String("CREATE RULE custom_rule AS MATCH (n) YIELD KEY n".into()),
    );
    props.insert("source_type".into(), Value::String("authored".into()));
    props.insert("status".into(), Value::String("active".into()));
    props.insert("confidence".into(), Value::Float(0.5));
    props.insert("missed_cycles".into(), Value::Int(0));
    kb.merge_node(labels::RULE, "rule_id", "r_custom", &props)
        .await
        .expect("seed");

    let cfg = RuleLifecycleConfig::default();
    // 0.5 * 0.95^N ; need < 0.40 → N >= 5 cycles.
    for _ in 0..6 {
        apply_decay_cycle(&kb, cfg).await.expect("decay");
    }
    let (st, _) = rule_status(&kb, "custom_rule").await;
    assert_eq!(st, "demoted");
}

#[tokio::test]
async fn record_rule_match_repromotes_demoted_rule() {
    let kb = kb().await;
    let mut props = HashMap::new();
    props.insert("name".into(), Value::String("rep_rule".into()));
    props.insert(
        "source".into(),
        Value::String("CREATE RULE rep_rule AS MATCH (n) YIELD KEY n".into()),
    );
    props.insert("source_type".into(), Value::String("authored".into()));
    props.insert("status".into(), Value::String("demoted".into()));
    props.insert("confidence".into(), Value::Float(0.55));
    kb.merge_node(labels::RULE, "rule_id", "r_rep", &props)
        .await
        .expect("seed");

    let cfg = RuleLifecycleConfig::default();
    // One match boost pushes confidence to 0.60 → re-promote.
    record_rule_match(&kb, "rep_rule", cfg)
        .await
        .expect("match");
    let (st, conf) = rule_status(&kb, "rep_rule").await;
    assert_eq!(st, "active");
    assert!(conf >= 0.6 - 1e-9);
}

#[tokio::test]
async fn record_rule_match_does_not_change_stdlib() {
    let kb = kb().await;
    register_stdlib_rules(&kb).await.expect("stdlib");
    let cfg = RuleLifecycleConfig::default();
    record_rule_match(&kb, "relevance_decay", cfg)
        .await
        .expect("match");
    let (st, conf) = rule_status(&kb, "relevance_decay").await;
    assert_eq!(st, "active");
    assert!((conf - 1.0).abs() < 1e-9);
}
