//! Integration tests for stdlib Locy rule execution (the migration from
//! aspirational rules to consumer-backed rules).
//!
//! Each test seeds a small graph, runs the rule consumer, and asserts the
//! side effect: episodes pruned, a `:Pattern` upserted, a Fact invalidated +
//! `CONTRADICTED_BY` edges drawn, and an authored candidate rule promoted on
//! match (the lifecycle is now bidirectional).

use std::collections::HashMap;

use chrono::{Duration, Utc};
use serde_json::json;
use uni_db::Value;

use uniko_memory::rules::{
    AddRuleParams, RuleLifecycleConfig, add_rule, consume_relevance_decay, register_stdlib_rules,
    run_active_rules,
};
use uniko_memory::{AssertFactParams, RecordEpisodeParams, assert_fact, record_episode};
use uniko_store::config::UnikoConfig;
use uniko_store::schema::constants::labels;
use uniko_store::KnowledgeBase;

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

async fn count(kb: &KnowledgeBase, cypher: &str) -> i64 {
    let session = kb.db().session();
    let result = session.query_with(cypher).fetch_all().await.expect("query");
    result
        .rows()
        .first()
        .and_then(|r| r.get::<i64>("c").ok())
        .unwrap_or(0)
}

// relevance_decay is a registered Locy rule; its consumer
// `consume_relevance_decay` runs the rule (decay via
// `duration.inDays(...).days`) and deletes the yielded episodes.
#[tokio::test]
async fn relevance_decay_prunes_decayed_episodes() {
    let kb = kb().await;
    register_stdlib_rules(&kb).await.expect("stdlib");
    seed_participant(&kb, "a1").await;

    // Stale, low-importance episode: importance 0.1, ~400 days old. With a
    // 30-day half-life the decayed relevance is ~2e-5 — well below 0.05.
    record_episode(
        &kb,
        "a1",
        RecordEpisodeParams {
            action_type: "old".into(),
            outcome: Some("success".into()),
            importance: Some(0.1),
            timestamp: Some(Utc::now() - Duration::days(400)),
            state: Some(json!({"topic": "old"})),
            ..Default::default()
        },
    )
    .await
    .expect("record old episode");

    // Fresh, important episode that must survive (decayed ≈ 1.0).
    record_episode(
        &kb,
        "a1",
        RecordEpisodeParams {
            action_type: "fresh".into(),
            outcome: Some("success".into()),
            importance: Some(1.0),
            timestamp: Some(Utc::now()),
            state: Some(json!({"topic": "fresh"})),
            ..Default::default()
        },
    )
    .await
    .expect("record fresh episode");

    assert_eq!(count(&kb, "MATCH (e:Episode) RETURN count(e) AS c").await, 2);

    let pruned = consume_relevance_decay(&kb, "a1", 30.0, 0.05)
        .await
        .expect("decay consumer");
    assert_eq!(pruned, 1, "exactly the stale episode should be pruned");
    assert_eq!(
        count(&kb, "MATCH (e:Episode) RETURN count(e) AS c").await,
        1,
        "the fresh episode survives"
    );
}

#[tokio::test]
async fn episode_pattern_detector_upserts_pattern() {
    let kb = kb().await;
    register_stdlib_rules(&kb).await.expect("stdlib");
    seed_participant(&kb, "a2").await;

    // 3 episodes of the same (action_type, outcome) → crosses the rule's
    // n >= 3 and avg_importance > 0.3 thresholds.
    for i in 0..3 {
        record_episode(
            &kb,
            "a2",
            RecordEpisodeParams {
                action_type: "build".into(),
                outcome: Some("success".into()),
                importance: Some(0.6),
                timestamp: Some(Utc::now() - Duration::minutes(i)),
                state: Some(json!({"topic": "build"})),
                ..Default::default()
            },
        )
        .await
        .expect("record build episode");
    }

    let report = run_active_rules(&kb, "a2", RuleLifecycleConfig::default())
        .await
        .expect("run rules");
    assert!(report.effects >= 1, "a pattern should be upserted");

    let session = kb.db().session();
    let rows = session
        .query_with(
            "MATCH (p:Pattern) WHERE p.action_type = 'build' AND p.outcome = 'success' \
             RETURN p.support AS support",
        )
        .fetch_all()
        .await
        .expect("query pattern");
    let support: i64 = rows
        .rows()
        .first()
        .expect("a :Pattern node exists")
        .get("support")
        .expect("support");
    assert!(support >= 3, "support should be at least 3, got {support}");
}

#[tokio::test]
async fn contradiction_detector_invalidates_fact_and_marks_edges() {
    let kb = kb().await;
    register_stdlib_rules(&kb).await.expect("stdlib");
    seed_participant(&kb, "a3").await;

    // A currently-valid outcome_pattern Fact: deploy ⇒ success.
    assert_fact(
        &kb,
        AssertFactParams {
            subject: "deploy".into(),
            predicate: "outcome_pattern".into(),
            object: Some("success".into()),
            ..Default::default()
        },
    )
    .await
    .expect("assert fact");

    // 2 episodes where deploy actually failed — contradicting the fact.
    for i in 0..2 {
        record_episode(
            &kb,
            "a3",
            RecordEpisodeParams {
                action_type: "deploy".into(),
                outcome: Some("failure".into()),
                importance: Some(0.5),
                timestamp: Some(Utc::now() - Duration::minutes(i)),
                state: Some(json!({"topic": "deploy"})),
                ..Default::default()
            },
        )
        .await
        .expect("record deploy episode");
    }

    let report = run_active_rules(&kb, "a3", RuleLifecycleConfig::default())
        .await
        .expect("run rules");
    assert!(report.effects >= 1, "one fact should be invalidated");

    // Two CONTRADICTED_BY marker edges (one per contradicting episode).
    assert_eq!(
        count(
            &kb,
            "MATCH (:Fact)-[r:CONTRADICTED_BY]->(:Episode) RETURN count(r) AS c"
        )
        .await,
        2,
        "a marker edge per contradicting episode"
    );

    // The fact's validity interval was closed (invalidated_at stamped).
    let session = kb.db().session();
    let rows = session
        .query_with("MATCH (f:Fact) WHERE f.subject = 'deploy' RETURN f.invalidated_at AS ia")
        .fetch_all()
        .await
        .expect("query fact");
    let ia: Option<String> = rows.rows().first().and_then(|r| r.get("ia").ok());
    assert!(ia.is_some(), "the fact should be invalidated");
}

#[tokio::test]
async fn authored_candidate_rule_is_executed_and_rewarded() {
    let kb = kb().await;
    seed_participant(&kb, "a4").await;

    // An authored rule that matches the seeded participant. It starts as a
    // candidate (confidence 0.5); a recorded match must boost it — proving the
    // lifecycle is bidirectional and that consumer-less user rules execute.
    add_rule(
        &kb,
        AddRuleParams {
            name: "agent_present".into(),
            source: "CREATE RULE agent_present AS \
                     MATCH (p:Participant {participant_id: $agent_id}) \
                     YIELD KEY p.participant_id"
                .into(),
            source_type: "authored".into(),
            ..Default::default()
        },
    )
    .await
    .expect("add_rule");

    let report = run_active_rules(&kb, "a4", RuleLifecycleConfig::default())
        .await
        .expect("run rules");
    assert!(report.rules_matched >= 1, "the authored rule should match");

    let session = kb.db().session();
    let rows = session
        .query_with(
            "MATCH (r:Rule) WHERE r.name = 'agent_present' \
             RETURN r.confidence AS conf, r.last_scored_at AS lsa",
        )
        .fetch_all()
        .await
        .expect("query rule");
    let row = rows.rows().first().expect("rule node");
    let conf: f64 = row.get("conf").expect("confidence");
    let lsa: Option<String> = row.get("lsa").ok();
    assert!(
        conf > 0.5,
        "confidence should be boosted above the 0.5 start, got {conf}"
    );
    assert!(lsa.is_some(), "last_scored_at should be stamped on a match");
}
