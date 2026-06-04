//! Integration tests for Locy runtime: rule management and execution.

use std::collections::HashMap;
use uni_db::Value;
use uniko_store::config::UnikoConfig;
use uniko_store::storage::KnowledgeBase;

async fn test_kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

#[tokio::test]
async fn test_create_and_list_rules() {
    let kb = test_kb().await;

    // Use Cypher-based Locy CREATE RULE syntax.
    kb.create_rule(
        "CREATE RULE test_rule AS \
         MATCH (p:Participant) \
         WHERE p.kind = 'agent' \
         YIELD KEY p.participant_id AS pid",
    )
    .expect("rule registration");

    let rules = kb.list_rules();
    let names: Vec<&str> = rules.iter().map(|r| r.name.as_str()).collect();
    assert!(
        names.contains(&"test_rule"),
        "registered rule must appear in list: {names:?}"
    );

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_delete_rule() {
    let kb = test_kb().await;
    kb.create_rule(
        "CREATE RULE del_rule AS \
         MATCH (p:Participant) \
         YIELD KEY p.participant_id AS pid",
    )
    .unwrap();

    let removed = kb.delete_rule("del_rule").unwrap();
    assert!(removed);

    let rules = kb.list_rules();
    let names: Vec<&str> = rules.iter().map(|r| r.name.as_str()).collect();
    assert!(!names.contains(&"del_rule"));

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_execute_cypher_query() {
    let kb = test_kb().await;

    // Seed data.
    let mut props = HashMap::new();
    props.insert("participant_id".into(), Value::String("p-locy".into()));
    props.insert("kind".into(), Value::String("agent".into()));
    props.insert("name".into(), Value::String("LocyBot".into()));
    kb.create_node("Participant", &props).await.unwrap();

    // Execute a Cypher query directly via session (verifying the db is
    // accessible from KnowledgeBase for advanced usage).
    let result = kb
        .db()
        .session()
        .query("MATCH (p:Participant {kind: 'agent'}) RETURN p.name AS name")
        .await
        .unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result.rows()[0].get::<String>("name").unwrap(), "LocyBot");

    kb.shutdown().await.unwrap();
}
