//! Integration tests for the P3 observation extraction pipeline.

use std::collections::HashMap;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use uniko_extract::observations::ObservationExtractionStep;
use uniko_pipes::circuit_breaker::CircuitBreaker;
use uniko_pipes::step::PipelineContext;
use uniko_pipes::types::StepOutcome;
use uniko_store::config::UnikoConfig;
use uniko_store::storage::edges::Direction;
use uniko_store::storage::KnowledgeBase;

async fn test_kb() -> Arc<KnowledgeBase> {
    Arc::new(
        KnowledgeBase::in_memory(UnikoConfig::default())
            .await
            .expect("in-memory KB"),
    )
}

fn make_ctx(
    kb: Arc<KnowledgeBase>,
    node_id: i64,
    content: &str,
) -> PipelineContext {
    PipelineContext::new(
        node_id,
        content.to_string(),
        "text".to_string(),
        CancellationToken::new(),
        kb,
        Arc::new(CircuitBreaker::new(5, 60_000)),
    )
}

/// Create a Message node and return its NodeId.
async fn create_message(kb: &KnowledgeBase, content: &str) -> i64 {
    let mut props = HashMap::new();
    props.insert(
        "message_id".into(),
        uni_db::Value::String(uniko_store::id::new_id()),
    );
    props.insert("content".into(), uni_db::Value::String(content.into()));
    props.insert("content_type".into(), uni_db::Value::String("text".into()));
    props.insert(
        "timestamp".into(),
        uni_db::Value::String(chrono::Utc::now().to_rfc3339()),
    );
    kb.create_node("Message", &props).await.unwrap()
}

/// Create an Entity node and return its NodeId.
async fn create_entity(kb: &KnowledgeBase, name: &str, entity_type: &str) -> i64 {
    let mut props = HashMap::new();
    props.insert(
        "entity_id".into(),
        uni_db::Value::String(format!("{entity_type}:{name}")),
    );
    props.insert("name".into(), uni_db::Value::String(name.into()));
    props.insert(
        "entity_type".into(),
        uni_db::Value::String(entity_type.into()),
    );
    props.insert("frequency".into(), uni_db::Value::Int(1));
    props.insert("confidence".into(), uni_db::Value::Float(0.9));
    kb.create_node("Entity", &props).await.unwrap()
}

// ── Tests ───────────────────────────────────────────────────────────

#[tokio::test]
async fn test_observation_from_prose() {
    let kb = test_kb().await;
    let text = "Caroline attended an LGBTQ support group yesterday.";
    let msg_nid = create_message(&kb, text).await;
    let entity_nid = create_entity(&kb, "Caroline", "person").await;

    let step = ObservationExtractionStep;
    let mut ctx = make_ctx(kb.clone(), msg_nid, text);
    ctx.extracted_entities = vec![entity_nid];

    let outcome = uniko_pipes::Step::execute(&step, &mut ctx).await.unwrap();
    assert!(matches!(outcome, StepOutcome::Completed));
    assert!(
        !ctx.extracted_observations.is_empty(),
        "should extract at least one observation"
    );

    // Verify Observation node exists.
    let obs_nid = ctx.extracted_observations[0];
    let (label, props) = kb.get_node(obs_nid).await.unwrap().expect("obs node");
    assert_eq!(label, "Observation");
    assert!(props.get("content").and_then(|v| v.as_str()).is_some());
    assert!(props.get("subject").and_then(|v| v.as_str()).is_some());

    // Verify OBSERVED_IN edge (Observation → Message).
    let edges = kb
        .get_edges(obs_nid, "OBSERVED_IN", Direction::Outgoing)
        .await
        .unwrap();
    assert!(!edges.is_empty(), "OBSERVED_IN edge should exist");
    assert_eq!(edges[0].to, msg_nid);

    // Verify ABOUT edge (Observation → Entity).
    let about_edges = kb
        .get_edges(obs_nid, "ABOUT", Direction::Outgoing)
        .await
        .unwrap();
    assert!(!about_edges.is_empty(), "ABOUT edge should exist");
}

#[tokio::test]
async fn test_observation_filtering_greetings() {
    let kb = test_kb().await;
    let msg_nid = create_message(&kb, "Hey!").await;
    let entity_nid = create_entity(&kb, "Alice", "person").await;

    let step = ObservationExtractionStep;
    let mut ctx = make_ctx(kb.clone(), msg_nid, "Hey!");
    ctx.extracted_entities = vec![entity_nid];

    let outcome = uniko_pipes::Step::execute(&step, &mut ctx).await.unwrap();
    assert!(
        matches!(outcome, StepOutcome::Skipped { .. }),
        "greetings should be skipped"
    );
    assert!(ctx.extracted_observations.is_empty());
}

#[tokio::test]
async fn test_observation_filtering_pure_question() {
    let kb = test_kb().await;
    let msg_nid = create_message(&kb, "Where is the store?").await;
    let entity_nid = create_entity(&kb, "store", "place").await;

    let step = ObservationExtractionStep;
    let mut ctx = make_ctx(kb.clone(), msg_nid, "Where is the store?");
    ctx.extracted_entities = vec![entity_nid];

    let outcome = uniko_pipes::Step::execute(&step, &mut ctx).await.unwrap();
    assert!(matches!(outcome, StepOutcome::Skipped { .. }));
}

#[tokio::test]
async fn test_observation_no_entities_skips() {
    let kb = test_kb().await;
    let msg_nid = create_message(&kb, "Caroline went to the store yesterday.").await;

    let step = ObservationExtractionStep;
    let mut ctx = make_ctx(kb.clone(), msg_nid, "Caroline went to the store yesterday.");
    // No entities from P2.
    ctx.extracted_entities = vec![];

    let outcome = uniko_pipes::Step::execute(&step, &mut ctx).await.unwrap();
    assert!(matches!(outcome, StepOutcome::Skipped { .. }));
}

#[tokio::test]
async fn test_observation_step_populates_context() {
    let kb = test_kb().await;
    let text = "Alice works at the hospital and lives in New York.";
    let msg_nid = create_message(&kb, text).await;
    let alice_nid = create_entity(&kb, "Alice", "person").await;

    let step = ObservationExtractionStep;
    let mut ctx = make_ctx(kb.clone(), msg_nid, text);
    ctx.extracted_entities = vec![alice_nid];

    uniko_pipes::Step::execute(&step, &mut ctx).await.unwrap();

    // extracted_observations should be populated.
    assert!(
        !ctx.extracted_observations.is_empty(),
        "extracted_observations should be populated"
    );

    // Each should be a valid Observation node.
    for &oid in &ctx.extracted_observations {
        let node = kb.get_node(oid).await.unwrap();
        assert!(node.is_some());
        let (label, _) = node.unwrap();
        assert_eq!(label, "Observation");
    }
}

#[tokio::test]
async fn test_observation_node_id_zero_skips() {
    let kb = test_kb().await;
    let step = ObservationExtractionStep;
    let ctx = make_ctx(kb, 0, "Some content.");
    assert!(!uniko_pipes::Step::should_run(&step, &ctx));
}
