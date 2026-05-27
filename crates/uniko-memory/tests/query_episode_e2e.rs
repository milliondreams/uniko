//! Integration tests for `record_query_episode` and `answer_query`.
//!
//! These cover the surface that uniko-bench (and, later, any
//! production API layer) uses to record query-time Episodes so cortex
//! P5 procedure promotion sees real traffic.
//!
//! Embedding is required to land the topic vector; tests follow the
//! existing `episode_e2e.rs` pattern and treat
//! [`UnikoError::Embedding`] as "skip embedding-dependent assertions"
//! so they remain runnable in environments without the xervo
//! embedding model warm.
use std::collections::HashMap;

use serde_json::json;
use uni_db::{Node, Value};

use uniko_memory::recall::RecallConfig;
use uniko_memory::{
    GeneratedAnswer, QueryRecordOptions, RecordQueryEpisodeParams, answer_query,
    record_query_episode,
};
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

fn is_embedding_unavailable(e: &UnikoError) -> bool {
    matches!(e, UnikoError::Embedding(_))
}

/// Fetch the whole Episode node so callers can inspect both top-level
/// properties (`action_type`, `outcome`, `importance`) and the nested
/// `state` Map without paying two roundtrips.
async fn fetch_episode(kb: &KnowledgeBase, episode_id: i64) -> Node {
    let session = kb.db().session();
    let rows = session
        .query_with("MATCH (e:Episode) WHERE id(e) = $vid RETURN e")
        .param("vid", episode_id)
        .fetch_all()
        .await
        .expect("fetch episode");
    let row = rows.rows().first().expect("episode row");
    // Annotation is load-bearing — `row.get` needs the concrete type
    // to pick the right `FromValue` impl (Value itself doesn't have
    // one).  Mirrors the pattern in `episode_e2e.rs`.
    let node: Node = row.get("e").expect("episode node");
    node
}

/// Pull the `state` property out as a `HashMap<String, Value>`,
/// panicking if it's missing or shaped unexpectedly. `state` is
/// always a Map because `record_query_episode` builds it from a JSON
/// object via `json_to_value`.
fn state_map(node: &Node) -> &HashMap<String, Value> {
    match node.properties.get("state") {
        Some(Value::Map(m)) => m,
        other => panic!("expected Map state, got {other:?}"),
    }
}

/// Convenience: pull a String property out of the state Map.
fn state_str<'a>(state: &'a HashMap<String, Value>, key: &str) -> &'a str {
    match state.get(key) {
        Some(Value::String(s)) => s.as_str(),
        other => panic!("expected String at state.{key}, got {other:?}"),
    }
}

#[tokio::test]
async fn record_query_episode_persists_recall_metadata_in_state() {
    let kb = kb().await;
    seed_participant(&kb, "bench-agent").await;

    let recall_ids: Vec<i64> = vec![101, 202, 303];
    let params = RecordQueryEpisodeParams {
        question: "what role did Alice take in 2024?",
        answer: "Alice became the team's tech lead in March 2024.",
        recall_node_ids: &recall_ids,
        recall_coverage: 0.82,
        recall_tokens: 4096,
        answer_input_tokens: Some(1200),
        answer_output_tokens: Some(48),
        answer_model: Some("gpt-4o-mini"),
        importance: Some(0.7),
        outcome: Some("success"),
        action_type: Some("retrieve"),
        extra_state: None,
    };

    let result = record_query_episode(&kb, "bench-agent", params).await;
    if let Err(ref e) = result
        && is_embedding_unavailable(e)
    {
        eprintln!("skipping: embedding unavailable in this env");
        return;
    }
    let episode_id = result.expect("record_query_episode");

    let node = fetch_episode(&kb, episode_id).await;
    let state = state_map(&node);

    // Question / answer round-trip.
    assert_eq!(
        state_str(state, "question"),
        "what role did Alice take in 2024?"
    );
    assert_eq!(
        state_str(state, "topic"),
        "what role did Alice take in 2024?",
        "topic must mirror question so embed_episode picks it up"
    );
    assert_eq!(
        state_str(state, "answer"),
        "Alice became the team's tech lead in March 2024."
    );

    // Recall metadata.
    match state.get("recall_node_ids") {
        Some(Value::List(items)) => {
            let landed: Vec<i64> = items
                .iter()
                .map(|v| match v {
                    Value::Int(i) => *i,
                    other => panic!("expected Int recall id, got {other:?}"),
                })
                .collect();
            assert_eq!(landed, vec![101, 202, 303]);
        }
        other => panic!("expected List recall_node_ids, got {other:?}"),
    }
    assert!(matches!(state.get("recall_count"), Some(Value::Int(3))));
    match state.get("recall_coverage") {
        Some(Value::Float(f)) => assert!((f - 0.82).abs() < 1e-9),
        other => panic!("expected Float recall_coverage, got {other:?}"),
    }
    assert!(matches!(state.get("recall_tokens"), Some(Value::Int(4096))));

    // Token usage + model id.
    assert!(matches!(
        state.get("answer_input_tokens"),
        Some(Value::Int(1200))
    ));
    assert!(matches!(
        state.get("answer_output_tokens"),
        Some(Value::Int(48))
    ));
    assert_eq!(state_str(state, "answer_model"), "gpt-4o-mini");

    // Episode-level outcome / action_type / importance persisted as
    // top-level node properties (not inside `state`).
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
}

#[tokio::test]
async fn record_query_episode_merges_extra_state_without_overwriting_builtins() {
    let kb = kb().await;
    seed_participant(&kb, "bench-agent").await;

    let recall_ids: Vec<i64> = vec![];
    // extra_state tries to overwrite `question` (built-in, should win)
    // and adds a fresh `sample_id` key (should land).
    let extra = json!({
        "question": "OVERWRITTEN — should not land",
        "sample_id": "conv-26",
        "question_index": 7,
        "category": "Temporal",
    });

    let params = RecordQueryEpisodeParams {
        question: "the real question",
        answer: "the real answer",
        recall_node_ids: &recall_ids,
        recall_coverage: 0.0,
        recall_tokens: 0,
        answer_input_tokens: None,
        answer_output_tokens: None,
        answer_model: None,
        importance: None,
        outcome: None,
        action_type: None,
        extra_state: Some(extra),
    };

    let result = record_query_episode(&kb, "bench-agent", params).await;
    if let Err(ref e) = result
        && is_embedding_unavailable(e)
    {
        return;
    }
    let episode_id = result.expect("record_query_episode");
    let node = fetch_episode(&kb, episode_id).await;
    let state = state_map(&node);

    assert_eq!(
        state_str(state, "question"),
        "the real question",
        "built-in `question` must not be overwritten by extra_state"
    );
    assert_eq!(state_str(state, "sample_id"), "conv-26");
    assert!(matches!(state.get("question_index"), Some(Value::Int(7))));
    assert_eq!(state_str(state, "category"), "Temporal");

    // Optional token fields were `None` — must NOT appear in state
    // (so downstream consumers can use `.contains_key` as a presence
    // signal rather than checking for null).
    assert!(!state.contains_key("answer_input_tokens"));
    assert!(!state.contains_key("answer_output_tokens"));
    assert!(!state.contains_key("answer_model"));
}

#[tokio::test]
async fn record_query_episode_rejects_missing_participant() {
    let kb = kb().await;
    let recall_ids: Vec<i64> = vec![1];
    let params = RecordQueryEpisodeParams {
        question: "q",
        answer: "a",
        recall_node_ids: &recall_ids,
        recall_coverage: 0.0,
        recall_tokens: 0,
        answer_input_tokens: None,
        answer_output_tokens: None,
        answer_model: None,
        importance: None,
        outcome: None,
        action_type: None,
        extra_state: None,
    };
    let err = record_query_episode(&kb, "no-such-agent", params)
        .await
        .expect_err("must reject missing Participant");
    let msg = format!("{err}");
    assert!(
        msg.contains("Participant 'no-such-agent' not found"),
        "unexpected error: {msg}"
    );
}

#[tokio::test]
async fn record_query_episode_handles_nan_coverage() {
    // Defensive: a NaN coverage must not poison the JSON (NaN is not
    // representable as serde_json::Number; the helper clamps to 0.0).
    let kb = kb().await;
    seed_participant(&kb, "bench-agent").await;
    let recall_ids: Vec<i64> = vec![];
    let params = RecordQueryEpisodeParams {
        question: "q",
        answer: "a",
        recall_node_ids: &recall_ids,
        recall_coverage: f64::NAN,
        recall_tokens: 0,
        answer_input_tokens: None,
        answer_output_tokens: None,
        answer_model: None,
        importance: None,
        outcome: None,
        action_type: None,
        extra_state: None,
    };
    let result = record_query_episode(&kb, "bench-agent", params).await;
    if let Err(ref e) = result
        && is_embedding_unavailable(e)
    {
        return;
    }
    let episode_id = result.expect("record_query_episode");
    let node = fetch_episode(&kb, episode_id).await;
    let state = state_map(&node);
    match state.get("recall_coverage") {
        Some(Value::Float(f)) => {
            assert!(f.abs() < 1e-9, "NaN must be clamped to 0.0, got {f}");
        }
        other => panic!("expected Float (clamped 0.0) coverage, got {other:?}"),
    }
}

#[tokio::test]
async fn answer_query_skips_recording_when_record_is_none() {
    // Verifies opt-in semantics: `record = None` produces no Episode.
    // Uses an empty-string question so the recall fast-path returns
    // an empty bundle without needing a populated KB.
    let kb = kb().await;

    let outcome = answer_query(
        &kb,
        "",
        &RecallConfig::from_uniko_config(&UnikoConfig::default()),
        |_bundle, _q| async move {
            Ok(GeneratedAnswer {
                text: "synthetic answer".into(),
                input_tokens: Some(10),
                output_tokens: Some(5),
                model: Some("test-model".into()),
            })
        },
        None,
    )
    .await
    .expect("answer_query");

    assert!(
        outcome.episode_id.is_none(),
        "record=None must NOT produce an Episode"
    );
    assert_eq!(outcome.answer.text, "synthetic answer");
    assert!(
        outcome.bundle.items.is_empty(),
        "empty query short-circuits to an empty bundle"
    );
}

#[tokio::test]
async fn answer_query_records_episode_when_opted_in() {
    let kb = kb().await;
    seed_participant(&kb, "bench-agent").await;

    let outcome = answer_query(
        &kb,
        "",
        &RecallConfig::from_uniko_config(&UnikoConfig::default()),
        |_bundle, _q| async move {
            Ok(GeneratedAnswer {
                text: "synthetic answer".into(),
                input_tokens: Some(10),
                output_tokens: Some(5),
                model: Some("test-model".into()),
            })
        },
        Some(QueryRecordOptions {
            participant_id: "bench-agent".into(),
            outcome: Some("success".into()),
            importance: Some(1.0),
            ..Default::default()
        }),
    )
    .await
    .expect("answer_query");

    // Recording is best-effort — when embedding is unavailable the
    // helper returns None rather than erroring.  Only check the
    // recorded path when an Episode was successfully created.
    assert_eq!(outcome.answer.text, "synthetic answer");
    if let Some(episode_id) = outcome.episode_id {
        let node = fetch_episode(&kb, episode_id).await;
        let state = state_map(&node);
        assert_eq!(state_str(state, "answer"), "synthetic answer");
        assert_eq!(state_str(state, "answer_model"), "test-model");
    }
}
