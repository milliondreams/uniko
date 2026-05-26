//! Integration tests for [`uniko_memory::working_memory`].
//!
//! Validates: Goal scope expansion via PARENT_GOAL traversal, per-tier
//! candidate fetch (Tasks/Sessions/Messages/Facts/Entities), token-
//! budget enforcement, and the empty-goal abstention path.

use std::collections::HashMap;

use chrono::Utc;
use uni_db::Value;

use uniko_memory::{WorkingMemoryParams, working_memory};
use uniko_store::config::UnikoConfig;
use uniko_store::schema::constants::{edges, labels};
use uniko_store::types::datetime_value;
use uniko_store::KnowledgeBase;

async fn kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

async fn mk_goal(kb: &KnowledgeBase, goal_id: &str, title: &str) -> i64 {
    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("title".into(), Value::String(title.into()));
    props.insert("status".into(), Value::String("active".into()));
    kb.merge_node(labels::GOAL, "goal_id", goal_id, &props)
        .await
        .expect("goal created")
}

async fn mk_task(kb: &KnowledgeBase, task_id: &str, title: &str, parent_goal: i64) -> i64 {
    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("title".into(), Value::String(title.into()));
    props.insert("status".into(), Value::String("active".into()));
    props.insert("priority".into(), Value::Float(0.7));
    let t = kb
        .merge_node(labels::TASK, "task_id", task_id, &props)
        .await
        .expect("task created");
    kb.create_edge(edges::PART_OF, t, parent_goal, &HashMap::new())
        .await
        .expect("PART_OF edge");
    t
}

async fn mk_session(kb: &KnowledgeBase, session_id: &str, for_goal: i64) -> i64 {
    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("topic".into(), Value::String("session".into()));
    props.insert("started_at".into(), datetime_value(Utc::now()));
    let s = kb
        .merge_node(labels::SESSION, "session_id", session_id, &props)
        .await
        .expect("session created");
    kb.create_edge(edges::FOR_GOAL, s, for_goal, &HashMap::new())
        .await
        .expect("FOR_GOAL edge");
    s
}

async fn mk_message(kb: &KnowledgeBase, msg_id: &str, content: &str, session: i64) -> i64 {
    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("content".into(), Value::String(content.into()));
    props.insert("timestamp".into(), datetime_value(Utc::now()));
    let m = kb
        .merge_node(labels::MESSAGE, "message_id", msg_id, &props)
        .await
        .expect("message created");
    kb.create_edge(edges::IN_SESSION, m, session, &HashMap::new())
        .await
        .expect("IN_SESSION edge");
    m
}

async fn mk_entity(kb: &KnowledgeBase, ent_id: &str, name: &str, mentioned_by: i64) -> i64 {
    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("name".into(), Value::String(name.into()));
    props.insert("frequency".into(), Value::Int(3));
    let e = kb
        .merge_node(labels::ENTITY, "entity_id", ent_id, &props)
        .await
        .expect("entity created");
    let mut edge_props = HashMap::new();
    edge_props.insert("count".into(), Value::Int(1));
    kb.create_edge(edges::MENTIONS, mentioned_by, e, &edge_props)
        .await
        .expect("MENTIONS edge");
    e
}

#[tokio::test]
async fn working_memory_returns_empty_for_unknown_goal() {
    let kb = kb().await;
    let bundle = working_memory(&kb, WorkingMemoryParams::new("nope"))
        .await
        .expect("traversal");
    assert!(bundle.items.is_empty());
    assert!((bundle.coverage - 0.0).abs() < 1e-9);
}

#[tokio::test]
async fn working_memory_collects_goal_task_session_message_entity() {
    let kb = kb().await;
    let g = mk_goal(&kb, "g1", "Reduce refund cycle time").await;
    let _t = mk_task(&kb, "t1", "Diagnose bottleneck", g).await;
    let s = mk_session(&kb, "s1", g).await;
    let m = mk_message(&kb, "m1", "the refund pipeline is slow", s).await;
    let _e = mk_entity(&kb, "ent_pipeline", "refund pipeline", m).await;

    let bundle = working_memory(&kb, WorkingMemoryParams::new("g1"))
        .await
        .expect("traversal");

    // We expect at least: 1 Goal, 1 Task, 1 Session, 1 Message, 1 Entity.
    let labels_present: std::collections::HashSet<_> =
        bundle.items.iter().map(|i| i.node_type.as_str()).collect();
    assert!(labels_present.contains(labels::GOAL), "Goal missing");
    assert!(labels_present.contains(labels::TASK), "Task missing");
    assert!(labels_present.contains(labels::SESSION), "Session missing");
    assert!(labels_present.contains(labels::MESSAGE), "Message missing");
    assert!(labels_present.contains(labels::ENTITY), "Entity missing");

    // Coverage is fraction of distinct tiers; 5 categories cover all
    // 5 tiers ≥ 4 (Goal/Fact map to Semantic).
    assert!(
        bundle.coverage >= 0.6,
        "coverage too low: {}",
        bundle.coverage
    );
}

#[tokio::test]
async fn working_memory_traverses_subgoals_when_enabled() {
    let kb = kb().await;
    let parent = mk_goal(&kb, "parent_goal", "Top-level").await;
    let child = mk_goal(&kb, "child_goal", "Sub-goal").await;
    kb.create_edge(edges::PARENT_GOAL, child, parent, &HashMap::new())
        .await
        .expect("PARENT_GOAL edge");
    // Sub-goal has its own session+message.
    let s = mk_session(&kb, "sub_session", child).await;
    let _m = mk_message(&kb, "sub_msg", "hello from sub", s).await;

    // With include_subgoals=true (default) we should see the child's
    // session and message via parent's working memory.
    let bundle = working_memory(&kb, WorkingMemoryParams::new("parent_goal"))
        .await
        .expect("traversal");
    let has_sub_msg = bundle
        .items
        .iter()
        .any(|i| i.node_type == labels::MESSAGE);
    assert!(has_sub_msg, "expected to see sub-goal's message");

    // With include_subgoals=false we should not.
    let mut p = WorkingMemoryParams::new("parent_goal");
    p.include_subgoals = false;
    let bundle = working_memory(&kb, p).await.expect("traversal");
    let has_sub_msg = bundle
        .items
        .iter()
        .any(|i| i.node_type == labels::MESSAGE);
    assert!(
        !has_sub_msg,
        "should NOT see sub-goal's message when subgoals disabled"
    );
}

#[tokio::test]
async fn working_memory_respects_budget() {
    let kb = kb().await;
    let g = mk_goal(&kb, "g_budget", "Budget test").await;
    let s = mk_session(&kb, "s_budget", g).await;
    // 30 messages so the per-tier query has more than fits in budget.
    for i in 0..30 {
        mk_message(&kb, &format!("m{i}"), &format!("msg {i}"), s).await;
    }

    // Budget = 100 tokens → at ~50 tokens/item, max 2 items.
    let mut p = WorkingMemoryParams::new("g_budget");
    p.budget = Some(100);
    let bundle = working_memory(&kb, p).await.expect("traversal");
    assert!(
        bundle.items.len() <= 2,
        "expected ≤2 items at 100-token budget, got {}",
        bundle.items.len()
    );
    assert!(bundle.total_tokens <= 100);
}
