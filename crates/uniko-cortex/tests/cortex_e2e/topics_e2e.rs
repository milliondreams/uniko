//! Integration tests for topic detection (P6, F40).

use std::collections::HashMap;

use chrono::Utc;
use uni_db::Value;

use uniko_cortex::topics::{TopicConfig, detect_topics_once};
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;
use uniko_store::schema::constants::{edges, labels};
use uniko_store::types::datetime_value;

async fn kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
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

#[tokio::test]
async fn detect_topics_returns_empty_on_empty_graph() {
    let kb = kb().await;
    let report = detect_topics_once(&kb, TopicConfig::default())
        .await
        .expect("detect");
    assert_eq!(report.created, 0);
    assert_eq!(report.entities_assigned, 0);
}

#[tokio::test]
async fn detect_topics_groups_co_occurring_entities() {
    let kb = kb().await;

    // Cluster A: entities (a1, a2, a3) all co-mentioned together.
    let m1 = mk_message(&kb, "m1").await;
    let m2 = mk_message(&kb, "m2").await;
    let m3 = mk_message(&kb, "m3").await;
    let a1 = mk_entity(&kb, "ea1", "apple").await;
    let a2 = mk_entity(&kb, "ea2", "banana").await;
    let a3 = mk_entity(&kb, "ea3", "cherry").await;
    for src in [m1, m2, m3] {
        mention(&kb, src, a1).await;
        mention(&kb, src, a2).await;
        mention(&kb, src, a3).await;
    }

    // Cluster B: (b1, b2) co-mentioned in their own messages.
    let m4 = mk_message(&kb, "m4").await;
    let m5 = mk_message(&kb, "m5").await;
    let b1 = mk_entity(&kb, "eb1", "rust").await;
    let b2 = mk_entity(&kb, "eb2", "go").await;
    for src in [m4, m5] {
        mention(&kb, src, b1).await;
        mention(&kb, src, b2).await;
    }

    let report = detect_topics_once(&kb, TopicConfig::default())
        .await
        .expect("detect");
    assert_eq!(report.created, 2, "expected 2 topics created");
    assert_eq!(report.entities_assigned, 5);

    // Topic membership: both clusters should be present.
    let session = kb.db().session();
    let rows = session
        .query_with(
            "MATCH (e:Entity)-[:BELONGS_TO]->(t:Topic) \
             RETURN e.entity_id AS eid, t.topic_id AS tid",
        )
        .fetch_all()
        .await
        .expect("query");
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();
    for row in rows.rows() {
        let eid: String = row.get("eid").expect("eid");
        let tid: String = row.get("tid").expect("tid");
        groups.entry(tid).or_default().push(eid);
    }
    let mut sizes: Vec<usize> = groups.values().map(Vec::len).collect();
    sizes.sort_unstable();
    assert_eq!(
        sizes,
        vec![2, 3],
        "expected one 2-cluster and one 3-cluster"
    );
}

#[tokio::test]
async fn detect_topics_is_idempotent() {
    let kb = kb().await;

    let m1 = mk_message(&kb, "m1").await;
    let e1 = mk_entity(&kb, "e1", "alpha").await;
    let e2 = mk_entity(&kb, "e2", "beta").await;
    mention(&kb, m1, e1).await;
    mention(&kb, m1, e2).await;

    let r1 = detect_topics_once(&kb, TopicConfig::default())
        .await
        .expect("detect");
    let r2 = detect_topics_once(&kb, TopicConfig::default())
        .await
        .expect("detect");
    // Second pass should re-merge into the same Topic, not create a new
    // one.  `created` may be 0 and `updated` may be 1.
    assert_eq!(r1.created, 1);
    assert_eq!(r2.created, 0);
}

#[tokio::test]
async fn detect_topics_filters_below_min_community_size() {
    let kb = kb().await;
    // Singleton entity — never co-occurs with anything.
    let m1 = mk_message(&kb, "m_solo").await;
    let e1 = mk_entity(&kb, "e_solo", "lonely").await;
    mention(&kb, m1, e1).await;

    let report = detect_topics_once(&kb, TopicConfig::default())
        .await
        .expect("detect");
    assert_eq!(report.created, 0);
}
