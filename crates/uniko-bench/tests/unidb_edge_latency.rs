//! Isolated repro: create_edge latency grows linearly with graph size.
//!
//! create_node stays flat at ~20ms, but create_edge grows from 6ms to
//! 2,700ms+ over 300 messages. All edge types affected equally —
//! not just read-before-write patterns.
//!
//! Run: cargo nextest run -p uniko-bench --test unidb_edge_latency --nocapture --run-ignored all

use std::collections::HashMap;
use std::time::Instant;

use uni_db::{ModelAliasSpec, Value};
use uniko_store::config::UnikoConfig;
use uniko_store::KnowledgeBase;

const NUM_MESSAGES: usize = 300;

/// Pure create_edge latency test on persistent KB.
///
/// Creates N Message nodes, each with one SENT_BY edge to a single
/// Participant. No read-before-write, no queries — just create_node +
/// create_edge in a loop. Measures per-iteration edge creation time.
#[tokio::test]
#[ignore]
async fn create_edge_latency_grows_with_graph_size() {
    let ws = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    std::env::set_current_dir(&ws).expect("cd");

    let tmp_dir = std::env::temp_dir().join("unidb-edge-latency-isolated");
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();

    let config = UnikoConfig {
        catalog_path: Some(ws.join("config/catalog.json")),
        schema_path: Some(ws.join("config/schema.json")),
        ..Default::default()
    };
    let kb = KnowledgeBase::open_with_xervo(&tmp_dir, config, Vec::<ModelAliasSpec>::new())
        .await
        .expect("persistent KB");

    // Create one Participant to wire all edges to.
    let mut p_props = HashMap::new();
    p_props.insert("participant_id".into(), Value::String("user-1".into()));
    p_props.insert("name".into(), Value::String("TestUser".into()));
    p_props.insert("kind".into(), Value::String("human".into()));
    p_props.insert("last_seen".into(), Value::String("2024-01-01".into()));
    let participant_nid = kb.create_node("Participant", &p_props).await.unwrap();

    let mut node_latencies = Vec::with_capacity(NUM_MESSAGES);
    let mut edge_latencies = Vec::with_capacity(NUM_MESSAGES);

    for i in 0..NUM_MESSAGES {
        // Create Message node (auto-embed on content).
        let mut props = HashMap::new();
        props.insert(
            "message_id".into(),
            Value::String(format!("msg-{i:04}")),
        );
        props.insert(
            "content".into(),
            Value::String("Test message for edge latency measurement.".into()),
        );
        props.insert("content_type".into(), Value::String("text".into()));
        props.insert(
            "timestamp".into(),
            Value::Temporal(uni_db::common::TemporalValue::DateTime {
                nanos_since_epoch: i as i64 * 1_000_000_000,
                offset_seconds: 0,
                timezone_name: None,
            }),
        );

        let t = Instant::now();
        let msg_nid = kb.create_node("Message", &props).await.unwrap();
        node_latencies.push(t.elapsed().as_millis() as f64);

        // Single create_edge — no reads, no queries.
        let t = Instant::now();
        let mut edge_props = HashMap::new();
        edge_props.insert("role".into(), Value::String("user".into()));
        kb.create_edge("SENT_BY", msg_nid, participant_nid, &edge_props)
            .await
            .unwrap();
        edge_latencies.push(t.elapsed().as_millis() as f64);
    }

    let avg = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let b = 20; // bucket size

    eprintln!("\n=== create_edge latency ({NUM_MESSAGES} Messages, 1 SENT_BY edge each) ===");
    eprintln!("  Node create (auto-embed):");
    eprintln!(
        "    First {b}: avg {:.1}ms | Last {b}: avg {:.1}ms | Ratio: {:.2}x",
        avg(&node_latencies[..b]),
        avg(&node_latencies[NUM_MESSAGES - b..]),
        avg(&node_latencies[NUM_MESSAGES - b..]) / avg(&node_latencies[..b]).max(1.0)
    );
    eprintln!("  Edge create (SENT_BY only):");
    eprintln!(
        "    First {b}: avg {:.1}ms | Last {b}: avg {:.1}ms | Ratio: {:.2}x",
        avg(&edge_latencies[..b]),
        avg(&edge_latencies[NUM_MESSAGES - b..]),
        avg(&edge_latencies[NUM_MESSAGES - b..]) / avg(&edge_latencies[..b]).max(1.0)
    );

    eprintln!("\n  Per-insert trend:");
    for i in (0..NUM_MESSAGES).step_by(20) {
        eprintln!(
            "    insert {:>4}: node={:.0}ms  edge={:.0}ms",
            i, node_latencies[i], edge_latencies[i]
        );
    }
    eprintln!(
        "    insert {:>4}: node={:.0}ms  edge={:.0}ms",
        NUM_MESSAGES - 1,
        node_latencies[NUM_MESSAGES - 1],
        edge_latencies[NUM_MESSAGES - 1]
    );

    let _ = std::fs::remove_dir_all(&tmp_dir);

    let ratio = avg(&edge_latencies[NUM_MESSAGES - b..]) / avg(&edge_latencies[..b]).max(1.0);
    assert!(
        ratio < 5.0,
        "create_edge latency grew {ratio:.1}x over {NUM_MESSAGES} inserts. \
         First {b} avg: {:.1}ms, Last {b} avg: {:.1}ms. \
         This is a uni-db issue — pure create_edge with no reads scales with graph size.",
        avg(&edge_latencies[..b]),
        avg(&edge_latencies[NUM_MESSAGES - b..]),
    );
}

/// Control: in-memory KB — same test to confirm persistent-only issue.
#[tokio::test]
#[ignore]
async fn create_edge_latency_inmemory_baseline() {
    let config = UnikoConfig::default();
    let kb = KnowledgeBase::in_memory_with_xervo(config, Vec::<ModelAliasSpec>::new())
        .await
        .expect("in-memory KB");

    let mut p_props = HashMap::new();
    p_props.insert("participant_id".into(), Value::String("user-1".into()));
    p_props.insert("name".into(), Value::String("TestUser".into()));
    p_props.insert("kind".into(), Value::String("human".into()));
    p_props.insert("last_seen".into(), Value::String("2024-01-01".into()));
    let participant_nid = kb.create_node("Participant", &p_props).await.unwrap();

    let mut edge_latencies = Vec::with_capacity(NUM_MESSAGES);

    for i in 0..NUM_MESSAGES {
        let mut props = HashMap::new();
        props.insert("message_id".into(), Value::String(format!("msg-{i:04}")));
        props.insert("content".into(), Value::String("Test message.".into()));
        props.insert("content_type".into(), Value::String("text".into()));
        props.insert(
            "timestamp".into(),
            Value::Temporal(uni_db::common::TemporalValue::DateTime {
                nanos_since_epoch: i as i64 * 1_000_000_000,
                offset_seconds: 0,
                timezone_name: None,
            }),
        );
        let msg_nid = kb.create_node("Message", &props).await.unwrap();

        let t = Instant::now();
        let mut edge_props = HashMap::new();
        edge_props.insert("role".into(), Value::String("user".into()));
        kb.create_edge("SENT_BY", msg_nid, participant_nid, &edge_props)
            .await
            .unwrap();
        edge_latencies.push(t.elapsed().as_millis() as f64);
    }

    let avg = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let b = 20;

    eprintln!("\n=== In-memory baseline: create_edge ({NUM_MESSAGES} SENT_BY edges) ===");
    eprintln!(
        "  First {b}: avg {:.1}ms | Last {b}: avg {:.1}ms | Ratio: {:.2}x",
        avg(&edge_latencies[..b]),
        avg(&edge_latencies[NUM_MESSAGES - b..]),
        avg(&edge_latencies[NUM_MESSAGES - b..]) / avg(&edge_latencies[..b]).max(0.01)
    );
}
