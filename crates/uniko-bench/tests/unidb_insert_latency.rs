//! Reproduction test for potential uni-db insert latency regression.
//!
//! During LoCoMo ingestion (369 turns), per-message insert latency
//! grows monotonically: ~1s at turn 1 → ~11s at turn 360, despite
//! uniform message sizes (~118 chars avg). Graph size should not
//! affect insert performance.
//!
//! This test isolates the variable: inserts identical messages into
//! an in-memory KB with auto-embed enabled and measures per-insert
//! latency to determine if the growth is in uni-db or our pipeline.
//!
//! Run: cargo nextest run -p uniko-bench --test unidb_insert_latency --nocapture

use std::collections::HashMap;
use std::time::Instant;

use uni_db::{ModelAliasSpec, Value};
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;

const NUM_INSERTS: usize = 200;
const NUM_PERSISTENT: usize = 300;
const MESSAGE_TEXT: &str = "I went to a LGBTQ support group yesterday and it was so powerful. \
     The transgender stories were so inspiring and I felt really accepted.";

/// Insert N identical Message nodes with auto-embed and measure per-insert latency.
///
/// If latency grows linearly with row count, this is a uni-db issue.
/// If latency stays constant, the slowdown is in our NLP/extraction pipeline.
#[tokio::test]
async fn insert_latency_should_not_grow_with_row_count() {
    let config = UnikoConfig::default();
    let kb = KnowledgeBase::in_memory_with_xervo(config, Vec::<ModelAliasSpec>::new())
        .await
        .expect("KB with xervo");

    let mut latencies_ms = Vec::with_capacity(NUM_INSERTS);

    for i in 0..NUM_INSERTS {
        let mut props = HashMap::new();
        props.insert("message_id".into(), Value::String(format!("msg-{i:04}")));
        props.insert("content".into(), Value::String(MESSAGE_TEXT.into()));
        props.insert("content_type".into(), Value::String("text".into()));
        props.insert(
            "timestamp".into(),
            Value::String(chrono::Utc::now().to_rfc3339()),
        );

        let start = Instant::now();
        kb.create_node("Message", &props).await.unwrap();
        let elapsed = start.elapsed();

        latencies_ms.push(elapsed.as_millis() as f64);
    }

    // Report buckets: first 20, middle 20, last 20.
    let first = &latencies_ms[..20];
    let mid_start = NUM_INSERTS / 2 - 10;
    let middle = &latencies_ms[mid_start..mid_start + 20];
    let last = &latencies_ms[NUM_INSERTS - 20..];

    let avg = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let max = |s: &[f64]| s.iter().cloned().fold(0.0_f64, f64::max);

    eprintln!("\n=== Insert Latency ({NUM_INSERTS} Message nodes with auto-embed) ===");
    eprintln!(
        "  First 20:  avg {:.1}ms  max {:.1}ms",
        avg(first),
        max(first)
    );
    eprintln!(
        "  Mid 20:    avg {:.1}ms  max {:.1}ms",
        avg(middle),
        max(middle)
    );
    eprintln!(
        "  Last 20:   avg {:.1}ms  max {:.1}ms",
        avg(last),
        max(last)
    );
    eprintln!(
        "  Overall:   avg {:.1}ms  max {:.1}ms",
        avg(&latencies_ms),
        max(&latencies_ms)
    );

    let ratio = avg(last) / avg(first).max(1.0);
    eprintln!("  Slowdown ratio (last/first): {ratio:.2}x");
    eprintln!();

    // Print every 20th latency for trend visibility.
    eprintln!("  Per-insert trend:");
    for (i, &ms) in latencies_ms.iter().enumerate() {
        if i % 20 == 0 || i == NUM_INSERTS - 1 {
            eprintln!("    insert {i:>4}: {ms:.1}ms");
        }
    }
    eprintln!();

    // Fail if the last bucket is >3x slower than the first bucket.
    // A healthy system should show <1.5x variance.
    assert!(
        ratio < 3.0,
        "Insert latency grew {ratio:.1}x over {NUM_INSERTS} inserts — \
         first 20 avg {:.1}ms, last 20 avg {:.1}ms. \
         This suggests O(n) insert behavior in uni-db.",
        avg(first),
        avg(last),
    );
}

/// Same test but without auto-embed — pure graph insert baseline.
///
/// Uses a label without auto-embed (Observation, which we stripped
/// indexes from). If this stays flat but the Message test grows,
/// the issue is in auto-embed, not in graph inserts.
#[tokio::test]
async fn insert_latency_without_autoembed_baseline() {
    let config = UnikoConfig::default();
    let kb = KnowledgeBase::in_memory_with_xervo(config, Vec::<ModelAliasSpec>::new())
        .await
        .expect("KB with xervo");

    let mut latencies_ms = Vec::with_capacity(NUM_INSERTS);

    for i in 0..NUM_INSERTS {
        let mut props = HashMap::new();
        props.insert(
            "observation_id".into(),
            Value::String(format!("obs-{i:04}")),
        );
        props.insert("content".into(), Value::String(MESSAGE_TEXT.into()));
        props.insert("subject".into(), Value::String("Caroline".into()));
        props.insert("confidence".into(), Value::Float(0.85));

        let start = Instant::now();
        kb.create_node("Observation", &props).await.unwrap();
        let elapsed = start.elapsed();

        latencies_ms.push(elapsed.as_millis() as f64);
    }

    let first = &latencies_ms[..20];
    let last = &latencies_ms[NUM_INSERTS - 20..];
    let avg = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let max = |s: &[f64]| s.iter().cloned().fold(0.0_f64, f64::max);

    eprintln!("\n=== Insert Latency ({NUM_INSERTS} Observation nodes, NO auto-embed) ===");
    eprintln!(
        "  First 20:  avg {:.1}ms  max {:.1}ms",
        avg(first),
        max(first)
    );
    eprintln!(
        "  Last 20:   avg {:.1}ms  max {:.1}ms",
        avg(last),
        max(last)
    );

    let ratio = avg(last) / avg(first).max(0.01);
    eprintln!("  Slowdown ratio (last/first): {ratio:.2}x");
    eprintln!();

    // Pure graph inserts should be essentially flat.
    assert!(
        ratio < 3.0,
        "Pure graph insert latency grew {ratio:.1}x — this is a graph engine issue, not auto-embed."
    );
}

/// Persistent KB: nodes + edges latency growth.
///
/// Reproduces the follow-up to #43: even after the retry backoff fix,
/// create_edge latency grows continuously on persistent KBs. Each
/// iteration creates a Message node (auto-embed) + 2 edges, mimicking
/// the real ingestion workload.
///
/// Run: cargo nextest run -p uniko-bench --test unidb_insert_latency \
///   insert_latency_persistent_with_edges --nocapture --run-ignored all
#[tokio::test]
#[ignore] // Slow — creates a temp dir and writes ~300 nodes+edges
async fn insert_latency_persistent_with_edges() {
    let ws = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    std::env::set_current_dir(&ws).expect("cd");

    let tmp_dir = std::env::temp_dir().join("unidb-edge-latency-test");
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

    // Create a Session and Participant node to wire edges to.
    let mut session_props = HashMap::new();
    session_props.insert("session_id".into(), Value::String("test-session".into()));
    session_props.insert(
        "started_at".into(),
        Value::Temporal(uni_db::common::TemporalValue::DateTime {
            nanos_since_epoch: 0,
            offset_seconds: 0,
            timezone_name: None,
        }),
    );
    let session_nid = kb.create_node("Session", &session_props).await.unwrap();

    let mut participant_props = HashMap::new();
    participant_props.insert("participant_id".into(), Value::String("user-1".into()));
    participant_props.insert("name".into(), Value::String("TestUser".into()));
    participant_props.insert("kind".into(), Value::String("human".into()));
    participant_props.insert("last_seen".into(), Value::String("2024-01-01".into()));
    let participant_nid = kb
        .create_node("Participant", &participant_props)
        .await
        .unwrap();

    let mut node_latencies = Vec::with_capacity(NUM_PERSISTENT);
    let mut edge_latencies = Vec::with_capacity(NUM_PERSISTENT);
    let mut total_latencies = Vec::with_capacity(NUM_PERSISTENT);

    for i in 0..NUM_PERSISTENT {
        let iter_start = Instant::now();

        // Create Message node (has auto-embed on content).
        let mut props = HashMap::new();
        props.insert("message_id".into(), Value::String(format!("msg-{i:04}")));
        props.insert("content".into(), Value::String(MESSAGE_TEXT.into()));
        props.insert("content_type".into(), Value::String("text".into()));
        props.insert(
            "timestamp".into(),
            Value::Temporal(uni_db::common::TemporalValue::DateTime {
                nanos_since_epoch: i as i64 * 1_000_000_000,
                offset_seconds: 0,
                timezone_name: None,
            }),
        );

        let node_start = Instant::now();
        let msg_nid = kb.create_node("Message", &props).await.unwrap();
        let node_ms = node_start.elapsed().as_millis() as f64;

        // Create 2 edges: SENT_BY + IN_SESSION.
        let edge_start = Instant::now();
        let mut sent_by_props = HashMap::new();
        sent_by_props.insert("role".into(), Value::String("user".into()));
        kb.create_edge("SENT_BY", msg_nid, participant_nid, &sent_by_props)
            .await
            .unwrap();
        kb.create_edge("IN_SESSION", msg_nid, session_nid, &HashMap::new())
            .await
            .unwrap();
        let edge_ms = edge_start.elapsed().as_millis() as f64;

        let total_ms = iter_start.elapsed().as_millis() as f64;

        node_latencies.push(node_ms);
        edge_latencies.push(edge_ms);
        total_latencies.push(total_ms);
    }

    let avg = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let max = |s: &[f64]| s.iter().cloned().fold(0.0_f64, f64::max);

    let first = 20;
    let n = NUM_PERSISTENT;

    eprintln!("\n=== Persistent KB: {NUM_PERSISTENT} Messages + 2 edges each ===");
    eprintln!("  Node create (auto-embed):");
    eprintln!(
        "    First {first}: avg {:.1}ms  max {:.1}ms",
        avg(&node_latencies[..first]),
        max(&node_latencies[..first])
    );
    eprintln!(
        "    Last {first}:  avg {:.1}ms  max {:.1}ms",
        avg(&node_latencies[n - first..]),
        max(&node_latencies[n - first..])
    );
    eprintln!(
        "    Ratio: {:.2}x",
        avg(&node_latencies[n - first..]) / avg(&node_latencies[..first]).max(1.0)
    );

    eprintln!("  Edge create (SENT_BY + IN_SESSION):");
    eprintln!(
        "    First {first}: avg {:.1}ms  max {:.1}ms",
        avg(&edge_latencies[..first]),
        max(&edge_latencies[..first])
    );
    eprintln!(
        "    Last {first}:  avg {:.1}ms  max {:.1}ms",
        avg(&edge_latencies[n - first..]),
        max(&edge_latencies[n - first..])
    );
    eprintln!(
        "    Ratio: {:.2}x",
        avg(&edge_latencies[n - first..]) / avg(&edge_latencies[..first]).max(1.0)
    );

    eprintln!("  Total (node + edges):");
    eprintln!(
        "    First {first}: avg {:.1}ms  max {:.1}ms",
        avg(&total_latencies[..first]),
        max(&total_latencies[..first])
    );
    eprintln!(
        "    Last {first}:  avg {:.1}ms  max {:.1}ms",
        avg(&total_latencies[n - first..]),
        max(&total_latencies[n - first..])
    );
    let ratio = avg(&total_latencies[n - first..]) / avg(&total_latencies[..first]).max(1.0);
    eprintln!("    Ratio: {ratio:.2}x");

    // Per-insert trend.
    eprintln!("\n  Trend (node_ms + edge_ms = total_ms):");
    for i in (0..n).step_by(20) {
        eprintln!(
            "    insert {:>4}: node={:.0}ms edge={:.0}ms total={:.0}ms",
            i, node_latencies[i], edge_latencies[i], total_latencies[i]
        );
    }
    eprintln!(
        "    insert {:>4}: node={:.0}ms edge={:.0}ms total={:.0}ms",
        n - 1,
        node_latencies[n - 1],
        edge_latencies[n - 1],
        total_latencies[n - 1]
    );

    // Clean up.
    let _ = std::fs::remove_dir_all(&tmp_dir);

    assert!(
        ratio < 5.0,
        "Persistent KB insert+edge latency grew {ratio:.1}x over {NUM_PERSISTENT} inserts. \
         This suggests O(n) write behavior in uni-db on persistent storage."
    );
}
