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
use uniko_store::config::UnikoConfig;
use uniko_store::KnowledgeBase;

const NUM_INSERTS: usize = 200;
const MESSAGE_TEXT: &str =
    "I went to a LGBTQ support group yesterday and it was so powerful. \
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
        props.insert(
            "message_id".into(),
            Value::String(format!("msg-{i:04}")),
        );
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
        props.insert(
            "confidence".into(),
            Value::Float(0.85),
        );

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
