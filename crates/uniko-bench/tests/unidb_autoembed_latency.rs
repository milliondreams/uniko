//! Isolated repro for rustic-ai/uni-db#49: auto-embed insert latency
//! grows O(n) when the label has a DateTime property.
//!
//! Standalone uni-db test — no uniko-store or uniko-extract dependencies.
//!
//! Root cause: the combination of a Vector index with EmbeddingCfg
//! (auto-embed) AND a DateTime property on the same label causes
//! per-insert latency to grow linearly with row count. Without the
//! DateTime property, inserts stay flat.
//!
//! Run: cargo nextest run -p uniko-bench --test unidb_autoembed_latency --nocapture --run-ignored all

use std::time::Instant;

use uni_db::api::schema::EmbeddingCfg;
use uni_db::{
    DataType, IndexType, ModelAliasSpec, ModelTask, Uni, Value, VectorAlgo, VectorIndexCfg,
    VectorMetric, WarmupPolicy,
};

const NUM_INSERTS: usize = 200;
const MESSAGE_TEXT: &str = "I went to a LGBTQ support group yesterday and it was so powerful. \
     The transgender stories were so inspiring and I felt really accepted.";

fn nomic_catalog() -> Vec<ModelAliasSpec> {
    vec![ModelAliasSpec {
        alias: "embed/default".to_string(),
        task: ModelTask::Embed,
        provider_id: "local/onnx".to_string(),
        model_id: "NomicEmbedTextV15".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({}),
    }]
}

fn auto_embed_vector_index() -> IndexType {
    IndexType::Vector(VectorIndexCfg {
        algorithm: VectorAlgo::HnswSq {
            m: 16,
            ef_construction: 100,
            partitions: None,
        },
        metric: VectorMetric::Cosine,
        embedding: Some(EmbeddingCfg {
            alias: "embed/default".to_string(),
            source_properties: vec!["content".to_string()],
            batch_size: 32,
            document_prefix: Some("search_document: ".to_string()),
            query_prefix: Some("search_query: ".to_string()),
        }),
    })
}

/// Repro: auto-embed + DateTime property → O(n) insert latency.
///
/// Inserts 200 nodes with `message_id` (String), `content` (String,
/// auto-embedded), and `timestamp` (DateTime). Latency is flat at ~35ms
/// through insert ~100, then climbs linearly to 200ms+ by insert 180.
#[tokio::test]
#[ignore]
async fn autoembed_with_datetime_grows() {
    let db = Uni::in_memory()
        .xervo_catalog(nomic_catalog())
        .build()
        .await
        .unwrap();

    db.schema()
        .label("Message")
        .property("message_id", DataType::String)
        .property("content", DataType::String)
        .property("timestamp", DataType::DateTime)
        .property_nullable("embedding", DataType::Vector { dimensions: 768 })
        .index("content", IndexType::FullText)
        .index("embedding", auto_embed_vector_index())
        .done()
        .apply()
        .await
        .unwrap();

    let session = db.session();
    let latencies_ms = insert_loop(&session, NUM_INSERTS, true).await;

    report("Auto-embed WITH DateTime (repro)", &latencies_ms);
    let r = ratio(&latencies_ms);
    db.shutdown().await.unwrap();

    assert!(
        r < 2.0,
        "Auto-embed + DateTime grew {r:.1}x over {NUM_INSERTS} inserts (uni-db#49)",
    );
}

/// Control: same schema but WITHOUT the DateTime property → flat latency.
#[tokio::test]
#[ignore]
async fn autoembed_without_datetime_flat() {
    let db = Uni::in_memory()
        .xervo_catalog(nomic_catalog())
        .build()
        .await
        .unwrap();

    db.schema()
        .label("Message")
        .property("message_id", DataType::String)
        .property("content", DataType::String)
        .property_nullable("embedding", DataType::Vector { dimensions: 768 })
        .index("content", IndexType::FullText)
        .index("embedding", auto_embed_vector_index())
        .done()
        .apply()
        .await
        .unwrap();

    let session = db.session();
    let latencies_ms = insert_loop(&session, NUM_INSERTS, false).await;

    report("Auto-embed WITHOUT DateTime (control)", &latencies_ms);
    let r = ratio(&latencies_ms);
    db.shutdown().await.unwrap();

    assert!(
        r < 2.0,
        "Without DateTime, insert latency still grew {r:.1}x — unexpected.",
    );
}

/// Baseline: no auto-embed, no DateTime — pure graph insert overhead.
#[tokio::test]
#[ignore]
async fn no_autoembed_baseline() {
    let db = Uni::in_memory().build().await.unwrap();

    db.schema()
        .label("Message")
        .property("message_id", DataType::String)
        .property("content", DataType::String)
        .index("content", IndexType::FullText)
        .done()
        .apply()
        .await
        .unwrap();

    let session = db.session();
    let latencies_ms = insert_loop(&session, NUM_INSERTS, false).await;

    report("No auto-embed baseline", &latencies_ms);
    let r = ratio(&latencies_ms);
    db.shutdown().await.unwrap();

    assert!(
        r < 2.0,
        "Even without auto-embed, insert latency grew {r:.1}x.",
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn insert_loop(
    session: &uni_db::api::session::Session,
    n: usize,
    with_timestamp: bool,
) -> Vec<f64> {
    let mut latencies = Vec::with_capacity(n);

    for i in 0..n {
        let tx = session.tx().await.unwrap();
        let start = Instant::now();

        if with_timestamp {
            tx.execute_with(
                "CREATE (:Message {message_id: $p0, content: $p1, timestamp: $p2})",
            )
            .param("p0", Value::String(format!("msg-{i:04}")))
            .param("p1", Value::String(MESSAGE_TEXT.into()))
            .param("p2", Value::String(chrono::Utc::now().to_rfc3339()))
            .run()
            .await
            .unwrap();
        } else {
            tx.execute_with(
                "CREATE (:Message {message_id: $p0, content: $p1})",
            )
            .param("p0", Value::String(format!("msg-{i:04}")))
            .param("p1", Value::String(MESSAGE_TEXT.into()))
            .run()
            .await
            .unwrap();
        }

        tx.commit().await.unwrap();
        latencies.push(start.elapsed().as_millis() as f64);
    }

    latencies
}

fn report(title: &str, latencies_ms: &[f64]) {
    let n = latencies_ms.len();
    let first = &latencies_ms[..20];
    let mid_start = n / 2 - 10;
    let middle = &latencies_ms[mid_start..mid_start + 20];
    let last = &latencies_ms[n - 20..];
    let avg = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let max = |s: &[f64]| s.iter().cloned().fold(0.0_f64, f64::max);

    eprintln!("\n=== {title} ({n} nodes) ===");
    eprintln!("  First 20:  avg {:.1}ms  max {:.1}ms", avg(first), max(first));
    eprintln!("  Mid 20:    avg {:.1}ms  max {:.1}ms", avg(middle), max(middle));
    eprintln!("  Last 20:   avg {:.1}ms  max {:.1}ms", avg(last), max(last));
    eprintln!("  Overall:   avg {:.1}ms  max {:.1}ms", avg(latencies_ms), max(latencies_ms));

    eprintln!("\n  Per-insert trend:");
    for i in (0..n).step_by(20) {
        eprintln!("    insert {:>4}: {:.0}ms", i, latencies_ms[i]);
    }

    let r = ratio(latencies_ms);
    eprintln!("\n  Slowdown ratio (last 20 / first 20): {r:.2}x");
}

fn ratio(latencies_ms: &[f64]) -> f64 {
    let first = &latencies_ms[..20];
    let last = &latencies_ms[latencies_ms.len() - 20..];
    let avg = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    avg(last) / avg(first).max(1.0)
}
