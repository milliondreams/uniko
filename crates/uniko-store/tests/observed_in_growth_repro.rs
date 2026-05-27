//! Repro: `batch_create_edges` MATCH+CREATE query time grows ~5x
//! within a single conversation, even with all known optimizations
//! (multi-MATCH, both-side label hints, skip RETURN).
//!
//! This replays the exact pattern uniko's observation pipeline emits:
//!
//! 1. Auto-embedded `:Message` nodes are created over time.
//! 2. Per "turn", 2-3 auto-embedded `:Observation` nodes are created
//!    via `UNWIND $items AS item CREATE (n:Observation {...})`.
//! 3. Per "turn", 2-3 OBSERVED_IN edges are created via
//!    `UNWIND $edges AS e
//!     MATCH (a:Observation) WHERE id(a) = e.src
//!     MATCH (b:Message) WHERE id(b) = e.dst
//!     CREATE (a)-[r:OBSERVED_IN]->(b)`
//!
//! On each turn the query inputs are constant (~3 edges, 1 src node id
//! per Message, 3 dst node ids per Observation). Yet the query_us grows
//! monotonically with the number of prior turns.
//!
//! Filed as a follow-up to <https://github.com/rustic-ai/uni-db/issues/55>.
//!
//! Run: `cargo nextest run -p uniko-store --test observed_in_growth_repro --run-ignored all`

use std::collections::HashMap;
use std::time::Instant;

use uni_db::api::schema::EmbeddingCfg;
use uni_db::{
    DataType, IndexType, ModelAliasSpec, ModelTask, Uni, Value, VectorAlgo, VectorIndexCfg,
    VectorMetric, WarmupPolicy,
};

const TURNS: usize = 400;
const OBS_PER_TURN: usize = 3;

const MESSAGE_TEXT: &str =
    "The quick brown fox jumps over the lazy dog at the local park yesterday afternoon.";

fn nomic_catalog() -> Vec<ModelAliasSpec> {
    // Use AllMiniLML6V2 (384d) — the model used in our LoCoMo run.
    vec![ModelAliasSpec {
        alias: "embed/default".to_string(),
        task: ModelTask::Embed,
        provider_id: "local/onnx".to_string(),
        model_id: "AllMiniLML6V2".to_string(),
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
            document_prefix: None,
            query_prefix: None,
        }),
    })
}

async fn setup_db() -> Uni {
    let db = Uni::in_memory()
        .xervo_catalog(nomic_catalog())
        .build()
        .await
        .unwrap();

    db.schema()
        .label("Message")
        .property("message_id", DataType::String)
        .property("content", DataType::String)
        .property_nullable("embedding", DataType::Vector { dimensions: 384 })
        .index("embedding", auto_embed_vector_index())
        .done()
        .label("Observation")
        .property("observation_id", DataType::String)
        .property("content", DataType::String)
        .property_nullable("embedding", DataType::Vector { dimensions: 384 })
        .index("embedding", auto_embed_vector_index())
        .done()
        .edge_type("OBSERVED_IN", &["Observation"], &["Message"])
        .done()
        .apply()
        .await
        .unwrap();

    db
}

/// Create a single Message node and return its id.
async fn create_message(db: &Uni, turn: usize) -> i64 {
    let session = db.session();
    let tx = session.tx().await.unwrap();
    let result = tx
        .query_with("CREATE (n:Message {message_id: $id, content: $c}) RETURN id(n) AS vid")
        .param("id", Value::String(format!("msg-{turn:04}")))
        .param("c", Value::String(format!("{MESSAGE_TEXT} (turn {turn})")))
        .fetch_all()
        .await
        .unwrap();
    tx.commit().await.unwrap();
    result.rows().first().unwrap().get::<i64>("vid").unwrap()
}

/// Batch-create Observation nodes and return their ids. Mirrors
/// `batch_create_nodes`.
async fn batch_create_observations(db: &Uni, turn: usize, n: usize) -> Vec<i64> {
    let session = db.session();
    let tx = session.tx().await.unwrap();
    let items: Vec<Value> = (0..n)
        .map(|i| {
            let mut m = HashMap::new();
            m.insert(
                "observation_id".to_string(),
                Value::String(format!("obs-{turn:04}-{i}")),
            );
            m.insert(
                "content".to_string(),
                Value::String(format!("Observation {i} from turn {turn}: {MESSAGE_TEXT}")),
            );
            Value::Map(m)
        })
        .collect();

    let result = tx
        .query_with(
            "UNWIND $items AS item \
             CREATE (n:Observation {observation_id: item.observation_id, content: item.content}) \
             RETURN id(n) AS vid",
        )
        .param("items", Value::List(items))
        .fetch_all()
        .await
        .unwrap();
    tx.commit().await.unwrap();
    result
        .rows()
        .iter()
        .map(|r| r.get::<i64>("vid").unwrap())
        .collect()
}

/// Batch-create OBSERVED_IN edges using the optimized form
/// (multi-MATCH, label hints on both endpoints, no RETURN).
/// Returns the elapsed wall time of the *query* phase only.
async fn batch_create_observed_in(db: &Uni, obs_ids: &[i64], msg_id: i64) -> u128 {
    let session = db.session();
    let tx = session.tx().await.unwrap();

    let edges: Vec<Value> = obs_ids
        .iter()
        .map(|&oid| {
            let mut m = HashMap::new();
            m.insert("src".to_string(), Value::Int(oid));
            m.insert("dst".to_string(), Value::Int(msg_id));
            Value::Map(m)
        })
        .collect();

    let cypher = "UNWIND $edges AS e \
                  MATCH (a:Observation) WHERE id(a) = e.src \
                  MATCH (b:Message) WHERE id(b) = e.dst \
                  CREATE (a)-[r:OBSERVED_IN]->(b)";

    let t0 = Instant::now();
    tx.execute_with(cypher)
        .param("edges", Value::List(edges))
        .run()
        .await
        .unwrap();
    let query_us = t0.elapsed().as_micros();

    tx.commit().await.unwrap();
    query_us
}

#[tokio::test]
#[ignore] // Requires fastembed model cached locally; takes ~3 min.
async fn repro_observed_in_query_time_grows() {
    let db = setup_db().await;

    let mut samples: Vec<u128> = Vec::with_capacity(TURNS);

    for turn in 0..TURNS {
        let msg_id = create_message(&db, turn).await;
        let obs_ids = batch_create_observations(&db, turn, OBS_PER_TURN).await;
        let q_us = batch_create_observed_in(&db, &obs_ids, msg_id).await;
        samples.push(q_us);

        if turn % 50 == 0 || turn == TURNS - 1 {
            // Window mean of last 20 calls (or all if early).
            let window_start = samples.len().saturating_sub(20);
            let window: f64 = samples[window_start..]
                .iter()
                .map(|v| *v as f64)
                .sum::<f64>()
                / (samples.len() - window_start) as f64
                / 1000.0;
            eprintln!(
                "turn {turn:>4}: latest_query_ms={:.1}  window_mean(last20)_ms={:.1}",
                samples.last().unwrap() / 1000,
                window
            );
        }
    }

    let first_50_mean: f64 = samples[..50].iter().map(|v| *v as f64).sum::<f64>() / 50.0 / 1000.0;
    let last_50_mean: f64 = samples[samples.len() - 50..]
        .iter()
        .map(|v| *v as f64)
        .sum::<f64>()
        / 50.0
        / 1000.0;
    let growth = last_50_mean / first_50_mean;

    eprintln!("\n--- Summary ---");
    eprintln!("turns: {TURNS}, observations per turn: {OBS_PER_TURN}");
    eprintln!("first 50 calls mean query time: {first_50_mean:.1}ms");
    eprintln!("last  50 calls mean query time: {last_50_mean:.1}ms");
    eprintln!("growth: {growth:.1}x");

    if growth > 2.0 {
        eprintln!(
            "\n⚠ CONFIRMED: batch_create_edges query time grew {growth:.1}x even though \
             input shape (3 edges, 1 src, 3 dst) is constant. Both endpoints have label \
             hints; query uses multi-MATCH and skips RETURN."
        );
    }

    db.shutdown().await.unwrap();
}
